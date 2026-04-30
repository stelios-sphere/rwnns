"""Unified comparison trainer.

Trains either nanoGPT (vendored from Karpathy 2022) or our RWNN-LM on
the same dataset with the same recipe. Writes one CSV per run with
columns:

    step,wall_seconds,lr,train_loss,val_loss

so later we can plot loss-vs-step *and* loss-vs-wall on the same axes.

Usage:
    python3 train.py --arch nanogpt --config char_shakespeare
    python3 train.py --arch rwnn    --config char_shakespeare
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
LLM_SRC = os.path.normpath(os.path.join(HERE, "..", "LLM", "src"))
sys.path.insert(0, HERE)
sys.path.insert(0, LLM_SRC)

from configs import CONFIGS, CompareConfig         # noqa: E402
from nanogpt_model import GPT, GPTConfig           # noqa: E402

# Lazy-import RWNN side; only needed for --arch rwnn.
def _import_rwnn():
    from llm import RWNNLM, RWNNLMConfig
    return RWNNLM, RWNNLMConfig


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare_char_shakespeare(seed: int = 1337):
    """Char-level vocab + 90/10 split. Returns (train_ids, val_ids, vocab_size, decode_fn)."""
    path = os.path.join(HERE, "data", "tinyshakespeare.txt")
    with open(path) as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train, val = data[:n], data[n:]
    decode = lambda ids: "".join(itos[int(i)] for i in ids)
    return train, val, len(chars), decode


def prepare_wikitext_bpe(seed: int = 1337):
    """Reuse the BPE-tokenised WikiText-103 corpus from the LLM/example/."""
    cache_dir = os.path.normpath(os.path.join(HERE, "..", "LLM", "example"))
    train = torch.load(os.path.join(cache_dir, "train_ids.pt"),
                       map_location="cpu", weights_only=True)
    val = torch.load(os.path.join(cache_dir, "val_ids.pt"),
                     map_location="cpu", weights_only=True)
    # The encoded tensors are on GPU and torch.long — cast to long on CPU here;
    # the runner moves them to GPU.
    train = train.detach().cpu().long()
    val = val.detach().cpu().long()
    # Vocab is 1024 (the BPE we trained earlier).
    sys.path.insert(0, os.path.join(LLM_SRC))
    from tokenizer import BPETokenizer
    tok = BPETokenizer.load(os.path.join(cache_dir, "tokenizer.json"))
    return train, val, tok.vocab_size, tok.decode


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device):
    ix = torch.randint(0, data.numel() - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


# ---------------------------------------------------------------------------
# LR schedule (same as nanoGPT)
# ---------------------------------------------------------------------------

def compute_lr(step: int, cfg: CompareConfig) -> float:
    if step < cfg.warmup_iters:
        return cfg.peak_lr * (step + 1) / cfg.warmup_iters
    if step >= cfg.max_iters:
        return cfg.min_lr
    progress = (step - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.peak_lr - cfg.min_lr)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_nanogpt(cfg: CompareConfig, vocab_size: int, device: str) -> GPT:
    gcfg = GPTConfig(
        block_size=cfg.block_size,
        vocab_size=vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )
    return GPT(gcfg).to(device)


def build_rwnn(cfg: CompareConfig, vocab_size: int, device: str):
    RWNNLM, RWNNLMConfig = _import_rwnn()
    rcfg = RWNNLMConfig(
        vocab_size=vocab_size,
        context_length=cfg.block_size,
        d_model=cfg.rwnn_d_model,
        n_nodes=cfg.rwnn_n_nodes,
        n_layers=cfg.rwnn_n_layers,
        edge_prob=cfg.rwnn_edge_prob,
        bilinear_fraction=cfg.rwnn_bilinear_fraction,
        attention_fraction=cfg.rwnn_attention_fraction,
        pos_encoding=cfg.rwnn_pos_encoding,
        seed=cfg.seed,
    )
    return RWNNLM(rcfg, device=device)


# ---------------------------------------------------------------------------
# Loss estimation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, train, val, cfg: CompareConfig, device, arch: str):
    model.eval()
    out = {}
    for split_name, split_data in [("train", train), ("val", val)]:
        losses = []
        for _ in range(cfg.eval_iters):
            x, y = get_batch(split_data, cfg.block_size, cfg.batch_size, device)
            if arch == "nanogpt":
                _, loss = model(x, targets=y)
            else:
                # RWNN-LM produces logits for ONE next token given a context.
                logits = model(x)                     # [B, V]
                loss = F.cross_entropy(logits, y[:, -1])
            losses.append(loss.item())
        out[split_name] = float(np.mean(losses))
    model.train()
    return out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["nanogpt", "rwnn"], required=True)
    p.add_argument("--config", choices=list(CONFIGS.keys()), required=True)
    p.add_argument("--max-iters", type=int, default=None,
                   help="Override config max_iters")
    p.add_argument("--out", default=None,
                   help="CSV path (default: runs/<arch>_<config>.csv)")
    args = p.parse_args()

    cfg = CONFIGS[args.config]
    if args.max_iters is not None:
        cfg.max_iters = args.max_iters

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    device = "cuda"

    # Data
    if cfg.dataset == "char_shakespeare":
        train, val, vocab_size, decode = prepare_char_shakespeare()
    elif cfg.dataset == "wikitext_bpe":
        train, val, vocab_size, decode = prepare_wikitext_bpe()
    else:
        raise ValueError(f"unknown dataset {cfg.dataset!r}")
    cfg.vocab_size = vocab_size

    # Model
    if args.arch == "nanogpt":
        model = build_nanogpt(cfg, vocab_size, device)
        n_params = sum(p.numel() for p in model.parameters())
    else:
        model = build_rwnn(cfg, vocab_size, device)
        n_params = sum(p.numel() for p in model.parameters())

    print(f"=== {args.arch} on {cfg.dataset} ===")
    print(f"  params  : {n_params:,}")
    print(f"  vocab   : {vocab_size}")
    print(f"  ctx     : {cfg.block_size}")
    print(f"  iters   : {cfg.max_iters}  (warmup {cfg.warmup_iters})")
    print(f"  batch   : {cfg.batch_size}")
    print(f"  peak LR : {cfg.peak_lr}")
    print(f"  seed    : {cfg.seed}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )

    # CSV log
    out_path = args.out or os.path.join(
        HERE, "runs", f"{args.arch}_{cfg.name}.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_f = open(out_path, "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "wall_seconds", "lr", "train_loss", "val_loss"])
    log_f.flush()

    print(f"  log     : {out_path}")
    print()

    # Warm up
    x, y = get_batch(train, cfg.block_size, cfg.batch_size, device)
    if args.arch == "nanogpt":
        _, loss = model(x, targets=y)
    else:
        logits = model(x); loss = F.cross_entropy(logits, y[:, -1])
    loss.backward()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    t0 = time.time()
    best_val = float("inf")
    for step in range(cfg.max_iters):
        lr = compute_lr(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train, cfg.block_size, cfg.batch_size, device)
        if args.arch == "nanogpt":
            _, loss = model(x, targets=y)
        else:
            logits = model(x); loss = F.cross_entropy(logits, y[:, -1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            losses = estimate_loss(model, train, val, cfg, device, args.arch)
            wall = time.time() - t0
            log.writerow([step, f"{wall:.3f}", f"{lr:.6e}",
                          f"{losses['train']:.4f}", f"{losses['val']:.4f}"])
            log_f.flush()
            star = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                star = "  *"
            print(f"  step {step:6d}  train {losses['train']:.4f}  "
                  f"val {losses['val']:.4f}  best {best_val:.4f}  "
                  f"wall {wall:6.1f}s{star}", flush=True)

    log_f.close()
    print(f"\n=== done. best val: {best_val:.4f}. log: {out_path} ===")


if __name__ == "__main__":
    main()
