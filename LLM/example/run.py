"""Weekend-scale RWNN-LLM training on tinyshakespeare.

Features relevant to long unattended runs:
- **Resume**: if ``latest_model.pt`` exists, reload model + optimizer state
  and keep going. If only ``best_model.pt`` exists, fall back to that.
  If neither, start fresh.
- **Atomic checkpoint saves**: write to ``*.tmp`` then rename. Crashes
  during write don't corrupt the on-disk checkpoint.
- **Time-based eval**: eval + log every ~``EVAL_INTERVAL_SECONDS`` wall
  seconds (default 60 s), regardless of step count. Adapts to whatever
  per-step speed we actually achieve.
- **Best tracking**: whenever val improves, snapshot to ``best_model.pt``.
  Always kept separate from ``latest_model.pt`` so "the best so far" is
  never overwritten by a later worse epoch.
- **Cosine LR schedule with warmup**: standard long-run recipe.
- **NaN guard**: if train or val loss becomes non-finite, stop with a
  clear error so the outer shell-level retry can relaunch from the last
  good checkpoint.

Config drives RWNN topology via ``context_length``, ``n_nodes``,
``n_layers`` (and ``edge_prob`` for density) — see llm.py.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
sys.path.insert(0, SRC)

from llm import RWNNLM, RWNNLMConfig            # noqa: E402
from tokenizer import BPETokenizer              # noqa: E402


# ---------------------------------------------------------------------------
# Config — ~66M params, RWNN 96%, with 5% bilinear gating nodes
# ---------------------------------------------------------------------------
DATA_TRAIN = "TinyStories-train.txt"
DATA_VAL = "TinyStories-valid.txt"
BPE_TRAIN_BYTES = 80 * 1024 * 1024   # train BPE on first 80 MB only (speed)

VOCAB_SIZE = 1024
CONTEXT_LENGTH = 128
D_MODEL = 48
N_IN_RWNN = 384
N_OUT_RWNN = 384
N_NODES = 45000
N_LAYERS = 8
EDGE_PROB = 0.075
BILINEAR_FRACTION = 0.05   # ~5% of compute nodes are bilinear gates

BATCH_SIZE = 64
# LR sized for 66M params. The earlier 3e-3 diverged past step ~2000
# (above random baseline) because peak was too aggressive for this scale.
# Standard transformer practice at ~60M is 3e-4 – 1e-3; we go 5e-4 and
# warm up longer to stay safe.
PEAK_LR = 5e-4
MIN_LR = 1e-5
WARMUP_STEPS = 4000
COSINE_DECAY_STEPS = 2_000_000
GRAD_CLIP = 1.0
MAX_STEPS = 5_000_000

EVAL_INTERVAL_SECONDS = 60.0
EVAL_ITERS = 40
SEED = 0

SAMPLE_PROMPT = "We are accounted poor citizens, the patricians good."
SAMPLE_TOKENS = 300
SAMPLE_TEMPERATURE = 0.8
SAMPLE_TOP_K = 40


def atomic_save(path: str, obj) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def compute_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return PEAK_LR * (step + 1) / WARMUP_STEPS
    if step >= COSINE_DECAY_STEPS:
        return MIN_LR
    # cosine decay between warmup and COSINE_DECAY_STEPS
    progress = (step - WARMUP_STEPS) / (COSINE_DECAY_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + (PEAK_LR - MIN_LR) * coeff


def get_batch(tokens: torch.Tensor, batch_size: int, context_length: int):
    N = tokens.numel()
    idx = torch.randint(0, N - context_length - 1, (batch_size,), device=tokens.device)
    ctx = torch.stack([tokens[i : i + context_length] for i in idx])
    tgt = tokens[idx + context_length]
    return ctx, tgt


@torch.no_grad()
def estimate_loss(model, splits, context_length, iters):
    out = {}
    model.eval()
    for name, toks in splits.items():
        losses = []
        for _ in range(iters):
            ctx, tgt = get_batch(toks, BATCH_SIZE, context_length)
            logits = model(ctx)
            losses.append(F.cross_entropy(logits, tgt).item())
        out[name] = float(np.mean(losses))
    model.train()
    return out


def load_or_build_tokenizer(corpus_text: str, path: str) -> BPETokenizer:
    if os.path.exists(path):
        tok = BPETokenizer.load(path)
        print(f"loaded tokenizer from {path}  vocab={tok.vocab_size}")
        return tok
    tok = BPETokenizer()
    t0 = time.time()
    tok.train(corpus_text, vocab_size=VOCAB_SIZE)
    tok.save(path)
    print(f"trained + saved tokenizer  vocab={tok.vocab_size}  "
          f"{time.time()-t0:.1f}s  -> {path}")
    return tok


def load_or_encode_corpus(tok: BPETokenizer, text: str, cache_path: str,
                          device: str) -> torch.Tensor:
    if os.path.exists(cache_path):
        ids = torch.load(cache_path, map_location=device, weights_only=True)
        print(f"loaded encoded corpus from {cache_path}  {ids.numel():,} tokens")
        return ids
    ids = tok.encode_corpus_to_gpu(text, device=device)
    atomic_save(cache_path, ids)
    print(f"encoded + cached corpus  {ids.numel():,} tokens  -> {cache_path}")
    return ids


def resume_or_fresh(cfg: RWNNLMConfig, device: str,
                    latest_path: str, best_path: str):
    model = RWNNLM(cfg, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95))

    start_step = 0
    best_val = float("inf")
    best_step = -1
    history: list[dict] = []

    # Prefer latest (most recent progress). Fall back to best. Fall back to fresh.
    for src in (latest_path, best_path):
        if not os.path.exists(src):
            continue
        try:
            ckpt = torch.load(src, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                opt.load_state_dict(ckpt["optimizer_state"])
            start_step = int(ckpt.get("step", 0)) + 1
            best_val = float(ckpt.get("best_val", ckpt.get("val_loss", float("inf"))))
            best_step = int(ckpt.get("best_step", ckpt.get("step", -1)))
            history = list(ckpt.get("history", []))
            print(f"resumed from {os.path.basename(src)}  "
                  f"step={start_step}  best_val={best_val:.4f}@{best_step}")
            return model, opt, start_step, best_val, best_step, history
        except Exception as e:
            print(f"failed to load {src}: {e}; trying next")

    print("starting fresh")
    return model, opt, start_step, best_val, best_step, history


def plot_history(history, out_path: str, counts: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [h["step"] for h in history]
    tr = [h["train"] for h in history]
    vl = [h["val"] for h in history]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(xs, tr, label="train", lw=1.5)
    ax.plot(xs, vl, label="val", lw=1.5, linestyle="--")
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"RWNN-LLM (total {counts['total']:,} params, "
                 f"rwnn {counts['rwnn']:,})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    device = "cuda"

    latest_path = os.path.join(HERE, "latest_model.pt")
    best_path = os.path.join(HERE, "best_model.pt")
    tok_path = os.path.join(HERE, "tokenizer.json")
    train_cache = os.path.join(HERE, "train_ids.pt")
    val_cache = os.path.join(HERE, "val_ids.pt")

    # --- corpus + tokenizer ---
    train_path = os.path.join(DATA, DATA_TRAIN)
    val_path = os.path.join(DATA, DATA_VAL)
    with open(train_path) as f:
        train_text = f.read()
    with open(val_path) as f:
        val_text = f.read()
    print(f"corpus: train={len(train_text):,} chars, val={len(val_text):,} chars")
    # Train BPE on a head subset of the train file for speed; this gives a
    # representative vocab without scanning all 2.6 GB.
    bpe_subset = train_text[: min(BPE_TRAIN_BYTES, len(train_text))]
    print(f"training BPE on {len(bpe_subset):,}-char subset of train")
    tok = load_or_build_tokenizer(bpe_subset, tok_path)
    train_ids = load_or_encode_corpus(tok, train_text, train_cache, device)
    val_ids = load_or_encode_corpus(tok, val_text, val_cache, device)

    # --- model ---
    cfg = RWNNLMConfig(
        vocab_size=tok.vocab_size,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        n_in_rwnn=N_IN_RWNN,
        n_out_rwnn=N_OUT_RWNN,
        n_nodes=N_NODES,
        n_layers=N_LAYERS,
        edge_prob=EDGE_PROB,
        bilinear_fraction=BILINEAR_FRACTION,
        seed=SEED,
    )
    model, opt, start_step, best_val, best_step, history = resume_or_fresh(
        cfg, device, latest_path, best_path
    )
    counts = model.num_parameters()
    print(f"model: total={counts['total']:,}  rwnn={counts['rwnn']:,} "
          f"({counts['rwnn']/counts['total']*100:.0f}%)  "
          f"in={counts['in_projection']:,}  out={counts['out_projection']:,}")
    print(f"rwnn graph: {model.rwnn.n_nodes} nodes, {model.rwnn.n_edges:,} edges, "
          f"{model.rwnn.n_levels} levels")

    # --- one-time architecture metadata dump ---
    arch_json = os.path.join(HERE, "architecture.json")
    if not os.path.exists(arch_json):
        per_level = [
            int(model.rwnn.level_starts[i + 1].item()
                - model.rwnn.level_starts[i].item())
            for i in range(model.rwnn.n_levels)
        ]
        with open(arch_json, "w") as f:
            json.dump({
                "config": dataclasses.asdict(cfg),
                "parameters": counts,
                "rwnn_graph": {
                    "n_nodes": int(model.rwnn.n_nodes),
                    "n_edges": int(model.rwnn.n_edges),
                    "n_levels": int(model.rwnn.n_levels),
                    "nodes_per_level": per_level,
                },
                "training": {
                    "batch_size": BATCH_SIZE,
                    "peak_lr": PEAK_LR, "min_lr": MIN_LR,
                    "warmup": WARMUP_STEPS, "cosine_decay": COSINE_DECAY_STEPS,
                    "eval_interval_seconds": EVAL_INTERVAL_SECONDS,
                    "eval_iters": EVAL_ITERS,
                    "grad_clip": GRAD_CLIP, "seed": SEED,
                },
            }, f, indent=2)
        print(f"saved  {arch_json}")

    # --- reference / comparison context ---
    if start_step == 0:
        print(f"\nReference (nanoGPT char, vocab 65):")
        print(f"  step    0:  train 4.29  val 4.28  (random ≈ ln 65 = 4.17)")
        print(f"  step  500:  train 1.40  val 1.59")
        print(f"  step 2000:  train 0.86  val 1.50")
        print(f"  step 5000:  train 0.62  val 1.49")
        print(f"Our BPE vocab is {tok.vocab_size}; random baseline "
              f"ln({tok.vocab_size}) = {math.log(tok.vocab_size):.2f} nats/token.")
        print(f"A nanoGPT-scale 'good' val loss in our units is roughly "
              f"{1.5*1.6:.2f}-{2.0*1.6:.2f} nats/token.\n")

    # --- training loop ---
    print(f"training (batch={BATCH_SIZE}, ctx={CONTEXT_LENGTH}, "
          f"eval every ~{EVAL_INTERVAL_SECONDS:.0f}s)")

    # Warm up Triton JIT before the first timing window.
    ctx, tgt = get_batch(train_ids, BATCH_SIZE, CONTEXT_LENGTH)
    model(ctx); torch.cuda.synchronize()

    loss_plot = os.path.join(HERE, "loss_curve.png")
    t_last_eval = time.time()
    t_wall_start = time.time()
    step = start_step
    try:
        while step < MAX_STEPS:
            lr = compute_lr(step)
            for pg in opt.param_groups:
                pg["lr"] = lr

            ctx, tgt = get_batch(train_ids, BATCH_SIZE, CONTEXT_LENGTH)
            logits = model(ctx)
            loss = F.cross_entropy(logits, tgt)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at step {step}: "
                                   f"{loss.item()}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            step += 1

            now = time.time()
            if now - t_last_eval >= EVAL_INTERVAL_SECONDS:
                losses = estimate_loss(
                    model,
                    {"train": train_ids, "val": val_ids},
                    CONTEXT_LENGTH, EVAL_ITERS,
                )
                wall = now - t_wall_start
                record = {
                    "step": step, "wall": wall, "lr": lr,
                    "train": losses["train"], "val": losses["val"],
                }
                history.append(record)

                # Always save latest for crash resume.
                atomic_save(latest_path, {
                    "config": dataclasses.asdict(cfg),
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "step": step,
                    "best_val": best_val, "best_step": best_step,
                    "history": history,
                })

                star = ""
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    best_step = step
                    atomic_save(best_path, {
                        "config": dataclasses.asdict(cfg),
                        "model_state": model.state_dict(),
                        "step": step,
                        "val_loss": best_val,
                    })
                    star = "  *"

                # 1-line report line, easy to grep from the log.
                print(f"step {step:8d}  lr {lr:.2e}  "
                      f"train {losses['train']:.4f}  val {losses['val']:.4f}  "
                      f"best {best_val:.4f}@{best_step}  "
                      f"wall {wall/3600:5.2f}h{star}", flush=True)

                # Plot every N evals (cheap).
                if len(history) % 10 == 0:
                    try:
                        plot_history(history, loss_plot, counts)
                    except Exception as e:
                        print(f"  (plot failed: {e})")

                if not math.isfinite(losses["train"]) or \
                   not math.isfinite(losses["val"]):
                    raise RuntimeError(f"non-finite eval loss: {losses}")

                t_last_eval = now

    except KeyboardInterrupt:
        print(f"\ninterrupted at step {step}; saving latest")
        atomic_save(latest_path, {
            "config": dataclasses.asdict(cfg),
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "step": step, "best_val": best_val, "best_step": best_step,
            "history": history,
        })
    except Exception as e:
        print(f"\nERROR at step {step}: {e}")
        try:
            atomic_save(latest_path, {
                "config": dataclasses.asdict(cfg),
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "step": step, "best_val": best_val, "best_step": best_step,
                "history": history,
            })
            print(f"saved latest_model.pt for resume at step {step}")
        except Exception as e2:
            print(f"also failed to save latest: {e2}")
        raise

    plot_history(history, loss_plot, counts)

    # --- one final sample from the best model ---
    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        prompt_ids = tok.encode_to_tensor(SAMPLE_PROMPT, device=device)
        out = model.generate(prompt_ids, max_new_tokens=SAMPLE_TOKENS,
                             temperature=SAMPLE_TEMPERATURE, top_k=SAMPLE_TOP_K)
        sample = tok.decode(out[0])
        with open(os.path.join(HERE, "sample.txt"), "w") as f:
            f.write(sample)
        print("=" * 70)
        print(sample)
        print("=" * 70)
    except Exception as e:
        print(f"final sampling failed: {e}")


if __name__ == "__main__":
    main()
