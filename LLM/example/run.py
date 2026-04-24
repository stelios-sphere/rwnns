"""Train the RWNN-LLM on tinyshakespeare and generate samples.

End-to-end:
    1. Load tinyshakespeare.txt.
    2. Train a byte-level BPE tokenizer to ``vocab_size``.
    3. Pre-tokenize the whole corpus, ship the resulting ids to GPU.
    4. Build an RWNN-LLM with (context_length, n_nodes, n_layers) driving
       the RWNN topology.
    5. Train next-token prediction with cross-entropy via Adam.
    6. Sample text from the trained model.
    7. Save: loss curve, trained tokenizer, generated samples.
"""

from __future__ import annotations

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

import dataclasses
import json

from llm import RWNNLM, RWNNLMConfig  # noqa: E402
from tokenizer import BPETokenizer    # noqa: E402
from visualize import draw_architecture  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Sized to roughly match a small-transformer config on tinyshakespeare:
# nanoGPT-Shakespeare default has n_layer=6, ~10.7M params. We aim for
# ~2M params total with the RWNN carrying the majority (~60-70%) — the
# "number of parameters and layers usually used" for a similar small
# LLM. The in/out linear projections are intentionally slim so the
# RWNN core isn't dwarfed by them.
VOCAB_SIZE = 512
CONTEXT_LENGTH = 128
D_MODEL = 24
N_IN_RWNN = 128
N_OUT_RWNN = 128
N_NODES = 20000
N_LAYERS = 6
EDGE_PROB = 0.015

BATCH_SIZE = 64
LR = 3e-3
STEPS = 3000
EVAL_INTERVAL = 250
EVAL_ITERS = 20
SEED = 0

SAMPLE_PROMPT = "ROMEO:\n"
SAMPLE_TOKENS = 400
SAMPLE_TEMPERATURE = 0.9
SAMPLE_TOP_K = 40


def get_batch(tokens: torch.Tensor, batch_size: int, context_length: int):
    """Sample a batch of (context, target_next_token) from a flat token tensor."""
    N = tokens.numel()
    # Each example needs context_length tokens of input + 1 target.
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


def main():
    torch.manual_seed(SEED)
    device = "cuda"

    # 1. Load corpus.
    corpus_path = os.path.join(DATA, "tinyshakespeare.txt")
    with open(corpus_path) as f:
        text = f.read()
    print(f"corpus: {len(text):,} chars")

    # 2. Train tokenizer on 90/10 train split of raw text.
    split_point = int(0.9 * len(text))
    train_text = text[:split_point]
    val_text = text[split_point:]

    tok = BPETokenizer()
    t0 = time.time()
    tok.train(train_text, vocab_size=VOCAB_SIZE)
    print(f"tokenizer trained: vocab={tok.vocab_size}  merges={len(tok.merges)}  "
          f"{time.time()-t0:.1f}s")

    # 3. Pre-tokenize both splits to GPU tensors.
    t0 = time.time()
    train_ids = tok.encode_corpus_to_gpu(train_text, device=device, verbose=True)
    val_ids = tok.encode_corpus_to_gpu(val_text, device=device, verbose=True)
    print(f"corpus tokenized in {time.time()-t0:.1f}s: "
          f"train={train_ids.numel():,} tokens, val={val_ids.numel():,}")

    tok.save(os.path.join(HERE, "tokenizer.json"))

    # 4. Build model.
    cfg = RWNNLMConfig(
        vocab_size=tok.vocab_size,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        n_in_rwnn=N_IN_RWNN,
        n_out_rwnn=N_OUT_RWNN,
        n_nodes=N_NODES,
        n_layers=N_LAYERS,
        edge_prob=EDGE_PROB,
        seed=SEED,
    )
    model = RWNNLM(cfg, device=device)
    counts = model.num_parameters()
    print(f"model: {counts['total']:,} params "
          f"(emb={counts['embedding']:,}, in={counts['in_projection']:,}, "
          f"rwnn={counts['rwnn']:,}, out={counts['out_projection']:,})")
    print(f"rwnn graph: {model.rwnn.n_nodes} nodes, {model.rwnn.n_edges} edges, "
          f"{model.rwnn.n_levels} levels")

    # 4a. Save architecture artefacts BEFORE training.
    #     - architecture.png : networkx render of the RWNN core (random init)
    #     - architecture.json: full config + graph stats, reproducible record
    arch_png = os.path.join(HERE, "architecture.png")
    draw_architecture(
        model.rwnn,
        model.rwnn.weights,
        arch_png,
        title=(f"RWNNLM core — {model.rwnn.n_nodes} nodes, "
               f"{model.rwnn.n_edges} edges, {model.rwnn.n_levels} layers"),
    )
    print(f"saved  {arch_png}")

    arch_json = os.path.join(HERE, "architecture.json")
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
                "batch_size": BATCH_SIZE, "steps": STEPS, "lr": LR,
                "eval_interval": EVAL_INTERVAL, "eval_iters": EVAL_ITERS,
                "seed": SEED,
            },
        }, f, indent=2)
    print(f"saved  {arch_json}")

    # 5. Training loop with best-checkpoint tracking.
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    history_train: list[tuple[int, float]] = []
    history_val: list[tuple[int, float]] = []

    # Warm up Triton JIT.
    ctx, tgt = get_batch(train_ids, BATCH_SIZE, CONTEXT_LENGTH)
    model(ctx); torch.cuda.synchronize()

    best_ckpt_path = os.path.join(HERE, "best_model.pt")
    final_ckpt_path = os.path.join(HERE, "final_model.pt")

    def save_ckpt(path: str, *, step: int, val_loss: float):
        torch.save({
            "config": dataclasses.asdict(cfg),
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "step": step,
            "val_loss": val_loss,
        }, path)

    # Reference points for loss comparison. nanoGPT's char-level Shakespeare
    # run reports (train/val) cross-entropy (natural log) at (step):
    #   step   0: 4.29 / 4.28   (random init, vocab 65, ln 65 = 4.17)
    #   step 250: 1.64 / 1.82
    #   step 500: 1.40 / 1.59
    #   step 2000: 0.86 / 1.50
    #   step 5000: 0.62 / 1.49
    # We use BPE (vocab={V}). Uniform-random baseline on our vocab is
    # ln({V}) = {baseline:.2f} nats/token. Because one BPE token ≈ 1.6
    # chars here, a "char-loss 1.5" nanoGPT-level model corresponds
    # roughly to val loss {ref_val:.2f}-{ref_val2:.2f} nats/token for us;
    # a reasonable cutoff to call training "working".
    V = tok.vocab_size
    baseline = float(np.log(V))
    ref_val = 1.5 * 1.6   # good char-loss × compression factor
    ref_val2 = 2.0 * 1.6  # mediocre
    print(f"\nReference (nanoGPT char, vocab 65):")
    print(f"  step    0:  train 4.29  val 4.28  (random ≈ ln 65 = 4.17)")
    print(f"  step  500:  train 1.40  val 1.59")
    print(f"  step 2000:  train 0.86  val 1.50")
    print(f"  step 5000:  train 0.62  val 1.49")
    print(f"Our vocab is BPE({V}); random baseline ln({V}) = {baseline:.2f} nats/token.")
    print(f"A nanoGPT-equivalent 'good' val loss in our units is roughly "
          f"{ref_val:.2f}-{ref_val2:.2f} nats/token.\n")

    print(f"training {STEPS} steps (batch={BATCH_SIZE}, ctx={CONTEXT_LENGTH})")
    t0 = time.time()
    best_val = float("inf")
    best_step = -1
    for step in range(STEPS):
        ctx, tgt = get_batch(train_ids, BATCH_SIZE, CONTEXT_LENGTH)
        logits = model(ctx)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % EVAL_INTERVAL == 0 or step == STEPS - 1:
            losses = estimate_loss(
                model,
                {"train": train_ids, "val": val_ids},
                CONTEXT_LENGTH, EVAL_ITERS,
            )
            history_train.append((step, losses["train"]))
            history_val.append((step, losses["val"]))
            elapsed = time.time() - t0
            star = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                best_step = step
                save_ckpt(best_ckpt_path, step=step, val_loss=best_val)
                star = "  * best, checkpoint saved"
            print(f"  step {step:5d}  train {losses['train']:.4f}  "
                  f"val {losses['val']:.4f}  [{elapsed:.1f}s]{star}")
    torch.cuda.synchronize()
    save_ckpt(final_ckpt_path, step=STEPS - 1, val_loss=history_val[-1][1])
    print(f"training done in {time.time()-t0:.1f}s  "
          f"(best val {best_val:.4f} @ step {best_step})")
    print(f"saved  {best_ckpt_path}")
    print(f"saved  {final_ckpt_path}")

    # Load the best model before sampling so generation uses the best checkpoint.
    print(f"loading best checkpoint for sampling")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # 6. Sample.
    prompt_ids = tok.encode_to_tensor(SAMPLE_PROMPT, device=device)
    out = model.generate(
        prompt_ids,
        max_new_tokens=SAMPLE_TOKENS,
        temperature=SAMPLE_TEMPERATURE,
        top_k=SAMPLE_TOP_K,
    )
    sample_text = tok.decode(out[0])
    print("=" * 70)
    print(sample_text)
    print("=" * 70)

    with open(os.path.join(HERE, "sample.txt"), "w") as f:
        f.write(sample_text)

    # 7. Plot loss.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs_t, ys_t = zip(*history_train)
    xs_v, ys_v = zip(*history_val)
    ax.plot(xs_t, ys_t, label="train", lw=2)
    ax.plot(xs_v, ys_v, label="val", lw=2, linestyle="--")
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"RWNN-LLM training — vocab={tok.vocab_size}, "
                 f"ctx={CONTEXT_LENGTH}, {counts['total']:,} params")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "loss_curve.png"), dpi=130)
    print(f"saved  {os.path.join(HERE, 'loss_curve.png')}")


if __name__ == "__main__":
    main()
