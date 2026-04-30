"""Comparison configs.

Two paired runs each. We hold dataset, sequence length, vocabulary,
batch size, training horizon, optimizer, gradient clip, eval cadence
and random seed constant; only the model architecture differs.

Char-Shakespeare: nanoGPT's published default, 10.6 M params, 5000
training iterations, char-level vocab (~65 distinct chars). Both runs
trained on the same character stream.

WikiText-103-BPE: ~50 M params at vocab 1024, ctx 512. Pairs the
existing RWNN best result (val 2.4837) against a freshly-trained
nanoGPT at the same parameter budget.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompareConfig:
    name: str
    dataset: str            # "char_shakespeare" | "wikitext_bpe"

    # Sequence + vocab.
    block_size: int
    vocab_size: int

    # Training schedule.
    batch_size: int
    max_iters: int
    eval_interval: int
    eval_iters: int
    warmup_iters: int
    peak_lr: float
    min_lr: float
    grad_clip: float
    weight_decay: float
    seed: int

    # nanoGPT-specific.
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False         # GPT-2 uses bias; LLaMA-style uses None. nanoGPT default = False.

    # RWNN-specific.
    rwnn_d_model: int = 64
    rwnn_n_nodes: int = 35_000
    rwnn_n_layers: int = 6
    rwnn_edge_prob: float = 0.025
    rwnn_bilinear_fraction: float = 0.10
    rwnn_attention_fraction: float = 0.10
    rwnn_pos_encoding: str = "sinusoidal"


# ---------------------------------------------------------------------------
# Char-level Shakespeare — nanoGPT canonical config (Karpathy 2022).
#   nanoGPT side: n_layer=6 n_head=6 n_embd=384, ~10.6M params.
#   RWNN side   : matched ~10.6M params via n_nodes/edge_prob.
# ---------------------------------------------------------------------------
char_shakespeare = CompareConfig(
    name="char_shakespeare",
    dataset="char_shakespeare",
    block_size=256,
    vocab_size=65,                  # determined by chars in tinyshakespeare.txt
    batch_size=64,
    max_iters=5000,
    eval_interval=250,
    eval_iters=200,
    warmup_iters=100,
    peak_lr=1e-3,
    min_lr=1e-4,
    grad_clip=1.0,
    weight_decay=1e-1,
    seed=1337,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.0,
    bias=False,
    rwnn_d_model=64,
    rwnn_n_nodes=35_000,
    rwnn_n_layers=6,
    rwnn_edge_prob=0.025,
    rwnn_bilinear_fraction=0.10,
    rwnn_attention_fraction=0.10,
    rwnn_pos_encoding="sinusoidal",
)


# ---------------------------------------------------------------------------
# WikiText-103-BPE — matched to our headline RWNN result (53M params, ctx 512).
# ---------------------------------------------------------------------------
wikitext_bpe = CompareConfig(
    name="wikitext_bpe",
    dataset="wikitext_bpe",
    block_size=512,
    vocab_size=1024,                # matches the BPE we trained for the RWNN
    batch_size=32,
    max_iters=50_000,
    eval_interval=500,
    eval_iters=40,
    warmup_iters=2_000,
    peak_lr=3e-4,                   # standard transformer LR at ~50M params
    min_lr=3e-5,
    grad_clip=1.0,
    weight_decay=1e-1,
    seed=1337,
    # ~50 M params: n_layer=10, n_head=10, n_embd=640
    n_layer=10,
    n_head=10,
    n_embd=640,
    dropout=0.0,
    bias=False,
    # RWNN: same defaults that produced val 2.4837 — we'll just rerun if needed
    rwnn_d_model=48,
    rwnn_n_nodes=80_000,
    rwnn_n_layers=8,
    rwnn_edge_prob=0.020,
    rwnn_bilinear_fraction=0.10,
    rwnn_attention_fraction=0.10,
    rwnn_pos_encoding="sinusoidal",
)


CONFIGS = {
    "char_shakespeare": char_shakespeare,
    "wikitext_bpe": wikitext_bpe,
}
