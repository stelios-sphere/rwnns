"""RWNN-LLM: a small autoregressive language model with an RWNN core.

Architecture — *no projection layers*. Every (token-position,
embedding-dimension) pair is its own RWNN input node, every vocab entry
is its own RWNN output node:

    ids  [B, T]                             T token ids, the context
      │  token embedding  (V, d_model)
      ▼
    emb  [B, T, d_model]
      │  flatten
      ▼
    flat [B, T·d_model]                     RWNN input (n_in = T·d_model)
      │  RWNN  (tanh hidden, linear output)
      ▼
    logits [B, V]                           RWNN output (n_out = V)

The configurable knobs ``context_length``, ``n_nodes``, ``n_layers`` and
``edge_prob`` drive the RWNN topology directly. ``n_in`` and ``n_out``
are *derived*: ``n_in = context_length * d_model``,
``n_out = vocab_size``. There is no in-projection or out-projection
linear layer — those would compress / expand tokens through a
fixed-width bottleneck before the random graph could reason over them.

For training, sample random fixed-length contexts out of a pre-tokenised
GPU tensor and minimise cross-entropy of the next token. For generation,
maintain a rolling context of the last T tokens and sample step-by-step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from rwnn import RWNN, build_layered_rwnn


@dataclass
class RWNNLMConfig:
    vocab_size: int
    context_length: int = 64        # T — number of tokens the model looks at
    d_model: int = 32               # token embedding dimension
    n_nodes: int = 1500             # total RWNN nodes (in + bias + hidden + out)
    n_layers: int = 5               # RWNN topological depth
    edge_prob: float = 0.03         # RWNN connection density
    bilinear_fraction: float = 0.0  # fraction of compute nodes that are bilinear gates
    product_fraction: float = 0.0   # fraction of compute nodes that are pure-product
    attention_fraction: float = 0.0 # fraction of compute nodes that are softmax-aggregator
    n_bias: int = 2                 # bias nodes inside RWNN
    seed: int = 0
    # Positional encoding scheme. "learned" = nn.Embedding(T, d_model), trained
    # by gradient descent. "sinusoidal" = the closed-form Vaswani-2017 table,
    # parameter-free and extrapolates to longer contexts at inference.
    pos_encoding: str = "learned"
    # Architecture: single RWNN, or two parallel mirror RWNNs (one all-linear,
    # one all-bilinear) with the same topology, summed at the output. When
    # parallel=True, ``bilinear_fraction`` is ignored — the linear branch is
    # 0%, the bilinear branch is 100%.
    parallel: bool = False


class RWNNLM(nn.Module):
    def __init__(self, cfg: RWNNLMConfig, device: str | torch.device = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)

        # Derived RWNN dimensions: every embedded feature is an input node,
        # every vocab entry is an output node. No bottleneck.
        n_in = cfg.context_length * cfg.d_model
        n_out = cfg.vocab_size
        n_role = n_in + cfg.n_bias + n_out
        if cfg.n_nodes < n_role + 1:
            raise ValueError(
                f"n_nodes={cfg.n_nodes} too small: need at least "
                f"{n_role + 1} (n_in={n_in} + n_bias={cfg.n_bias} + "
                f"n_out={n_out} + 1 hidden)"
            )

        if cfg.parallel:
            # Two RWNNs of identical topology, one all-linear and one all-
            # bilinear. They share only the input embedding and merge at the
            # output by summation. Same seed → identical edge sets;
            # bilinear_fraction differs so node kinds are complementary.
            graph_L = build_layered_rwnn(
                n_nodes=cfg.n_nodes, edge_prob=cfg.edge_prob,
                n_layers=cfg.n_layers, n_in=n_in,
                n_bias=cfg.n_bias, n_out=n_out,
                seed=cfg.seed, bilinear_fraction=0.0,
            )
            graph_B = build_layered_rwnn(
                n_nodes=cfg.n_nodes, edge_prob=cfg.edge_prob,
                n_layers=cfg.n_layers, n_in=n_in,
                n_bias=cfg.n_bias, n_out=n_out,
                seed=cfg.seed, bilinear_fraction=1.0,
            )
            self.rwnn_L = RWNN(graph_L, device=self.device)
            self.rwnn_B = RWNN(graph_B, device=self.device)
            # Scale each branch's weights by 1/√2 so the summed output has
            # variance comparable to a single-branch network's output.
            with torch.no_grad():
                self.rwnn_L.weights.mul_(1.0 / math.sqrt(2.0))
                self.rwnn_B.weights.mul_(1.0 / math.sqrt(2.0))
        else:
            graph = build_layered_rwnn(
                n_nodes=cfg.n_nodes,
                edge_prob=cfg.edge_prob,
                n_layers=cfg.n_layers,
                n_in=n_in,
                product_fraction=cfg.product_fraction,
                attention_fraction=cfg.attention_fraction,
                n_bias=cfg.n_bias,
                n_out=n_out,
                seed=cfg.seed,
                bilinear_fraction=cfg.bilinear_fraction,
            )
            self.rwnn = RWNN(graph, device=self.device)

        # Token embedding + per-position offset. Without a positional signal,
        # two tokens of the same id at different positions have identical
        # d_model vectors; under random connectivity the only positional
        # information (the slot index in the flattened input) gets washed
        # out. The pos table gives each (position, dim) input node a unique
        # additive signature.
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        if cfg.pos_encoding == "learned":
            # Trainable [T, d_model] embedding.
            self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
            self._sinusoidal = False
        elif cfg.pos_encoding == "sinusoidal":
            # Vaswani-2017 closed form: even dims = sin, odd dims = cos,
            # at log-spaced frequencies. Non-learnable buffer; zero params.
            pe = _build_sinusoidal_table(cfg.context_length, cfg.d_model)
            self.register_buffer("pos_emb_table", pe)
            self._sinusoidal = True
        else:
            raise ValueError(
                f"pos_encoding must be 'learned' or 'sinusoidal', "
                f"got {cfg.pos_encoding!r}"
            )

        self.to(self.device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, T] -> logits [B, V] for the next token."""
        assert ids.dim() == 2, f"expected [B, T], got {tuple(ids.shape)}"
        B, T = ids.shape
        assert T == self.cfg.context_length, (
            f"context_length mismatch: model={self.cfg.context_length}, got T={T}"
        )
        if self._sinusoidal:
            pos = self.pos_emb_table          # [T, d_model], on device via buffer
        else:
            positions = torch.arange(T, device=ids.device)
            pos = self.pos_emb(positions)     # [T, d_model]
        emb = self.token_emb(ids) + pos                       # broadcasts over B
        flat = emb.reshape(B, T * self.cfg.d_model)           # [B, n_in]
        if self.cfg.parallel:
            logits = self.rwnn_L(flat) + self.rwnn_B(flat)    # [B, V]
        else:
            logits = self.rwnn(flat)                          # [B, V]
        return logits

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling starting from ``prompt_ids`` of shape ``[T0]`` or ``[B, T0]``.

        Returns the concatenation ``[prompt | generated]`` of shape
        ``[B, T0 + max_new_tokens]``.
        """
        self.eval()
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        B = prompt_ids.shape[0]
        T = self.cfg.context_length
        device = self.device

        out = prompt_ids.to(device)
        for _ in range(max_new_tokens):
            # Take last T tokens; left-pad with 0 if the prompt is shorter.
            if out.shape[1] >= T:
                ctx = out[:, -T:]
            else:
                pad = torch.zeros((B, T - out.shape[1]), dtype=out.dtype, device=device)
                ctx = torch.cat([pad, out], dim=1)
            logits = self(ctx)                        # [B, V]
            logits = logits / max(temperature, 1e-8)
            if top_k is not None and 0 < top_k < logits.shape[-1]:
                v, _ = torch.topk(logits, top_k, dim=-1)
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < thresh,
                                     torch.full_like(logits, float("-inf")),
                                     logits)
            probs = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)  # [B, 1]
            out = torch.cat([out, next_ids], dim=1)
        return out

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def num_parameters(self) -> dict[str, int]:
        tok = self.token_emb.weight.numel()
        pos = 0 if self._sinusoidal else self.pos_emb.weight.numel()
        if self.cfg.parallel:
            rwnn_L_p = self.rwnn_L.weights.numel()
            rwnn_B_p = self.rwnn_B.weights.numel()
            rwnn_p = rwnn_L_p + rwnn_B_p
            return {
                "token_embedding": tok,
                "pos_embedding": pos,
                "rwnn_L": rwnn_L_p,
                "rwnn_B": rwnn_B_p,
                "rwnn": rwnn_p,          # combined linear + bilinear
                "total": tok + pos + rwnn_p,
            }
        rwnn_p = self.rwnn.weights.numel()
        return {
            "token_embedding": tok,
            "pos_embedding": pos,
            "rwnn": rwnn_p,
            "total": tok + pos + rwnn_p,
        }


def _build_sinusoidal_table(T: int, d: int) -> torch.Tensor:
    """Vaswani-2017 closed-form positional encoding.

    PE(t, 2i)   = sin(t / 10000^(2i/d))
    PE(t, 2i+1) = cos(t / 10000^(2i/d))

    Different dimensions oscillate at log-spaced frequencies — fast at
    low indices, slow at high indices. The dot product
    PE(t) · PE(t+k) depends only on k, which gives the model an
    implicit relative-position structure even though the encoding
    itself is absolute. Returns a [T, d] float tensor.
    """
    pos = torch.arange(T).unsqueeze(1).float()             # [T, 1]
    half_d = d // 2
    i = torch.arange(half_d).float()                       # [d/2]
    div = torch.exp(-math.log(10000.0) * 2.0 * i / d)      # [d/2]
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(pos * div)                     # even dims
    pe[:, 1::2] = torch.cos(pos * div)[:, : d - half_d]    # odd dims
    return pe
