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
    n_bias: int = 2                 # bias nodes inside RWNN
    seed: int = 0


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

        graph = build_layered_rwnn(
            n_nodes=cfg.n_nodes,
            edge_prob=cfg.edge_prob,
            n_layers=cfg.n_layers,
            n_in=n_in,
            n_bias=cfg.n_bias,
            n_out=n_out,
            seed=cfg.seed,
            bilinear_fraction=cfg.bilinear_fraction,
        )
        self.rwnn = RWNN(graph, device=self.device)

        # Token embedding + learnable absolute positional embedding.
        # Without positional embeddings, two tokens of the same id at
        # different positions have identical 48-dim embeddings; the
        # only positional signal reaches the RWNN through *which* input
        # nodes a value lands at — easy to wash out under random
        # connectivity. Adding a per-position learnable vector gives
        # each (position, dim) input node a unique signature.
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)

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
        positions = torch.arange(T, device=ids.device)
        emb = self.token_emb(ids) + self.pos_emb(positions)  # [B, T, d_model]
        flat = emb.reshape(B, T * self.cfg.d_model)          # [B, n_in]
        logits = self.rwnn(flat)                             # [B, V]
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
        pos = self.pos_emb.weight.numel()
        rwnn_p = self.rwnn.weights.numel()
        return {
            "token_embedding": tok,
            "pos_embedding": pos,
            "rwnn": rwnn_p,
            "total": tok + pos + rwnn_p,
        }
