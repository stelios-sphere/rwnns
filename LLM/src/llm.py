"""RWNN-LLM: a small autoregressive language model with an RWNN core.

Architecture (one forward pass predicts ONE next token):

    ids  [B, T]                             T token ids, the context
      │  token embedding  (V, d_model)
      ▼
    emb  [B, T, d_model]
      │  flatten + in_proj  Linear(T·d_model -> n_in)
      ▼
    rwnn_in  [B, n_in]
      │  RWNN  (tanh hidden, linear output)
      ▼
    rwnn_out [B, n_out]
      │  out_proj  Linear(n_out -> V)
      ▼
    logits   [B, V]        logits for the next token

The three knobs ``context_length``, ``n_nodes``, ``n_layers`` (plus
``edge_prob`` for density) drive the RWNN topology: they are forwarded
straight to ``build_layered_rwnn`` along with ``n_in_rwnn`` and
``n_out_rwnn`` so the RWNN's inputs and outputs match the interface
layers around it.

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
    n_in_rwnn: int = 128            # RWNN input width
    n_out_rwnn: int = 128           # RWNN output width
    n_nodes: int = 1500             # total RWNN nodes (incl. in + bias + hidden + out)
    n_layers: int = 5               # RWNN topological depth
    edge_prob: float = 0.03         # RWNN connection density
    n_bias: int = 2                 # bias nodes inside RWNN
    seed: int = 0


class RWNNLM(nn.Module):
    def __init__(self, cfg: RWNNLMConfig, device: str | torch.device = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)

        # Build the RWNN core first so we can report its edge count.
        graph = build_layered_rwnn(
            n_nodes=cfg.n_nodes,
            edge_prob=cfg.edge_prob,
            n_layers=cfg.n_layers,
            n_in=cfg.n_in_rwnn,
            n_bias=cfg.n_bias,
            n_out=cfg.n_out_rwnn,
            seed=cfg.seed,
        )
        self.rwnn = RWNN(graph, device=self.device)

        # Interface layers.
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.in_proj = nn.Linear(cfg.context_length * cfg.d_model, cfg.n_in_rwnn)
        self.out_proj = nn.Linear(cfg.n_out_rwnn, cfg.vocab_size)

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
        emb = self.token_emb(ids)            # [B, T, d_model]
        flat = emb.reshape(B, T * self.cfg.d_model)
        x = self.in_proj(flat)               # [B, n_in]
        x = self.rwnn(x)                     # [B, n_out]
        logits = self.out_proj(x)            # [B, V]
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
        emb = self.token_emb.weight.numel()
        in_p = self.in_proj.weight.numel() + self.in_proj.bias.numel()
        rwnn_p = self.rwnn.weights.numel()
        out_p = self.out_proj.weight.numel() + self.out_proj.bias.numel()
        return {
            "embedding": emb,
            "in_projection": in_p,
            "rwnn": rwnn_p,
            "out_projection": out_p,
            "total": emb + in_p + rwnn_p + out_p,
        }
