"""The autograd.Function wrapping the Triton kernels, plus an nn.Module wrapper."""

from __future__ import annotations

import math
from typing import Any

import torch
import triton

from .graph import RandomDAG
from .kernels import (
    backward_dpre_kernel,
    backward_propagate_kernel,
    backward_weight_kernel,
    forward_level_kernel,
)


_ACT_TANH = 1
_ACT_LINEAR = 0
_BLOCK_B = 128


class RWNNFunction(torch.autograd.Function):
    """Custom autograd.Function that drives the four Triton kernels.

    The graph tensors are passed as a single tuple to keep the Function
    signature compact; only ``x`` and ``weights`` are differentiable
    w.r.t. loss (and only ``weights`` receives gradients in practice —
    input grads are produced too, for completeness / gradcheck).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,             # [B, n_in]
        weights: torch.Tensor,       # [E_live]
        parent_offsets: torch.Tensor,
        parent_ids: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        level_nodes: torch.Tensor,
        level_starts: torch.Tensor,
        level_is_output: torch.Tensor,
        input_ids: torch.Tensor,
        bias_ids: torch.Tensor,
        output_ids: torch.Tensor,
        node_kinds: torch.Tensor,
        node_weight_offsets: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        assert x.is_cuda and weights.is_cuda, "RWNN requires CUDA tensors"
        assert x.dtype == torch.float32 and weights.dtype == torch.float32

        B = x.shape[0]
        N = n_nodes
        device = x.device

        a = torch.zeros((N, B), device=device, dtype=torch.float32)
        pre = torch.zeros((N, B), device=device, dtype=torch.float32)

        # Inputs: scatter x.T into the input rows.
        a.index_copy_(0, input_ids.long(), x.transpose(0, 1).contiguous())
        # Bias rows = 1.
        if bias_ids.numel() > 0:
            a.index_fill_(0, bias_ids.long(), 1.0)

        L = int(level_starts.numel()) - 1
        # Level 0 is inputs + biases — nothing to compute, values already set.
        for ℓ in range(1, L):
            s = int(level_starts[ℓ].item())
            e = int(level_starts[ℓ + 1].item())
            n_level = e - s
            if n_level == 0:
                continue
            level_slice = level_nodes[s:e].contiguous()
            act = _ACT_LINEAR if bool(level_is_output[ℓ].item()) else _ACT_TANH
            grid = (n_level, triton.cdiv(B, _BLOCK_B))
            forward_level_kernel[grid](
                a,
                pre,
                weights,
                level_slice,
                parent_offsets,
                parent_ids,
                node_kinds,
                node_weight_offsets,
                B,
                BLOCK_B=_BLOCK_B,
                ACT=act,
            )

        out = a.index_select(0, output_ids.long()).transpose(0, 1).contiguous()

        ctx.save_for_backward(
            a, pre, weights,
            parent_offsets, parent_ids, edge_src, edge_dst,
            level_nodes, level_starts, level_is_output,
            input_ids, output_ids, node_kinds, node_weight_offsets,
        )
        ctx.B = B
        ctx.N = N
        ctx.n_edges = int(edge_src.numel())
        ctx.n_live_edges = int(weights.numel())
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple:
        (a, pre, weights,
         parent_offsets, parent_ids, edge_src, edge_dst,
         level_nodes, level_starts, level_is_output,
         input_ids, output_ids, node_kinds,
         node_weight_offsets) = ctx.saved_tensors
        B = ctx.B
        N = ctx.N
        E = ctx.n_edges
        E_live = ctx.n_live_edges
        device = a.device

        d_a = torch.zeros((N, B), device=device, dtype=torch.float32)
        d_pre = torch.zeros((N, B), device=device, dtype=torch.float32)

        # Seed d_a at the output rows with grad_out^T.
        d_a.index_copy_(0, output_ids.long(), grad_out.transpose(0, 1).contiguous())

        L = int(level_starts.numel()) - 1
        # Reverse sweep over compute levels. Level 0 is input/bias (no compute).
        for ℓ in range(L - 1, 0, -1):
            s = int(level_starts[ℓ].item())
            e = int(level_starts[ℓ + 1].item())
            n_level = e - s
            if n_level == 0:
                continue
            level_slice = level_nodes[s:e].contiguous()
            act = _ACT_LINEAR if bool(level_is_output[ℓ].item()) else _ACT_TANH
            grid = (n_level, triton.cdiv(B, _BLOCK_B))

            backward_dpre_kernel[grid](
                d_a, pre, d_pre, level_slice,
                B, BLOCK_B=_BLOCK_B, ACT=act,
            )
            backward_propagate_kernel[grid](
                d_a, a, pre, d_pre, weights, level_slice,
                parent_offsets, parent_ids, node_kinds, node_weight_offsets,
                B, BLOCK_B=_BLOCK_B,
            )

        # Per-(live-)edge weight gradient. d_w has compact shape [E_live].
        # Kernel grid still iterates all E edges; for kind 2/3 destinations
        # the kernel returns early without writing.
        d_w = torch.zeros(E_live, device=device, dtype=torch.float32)
        if E > 0:
            backward_weight_kernel[(E,)](
                a, d_pre, weights, edge_src, edge_dst,
                parent_offsets, parent_ids, node_kinds, node_weight_offsets,
                d_w,
                B, BLOCK_B=_BLOCK_B,
            )

        # Input gradient, mainly for gradcheck.
        d_x = d_a.index_select(0, input_ids.long()).transpose(0, 1).contiguous()

        # Return grads for every argument of forward() in order.
        return (d_x, d_w,
                None, None, None, None, None, None, None,
                None, None, None, None, None, None)


class RWNN(torch.nn.Module):
    """Randomly Wired Neural Network — feed-forward DAG, tanh hidden, linear output.

    The graph topology is fixed at construction time (built by
    ``build_random_dag``); only the per-edge weights are learnable.
    """

    def __init__(self, graph: RandomDAG, device: torch.device | str = "cuda"):
        super().__init__()
        device = torch.device(device)
        graph = graph.to(device)
        self._n_nodes = graph.n_nodes
        self._n_edges = graph.n_edges
        self._n_live_edges = graph.n_live_edges
        self._n_levels = graph.n_levels
        self._n_in = graph.n_in
        self._n_bias = graph.n_bias
        self._n_hidden = graph.n_hidden
        self._n_out = graph.n_out

        # Per-destination 1/√fan_in init, only for *live* edges (i.e. those
        # going into kind 0 or 1 destinations). Product (kind 2) and
        # softmax-aggregator (kind 3) nodes have no learnable per-edge
        # weights; we don't allocate slots for them.
        edge_dst_long = graph.edge_dst.long()
        fan_in_per_edge = graph.fan_in[edge_dst_long].clamp(min=1).float()
        edge_kind = graph.node_kinds.to(torch.int64)[edge_dst_long]
        is_live_edge = (edge_kind <= 1)
        live_fan_in = fan_in_per_edge[is_live_edge]
        scale = (1.0 / live_fan_in.sqrt()) * math.sqrt(3.0)
        n_live = int(is_live_edge.sum().item())
        assert n_live == graph.n_live_edges, (
            f"live edge count mismatch: {n_live} vs {graph.n_live_edges}"
        )
        w = (torch.rand(n_live, device=device) * 2 - 1) * scale
        self.weights = torch.nn.Parameter(w.contiguous())

        # Graph tensors as non-learnable buffers.
        self.register_buffer("parent_offsets", graph.parent_offsets.contiguous())
        self.register_buffer("parent_ids", graph.parent_ids.contiguous())
        self.register_buffer("edge_src", graph.edge_src.contiguous())
        self.register_buffer("edge_dst", graph.edge_dst.contiguous())
        self.register_buffer("level_nodes", graph.level_nodes.contiguous())
        self.register_buffer("level_starts", graph.level_starts.contiguous())
        self.register_buffer("level_is_output", graph.level_is_output.contiguous())
        self.register_buffer("input_ids", graph.input_ids.contiguous())
        self.register_buffer("bias_ids", graph.bias_ids.contiguous())
        self.register_buffer("output_ids", graph.output_ids.contiguous())
        self.register_buffer("node_kinds", graph.node_kinds.contiguous())
        self.register_buffer("node_weight_offsets",
                             graph.node_weight_offsets.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return RWNNFunction.apply(
            x.contiguous(),
            self.weights,
            self.parent_offsets, self.parent_ids,
            self.edge_src, self.edge_dst,
            self.level_nodes, self.level_starts, self.level_is_output,
            self.input_ids, self.bias_ids, self.output_ids,
            self.node_kinds, self.node_weight_offsets,
            self._n_nodes,
        )

    @property
    def n_edges(self) -> int:
        return self._n_edges

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    @property
    def n_levels(self) -> int:
        return self._n_levels

    @property
    def n_in(self) -> int:
        return self._n_in

    @property
    def n_bias(self) -> int:
        return self._n_bias

    @property
    def n_hidden(self) -> int:
        return self._n_hidden

    @property
    def n_out(self) -> int:
        return self._n_out
