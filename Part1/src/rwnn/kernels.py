"""Triton kernels for the random feed-forward DAG.

Four kernels, each a true custom GPU kernel (compiled to PTX / SASS by
Triton's LLVM pipeline, same machine code class as hand-written CUDA):

1. ``forward_level_kernel``       — one forward sweep over one topological level.
2. ``backward_dpre_kernel``       — d_pre = d_a * sigma'(pre) for one level.
3. ``backward_propagate_kernel``  — atomic-accumulate d_a of parents from d_pre of a level.
4. ``backward_weight_kernel``     — per-edge weight gradient (batch reduction).

Activation codes:
    0 = linear
    1 = tanh
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

@triton.jit
def forward_level_kernel(
    a_ptr,               # float32 [N, B] — activations (written for this level)
    pre_ptr,             # float32 [N, B] — pre-activations (written for this level)
    weights_ptr,         # float32 [E]
    level_nodes_ptr,     # int32   [n_level]
    parent_offsets_ptr,  # int32   [N+1]
    parent_ids_ptr,      # int32   [E]
    B,                   # int — batch size
    BLOCK_B: tl.constexpr,
    ACT: tl.constexpr,   # 0 = linear, 1 = tanh
):
    pid_node = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    i = tl.load(level_nodes_ptr + pid_node).to(tl.int64)
    start = tl.load(parent_offsets_ptr + i).to(tl.int64)
    end = tl.load(parent_offsets_ptr + i + 1).to(tl.int64)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B

    acc = tl.zeros([BLOCK_B], dtype=tl.float32)

    # Dynamic loop over this node's parents in CSR order. Weights are
    # stored in the same order so the kernel streams both sequentially.
    for k in range(start, end):
        j = tl.load(parent_ids_ptr + k).to(tl.int64)
        w = tl.load(weights_ptr + k)
        a_j = tl.load(a_ptr + j * B + offs_b, mask=mask_b, other=0.0)
        acc += w * a_j

    row = i * B + offs_b
    tl.store(pre_ptr + row, acc, mask=mask_b)

    if ACT == 1:
        out = libdevice.tanh(acc)
    else:
        out = acc
    tl.store(a_ptr + row, out, mask=mask_b)


# ---------------------------------------------------------------------------
# Backward: d_pre = d_a * sigma'(pre)
# ---------------------------------------------------------------------------

@triton.jit
def backward_dpre_kernel(
    d_a_ptr,             # float32 [N, B] — upstream grad of activations
    pre_ptr,             # float32 [N, B]
    d_pre_ptr,           # float32 [N, B] — output
    level_nodes_ptr,     # int32 [n_level]
    B,
    BLOCK_B: tl.constexpr,
    ACT: tl.constexpr,
):
    pid_node = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    i = tl.load(level_nodes_ptr + pid_node).to(tl.int64)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    row = i * B + offs_b

    d_a = tl.load(d_a_ptr + row, mask=mask_b, other=0.0)

    if ACT == 1:
        pre = tl.load(pre_ptr + row, mask=mask_b, other=0.0)
        t = libdevice.tanh(pre)
        dsig = 1.0 - t * t
        d_pre = d_a * dsig
    else:
        d_pre = d_a

    tl.store(d_pre_ptr + row, d_pre, mask=mask_b)


# ---------------------------------------------------------------------------
# Backward: propagate d_pre of this level into d_a of parents
# ---------------------------------------------------------------------------

@triton.jit
def backward_propagate_kernel(
    d_a_ptr,             # float32 [N, B] — atomically incremented at parent rows
    d_pre_ptr,           # float32 [N, B]
    weights_ptr,         # float32 [E]
    level_nodes_ptr,     # int32 [n_level]
    parent_offsets_ptr,  # int32 [N+1]
    parent_ids_ptr,      # int32 [E]
    B,
    BLOCK_B: tl.constexpr,
):
    pid_node = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    i = tl.load(level_nodes_ptr + pid_node).to(tl.int64)
    start = tl.load(parent_offsets_ptr + i).to(tl.int64)
    end = tl.load(parent_offsets_ptr + i + 1).to(tl.int64)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B

    d_pre_i = tl.load(d_pre_ptr + i * B + offs_b, mask=mask_b, other=0.0)

    for k in range(start, end):
        j = tl.load(parent_ids_ptr + k).to(tl.int64)
        w = tl.load(weights_ptr + k)
        # Multiple distinct children in different level invocations may
        # share this parent — accumulate atomically.
        tl.atomic_add(d_a_ptr + j * B + offs_b, w * d_pre_i, mask=mask_b)


# ---------------------------------------------------------------------------
# Backward: per-edge weight gradient
# ---------------------------------------------------------------------------

@triton.jit
def backward_weight_kernel(
    a_ptr,           # float32 [N, B]
    d_pre_ptr,       # float32 [N, B]
    edge_src_ptr,    # int32 [E]
    edge_dst_ptr,    # int32 [E]
    d_w_ptr,         # float32 [E] — output
    B,
    BLOCK_B: tl.constexpr,
):
    e = tl.program_id(axis=0)

    src = tl.load(edge_src_ptr + e).to(tl.int64)
    dst = tl.load(edge_dst_ptr + e).to(tl.int64)

    acc = tl.zeros([BLOCK_B], dtype=tl.float32)
    # Chunked reduction over the batch dimension.
    off = 0
    while off < B:
        offs_b = off + tl.arange(0, BLOCK_B)
        mask_b = offs_b < B
        a_src = tl.load(a_ptr + src * B + offs_b, mask=mask_b, other=0.0)
        d_pre_dst = tl.load(d_pre_ptr + dst * B + offs_b, mask=mask_b, other=0.0)
        acc += a_src * d_pre_dst
        off += BLOCK_B

    total = tl.sum(acc, axis=0)
    tl.store(d_w_ptr + e, total)
