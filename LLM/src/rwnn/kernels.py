"""Triton kernels for the random feed-forward DAG.

Four kernels, each a true custom GPU kernel (compiled to PTX / SASS by
Triton's LLVM pipeline, same machine code class as hand-written CUDA):

1. ``forward_level_kernel``       — one forward sweep over one topological level.
2. ``backward_dpre_kernel``       — d_pre = d_a * tanh'(pre) (or 1 for linear output).
3. ``backward_propagate_kernel``  — atomic-accumulate d_a of parents from d_pre.
4. ``backward_weight_kernel``     — per-edge weight gradient (batch reduction).

Node kinds (per-destination dispatch inside each kernel):
    KIND_LINEAR      = 0:  pre = Σ_k  w_k · a_{p_k}
    KIND_BILINEAR    = 1:  pre = Σ_pairs  σ(w_g · a_g) · w_v · a_v
                          (parents iterated as paired (gate, value);
                           each pair contributes one gated value;
                           tanh applied at the output for stability.)
    KIND_PRODUCT     = 2:  pre = Σ_pairs  a_g · a_v
                          (parents in pairs; pure activation product,
                           no weights. Symmetric in its two parents.)
    KIND_SOFTMAX_AGG = 3:  pre = Σ_k  softmax(scores)_k · value_k
                          (parents in (score, value) pairs; softmax
                           taken across the K pairs; no weights;
                           internal max-subtract for stability.)

Activation codes (the *output* nonlinearity, applied after the
node-kind-specific accumulation):
    ACT_LINEAR = 0
    ACT_TANH   = 1
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
    a_ptr,                 # float32 [N, B] — activations (written for this level)
    pre_ptr,               # float32 [N, B] — pre-activations (written for this level)
    weights_ptr,           # float32 [E_live]   — only weighted edges (kind 0,1 destinations)
    level_nodes_ptr,       # int32   [n_level]
    parent_offsets_ptr,    # int32   [N+1]
    parent_ids_ptr,        # int32   [E]
    node_kinds_ptr,        # int8    [N] — 0 linear, 1 bilinear, 2 product, 3 attention
    node_weight_offsets_ptr,  # int32 [N+1] — start of node i's weights in weights_ptr
    B,                     # int — batch size
    BLOCK_B: tl.constexpr,
    ACT: tl.constexpr,     # 0 = linear, 1 = tanh
):
    pid_node = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    i = tl.load(level_nodes_ptr + pid_node).to(tl.int64)
    start = tl.load(parent_offsets_ptr + i).to(tl.int64)
    end = tl.load(parent_offsets_ptr + i + 1).to(tl.int64)
    kind = tl.load(node_kinds_ptr + i).to(tl.int32)
    w_start = tl.load(node_weight_offsets_ptr + i).to(tl.int64)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B

    acc = tl.zeros([BLOCK_B], dtype=tl.float32)

    if kind == 0:
        # LINEAR: simple weighted sum over parents.
        for k in range(start, end):
            j = tl.load(parent_ids_ptr + k).to(tl.int64)
            w = tl.load(weights_ptr + w_start + (k - start))
            a_j = tl.load(a_ptr + j * B + offs_b, mask=mask_b, other=0.0)
            acc += w * a_j
    elif kind == 1:
        # BILINEAR: parents in (gate, value) pairs, step by 2.
        for k in range(start, end, 2):
            jg = tl.load(parent_ids_ptr + k).to(tl.int64)
            jv = tl.load(parent_ids_ptr + k + 1).to(tl.int64)
            wg = tl.load(weights_ptr + w_start + (k - start))
            wv = tl.load(weights_ptr + w_start + (k - start) + 1)
            a_g = tl.load(a_ptr + jg * B + offs_b, mask=mask_b, other=0.0)
            a_v = tl.load(a_ptr + jv * B + offs_b, mask=mask_b, other=0.0)
            zg = wg * a_g
            g = 1.0 / (1.0 + tl.exp(-zg))   # sigmoid
            acc += g * (wv * a_v)
    elif kind == 2:
        # PRODUCT: pairs (g, v), no weights — pure activation product.
        for k in range(start, end, 2):
            jg = tl.load(parent_ids_ptr + k).to(tl.int64)
            jv = tl.load(parent_ids_ptr + k + 1).to(tl.int64)
            a_g = tl.load(a_ptr + jg * B + offs_b, mask=mask_b, other=0.0)
            a_v = tl.load(a_ptr + jv * B + offs_b, mask=mask_b, other=0.0)
            acc += a_g * a_v
    else:
        # SOFTMAX_AGG (kind == 3): pairs (score, value).
        # 3-pass softmax with max-subtract for numerical stability:
        #   pass 1: m   = max_k(score_k)
        #   pass 2: Z   = Σ_k exp(score_k - m)
        #   pass 3: pre = Σ_k (exp(score_k - m) / Z) · value_k
        m = tl.full([BLOCK_B], -1.0e30, dtype=tl.float32)
        for k in range(start, end, 2):
            js = tl.load(parent_ids_ptr + k).to(tl.int64)
            a_s = tl.load(a_ptr + js * B + offs_b, mask=mask_b, other=-1.0e30)
            m = tl.maximum(m, a_s)
        Z = tl.zeros([BLOCK_B], dtype=tl.float32)
        for k in range(start, end, 2):
            js = tl.load(parent_ids_ptr + k).to(tl.int64)
            a_s = tl.load(a_ptr + js * B + offs_b, mask=mask_b, other=-1.0e30)
            Z += tl.exp(a_s - m)
        for k in range(start, end, 2):
            js = tl.load(parent_ids_ptr + k).to(tl.int64)
            jv = tl.load(parent_ids_ptr + k + 1).to(tl.int64)
            a_s = tl.load(a_ptr + js * B + offs_b, mask=mask_b, other=-1.0e30)
            a_v = tl.load(a_ptr + jv * B + offs_b, mask=mask_b, other=0.0)
            w = tl.exp(a_s - m) / Z
            acc += w * a_v

    row = i * B + offs_b
    tl.store(pre_ptr + row, acc, mask=mask_b)

    if ACT == 1:
        out = libdevice.tanh(acc)
    else:
        out = acc
    tl.store(a_ptr + row, out, mask=mask_b)


# ---------------------------------------------------------------------------
# Backward: d_pre = d_a * (1 - tanh^2(pre))
# (No node-kind branching — output nonlinearity is the same for both kinds.)
# ---------------------------------------------------------------------------

@triton.jit
def backward_dpre_kernel(
    d_a_ptr,
    pre_ptr,
    d_pre_ptr,
    level_nodes_ptr,
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
    d_a_ptr,                # float32 [N, B] — atomically incremented at parent rows
    a_ptr,                  # float32 [N, B] — needed to recompute g, v, scores
    pre_ptr,                # float32 [N, B] — needed for softmax-agg backward
    d_pre_ptr,              # float32 [N, B]
    weights_ptr,            # float32 [E_live]
    level_nodes_ptr,        # int32 [n_level]
    parent_offsets_ptr,     # int32 [N+1]
    parent_ids_ptr,         # int32 [E]
    node_kinds_ptr,         # int8  [N]
    node_weight_offsets_ptr,  # int32 [N+1]
    B,
    BLOCK_B: tl.constexpr,
):
    pid_node = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    i = tl.load(level_nodes_ptr + pid_node).to(tl.int64)
    start = tl.load(parent_offsets_ptr + i).to(tl.int64)
    end = tl.load(parent_offsets_ptr + i + 1).to(tl.int64)
    kind = tl.load(node_kinds_ptr + i).to(tl.int32)
    w_start = tl.load(node_weight_offsets_ptr + i).to(tl.int64)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B

    d_pre_i = tl.load(d_pre_ptr + i * B + offs_b, mask=mask_b, other=0.0)

    if kind == 0:
        # LINEAR: each edge contributes w * d_pre to its parent.
        for k in range(start, end):
            j = tl.load(parent_ids_ptr + k).to(tl.int64)
            w = tl.load(weights_ptr + w_start + (k - start))
            tl.atomic_add(d_a_ptr + j * B + offs_b, w * d_pre_i, mask=mask_b)
    elif kind == 1:
        # BILINEAR: each (gate, value) pair contributes
        #   d/d a_g  =  w_g · σ'(z_g) · v ,    v = w_v · a_v
        #   d/d a_v  =  σ(z_g) · w_v
        # all multiplied by d_pre_i.
        for k in range(start, end, 2):
            jg = tl.load(parent_ids_ptr + k).to(tl.int64)
            jv = tl.load(parent_ids_ptr + k + 1).to(tl.int64)
            wg = tl.load(weights_ptr + w_start + (k - start))
            wv = tl.load(weights_ptr + w_start + (k - start) + 1)
            a_g = tl.load(a_ptr + jg * B + offs_b, mask=mask_b, other=0.0)
            a_v = tl.load(a_ptr + jv * B + offs_b, mask=mask_b, other=0.0)
            zg = wg * a_g
            g = 1.0 / (1.0 + tl.exp(-zg))
            sig_deriv = g * (1.0 - g)
            v = wv * a_v
            d_a_g = wg * sig_deriv * v * d_pre_i
            d_a_v = g * wv * d_pre_i
            tl.atomic_add(d_a_ptr + jg * B + offs_b, d_a_g, mask=mask_b)
            tl.atomic_add(d_a_ptr + jv * B + offs_b, d_a_v, mask=mask_b)
    elif kind == 2:
        # PRODUCT: ∂(a_g · a_v)/∂a_g = a_v, ∂/∂a_v = a_g.
        for k in range(start, end, 2):
            jg = tl.load(parent_ids_ptr + k).to(tl.int64)
            jv = tl.load(parent_ids_ptr + k + 1).to(tl.int64)
            a_g = tl.load(a_ptr + jg * B + offs_b, mask=mask_b, other=0.0)
            a_v = tl.load(a_ptr + jv * B + offs_b, mask=mask_b, other=0.0)
            tl.atomic_add(d_a_ptr + jg * B + offs_b, a_v * d_pre_i, mask=mask_b)
            tl.atomic_add(d_a_ptr + jv * B + offs_b, a_g * d_pre_i, mask=mask_b)
    else:
        # SOFTMAX_AGG (kind == 3): pairs (score, value).
        #   pre  = Σ_k w_k · v_k,  w_k = exp(s_k - m) / Z
        #   ∂pre/∂v_k = w_k
        #   ∂pre/∂s_k = w_k · (v_k - pre)
        # Recompute m, Z (3-pass softmax) and read pre from saved buffer.
        pre_i = tl.load(pre_ptr + i * B + offs_b, mask=mask_b, other=0.0)
        m = tl.full([BLOCK_B], -1.0e30, dtype=tl.float32)
        for k in range(start, end, 2):
            js = tl.load(parent_ids_ptr + k).to(tl.int64)
            a_s = tl.load(a_ptr + js * B + offs_b, mask=mask_b, other=-1.0e30)
            m = tl.maximum(m, a_s)
        Z = tl.zeros([BLOCK_B], dtype=tl.float32)
        for k in range(start, end, 2):
            js = tl.load(parent_ids_ptr + k).to(tl.int64)
            a_s = tl.load(a_ptr + js * B + offs_b, mask=mask_b, other=-1.0e30)
            Z += tl.exp(a_s - m)
        for k in range(start, end, 2):
            js = tl.load(parent_ids_ptr + k).to(tl.int64)
            jv = tl.load(parent_ids_ptr + k + 1).to(tl.int64)
            a_s = tl.load(a_ptr + js * B + offs_b, mask=mask_b, other=-1.0e30)
            a_v = tl.load(a_ptr + jv * B + offs_b, mask=mask_b, other=0.0)
            w = tl.exp(a_s - m) / Z
            d_a_v = w * d_pre_i
            d_a_s = w * (a_v - pre_i) * d_pre_i
            tl.atomic_add(d_a_ptr + js * B + offs_b, d_a_s, mask=mask_b)
            tl.atomic_add(d_a_ptr + jv * B + offs_b, d_a_v, mask=mask_b)


# ---------------------------------------------------------------------------
# Backward: per-edge weight gradient
# ---------------------------------------------------------------------------

@triton.jit
def backward_weight_kernel(
    a_ptr,                  # float32 [N, B]
    d_pre_ptr,              # float32 [N, B]
    weights_ptr,            # float32 [E_live]   — needed for bilinear partner weight
    edge_src_ptr,           # int32 [E]
    edge_dst_ptr,           # int32 [E]
    parent_offsets_ptr,     # int32 [N+1]    — to compute partner edge index
    parent_ids_ptr,         # int32 [E]      — to look up partner src
    node_kinds_ptr,         # int8  [N]
    node_weight_offsets_ptr,  # int32 [N+1]
    d_w_ptr,                # float32 [E_live]   — gradient for compact weights
    B,
    BLOCK_B: tl.constexpr,
):
    e = tl.program_id(axis=0)

    src = tl.load(edge_src_ptr + e).to(tl.int64)
    dst = tl.load(edge_dst_ptr + e).to(tl.int64)
    kind = tl.load(node_kinds_ptr + dst).to(tl.int32)

    # PRODUCT (2) and SOFTMAX_AGG (3): no weight allocated, no gradient
    # to compute. Skip entirely (no store needed — weights tensor doesn't
    # have a slot for this edge).
    if kind >= 2:
        return

    dst_parent_start = tl.load(parent_offsets_ptr + dst).to(tl.int64)
    dst_w_start = tl.load(node_weight_offsets_ptr + dst).to(tl.int64)
    pos_in_dst = e - dst_parent_start
    w_idx = dst_w_start + pos_in_dst

    if kind == 0:
        # LINEAR: d w_e = Σ_b a[src, b] * d_pre[dst, b]
        acc = tl.zeros([BLOCK_B], dtype=tl.float32)
        off = 0
        while off < B:
            offs_b = off + tl.arange(0, BLOCK_B)
            mask_b = offs_b < B
            a_src = tl.load(a_ptr + src * B + offs_b, mask=mask_b, other=0.0)
            d_pre_dst = tl.load(d_pre_ptr + dst * B + offs_b, mask=mask_b, other=0.0)
            acc += a_src * d_pre_dst
            off += BLOCK_B
        total = tl.sum(acc, axis=0)
        tl.store(d_w_ptr + w_idx, total)
    else:
        # BILINEAR: depends on whether this edge is the gate (even pos in
        # parent list) or the value (odd pos). Partner edge gives us the
        # other side of the pair.
        is_gate = (pos_in_dst % 2) == 0
        if is_gate:
            partner_e = e + 1
            partner_w_idx = w_idx + 1
        else:
            partner_e = e - 1
            partner_w_idx = w_idx - 1
        partner_src = tl.load(parent_ids_ptr + partner_e).to(tl.int64)
        w_partner = tl.load(weights_ptr + partner_w_idx)
        w_self = tl.load(weights_ptr + w_idx)

        acc = tl.zeros([BLOCK_B], dtype=tl.float32)
        off = 0
        while off < B:
            offs_b = off + tl.arange(0, BLOCK_B)
            mask_b = offs_b < B
            a_self = tl.load(a_ptr + src * B + offs_b, mask=mask_b, other=0.0)
            a_partner = tl.load(a_ptr + partner_src * B + offs_b,
                                mask=mask_b, other=0.0)
            d_pre_dst = tl.load(d_pre_ptr + dst * B + offs_b,
                                mask=mask_b, other=0.0)
            if is_gate:
                # d w_g  =  Σ_b a_g · σ'(z_g) · v · d_pre
                z_g = w_self * a_self
                g = 1.0 / (1.0 + tl.exp(-z_g))
                sig_deriv = g * (1.0 - g)
                v = w_partner * a_partner
                acc += a_self * sig_deriv * v * d_pre_dst
            else:
                # d w_v  =  Σ_b σ(z_g) · a_v · d_pre
                z_g = w_partner * a_partner
                g = 1.0 / (1.0 + tl.exp(-z_g))
                acc += g * a_self * d_pre_dst
            off += BLOCK_B
        total = tl.sum(acc, axis=0)
        tl.store(d_w_ptr + w_idx, total)
