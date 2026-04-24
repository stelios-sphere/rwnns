# Part 1 — Implementation Notes

This document is the design record for the Part 1 implementation. Read
`README.md` first for what the article is about; this file is about how
we turn that into GPU code.

## The feed-forward logic we must reproduce

From the article:

- Nodes have unique integer indices `0 … N-1`.
- **Input nodes** and **bias nodes** have only outgoing edges.
- **Output nodes** have only incoming edges.
- **Hidden** (and output) nodes with index `i` may only receive edges
  from nodes with strictly lower index `j < i`. This DAG rule is the
  sole topological constraint; within it the wiring is random.

For a hidden/output node `i` with parent set `P(i) ⊆ {0, …, i-1}`:

```
pre_i[b] = Σ_{j ∈ P(i)} w_{j→i} · a_j[b]
a_i[b]   = σ( pre_i[b] )
```

`a_j[b]` is the already-computed activation of parent `j` for batch
element `b`. Bias nodes have `a = 1` constantly. No separate per-node
bias parameter — bias comes in through edges from bias nodes.

`σ` is **tanh** for hidden nodes and **identity** (linear) for output
nodes, which is the natural choice for the regression target
`f(x) = x₁² + x₂²`.

Because parents have strictly lower indices, iterating
`i = 0, 1, 2, …` in index order is a valid topological order. For GPU
parallelism we go one better — we precompute **topological levels**:

```
level(i) = 0                         if i is input or bias
         = 1 + max_{j ∈ P(i)} level(j)   otherwise
```

All nodes at the same level can be computed in parallel; their inputs
live exclusively at earlier levels. The forward pass is a sweep over
levels `1 → L_max`. The backward pass is the same sweep in reverse.

## Implementation shape

```
Part1/
├── README.md              # what the article is about
├── IMPLEMENTATION.md      # this file — design record
├── requirements.txt
├── rwnn/
│   ├── __init__.py
│   ├── graph.py           # random DAG builder + CSR + levels
│   ├── kernels.py         # Triton kernels (custom GPU code)
│   └── model.py           # autograd.Function + nn.Module
├── test_kernels.py        # correctness vs. pure-PyTorch reference
└── train.py               # the Part 1 experiment
```

## Why Triton, not raw `.cu`

The target machine has PyTorch 2.10 + CUDA 12.8 runtime + Triton 3.6
but **no `nvcc` toolchain** installed. Triton compiles Python-authored
kernels directly to PTX / SASS via LLVM — it is not a wrapper over
cuBLAS or ATen, the emitted machine code is the same kind that `nvcc`
would produce from hand-written `__global__` kernels. FlashAttention,
PyTorch's own custom ops, and countless modern GPU kernels are written
in it.

If we need `.cu` files later, the kernel logic translates nearly
one-to-one — only the surrounding build system changes.

## Graph representation

Built once on the host, moved to device as int/float tensors.

- `n_in`, `n_bias`, `n_hidden`, `n_out`, total `N`.
- **Parent CSR** (how the forward kernel reads the graph):
  - `parent_offsets: int32[N+1]` — for node `i`, its parent list lives at
    `parent_ids[parent_offsets[i] : parent_offsets[i+1]]`.
  - `parent_ids: int32[E]` — parent node index for each incoming edge.
  - Weights `w: float32[E]` are stored in the **same CSR order** as
    `parent_ids`, so the kernel's inner loop reads `parent_ids[k]` and
    `w[k]` together with coalesced access.
- **Edge tables** (how the weight-gradient kernel iterates):
  - `edge_src: int32[E]`, `edge_dst: int32[E]` — destination-sorted,
    matching the weight order.
- **Levels** (how the forward / backward drivers schedule work):
  - `level_nodes: int32[N]` — all node indices, sorted by level.
  - `level_starts: int32[L_max+2]` — offsets, so level `ℓ` is
    `level_nodes[level_starts[ℓ] : level_starts[ℓ+1]]`.
  - Level 0 is always inputs + biases (no compute beyond setting values).
  - Output nodes are all at the highest level(s) and use linear
    activation; hidden nodes use tanh.

## Activation memory

`a: float32[N, B]` and `pre: float32[N, B]`, node-major. Batch is the
contiguous dimension so the inner loop in each kernel does coalesced
loads across a block of batch lanes.

## The four kernels

### 1. `forward_level_kernel(ACT)`

Grid: `(n_level_nodes, ceil(B / BLOCK_B))`. Each program handles one
node at this level and one batch block.

```
i     = level_nodes[pid_node]
start = parent_offsets[i]
end   = parent_offsets[i+1]
acc   = 0
for k in [start, end):
    j = parent_ids[k]
    w = weights[k]
    acc += w * a[j, offs_b]
pre[i, offs_b] = acc
a[i, offs_b]   = σ_ACT(acc)
```

Launched once per level with `ACT = tanh` for hidden levels, `ACT =
linear` for the output level.

### 2. `backward_dpre_kernel(ACT)`

Grid same as forward. Given upstream `d_a`, compute
`d_pre = d_a · σ'(pre)`.

`σ'` for tanh: `1 - tanh(pre)^2`. For linear: `1`. We compute `tanh(pre)`
directly rather than reading `a` back — slightly cheaper and avoids
ambiguity if `a` ever gets overwritten in-place.

### 3. `backward_propagate_kernel`

Grid same as forward. For each node `i` at this level, read its
`d_pre`, loop over parents, and **atomically add** `w · d_pre_i` into
`d_a[parent, offs_b]`. Atomics are needed because multiple children
across different levels (already processed) may share a parent — the
partial contributions accumulate.

### 4. `backward_weight_kernel`

Grid: `(E,)`, one program per edge. Each program chunks the batch into
tiles of `BLOCK_B`, computes `Σ_b a[src, b] · d_pre[dst, b]` locally,
writes the reduced scalar to `d_w[edge]`. No atomics — one writer per
edge.

## Driver (`RWNNFunction`)

Forward:
1. Allocate `a`, `pre` as `[N, B]` zeros.
2. Scatter `x.T` into `a[input_rows, :]`; set `a[bias_rows, :] = 1`.
3. For `ℓ = 1 … L_max`, launch `forward_level_kernel` with the
   appropriate activation.
4. Return `a[output_rows, :].T` (shape `[B, n_out]`).
5. Save `a`, `pre`, `weights`, graph tensors for backward.

Backward:
1. Allocate `d_a`, `d_pre` as zeros. Scatter `grad_output.T` into
   `d_a[output_rows, :]`.
2. For `ℓ = L_max … 1`:
   - `backward_dpre_kernel` for that level.
   - `backward_propagate_kernel` for that level.
3. `backward_weight_kernel` once over all edges → `d_w`.
4. Gather `d_a[input_rows, :].T` as `d_x` (if input requires grad,
   usually not for training).

## Correctness strategy

`test_kernels.py` builds a small random graph, runs forward/backward
through the Triton path and through a naïve pure-PyTorch reference
(nested Python loop over nodes, dense weight matrix). Asserts:

- Forward outputs match within `1e-5`.
- `torch.autograd.gradcheck` passes on the custom Function.
- Weight gradients match the reference within `1e-5`.

This is non-negotiable: custom kernels are easy to get subtly wrong, so
we refuse to trust them until the reference agrees.

## Training

`train.py` reproduces the Part 1 experiment verbatim:

- 100 training points uniformly in `[-1, 1]²`.
- Target `y = x₁² + x₂²`.
- Architecture: `n_in=2`, `n_bias=2`, `n_hidden=12`, `n_out=1`,
  edge probability `p=0.75`, with guaranteed-non-empty parent sets.
- Activation: tanh hidden, linear output.
- Loss: MSE. Optimizer: Adam, `lr=5e-3`.
- Train for ~3000 steps (full batch).
- Evaluate on 5000 test points in `[-1, 1]²`, save predictions and a
  comparison plot.
