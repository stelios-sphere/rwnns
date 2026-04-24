# `src/` — RWNN package (agent-facing reference)

Everything a coding agent needs to use this package: what each file
does, how to install, how to instantiate a model, train it, test it,
and plot it. Read this first, then jump into the file you need.

## File map

```
src/
├── README.md          ← this file
├── requirements.txt   ← pinned-ish dependency floors
├── rwnn/              ← the importable package
│   ├── __init__.py    ← public API (see "Public API" below)
│   ├── graph.py       ← random DAG construction (two builders)
│   ├── kernels.py     ← four custom Triton GPU kernels
│   └── model.py       ← autograd.Function + nn.Module
├── visualize.py       ← drawing helpers (networkx + matplotlib 3D)
└── tests.py           ← correctness tests vs. a pure-PyTorch reference
```

A minimal runnable example lives under `../example/run.py`.

## Install

This code targets **Python 3.10+, PyTorch 2.10, Triton 3.6, CUDA 12.8**
on an NVIDIA GPU. Triton kernels compile at first use via LLVM → PTX
→ SASS; **no separate `nvcc` toolchain is required** (`ptxas` is
bundled in PyTorch's CUDA wheels on recent setups).

```bash
pip install -r src/requirements.txt
```

If `ptxas` is missing at runtime, install the NVIDIA pip toolchain
sliver:

```bash
pip install nvidia-cuda-nvcc-cu12
```

## Import path

The package isn't pip-installed — just put `src/` on `sys.path`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rwnn import build_layered_rwnn, RWNN
```

`../example/run.py` shows the exact pattern.

## Public API (`rwnn/__init__.py`)

| Name | Kind | Purpose |
|---|---|---|
| `RandomDAG` | dataclass | Host/device description of a DAG: CSR of parents, edge tables, topological levels, role index ranges. |
| `build_random_dag(n_in, n_bias, n_hidden, n_out, edge_prob, seed)` | fn | Random DAG with the `j < i` role constraint; number of topological levels emerges from the draw. |
| `build_layered_rwnn(n_nodes, edge_prob, n_layers, n_in, n_bias, n_out, seed)` | fn | Random DAG with **exactly** `n_layers` topological levels. |
| `RWNN(graph, device)` | `nn.Module` | Holds weights as `nn.Parameter`, graph tensors as buffers. `forward(x)` returns `[B, n_out]`. |
| `RWNNFunction` | `autograd.Function` | Low-level driver around the Triton kernels; `RWNN.forward` calls `RWNNFunction.apply`. |

All builders return a `RandomDAG`. Pass that to `RWNN(graph)`.

## Minimum working example

```python
import torch
from rwnn import build_layered_rwnn, RWNN

graph = build_layered_rwnn(
    n_nodes=1000, edge_prob=0.02, n_layers=5,
    n_in=2, n_bias=2, n_out=1, seed=0,
)
net = RWNN(graph, device="cuda")

x = torch.randn(256, 2, device="cuda")
y = net(x)                # [256, 1]
loss = (y ** 2).mean()
loss.backward()           # net.weights.grad is populated
```

## Training loop recipe

```python
opt = torch.optim.Adam(net.parameters(), lr=5e-3)

x_train = (torch.rand(5000, 2, device="cuda") * 2 - 1)
y_train = (x_train ** 2).sum(dim=1, keepdim=True)

for epoch in range(3000):
    pred = net(x_train)
    loss = ((pred - y_train) ** 2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
```

Anything PyTorch supports (Adam, SGD, weight decay, LR schedulers,
gradient clipping, `torch.compile` around the non-Triton parts) works
here — `RWNN` is a normal `nn.Module`.

## Visualisation

Two renderers in `visualize.py`:

```python
from visualize import draw_architecture, draw_prediction_3d
import numpy as np

# 1) Architecture (networkx, layered by topological level).
draw_architecture(graph, net.weights, "architecture.png")

# 2) 3D prediction surface (matplotlib, blog-post style).
def predict(pts: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.as_tensor(pts, device="cuda", dtype=torch.float32)
        return net(t).squeeze(-1).cpu().numpy()

draw_prediction_3d(
    predict,
    "predictions_3d.png",
    train_points=(x_train.cpu().numpy(), y_train.cpu().numpy()),
)
```

- `draw_architecture` auto-scales node size and hides labels for large
  graphs (> 60 nodes). Edge colour = sign, width ∝ `|w|`.
- `draw_prediction_3d` evaluates the model on a `grid × grid` mesh,
  plots the RWNN output as a 3D surface, overlays the target function
  as a translucent wireframe, and optionally scatters training points.

## Running the tests

```bash
cd src
python3 tests.py
```

Runs six checks comparing the Triton kernels against a naïve PyTorch
reference:

- forward output, weight gradient, input gradient
- multiple shapes (`n_hidden ∈ {1, 20, 50}`, various batch sizes)
- `build_layered_rwnn` produces **exactly** the requested number of
  levels
- layered forward matches the reference

All errors should be ≤ ~3e-7 (float32 roundoff).

## Key design notes (why the code looks the way it does)

- **Node indexing.** `[inputs | biases | hidden | outputs]` in a single
  flat index space `[0, N)`. Hidden/output node `i` may only receive
  from nodes with `j < i` — that single rule makes every graph a DAG.
- **Topological level** = longest path from any input/bias. All nodes
  at the same level are data-independent and run in parallel in one
  kernel launch. Forward is `n_levels − 1` launches.
- **Parent-CSR layout.** Per-edge weights are stored in CSR order so
  the inner loop of each kernel reads parent IDs and weights together
  with coalesced access.
- **Four kernels:** `forward_level`, `backward_dpre`, `backward_propagate`
  (atomics into parents' `d_a`), `backward_weight` (per-edge reduction
  over batch). See `kernels.py` for the implementations.
- **Activation.** Tanh for hidden levels, linear for output levels.
  Chosen per-launch via a Triton `tl.constexpr` (no per-node branching
  inside the kernel).
