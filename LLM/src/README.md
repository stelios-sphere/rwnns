# `LLM/src/` — RWNN-LM package (agent-facing reference)

Everything a coding agent needs to use this package: what each file
does, how to install, how to instantiate models, train them, test them,
and plot them. Read this first, then jump into the file you need.

## File map

```
LLM/src/
├── README.md            ← this file
├── requirements.txt     ← dependency floors
├── rwnn/                ← the random-DAG core (graph + Triton kernels + nn.Module)
│   ├── __init__.py      ← public API
│   ├── graph.py         ← random DAG construction (two builders) with
│   │                       node-kind support (linear / bilinear-gating)
│   ├── kernels.py       ← four custom Triton GPU kernels, branched on
│   │                       node kind
│   └── model.py         ← autograd.Function + RWNN nn.Module
├── tokenizer/           ← byte-level BPE
│   ├── __init__.py
│   └── bpe.py           ← from-scratch GPT-2-style BPE
├── llm.py               ← RWNNLMConfig + RWNNLM (the language model)
├── visualize.py         ← networkx architecture render + 3D prediction render
└── tests.py             ← correctness tests vs. pure-PyTorch reference
```

A runnable end-to-end example is `../example/run.py` (TinyStories
training).

## Install

Targets **Python 3.10+, PyTorch 2.10, Triton 3.6, CUDA 12.8** on an
NVIDIA GPU. Triton kernels compile at first use via LLVM → PTX → SASS;
**no separate `nvcc` toolchain required** (`ptxas` ships in PyTorch's
CUDA wheels).

```bash
pip install -r LLM/src/requirements.txt
```

If `ptxas` is missing at runtime, install the NVIDIA pip toolchain
sliver:

```bash
pip install nvidia-cuda-nvcc-cu12
```

## Import path

The package isn't pip-installed — put `LLM/src/` on `sys.path`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rwnn       import build_layered_rwnn, RWNN
from tokenizer  import BPETokenizer
from llm        import RWNNLM, RWNNLMConfig
```

`../example/run.py` shows the exact pattern.

## RWNN package (`rwnn/`)

### Public API

| Name | Kind | Purpose |
|---|---|---|
| `RandomDAG` | dataclass | DAG description: parent CSR, edge tables, topological levels, role index ranges, **per-node `node_kinds`** (0=LINEAR, 1=BILINEAR). |
| `build_random_dag(n_in, n_bias, n_hidden, n_out, edge_prob, seed)` | fn | Random DAG with `j < i` rule; number of levels emerges from the draw. |
| `build_layered_rwnn(n_nodes, edge_prob, n_layers, n_in, n_bias, n_out, seed, bilinear_fraction)` | fn | Random DAG with **exactly** `n_layers` topological levels. `bilinear_fraction` (default 0.0) makes that fraction of compute nodes bilinear-gating. |
| `RWNN(graph, device)` | `nn.Module` | Holds weights as `nn.Parameter`, graph tensors as buffers. `forward(x)` returns `[B, n_out]`. Exposes `n_in`, `n_bias`, `n_hidden`, `n_out`, `n_nodes`, `n_edges`, `n_levels`. |
| `RWNNFunction` | `autograd.Function` | Low-level driver around the Triton kernels; `RWNN.forward` calls `RWNNFunction.apply`. |

### Bilinear gating nodes

Each compute node in the DAG has a *kind*:

- **Linear (default).** `pre = Σ_k w_k · a_{p_k}`, then `tanh` (or
  identity at output level).
- **Bilinear gating.** Parents come in `(gate, value)` pairs; one weight
  per individual edge. Pre-activation is
  `pre = Σ_pairs σ(w_g · a_g) · w_v · a_v`, then `tanh`. This is
  structurally the same operation as a SwiGLU / GLU unit and is what
  modern transformer FFNs use.

Activation by `bilinear_fraction=p` makes ~`p` fraction of compute
nodes (hidden + output) bilinear, scattered through the graph by chance.
Each such node is enforced to have an even number of parents (≥ 2). The
kernels branch on the node-kind tensor inside the inner loop, so a
single graph can mix both kinds with no extra launches.

### Minimum working example (regression)

```python
import torch
from rwnn import build_layered_rwnn, RWNN

graph = build_layered_rwnn(
    n_nodes=1000, edge_prob=0.02, n_layers=5,
    n_in=2, n_bias=2, n_out=1, seed=0,
    bilinear_fraction=0.05,    # 5 % of compute nodes are bilinear gates
)
net = RWNN(graph, device="cuda")

x = torch.randn(256, 2, device="cuda")
y = net(x)                # [256, 1]
loss = (y ** 2).mean()
loss.backward()           # net.weights.grad is populated
```

### Key design notes

- **Node indexing.** `[inputs | biases | hidden | outputs]` in a flat
  index space `[0, N)`. Hidden/output node `i` only receives from
  nodes with `j < i` — single rule, guarantees DAG.
- **Topological levels** = longest path from any input/bias. All nodes
  at the same level are data-independent and run in parallel in one
  kernel launch. Forward is `n_levels − 1` launches.
- **Reachable-from-inputs and reachable-to-outputs invariants** are
  both enforced by `build_layered_rwnn`, eliminating "vestigial" nodes
  (no path to output → zero gradient forever).
- **Parent-CSR layout.** Per-edge weights stored in CSR order matching
  parent IDs, so the inner loop of each kernel streams both
  sequentially and coalesces across the batch dimension.
- **Four kernels:** `forward_level_kernel`, `backward_dpre_kernel`,
  `backward_propagate_kernel` (atomics into parents' `d_a`),
  `backward_weight_kernel` (per-edge batch reduction). All branch on
  `node_kinds` inline.
- **Activation.** Tanh for hidden levels, linear for output levels.
  Chosen per-launch via a `tl.constexpr` (no per-node branching for
  activation).

## Tokenizer (`tokenizer/`)

`BPETokenizer` is a from-scratch byte-level BPE in pure Python:

```python
from tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train(corpus_text, vocab_size=1024)         # greedy pair merging
ids = tok.encode("Once upon a time")            # list[int]
text = tok.decode(ids)                          # roundtrip-safe

# Convenience for training: pre-encode a corpus straight onto the GPU.
ids_gpu = tok.encode_corpus_to_gpu(big_text, device="cuda")  # torch.long [N]
```

Algorithm: same family as GPT-2 / `tiktoken` — initial vocab is the 256
bytes; greedily merge the most-frequent adjacent pair until the target
vocab size is reached. Pre-tokenisation uses a stdlib regex to split
text into word-like chunks before merging. Per-word encoding cache makes
encoding a 2 GB corpus seconds instead of minutes. Save / load via
`tok.save("tok.json")` / `BPETokenizer.load("tok.json")`.

GPU acceleration story is honest: BPE encoding is sequential per text;
what we accelerate is the data pipeline by storing the **already-tokenised
corpus** as a GPU tensor. Training-time data access is then a free GPU
slice.

## Language model (`llm.py`)

`RWNNLM` is the autoregressive next-token predictor. **No projection
layers** — every embedded feature is its own RWNN input node, every
vocab entry is its own RWNN output node.

```
ids   [B, T]
  │  token_emb  (V, d_model)
  ▼
emb   [B, T, d_model]
  │  flatten
  ▼
flat  [B, T·d_model]                ← RWNN input (n_in = T·d_model)
  │  RWNN  (tanh hidden, linear output)
  ▼
logits[B, V]                        ← RWNN output (n_out = V)
```

### Config

```python
from llm import RWNNLM, RWNNLMConfig

cfg = RWNNLMConfig(
    vocab_size=1024,
    context_length=128,        # T
    d_model=48,                # token-embedding dim
    n_nodes=45000,             # total RWNN nodes (in + bias + hidden + out)
    n_layers=8,                # RWNN topological depth
    edge_prob=0.075,           # connection density
    bilinear_fraction=0.05,    # 5 % bilinear gating nodes
    n_bias=2,
    seed=0,
)
model = RWNNLM(cfg, device="cuda")
```

`RWNN.n_in` is computed automatically as `context_length · d_model`;
`RWNN.n_out` as `vocab_size`. `n_nodes` must be at least
`n_in + n_bias + n_out + 1` so the build has room for at least one
hidden node.

### Forward / generate

```python
logits = model(ids)                               # [B, T] -> [B, V]

# Sampling: rolling context, top-k + temperature.
out = model.generate(prompt_ids, max_new_tokens=200,
                     temperature=0.85, top_k=50)
text = tok.decode(out[0])
```

### Parameter budget

```python
counts = model.num_parameters()
# {'embedding': 49152, 'rwnn': 65531933, 'total': 65581085}
```

The RWNN holds 100 % of compute parameters (only the small embedding
table sits outside it).

## Visualisation (`visualize.py`)

Two renderers:

```python
from visualize import draw_architecture, draw_prediction_3d

# Architecture: networkx layered by topological level.
# Bilinear nodes get a gold ring + legend entry.
# Linear nodes have a black ring.
# Edge colour: blue (w > 0), red (w < 0). Edge width ∝ |w|.
# Auto-scales for large graphs (> 20 k edges → fast LineCollection
# straight-line mode with edge subsampling).
draw_architecture(graph, net.weights, "architecture.png")

# 3D prediction surface (regression only).
draw_prediction_3d(predict_fn, "predictions_3d.png",
                   train_points=(X, y))
```

## Tests

```bash
cd LLM/src
python3 tests.py
```

Runs eight checks comparing the Triton kernels against a naïve PyTorch
reference:

- forward output, weight gradient, input gradient
- shape variants (`n_hidden ∈ {1, 20, 50}`, various batch sizes)
- `build_layered_rwnn` topological depth is **exact**
- layered forward matches the reference

(Bilinear nodes are exercised inline within these tests because
`bilinear_fraction=0` is the default — to specifically test bilinear,
use the inline benchmark in any of the recent commits referenced from
the LLM example.)

All errors should be ≤ ~3 e-7 (float32 roundoff).
