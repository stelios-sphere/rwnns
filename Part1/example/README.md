# `example/` — latest run

A single end-to-end run showing everything in `src/` working together.

## What it does

`run.py` builds a **1000-node / 5-layer** layered RWNN
(`build_layered_rwnn`, `edge_prob=0.02`), trains it for 3,000 epochs of
full-batch Adam on 5,000 random points in `[-1, 1]²` targeting
`f(x) = x₁² + x₂²`, then produces:

| file | what it is |
|---|---|
| `architecture.png` | networkx render of the trained 1000-node DAG, layered by topological level. Very dense — good for seeing aggregate structure. |
| `architecture_small.png` | networkx render of a small 30-node / 5-layer DAG built with the same builder. Individual nodes and curved edges are clearly visible — use this to understand the layout style. |
| `predictions_3d.png` | Matplotlib 3D surface of the RWNN prediction vs. the target wireframe, plus a training-point scatter and a residual surface. |
| `results.npz` | Raw `x_test`, `y_test`, `pred_test`, learned weights, edge tables, MSE values, per-level node counts. |

## Running it

```bash
cd Part1/example
python3 run.py
```

Requires only the packages in `../src/requirements.txt` and a CUDA GPU.
The first call JIT-compiles the Triton kernels (takes a few seconds);
subsequent steps run at ~1 ms/step.

## Latest numbers (seed = 0)

```
n_nodes    : 1000
n_edges    : ~6,900   (tangential, varies slightly with edge_prob draw)
n_levels   : 5        (exact; enforced by build_layered_rwnn)
per_level  : [4, 332, 332, 331, 1]
train_mse  : ~4e-5
test_mse   : ~4e-5    (no overfitting at 5000 training points)
wall time  : ~3 s for 3000 epochs on an RTX 4090
```

Exact numbers for the run committed here are inside `results.npz`
under the `train_mse` / `test_mse` keys.
