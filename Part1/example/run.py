"""Reproduce the latest run: 1000-node / 5-layer RWNN on f(x) = x1^2 + x2^2.

Produces:
    example/architecture.png   - networkx DAG render, coloured by role
    example/predictions_3d.png - 3D surface plot, blog-post style
    example/results.npz        - raw numpy arrays for further analysis
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, SRC)

from rwnn import RWNN, build_layered_rwnn  # noqa: E402
from visualize import draw_architecture, draw_prediction_3d  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_NODES = 1000
N_LAYERS = 5
EDGE_PROB = 0.02
SEED = 0

N_TRAIN = 5000
N_TEST = 5000
EPOCHS = 3000
LR = 5e-3


def target_fn(x: torch.Tensor) -> torch.Tensor:
    return (x ** 2).sum(dim=1, keepdim=True)


def main():
    torch.manual_seed(SEED)
    device = "cuda"

    print(f"building layered RWNN (n_nodes={N_NODES}, n_layers={N_LAYERS}, "
          f"edge_prob={EDGE_PROB})")
    graph = build_layered_rwnn(
        n_nodes=N_NODES,
        edge_prob=EDGE_PROB,
        n_layers=N_LAYERS,
        n_in=2, n_bias=2, n_out=1,
        seed=SEED,
    )
    per_level = [graph.level_starts[i + 1].item() - graph.level_starts[i].item()
                 for i in range(graph.n_levels)]
    print(f"  n_nodes={graph.n_nodes}  n_edges={graph.n_edges}  "
          f"n_levels={graph.n_levels}  per_level={per_level}")

    net = RWNN(graph, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    x_train = (torch.rand(N_TRAIN, 2, device=device) * 2 - 1)
    y_train = target_fn(x_train)
    x_test = (torch.rand(N_TEST, 2, device=device) * 2 - 1)
    y_test = target_fn(x_test)

    # Warm up (first call JITs the kernels).
    net(x_train[:1]); torch.cuda.synchronize()

    print(f"training {EPOCHS} epochs on {N_TRAIN} points, lr={LR}")
    t0 = time.time()
    for epoch in range(EPOCHS):
        pred = net(x_train)
        loss = ((pred - y_train) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch % max(1, EPOCHS // 10) == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:5d}  train mse {loss.item():.6f}")
    torch.cuda.synchronize()
    wall = time.time() - t0

    with torch.no_grad():
        pred_test = net(x_test)
        test_mse = ((pred_test - y_test) ** 2).mean().item()
    train_mse = loss.item()
    print(f"done.  train_mse={train_mse:.6f}  test_mse={test_mse:.6f}  "
          f"wall={wall:.2f}s  ({wall / EPOCHS * 1e3:.2f} ms/epoch)")

    # Save raw arrays.
    npz_path = os.path.join(HERE, "results.npz")
    np.savez(
        npz_path,
        x_test=x_test.cpu().numpy(),
        y_test=y_test.cpu().numpy(),
        pred_test=pred_test.cpu().numpy(),
        weights=net.weights.detach().cpu().numpy(),
        edge_src=graph.edge_src.cpu().numpy(),
        edge_dst=graph.edge_dst.cpu().numpy(),
        train_mse=train_mse,
        test_mse=test_mse,
        n_nodes=graph.n_nodes,
        n_edges=graph.n_edges,
        n_levels=graph.n_levels,
        per_level=np.array(per_level),
    )
    print(f"saved  {npz_path}")

    # Architecture plot of the trained 1000-node model (networkx, dense).
    arch_path = os.path.join(HERE, "architecture.png")
    draw_architecture(
        graph,
        net.weights,
        arch_path,
        title=(f"RWNN architecture — {graph.n_nodes} nodes, "
               f"{graph.n_edges} edges, {graph.n_levels} layers"),
    )
    print(f"saved  {arch_path}")

    # Also render a small representative DAG so the networkx structure is
    # visually legible. Same builder, same number of layers, untrained —
    # this is a structural view, not a trained-weight heatmap.
    small_graph = build_layered_rwnn(
        n_nodes=30, edge_prob=0.3, n_layers=N_LAYERS, seed=SEED,
    )
    small_net = RWNN(small_graph, device=device)
    small_path = os.path.join(HERE, "architecture_small.png")
    draw_architecture(
        small_graph,
        small_net.weights,
        small_path,
        title=(f"Small representative DAG — {small_graph.n_nodes} nodes, "
               f"{small_graph.n_edges} edges, {small_graph.n_levels} layers "
               f"(structure preview, untrained)"),
    )
    print(f"saved  {small_path}")

    # 3D surface plot.
    def predict_np(pts: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(pts, device=device, dtype=torch.float32)
            return net(t).squeeze(-1).cpu().numpy()

    pred_path = os.path.join(HERE, "predictions_3d.png")
    # Use a subsample of training points so they remain visible on the surface.
    sub = np.random.default_rng(SEED).choice(N_TRAIN, size=200, replace=False)
    draw_prediction_3d(
        predict_np,
        pred_path,
        domain=(-1.0, 1.0),
        grid=60,
        train_points=(x_train[sub].cpu().numpy(), y_train[sub].cpu().numpy()),
        title=(f"Target f(x) = x1² + x2²  —  test MSE {test_mse:.2e}  "
               f"(RWNN: {graph.n_nodes} nodes, {graph.n_layers if hasattr(graph, 'n_layers') else graph.n_levels} layers)"),
    )
    print(f"saved  {pred_path}")


if __name__ == "__main__":
    main()
