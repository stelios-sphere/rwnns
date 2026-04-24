"""Train the 1000-node / 5-layer layered RWNN on the Part 1 task.

Fits f(x1, x2) = x1^2 + x2^2 using the same protocol as Part 1's
train.py, but with the layered network produced by build_layered_rwnn.
With 6,924 parameters against 100 training points the model is heavily
over-parameterised, so we also show a run with more training data for
honesty.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from rwnn import RWNN, build_layered_rwnn


def target_fn(x: torch.Tensor) -> torch.Tensor:
    return (x ** 2).sum(dim=1, keepdim=True)


def run(n_train: int, args) -> dict:
    torch.manual_seed(args.seed)
    graph = build_layered_rwnn(
        n_nodes=args.n_nodes,
        edge_prob=args.edge_prob,
        n_layers=args.n_layers,
        n_in=2, n_bias=2, n_out=1,
        seed=args.seed,
    )
    net = RWNN(graph, device="cuda")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    x_train = (torch.rand(n_train, 2, device="cuda") * 2 - 1)
    y_train = target_fn(x_train)
    x_test = (torch.rand(args.n_test, 2, device="cuda") * 2 - 1)
    y_test = target_fn(x_test)

    print(f"--- n_train={n_train:5d}  params={net.weights.numel():,}  "
          f"edges={graph.n_edges}  levels={graph.n_levels} ---")

    # Warm up Triton JIT.
    net(x_train[:1]); torch.cuda.synchronize()

    t0 = time.time()
    for epoch in range(args.epochs):
        pred = net(x_train)
        loss = ((pred - y_train) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:5d}  train mse {loss.item():.6f}")
    torch.cuda.synchronize()
    wall = time.time() - t0

    with torch.no_grad():
        pred_test = net(x_test)
        test_mse = ((pred_test - y_test) ** 2).mean().item()
    train_mse = loss.item()
    print(f"  train_mse={train_mse:.6f}  test_mse={test_mse:.6f}  "
          f"wall={wall:.2f}s  {wall / args.epochs * 1e3:.2f} ms/epoch")

    return dict(
        n_train=n_train,
        train_mse=train_mse,
        test_mse=test_mse,
        wall=wall,
        x_test=x_test.cpu().numpy(),
        y_test=y_test.cpu().numpy(),
        pred_test=pred_test.cpu().numpy(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-nodes", type=int, default=1000)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--edge-prob", type=float, default=0.02)
    p.add_argument("--n-test", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = p.parse_args()

    results = []
    for n_train in (100, 5000):
        results.append(run(n_train, args))
        print()

    if args.plot:
        _plot(results, args.out_dir)


def _plot(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(len(results), 3, figsize=(15, 4 * len(results)))
    if len(results) == 1:
        axs = [axs]
    for row, r in zip(axs, results):
        x = r["x_test"]; y = r["y_test"].ravel(); p = r["pred_test"].ravel()
        s0 = row[0].scatter(x[:, 0], x[:, 1], c=y, s=4, cmap="viridis")
        row[0].set_title(f"truth (n_train={r['n_train']})"); fig.colorbar(s0, ax=row[0])
        s1 = row[1].scatter(x[:, 0], x[:, 1], c=p, s=4, cmap="viridis")
        row[1].set_title(f"prediction (test mse {r['test_mse']:.4f})")
        fig.colorbar(s1, ax=row[1])
        s2 = row[2].scatter(x[:, 0], x[:, 1], c=np.abs(p - y), s=4, cmap="magma")
        row[2].set_title("|error|"); fig.colorbar(s2, ax=row[2])
        for ax in row:
            ax.set_aspect("equal"); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    fig.tight_layout()
    out = os.path.join(out_dir, "predictions_large.png")
    fig.savefig(out, dpi=120)
    print(f"plot saved to {out}")


if __name__ == "__main__":
    main()
