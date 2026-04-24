"""Part 1 experiment: fit f(x1, x2) = x1^2 + x2^2 with a randomly wired net.

Matches the article's protocol: 100 random points in [-1, 1]^2 for
training, 5000 for test. Tanh hidden, linear output, MSE loss, Adam.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from rwnn import RWNN, build_random_dag


def target_fn(x: torch.Tensor) -> torch.Tensor:
    return (x ** 2).sum(dim=1, keepdim=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-hidden", type=int, default=12)
    p.add_argument("--edge-prob", type=float, default=0.75)
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-test", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"

    graph = build_random_dag(
        n_in=2, n_bias=2, n_hidden=args.n_hidden, n_out=1,
        edge_prob=args.edge_prob, seed=args.seed,
    )
    print(f"graph: {graph.n_nodes} nodes, {graph.n_edges} edges, "
          f"{graph.n_levels} levels")

    net = RWNN(graph, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    x_train = (torch.rand(args.n_train, 2, device=device) * 2 - 1)
    y_train = target_fn(x_train)
    x_test = (torch.rand(args.n_test, 2, device=device) * 2 - 1)
    y_test = target_fn(x_test)

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
    train_secs = time.time() - t0
    print(f"trained in {train_secs:.2f}s  "
          f"({train_secs / args.epochs * 1e3:.2f} ms/epoch)")

    with torch.no_grad():
        pred_test = net(x_test)
        test_mse = ((pred_test - y_test) ** 2).mean().item()
    print(f"test mse ({args.n_test} points): {test_mse:.6f}")

    # Save predictions for inspection.
    np.savez(
        os.path.join(args.out_dir, "results.npz"),
        x_test=x_test.cpu().numpy(),
        y_test=y_test.cpu().numpy(),
        pred_test=pred_test.cpu().numpy(),
        train_mse=loss.item(),
        test_mse=test_mse,
        n_edges=graph.n_edges,
        n_nodes=graph.n_nodes,
    )

    if args.plot:
        _plot(x_test, y_test, pred_test, args.out_dir)


def _plot(x_test, y_test, pred_test, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = x_test.cpu().numpy()
    y = y_test.cpu().numpy().ravel()
    p = pred_test.cpu().numpy().ravel()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    s0 = axs[0].scatter(x[:, 0], x[:, 1], c=y, s=4, cmap="viridis")
    axs[0].set_title("truth: x1^2 + x2^2"); fig.colorbar(s0, ax=axs[0])
    s1 = axs[1].scatter(x[:, 0], x[:, 1], c=p, s=4, cmap="viridis")
    axs[1].set_title("RWNN prediction");    fig.colorbar(s1, ax=axs[1])
    s2 = axs[2].scatter(x[:, 0], x[:, 1], c=np.abs(p - y), s=4, cmap="magma")
    axs[2].set_title("|error|");            fig.colorbar(s2, ax=axs[2])
    for ax in axs:
        ax.set_aspect("equal"); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    fig.tight_layout()
    out = os.path.join(out_dir, "predictions.png")
    fig.savefig(out, dpi=120)
    print(f"plot saved to {out}")


if __name__ == "__main__":
    main()
