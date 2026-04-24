"""Draw the RWNN graph in the style of the blog post.

Colour scheme matches the article:
    red    -> input nodes
    purple -> bias nodes
    grey   -> hidden nodes
    blue   -> output node(s)

Nodes are laid out by topological level (x = level, y = position within
level). Edge colour encodes the sign of the learned weight (blue
positive, red negative), edge thickness encodes its magnitude.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rwnn import RWNN, build_random_dag


INPUT_COLOR = "#E64A3C"   # red
BIAS_COLOR = "#8C52A8"    # purple
HIDDEN_COLOR = "#B0B0B0"  # grey
OUTPUT_COLOR = "#3B78C2"  # blue


def _role_of(idx: int, graph) -> str:
    if idx < graph.n_in:
        return "input"
    if idx < graph.n_in + graph.n_bias:
        return "bias"
    if idx < graph.n_in + graph.n_bias + graph.n_hidden:
        return "hidden"
    return "output"


def _color_for(role: str) -> str:
    return {
        "input": INPUT_COLOR,
        "bias": BIAS_COLOR,
        "hidden": HIDDEN_COLOR,
        "output": OUTPUT_COLOR,
    }[role]


def layered_positions(graph) -> dict[int, tuple[float, float]]:
    """Group nodes by topological level and spread them vertically per level."""
    level_nodes = graph.level_nodes.tolist()
    level_starts = graph.level_starts.tolist()
    pos = {}
    n_levels = len(level_starts) - 1
    for ℓ in range(n_levels):
        s, e = level_starts[ℓ], level_starts[ℓ + 1]
        nodes = level_nodes[s:e]
        if not nodes:
            continue
        n = len(nodes)
        for k, nd in enumerate(nodes):
            y = 0.0 if n == 1 else (k - (n - 1) / 2.0) / max(1, n - 1)
            pos[int(nd)] = (float(ℓ), float(y))
    return pos


def draw(graph, weights: np.ndarray, out_path: str, title: str | None = None):
    pos = layered_positions(graph)

    edge_src = graph.edge_src.tolist()
    edge_dst = graph.edge_dst.tolist()

    w_abs = np.abs(weights)
    w_max = float(w_abs.max()) if w_abs.size else 1.0
    # Linewidth in [0.3, 2.8] scaled by |weight|.
    lw = 0.3 + 2.5 * (w_abs / (w_max + 1e-12))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_axis_off()

    # Edges first (so nodes draw on top).
    for e in range(len(edge_src)):
        j = edge_src[e]
        i = edge_dst[e]
        xj, yj = pos[j]
        xi, yi = pos[i]
        color = "#2E6FB7" if weights[e] >= 0 else "#C0392B"
        ax.plot([xj, xi], [yj, yi], color=color, linewidth=lw[e],
                alpha=0.75, zorder=1, solid_capstyle="round")

    # Nodes.
    for nd, (x, y) in pos.items():
        role = _role_of(nd, graph)
        ax.scatter([x], [y], s=500, c=_color_for(role),
                   edgecolors="black", linewidths=1.2, zorder=2)
        ax.text(x, y, str(nd), ha="center", va="center",
                fontsize=9, color="white", weight="bold", zorder=3)

    # Legend.
    legend_entries = [
        ("input",  INPUT_COLOR),
        ("bias",   BIAS_COLOR),
        ("hidden", HIDDEN_COLOR),
        ("output", OUTPUT_COLOR),
    ]
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=c, markeredgecolor="black",
                           markersize=12, label=lbl)
               for lbl, c in legend_entries]
    handles += [
        plt.Line2D([0], [0], color="#2E6FB7", linewidth=2, label="w > 0"),
        plt.Line2D([0], [0], color="#C0392B", linewidth=2, label="w < 0"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9,
              fontsize=9, ncols=3)

    if title is None:
        title = (f"Random DAG: {graph.n_nodes} nodes, {graph.n_edges} edges, "
                 f"{graph.n_levels} levels")
    ax.set_title(title)
    ax.set_xlim(-0.5, graph.n_levels - 0.5)
    ax.set_ylim(-1.1, 1.1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"graph saved to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-hidden", type=int, default=12)
    p.add_argument("--edge-prob", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--untrained", action="store_true",
                   help="Draw the randomly-initialized graph (no training).")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "graph.png"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph = build_random_dag(
        n_in=2, n_bias=2, n_hidden=args.n_hidden, n_out=1,
        edge_prob=args.edge_prob, seed=args.seed,
    )

    if args.untrained:
        net = RWNN(graph, device=device)
        weights = net.weights.detach().cpu().numpy()
        title = (f"Randomly wired, untrained — {graph.n_nodes} nodes, "
                 f"{graph.n_edges} edges, {graph.n_levels} levels")
    else:
        net = RWNN(graph, device=device)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        x = (torch.rand(100, 2, device=device) * 2 - 1)
        y = (x ** 2).sum(dim=1, keepdim=True)
        for _ in range(args.epochs):
            pred = net(x)
            loss = ((pred - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        weights = net.weights.detach().cpu().numpy()
        title = (f"Trained RWNN — {graph.n_nodes} nodes, "
                 f"{graph.n_edges} edges, {graph.n_levels} levels, "
                 f"train mse {loss.item():.2e}")

    draw(graph, weights, args.out, title=title)


if __name__ == "__main__":
    main()
