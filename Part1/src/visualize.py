"""Visualisation helpers for RWNNs.

Two public entry points:

- :func:`draw_architecture` — renders the DAG with :mod:`networkx` using
  a topological-level layout. Node colours match the blog post (red
  input, purple bias, grey hidden, blue output); edge colour is the
  sign of the learned weight, edge width scales with ``|w|``.

- :func:`draw_prediction_3d` — renders the trained model on a mesh grid
  as a 3D surface, with the target function as a wireframe overlay and
  the training points as a scatter — the "blog-post" picture.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers projection
import networkx as nx
import numpy as np
import torch


INPUT_COLOR = "#E64A3C"   # red
BIAS_COLOR = "#8C52A8"    # purple
HIDDEN_COLOR = "#B0B0B0"  # grey
OUTPUT_COLOR = "#3B78C2"  # blue

_ROLE_SORT_KEY = {"input": 0, "bias": 1, "hidden": 2, "output": 3}


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
    """(node -> (x, y)) with x = topological level, y = within-level index.

    Within each level, nodes are ordered by role (inputs, biases, hidden,
    outputs) then by id so the layout is deterministic.
    """
    level_nodes = graph.level_nodes.tolist()
    level_starts = graph.level_starts.tolist()
    pos: dict[int, tuple[float, float]] = {}
    for ℓ in range(graph.n_levels):
        s, e = level_starts[ℓ], level_starts[ℓ + 1]
        nodes = sorted(
            level_nodes[s:e],
            key=lambda i: (_ROLE_SORT_KEY[_role_of(i, graph)], i),
        )
        n = len(nodes)
        if n == 0:
            continue
        for k, nd in enumerate(nodes):
            y = 0.0 if n == 1 else (k - (n - 1) / 2.0) / max(1, n - 1)
            pos[int(nd)] = (float(ℓ), float(y))
    return pos


def draw_architecture(
    graph,
    weights: np.ndarray | torch.Tensor,
    out_path: str,
    *,
    title: str | None = None,
    node_size: int | None = None,
    show_labels: bool | None = None,
) -> None:
    """Render the DAG with networkx, coloured / sized like the blog post."""
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    # Build the networkx graph.
    g = nx.DiGraph()
    for n in range(graph.n_nodes):
        g.add_node(int(n), role=_role_of(int(n), graph))
    edge_src = graph.edge_src.tolist()
    edge_dst = graph.edge_dst.tolist()
    for e in range(len(edge_src)):
        g.add_edge(int(edge_src[e]), int(edge_dst[e]), weight=float(weights[e]))

    pos = layered_positions(graph)

    # Size/label heuristics scale down for large graphs.
    N = graph.n_nodes
    if node_size is None:
        node_size = 500 if N <= 40 else max(20, int(4000 / N))
    if show_labels is None:
        show_labels = N <= 60

    fig, ax = plt.subplots(figsize=(13, 6))
    for ℓ in range(graph.n_levels):
        ax.axvline(ℓ, color="#DDDDDD", linewidth=0.8, zorder=0)

    # Edges — styled by sign + magnitude.
    w_arr = np.array([g.edges[e]["weight"] for e in g.edges])
    w_abs = np.abs(w_arr)
    w_max = float(w_abs.max()) if w_abs.size else 1.0
    widths = 0.3 + 2.5 * (w_abs / (w_max + 1e-12))
    edge_colors = ["#2E6FB7" if w >= 0 else "#C0392B" for w in w_arr]

    nx.draw_networkx_edges(
        g, pos, ax=ax,
        edge_color=edge_colors, width=widths, alpha=0.75,
        arrows=False,  # feed-forward is clear from layout; arrows clutter at scale
    )

    # Nodes — colour by role.
    for role in ("input", "bias", "hidden", "output"):
        nodes = [n for n, d in g.nodes(data=True) if d["role"] == role]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            g, pos, nodelist=nodes, ax=ax,
            node_color=_color_for(role),
            node_size=node_size,
            edgecolors="black", linewidths=1.0,
        )

    if show_labels:
        nx.draw_networkx_labels(
            g, pos, ax=ax,
            font_size=9, font_color="white", font_weight="bold",
        )

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
        title = (f"RWNN architecture — {graph.n_nodes} nodes, "
                 f"{graph.n_edges} edges, {graph.n_levels} levels")
    ax.set_title(title)
    ax.set_xlim(-0.5, graph.n_levels - 0.5)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xticks(range(graph.n_levels))
    ax.set_xticklabels([f"L{ℓ}" for ℓ in range(graph.n_levels)], fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("topological level (longest path from inputs/biases)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_prediction_3d(
    predict_fn,
    out_path: str,
    *,
    domain: tuple[float, float] = (-1.0, 1.0),
    grid: int = 60,
    target_fn=lambda x1, x2: x1 ** 2 + x2 ** 2,
    train_points: tuple[np.ndarray, np.ndarray] | None = None,
    title: str | None = None,
) -> None:
    """Render the model as a 3D surface over the input domain.

    Parameters
    ----------
    predict_fn : callable
        Takes an (N, 2) numpy array and returns an (N,) numpy array of
        predictions. The caller is responsible for wrapping any torch
        model so this stays plain numpy in / out.
    domain : (lo, hi)
        Square domain ``[lo, hi]^2`` evaluated on a ``grid x grid`` mesh.
    target_fn : callable(x1, x2) -> ndarray
        The ground-truth function, rendered as a translucent wireframe
        overlay for visual comparison.
    train_points : (X_train, y_train) | None
        If given, plotted as red dots.
    """
    lo, hi = domain
    xs = np.linspace(lo, hi, grid)
    x1, x2 = np.meshgrid(xs, xs)
    pts = np.stack([x1.ravel(), x2.ravel()], axis=1).astype(np.float32)
    y_pred = predict_fn(pts).reshape(grid, grid)
    y_true = target_fn(x1, x2)

    fig = plt.figure(figsize=(12, 5))

    # Left panel: prediction surface (blog-post style).
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax1.plot_surface(
        x1, x2, y_pred, cmap="viridis",
        edgecolor="none", alpha=0.9, antialiased=True,
    )
    ax1.plot_wireframe(
        x1, x2, y_true,
        color="black", linewidth=0.4, alpha=0.35, rstride=4, cstride=4,
    )
    if train_points is not None:
        X_tr, y_tr = train_points
        ax1.scatter(X_tr[:, 0], X_tr[:, 1], y_tr.ravel(),
                    color="red", s=18, depthshade=True, label="train points")
        ax1.legend(loc="upper left")
    ax1.set_xlabel("x1"); ax1.set_ylabel("x2"); ax1.set_zlabel("y")
    ax1.set_title("RWNN prediction (surface) + target (wireframe)")
    fig.colorbar(surf, ax=ax1, shrink=0.6, pad=0.1)

    # Right panel: absolute error.
    err = np.abs(y_pred - y_true)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    err_surf = ax2.plot_surface(
        x1, x2, err, cmap="magma", edgecolor="none", antialiased=True,
    )
    ax2.set_xlabel("x1"); ax2.set_ylabel("x2"); ax2.set_zlabel("|error|")
    ax2.set_title(f"|prediction − target|  (max {err.max():.4f})")
    fig.colorbar(err_surf, ax=ax2, shrink=0.6, pad=0.1)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
