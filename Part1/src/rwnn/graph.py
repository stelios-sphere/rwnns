"""Random feed-forward DAG construction for RWNNs (Part 1).

Node layout, in a single flat index space [0, N):

    [ inputs | biases | hidden | outputs ]

Role constraints (the DAG rule):
- Input/bias nodes have no parents.
- Hidden/output node i may receive from any j < i.

For each hidden/output node we include every lower-index node as a parent
with probability ``edge_prob``. If that produces an empty parent set we
insert one uniformly-random parent so every compute node participates in
the forward pass.

Output nodes additionally get one guaranteed edge from the last hidden
node so gradients can always flow back through hidden compute even if
the random draw connected them only to inputs/biases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass
class RandomDAG:
    """Host-side description of the random DAG + tensors for the GPU.

    Index ranges (all inclusive/exclusive Python-style):
        inputs : [0, n_in)
        biases : [n_in, n_in + n_bias)
        hidden : [n_in + n_bias, n_in + n_bias + n_hidden)
        outputs: [N - n_out, N)
    """

    n_in: int
    n_bias: int
    n_hidden: int
    n_out: int

    # CSR over destination nodes.
    parent_offsets: torch.Tensor  # int32 [N+1]
    parent_ids: torch.Tensor      # int32 [E]

    # Destination-sorted edge tables (same order as CSR / weights).
    edge_src: torch.Tensor        # int32 [E]
    edge_dst: torch.Tensor        # int32 [E]

    # Topological levels.
    level_nodes: torch.Tensor     # int32 [N], nodes sorted by level
    level_starts: torch.Tensor    # int32 [L+1]
    # Which compute levels produce output-node activations (linear) vs.
    # hidden (tanh). A level is "output" if all of its nodes are outputs.
    level_is_output: torch.Tensor # bool [L]

    # Per-node fan-in, handy for weight init.
    fan_in: torch.Tensor          # int32 [N]

    # Convenience.
    input_ids: torch.Tensor       # int32 [n_in]
    bias_ids: torch.Tensor        # int32 [n_bias]
    output_ids: torch.Tensor      # int32 [n_out]

    @property
    def n_nodes(self) -> int:
        return self.n_in + self.n_bias + self.n_hidden + self.n_out

    @property
    def n_edges(self) -> int:
        return int(self.parent_ids.numel())

    @property
    def n_levels(self) -> int:
        return int(self.level_starts.numel()) - 1

    def to(self, device: torch.device | str) -> "RandomDAG":
        kw = {}
        for f in self.__dataclass_fields__:
            v = getattr(self, f)
            if isinstance(v, torch.Tensor):
                kw[f] = v.to(device)
            else:
                kw[f] = v
        return RandomDAG(**kw)


def _assemble_dag(
    n_in: int,
    n_bias: int,
    n_hidden: int,
    n_out: int,
    parents: List[List[int]],
) -> RandomDAG:
    """Pack per-node parent lists into CSR + levels + edge tables."""
    N = n_in + n_bias + n_hidden + n_out
    inputs_end = n_in
    bias_end = n_in + n_bias
    hidden_end = n_in + n_bias + n_hidden

    parent_offsets_np = np.zeros(N + 1, dtype=np.int64)
    for i in range(N):
        parent_offsets_np[i + 1] = parent_offsets_np[i] + len(parents[i])
    E = int(parent_offsets_np[-1])

    parent_ids_np = np.empty(E, dtype=np.int64)
    edge_src_np = np.empty(E, dtype=np.int64)
    edge_dst_np = np.empty(E, dtype=np.int64)
    k = 0
    for i in range(N):
        for j in parents[i]:
            parent_ids_np[k] = j
            edge_src_np[k] = j
            edge_dst_np[k] = i
            k += 1
    assert k == E

    fan_in_np = np.diff(parent_offsets_np).astype(np.int64)

    # Topological levels from longest-path-from-inputs.
    level_np = np.zeros(N, dtype=np.int64)
    for i in range(bias_end, N):
        if parents[i]:
            level_np[i] = int(max(level_np[p] for p in parents[i])) + 1
    L = int(level_np.max()) + 1

    order = np.lexsort((np.arange(N), level_np))
    level_nodes_np = order.astype(np.int64)
    level_starts_np = np.zeros(L + 1, dtype=np.int64)
    for i in range(N):
        level_starts_np[level_np[level_nodes_np[i]] + 1] = i + 1
    for ℓ in range(1, L + 1):
        if level_starts_np[ℓ] < level_starts_np[ℓ - 1]:
            level_starts_np[ℓ] = level_starts_np[ℓ - 1]

    output_set = set(range(hidden_end, N))
    level_is_output_np = np.zeros(L, dtype=bool)
    for ℓ in range(L):
        nodes_at_ℓ = level_nodes_np[level_starts_np[ℓ] : level_starts_np[ℓ + 1]]
        if nodes_at_ℓ.size > 0 and all(int(n) in output_set for n in nodes_at_ℓ):
            level_is_output_np[ℓ] = True

    def t(a, dtype=torch.int32):
        return torch.as_tensor(a, dtype=dtype)

    return RandomDAG(
        n_in=n_in,
        n_bias=n_bias,
        n_hidden=n_hidden,
        n_out=n_out,
        parent_offsets=t(parent_offsets_np),
        parent_ids=t(parent_ids_np),
        edge_src=t(edge_src_np),
        edge_dst=t(edge_dst_np),
        level_nodes=t(level_nodes_np),
        level_starts=t(level_starts_np),
        level_is_output=torch.as_tensor(level_is_output_np),
        fan_in=t(fan_in_np),
        input_ids=t(np.arange(0, inputs_end, dtype=np.int64)),
        bias_ids=t(np.arange(inputs_end, bias_end, dtype=np.int64)),
        output_ids=t(np.arange(hidden_end, N, dtype=np.int64)),
    )


def build_random_dag(
    n_in: int,
    n_bias: int,
    n_hidden: int,
    n_out: int,
    edge_prob: float = 0.75,
    seed: int | None = None,
) -> RandomDAG:
    rng = np.random.default_rng(seed)

    N = n_in + n_bias + n_hidden + n_out
    inputs_end = n_in
    bias_end = n_in + n_bias
    hidden_end = n_in + n_bias + n_hidden
    # outputs: [hidden_end, N)

    # Build parent sets.
    parents: List[List[int]] = [[] for _ in range(N)]

    def draw_parents(dst: int, candidates: np.ndarray) -> np.ndarray:
        # Bernoulli keep with prob edge_prob, fallback to one random if empty.
        if candidates.size == 0:
            return candidates
        keep = rng.random(candidates.size) < edge_prob
        chosen = candidates[keep]
        if chosen.size == 0:
            chosen = rng.choice(candidates, size=1, replace=False)
        return chosen

    # Hidden nodes: parents from any earlier node.
    for i in range(bias_end, hidden_end):
        cand = np.arange(0, i, dtype=np.int64)
        parents[i] = sorted(draw_parents(i, cand).tolist())

    # Output nodes: parents from any earlier node, plus (if hidden exist and
    # no hidden was drawn) one guaranteed edge from a random hidden node so
    # the output isn't a linear function of only the inputs/biases.
    for i in range(hidden_end, N):
        cand = np.arange(0, i, dtype=np.int64)
        chosen = draw_parents(i, cand).tolist()
        if n_hidden > 0 and not any(bias_end <= p < hidden_end for p in chosen):
            chosen.append(int(rng.integers(bias_end, hidden_end)))
        parents[i] = sorted(set(chosen))

    return _assemble_dag(n_in, n_bias, n_hidden, n_out, parents)


def build_layered_rwnn(
    n_nodes: int,
    edge_prob: float,
    n_layers: int,
    n_in: int = 2,
    n_bias: int = 2,
    n_out: int = 1,
    seed: int | None = None,
) -> RandomDAG:
    """Random DAG with a user-specified number of topological levels.

    Parameters
    ----------
    n_nodes : int
        Total number of nodes (inputs + biases + hidden + outputs).
    edge_prob : float
        Probability with which each allowable earlier-layer node is kept
        as a parent. Controls connection density. Lower → sparser.
    n_layers : int
        Desired number of topological levels (≥ 2). Layer 0 holds
        inputs + biases, layer ``n_layers - 1`` holds outputs, the
        remaining ``n_layers - 2`` layers are hidden.
    n_in, n_bias, n_out : int
        Role counts. Defaults: 2 inputs, 2 biases, 1 output.
    seed : int | None
        RNG seed.

    Wiring rules
    ------------
    A node at layer ``L`` draws every node at layers ``< L`` as a
    candidate parent, keeping each with probability ``edge_prob``. To
    guarantee the node actually lands at level ``L`` under the longest-
    path label (``level(i) = 1 + max(level(parents))``), we force at
    least one parent at layer ``L - 1``. The result is a random DAG
    whose topological depth is exactly ``n_layers``.
    """
    if n_layers < 2:
        raise ValueError("n_layers must be >= 2 (one layer for inputs/biases, one for outputs)")
    n_hidden = n_nodes - n_in - n_bias - n_out
    if n_hidden < 0:
        raise ValueError(
            f"n_nodes={n_nodes} too small for n_in+n_bias+n_out="
            f"{n_in + n_bias + n_out}"
        )
    n_hidden_layers = n_layers - 2
    if n_hidden_layers == 0 and n_hidden > 0:
        raise ValueError("n_layers=2 leaves no hidden layers; set n_hidden=0 or increase n_layers")
    if n_hidden_layers > 0 and n_hidden < n_hidden_layers:
        raise ValueError(
            f"need at least {n_hidden_layers} hidden nodes for {n_hidden_layers} hidden layers"
        )

    rng = np.random.default_rng(seed)
    N = n_nodes
    bias_end = n_in + n_bias
    hidden_end = bias_end + n_hidden

    # Assign each node to a layer.
    layer = np.zeros(N, dtype=np.int64)
    layer[:bias_end] = 0
    if n_hidden_layers > 0:
        base = n_hidden // n_hidden_layers
        extra = n_hidden % n_hidden_layers
        idx = bias_end
        for ℓ in range(n_hidden_layers):
            size = base + (1 if ℓ < extra else 0)
            layer[idx : idx + size] = ℓ + 1
            idx += size
        assert idx == hidden_end
    layer[hidden_end:] = n_layers - 1

    # Precompute per-layer node lists (for the mandatory layer-(L-1) pick).
    nodes_at_layer: List[np.ndarray] = [
        np.where(layer == ℓ)[0] for ℓ in range(n_layers)
    ]

    parents: List[List[int]] = [[] for _ in range(N)]
    for i in range(bias_end, N):
        Li = int(layer[i])
        # Candidates live at any earlier layer. We gather per layer so we
        # can enforce the layer-(L-1) presence without re-scanning.
        chosen: set = set()
        for ℓp in range(Li):
            pool = nodes_at_layer[ℓp]
            if pool.size == 0:
                continue
            # Only candidates with index < i are allowed by our DAG rule,
            # which is automatic here because layers are contiguous in index.
            keep = rng.random(pool.size) < edge_prob
            chosen.update(int(x) for x in pool[keep])
        # Enforce at least one parent at layer L-1 so the longest-path
        # rule places this node at exactly level L.
        prev_pool = nodes_at_layer[Li - 1]
        if prev_pool.size > 0 and not any(layer[p] == Li - 1 for p in chosen):
            chosen.add(int(rng.choice(prev_pool)))
        if not chosen:
            # Extreme edge case: no earlier layer has any node. Shouldn't
            # happen because layer 0 always has inputs + biases.
            chosen.add(int(rng.integers(0, i)))
        parents[i] = sorted(chosen)

    return _assemble_dag(n_in, n_bias, n_hidden, n_out, parents)
