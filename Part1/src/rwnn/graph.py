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

    # Per-node kind. 0 = LINEAR, 1 = BILINEAR (gating). Bilinear nodes
    # have an even number of parents paired as (gate, value); they
    # compute  tanh( Σ_pairs σ(w_g · a_g) · w_v · a_v ).
    node_kinds: torch.Tensor      # int8 [N]

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
    node_kinds: np.ndarray | None = None,
) -> RandomDAG:
    """Pack per-node parent lists into CSR + levels + edge tables.

    ``node_kinds`` is an int8 array of length N, with 0 = LINEAR (default)
    and 1 = BILINEAR. If None, all nodes are LINEAR.
    """
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

    if node_kinds is None:
        node_kinds_np = np.zeros(N, dtype=np.int8)
    else:
        node_kinds_np = np.asarray(node_kinds, dtype=np.int8)
        assert node_kinds_np.shape == (N,)

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
        node_kinds=torch.as_tensor(node_kinds_np, dtype=torch.int8),
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
    bilinear_fraction: float = 0.0,
    product_fraction: float = 0.0,
    attention_fraction: float = 0.0,
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
    candidate parent, keeping each with probability ``edge_prob``.

    Two constraints are then enforced to keep the graph non-degenerate:

    * **Reachable from inputs.** At least one parent at layer ``L - 1``
      so the longest-path label ``level(i) = 1 + max(level(parents))``
      places this node at exactly level ``L``.
    * **Reachable to outputs.** Every non-output node has at least one
      outgoing edge to some layer ``> L``. Combined with the previous
      rule, this eliminates "vestigial" nodes — those with no path to
      any output and therefore zero gradient forever.

    The result is a random DAG whose topological depth is exactly
    ``n_layers`` and in which every node participates in at least one
    input-to-output path.

    Mixed node kinds
    ----------------
    Three optional fractions designate non-linear node kinds. They are
    drawn independently per compute node, so the kind chosen for any
    given node is exactly one of:

    * Bilinear (``bilinear_fraction``):
      ``pre = Σ_pairs σ(w_g · a_g) · w_v · a_v``  (parents in pairs,
      two weights per pair)
    * Product (``product_fraction``):
      ``pre = Σ_pairs a_g · a_v``  (parents in pairs, no weights —
      pure activation product, symmetric)
    * Attention / softmax-aggregator (``attention_fraction``):
      ``pre = Σ_k softmax(score_k) · value_k``  (parents in
      ``(score, value)`` pairs, no weights — softmax taken across
      the K pairs of one node)

    The sum of the three must be ≤ 1. Whatever fraction is left over
    stays linear: ``pre = Σ_k w_k · a_k``.

    All non-linear kinds need an even number of parents (≥ 2). When
    the random parent draw lands on an odd count we pad with one extra
    earlier-layer parent that preserves the layer-(L − 1) invariant.

    Bilinear gating nodes (legacy paragraph kept for context)
    --------------------------------------------------------
    Each bilinear node ``tanh``'s the gated sum just like a linear
    node, so its output activation is in the same range. After pair
    adjustment every bilinear node has at least one (gate, value)
    pair, with at least one pair pointing back to layer ``L − 1`` to
    preserve the
    longest-path placement.
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

    # Reachable-to-outputs pass. For every non-output node with no
    # outgoing edge, add one to a uniformly random node at some later
    # layer. Because output nodes sit at the max layer and already have
    # at least one parent at L_max - 1, this pass terminates every
    # forward chain at an output node — no vestigial nodes.
    children_count = np.zeros(N, dtype=np.int64)
    for p_list in parents:
        for p in p_list:
            children_count[p] += 1

    for i in range(hidden_end):  # inputs + biases + hidden (not outputs)
        if children_count[i] > 0:
            continue
        Li = int(layer[i])
        later = np.concatenate(
            [nodes_at_layer[ℓ] for ℓ in range(Li + 1, n_layers)]
        ) if Li + 1 < n_layers else np.empty(0, dtype=np.int64)
        if later.size == 0:
            continue
        j = int(rng.choice(later))
        parents[j] = sorted(set(parents[j]) | {i})
        children_count[i] += 1

    # Per-node kind assignment — independent draws per compute node.
    # Kind codes match kernels.py:
    #   0 = LINEAR (default), 1 = BILINEAR, 2 = PRODUCT, 3 = ATTENTION
    # Validate fractions.
    nonlin_total = bilinear_fraction + product_fraction + attention_fraction
    if nonlin_total > 1.0 + 1e-9:
        raise ValueError(
            f"bilinear+product+attention fractions sum to {nonlin_total:.3f} > 1; "
            f"each compute node has exactly one kind"
        )

    node_kinds = np.zeros(N, dtype=np.int8)
    if nonlin_total > 0.0:
        # Cumulative thresholds for a single uniform draw → bucket.
        t1 = bilinear_fraction                              # < t1 → BILINEAR
        t2 = t1 + product_fraction                          # < t2 → PRODUCT
        t3 = t2 + attention_fraction                        # < t3 → ATTENTION

        for i in range(bias_end, N):
            r = rng.random()
            if r < t1:
                kind = 1
            elif r < t2:
                kind = 2
            elif r < t3:
                kind = 3
            else:
                kind = 0
            if kind == 0:
                continue

            # Non-linear kinds all want pair-aligned parents (≥ 2, even).
            # Pad with extra earlier-layer parents to satisfy the constraint.
            if len(parents[i]) < 2:
                Li = int(layer[i])
                cand = np.concatenate(
                    [nodes_at_layer[ℓ] for ℓ in range(Li)]
                )
                cand = cand[~np.isin(cand, parents[i])]
                if cand.size == 0:
                    # Can't make this node non-linear meaningfully — leave linear.
                    continue
                parents[i] = sorted(set(parents[i]) | {int(rng.choice(cand))})
            if len(parents[i]) % 2 == 1:
                Li = int(layer[i])
                cand = np.concatenate(
                    [nodes_at_layer[ℓ] for ℓ in range(Li)]
                )
                cand = cand[~np.isin(cand, parents[i])]
                if cand.size == 0:
                    parents[i] = parents[i][:-1]
                else:
                    parents[i] = sorted(set(parents[i]) | {int(rng.choice(cand))})
            node_kinds[i] = kind

    return _assemble_dag(n_in, n_bias, n_hidden, n_out, parents, node_kinds)
