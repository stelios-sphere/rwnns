# Ideas for future research

A running notebook of architectural extensions and research directions
that fell out of conversations while building the RWNN series. Nothing
here is implemented yet; this is where ideas go to wait their turn.

---

## 1. Multiplicative node types → attention-like dynamics

**Topology + multiplicative node type**: now you're in the neighbourhood of
real attention, and it's natively a random-graph-compatible extension.
This is actually a research-interesting direction.

### What to actually prototype

If I were doing this seriously:

- **Bilinear "gating" nodes** as a second node type. Kernel change is
  trivial: one more branch, parents split into (multiplicand, multiplier)
  pairs. This is where you get most of the attention-like benefit.
- **Leave trilinear+ for later**, or only via Part 3's evolutionary
  search: if the evolutionary algorithm finds that adding trilinear
  nodes in specific positions helps, it'll do it. Don't hand-design them.

### Why this matters

So the answer to "why only pairs of tokens and not triplets etc?": pairs
are not the natural stopping point — they're just where attention got
stuck for historical/compute reasons. RWNNs naturally generalise to
arbitrary arity, and that's a genuine architectural advantage.

The fact that you could have a DAG with 95% linear nodes, 4% bilinear
nodes, 1% trilinear nodes, all co-trained, all topology-searched, is
the kind of thing you can't easily write down as a transformer.

### Notes toward an implementation

- **Two node kinds in one graph.** Extend `RandomDAG` with a `node_kind`
  tensor: `0 = LINEAR`, `1 = BILINEAR`. Bilinear nodes have parents
  paired as (multiplicand, multiplier).
- **Forward kernel branch.** In `forward_level_kernel`, bilinear nodes
  compute `Σ_pairs w · a_j · a_k` instead of `Σ w · a_j`. One extra code
  path; same CSR layout works if we interleave pair parents.
- **Backward.** `∂(a_j · a_k)/∂a_j = a_k` and vice versa — both parents
  need their gradient scaled by the *other* parent's activation.
  Straightforward in Triton; requires storing pair-paired parent lists.
- **Evolutionary search hook.** Part 3's mutation operators can now
  flip a node's kind, not just add/remove edges. The Pareto front
  becomes (error, edges, bilinear-fraction).

---

(Add future ideas below this line.)
