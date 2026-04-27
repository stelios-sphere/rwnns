# Randomly / intentionally wired neural networks as a research agenda

*A short memo on why this direction is worth a sustained look, what the
concrete advantages are over the transformer default, and what the
near-term experiments would be.*

---

## TL;DR

**Thesis.** A neural network architecture where the connectivity graph
is a first-class object — sampled randomly, designed deliberately, or
searched evolutionarily — unlocks three axes of freedom that
transformers don't naturally give you:

1. **Parameter granularity.** Weights are per-edge. Pruning, growing,
   sparsifying, and evolutionary search all operate on the same
   primitive.
2. **Node-type heterogeneity.** A single graph can mix linear,
   bilinear (attention-like), trilinear, and higher-order nodes — an
   architectural degree of freedom that transformers are structurally
   locked out of (they are linear + bilinear and nothing else).
3. **Communication topology.** The network *is* a graph; graph
   partitioning gives you model-parallel plumbing that
   uniformly-dense architectures require specialised systems (Megatron,
   FSDP, pipeline parallel) to achieve.

**Claim.** The *systems* case for this architecture is stronger than
the *model-quality* case. Attention remains a good prior for language,
but it is also the root cause of most of the distributed-training pain
at trillion-parameter scale. An RWNN-style architecture aligned to the
physical hardware topology could train for a fraction of the
communication cost.

---

## Context

Random feed-forward graphs as neural networks were shown by Facebook AI
Research (Xie et al., 2019) to match hand-designed architectures like
ResNet and ShuffleNet on ImageNet. The series this repo builds on
(Kyriacou, 2020) reproduces this at a tiny scale, then extends it with
ensemble uncertainty quantification and multi-objective evolutionary
architecture search.

Nothing in those results disputes that attention is a strong language
prior. Our own experiments with an RWNN-core LLM on Shakespeare confirm
that pure RWNN-LMs (flattened-context MLP lineage) underperform
transformers on language. That is the *known loss*.

The *upside* is in three places transformers structurally cannot match:
architecture search, node heterogeneity, and communication topology.

---

## The case, in three points

### 1. Architecture search is natural here, hard there

Transformers present a small, rigid hyperparameter surface: layer count,
head count, embedding dim, block size. "Neural architecture search"
over this space is a combinatorial optimisation over ~5 integers.

RWNNs present architecture as **a literal graph**. The design space is
the space of DAGs, which is continuous-ish under evolutionary
operators (add edge, remove edge, reassign node kind, swap parent
groups). Part 3's bi-objective Pareto front (error vs. edge count)
already demonstrates this in a toy regression setting — the search
discovers sparse architectures matching dense ones at ~1/3 the
connections.

Extending this to an RWNN-as-transformer-FFN replacement is a
tractable, well-posed research project with a clear evaluation protocol
(perplexity on OpenWebText-scale data with a standardised attention
stack held fixed).

### 2. Heterogeneous node kinds → a single graph, multiple operator types

A transformer has exactly two primitives: attention (bilinear in
activations) and MLP (linear in activations). Everything it can express
is some composition of these.

An RWNN's graph abstraction is indifferent to node kind. A single graph
can mix:

- **Linear nodes.** `a_i = σ(Σ w_j a_j)`. Current default.
- **Bilinear nodes.** `a_i = σ(Σ w_jk · a_j · a_k)`. Gives you
  content-addressable routing of exactly the kind attention uses.
- **Trilinear / higher-order nodes.** 3-way conjunction detectors.
  Mostly useful in evolutionary-searched positions; hand-designing
  them is unlikely to pay off.

One extra branch in the forward / backward Triton kernel unlocks
bilinear. Combined with Part 3's evolutionary search, the algorithm
itself can find the right mix of node kinds per position in the graph.
The Pareto front becomes **(error, edges, bilinear-fraction,
cross-partition-edges)** — a rich, genuinely novel search space.

### 3. The model is the communication graph

This is the strongest pitch, and the one that differentiates RWNNs from
"just another NAS idea".

Modern LLM training is communication-bound, not compute-bound, past a
certain scale. Why?

- Attention forces all-to-all interactions in the sequence dimension.
- The 4× MLP expansion makes the down-projection the largest
  cross-GPU transfer.
- Uniformly dense layers treat all communication as equally expensive;
  modern clusters do not.

An RWNN makes the network's communication pattern **an explicit design
variable**. Concretely:

- **Graph partitioning = free model parallelism.** Given a DAG and k
  GPUs, a classical min-cut partitioner assigns nodes to GPUs so that
  cross-partition edges (= inter-GPU traffic) are minimised. No
  bespoke tensor-parallel plumbing required.
- **Hierarchical edge probability.** Make `edge_prob` a function of
  physical distance:
  `intra-GPU ≫ intra-node ≫ inter-node ≫ inter-site`. The graph
  naturally matches bandwidth tiers.
- **Pipeline parallelism is geometric.** Topological levels are the
  natural sync barrier. Assign levels to pipeline stages — no
  micro-batch schedulers, no bubble overhead beyond inter-level sync.
- **Sparse all-reduce.** Only cross-partition edges need collective
  comms during gradient sync. 95 % intra-partition edges = ~20× lower
  all-reduce volume.
- **Communication-cost term in NAS.** Add cross-partition edges as a
  third objective in the evolutionary search; the algorithm evolves
  hardware-aware architectures.

This is the rare case where a sparse topological architecture
*structurally beats* a dense one on something that matters at scale.

---

## Near-term experiments (ordered)

In ascending order of research risk:

1. **RWNN as transformer FFN replacement.** Keep attention; replace the
   2/3-of-parameters dense MLP with a random / evolutionary-searched
   DAG of the same param budget. Measure perplexity on OpenWebText at
   GPT-2-small scale. Expected outcome: comparable perplexity at lower
   FLOP / comms budget due to sparsity.
2. **Bilinear node kind.** Add a multiplicative node kind to the RWNN
   kernel. Re-run the FFN-replacement experiment and let the
   evolutionary search decide how many bilinear nodes are useful.
   Expected outcome: small perplexity win, specifically on tasks where
   local pairwise interactions matter (syntax, coreference).
3. **Hierarchical edge probability.** Build the RWNN with edge
   probability shaped to actual GPU topology (intra-/inter-NVLink,
   inter-node). Measure training throughput and communication volume
   vs. a dense transformer baseline on the same hardware. Expected
   outcome: substantial throughput improvement.
4. **Multi-objective NAS with communication cost.** Evolutionary search
   over (accuracy, edges, cross-partition-edges). Harvest the Pareto
   front. Expected outcome: characterise how much accuracy must be
   traded for training-economics gains.
5. **Scaling laws for RWNN-LMs.** Kaplan / Hoffmann-style study across
   multiple (params, data, compute) budgets. This is the most
   uncertain and highest-value outcome: we have no scaling laws for
   this architecture family.

---

## Caveats, honestly

- Attention remains a strong language prior; a pure-RWNN LLM is
  unlikely to match a same-size transformer on perplexity. The story
  is in **component replacement + training economics**, not drop-in
  end-to-end displacement.
- Load-balancing partitioned DAGs is harder than partitioning uniform
  layers. Graph partitioners handle it, but node compute cost must be
  weighted correctly.
- Fault tolerance changes: replacing a dead GPU's specific nodes is
  non-trivial without redundant assignment.
- Collective libraries (NCCL) prefer uniform patterns; many small
  per-level collectives may hurt throughput unless batched across
  levels.
- Training is currently finickier than transformers — LR tuning
  transfers poorly across scales (observed: 3e-3 that worked at 2.7M
  diverged at 66M, in this very repo). Stabilising training at scale
  is its own research project.

---

## Summary

The case for RWNNs is not "drop-in transformer replacement for
language". It is:

1. **A more expressive architecture search space** (graph topology +
   node kinds).
2. **A more faithful match to distributed-training hardware**
   (partitioning + hierarchical connectivity).
3. **A systems-grounded research direction** that is complementary to
   attention, not competitive with it.

The implementation cost is modest. The repository already contains a
working custom-kernel RWNN trainer at ~66M parameters, topology search
infrastructure (Part 3), and a dataset loop for Shakespeare-scale data.
The near-term experiments above are tractable in a focused two- or
three-person quarter.

The upside, if the systems case lands, is not a small perplexity
improvement — it is a meaningful reduction in the communication cost
of training the next generation of large models. That is a prize worth
two or three quarters of serious effort.
