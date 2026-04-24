# Part 3 — Optimally Wired Neural Networks (OWNNs)

**Source article:** [On Randomly Wired and Optimally Wired Feed-Forward Neural Networks — Part 3](https://medium.com/becoming-human/on-randomly-wired-and-optimally-wired-feed-forward-neural-networks-part-3-57c616c581b6)

## From random to optimal

Part 1 established that random wiring works. Part 2 used the variability
of random wiring as a source of uncertainty. Part 3 asks the natural next
question: **can we search over wirings to find good ones instead of
relying on chance?**

This is a miniature version of **Neural Architecture Search (NAS)**. The
twist is that here the search variable is the set of synaptic
connections — the graph itself — not a choice between pre-baked layer
templates. The optimisation mixes two very different kinds of variables:

- **Discrete:** which connections exist? This is a combinatorial problem
  on graphs.
- **Continuous:** what weight does each connection carry? This is the
  usual smooth optimisation problem backprop solves.

The approach combines **evolutionary algorithms** (good at discrete,
non-differentiable search) with **gradient descent** (good at continuous,
differentiable refinement).

## Representation: the genome

Each candidate network is represented as a list of synaptic connections,
where each synapse is a tuple `(from_node, to_node)`. This list is the
genome that evolution operates on. Weights live alongside the genome but
are refined by backprop after the topology is fixed for a given
generation.

## Evolutionary operators

### Crossover

Two parent networks contribute their connection lists. A random splice
point is chosen, and the offspring inherits the first part of parent A's
connections and the second part of parent B's connections (and vice versa
for the sibling offspring). The result is a new topology that shares
sub-structures with both parents.

### Mutation

A random perturbation of the connection list:

- **Add** a new `(from, to)` synapse, initialised with a random weight.
- **Remove** an existing synapse.

Mutation keeps the population from collapsing onto a single lineage and
allows the search to discover connections no parent had.

### Weight refinement

After each generation produces new topologies, the weights of every
candidate are refined by ordinary backpropagation on the training data.
This is the hybrid nature of the algorithm: evolution searches the
topology space, gradient descent fits the continuous parameters inside
each topology.

## Objectives — from single- to multi-objective

The article frames architecture search as an **optimisation problem with
more than one goal**:

1. **Minimise training error.** We still want the network to fit the
   data.
2. **Minimise network complexity** — measured by the number of synaptic
   connections. Smaller networks are cheaper to run, easier to interpret,
   and generalise better when data is limited.

With only one objective (error), the search collapses onto big, dense
networks that fit best. With two objectives, the search produces a
**Pareto front**: the set of networks where you cannot improve one
objective without sacrificing the other. The article also mentions
extending the idea to **tri-objective** optimisation, but the worked
example focuses on the bi-objective case.

## The Pareto front and what it reveals

Plotting every evolved network on axes of (error, number of synapses)
exposes a frontier of non-dominated solutions. The striking finding in
the article:

- **Solution A** achieves very low error with many synapses.
- **Solution B** on the same Pareto front achieves **comparable error
  with roughly one third of the synapses**.

This is the whole point of multi-objective architecture search:
uncovering models that a single-objective loss would never surface
because it has no incentive to be small. For deployment, Solution B is
strictly more useful than Solution A — same accuracy, a third of the
compute.

## Findings

- Evolutionary search + backprop can **learn both the topology and the
  weights** of a feed-forward network without human architectural input.
- Framing the problem as multi-objective produces a **spectrum of
  solutions** trading accuracy against size.
- Sparse architectures on the Pareto front can match the accuracy of
  dense ones at a fraction of the cost, confirming the intuition that
  most connections in a manually designed network are doing little work.

## Why this matters

- It replaces human architecture intuition with a search procedure that
  is explicit, reproducible, and has a knob (the trade-off between
  objectives) that the designer can tune instead of guess.
- The bi-objective framing gives the designer a menu of architectures
  rather than a single answer — perfect for choosing based on deployment
  constraints.
- Combined with Parts 1 and 2, the series shows a full arc: random
  topologies work → their randomness is useful → their randomness can be
  replaced by targeted search.

## Key takeaways

- Represent the network as a connection list; evolve it with crossover
  and mutation; refine weights with backprop each generation.
- Minimise **both** error and connection count — the Pareto front is
  the deliverable, not a single best model.
- Small, evolved topologies can rival dense hand-designed ones. The
  series ends having demonstrated end-to-end automatic architecture
  discovery on a toy regression problem, with clear paths to scale up.
