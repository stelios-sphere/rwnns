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

## 2. Topology-aware distributed training

The model **is** the communication graph. That flips the usual
distributed-LLM-training playbook: instead of shoehorning a
uniformly-dense architecture onto a hierarchical hardware topology and
paying for an expensive any-to-any plumbing layer (all-reduce for data
parallel, all-gather + reduce-scatter for tensor parallel, micro-batch
pipeline bubbles, etc.), an RWNN lets you design the network's
connectivity to *match* the hardware from the start.

### What this buys you

- **Graph partitioning = free model parallelism.** Given a DAG and k
  GPUs, run METIS (or any min-cut partitioner) with a balance
  constraint: "assign nodes to partitions, minimise the number of
  cross-partition edges." The result *is* the inter-GPU communication
  plan, and the optimiser already minimised it. Cross-partition edges
  become the only traffic; everything else stays local.
- **Hierarchical edge probability.** ``edge_prob`` doesn't have to be
  uniform. Make it a function of physical distance:

      p_intra_gpu  ≫ p_intra_node ≫ p_inter_node ≫ p_inter_site

  so the graph naturally ends up dense within a GPU, sparse within a
  host, very sparse across hosts. The topology is shaped to bandwidth
  tiers by construction.
- **Communication cost as a third NAS objective.** Part 3's evolutionary
  search already does multi-objective Pareto fronts. Add
  cross-partition edge count as a third objective and the algorithm
  evolves architectures that are accurate *and* physically cheap to
  train.
- **Pipeline parallelism for free.** Topological levels are already the
  natural sync barrier. Assign levels to pipeline stages and you get
  pipeline parallelism with no bubble overhead beyond inter-level sync.
- **Sparse all-reduce.** Only edges crossing partitions need cross-GPU
  all-reduce. With 95 % of edges intra-GPU, the all-reduce volume
  drops ~20×. At trillion-parameter scale, "can you train at all"
  territory.

### Intentional topologies worth trying

Once random wiring is known to work as a model, the same training
machinery works for **any** DAG. Candidates that match real hardware
patterns:

- **Star.** Small number of central nodes bridge local clusters.
  Minimises cross-GPU traffic.
- **Butterfly / FFT.** log(k) depth; every GPU reaches every other in
  log(k) hops. Optimal spread for all-reduce-like patterns.
- **Hypercube.** Neighbours in bit-flipped positions. Classical HPC.
- **Fat-tree.** Literally matches modern datacenter network topology.

### Caveats

1. Near-term, the RWNN must be useful as a *model* for any of this to
   matter. The natural first step is **RWNN replacing the transformer
   FFN** (where most params live) while attention handles sequence
   mixing as today.
2. Load balancing is harder than in a uniform transformer — partitioners
   handle it but it's not automatic.
3. Fault tolerance changes: replacing a dead GPU's specific nodes is
   harder than resharding a uniform tensor.
4. NCCL/collective libraries prefer uniform patterns; lots of small
   per-level collectives may hurt throughput unless batched.
5. Savings are realised only if the topology is committed at design
   time. Retrofitting a trained dense model into a sparse one captures
   little of this.

### Why this is the strongest RWNN pitch

Model-quality-wise, attention remains a very good prior for language.
Systems-wise, transformers force all-to-all and pay for it forever.
RWNNs make the network's communication pattern an explicit design
variable; graph partitioning and evolutionary search can then shape
the network to the hardware. That is the rare case where sparse,
random-topology architectures don't just compete with transformers —
they can structurally *beat* them on training economics.

---

(Add future ideas below this line.)
