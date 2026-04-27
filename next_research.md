# Next research direction: parallel-mirror RWNN-LM

*The architecture I'd build next. Self-contained design memo so this can
be picked up after the current Option-1 (random-bilinear-fraction) run
gives us a baseline number.*

---

## Recommendation

**Option 3 — parallel mirror networks.**
Clean architecture, clean gradients, easy implementation, easy ablation,
fits the GPU. What I'd build.

## The architecture

Two complete RWNNs running in parallel, **mirror-image topology**,
sharing only the input embedding (and merging at the logits):

```
ids
 │
 ▼
embedding [B, T, d_model] → flatten → flat [B, T·d_model]
 │                                       │
 ├──► RWNN_L  (all-linear, own weights)  ──► logits_L [B, V]
 └──► RWNN_B  (all-bilinear, own weights) ──► logits_B [B, V]
                                                    │
                                          (sum, or learnable α-mix)
                                                    ▼
                                              logits [B, V]
```

Same node count and edge topology in both networks (built from the same
``build_layered_rwnn`` call with the same seed). Each branch has its own
weight tensor, its own forward pass, its own kernel set. They never see
each other's intermediate activations — only at the output do they
merge. That's the "2 mirror networks connected on the input side"
reading.

For an inputs-only-shared, outputs-summed model:

- ``RWNN_L`` = "the linear path" (cheap, statistically rich features)
- ``RWNN_B`` = "the multiplicative path" (content-addressable interactions)

Same job split as transformer attention + FFN, just with different
primitives.

## Why this design wins (vs. the alternatives we considered)

### 1. Cleanest research story

Each branch is fully expressive in its own function class. No shared
weights doing double duty, no random subset of nodes flipping kind. The
architecture description fits in one sentence: *"two parallel RWNNs of
the same topology, one all-linear and one all-bilinear, summed at the
output."* That's a publishable design — easy to defend, easy to
ablate (drop a branch, you get the unimix baseline for free).

### 2. Independent gradients = better optimization

The biggest reason. With "both modes at every node", the gradient on
each shared weight comes from both linear and bilinear contributions;
the optimizer is doing implicit credit assignment between two
computational primitives that are summed together. That's a notoriously
fiddly setup — gradient interference, weird canceling minima, awkward
init. Parallel-mirrors' branches don't see each other until the output
sum, so each branch's weights only feel gradients from their own forward
pass. Much closer to two well-understood single-mode networks plus a
residual.

### 3. We have the headroom

The current run uses **2 GB of 24 GB GPU memory**. Doubling to ~4 GB
still leaves 20 GB unused. Compute time per step doubles to ~150 ms,
but in absolute terms that's still ~400 steps/min — comfortable for any
reasonable training duration. We are not memory-bound or compute-bound;
nothing structural is gained by squeezing into a smaller, more coupled
design.

### 4. Implementation is the easiest of the three options

No kernel changes. No ``node_kinds`` branching inside the inner loop
(we can simplify those kernels for both cases). Just two ``RWNN(graph)``
modules in ``RWNNLM.__init__`` plus
``logits = self.rwnn_L(flat) + self.rwnn_B(flat)`` in forward. ~30 lines
of code change total.

```python
class RWNNLM_parallel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        graph_lin = build_layered_rwnn(
            n_nodes=cfg.n_nodes, edge_prob=cfg.edge_prob,
            n_layers=cfg.n_layers,
            n_in=cfg.context_length * cfg.d_model,
            n_bias=cfg.n_bias, n_out=cfg.vocab_size,
            seed=cfg.seed, bilinear_fraction=0.0,
        )
        graph_bil = build_layered_rwnn(
            n_nodes=cfg.n_nodes, edge_prob=cfg.edge_prob,
            n_layers=cfg.n_layers,
            n_in=cfg.context_length * cfg.d_model,
            n_bias=cfg.n_bias, n_out=cfg.vocab_size,
            seed=cfg.seed, bilinear_fraction=1.0,
        )
        self.rwnn_L = RWNN(graph_lin, device=cfg.device)
        self.rwnn_B = RWNN(graph_bil, device=cfg.device)
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

    def forward(self, ids):
        B, T = ids.shape
        flat = self.token_emb(ids).reshape(B, T * self.cfg.d_model)
        return self.rwnn_L(flat) + self.rwnn_B(flat)
```

### 5. Mirrors a known-good design pattern

Modern transformers run attention and FFN in parallel and sum their
contributions (with residuals between blocks). Parallel mirrors apply
that same idiom to our substrate: two computational primitives, run in
parallel, summed at the output. We're not inventing an architecture
style — we're applying a battle-tested one in a new model class.

## Honest tradeoffs

1. **2× compute per step.** Two full forward passes, two full backward
   passes. At ~75 ms/step now, expect ~150 ms/step. Still very fast
   for ~130 M edges of compute on a 4090.

2. **2× total parameters at the same ``n_nodes``.** Goes from 65 M to
   ~130 M (each branch ~65 M edges). AdamW state doubles too → ~1 GB
   more memory. Comfortable on 24 GB.

3. **Param-budget option.** If keeping ~65 M total params is a hard
   requirement, halve ``N_NODES`` to ~22,500. Each branch then has
   ~16 M edges, 33 M total = 66 M, matching the current model size.
   We give up per-branch capacity to keep the count constant, but get
   bilinear coverage at the same total budget.

4. **Output combination decision.** Simple sum is the obvious starting
   point (residual style). A scalar learnable ``α · L + (1-α) · B`` is
   one extra parameter and lets the model pick the mix. Per-vocab-position
   is ``V`` extra params — still negligible. **Start with the simple
   sum**; it's the cleanest baseline.

5. **Init balancing.** Linear and bilinear branches have different
   natural pre-activation scales at init (sigmoid gating compresses to
   ~0.5 at zero). Probably want to scale-balance the two branches' init
   weights so neither dominates early. ``1/√(2·fan_in)`` for both is a
   reasonable starting point — halves the standard init since both
   branches sum into the same logits.

6. **No interaction between branches.** A possible downside: the bilinear
   branch can't see the linear branch's hidden activations, and vice
   versa. They can only coordinate through the output error signal. This
   is a slightly weaker form of cross-talk than what transformers do
   (where each block reads from the residual stream). Feature, not bug,
   for clean experimentation.

## Comparison to the other options on the table

| design | linear-bilinear interaction | params | compute | implementation |
|---|---|---:|---:|---|
| Random ``bilinear_fraction=0.05`` placement (current shipped) | none — each node is one or the other; only ~5 % of nodes are bilinear | 1× | 1× | shipped |
| Both modes at every node (single output) | shared activation, coupled gradients | 1× shared, 1.3× separate | 1.5× | small kernel patch |
| **Parallel mirror networks (this proposal)** | independent until output | 2× | 2× | two ``RWNN`` modules + sum |

## Empirical predictions

If we run all three at the same dataset (TinyStories) and same
``n_nodes`` skeleton, the expected ranking by val cross-entropy:

- **Loss floor**: parallel mirrors lowest (most capacity, cleanest
  gradients). Random-5 %-bilinear highest (effectively the same as
  linear-only because so few bilinear nodes exist).
- **Convergence speed per step**: mirrors slowest in wall time per step
  but possibly fewer steps to converge — more capacity, cleaner signal.
  Net is likely a wash or slight win.
- **Sample quality**: mirrors should produce noticeably better short
  bigram-to-trigram coherence because the bilinear branch finally has
  enough bandwidth to express pairwise interactions. The current 5 %
  scattering doesn't get this signal across.

## Strategic plan

1. **Let the current Option-1 run finish or plateau** so we have a
   baseline number for the same model size + dataset.
2. **Implement parallel mirrors** as ``RWNNLM_parallel`` (or as a flag
   on the existing ``RWNNLM`` class) — see code sketch above. New
   ``run_parallel.py`` or a ``--parallel`` switch on ``run.py``.
3. **Train the parallel-mirror version** on TinyStories with everything
   else held constant: same tokenizer, same context length, same
   ``n_nodes`` (or halved if budget-constrained), same LR schedule.
4. **Compare loss curves and samples**. The val-loss delta between
   single-network and parallel-mirror tells us whether bilinear gating
   at full bandwidth earns its keep on language modelling.
5. **If parallel mirrors win clearly**, the natural next step is to
   add evolutionary search (Part 3 style) over each branch's edge
   topology — the architecture is now a clean two-branch graph, perfect
   substrate for multi-objective NAS over (val loss, edges per branch).

## Implementation checklist (for the next session)

- [ ] Add ``RWNNLM_parallel`` class to ``LLM/src/llm.py`` (or extend
      ``RWNNLMConfig`` with a ``parallel: bool`` flag).
- [ ] In the parallel branch, build two ``RandomDAG`` objects with the
      same seed, ``bilinear_fraction=0.0`` and ``1.0``.
- [ ] Tweak weight init: scale by ``1/√2`` so the summed output has
      similar variance to a single-branch network.
- [ ] Add a ``--parallel`` flag to ``run.py`` (or a sibling
      ``run_parallel.py``) — same training loop otherwise.
- [ ] Reuse the existing tokenizer and pre-encoded corpus caches
      (no need to retrain BPE).
- [ ] Run, compare val curve to the Option-1 baseline.
- [ ] Add ``sample.txt`` and ``loss_curve.png`` outputs to a separate
      ``LLM/example_parallel/`` directory (or suffix filenames) so
      both runs' artefacts coexist.

That's the full plan. Pick this up after the current run gives a number.

---

## Constraint: keep this a *pure* RWNN

This whole research line is about what a randomly-wired graph **on its
own** can express, with no transformer scaffolding. Every extension
explored under "next_research" must therefore stay inside the RWNN
abstraction:

- **No residual connections** between sub-graphs. The graph IS the
  computation.
- **No layer / RMS normalisation.** If gradient stability requires
  norms, the right answer is to fix the graph topology or activation
  rather than bolt on a normaliser.
- **No "RWNN as FFN inside a transformer block".** That line of
  research exists in the literature already (sparse-MoE FFNs, etc.);
  it's not what this repo is investigating.

Acceptable extensions:
- New node *kinds* (linear, bilinear, trilinear, gated, …) co-existing
  in one DAG.
- Connectivity rules (position-local, hierarchical edge probability,
  evolved topologies).
- Whole-network primitives (parallel mirrors, multiple co-trained
  graphs sharing input/output).
- Different builders (skip-connection-rich, fully sparse, etc.) that
  produce a pure DAG.

The boundary is: *if it isn't expressible as a single random feed-forward
DAG (possibly with multiple kinds of nodes and possibly multiple co-trained
graphs sharing only inputs/outputs), it's out of scope here.*

---

## RWNN attention — next thing to try if the current run fails

Bilinear gating nodes (current) are **not** attention. They cover
GLU/SwiGLU territory: gated multiplicative interactions per node. They
do not compute **content-based similarity between two activations**, do
not normalise via softmax, and do not perform **soft routing across
positions**. Attention is the missing primitive for sustained
syntactic dependency in language modelling, and the post-step-5000 val
plateau we're hitting at every ctx≥512 configuration likely reflects
its absence.

### Two new pure-RWNN node kinds

This proposal adds attention as a primitive *without* introducing any
transformer scaffolding (no residuals, no normalisation outside the
node, no separate sub-blocks). Just two node kinds in the same DAG:

1. **Product node**: ``a_i = a_g · a_v`` (or ``Σ a_g · a_v`` over
   paired parents). **No weights** — pure activation product. Used
   for similarity / Q·K-style scoring. Symmetric in its two parents.
2. **Softmax-aggregator node**: takes K pairs of ``(score, value)``
   parents, computes ``weights = softmax(scores)``, returns
   ``Σ_k weights[k] · values[k]``. The normalisation lives *inside*
   the node — no external norm needed. Output is the soft-weighted
   sum of values.

### What an "attention head" looks like inside the random DAG

A single attention-head subgraph becomes:

- **N product nodes**, each computing a Q·K-style similarity between
  two specific projection-pair parents (where the projections are
  themselves earlier-layer linear nodes acting as Q and K).
- **N corresponding linear nodes** computing V-style values for those
  same N positions.
- **1 softmax-aggregator node** that consumes the N (score, value)
  pairs and emits one attended output.

The "Q-projection" and "K-projection" are just learned linear nodes
applied to the input; the model figures out per-head what to compare.
With a few hundred such heads scattered through the DAG (each head's
N varies — small for local heads, large for global), the network has
real attention machinery built from RWNN primitives.

### Why this is pure-RWNN compatible

- All four node kinds (linear, bilinear, product, softmax-aggregator)
  live in **one DAG**. No sub-blocks, no residuals between them.
- **No external normalisation** — softmax is internal to the
  aggregator node, the same way `tanh` is internal to a linear node
  today.
- The **graph is still the computation**. Connectivity rules
  (random, position-local, evolved) work the same as before; only
  the per-node operator set has grown.

### Topology strategies

Two options for getting attention-shaped subgraphs into the random DAG:

A. **Pure random**. Just allow the four node kinds to be sampled in
   any proportion. The RWNN may or may not develop attention-shaped
   subgraphs spontaneously. Cheap to try; might not work without
   structural prior.
B. **Seeded attention heads**. Pre-place a small number of
   hand-designed attention subgraphs in the DAG (e.g. 8 heads, each
   with N=64 product+linear+softmax nodes wired correctly), then
   randomly wire the rest of the graph around them. The hand-design
   gives the model attention as a primitive; the surrounding random
   wiring lets it find non-attention computation paths too.
C. **Evolutionary search** (Part 3 style). Multi-objective NAS over
   (val loss, total edges, fraction-attention-shaped). Let evolution
   discover the right mix.

I'd start with B (seeded) — it's the most direct test of whether
attention is the missing primitive. If seeded heads improve val
substantially over the current bilinear-only ceiling, that's the
hypothesis confirmed; A and C are then worth pursuing.

### Implementation cost

Two new node kinds means:
- Two more branches in ``forward_level_kernel``.
- Two more branches in ``backward_propagate_kernel`` and
  ``backward_weight_kernel``.
- Softmax requires a small intra-node reduction across paired
  parents. Doable in Triton with `tl.softmax` over a fixed-size
  vector, or manually with `max + exp + sum`.
- Backward through softmax: standard
  `softmax(s)_k · (1 - softmax(s)_k)` for the diagonal,
  `-softmax(s)_j · softmax(s)_k` off-diagonal. ~30 lines of Triton.

Estimated effort: ~half a day to add the kernels and wire them through
``RandomDAG`` / ``RWNN`` / ``llm.py``. Add a flag like
``cfg.attention_heads = 8`` for the seeded variant.

### Implementation checklist

- [ ] Extend ``node_kinds`` to support {0=linear, 1=bilinear,
      2=product, 3=softmax_aggregator}.
- [ ] Add product-node forward/backward kernel branches.
- [ ] Add softmax-aggregator node forward (reduction across pairs)
      and backward (Jacobian of softmax).
- [ ] Builder: add a "seed_attention_heads" option to
      ``build_layered_rwnn`` that pre-places K attention subgraphs
      with the right wiring.
- [ ] Verify gradients vs. a pure-PyTorch reference at small scale
      (just like the existing tests for linear and bilinear).
- [ ] Train the seeded-attention variant on WikiText-103 with
      otherwise-identical config to the current parallel-mirrors run.
- [ ] If val drops substantially below ~4.5: the hypothesis is
      confirmed and we expand attention coverage.

### Empirical predictions

- **Val loss** should drop into the ~3.0-3.5 zone — comparable to
  small transformers — *if* attention is what the model was missing.
  If val stays around ~4.5, the bottleneck is somewhere else (maybe
  simply more data needed, or a deeper structural problem).
- **Sample quality**: should produce locally coherent multi-token
  spans, with proper coreference (`The X ... it ...`) and
  consistent named entities across a paragraph — neither of which
  the current bilinear-only model can sustain.
- **Training speed**: expect ~30-50 % slower per step than current,
  largely from the softmax-aggregator's per-node reduction work.
