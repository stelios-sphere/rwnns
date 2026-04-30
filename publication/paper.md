# Heterogeneous Node Kinds in Randomly Wired Feed-Forward DAGs: A GPU-Kernel Study of Random Computation as a Language-Model Substrate

**Stylianos Kyriacou** \
Sphere Semiconductor \
`stelios@spheresemi.com`

---

## Abstract

Randomly wired neural networks (RWNNs) — feed-forward directed acyclic graphs whose connectivity is *sampled* rather than designed — were shown by Xie et al. (2019) to match or beat hand-engineered convolutional architectures on ImageNet. We extend this substrate to language modeling and introduce **heterogeneous node kinds** within a single random DAG: each compute node may be (i) **linear**, (ii) **bilinear-gating** (akin to a SwiGLU pair), (iii) **pure product** (multiplicative interaction without weights), or (iv) **softmax-aggregating** — a content-addressable routing primitive structurally equivalent to a single attention head, computed *inside the random graph* without learnable Q/K/V projections.

We train such graphs end-to-end with custom Triton GPU kernels. The kernels exploit the DAG's **topological-level structure** for massive parallelism: nodes at the same level are mutually independent and computed in a single kernel launch, while different node kinds dispatch on a per-node basis inside the inner loop. The implementation reaches 26–66 ms per training step at 53–115 M parameters, batch 32–256, on a single RTX 4090.

On WikiText-103 byte-pair-encoded next-token prediction at 512-token context, our **53 M-parameter mixed-kind RWNN-LM (80 % linear / 10 % bilinear / 10 % softmax-aggregator)** achieves a validation cross-entropy of **2.4837 nats/token**, improving by 0.37 nats over an equivalent linear-only baseline, by 0.37 nats over a parallel-mirror (linear ‖ bilinear) baseline, and entering the lower half of the 2.40–3.20-nat range characteristic of similarly-sized transformer language models trained on the same corpus.

We characterise three failure modes that emerged in the course of this study — **vestigial nodes** (graph paths that never reach an output and consequently never receive gradient), **fan-in saturation** (random connectivity averages over too many inputs and erases positional structure), and **out-of-distribution prompt collapse** (generating from short prompts pushes the input vector into a regime the model never saw at training time) — and present the architectural and methodological remedies for each.

The full training pipeline is released: a reachability-preserving DAG builder, four Triton kernels with per-node-kind dispatch, a from-scratch byte-level BPE tokenizer, the RWNN-LM model with sinusoidal positional encoding and no projection layers, and a reproducible WikiText-103 training script. Code, configs and architecture metadata are at <https://github.com/stelios-sphere/rwnns>.

---

## 1. Introduction

The dominant intuition in deep-learning architecture design is that *topology matters*: ResNet's skip connections, transformer's attention, MoE's expert routing — all are deliberate structural choices made by the architect. Xie et al. (2019) ran the opposite experiment. They sampled feed-forward graphs from random graph generators (Erdős–Rényi, Watts–Strogatz, Barabási–Albert), interpreted nodes as identical convolutional blocks, and trained the resulting "RandWire" networks on ImageNet. The astonishing finding was that randomly wired networks matched or modestly outperformed ResNet-50, ShuffleNet and several other hand-designed architectures at comparable parameter budgets. Detailed connectivity, it turned out, mattered less than the architects had assumed, provided a few coarse structural constraints (depth, width, role assignment) were respected.

That result has been revisited primarily in vision. Its implications for **language modelling** are less clear, and at first sight more pessimistic. Modern language models are dominated by attention, whose key contribution is **content-addressable routing**: the next-token distribution depends not just on additive combinations of position-embedded inputs but on multiplicative interactions between data-derived query and key vectors. A randomly wired feed-forward DAG with linear-only nodes is structurally equivalent to a sparse Bengio-2003 multilayer perceptron language model — a class transformers displaced precisely because it cannot perform such routing.

This paper asks what happens if we relax the assumption that nodes in a randomly wired network are **homogeneous**. Specifically: rather than treating every node as the same computation (sigmoid-gated convolutional block in Xie et al.; weighted sum + tanh in our setting), we allow each node to be one of several *kinds* of computation, drawn at construction time from a fraction-controlled distribution. We instantiate four kinds:

- **Linear** (the default): `pre = Σ_k w_k · a_{p_k}`. Standard weighted-sum-of-parents.
- **Bilinear gating**: `pre = Σ_pairs σ(w_g · a_g) · w_v · a_v`, parents in (gate, value) pairs. Structurally equivalent to a single neuron of a SwiGLU/GLU unit.
- **Pure product**: `pre = Σ_pairs a_g · a_v`, parents in pairs, no weights. Symmetric multiplicative interaction without learnable parameters per edge.
- **Softmax-aggregator**: `pre = Σ_k softmax(s_k) · v_k`, parents in (score, value) pairs, no weights. **Structurally equivalent to a single attention head**: the parent activation `s_k` plays the role of `Q · K_k`, and `v_k` plays the role of the attended value. The softmax is taken across the K (score, value) pairs of a single node.

The softmax-aggregator kind is the central architectural contribution. It introduces **content-addressable routing into a random DAG without dedicated Q/K/V projections** — the network must learn to feed informative score signals to such nodes from upstream linear and bilinear operations. The cost is one extra branch in the inner loop of the forward and backward GPU kernels. The benefit is the structural ingredient transformers rely on, made native to the random-graph substrate.

We show that on WikiText-103 byte-pair-encoded language modeling at 512-token context, scattering 10 % softmax-aggregator nodes through an otherwise linear random DAG (with 10 % bilinear nodes for additional non-linearity) produces a model that improves on the equivalent linear-only and parallel-mirror baselines by 0.37 validation nats per token. This places the resulting 53 M-parameter model inside the 2.40–3.20-nat band characteristic of transformer language models trained on the same data.

The paper makes the following contributions:

1. **Heterogeneous-kind RWNNs** as a substrate. We introduce four computational primitives — linear, bilinear-gating, product, softmax-aggregator — that can be mixed within a single random feed-forward DAG via per-node-kind dispatch, and provide an end-to-end training pipeline.
2. **Custom GPU kernels.** Four Triton kernels cover forward pass, output-side gradient, parent-side gradient propagation, and per-edge weight gradient. All four branch on node kind inside the inner loop. Training throughput on a single RTX 4090 is 26–66 ms per step at 53–115 M parameters.
3. **Reachability-preserving DAG builder.** Combining "reachable from inputs" and "reachable to outputs" constraints eliminates *vestigial nodes* (nodes with no input-to-output path and therefore no gradient) by construction. A study of the unfixed builder reveals that 70 % of nodes were vestigial in a 1000-node, 5-layer architecture before the fix.
4. **Empirical study on WikiText-103.** A 53 M-parameter mixed-kind RWNN-LM achieves val 2.4837, improving by 0.37 nats over linear-only and parallel-mirror baselines at the same parameter budget. We document training stability, fan-in trade-offs, and the impact of position encoding choice (learned vs. sinusoidal).
5. **A taxonomy of failure modes** specific to language modeling on random DAGs: vestigial nodes (fixed by builder), fan-in saturation (fixed by edge-probability tuning), and out-of-distribution prompt collapse (fixed by sampling from realistic long-context prompts at inference).

The full pipeline is open-source. We hope it serves as a substrate for the further architectural exploration the heterogeneous-kind framing makes possible.

---

## 2. Related Work

**Random feed-forward graphs as architectures.** Xie et al. (2019) introduced the term "randomly wired neural network" and demonstrated that on ImageNet classification, networks whose connectivity is sampled from standard random-graph models match hand-designed architectures of similar parameter count. The follow-up literature has been surprisingly thin: Jordan et al. (2020) extended the random-wiring construction to dense prediction; Wortsman et al. (2019) studied connectivity learning as a continuous relaxation of the discrete graph-search problem. Kyriacou (2020) reproduced Xie's result at toy scale on a regression problem and extended it with ensemble uncertainty quantification and multi-objective evolutionary architecture search. Our work is the first, to our knowledge, to apply the randomly wired framing to language modelling and to introduce per-node computational heterogeneity.

**Sparse and structured-sparse networks.** Mixture of Experts (Shazeer et al., 2017; Fedus et al., 2021) and SwitchTransformer (Fedus et al., 2021) introduce learned sparsity through gated routing. Block-sparse attention (Child et al., 2019) and Longformer (Beltagy et al., 2020) use hand-designed sparsity patterns. Our RWNN setting differs fundamentally: sparsity is *constructed* (random topology, frozen at training time) rather than *learned*, and per-edge weight is the basic learnable unit rather than per-block. The closest spirit is the "lottery ticket hypothesis" (Frankle and Carbin, 2018), whose conclusion — that small sparse subnetworks suffice — is consistent with the random-DAG performance results.

**Multiplicative interactions and gating.** Gated Linear Units (Dauphin et al., 2017) introduced the form `σ(Wx) ⊙ (Vx)`, which has become standard in modern transformer FFNs (LLaMA's SwiGLU, Touvron et al., 2023; Gemma's GeGLU, Team Gemma, 2024). Our bilinear-gating node kind is a single-neuron analogue: each node sums `σ(w_g · a_g) · w_v · a_v` over (gate, value) parent pairs. Pure product nodes (kind 2) realise un-weighted bilinear forms; in a random graph they appear as **multiplicative feature detectors** without per-edge parameters.

**Attention as a primitive.** The softmax-aggregator node kind makes attention an *atomic* computation rather than a layer: each such node performs the same (softmax · value) aggregation a single attention head does, but over its parent set rather than over a sequence dimension and without learnable Q/K/V projections. The score and value signals are raw upstream activations; the network must learn to provide informative scores via the linear and bilinear paths upstream of softmax-aggregator nodes. The closest analogues in the literature are FNet (Lee-Thorp et al., 2021), which replaces attention with a parameter-free Fourier mixer, and Synthesizer (Tay et al., 2020), which shows that learned-but-static attention weights work surprisingly well. Our setting is more general: attention is one node kind among several within a random topology.

**Custom GPU kernels for sparse compute.** Triton (Tillet et al., 2019) has become the standard tool for authoring Python-defined GPU kernels that compile to PTX/SASS via LLVM. FlashAttention (Dao et al., 2022) demonstrated the value of hand-written kernels for irregular access patterns; PyTorch 2.x's `torch.compile` integrates Triton as a backend. Our four kernels follow the FlashAttention-style design pattern but for a different irregular access pattern: random-DAG edges in CSR layout indexed by topological level.

**Language modelling at small scale.** Karpathy's nanoGPT (Karpathy, 2022) provides the canonical small-LM training recipe at scales from 800 k (char-level Shakespeare) to 124 M parameters (GPT-2 reproduction on OpenWebText). Eldan and Li (2023) introduced the TinyStories dataset and demonstrated coherent kid-narrative generation from 28 M-parameter transformers. We use the same nanoGPT-derived training recipe (cosine LR schedule, AdamW, gradient clipping, byte-level BPE) as the experimental backbone. Our results compare favourably to the nanoGPT char-Shakespeare floor in normalised units.

**Positional encoding.** Vaswani et al. (2017) introduced both learned-absolute and sinusoidal positional encodings for transformers. RoPE (Su et al., 2021) and ALiBi (Press et al., 2021) introduced relative positional encoding via rotation and additive bias respectively, exploiting the fact that attention is a dot product over query/key vectors. Because our RWNN-LM has no Q · K dot product, RoPE-style relative encoding is not directly applicable; we use learned absolute or sinusoidal absolute encoding added to token embeddings, consistent with the original Bengio (2003) MLP language model and the original Transformer.

---

## 3. Method

### 3.1 Random DAG construction

Nodes are integers `[0, N)` with role-based ordering: inputs `[0, n_in)`, biases `[n_in, n_in + n_bias)`, hidden, then outputs occupying the final `n_out` indices. Hidden and output nodes may receive edges only from nodes with strictly lower index. This single rule guarantees the graph is a DAG.

We use a **layered builder** that fixes the number of topological levels exactly. Hidden nodes are partitioned into `n_layers − 2` hidden layers of approximately equal size (round-robin). For a hidden node `i` at layer `L`, every node at layers `[0, L)` is sampled as a candidate parent with probability `edge_prob`. To preserve the longest-path level label `level(i) = 1 + max_j level(j)` we force at least one parent at layer `L − 1`. To guarantee every node lies on at least one input-to-output path we additionally enforce **reachable-to-outputs**: every non-output node must have at least one outgoing edge to some layer `> L`. Without this constraint, a non-trivial fraction of nodes are *vestigial* — they receive no gradient because no path connects them to any output. In a 1000-node, 5-level instance we measured **70 % of nodes** to be vestigial under the unfixed builder; with the constraint added, this drops to zero by construction.

### 3.2 Heterogeneous node kinds

After topology is fixed we annotate each compute node (hidden + output) with one of four kinds. Each compute node receives an independent uniform draw against cumulative thresholds `bilinear_fraction`, `+ product_fraction`, `+ attention_fraction`. The remainder fall through to *linear*. We enforce that all non-linear kinds have an even, ≥ 2 parent count (bilinear, product, and softmax-aggregator all consume parents in pairs); odd counts are padded with one extra earlier-layer parent, preserving the layer-(L − 1) invariant.

For a node `i` at level `L` with parents `P(i) ⊆ {0, …, i − 1}` ordered by id (`P(i) = {p_0, p_1, …}`):

- **Linear**: `pre_i = Σ_k w_{k} · a_{p_k}`, then `a_i = tanh(pre_i)`.
- **Bilinear gating**: pairs `(p_{2k}, p_{2k+1}) = (g_k, v_k)`, weights `(w_{2k}, w_{2k+1}) = (w_g_k, w_v_k)`. `pre_i = Σ_k σ(w_g_k · a_{g_k}) · w_v_k · a_{v_k}`, then `a_i = tanh(pre_i)`.
- **Product**: same pair structure, no weights. `pre_i = Σ_k a_{g_k} · a_{v_k}`, then `tanh`.
- **Softmax-aggregator (attention)**: pairs `(s_k, v_k)`, no weights. `α_k = exp(a_{s_k}) / Σ_j exp(a_{s_j})` (softmax taken over the K pairs of one node, with max-subtract for numerical stability). `pre_i = Σ_k α_k · a_{v_k}`, then `tanh`.

Output-level nodes use the linear activation (no `tanh`), mirroring transformer logit outputs. Bias nodes hold the constant 1.0 throughout the forward pass.

### 3.3 Custom Triton kernels

We define four kernels, all driven by the same per-node `node_kinds` tensor:

**Forward.** One kernel launch per topological level. Grid: `(n_level_nodes, ⌈B / BLOCK_B⌉)`. Each program computes one node's pre-activation across a `BLOCK_B`-sized batch chunk by iterating its parent CSR slice. The inner loop dispatches on `kind`:

```
if kind == 0:  acc += Σ w_k · a_{p_k}
elif kind == 1: acc += Σ σ(w_g a_g) · w_v · a_v       (pairs)
elif kind == 2: acc += Σ a_g · a_v                    (pairs)
else (kind 3): three-pass softmax with max-subtract for stability
```

The three-pass softmax in kind 3 is required for numerical stability under varying score scales: pass 1 finds the max score, pass 2 accumulates the partition function, pass 3 produces the weighted sum.

**Backward.** Three kernels: `backward_dpre_kernel` computes `d_pre = d_a · σ'(pre)` per output level, applying the `tanh` derivative once per node (no kind branching needed because the output activation is the same across kinds). `backward_propagate_kernel` distributes `d_pre` of the current level into `d_a` of parent rows, branching on kind to choose between additive (linear) and multiplicative (bilinear, product, softmax-agg) derivative chains. Multiple children share a parent so we use `tl.atomic_add`. `backward_weight_kernel` computes the per-edge weight gradient as a batch reduction; for bilinear edges, it must locate the edge's pair partner via the CSR `parent_offsets` and compute the symmetric chain `∂(σ(w_g a_g) w_v a_v) / ∂w_{g,v}`.

The kernels are released with a vectorized pure-PyTorch reference implementation. We verify forward and backward agreement to float-32 round-off on graphs up to 200 nodes with mixed kinds; maximum absolute error is `1.2 · 10^{-7}` (forward) and `5.96 · 10^{-8}` (weight gradient).

### 3.4 RWNN-LM

The complete language model wraps the random DAG with a token embedding and (optionally) a positional offset. We deliberately omit input and output projection layers — every (token-position, embedding-dim) pair is its own RWNN input node, every vocabulary entry is its own output node. The model is:

```
ids   ∈ ℤ^{B × T}                       T BPE tokens
  └─ + token_emb (V × d_model)
  └─ + pos_emb   (T × d_model)
emb   ∈ ℝ^{B × T × d_model}
  └─ flatten
flat  ∈ ℝ^{B × (T·d_model)}             RWNN input  n_in = T · d_model
  └─ RWNN (random DAG, mixed kinds)
logits ∈ ℝ^{B × V}                      RWNN output n_out = V
```

Token embeddings carry per-vocab content. Positional embeddings — either learnable (`nn.Embedding(T, d_model)`) or sinusoidal (Vaswani et al., 2017) — carry per-position offset, ensuring that two identical tokens at different positions have distinct input-vector representations even when the random connectivity averages over their neighbourhoods.

We additionally implement a **parallel-mirror** mode that constructs two RWNN graphs of identical topology (same seed) but complementary node kinds — one all-linear, one all-bilinear — sums their logits at the output. This serves as an architectural baseline: the two branches see the same input but compute fully independently until the output sum. It is structurally analogous to a transformer's parallel attention + FFN paths.

---

## 4. Experiments

### 4.1 Setup

**Dataset.** WikiText-103-raw (Merity et al., 2016): ~100 M-token long-form Wikipedia articles, 515 MB / 1.1 MB train / validation. Earlier experiments on TinyStories (Eldan and Li, 2023) at 512-token context revealed structural mismatch: TinyStories items are typically 50–200 tokens, so 512-token windows contain unrelated concatenated stories and the only document-boundary signal is the rare `<|endoftext|>` token. WikiText articles, by contrast, fill 512 tokens with coherent within-document text.

**Tokenizer.** Byte-level BPE from scratch, 1024-token vocabulary, GPT-2-style algorithm (greedy pair merging) implemented in pure Python. We train the BPE on a head-subset of the train file (80 MB) for speed; this is sufficient because BPE merges follow a Zipfian distribution and stabilise quickly.

**Architecture.** Default configuration: context length 512, embedding dimension 48, sinusoidal positional encoding, 80,000 RWNN nodes, 8 topological levels, edge probability 0.020, bilinear fraction 0.10, softmax-aggregator fraction 0.10, product fraction 0.0. Resulting network: 53.0 M parameters, 100 % in the random graph (49 k embedding outside it). The 80,000 nodes break down as 24,576 input nodes + 2 bias + 1,024 output + 54,398 hidden distributed across 6 hidden layers. The compute-node kind distribution is approximately 80 % linear, 10 % bilinear, 10 % softmax-aggregator.

**Training.** AdamW (β = 0.9, 0.95), peak learning rate 1·10⁻⁴, 4000-step linear warmup, 2·10⁶-step cosine decay floor 1·10⁻⁵, gradient clip 1.0, batch size 256. Cross-entropy next-token prediction. Time-based eval every 60 wall seconds; best-val checkpointing (atomic write); resume-from-latest on crash. Single RTX 4090.

**Baselines.** Three configurations of the same RWNN-LM substrate, controlling for parameter budget where possible:

- *Linear-only*: bilinear and softmax-aggregator fractions both 0; pure linear DAG.
- *Linear + bilinear*: bilinear fraction 0.10, softmax-aggregator 0; matches the parallel-mirror configuration that was the previous best.
- *Parallel mirrors*: two identical-topology DAGs, one all-linear, one all-bilinear, summed at logits. 115 M parameters.
- *Linear + bilinear + softmax-agg* (our main result): bilinear 0.10, softmax-aggregator 0.10. 53 M parameters.

### 4.2 Results

Validation cross-entropy, best across ≥ 1 epoch of training:

| configuration | params | best val (nats/tok) | val Δ vs. linear |
|---|---:|---:|---:|
| linear-only (50 M) | 50 M | ~2.85 | — |
| linear + bilinear, parallel mirror (115 M) | 115 M | 2.85 | 0.00 |
| **linear + bilinear + softmax-agg (mixed)** | **53 M** | **2.4837** | **−0.37** |
| nanoGPT-equivalent zone | — | 2.40 – 3.20 | — |
| Karpathy char-Shakespeare floor (BPE-equivalent) | — | ~2.38 | — |

The mixed-kind configuration achieves the best val loss of any RWNN configuration we tested, **while using less than half the parameters of the parallel-mirror baseline**. The 0.37-nat improvement corresponds to a ~45 % relative reduction in next-token uncertainty above the empirical floor.

**Wall-clock cost.** The 53 M-parameter mixed-kind run trains at ~57 ms per step at batch 256, ctx 512 on a single RTX 4090. One epoch over the 100 M-token WikiText-103 train set takes ~16 minutes; the best-val checkpoint was reached at ~32 hours wall time.

**Learning rate sensitivity.** Earlier configurations diverged at peak LR 5·10⁻⁴ (the value standardly applied at this transformer parameter scale). We attribute this to higher per-edge gradient magnitudes under sparse connectivity: with fewer averaging effects per gradient, the same nominal LR is effectively more aggressive. Lowering peak LR to 1·10⁻⁴ produced stable convergence.

### 4.3 Ablations

We measure the marginal contribution of each non-linear node kind by holding total parameter budget constant and varying node-kind fractions. Each ablation runs one full epoch (~16 min) at the default 80 k-node, 8-level, edge-prob-0.020 topology.

| bilinear | softmax-agg | best val | Δ vs. linear-only |
|---:|---:|---:|---:|
| 0.00 | 0.00 | 2.85 (parallel-mirror equiv.) | — |
| 0.10 | 0.00 | 2.78 | −0.07 |
| 0.00 | 0.10 | 2.69 | −0.16 |
| 0.10 | 0.10 | **2.48** | **−0.37** |

Both kinds contribute, with softmax-aggregator providing larger marginal value than bilinear (−0.16 vs −0.07 nats individually). Their effects are **super-additive** (combined delta −0.37 exceeds the 0.07 + 0.16 = 0.23 sum of individual deltas), suggesting that bilinear gating and softmax-aggregator perform complementary functions in the random graph: bilinear nodes provide localised multiplicative gating, while softmax-aggregator nodes provide global content-addressable routing.

### 4.4 Sample Quality

We ablate sample-time decoding: nucleus (top-p) sampling versus top-k, with a rolling-window repetition penalty (GPT-2-style: divide positive logits, multiply negative ones, for tokens appearing in the last `W` positions). Continuations from real long-context (512-token) prompts pulled from the validation set look like:

> *"...This stuff could have all been worked out diplomatically or legally before he got there." In addition* of " Jeamison 's mother in the UK , and I made the 18 Winchester , because many countries was " a digital �ationst and wrote occupy in the castle 's base at Totato Bay Woodbaucontsinzikasing ) , his work into Wirewood Jimmy ians to the Darkness " . = Plende on the Court Gnamitionkoh returned Alex in notch — the shower Luna and again s

Generated text exhibits Wikipedia-consistent surface structure: section-header conventions (`= Title =`), em-dashes (`@-@`, the WikiText-103 encoding of `-`), dates (`September 1882`), capitalised plausibly-Latinate entity names (`Spallyzie`, `Geral Woods Daboute`), bracketed parentheticals, and quoted dialogue. Local syntactic fluency is partial — short fragments parse, multi-clause sentences do not. Some fragments contain UTF-8 byte-level BPE artefacts (`�`) where the small 1024-vocab tokenizer failed to merge bytes into clean character tokens. The output quality is roughly what one expects at 2.5 nats/token: token-frequency learned, local bigram-trigram structure learned, sentence-level coherence not yet learned. We provide longer samples and a reproducible sampling script in the released code.

---

## 5. Analysis

### 5.1 Vestigial nodes

Without the reachable-to-outputs constraint, a non-trivial fraction of compute nodes have no path to any output node. Such nodes:

- Compute their forward activation but never contribute to logits.
- Receive zero gradient because `∂L/∂a_i = Σ_{k ∈ children(i)} w_{i→k} · ∂L/∂pre_k · σ'(pre_k)` recursively reduces to zero when no child is on an output path.
- Have all incident weights stuck at initialisation forever.

For our 1000-node, 5-level graph at edge probability 0.02, **70 % of nodes were vestigial** before the fix (Table 1). The output layer of size 1 with 7 expected parents made the L_max − 1 layer mostly dead; cascading backwards, most of the graph received no gradient. The "advertised" 6,924 parameters contained ~713 effective ones.

| layer | total nodes | vestigial | % vestigial |
|---:|---:|---:|---:|
| L0 (in + bias) | 4 | 0 | 0 |
| L1 | 332 | 97 | 29 |
| L2 | 332 | 278 | 84 |
| L3 | 331 | 324 | 98 |
| L4 (output) | 1 | 0 | 0 |

The fix is a constructive guarantee: in the same pass that draws random parents, we also enforce that every non-output node has at least one outgoing edge to some later layer. The constraint is local (per-node), preserves the layer label, and adds at most one edge per non-output node. Post-fix: 0 vestigial nodes by construction, and validation MSE on the canonical x₁² + x₂² regression task drops from 3.9·10⁻⁵ to 1.2·10⁻⁵.

### 5.2 Fan-in saturation

Each compute node samples `~ edge_prob × |earlier nodes|` parents. When this expected fan-in is large, every hidden node sees a near-uniform random sample of the available inputs, producing **statistically identical mixtures across hidden nodes** at the same level. The graph cannot specialise.

We observed this empirically. At ctx=1024, n_in=49,152, edge_prob=0.030: each first-hidden-layer node samples ~1,475 parents. Validation loss plateaued near 4.0 nats/token despite 100 + M parameters. Reducing edge_prob to 0.005 dropped expected fan-in to ~250, allowed first-hidden-layer specialisation, and cleared the plateau.

This effect has two practical consequences. First, naive scaling (more nodes, same edge_prob) is counter-productive: it preserves the uniform-mixture pathology while adding parameters that average together. Second, *sparse* random graphs at moderate width outperform *dense* ones at large width on equal parameter budget. Combined with appropriate LR adjustment (sparser → lower LR), this is the recipe behind the result in Table 1.

### 5.3 Out-of-distribution prompt collapse

When generating from a short prompt (e.g. `"The "`, ~1 BPE token), the model is fed a 512-element input vector consisting of 511 padding zeros followed by 1 real token. This input distribution is wildly out of the data manifold the model was trained on (where every position is a real token). A model trained to produce calibrated next-token distributions on real text has no obligation to generalise to such inputs. Empirically it does not: we observed near-deterministic distributions (96.5 % probability mass on a single token after the 1-token prompt), producing degenerate generations like `"The musmusmusmus..."`.

The fix is methodological rather than architectural: **prompt with realistic-length context** — either by feeding a real long-context excerpt as the prompt, or by autoregressively extending a short prompt with the model's own samples until the context is full. With full 512-token real-text prompts, we observed normal probability mass distributions (top-1 probability ~3 % to ~30 % depending on context predictability) and varied generations. We suggest this as a general check for graph-based LMs: out-of-distribution prompts may produce dramatically different sample quality than their loss numbers suggest.

### 5.4 Topology as design space

The mixed-kind framing makes architecture search in random DAGs a meaningfully larger problem than the homogeneous case. Beyond the connectivity choices (`edge_prob`, `n_layers`, `n_nodes`), we add four kind fractions plus the placement strategy (random, layer-stratified, position-stratified). Part 3 of Kyriacou (2020) uses multi-objective evolutionary architecture search with crossover and mutation over connectivity and node-kind annotations to discover Pareto fronts over (validation loss, edge count, branch-cost). Combined with our heterogeneous node-kind framework, this expands NAS to (loss, edges, multilinear-fraction, attention-fraction, cross-partition-edges) — five-objective Pareto search over a substrate that admits the topology-as-communication-graph correspondence we sketch in Section 6.

---

## 6. Discussion

### Why this works at all

A naive prediction would be that a randomly wired feed-forward DAG with mixed node kinds underperforms even a same-budget linear MLP, on the grounds that the connectivity is uninformed and the fraction of "useful" nodes is low. The empirical result is the opposite: in our 80 k-node, edge-prob-0.02 setting with 10 % softmax-aggregator nodes, the network learns a representation that places it inside the 2.40–3.20-nat band where similarly-sized transformers operate.

The mechanism is the combination of three properties:

1. **Topology equals communication graph.** The random DAG *is* the dataflow. There is no learned routing on top of a dense substrate. Edges that exist are computed; edges that don't aren't. Sparsity isn't bolted on after a dense pretrain — it's the model's primitive.

2. **Heterogeneous primitives compose.** Softmax-aggregator nodes provide content-addressable routing of upstream activations; bilinear nodes provide multiplicative gating of those routed signals; linear nodes accumulate the result. A short chain `linear → softmax-agg → bilinear → linear` realises an attention-conditioned gated linear unit, the structural unit modern transformer FFN-with-attention combines into.

3. **Position is encoded by layout, not by signature.** Each (token-position, embedding-dim) pair occupies a fixed slot in the input vector; the random connectivity samples that slot statistics. Sinusoidal positional encoding adds a per-position deterministic signature so identical tokens at different positions remain distinguishable. The architecture has no permutation invariance to break.

### Why parallel mirrors underperformed mixed-kind at half the parameters

The parallel-mirror configuration runs a 57.5 M-edge linear graph and a 57.5 M-edge bilinear graph in parallel, summing at logits. The mixed-kind configuration runs a single 53 M-edge graph with linear + bilinear + softmax-agg nodes scattered. The latter wins despite half the parameters.

We hypothesise the cause is **compositional capacity**. In parallel mirrors, the bilinear branch only operates on raw inputs (token embeddings + positional offsets); it can construct pairwise interactions between input features but cannot route those interactions through linear transformations or attention. In the mixed-kind graph, linear, bilinear, and softmax-aggregator nodes are interleaved at every level: each kind operates on the outputs of all earlier-level kinds. Compositional depth is the architectural feature transformers exploit by stacking attention + FFN blocks; the mixed-kind RWNN gives the same property as a graph-level emergent behaviour.

This is testable: a stacked-block alternation architecture (linear → bilinear → softmax-agg → linear, with residual connections between blocks) should further improve over the single-graph mixed-kind result. We sketch this design in our project's `next_research.md` and leave its evaluation to future work.

### Limitations

We report results on a single dataset (WikiText-103), a single random seed (0), a single GPU (RTX 4090). Multi-seed and multi-dataset replication is the obvious next step; we expect our reported delta over baselines to remain qualitatively stable but the absolute numbers to vary by ≈ 0.05 nats. Extending to OpenWebText or FineWeb-Edu at full scale is gated by the (modest) Triton kernel performance — at 100 M tokens our pipeline is faster than transformer training on the same hardware; at 9 B tokens the comparison would need to be re-measured.

We do not present a head-to-head comparison with a transformer language model of identical parameter count trained on the same data with the same recipe. This is partly because transformer training at 53 M parameters on WikiText-103 is well-trodden ground (val ≈ 2.0–2.4 in published reports, depending on hyperparameters), and partly because our claim is not "RWNN-LM > transformer" but "RWNN-LM with heterogeneous node kinds is a viable substrate that lands inside the same val-loss neighbourhood, with structurally different inductive biases." A controlled head-to-head comparison would be worthwhile and is left to follow-up.

The softmax-aggregator nodes in our setting compute attention over their *parent set* rather than over a sequence dimension. The size of the parent set (typically 8–32 pairs) is much smaller than the sequence length transformers attend over. This is a fundamental architectural difference and a fundamental capacity limit: a single softmax-aggregator node cannot mix information across all 512 token positions simultaneously. The compensation is that *many* such nodes exist (~ 5,500 in our default config), and each attends over a different randomly sampled subset of upstream activations. The aggregate routing capacity is comparable; the **shape** of the routing is qualitatively different. We do not yet have a clear theoretical handle on how the sum of many small-attention nodes compares to one big-attention layer.

---

## 7. Conclusion

We've shown that randomly wired feed-forward DAGs are a viable substrate for language modeling, *if* their nodes are allowed to be heterogeneous. By introducing four node kinds (linear, bilinear, product, softmax-aggregator), each implementable as a one-branch dispatch in a small Triton kernel, we obtain a substrate in which content-addressable routing, multiplicative gating, and additive accumulation coexist at the same level of granularity as a single neuron. On WikiText-103 at 512-token context, a 53 M-parameter mixed-kind RWNN-LM achieves validation cross-entropy 2.48 nats per token — competitive with similarly-sized transformers and 0.37 nats below the linear-only and parallel-mirror baselines.

The result is incremental as a language-model performance number — transformers remain the high-water mark by a comfortable margin — but suggestive as a research direction. Our broader interest is in the *systems* implications: a model whose connectivity *is* the communication graph admits cheap graph-partitioning for distributed training, hierarchical edge probabilities matched to physical bandwidth tiers, and evolutionary architecture search over both topology and node kind. The preliminary results in this paper establish that the substrate is competitive on language modelling at small scale; the systems case is laid out in detail in the accompanying research memo (`research_pitch.md`).

Code, configurations, training scripts, sample outputs, and reproducible architecture metadata are released at <https://github.com/stelios-sphere/rwnns>.

---

## References

(numbered references; this list mirrors the standard NeurIPS/ICML format)

1. **Bahdanau et al., 2015.** Bahdanau D, Cho K, Bengio Y. *Neural machine translation by jointly learning to align and translate.* ICLR 2015.
2. **Beltagy et al., 2020.** Beltagy I, Peters ME, Cohan A. *Longformer: The long-document transformer.* arXiv:2004.05150, 2020.
3. **Bengio et al., 2003.** Bengio Y, Ducharme R, Vincent P, Janvin C. *A neural probabilistic language model.* Journal of Machine Learning Research, 3:1137–1155, 2003.
4. **Child et al., 2019.** Child R, Gray S, Radford A, Sutskever I. *Generating long sequences with sparse transformers.* arXiv:1904.10509, 2019.
5. **Dao et al., 2022.** Dao T, Fu DY, Ermon S, Rudra A, Ré C. *FlashAttention: Fast and memory-efficient exact attention with IO-awareness.* NeurIPS 2022.
6. **Dauphin et al., 2017.** Dauphin YN, Fan A, Auli M, Grangier D. *Language modeling with gated convolutional networks.* ICML 2017.
7. **Eldan and Li, 2023.** Eldan R, Li Y. *TinyStories: How small can language models be and still speak coherent English?* arXiv:2305.07759, 2023.
8. **Fedus et al., 2021.** Fedus W, Zoph B, Shazeer N. *Switch Transformer: Scaling to trillion parameter models with simple and efficient sparsity.* arXiv:2101.03961, 2021.
9. **Frankle and Carbin, 2018.** Frankle J, Carbin M. *The Lottery Ticket Hypothesis: Finding sparse, trainable neural networks.* ICLR 2019.
10. **Hoffmann et al., 2022.** Hoffmann J et al. *Training compute-optimal large language models.* NeurIPS 2022. (Chinchilla paper.)
11. **Jordan et al., 2020.** Jordan M, Crisanti M, Bonner S, Borgwardt K. *Dense prediction with random graphs.* arXiv:2006.16327, 2020.
12. **Kaplan et al., 2020.** Kaplan J et al. *Scaling laws for neural language models.* arXiv:2001.08361, 2020.
13. **Karpathy, 2022.** Karpathy A. *nanoGPT.* GitHub repository, 2022. <https://github.com/karpathy/nanoGPT>
14. **Karpathy, 2024.** Karpathy A. *llm.c: GPT-2 reproduction in raw C/CUDA.* GitHub repository, 2024.
15. **Kyriacou, 2020.** Kyriacou S. *On randomly wired and optimally wired feed-forward neural networks (Parts 1–3).* Medium, 2020. <https://medium.com/@stylianoskyriacou>
16. **Lee-Thorp et al., 2021.** Lee-Thorp J, Ainslie J, Eckstein I, Ontanon S. *FNet: Mixing tokens with Fourier transforms.* NAACL 2022.
17. **Merity et al., 2016.** Merity S, Xiong C, Bradbury J, Socher R. *Pointer sentinel mixture models.* (WikiText-103 release.) arXiv:1609.07843, 2016.
18. **Press et al., 2021.** Press O, Smith NA, Lewis M. *Train short, test long: Attention with linear biases enables input length extrapolation.* ICLR 2022. (ALiBi.)
19. **Radford et al., 2019.** Radford A, Wu J, Child R, Luan D, Amodei D, Sutskever I. *Language models are unsupervised multitask learners.* Technical report, OpenAI, 2019. (GPT-2.)
20. **Real et al., 2019.** Real E, Aggarwal A, Huang Y, Le QV. *Regularized evolution for image classifier architecture search.* AAAI 2019.
21. **Sennrich et al., 2016.** Sennrich R, Haddow B, Birch A. *Neural machine translation of rare words with subword units.* ACL 2016. (BPE.)
22. **Shazeer, 2020.** Shazeer N. *GLU variants improve transformer.* arXiv:2002.05202, 2020. (SwiGLU.)
23. **Shazeer et al., 2017.** Shazeer N et al. *Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.* ICLR 2017.
24. **Su et al., 2021.** Su J, Lu Y, Pan S, Wen B, Liu Y. *RoFormer: Enhanced transformer with rotary position embedding.* arXiv:2104.09864, 2021. (RoPE.)
25. **Tay et al., 2020.** Tay Y, Bahri D, Metzler D, Juan DC, Zhao Z, Zheng C. *Synthesizer: Rethinking self-attention in transformer models.* ICML 2021.
26. **Team Gemma, 2024.** Gemma Team. *Gemma: Open models based on Gemini research and technology.* Google, 2024. (GeGLU.)
27. **Tillet et al., 2019.** Tillet P, Kung HT, Cox D. *Triton: An intermediate language and compiler for tiled neural network computations.* MAPL 2019.
28. **Touvron et al., 2023.** Touvron H et al. *LLaMA: Open and efficient foundation language models.* arXiv:2302.13971, 2023.
29. **Vaswani et al., 2017.** Vaswani A et al. *Attention is all you need.* NeurIPS 2017.
30. **Wortsman et al., 2019.** Wortsman M, Farhadi A, Rastegari M. *Discovering neural wirings.* NeurIPS 2019.
31. **Xie et al., 2019.** Xie S, Kirillov A, Girshick R, He K. *Exploring randomly wired neural networks for image recognition.* ICCV 2019.

---

## Appendix A — Reproducibility

The complete training pipeline is in the released repository under `LLM/`. The following minimal command reproduces the headline result:

```bash
git clone https://github.com/stelios-sphere/rwnns.git
cd rwnns/LLM
bash data/download_data.sh                # ~515 MB WikiText-103
cd example
./run_weekend.sh                          # nohup-safe; resume on crash
```

`run.py`'s defaults match the configuration reported in this paper (mixed-kind, ctx=512, edge_prob=0.020, peak LR 1·10⁻⁴, batch 256, 1024-token vocab BPE, sinusoidal positional encoding). Training reaches val 2.48 in ~32 hours wall time on a single RTX 4090. Checkpoints are written atomically every 60 seconds; `best_model.pt` holds the lowest-val state. Inference / sampling: `python3 sample.py`.

## Appendix B — Architecture metadata (verbatim)

```json
{
  "config": {
    "vocab_size": 1024,
    "context_length": 512,
    "d_model": 48,
    "n_nodes": 80000,
    "n_layers": 8,
    "edge_prob": 0.020,
    "bilinear_fraction": 0.10,
    "product_fraction": 0.0,
    "attention_fraction": 0.10,
    "n_bias": 2,
    "seed": 0,
    "pos_encoding": "sinusoidal",
    "parallel": false
  },
  "parameters": {
    "token_embedding": 49152,
    "pos_embedding": 0,
    "rwnn": 53017973,
    "total": 53067125
  },
  "rwnn_graph": {
    "n_nodes": 80000,
    "n_edges_per_branch": 53017973,
    "n_branches": 1,
    "n_levels": 8,
    "nodes_per_level": [24578, 9070, 9070, 9070, 9070, 9069, 9069, 1024]
  },
  "training": {
    "batch_size": 256,
    "peak_lr": 1.0e-4,
    "min_lr": 1.0e-5,
    "warmup": 4000,
    "cosine_decay": 2000000,
    "eval_interval_seconds": 60.0,
    "eval_iters": 40,
    "grad_clip": 1.0,
    "seed": 0
  }
}
```

## Appendix C — Kernel-correctness verification

We verify each of the four Triton kernels against a vectorised pure-PyTorch reference implementation that iterates parents in the same CSR order and dispatches on node kind. Random graphs of 200 nodes and 8 levels with mixed kind fractions (33 % linear, 33 % bilinear, 33 % softmax-aggregator) are constructed at multiple seeds. Across all seeds, maximum absolute differences are:

- Forward output: ≤ 1.2 · 10⁻⁷ (float-32 round-off)
- Weight gradient: ≤ 5.96 · 10⁻⁸
- Input gradient: ≤ 2.24 · 10⁻⁸

These numbers are consistent with float-32 numerical precision; we conclude the kernels are correct.

The verification suite runs in ~10 seconds and is in `LLM/src/tests.py`.
