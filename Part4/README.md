# Part 4 — RWNN as an LLM (placeholder)

> **Status:** scoping only. No implementation yet. We are deliberately
> taking this one step at a time.

## Goal

Extend the RWNN idea from Parts 1–3 to a **language model** setting:
instead of fitting a small regression function, use a randomly wired
feed-forward network as the core computation block of an LLM-style
model, and eventually evolve its wiring the way Part 3 evolves topology
for regression.

This file is a parking spot for that direction. It exists so we have a
place to collect notes, open questions, and design choices before any
code is written.

## Open questions (to resolve before we build anything)

- **Scope of "LLM".** Character-level model on a tiny corpus? Word-level?
  BPE tokens? Pretraining vs. fine-tuning? A toy next-token model is
  probably the honest starting point, matching the spirit of Parts 1–3.
- **Where does the RWNN live?** Options:
  - Replace the MLP/feed-forward block inside a standard transformer
    layer with an RWNN block.
  - Drop attention entirely and use a deep stack of RWNN blocks over
    token embeddings (closer to the spirit of Parts 1–3, further from a
    conventional LLM).
  - Use an RWNN as the read-out head on top of a frozen backbone.
- **Sequence handling.** RWNNs as defined in Parts 1–3 are feed-forward
  and pointwise. Language needs a way to mix information across
  positions. Candidates: causal attention kept as-is, a
  convolution/mixer layer, or extending the DAG constraint to span
  positions as well as features.
- **Training signal.** Cross-entropy on next-token prediction is the
  obvious loss. Nothing exotic needed at first.
- **Scale.** Parts 1–3 had ~10 hidden nodes. An LLM needs *thousands*
  of parameters per block, many blocks, and many tokens of context.
  The DAG constraint and the evolutionary search from Part 3 do not
  obviously scale — this will be a real engineering question, not a
  toy one.
- **Evolution vs. fixed random wiring.** Do we start with a single
  random topology (Part 1 style), an ensemble (Part 2 style), or an
  evolved topology (Part 3 style)? Part 1 style is the right first
  step: prove the block can learn language at all.

## Things we are explicitly NOT doing yet

- Writing training code.
- Choosing a dataset.
- Picking a tokenizer.
- Benchmarking against a baseline transformer.

We will decide each of these together, in order, once the scoping above
is settled.

## Next step

Agree on the smallest viable experiment: most likely a
character-level next-token model on a tiny text corpus, with a single
randomly wired feed-forward block replacing a standard MLP, and no
attention at first. Once the scope is locked, we write the minimal
training loop and see whether the network can learn anything at all —
mirroring the spirit of Part 1.
