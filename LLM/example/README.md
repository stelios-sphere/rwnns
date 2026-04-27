# `LLM/example/` — TinyStories training run

End-to-end training of the RWNN-LLM on Microsoft's TinyStories corpus.

## Architecture (no projection layers, with positional embeddings)

```
ids   [B, T]                                 T = 128 BPE tokens
  │   token_emb (V × d_model)  +  pos_emb (T × d_model)
  ▼
emb   [B, T, d_model]                        d_model = 48
  │   flatten
  ▼
flat  [B, T·d_model]                         RWNN input  n_in  = 6,144
  │   RWNN (random DAG, tanh hidden, linear output)
  ▼
logits[B, V]                                 RWNN output n_out = 1,024
```

Every (token-position, embedding-dim) pair is its own input node;
every vocab entry is its own output node. **No `in_proj` / `out_proj`
linear layers** — the embedding feeds the random graph directly, and
the random graph's outputs are the logits.

A learnable per-position embedding is added to the token embedding
before flattening, so each (position, dim) input node carries a unique
positional signature in addition to its token's content. Without this,
two identical tokens at different positions produced identical
embeddings, and the only positional signal the RWNN had was *which*
input nodes a value lands at — easy to wash out under random
connectivity.

Default config (`run.py`):

| | |
|---|---:|
| `vocab_size` | 1024 (BPE) |
| `context_length` | 128 |
| `d_model` | 48 |
| `n_nodes` | 45,000 |
| `n_layers` | 8 |
| `edge_prob` | 0.075 |
| `bilinear_fraction` | 0.05 |
| total params | ~65.6 M (100 % in RWNN modulo embedding) |

## Files

| file | what it is |
|---|---|
| `run.py` | The training script. Builds the tokenizer, encodes the corpus, builds and trains the model, samples mid-training and at end. |
| `run_weekend.sh` | Shell wrapper that relaunches `run.py` on any non-zero exit (resume-safe via the atomic `latest_model.pt` checkpoint). |
| `tokenizer.json` | Trained BPE tokenizer (cached after the first run; reused on resume). |
| `train_ids.pt` / `val_ids.pt` | Pre-tokenised corpus on GPU (cached). Together ~7 GB. |
| `architecture.json` | Snapshot of the model config + RWNN graph stats, written once at run start. |
| `latest_model.pt` | Most recent checkpoint, atomic-saved every eval (~60 s). For crash resume. |
| `best_model.pt` | Snapshot at the lowest validation loss seen so far. Used for end-of-run sampling. |
| `loss_curve.png` | Train / val loss vs. step, regenerated every 10 evals. |
| `sample.txt` | Sample text generated from `best_model.pt` at the end of the run. |
| `train.log` | Append-only log: setup, eval lines, mid-training samples. |

## Running

First, fetch the dataset (one-time, ~2 GB):

```bash
bash ../data/download_data.sh
```

Then run training:

```bash
# Foreground: prints to terminal, dies with the shell
python3 run.py

# Background, resume-safe, survives session end
nohup ./run_weekend.sh > /dev/null 2>&1 &
disown
tail -f train.log     # to monitor
```

## What `run.py` does, top to bottom

1. **Train BPE.** Reads the train corpus, BPE-trains on the first 80 MB
   for speed, saves to `tokenizer.json`. Cached on resume.
2. **Encode the corpus.** Tokenizes the full train and val corpora using
   the trained BPE; writes them to GPU as `train_ids.pt` and `val_ids.pt`.
   Cached on resume.
3. **Build the model.** `RWNNLM` with the config above — RWNN core with
   45 k nodes / ~65 M edges / 8 levels, 5 % bilinear gating nodes, and
   a single 1024 × 48 token embedding outside the graph.
4. **Save architecture.** `architecture.json` snapshots the config,
   parameter counts and per-level node counts.
5. **Resume-or-fresh.** Tries to load `latest_model.pt` first, then
   `best_model.pt`, falling back to fresh init.
6. **Train.**
   - AdamW, cosine LR schedule with warmup (peak 5e-4, min 1e-5).
   - Every ~60 s: estimate train + val cross-entropy on 40 batches each,
     print one log line, save `latest_model.pt`, save `best_model.pt`
     if val improved.
   - Every 100 steps: generate a 60-token sample from the prompt
     `"Once upon a time"` so you can watch the model learn in real time.
   - NaN/Inf guards on both train and val loss.
   - On Ctrl-C or any exception: save `latest_model.pt` before exiting
     so a relaunch resumes mid-flight.
7. **Plot the loss curve and final sample.** When training ends cleanly
   (or on `KeyboardInterrupt`), `loss_curve.png` is written and a 300-token
   sample is generated from the best checkpoint.

## Reading the log

Eval lines look like:

```
step  9354  lr 5.00e-04  train 2.91  val 2.99  best 2.86@8906  wall 0.30h
```

- `step` — total training steps (resume-aware).
- `lr` — current learning rate (cosine after warmup).
- `train` / `val` — cross-entropy in nats per token, averaged over
  `EVAL_ITERS` (40) batches at 60-second wall intervals.
- `best` — lowest val loss seen so far, with the step it was achieved.
- `wall` — accumulated wall-clock training time in hours.
- A trailing `*` marks an eval where val improved → `best_model.pt` was
  rewritten.

For BPE vocab 1024 the random baseline is `ln 1024 = 6.93 nats/token`.
A nanoGPT-equivalent "good" range (extrapolated from char-level
results) is roughly **2.40 – 3.20 nats/token** in our units.

Mid-training samples look like:

```
[step  6800] sample: Once upon a time, there was. The day, Lily ons they who a loty. One day, Lily reiy sounded to have. like they wereed the winra. One
```

## Resilience

- `run.py` saves `latest_model.pt` atomically every eval. A crash mid-write
  cannot corrupt it (writes to `.tmp`, renames).
- `run_weekend.sh` relaunches `run.py` after any non-zero exit, with an
  anti-thrash policy (3 fast failures in a minute → bail).
- `nohup` + `disown` make the run survive losing the shell or even the
  SSH session.

## Known limits

- The architecture has no attention. This is a pure feed-forward DAG
  with positional encoding via concatenation. Realistic loss floor on
  TinyStories is in the **2.0 – 2.4** range; coherent multi-sentence
  output is unlikely. See `../../research_pitch.md` for context.
- BPE is small (1024 vocab) so word interiors will sometimes break.
  Reasonable for a small model; bumping vocab to 4096 or 8192 would
  help if you scale up.
