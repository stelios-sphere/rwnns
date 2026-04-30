# `publication/` — paper

`paper.md` is a top-level-conference-style write-up of the work in this
repository, covering:

- the four heterogeneous node kinds (linear, bilinear-gating, product,
  softmax-aggregator) we introduce,
- the custom Triton GPU kernels that implement them,
- the WikiText-103 training results (best val 2.4837 at 53 M params),
- analysis of three failure modes (vestigial nodes, fan-in saturation,
  out-of-distribution prompt collapse),
- a complete bibliography (31 references covering randomly wired
  networks, sparse models, gating units, attention, BPE, scaling laws,
  Triton, and the modern small-LM toolchain).

## Layout

| file | what it is |
|---|---|
| `paper.md` | the full write-up — abstract, 7 sections, references, 3 appendices |

The companion documents `research_pitch.md`, `next_research.md` and
`ideas_for_future_research.md` at the repository root contain longer
forms of arguments referenced by the paper.

## Conversion to LaTeX

Markdown was used so the paper sits naturally next to the other repo
docs and tracks well in `git diff`. For conference submission the
straightforward path is:

```bash
pandoc publication/paper.md -o paper.pdf \
    --bibliography=publication/refs.bib \
    --citeproc \
    --pdf-engine=xelatex \
    --template=neurips_2024.tex
```

(`refs.bib` is not yet provided — the references in `paper.md` are
formatted as a numbered list inline. A bibtex export is mechanical and
left for whoever takes the paper through formal submission.)

## Reproducing the headline result

See `paper.md` Appendix A. In short, from the repository root:

```bash
cd LLM
bash data/download_data.sh
cd example
./run_weekend.sh
```

Best-val checkpoint is written atomically as `best_model.pt`; the run
is resume-safe on crash and survives shell exit via `nohup`. Sample
text from a trained checkpoint via `python3 sample.py`.
