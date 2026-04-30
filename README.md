# Heterogeneous Random-Wired Neural Networks for Language Modeling

A research project on whether randomly wired feed-forward DAGs — with
*heterogeneous node kinds* (linear, bilinear-gating, product,
softmax-aggregator) — can serve as a substrate for language modeling.
Includes a complete training pipeline, a head-to-head comparison
against [nanoGPT](https://github.com/karpathy/nanoGPT) on matched
datasets and parameter budgets, and a written-up paper.

## Layout

```
.
├── LLM/                  RWNN language-model package + WikiText-103 example
│   ├── data/                ← download_data.sh fetches WikiText-103-raw
│   ├── src/                 ← rwnn package (graph, kernels, model), tokenizer, llm.py
│   └── example/             ← run.py: end-to-end training script
├── comparison/           Head-to-head comparison vs. nanoGPT
│   ├── nanogpt/             ← vendored copy of Karpathy's nanoGPT (model.py, train.py)
│   ├── data/                ← char-Shakespeare + WikiText prep
│   ├── run_compare.py       ← unified runner for either architecture
│   └── plot.py              ← loss-vs-step + loss-vs-wall comparison plots
└── publication/          Paper + supplementary
    ├── paper.md             ← full conference-style write-up
    └── README.md            ← reproducibility notes
```

## Quick reproductions

### Train the headline RWNN-LM on WikiText-103

```bash
cd LLM
bash data/download_data.sh
cd example
./run_weekend.sh                         # nohup-safe, resume on crash
```

Reaches val cross-entropy ~2.48 nats/token (53M parameters,
mixed-kind: 80% linear / 10% bilinear / 10% softmax-aggregator) in
~32 hours on a single RTX 4090.

### Run the head-to-head comparison vs. nanoGPT

```bash
cd comparison
python3 run_compare.py --config char_shakespeare --model nanogpt
python3 run_compare.py --config char_shakespeare --model rwnn
python3 plot.py                          # comparison plots
```

## Paper

`publication/paper.md` is the full write-up. Key claims:

- 53M-parameter mixed-kind RWNN-LM achieves val 2.48 on WikiText-103
  (nats/token) — inside the nanoGPT-equivalent good range.
- Heterogeneous node kinds substantially outperform homogeneous random
  graphs: ablations show bilinear and softmax-aggregator nodes
  contribute super-additively.
- Custom Triton kernels deliver 26-66 ms/step at 53-115 M parameters
  on a single 4090.
- Three failure modes characterised (vestigial nodes, fan-in
  saturation, OOD prompt collapse), each with a targeted fix.

## Branches

- `main` — early Part 1 RWNN regression work + initial LLM scaffolding.
- `rwnn-attention` — heterogeneous-kind LLM development branch.
- `publication` (this branch) — clean LLM + paper + comparison.

## License

Code: MIT. Paper: CC BY 4.0.
