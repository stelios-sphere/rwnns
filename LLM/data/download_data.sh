#!/bin/bash
# Download datasets used by LLM/example/run.py.
# Files are too big for git — they live on HuggingFace.
#
# Default: WikiText-103 raw (long-form Wikipedia, suits ctx=1024).
# Optional: TinyStories (kid stories, suits ctx <= 256).
set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# WikiText-103 (Salesforce). ~515 MB train + ~1 MB val. Long-form articles
# that comfortably fill a 1024-token context. The HuggingFace mirror ships
# parquet shards; we concatenate them into plain text.
# ---------------------------------------------------------------------------
if [ ! -f WikiText-103-train.txt ] || [ ! -f WikiText-103-valid.txt ]; then
    echo "fetching WikiText-103 parquet shards..."
    mkdir -p .wikitext_tmp
    for f in train-00000-of-00002 train-00001-of-00002 validation-00000-of-00001; do
        if [ -f ".wikitext_tmp/${f}.parquet" ]; then continue; fi
        curl -L --progress-bar -o ".wikitext_tmp/${f}.parquet" \
            "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/${f}.parquet"
    done
    echo "decoding parquet -> txt..."
    python3 - <<'PY'
import pyarrow.parquet as pq
def cat(in_paths, out_path):
    with open(out_path, "w") as f:
        for p in in_paths:
            for s in pq.read_table(p, columns=["text"]).column("text").to_pylist():
                f.write(s)
cat([".wikitext_tmp/train-00000-of-00002.parquet",
     ".wikitext_tmp/train-00001-of-00002.parquet"],
    "WikiText-103-train.txt")
cat([".wikitext_tmp/validation-00000-of-00001.parquet"],
    "WikiText-103-valid.txt")
PY
    rm -rf .wikitext_tmp
fi

# ---------------------------------------------------------------------------
# TinyStories (Microsoft, 2023). ~1.9 GB train + ~19 MB val. Short kid
# stories. Comment in if you want to switch run.py back to TinyStories.
# ---------------------------------------------------------------------------
# for f in TinyStories-train.txt TinyStories-valid.txt; do
#     [ -f "$f" ] && continue
#     echo "fetching $f..."
#     curl -L --progress-bar -o "$f" \
#         "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/$f"
# done

echo "done."
