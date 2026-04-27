#!/bin/bash
# Download datasets used by LLM/example/run.py.
# Files are too big for git — they live on HuggingFace.
set -euo pipefail
cd "$(dirname "$0")"

# Microsoft TinyStories (Eldan & Li, 2023). ~2.6 GB train + ~19 MB val.
for f in TinyStories-train.txt TinyStories-valid.txt; do
    if [ -f "$f" ]; then
        echo "$f already present, skipping"
        continue
    fi
    echo "fetching $f ..."
    curl -L --progress-bar -o "$f" \
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/$f"
done
echo "done."
