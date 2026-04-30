"""Sample text from the trained best_model.pt with smarter decoding.

Key trick: this model was trained on FULL 512-token windows of WikiText.
Prompting it with a short string like "The " puts the input vector at
[0, 0, ..., 0, "The"] (511 padding zeros + 1 token), which is wildly
out-of-distribution. The model never saw such inputs during training,
so it falls back on its strongest token-level associations and
collapses to ~93% probability on a single token.

Fix: prompt with REAL long-context windows from the validation set,
then continue generating. This keeps the model in-distribution.
"""

from __future__ import annotations

import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, SRC)

from llm import RWNNLM, RWNNLMConfig         # noqa: E402
from tokenizer import BPETokenizer           # noqa: E402


def main():
    device = "cuda"

    tok = BPETokenizer.load(os.path.join(HERE, "tokenizer.json"))
    ckpt_path = os.path.join(HERE, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = RWNNLMConfig(**ckpt["config"])
    model = RWNNLM(cfg, device=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded best_model.pt  step={ckpt['step']}  "
          f"val_loss={ckpt['val_loss']:.4f}")

    # Take a few real-text prompts from val.
    val_ids = torch.load(os.path.join(HERE, "val_ids.pt"),
                         map_location=device, weights_only=True)
    rng = random.Random(0)
    n_prompts = 4
    prompt_len = cfg.context_length          # full context = no padding
    new_tokens = 200

    settings = [
        ("nucleus T=0.9 top_p=0.95 + reppen 1.3",
         dict(temperature=0.9, top_p=0.95,
              repetition_penalty=1.3, rep_penalty_window=64)),
        ("nucleus T=1.1 top_p=0.92 + reppen 1.4",
         dict(temperature=1.1, top_p=0.92,
              repetition_penalty=1.4, rep_penalty_window=64)),
    ]

    for k in range(n_prompts):
        start = rng.randint(0, val_ids.numel() - prompt_len - new_tokens)
        prompt_ids = val_ids[start : start + prompt_len]
        prompt_text = tok.decode(prompt_ids)
        # Show the tail of the prompt so the continuation is locally readable.
        print()
        print("=" * 78)
        print(f"PROMPT TAIL (~last 300 chars of {prompt_len}-token real-text window):")
        print("..." + prompt_text[-300:])
        print("=" * 78)
        for label, kwargs in settings:
            torch.manual_seed(7 + k)
            out = model.generate(prompt_ids, max_new_tokens=new_tokens, **kwargs)
            cont = tok.decode(out[0, prompt_ids.numel():])
            print(f"\n--- {label} ---")
            print(cont)


if __name__ == "__main__":
    main()
