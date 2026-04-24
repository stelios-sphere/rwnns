"""Byte-level Byte-Pair Encoding (BPE) — pure implementation.

Same algorithm family as GPT-2 / tiktoken: the initial vocabulary is the
256 possible bytes, then we greedily merge the most-frequent adjacent
pair of tokens until the target vocab size is reached. Decoding just
concatenates the byte-sequences of each token and UTF-8-decodes.

Pre-tokenisation (the step that splits text into "word-like" chunks so
BPE doesn't learn cross-word merges) uses a simple regex compatible
with the Python stdlib. It isn't byte-for-byte GPT-2 but it does the
same job for tinyshakespeare-scale corpora.

GPU acceleration, honestly
--------------------------
BPE encoding is inherently O(merges × text_length) per text with tight
sequential dependencies, so the encode-one-string path is CPU. What we
GPU-accelerate is everything around it:

* ``encode_corpus_to_gpu`` does the one-time pre-tokenisation of a full
  corpus and writes the result directly into a GPU ``torch.long`` tensor,
  so the training hot path never touches Python or the CPU.
* ``decode`` uses vectorised ``torch.Tensor.tolist()`` for the one batch
  copy back to host, then a single string join.
* Per-word caching during encoding turns 1.1 MB of tinyshakespeare into
  ~seconds rather than minutes.

If you ever need true GPU-parallel encoding (e.g. online data loading at
scale), swap ``_apply_merges`` for a Triton kernel that does the merge
scan in parallel across words; the interface below doesn't need to
change.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import torch


# Simple pre-tokeniser: runs of word chars, runs of punct, runs of whitespace.
# \w in Python 3 is Unicode-aware by default, so this works beyond ASCII too.
_PRE_TOKEN_RE = re.compile(r"\w+|[^\w\s]+|\s+", re.UNICODE)


def _pretokenize(text: str) -> list[str]:
    return _PRE_TOKEN_RE.findall(text)


def _apply_merges(
    tokens: list[int],
    merge_rank: dict[tuple[int, int], int],
    pair_to_new: dict[tuple[int, int], int],
) -> list[int]:
    """Repeatedly merge the highest-priority pair until no more merges apply."""
    # Greedy: at each step find the pair with the lowest rank (earliest trained)
    # and merge the first such pair in the sequence. Classic GPT-2 implementation.
    while len(tokens) >= 2:
        best_rank = None
        best_idx = -1
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            r = merge_rank.get(pair)
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best_idx = i
        if best_rank is None:
            break
        pair = (tokens[best_idx], tokens[best_idx + 1])
        tokens = tokens[:best_idx] + [pair_to_new[pair]] + tokens[best_idx + 2:]
    return tokens


@dataclass
class BPETokenizer:
    """Byte-level BPE tokenizer.

    After ``train``, the tokenizer can ``encode`` / ``decode`` text and
    serialise to / from JSON via ``save`` / ``load``.
    """

    vocab_size: int = 0
    # merges[k] = ((a, b), new_id) — in training priority order
    merges: list[tuple[tuple[int, int], int]] = field(default_factory=list)
    # token_id -> bytes
    vocab: dict[int, bytes] = field(default_factory=dict)
    # word (str) -> list[int] encoding cache
    _cache: dict[str, list[int]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int, *, verbose: bool = False) -> None:
        """Train BPE merges on ``text`` until the vocabulary has ``vocab_size`` tokens."""
        assert vocab_size >= 256, "vocab must include at least the 256 byte base"

        # Initial vocab: raw bytes.
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        merges: list[tuple[tuple[int, int], int]] = []

        # Pre-tokenize once; track each unique pre-token's byte sequence + count.
        pretokens = _pretokenize(text)
        counts = Counter(pretokens)
        # tokens[word] = current list of token ids for that word
        tokens: dict[str, list[int]] = {
            w: list(w.encode("utf-8")) for w in counts
        }

        # Global pair count table, kept in sync as merges are applied.
        pair_counts: Counter[tuple[int, int]] = Counter()
        for w, toks in tokens.items():
            freq = counts[w]
            for i in range(len(toks) - 1):
                pair_counts[(toks[i], toks[i + 1])] += freq

        target_merges = vocab_size - 256
        for step in range(target_merges):
            if not pair_counts:
                break
            # Deterministic tiebreak on (count, -a, -b) — prefers lower-id pairs
            # which means byte pairs over later composite pairs when ties occur.
            best_pair = max(pair_counts.items(),
                            key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            best_count = pair_counts[best_pair]
            if best_count <= 0:
                break

            new_id = 256 + len(merges)
            merges.append((best_pair, new_id))
            vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

            # Incrementally update pair_counts: only words containing best_pair change.
            a, b = best_pair
            changed_words = [w for w, toks in tokens.items()
                             if any(toks[i] == a and toks[i + 1] == b
                                    for i in range(len(toks) - 1))]
            for w in changed_words:
                freq = counts[w]
                old = tokens[w]
                # Subtract old pair contributions.
                for i in range(len(old) - 1):
                    pair_counts[(old[i], old[i + 1])] -= freq
                # Rewrite the word.
                new = []
                i = 0
                while i < len(old):
                    if i < len(old) - 1 and old[i] == a and old[i + 1] == b:
                        new.append(new_id)
                        i += 2
                    else:
                        new.append(old[i])
                        i += 1
                tokens[w] = new
                # Add new pair contributions.
                for i in range(len(new) - 1):
                    pair_counts[(new[i], new[i + 1])] += freq
            # Clean up zero entries.
            if best_pair in pair_counts:
                del pair_counts[best_pair]

            if verbose and (step + 1) % max(1, target_merges // 10) == 0:
                print(f"  merge {step + 1:4d}/{target_merges}  "
                      f"pair=({a:3d},{b:3d})  count={best_count}  "
                      f"token={vocab[new_id]!r}")

        self.merges = merges
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self._cache.clear()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _merge_rank(self) -> dict[tuple[int, int], int]:
        return {pair: rank for rank, (pair, _) in enumerate(self.merges)}

    def _pair_to_new(self) -> dict[tuple[int, int], int]:
        return {pair: new for pair, new in self.merges}

    def encode_word(self, word: str,
                    merge_rank: dict[tuple[int, int], int] | None = None,
                    pair_to_new: dict[tuple[int, int], int] | None = None) -> list[int]:
        if word in self._cache:
            return self._cache[word]
        if merge_rank is None:
            merge_rank = self._merge_rank()
        if pair_to_new is None:
            pair_to_new = self._pair_to_new()
        toks = list(word.encode("utf-8"))
        out = _apply_merges(toks, merge_rank, pair_to_new)
        self._cache[word] = out
        return out

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` to a list of token ids (host-side Python list)."""
        if not self.merges:
            # No merges trained: raw-byte fallback.
            return list(text.encode("utf-8"))
        merge_rank = self._merge_rank()
        pair_to_new = self._pair_to_new()
        out: list[int] = []
        for w in _pretokenize(text):
            out.extend(self.encode_word(w, merge_rank, pair_to_new))
        return out

    def encode_to_tensor(self, text: str,
                         device: str | torch.device = "cuda",
                         dtype: torch.dtype = torch.long) -> torch.Tensor:
        """Encode ``text`` and return the token ids as a ``[N]`` tensor on ``device``."""
        return torch.tensor(self.encode(text), dtype=dtype, device=device)

    def encode_corpus_to_gpu(self, text: str, *,
                             device: str | torch.device = "cuda",
                             dtype: torch.dtype = torch.long,
                             verbose: bool = False) -> torch.Tensor:
        """Encode a (potentially large) corpus and place it on the GPU.

        Uses the per-word cache so repeated words are encoded once.
        Tinyshakespeare encodes in a few seconds.
        """
        ids = self.encode(text)
        if verbose:
            ratio = len(ids) / max(1, len(text.encode("utf-8")))
            print(f"  corpus: {len(ids):,} tokens "
                  f"({ratio:.2f}× bytes; vocab={self.vocab_size})")
        return torch.tensor(ids, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: Iterable[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        buf = b"".join(self.vocab[int(i)] for i in ids)
        return buf.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        # Vocab maps ids (int) to bytes; JSON-friendly via hex.
        blob = {
            "vocab_size": self.vocab_size,
            "merges": [[list(p), n] for p, n in self.merges],
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
        }
        with open(path, "w") as f:
            json.dump(blob, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            blob = json.load(f)
        tok = cls()
        tok.vocab_size = int(blob["vocab_size"])
        tok.merges = [((int(p[0]), int(p[1])), int(n)) for p, n in blob["merges"]]
        tok.vocab = {int(k): bytes.fromhex(v) for k, v in blob["vocab"].items()}
        return tok
