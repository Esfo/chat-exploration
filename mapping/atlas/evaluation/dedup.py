"""Exact and near-duplicate grouping for training rows.

Duplicated and near-duplicated examples are a primary DOWNSAMPLE / DROP signal:
they waste training budget and over-represent whatever they say. We compute two
groupings, both pure-text and torch-free:

  * exact: a hash of the normalised (whitespace/case-folded) user+assistant text.
  * near:  a cheap MinHash over word-shingles, banded into a single bucket so two
    rows that share most of their shingles land in the same near-duplicate group.

Each row learns the size of its exact and near-duplicate groups; a size > 1 means
it has at least one (near-)twin elsewhere in the dataset.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)

#Fixed odd multipliers for a tiny, dependency-free MinHash family.
_MINHASH_SALTS = tuple(range(1, 17))
_MASK = (1 << 61) - 1  # a Mersenne prime modulus keeps the hash well-distributed


def _normalize_text(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def exact_hash(text: str) -> str:
    return hashlib.sha1(_normalize_text(text).encode("utf-8")).hexdigest()


def _shingles(text: str, k: int = 4) -> set[int]:
    words = _WORD.findall(text.lower())
    if len(words) < k:
        grams = [" ".join(words)] if words else []
    else:
        grams = [" ".join(words[i:i + k]) for i in range(len(words) - k + 1)]
    return {int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
            for g in grams}


def minhash_signature(text: str) -> tuple[int, ...]:
    sh = _shingles(text)
    if not sh:
        return tuple(0 for _ in _MINHASH_SALTS)
    sig = []
    for salt in _MINHASH_SALTS:
        sig.append(min(((h * (2 * salt + 1)) & _MASK) for h in sh))
    return tuple(sig)


def near_bucket(text: str) -> str:
    """A single banded MinHash bucket id; near-identical texts share a bucket."""
    sig = minhash_signature(text)
    return hashlib.sha1(",".join(map(str, sig)).encode()).hexdigest()


def group_sizes(texts: Iterable[str]):
    """Return ``(exact_sizes, near_sizes)`` parallel lists for ``texts``.

    Element ``i`` is the number of rows sharing row ``i``'s exact / near bucket.
    """
    texts = list(texts)
    exact_keys = [exact_hash(t) for t in texts]
    near_keys = [near_bucket(t) for t in texts]
    exact_counts: dict[str, int] = {}
    near_counts: dict[str, int] = {}
    for k in exact_keys:
        exact_counts[k] = exact_counts.get(k, 0) + 1
    for k in near_keys:
        near_counts[k] = near_counts.get(k, 0) + 1
    exact_sizes = [exact_counts[k] for k in exact_keys]
    near_sizes = [near_counts[k] for k in near_keys]
    return exact_sizes, near_sizes, exact_keys, near_keys
