"""
a small, dependency-free byte-pair-encoding (BPE) tokenizer.

this replaces the word-survival token process. it learns a vocabulary directly
from the corpus by repeatedly merging the most frequent adjacent symbol pair -
the standard BPE algorithm - so common words collapse into one or two tokens
instead of many fragments.

spacing is handled SentencePiece-style: a leading space is folded into the token
via the marker '▁', so the encode/decode round-trip is lossless and there is no
need for a separate space token.

train once and save the result to JSON (the merges + vocabulary). both training
(training.py) and inference (chat.py) load that same file so they tokenise
identically.
"""

import heapq
import json
import os
import re
import time

from read_paragraphs import read_paragraphs


#marker that stands in for a leading space, so word boundaries survive encoding
SPACE = "▁"  # ▁

#special tokens occupy the first, fixed IDs
PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

#precompiled once: a word is the SPACE marker plus the run of non-marker chars
_WORD_RE = re.compile(SPACE + "[^" + SPACE + "]+")


def _pretokenize(text):
    """
    split text into words, each carrying its leading space as the SPACE marker.

    'the cat' -> ['▁the', '▁cat']. whitespace runs are collapsed to one space,
    which is fine for training a language model on prose.
    """

    text = " ".join(text.split())
    if not text:
        return []

    #every word boundary becomes a SPACE marker, then split on it (keeping it
    #attached to the front of each word)
    marked = SPACE + text.replace(" ", SPACE)
    return _WORD_RE.findall(marked)


class BPETokenizer:
    """
    holds the learned vocabulary + merge rules and does encode/decode.
    """

    def __init__(self, id_to_token, merges):
        #id_to_token[i] is the string for token id i (specials included)
        self.id_to_token = list(id_to_token)
        self.token_to_id = {tok: i for i, tok in enumerate(self.id_to_token)}

        #merges in learned order; the rank is the priority (lower = merge first)
        self.merges = [tuple(m) for m in merges]
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]

        #per-word encode cache: the corpus repeats words constantly, so this
        #turns most encode work into a dict lookup
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.id_to_token)

    #--- encoding -------------------------------------------------------------

    def _encode_word(self, word):
        """apply the learned merges to one word, returning a list of token strings."""
        cached = self._cache.get(word)
        if cached is not None:
            return cached

        symbols = list(word)
        #greedily apply the highest-priority applicable merge until none remain
        while len(symbols) >= 2:
            best_rank = None
            best_i = None
            for i in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i is None:
                break
            symbols[best_i: best_i + 2] = [symbols[best_i] + symbols[best_i + 1]]

        self._cache[word] = symbols
        return symbols

    def encode(self, text, add_eos=True):
        """convert text to token IDs (appending EOS like the old tokenizer did)."""
        ids = []
        for word in _pretokenize(text):
            for symbol in self._encode_word(word):
                ids.append(self.token_to_id.get(symbol, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    #--- decoding -------------------------------------------------------------

    def decode(self, token_ids):
        """convert token IDs back to text (lossless apart from collapsed whitespace)."""
        pieces = []
        for token_id in token_ids:
            token_id = int(token_id)
            #drop special markers entirely
            if token_id in (self.pad_id, self.eos_id, self.unk_id):
                continue
            pieces.append(self.id_to_token[token_id])
        return "".join(pieces).replace(SPACE, " ").strip()

    #--- persistence ----------------------------------------------------------

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "id_to_token": self.id_to_token,
                    "merges": [list(m) for m in self.merges],
                },
                f,
                ensure_ascii=False,
            )
        return path

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(id_to_token=data["id_to_token"], merges=data["merges"])


def train_from_paragraphs(paragraphs, vocab_size, min_frequency=2):
    """
    learn a BPE vocabulary of (about) vocab_size tokens from an iterable of
    paragraph strings.

    works on unique word *types* weighted by frequency (not the raw token
    stream), and updates pair statistics incrementally with a lazy heap, so it
    scales to large corpora without rescanning everything each merge.

    min_frequency drops word types rarer than this from the merge search. the
    base alphabet is still taken from every word, so coverage is unaffected -
    this only skips the long tail of once-seen words that barely influence the
    merges but dominate the type count, which is a big speedup.
    """

    #1) count how often each pre-tokenised word appears
    t0 = time.time()
    word_counts = {}
    for paragraph in paragraphs:
        for word in _pretokenize(paragraph):
            word_counts[word] = word_counts.get(word, 0) + 1

    if not word_counts:
        raise ValueError("no words found to train a tokenizer on; check textsource")

    #the base alphabet comes from ALL words so every character stays coverable
    alphabet = set()
    for word in word_counts:
        alphabet.update(word)

    print(
        f"  counted {len(word_counts)} word types in {time.time() - t0:.1f}s",
        flush=True,
    )

    #2) start every (frequent-enough) word as a list of single characters
    t1 = time.time()
    word_syms = []
    word_freq = []
    for word, count in word_counts.items():
        if count < min_frequency:
            continue
        word_syms.append(list(word))
        word_freq.append(count)

    print(
        f"  merging over {len(word_syms)} word types "
        f"(min_frequency={min_frequency})",
        flush=True,
    )

    #3) initial adjacent-pair statistics
    pair_counts = {}
    pair_words = {}
    for wi, syms in enumerate(word_syms):
        freq = word_freq[wi]
        for a, b in zip(syms, syms[1:]):
            pair = (a, b)
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            pair_words.setdefault(pair, set()).add(wi)

    #lazy max-heap: entries can go stale, so we re-check the count on pop
    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    #4) how many merges we can afford given the special tokens + base alphabet
    n_specials = 3  # PAD, EOS, UNK
    num_merges = max(0, vocab_size - n_specials - len(alphabet))

    t2 = time.time()
    merges = []
    while len(merges) < num_merges:
        #pop the most frequent still-valid pair
        best = None
        while heap:
            neg_count, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) == -neg_count and -neg_count > 0:
                best = pair
                break
        if best is None:
            break

        merges.append(best)
        new_token = best[0] + best[1]
        touched = set()

        #apply the merge inside every word that contains it
        for wi in list(pair_words.get(best, ())):
            syms = word_syms[wi]
            freq = word_freq[wi]

            #remove this word's current pair contributions. discarding the word
            #from each pair's membership keeps those sets from filling up with
            #stale entries, so future merges only visit words that really contain
            #the pair (the main speedup).
            for a, b in zip(syms, syms[1:]):
                pair = (a, b)
                pair_counts[pair] -= freq
                pair_words[pair].discard(wi)
                touched.add(pair)

            #merge adjacent occurrences of best
            merged = []
            i = 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == best[0] and syms[i + 1] == best[1]:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(syms[i])
                    i += 1
            word_syms[wi] = merged

            #add the word's new pair contributions
            for a, b in zip(merged, merged[1:]):
                pair = (a, b)
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                pair_words.setdefault(pair, set()).add(wi)
                touched.add(pair)

        #the merged pair is gone now
        pair_counts.pop(best, None)
        pair_words.pop(best, None)
        touched.discard(best)

        #push updated counts for everything we changed (stale entries get
        #filtered out on pop)
        for pair in touched:
            count = pair_counts.get(pair, 0)
            if count > 0:
                heapq.heappush(heap, (-count, pair))

    print(
        f"  learned {len(merges)} merges in {time.time() - t2:.1f}s",
        flush=True,
    )

    #5) assemble the final id<->token tables
    id_to_token = [PAD_TOKEN, EOS_TOKEN, UNK_TOKEN]
    seen = set(id_to_token)
    for ch in sorted(alphabet):
        if ch not in seen:
            id_to_token.append(ch)
            seen.add(ch)
    for a, b in merges:
        token = a + b
        if token not in seen:
            id_to_token.append(token)
            seen.add(token)

    return BPETokenizer(id_to_token=id_to_token, merges=merges)


def train(textsource, vocab_size, min_frequency=2):
    """learn a tokenizer straight from the corpus on disk."""
    return train_from_paragraphs(read_paragraphs(textsource), vocab_size, min_frequency)
