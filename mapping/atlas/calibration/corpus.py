"""Procedural calibration corpus.

The plan is explicit that the right unit of data is the *token position*, not the
prompt, and that a compact but structurally diverse corpus is enough to make the
model visit varied internal regimes. The regime labels here ("prose", "code",
"brackets", ...) are *input* categories chosen to spread internal states; they
are not semantic labels for the final analysis.

This generator is fully procedural and seeded, so a run is reproducible and
needs no external dataset download. It produces text snippets across regimes; the
extractor tokenizes them to reach the configured ``target_tokens`` budget.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass


REGIMES = [
    "prose", "code", "markup", "lists", "dialogue", "numbers", "dates",
    "tables", "repeated", "brackets", "punctuation", "names", "random",
    "corrupted", "multilingual",
]


@dataclass
class CorpusItem:
    corpus_item_id: int
    regime: str
    text: str


_WORDS = (
    "system model vector signal residual stream layer attention head neuron "
    "cluster activation weight gradient token sequence probe direction matrix "
    "norm sparse dense graph edge source sink relay routing summary histogram "
    "library index manifest catalog pipeline checkpoint tensor analysis"
).split()

_NAMES = ["Alice", "Bjorn", "Chen", "Dara", "Eitan", "Fatima", "Goro", "Hana",
          "Ivan", "Jun", "Kira", "Liam", "Mei", "Noor", "Omar", "Priya"]

_MULTILINGUAL = [
    "le modèle apprend des représentations internes",
    "das Modell verarbeitet versteckte Zustände",
    "el modelo genera vectores residuales",
    "モデルは内部状態を処理します",
    "модель обрабатывает скрытые состояния",
]


def _prose(rng: random.Random, n: int) -> str:
    sent = []
    for _ in range(n):
        words = rng.choices(_WORDS, k=rng.randint(6, 16))
        words[0] = words[0].capitalize()
        sent.append(" ".join(words) + ".")
    return " ".join(sent)


def _code(rng: random.Random, n: int) -> str:
    lines = []
    for _ in range(n):
        var = rng.choice(_WORDS)
        lines.append(f"def {var}_{rng.randint(0, 99)}(x, y=None):")
        lines.append(f"    return [{var}(i) for i in range({rng.randint(1, 64)}) if i % 2 == 0]")
    return "\n".join(lines)


def _markup(rng: random.Random, n: int) -> str:
    tags = ["div", "span", "section", "p", "li", "code", "em"]
    out = []
    for _ in range(n):
        t = rng.choice(tags)
        out.append(f"<{t} class=\"{rng.choice(_WORDS)}\">{_prose(rng, 1)}</{t}>")
    return "\n".join(out)


def _lists(rng: random.Random, n: int) -> str:
    return "\n".join(f"{i+1}. {rng.choice(_WORDS)} {rng.choice(_WORDS)}" for i in range(n * 4))


def _dialogue(rng: random.Random, n: int) -> str:
    out = []
    for _ in range(n * 2):
        spk = rng.choice(_NAMES)
        out.append(f"{spk}: {_prose(rng, 1)}")
    return "\n".join(out)


def _numbers(rng: random.Random, n: int) -> str:
    return " ".join(str(rng.randint(-10**6, 10**6)) for _ in range(n * 12))


def _dates(rng: random.Random, n: int) -> str:
    return " ".join(
        f"{rng.randint(1900,2099)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        for _ in range(n * 8)
    )


def _tables(rng: random.Random, n: int) -> str:
    rows = ["| " + " | ".join(rng.choice(_WORDS) for _ in range(4)) + " |" for _ in range(n * 4)]
    return "\n".join(rows)


def _repeated(rng: random.Random, n: int) -> str:
    tok = rng.choice(_WORDS)
    return (tok + " ") * (n * 20)


def _brackets(rng: random.Random, n: int) -> str:
    opens = "([{<"
    closes = ")]}>"
    out = []
    for _ in range(n * 10):
        i = rng.randint(0, 3)
        out.append(opens[i] + rng.choice(_WORDS) + closes[i])
    return " ".join(out)


def _punctuation(rng: random.Random, n: int) -> str:
    p = "!?.,;:—…\"'`*#@&%"
    return "".join(rng.choice(p + " ") for _ in range(n * 40))


def _names(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(_NAMES) for _ in range(n * 12))


def _random(rng: random.Random, n: int) -> str:
    chars = string.ascii_letters + string.digits + " "
    return "".join(rng.choice(chars) for _ in range(n * 60))


def _corrupted(rng: random.Random, n: int) -> str:
    base = _prose(rng, n)
    chars = list(base)
    for _ in range(len(chars) // 8):
        j = rng.randint(0, len(chars) - 1)
        chars[j] = rng.choice("@#%&*\x00 ".replace("\x00", ""))
    return "".join(chars)


def _multilingual(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(_MULTILINGUAL) for _ in range(n * 2))


_GENERATORS = {
    "prose": _prose, "code": _code, "markup": _markup, "lists": _lists,
    "dialogue": _dialogue, "numbers": _numbers, "dates": _dates, "tables": _tables,
    "repeated": _repeated, "brackets": _brackets, "punctuation": _punctuation,
    "names": _names, "random": _random, "corrupted": _corrupted,
    "multilingual": _multilingual,
}


def generate_corpus(seed: int, items_per_regime: int = 8,
                    complexity: int = 6) -> list[CorpusItem]:
    """Generate a balanced, seeded list of corpus items across all regimes."""
    rng = random.Random(seed)
    items: list[CorpusItem] = []
    item_id = 0
    for regime in REGIMES:
        gen = _GENERATORS[regime]
        for _ in range(items_per_regime):
            text = gen(random.Random(rng.random()), complexity)
            items.append(CorpusItem(item_id, regime, text))
            item_id += 1
    rng.shuffle(items)
    #Reassign contiguous IDs after shuffle for stable ordering.
    for i, it in enumerate(items):
        it.corpus_item_id = i
    return items
