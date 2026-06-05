"""Stage 5 — build the calibration corpus.

Generates a compact, structurally diverse corpus, tokenizes it to the configured
token budget, and writes the calibration tables. Activation events later
reference the token index rather than copying text, keeping the activation tables
small.
"""

from __future__ import annotations

from collections import defaultdict

from ..calibration.corpus import generate_corpus, REGIMES
from ..manifest import Library
from ..model_backend import ModelBackend
from . import register


@register("build-calibration", 5, requires=["manifest/library.json"],
          produces=["calibration/token_index.parquet", "calibration/corpus_index.parquet"])
def run(library: Library, backend: ModelBackend, tokens: int | None = None,
        items_per_regime: int = 8, **kwargs):
    config = library.config()
    target_tokens = tokens or config.target_tokens
    max_len = config.max_sequence_length
    tok = backend.tokenizer

    #Generate more items than needed; stop once the token budget is reached.
    items = generate_corpus(config.calibration_seed, items_per_regime=items_per_regime)

    corpus_rows = []
    token_rows = []
    group = defaultdict(lambda: {"sequences": 0, "tokens": 0, "len_sum": 0})

    seq_id = 0
    total_tokens = 0
    rng_split = 0
    for item in items:
        if total_tokens >= target_tokens:
            break
        enc = tok(item.text, truncation=True, max_length=max_len,
                  return_attention_mask=False)
        ids_list = enc["input_ids"]
        if not ids_list:
            continue
        split = "train" if (rng_split % 5) else "holdout"
        rng_split += 1
        corpus_rows.append({
            "corpus_item_id": item.corpus_item_id, "regime": item.regime,
            "source": "procedural", "char_length": len(item.text),
            "token_length": len(ids_list), "split": split,
        })
        for pos, tid in enumerate(ids_list):
            token_rows.append({
                "corpus_item_id": item.corpus_item_id, "sequence_id": seq_id,
                "token_position": pos, "token_id": int(tid),
                "token_text": tok.decode([tid]),
            })
        g = group[item.regime]
        g["sequences"] += 1
        g["tokens"] += len(ids_list)
        g["len_sum"] += len(ids_list)
        total_tokens += len(ids_list)
        seq_id += 1

    group_rows = [{
        "regime": r, "sequence_count": group[r]["sequences"],
        "token_count": group[r]["tokens"],
        "mean_token_length": (group[r]["len_sum"] / group[r]["sequences"]) if group[r]["sequences"] else 0.0,
    } for r in REGIMES if r in group]

    split_tokens = defaultdict(lambda: {"items": 0, "tokens": 0})
    for c in corpus_rows:
        split_tokens[c["split"]]["items"] += 1
        split_tokens[c["split"]]["tokens"] += c["token_length"]
    split_rows = [{
        "split": s, "corpus_item_count": v["items"], "token_count": v["tokens"],
    } for s, v in split_tokens.items()]

    library.commit_table("calibration/corpus_index.parquet", corpus_rows, "calibration", "build-calibration")
    library.commit_table("calibration/token_index.parquet", token_rows, "calibration", "build-calibration")
    library.commit_table("calibration/token_group_stats.parquet", group_rows, "calibration", "build-calibration")
    library.commit_table("calibration/calibration_splits.parquet", split_rows, "calibration", "build-calibration")
    library.log("build-calibration", "calibration corpus built",
                sequences=seq_id, tokens=total_tokens)
    library.update_artifact_versions("build-calibration")
    return {"sequences": seq_id, "tokens": total_tokens}
