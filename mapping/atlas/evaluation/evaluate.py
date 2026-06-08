"""``atlas evaluate`` — run one model over an eval pack and produce a decision.

For each eval item this computes the loss family (assistant-only loss is the
headline), generates a deterministic continuation, runs the behaviour /
degeneration metrics and the item's trait checks, then aggregates everything into
a traceable summary: pass rates by category and difficulty, repetition/format
failure rates, mean assistant loss, and the worst examples. The summary never
collapses to a single fake "quality score"; it reports the components and a
final, traceable recommendation.

Outputs (under ``<out>/eval_runs/<model_id>/``):
    eval_summary.json   - aggregate metrics + recommendation
    eval_by_item.parquet - one row per eval item (the drill-down table)
    generations.jsonl   - prompt / generated_text / config per item
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..storage import write_parquet
from .behavior import analyze_generation
from .chat_format import encode_prompt, USER_TAG, ASSISTANT_TAG
from .checks import run_checks
from .eval_pack import EvalItem, load_eval_pack
from .loss import assistant_loss
from . import schemas


#Per-category generation budgets (deterministic eval uses fixed lengths).
_DEFAULT_MAX_NEW_TOKENS = {
    "direct_qa": 160,
    "multi_turn": 96,
    "summarization": 200,
    "coding_basic": 256,
    "instruction_following": 200,
    "refusal_boundaries": 96,
    "repetition_traps": 128,
    "formatting": 160,
}


@dataclass
class EvalConfig:
    """Deterministic generation settings for an eval run (temperature 0)."""

    max_new_tokens: int = 160
    per_category_tokens: dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_MAX_NEW_TOKENS))
    temperature: float = 0.0
    repetition_penalty: float = 1.0

    def tokens_for(self, category: str) -> int:
        return int(self.per_category_tokens.get(category, self.max_new_tokens))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "per_category_tokens": self.per_category_tokens,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "deterministic": True,
        }


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "model"


def _decode(backend, ids: list[int]) -> str:
    if not ids:
        return ""
    text = backend.tokenizer.decode(ids)
    #Trim at the next role marker so a role-loop continuation is still *detected*
    #(the marker stays in the analysed text) but not double-counted past it.
    return text


def _generate_for_item(backend, item: EvalItem, cfg: EvalConfig):
    prompt_ids = encode_prompt(backend.tokenizer, item.prompt_messages())
    new_ids = backend.generate_greedy(
        prompt_ids, max_new_tokens=cfg.tokens_for(item.category),
        eos_ids=[backend.eos_id] if backend.eos_id is not None else None)
    stopped_on_eos = bool(new_ids) and backend.eos_id is not None and new_ids[-1] == backend.eos_id
    if stopped_on_eos:
        new_ids = new_ids[:-1]
    return _decode(backend, new_ids), stopped_on_eos


def evaluate_model(backend, eval_pack, *, out_dir: str | Path,
                   model_id: str | None = None, cfg: EvalConfig | None = None,
                   eval_run_id: str | None = None) -> dict[str, Any]:
    """Evaluate ``backend`` over ``eval_pack`` (path or list of items).

    Returns the summary dict; also writes the summary, the by-item parquet, and a
    generations log under ``out_dir/eval_runs/<model_id>/``.
    """
    cfg = cfg or EvalConfig()
    items = load_eval_pack(eval_pack) if isinstance(eval_pack, (str, Path)) else list(eval_pack)
    model_id = model_id or getattr(backend, "model_path", "model")
    pack_name = Path(eval_pack).stem if isinstance(eval_pack, (str, Path)) else "items"
    eval_run_id = eval_run_id or f"eval.{_slug(model_id)}.{pack_name}"

    rows: list[dict[str, Any]] = []
    gen_log: list[dict[str, Any]] = []
    for item in items:
        answer = item.answer_text()
        if answer:
            lm = assistant_loss(backend, item.prompt_messages(), answer)
        else:
            lm = None
        generated, stopped = _generate_for_item(backend, item, cfg)
        bm = analyze_generation(generated, stopped_on_eos=stopped)
        cr = run_checks(generated, item.expected_traits, bm)

        rows.append({
            "eval_run_id": eval_run_id,
            "model_id": str(model_id),
            "item_id": item.id,
            "category": item.category,
            "difficulty": item.difficulty,
            "full_loss": lm.full_loss if lm else float("nan"),
            "assistant_loss": lm.assistant_loss if lm else float("nan"),
            "user_prompt_loss": lm.user_prompt_loss if lm else float("nan"),
            "assistant_perplexity": lm.assistant_perplexity if lm else float("nan"),
            "tokens_scored": lm.tokens_scored if lm else 0,
            "generated_text": generated,
            "length_words": bm.length_words,
            "stopped_cleanly": bm.stopped_cleanly,
            "empty_output": bm.empty_output,
            "malformed_output": bm.malformed_output,
            "contains_user_role_leak": bm.contains_user_role_leak,
            "contains_assistant_role_loop": bm.contains_assistant_role_loop,
            "repetition_3gram_rate": bm.repetition_3gram_rate,
            "repetition_5gram_rate": bm.repetition_5gram_rate,
            "unique_token_ratio": bm.unique_token_ratio,
            "degenerate": bm.degenerate,
            "behavior_pass": cr.behavior_pass,
            "format_pass": cr.format_pass,
            "must_include_pass": cr.must_include_pass,
            "reason_codes": cr.reason_codes,
        })
        gen_log.append({
            "item_id": item.id, "category": item.category,
            "prompt": [m for m in item.prompt_messages()],
            "generated_text": generated,
            "reference_answer": item.reference_answer,
            "generation_config": cfg.to_dict(),
            "model_id": str(model_id), "eval_run_id": eval_run_id,
        })

    summary = _summarize(rows, eval_run_id, str(model_id), pack_name, cfg)

    base = Path(out_dir) / "eval_runs" / _slug(str(model_id))
    base.mkdir(parents=True, exist_ok=True)
    write_parquet(rows, base / "eval_by_item.parquet", schema=schemas.EVAL_BY_ITEM)
    (base / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    with (base / "generations.jsonl").open("w") as fh:
        for g in gen_log:
            fh.write(json.dumps(g) + "\n")
    summary["_artifacts"] = {"dir": str(base)}
    return summary


def _finite(xs):
    return [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]


def _rate(flags) -> float:
    flags = list(flags)
    return (sum(1 for f in flags if f) / len(flags)) if flags else 0.0


def _group_rates(rows, key):
    out: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    for name, grp in groups.items():
        losses = _finite([r["assistant_loss"] for r in grp])
        out[name] = {
            "n": len(grp),
            "behavior_pass_rate": _rate(r["behavior_pass"] for r in grp),
            "assistant_loss_mean": (sum(losses) / len(losses)) if losses else None,
            "degenerate_rate": _rate(r["degenerate"] for r in grp),
        }
    return out


def _summarize(rows, eval_run_id, model_id, pack_name, cfg) -> dict[str, Any]:
    n = len(rows)
    losses = _finite([r["assistant_loss"] for r in rows])
    behavior_pass_rate = _rate(r["behavior_pass"] for r in rows)
    repetition_failure_rate = _rate(
        r["repetition_5gram_rate"] > 0.2 for r in rows)
    format_failure_rate = _rate(not r["format_pass"] for r in rows)
    degenerate_rate = _rate(r["degenerate"] for r in rows)

    reason_counts: dict[str, int] = {}
    for r in rows:
        for rc in r["reason_codes"]:
            #Collapse parametrised reasons (missing:Orbit) to their family.
            fam = rc.split(":", 1)[0]
            reason_counts[fam] = reason_counts.get(fam, 0) + 1

    worst = sorted(
        [r for r in rows if math.isfinite(r["assistant_loss"])],
        key=lambda r: r["assistant_loss"], reverse=True)[:10]
    worst_examples = [{
        "item_id": r["item_id"], "category": r["category"],
        "assistant_loss": r["assistant_loss"], "behavior_pass": r["behavior_pass"],
        "reason_codes": r["reason_codes"],
        "generated_text": r["generated_text"][:400],
    } for r in worst]

    recommendation, recommendation_reasons = _recommend(
        behavior_pass_rate, repetition_failure_rate, format_failure_rate,
        degenerate_rate)

    return {
        "eval_run_id": eval_run_id,
        "model_id": model_id,
        "eval_pack": pack_name,
        "n_items": n,
        "generation_config": cfg.to_dict(),
        "metrics": {
            "assistant_loss_mean": (sum(losses) / len(losses)) if losses else None,
            "assistant_loss_median": (sorted(losses)[len(losses) // 2]) if losses else None,
            "behavior_pass_rate": behavior_pass_rate,
            "repetition_failure_rate": repetition_failure_rate,
            "format_failure_rate": format_failure_rate,
            "degenerate_rate": degenerate_rate,
        },
        "category_pass_rates": _group_rates(rows, "category"),
        "difficulty_pass_rates": _group_rates(rows, "difficulty"),
        "failure_reason_counts": dict(sorted(
            reason_counts.items(), key=lambda kv: kv[1], reverse=True)),
        "worst_examples": worst_examples,
        "recommendation": recommendation,
        "recommendation_reasons": recommendation_reasons,
    }


def _recommend(behavior_pass_rate, repetition_failure_rate, format_failure_rate,
               degenerate_rate):
    """Produce a traceable health verdict (PASS / WARNING / FAIL) from the
    component metrics — never a single opaque score."""
    reasons = []
    if degenerate_rate > 0.5 or format_failure_rate > 0.5:
        reasons.append("majority of outputs degenerate or malformed")
        return "FAIL", reasons
    if behavior_pass_rate < 0.4:
        reasons.append(f"low behavior pass rate ({behavior_pass_rate:.0%})")
        return "FAIL", reasons
    if repetition_failure_rate > 0.2:
        reasons.append(f"elevated repetition failures ({repetition_failure_rate:.0%})")
    if format_failure_rate > 0.15:
        reasons.append(f"elevated format failures ({format_failure_rate:.0%})")
    if behavior_pass_rate < 0.75:
        reasons.append(f"behavior pass rate below target ({behavior_pass_rate:.0%})")
    if reasons:
        return "WARNING", reasons
    return "PASS", ["pass rates healthy; no degeneration flags"]
