"""``atlas`` command-line interface.

Exposes the stage-level commands from the plan (init, scan-tensors, build-units,
static-analysis, build-static-graph, build-calibration, capture-activations,
run-probes, build-graphs, cluster, score-clusters, plan-interventions,
run-interventions, export-summaries, build-indexes) plus a ``run-all`` convenience
that executes the V1 stages in order.

Every command checks prerequisites, reads the manifests, logs its configuration,
and commits artifacts atomically through the ``Library`` handle. Running a stage
out of order fails fast with the list of missing inputs instead of a traceback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import stages
from .config import DEFAULT_LIBRARY_DIR, DEFAULT_MODEL_PATH, DEFAULT_TOKENIZER_PATH, ExtractionConfig
from .manifest import Library
from .model_backend import ModelBackend


def _backend_from_library(library: Library) -> ModelBackend:
    cfg = library.config()
    return _backend_from_config(cfg)


def _backend_from_config(cfg) -> ModelBackend:
    return ModelBackend(
        cfg.model_path, cfg.tokenizer_path, cfg.dtype, cfg.device,
        dequantize_f16=cfg.dequantize_f16, llama_quantize=cfg.llama_quantize,
        load_in_4bit=getattr(cfg, "load_in_4bit", False),
    )


def _run_stage(stage_name: str, args) -> None:
    stages.load_all()
    stage = stages.REGISTRY[stage_name]
    library = Library(args.out)

    if stage_name == "init":
        config = _config_from_init_args(args)
        backend = _backend_from_config(config)
        result = stage.run(library, backend, config=config)
        print(f"[init] {result}")
        return

    if not library.exists():
        sys.exit(f"No library at {args.out!r}. Run `atlas init` first.")

    missing = stages.check_prerequisites(library, stage)
    if missing:
        sys.exit(f"[{stage_name}] missing prerequisites: {missing}")

    backend = _backend_from_library(library)
    kwargs = _stage_kwargs(stage_name, args)
    result = stage.run(library, backend, **kwargs)
    print(f"[{stage_name}] {result}")


def _config_from_init_args(args) -> ExtractionConfig:
    cfg = ExtractionConfig()
    if args.model:
        cfg.model_path = args.model
    if args.tokenizer:
        cfg.tokenizer_path = args.tokenizer
    elif args.model:
        cfg.tokenizer_path = args.model
    if args.dtype:
        cfg.dtype = args.dtype
    if args.device:
        cfg.device = args.device
    if args.tokens:
        cfg.target_tokens = args.tokens
    if args.max_mlp_neurons_per_layer is not None:
        cfg.max_mlp_neurons_per_layer = args.max_mlp_neurons_per_layer
    if getattr(args, "dequantize_f16", False):
        cfg.dequantize_f16 = True
    if getattr(args, "llama_quantize", None):
        cfg.llama_quantize = args.llama_quantize
    if getattr(args, "load_in_4bit", False):
        cfg.load_in_4bit = True
    return cfg


def _stage_kwargs(stage_name: str, args) -> dict:
    kwargs = {}
    if stage_name == "build-calibration" and getattr(args, "tokens", None):
        kwargs["tokens"] = args.tokens
    if stage_name == "capture-activations" and getattr(args, "batch_size", None):
        kwargs["batch_size"] = args.batch_size
    return kwargs


def _run_all(args) -> None:
    """Run the V1 stage sequence in order (skips V2/V3 stages)."""
    stages.load_all()
    library = Library(args.out)
    if not library.exists():
        sys.exit(f"No library at {args.out!r}. Run `atlas init` first, then `run-all`.")
    #init is run explicitly by the user (it needs --model); run-all does the rest.
    import time
    v1 = [s for s in stages.ordered_stages()
          if s.version_scope == "v1" and s.name != "init"]
    #--from <stage>: force a rerun from that stage onward (handy when an upstream
    #stage was rerun and downstream outputs are now stale).
    from_stage = getattr(args, "from_stage", None)
    names = [s.name for s in v1]
    if from_stage and from_stage not in names:
        sys.exit(f"Unknown --from stage {from_stage!r}. Choices: {', '.join(names)}")
    force_from_idx = names.index(from_stage) if from_stage else len(names)

    def _schema_ok(rel):
        """True unless a produced parquet is missing columns its registered schema
        requires (i.e. it was written by older code) — a precise, timestamp-free
        staleness signal that catches schema changes without redoing valid work."""
        from . import schemas
        schema = schemas.SCHEMA_REGISTRY.get(rel)
        if schema is None:
            return True
        p = library.path(rel)
        try:
            import pyarrow.parquet as pq
            target = p if p.is_file() else next(p.rglob("*.parquet"))
            have = set(pq.read_schema(target).names)
        except Exception:  # noqa: BLE001 — unreadable: let the stage rerun
            return False
        return set(schema.names) <= have

    #Code fingerprint per stage: hash the stage's own source plus any helper
    #modules whose logic it depends on, and the schemas it writes. When the
    #fingerprint changes, the stage's *logic* changed (even if its output schema
    #did not — e.g. the exact-bitset capture change), so it reruns automatically.
    #This is what removes the need for --from after a code change.
    import hashlib
    import inspect
    from pathlib import Path
    from . import schemas as _schemas
    atlas_dir = Path(__file__).resolve().parent
    extra_deps = {
        "capture-activations": ["capture_accum.py", "sketches.py"],
        "build-graphs": ["sketches.py"],
        "static-analysis": ["sketches.py"],
    }

    def _fingerprint(stage):
        h = hashlib.sha256()
        files = [Path(inspect.getsourcefile(stage.run))]
        files += [atlas_dir / d for d in extra_deps.get(stage.name, [])]
        for f in files:
            try:
                h.update(f.read_bytes())
            except OSError:
                pass
        for rel in stage.produces or []:
            s = _schemas.SCHEMA_REGISTRY.get(rel)
            if s is not None:
                h.update(str(s).encode())
        return h.hexdigest()

    fp_path = library.path("stage_fingerprints.json")
    try:
        import json
        stored_fp = json.loads(fp_path.read_text())
    except Exception:  # noqa: BLE001 — first run / unreadable
        stored_fp = {}
    current_fp = {}

    cascade = False  # once a stage reruns, everything downstream must too
    for i, stage in enumerate(v1):
        #Resume by default. Rerun a stage when its outputs are missing, their
        #schema is out of date, its code fingerprint changed, or an earlier stage
        #already reran this pass. No timestamps, no flags needed for code changes.
        fp = _fingerprint(stage)
        current_fp[stage.name] = fp
        exists = stage.produces and all(library.path(p).exists() for p in stage.produces)
        schema_ok = exists and all(_schema_ok(p) for p in stage.produces)
        #Unknown stored fingerprint (first run after this feature) is treated as
        #current, so an already-complete library isn't needlessly rebuilt.
        code_changed = stage.name in stored_fp and stored_fp[stage.name] != fp
        forced = getattr(args, "force", False) or i >= force_from_idx
        if exists and schema_ok and not code_changed and not forced and not cascade:
            print(f"=== skipping {stage.name} (up to date) ===", flush=True)
            continue
        reason = ("forced" if forced else "upstream reran" if cascade
                  else "missing outputs" if not exists
                  else "schema out of date" if not schema_ok else "code changed")
        print(f"=== running {stage.name} ({reason}) ===", flush=True)
        t0 = time.time()
        _run_stage(stage.name, args)
        print(f"=== {stage.name} done in {(time.time()-t0)/60:.1f}m ===", flush=True)
        cascade = True
        #Persist fingerprints as we go so an interrupted run resumes correctly.
        import json
        stored_fp[stage.name] = fp
        fp_path.write_text(json.dumps({**stored_fp, **current_fp}, indent=2))

    #Stamp fingerprints for any stages that were up to date (migration / no-op).
    import json
    fp_path.write_text(json.dumps({**stored_fp, **current_fp}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atlas", description="MIBL extraction pipeline")
    p.add_argument("--out", default=str(DEFAULT_LIBRARY_DIR),
                   help="library directory (default: %(default)s)")
    p.add_argument("--threads", type=int, default=0,
                   help="CPU threads for torch/BLAS and the numpy stages "
                        "(0 = all cores)")
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="initialize a new library")
    #Default model is the Ollama GGUF (e.g. "llama3:8b"); pass a path for HF dirs
    #or a specific .gguf file.
    init.add_argument("--model", default=ExtractionConfig().model_path)
    init.add_argument("--tokenizer", default=None)
    init.add_argument("--dtype", default=None)
    init.add_argument("--device", default=None,
                      help="cpu | cuda | auto (default auto: GPU if available)")
    init.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true",
                      help="4-bit (nf4) quantize the model for GPU capture so an "
                           "8B checkpoint fits in a few GB of VRAM")
    init.add_argument("--tokens", type=int, default=None)
    init.add_argument("--dequantize-f16", dest="dequantize_f16", action="store_true",
                      help="dequantize the source GGUF to F16 via llama-quantize "
                           "before loading (fallback for unreadable quant types)")
    init.add_argument("--llama-quantize", dest="llama_quantize", default=None,
                      help="path/name of the llama-quantize binary")
    init.add_argument("--max-mlp-neurons-per-layer", dest="max_mlp_neurons_per_layer",
                      type=int, default=None,
                      help="cap MLP neurons per layer (0 = all) to bound cost")

    for name in ("scan-tensors", "build-units", "static-analysis",
                 "build-static-graph", "run-probes", "build-graphs", "cluster",
                 "score-clusters", "plan-interventions", "run-interventions",
                 "export-summaries", "build-indexes"):
        sub.add_parser(name, help=f"run the {name} stage")

    cal = sub.add_parser("build-calibration", help="build the calibration corpus")
    cal.add_argument("--tokens", type=int, default=None)

    cap = sub.add_parser("capture-activations", help="capture firing summaries")
    cap.add_argument("--batch-size", dest="batch_size", type=int, default=None)

    #--- evaluation commands (standalone: operate on a checkpoint directly) ---
    ev = sub.add_parser("evaluate",
                        help="evaluate one model on an eval pack (self-checking report)")
    ev.add_argument("--model-a", "--model", dest="model_a", required=True,
                    help="model checkpoint / GGUF / Ollama name to evaluate")
    ev.add_argument("--tokenizer", default=None)
    ev.add_argument("--eval-pack", dest="eval_pack", required=True,
                    help="path to a .jsonl eval pack")
    ev.add_argument("--eval-out", dest="eval_out", default="atlas_eval",
                    help="base output dir (default: %(default)s)")
    ev.add_argument("--device", default=None, help="cpu | cuda | auto")
    ev.add_argument("--dtype", default=None)
    ev.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=None)

    cmp_ = sub.add_parser("compare", help="compare two models on the same eval pack")
    cmp_.add_argument("--model-a", dest="model_a", required=True)
    cmp_.add_argument("--model-b", dest="model_b", required=True)
    cmp_.add_argument("--tokenizer-a", dest="tokenizer_a", default=None)
    cmp_.add_argument("--tokenizer-b", dest="tokenizer_b", default=None)
    cmp_.add_argument("--eval-pack", dest="eval_pack", required=True)
    cmp_.add_argument("--eval-out", dest="eval_out", default="atlas_eval")
    cmp_.add_argument("--device", default=None)
    cmp_.add_argument("--dtype", default=None)
    cmp_.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=None)

    sd = sub.add_parser("score-data",
                        help="score + label training rows (KEEP/DROP/REVIEW/...)")
    sd.add_argument("--model", required=True,
                    help="model used to compute assistant-only loss per row")
    sd.add_argument("--tokenizer", default=None)
    sd.add_argument("--dataset", required=True, help="training data .jsonl")
    sd.add_argument("--eval-summary", dest="eval_summary", default=None,
                    help="optional eval_summary.json to drive category COLLECT_MORE recs")
    sd.add_argument("--eval-out", dest="eval_out", default="atlas_eval",
                    help="base output dir (default: %(default)s)")
    sd.add_argument("--device", default=None)
    sd.add_argument("--dtype", default=None)

    fd = sub.add_parser("filter-data",
                        help="split a dataset into keep/drop/review using score-data output")
    fd.add_argument("--dataset", default=None,
                    help="original .jsonl (optional; falls back to score-data's source_rows)")
    fd.add_argument("--scores", required=True,
                    help="sample_recommendations.jsonl from score-data")
    fd.add_argument("--mode", choices=["conservative", "normal", "aggressive"],
                    default="conservative")
    fd.add_argument("--keep-out", dest="keep_out", required=True)
    fd.add_argument("--drop-out", dest="drop_out", required=True)
    fd.add_argument("--review-out", dest="review_out", default=None)

    ed = sub.add_parser("export-data-filter",
                        help="export kept/dropped datasets from a score-data run dir")
    ed.add_argument("--score-run", dest="score_run", required=True,
                    help="path to a data_scores/ dir (or its parent) from score-data")
    ed.add_argument("--mode", choices=["conservative", "normal", "aggressive"],
                    default="conservative")
    ed.add_argument("--keep-out", dest="keep_out", required=True)
    ed.add_argument("--drop-out", dest="drop_out", required=True)
    ed.add_argument("--review-out", dest="review_out", default=None)

    runall = sub.add_parser("run-all", help="run the full V1 pipeline in order")
    runall.add_argument("--force", action="store_true",
                        help="rerun every stage even if its outputs already exist")
    runall.add_argument("--from", dest="from_stage", default=None,
                        help="rerun from this stage onward (e.g. --from cluster), "
                             "ignoring existing outputs from it on; earlier stages "
                             "still skip if already done")
    runall.add_argument("--batch-size", dest="batch_size", type=int, default=None,
                        help="capture batch size (larger keeps CPU cores busier)")
    return p


def _eval_backend(model_path: str, tokenizer_path, args) -> ModelBackend:
    return ModelBackend(
        model_path, tokenizer_path or model_path,
        dtype=args.dtype or "bfloat16", device=args.device or "auto",
    )


def _eval_config(args):
    from .evaluation.evaluate import EvalConfig
    cfg = EvalConfig()
    if getattr(args, "max_new_tokens", None):
        cfg.max_new_tokens = args.max_new_tokens
        cfg.per_category_tokens = {}  # uniform budget when overridden
    return cfg


def _run_evaluate(args) -> None:
    from .evaluation import evaluate_model
    backend = _eval_backend(args.model_a, args.tokenizer, args)
    summary = evaluate_model(backend, args.eval_pack, out_dir=args.eval_out,
                             model_id=args.model_a, cfg=_eval_config(args))
    m = summary["metrics"]
    print(f"[evaluate] {summary['model_id']} on {summary['eval_pack']}: "
          f"{summary['recommendation']}")
    print(f"  behavior_pass_rate={m['behavior_pass_rate']:.2f} "
          f"assistant_loss_mean={m['assistant_loss_mean']} "
          f"format_fail={m['format_failure_rate']:.2f} "
          f"repetition_fail={m['repetition_failure_rate']:.2f}")
    print(f"  artifacts: {summary['_artifacts']['dir']}")


def _run_compare(args) -> None:
    from .evaluation import compare_models
    backend_a = _eval_backend(args.model_a, args.tokenizer_a, args)
    backend_b = _eval_backend(args.model_b, args.tokenizer_b, args)
    summary = compare_models(backend_a, backend_b, args.eval_pack,
                             out_dir=args.eval_out, model_a_id=args.model_a,
                             model_b_id=args.model_b, cfg=_eval_config(args))
    wr = summary["win_rate"]
    print(f"[compare] overall winner: {summary['overall_winner']}")
    print(f"  win_rate: A={wr['model_a']:.2f} B={wr['model_b']:.2f} tie={wr['tie']:.2f}")
    print(f"  improvements={summary['improvement_count']} "
          f"regressions={summary['regression_count']} "
          f"assistant_loss_delta_mean={summary['assistant_loss_delta_mean']}")
    print(f"  artifacts: {summary['_artifacts']['dir']}")


def _run_score_data(args) -> None:
    import json
    from .evaluation import score_dataset
    backend = _eval_backend(args.model, args.tokenizer, args)
    eval_summary = None
    if args.eval_summary:
        eval_summary = json.loads(Path(args.eval_summary).read_text())
    summary = score_dataset(backend, args.dataset, out_dir=args.eval_out,
                            eval_summary=eval_summary)
    print(f"[score-data] {summary['n_samples']} samples")
    for action, n in sorted(summary["action_counts"].items()):
        print(f"  {action}: {n}")
    print(f"  artifacts: {summary['_artifacts']['dir']}")


def _run_filter_data(args) -> None:
    from .evaluation import filter_dataset
    res = filter_dataset(args.scores, mode=args.mode, keep_out=args.keep_out,
                         drop_out=args.drop_out, review_out=args.review_out,
                         dataset_path=args.dataset)
    c = res["counts"]
    print(f"[filter-data] mode={res['mode']}: keep={c['keep']} drop={c['drop']} "
          f"review={c['review']} unscored={c['unscored']}")


def _run_export_data_filter(args) -> None:
    from .evaluation import filter_dataset
    run = Path(args.score_run)
    scores = run / "sample_recommendations.jsonl"
    if not scores.exists():
        scores = run / "data_scores" / "sample_recommendations.jsonl"
    if not scores.exists():
        sys.exit(f"no sample_recommendations.jsonl under {args.score_run!r}")
    res = filter_dataset(scores, mode=args.mode, keep_out=args.keep_out,
                         drop_out=args.drop_out, review_out=args.review_out)
    c = res["counts"]
    print(f"[export-data-filter] mode={res['mode']}: keep={c['keep']} drop={c['drop']} "
          f"review={c['review']} unscored={c['unscored']}")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    from .parallel import configure_threads
    threads = configure_threads(args.threads)
    print(f"[atlas] using up to {threads} CPU threads")
    dispatch = {
        "run-all": _run_all, "evaluate": _run_evaluate, "compare": _run_compare,
        "score-data": _run_score_data, "filter-data": _run_filter_data,
        "export-data-filter": _run_export_data_filter,
    }
    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        _run_stage(args.command, args)


if __name__ == "__main__":
    main()
