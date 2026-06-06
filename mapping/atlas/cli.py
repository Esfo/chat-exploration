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


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    from .parallel import configure_threads
    threads = configure_threads(args.threads)
    print(f"[atlas] using up to {threads} CPU threads")
    if args.command == "run-all":
        _run_all(args)
    else:
        _run_stage(args.command, args)


if __name__ == "__main__":
    main()
