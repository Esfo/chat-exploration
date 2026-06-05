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
    for stage in v1:
        print(f"=== running {stage.name} ===", flush=True)
        t0 = time.time()
        _run_stage(stage.name, args)
        print(f"=== {stage.name} done in {(time.time()-t0)/60:.1f}m ===", flush=True)


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
    init.add_argument("--device", default=None)
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

    sub.add_parser("run-all", help="run the full V1 pipeline in order")
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
