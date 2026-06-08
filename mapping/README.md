# Atlas — Model Internal Behavior Library (MIBL)

Atlas turns an open-weight transformer into a **durable, queryable, on-disk data
library** of its internal structure and firing behavior. It is not a single
report, visualization, or interpretation: it is a *model-internals warehouse*
that later pipelines (visualization, anomaly detection, cluster labeling, causal
testing, model comparison, safety analysis, plain statistics) can run on top of
without ever reloading the full model or recomputing expensive intermediates.

The system inspects the model at three levels — **static weights**, **observed
activations**, and **relationships between components** that fire together or pass
signals forward — and organizes everything into indexed, cross-referenced
artifacts with stable IDs and versioned extraction runs.

This project lives in `mapping/atlas` and follows the save-location conventions
of the sibling `rescaling/` pipeline: large artifacts are written under
`/home/sfo/data/models` (configurable via `--out` / `ATLAS_DATA_ROOT`), and the
analyzed model is the same Ollama-provided `llama3:8b` GGUF that `rescaling`
scales — **no Hugging Face download required**.

Activation capture needs forward hooks, which need a PyTorch model. Rather than
download HF weights, the backend resolves the Ollama GGUF the same way
`rescaling/weightscaling.py` does (`ollama show --modelfile` → the `FROM` blob)
and lets `transformers` load that GGUF directly, **dequantizing it in memory**
into a `LlamaForCausalLM` we can hook. If a source quantization type can't be
read directly, pass `--dequantize-f16` (or `DEQUANTIZE_F16=1 ./atlasexecution`)
to first dequantize to an F16 GGUF with `llama-quantize` — the same step
`rescaling` uses — and load that. Point `--model` at an HF directory or a
specific `.gguf` file to override.

## Concepts (kept separate on purpose)

Different kinds of evidence are stored separately before being combined:

| Evidence            | Meaning                                                        |
|---------------------|---------------------------------------------------------------|
| Co-firing           | these units activate together                                 |
| Static alignment    | this unit *could* feed that unit (weight geometry permits it)  |
| Lagged activation   | this unit tends to activate before that unit                  |
| Attention routing   | a head routes + writes a direction later units read           |
| Probe response      | a later unit responds to a source-like internal signal (V2)   |
| Causal intervention | changing the source changes the target (V2)                   |

The combined graph keeps every evidence column plus a weighted `combined_score`
and an `edge_confidence`, so downstream code chooses how strict to be.

## Storage layers

* **cold array layer** — Zarr (`*.zarr`): firing signatures, bitsets, centroids
* **columnar table layer** — Parquet: catalogs, stats, edges, clusters, summaries
* **query layer** — DuckDB views over the Parquet (`indexes/atlas.duckdb`)
* **vector / graph / bitmap index layers** — built in the final stage

Every high-level object traces downward: cluster → units → tensor slices →
checkpoint tensors; activation summaries → calibration token positions; edges →
the evidence that produced them. See `atlas/config.py::LAYOUT` for the full
on-disk contract and `atlas/schemas.py` for every table schema.

## Pipeline (resumable stages)

```
atlas init --model PATH --tokenizer PATH --out atlas_library/
atlas scan-tensors          # tensor catalog + weight baselines
atlas build-units           # MLP neuron + attention head registry, slice refs
atlas static-analysis       # read/write geometry, direction sketches
atlas build-static-graph    # possible-flow (static alignment) graph
atlas build-calibration --tokens 500000
atlas capture-activations   # forward hooks -> compressed firing summaries
atlas run-probes            # internal probe engine (V2; static approx in V1)
atlas build-graphs          # activation/lagged/routing/combined graphs
atlas cluster               # local -> cross-layer -> family clustering
atlas score-clusters        # role scores, coherence, cluster edges
atlas plan-interventions    # select high-confidence edges (V2)
atlas run-interventions     # causal validation (V2 placeholder)
atlas export-summaries      # histograms, quantiles, top-k, baselines, views
atlas build-indexes         # DuckDB views, vector + graph + bitmap indexes
```

Each stage checks prerequisites, reads the manifests, logs its config, writes
temporary outputs, validates schemas, and commits atomically. `atlas run-all`
runs the V1 sequence after `init`.

End-to-end driver (mirrors `rescaling/scalingexecution`):

```
./atlasexecution
```

## Querying a built library

```python
from atlas.query import AtlasQuery
q = AtlasQuery("/home/sfo/data/models/atlas_library")

q.units_in_cluster("cl.000007")        # cluster -> member units
q.downstream_clusters("cx.000003")     # signal-flow neighborhood
q.tensor_slices_for_unit("L12.mlp_neuron.000345")  # unit -> checkpoint slices
q.similar_units("L12.mlp_neuron.000345", k=10)     # vector similarity
```

Or straight from DuckDB:

```
duckdb /home/sfo/data/models/atlas_library/indexes/atlas.duckdb
> SELECT * FROM cluster_stats WHERE source_score > 0.7 ORDER BY mean_write_norm DESC;
```

## Self-checking evaluation (`atlas evaluate` / `atlas compare`)

The extraction pipeline describes a model's internals; the evaluation harness
answers the practical question *"is this model actually good, and is B better than
A?"*. These two commands are **standalone** — they operate on a checkpoint
directly and do not require a built library:

```
atlas evaluate --model-a runs/model.pt --eval-pack evals/chat_basic.jsonl
atlas compare  --model-a runs/old.pt --model-b runs/new.pt --eval-pack evals/chat_basic.jsonl
```

The harness never collapses everything into one fake "quality score"; it reports
separate, traceable metric families and a final PASS/WARNING/FAIL recommendation:

* **assistant-only loss** — masks the user prompt and scores only the assistant
  answer tokens (`assistant_loss`, `assistant_perplexity`, `tokens_scored`), the
  metric that actually reflects conversational answer quality.
* **deterministic generation** (temperature 0) + **behaviour / degeneration
  metrics** — empty/malformed output, role leak, assistant-role loops, repeated
  lines, 3/5-gram repetition rate, unique-token ratio.
* **trait checks** — each eval item's `expected_traits` (`should_answer`,
  `max_words`, `must_include`, `must_not_include`, `requires_code`) become
  pass/fail checks aggregated into `behavior_pass_rate` and per-reason tables.

`atlas compare` runs both models on the *same* prompts and picks a per-item
winner from loss **and** behaviour (never loss alone), classifying each item as
`A_STRONG_WIN` / `A_WEAK_WIN` / `TIE` / `B_WEAK_WIN` / `B_STRONG_WIN` /
`BOTH_BAD` / `METRICS_DISAGREE`, then writes `comparison_by_item.parquet`,
per-category win rates, and `regression_cases.jsonl` / `improvement_cases.jsonl`.

Outputs land under `--eval-out` (default `atlas_eval/`):

```
atlas_eval/eval_runs/<model>/eval_summary.json + eval_by_item.parquet + generations.jsonl
atlas_eval/comparisons/<a>_vs_<b>/comparison_summary.json + *.parquet + *.jsonl
```

### Eval packs

Eval data is kept separate from training data as `.jsonl` (one item per line); see
`evals/` for samples (`chat_basic`, `chat_multi_turn`, `refusal_boundaries`):

```json
{"id": "chat_basic_000001", "category": "direct_qa", "difficulty": "easy",
 "messages": [{"role": "user", "content": "Explain overfitting in simple terms."}],
 "expected_traits": {"should_answer": true, "max_words": 120},
 "reference_answer": "Overfitting is when a model memorizes training data ..."}
```

An item ending in a user turn is answered by the model (and scored for loss
against `reference_answer`); an item ending in a held-out assistant turn scores
that turn directly.

## Data scoring & filtering (`atlas score-data` / `filter-data`)

The other half of the feedback loop: decide which *training* rows to keep, fix,
downsample, drop, or collect more of.

```
atlas score-data  --model runs/model.pt --dataset data/train.jsonl
atlas filter-data --scores atlas_eval/data_scores/sample_recommendations.jsonl \
                  --mode conservative --keep-out train.clean.jsonl --drop-out train.drop.jsonl
```

`score-data` computes, per row: model-free **text-quality** signals (non-answer,
prompt-echo, role leakage, scrape artifacts, weird characters, repetition),
exact + near-duplicate group sizes (a dependency-free MinHash), and
**assistant-only loss** from the model. It then assigns one action label using
dataset-relative context (loss percentile, category frequency, per-category
weakness):

* **KEEP** — clean, useful, on-format.
* **KEEP_HARD** — high loss but clean and valuable; *not dropped just for being
  hard*. High loss is only a DROP signal when it co-occurs with quality failures.
* **DOWNSAMPLE** — clean but duplicated / too easy.
* **FIX** — good idea, broken structure (bad roles, missing fields).
* **DROP** — malformed, non-answer, prompt-echo, repetition, artifacts, or exact
  duplicates.
* **COLLECT_MORE_LIKE_THIS** — clean, rare category where the model is weak.
* **REVIEW** — metrics disagree / low confidence.

Outputs under `--eval-out/data_scores/`: `sample_scores.parquet`,
`sample_recommendations.jsonl`, `category_recommendations.json` (per-category
COLLECT_MORE / CLEAN_EXISTING / DOWNSAMPLE / HOLD with target counts), and
`drop_ids.txt` / `keep_ids.txt` / `review_ids.txt` / `collect_more.json`. Pass an
`--eval-summary` from `atlas evaluate` to fold eval failure rates into the
collection plan.

`filter-data` routes the original source lines to keep/drop/review files;
**conservative** (the default) only drops high-confidence DROP rows so the loop
never deletes hard-but-useful data early. `export-data-filter --score-run DIR`
does the same straight from a score-data run directory. The Streamlit dashboard
pages (Model Quality / Compare / Data Quality / Collection Plan) remain a planned
follow-up.

### Interactive explorer

A Streamlit dashboard browses the library without writing SQL — overview, clusters
(with members and signal flow), units (top firing tokens, similar units, weight
slices), the connection graph, and a free-form SQL console:

```
pip install streamlit
./atlasdashboard                       # default library location
./atlasdashboard /path/to/atlas_library
```

It reads only the committed artifacts (DuckDB views + `AtlasQuery`); it never loads
the model.

## Scope

* **V1 (implemented):** decoder-only support, tensor + unit catalogs, MLP neuron
  and attention-head units, static stats, calibration, activation capture,
  signatures + bitsets, fire-together + lagged graphs, multi-level clustering,
  cluster stats, histogram/summary export, DuckDB + vector + graph indexes.
* **V2:** live residual-space probe engine, attention-routing refinement,
  cluster-edge confidence, selected causal interventions (contracts already
  present: `run-probes`, `plan-interventions`, `run-interventions`).
* **V3:** SAE feature units, cluster stability across corpora, cross-model
  comparison, richer causal tests, graph export formats, semantic labels.

## Installation & tests

```
pip install -r requirements-atlas.txt      # numpy/pyarrow/zarr/duckdb + torch/transformers
python -m pytest tests/                     # runs the full pipeline on a tiny fake backend (no torch needed)
```

The test suite drives every stage against a torch-free `FakeBackend`, so the data
layers, schemas, clustering, and query interfaces are exercised end-to-end
without a GPU or a multi-gigabyte checkpoint.

## Performance rules (enforced by design)

No dense token×unit activation matrices, no dense unit×unit similarity matrices,
no repeated checkpoint reloads after build. Instead: streaming statistics,
chunked Zarr, partitioned Parquet, sparse top-k edges, random-projection
signatures, compressed bitsets, materialized summaries, stable IDs, and resumable
stages.
