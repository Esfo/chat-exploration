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
analyzed model is the same Llama 3 8B family. Because activation capture requires
forward hooks, the default backend loads the **HuggingFace/PyTorch** checkpoint
(`/home/sfo/data/models/Meta-Llama-3-8B`) rather than the Ollama GGUF that
`rescaling` scales.

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
