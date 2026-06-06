# TECHNICAL DOCUMENT 2: DASHBOARD AND DATA-LIBRARY DEVELOPMENT PLAN FOR THE LLM INTERNAL-BEHAVIOR ATLAS

> This is the spec of record for the dashboard rebuild. Committed to the repo so
> it persists across sessions. Source: user-provided technical document.

## 0. PURPOSE
Turn the existing model-internals data library into a serious dashboard and
analysis system that presents the internal mechanics of an LLM in a way that is
fast, continuous, cluster-centered, and visually meaningful. The next problem is
presentation quality and analytical trust. The dashboard should behave like a
professional model atlas: each page has a clear purpose, the title explains what
the viewer sees, and the most important plots show immediately. Organize around
fixed, meaningful dashboards rather than a pile of controls. The most important
visual subject is clusters, presented as continuous landscapes, density plots,
scatter fields, ridgelines, heatmaps, contour plots, flow maps, and linked
drilldowns — not stacked bars. Prefer fast precomputed summary artifacts over raw
table scans. Make the model navigable at scale.

## 1. PRODUCT DIRECTION — two layers
Fixed dashboards (stable names/layouts, immediately visible plots) are the main
interface. A powerful custom plotting/query section exists for exploration but is
not the first thing seen.

Page list:
1. Library Health
2. Extraction Quality
3. Cluster Mechanics
4. Cluster Landscape
5. Cluster Drilldown
6. Signal Flow
7. Layer Dynamics
8. Activation Structure
9. Weight Geometry
10. Evidence Strength
11. Unit Explorer
12. Histogram Lab
13. Custom Plot Studio
14. SQL Workbench

Pages 1–2 are trust pages. Middle pages are atlas pages. Final pages are expert
tools.

## 2. VISUAL DESIGN PRINCIPLES
Small set of high-quality plot types used repeatedly: continuous scatter
landscapes (position means something); density contours/hexbin where too many
points; ridgeline/overlaid density for distributions; heatmaps for layer↔layer
and family↔family; flow/Sankey sparingly; linked drilldown; one-sentence section
labels; consistent metric names; precomputed summaries.

## 3. GLOBAL UI STRUCTURE
Each page: one-line purpose under the title; compact metric strip (4–8 numbers);
main continuous plots (visible without expanding); secondary detail panels;
optional controls (refine, not required). Sidebar holds only global filters:
library path, model/run selector, cluster level, layer range, unit type, min
confidence, evidence type, sample/full toggle.

## 4. PAGE: LIBRARY HEALTH
State of the data library. Metric strip: model name, param count, layer count,
unit count, cluster count, edge count, calibration tokens, library size on disk.
Plot 1: artifact completion matrix (stage × artifact group, status heatmap).
Plot 2: artifact size by storage group (treemap/size bars). Plot 3: row-count
scale map (log). Details: run log, artifact versions, missing required
artifacts, schema warnings, config summary. Data: artifact_index, versions,
manifest json, run log, file sizes. Code: artifact scanner (size/rows/mtime/
schema/stage) + `library_health` view; no per-rerun directory walks.

## 5. PAGE: EXTRACTION QUALITY
Quality-control cockpit. Metric strip: observed tokens, units w/ stats, units w/
top events, avg activation rate, median bitset density, graph sampled/full,
cluster coverage %, edge confidence median. Plots: (1) activation-rate density by
layer (ridgeline); (2) bitset saturation by layer/type (critical); (3) top-event
coverage; (4) graph participation coverage per layer by unit type; (5) edge
confidence distribution by type (overlaid density). Details: warnings for
saturated bitsets, missing stats, sampled graphs, low top-event coverage, giant
components. Data: activation_quality, graph_quality, extraction_warnings.

## 6. PAGE: CLUSTER MECHANICS
How clusters behave internally. Metric strip: total/local/xlayer/family clusters,
median fire & structural coherence, median member count, largest cluster. Plots:
(1) coherence field fire vs structural, size=members, color=level/type; (2) size
vs specificity; (3) activation-rate density (hexbin); (4) read/write geometry
field; (5) layer-span field (layer_center vs layer_span). Details: top coherence/
specificity/large-low-coherence/giant clusters, selected summary. Data:
cluster_stats, cluster_quality, cluster_embedding_2d, materialized_cluster_views.
Code: cluster quality metrics (coherence percentiles, internal/external edge
density, member IQRs, giant flag); weighted/mutual-kNN clustering.

## 7. PAGE: CLUSTER LANDSCAPE
Clusters as a continuous behavior-space map. Plots: (1) 2D cluster behavior map
(precomputed embedding; color by role/level/layer); (2) same w/ density contours;
(3) role-gradient map (source-sink, relay); (4) layer-depth gradient; (5)
family hull overlay. Data: cluster_embedding_2d, embedding_metadata,
cluster_family_hulls, cluster_nearest_neighbors. Code: embedding stage (inputs:
activation centroid, static geometry, role scores, graph neighborhood, layer
features); store artifacts; UMAP/PaCMAP/t-SNE/spectral/force-directed; run meta.

## 8. PAGE: CLUSTER DRILLDOWN
Everything about one cluster. Selector: id search, nearest, top-by-score. Header
card. Plots: (1) member distribution by layer; (2) member activation landscape;
(3) member read/write geometry; (4) evidence stack (continuous profile, not
stacked bar); (5) upstream/downstream neighborhood; (6) histogram comparison vs
layer/type/family baselines. Details: top members, top events, strongest in/out
edges, tensor refs, child/parent clusters. Data: cluster_member_view,
cluster_edge_neighborhoods, cluster_histogram_comparisons, cluster_top_events,
cluster_tensor_refs. Code: materialized drilldown view, top-k up/down edges,
histogram overlays, representative members, search.

## 9. PAGE: SIGNAL FLOW
Where signals originate/travel/concentrate/terminate. Separate evidence types.
Plots: (1) layer-to-layer flow heatmap, side-by-side combined/activation/static/
lagged; (2) source/sink layer curves; (3) relay density by layer; (4) cluster
flow field (source vs sink, color relay); (5) top flow corridors (layer-
positioned network). Details: top source/sink/relay clusters, top cross-layer
edges, evidence disagreement lists. Data: layer_flow_matrix, layer_role_curves,
cluster_edges, cluster_flow_corridors, evidence_disagreement. Code: evidence-
specific layer flow matrices; source/sink/relay by layer; edge disagreement
scoring.

## 10. PAGE: LAYER DYNAMICS
How mechanics change with depth. Plots: (1) layer metric curves (small
multiples); (2) layer distribution ridgelines; (3) cluster count/size field by
layer; (4) layer transition strength; (5) depthwise family presence. Data:
layer_summary, layer_metric_distributions, family_layer_presence,
layer_transition_matrix. Code: layer summary export with quantiles + baselines.

## 11. PAGE: ACTIVATION STRUCTURE
Empirical firing behavior. Plots: (1) activation rate vs specificity field; (2)
burstiness landscape; (3) activation distribution by layer (ridgeline); (4) top-
event token-position map; (5) cluster activation similarity map. Data:
activation_stats, activation_top_events, activation_embedding_2d,
activation_distribution_by_layer, top_event_position_density. Code (done in Tech
Doc work): per-unit thresholds, deterministic signatures, exact bitset/MinHash,
calibration split ids.

## 12. PAGE: WEIGHT GEOMETRY
Static weight structure vs clusters. Plots: (1) read/write norm field; (2)
weight-tail field; (3) static alignment heatmap (layer→layer); (4) singular
spectrum by tensor type (optional); (5) cluster static baseline comparison. Data:
unit_static_stats, tensor_stats, singular_summaries,
unit_edges_static_alignment, static_layer_alignment_matrix. Code: chunked memory-
safe moments (done), exact norms, static outlier baselines, lower worker count.

## 13. PAGE: EVIDENCE STRENGTH
Why edges/clusters are believed. Make the evidence stack visible; never present
combined score as one measurement. Plots: (1) evidence score scatter matrix; (2)
combined vs confidence; (3) evidence composition field (ternary-like); (4)
evidence disagreement map. Details: multi-evidence/static-only/activation-only/
disagreement/causal edges, selected edge stack. Data: unit_edges_combined,
cluster_edges, evidence_disagreement, edge_evidence_profiles. Code:
dominant_evidence_type, evidence_entropy, evidence_support_count,
disagreement_score; confidence stored separately from combined.

## 14. PAGE: UNIT EXPLORER
Inspect individual units in context. Plots: activation histogram w/ baselines,
similar-unit neighborhood, unit edge neighborhood, unit position in cluster.
Data: unit_neighborhoods, unit_cluster_context, unit_histogram_comparisons.

## 15. PAGE: HISTOGRAM LAB
Opinionated, fast precomputed-distribution comparison. Plots: cluster vs
baseline density, role comparison density, layer-ranged distribution field,
family distribution comparison. Data: histogram_bins, histogram_index,
baseline_comparisons, quantile_summaries. Code: export histograms for all entity
types + role groups; normalized density bins; baseline references.

## 16. PAGE: CUSTOM PLOT STUDIO
Powerful custom analysis over curated datasets/summary views (not raw tables
first). Linked plots, faceting, baseline overlays, saved chart specs, exportable
SQL/spec, load from precomputed histograms. Curated datasets: clusters_full,
cluster_members_full, cluster_edges_full, units_full, units_with_cluster,
layer_summary, histogram_ready, evidence_profiles, activation_quality,
static_geometry, signal_flow_summary. Chart types: scatter, scatter density,
hexbin, contour density, line, ridgeline, heatmap, histogram, density overlay,
box/violin, network neighborhood, layer matrix, table. Distinguish raw vs summary
plots; recommend summaries for large tables. Code: chart-spec JSON schema, saved
charts under summaries/saved_charts/, curated plotting views.

## 17. STREAMLIT WARNING FIX (Stage A — DONE)
Replace use_container_width=True → width="stretch"; False → width="content".

## 18. EXTRACTION CRITIQUES (Tech Doc work — DONE)
Issue 1 bitset saturation (→ exact bitset). Issue 2 per-unit thresholds. Issue 3
deterministic signatures. Issue 4 top-m per batch. Issue 5 real reservoir. Issue
6 mutual-kNN clustering + quality flags. Issue 7 explicit graph caps/coverage.
Issue 8 memory-safe static. Issue 9 reduce raw dashboard scans. Issue 10 keep
edge evidence separated.

## 19. NEW SUMMARY ARTIFACTS
library_health, extraction_quality, layer_summary, cluster_quality,
cluster_drilldown_view, cluster_histogram_comparisons, evidence_profiles,
layer_flow_matrix, cluster_embedding_2d, unit_embedding_2d, topk_index.

## 20. EXECUTION STAGES
- A: dashboard deprecation cleanup (DONE)
- B: dashboard page restructure to the 14 pages
- C: summary artifact buildout (`export-dashboard-summaries` stage)
- D: cluster visualization upgrade (mechanics, landscape embedding, drilldown,
  histogram comparisons) — most important product work
- E: extraction quality metrics (DONE)
- F: graph & clustering correctness (DONE: mutual-kNN, coverage, quality)
- G: activation signature correction (DONE: deterministic sigs, thresholds,
  exact bitset, top-m events)
- H: Custom Plot Studio v2
- I: performance hardening (summaries/materialized views, caching, row limits)

## 21. FINAL TARGET EXPERIENCE
Opening a technical atlas of the model: first see the library is complete and
trustworthy; then the cluster space as a continuous landscape; then drill into
one cluster to see where it lives, how members activate, weight structure, what
feeds in/out, supporting evidence, and baseline comparisons — without manually
building plots, without stacked bars for continuous data, without scanning raw
tables, without knowing the whole schema.
