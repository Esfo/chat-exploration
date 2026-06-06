"""PyArrow schemas for every Parquet artifact in the library.

These schemas are the durable *contract* between the extraction system and the
downstream pipelines described in the plan (histogram, graph, similarity,
drilling, comparison, causal). Keeping them centralized means a stage cannot
silently drift from what a consumer expects, and a schema-validation step can
check committed artifacts against these definitions.

Where the plan lists "Important fields" we include those exact field names and
add the structural fields (IDs, run IDs) needed for traceability. Nested values
such as ``quantiles`` are stored as lists so a single row fully describes a
distribution without a side table.
"""

from __future__ import annotations

import pyarrow as pa


#Reusable building blocks.
_FLOAT_LIST = pa.list_(pa.float64())
_INT_LIST = pa.list_(pa.int64())
_STR_LIST = pa.list_(pa.string())


def _schema(fields: list[tuple[str, pa.DataType]]) -> pa.Schema:
    return pa.schema([pa.field(n, t) for n, t in fields])


#--------------------------------------------------------------------------
#catalog/
#--------------------------------------------------------------------------

TENSOR_INDEX = _schema([
    ("tensor_id", pa.string()),
    ("model_id", pa.string()),
    ("tensor_name", pa.string()),
    ("logical_name", pa.string()),
    ("layer_id", pa.int32()),
    ("component_type", pa.string()),
    ("architecture_role", pa.string()),
    ("shape", _INT_LIST),
    ("dtype", pa.string()),
    ("parameter_count", pa.int64()),
    ("checkpoint_file", pa.string()),
    ("checkpoint_offset", pa.int64()),
    ("hash", pa.string()),
])

LOGICAL_TENSOR_INDEX = _schema([
    ("logical_tensor_id", pa.string()),
    ("tensor_id", pa.string()),
    ("model_id", pa.string()),
    ("logical_name", pa.string()),
    ("layer_id", pa.int32()),
    ("component_type", pa.string()),
    ("slice_axis", pa.int32()),
    ("start_index", pa.int64()),
    ("end_index", pa.int64()),
])

UNIT_INDEX = _schema([
    ("unit_id", pa.string()),
    ("model_id", pa.string()),
    ("layer_id", pa.int32()),
    ("unit_type", pa.string()),
    ("component_name", pa.string()),
    ("local_index", pa.int64()),
    ("read_tensor_id", pa.string()),
    ("write_tensor_id", pa.string()),
    ("parent_tensor_ids", _STR_LIST),
    ("hidden_size", pa.int64()),
    ("unit_dim", pa.int64()),
    ("is_clusterable", pa.bool_()),
])

UNIT_WEIGHT_REFS = _schema([
    ("unit_id", pa.string()),
    ("tensor_id", pa.string()),
    ("slice_type", pa.string()),  # "read" | "write" | "qk" | "ov"
    ("dimension", pa.int32()),
    ("start_index", pa.int64()),
    ("end_index", pa.int64()),
    ("stride", pa.int64()),
    ("interpretation", pa.string()),
])

ARTIFACT_INDEX = _schema([
    ("artifact_path", pa.string()),
    ("artifact_class", pa.string()),
    ("format", pa.string()),
    ("stage", pa.string()),
    ("schema_version", pa.string()),
    ("row_count", pa.int64()),
    ("partition_keys", _STR_LIST),
    ("created_at", pa.string()),
    ("committed", pa.bool_()),
])


#--------------------------------------------------------------------------
#tensors/
#--------------------------------------------------------------------------

TENSOR_STATS = _schema([
    ("tensor_id", pa.string()),
    ("mean", pa.float64()),
    ("std", pa.float64()),
    ("min", pa.float64()),
    ("max", pa.float64()),
    ("abs_max", pa.float64()),
    ("quantiles", _FLOAT_LIST),
    ("skewness", pa.float64()),
    ("kurtosis", pa.float64()),
    ("row_norm_summary", _FLOAT_LIST),
    ("col_norm_summary", _FLOAT_LIST),
])

TENSOR_HISTOGRAMS = _schema([
    ("tensor_id", pa.string()),
    ("bin_left", _FLOAT_LIST),
    ("bin_right", _FLOAT_LIST),
    ("count", _INT_LIST),
])

ROW_COL_STATS = _schema([
    ("tensor_id", pa.string()),
    ("axis", pa.string()),
    ("index", pa.int64()),
    ("norm", pa.float64()),
    ("mean", pa.float64()),
    ("std", pa.float64()),
])

SINGULAR_SUMMARIES = _schema([
    ("tensor_id", pa.string()),
    ("singular_values", _FLOAT_LIST),
    ("effective_rank", pa.float64()),
    ("energy_top1", pa.float64()),
    ("energy_top8", pa.float64()),
])


#--------------------------------------------------------------------------
#units/
#--------------------------------------------------------------------------

UNIT_STATIC_STATS = _schema([
    ("unit_id", pa.string()),
    ("read_norm", pa.float64()),
    ("write_norm", pa.float64()),
    ("read_mean", pa.float64()),
    ("read_std", pa.float64()),
    ("write_mean", pa.float64()),
    ("write_std", pa.float64()),
    ("read_kurtosis", pa.float64()),
    ("write_kurtosis", pa.float64()),
    ("weight_tail_ratio", pa.float64()),
    ("static_outlier_score", pa.float64()),
])

UNIT_ROLE_SCORES = _schema([
    ("unit_id", pa.string()),
    ("static_source_score", pa.float64()),
    ("static_sink_score", pa.float64()),
    ("routing_score", pa.float64()),
    ("write_strength", pa.float64()),
])


#--------------------------------------------------------------------------
#calibration/
#--------------------------------------------------------------------------

CORPUS_INDEX = _schema([
    ("corpus_item_id", pa.int64()),
    ("regime", pa.string()),
    ("source", pa.string()),
    ("char_length", pa.int64()),
    ("token_length", pa.int64()),
    ("split", pa.string()),
])

TOKEN_INDEX = _schema([
    ("corpus_item_id", pa.int64()),
    ("sequence_id", pa.int64()),
    ("token_position", pa.int32()),
    ("token_id", pa.int64()),
    ("token_text", pa.string()),
])

TOKEN_GROUP_STATS = _schema([
    ("regime", pa.string()),
    ("sequence_count", pa.int64()),
    ("token_count", pa.int64()),
    ("mean_token_length", pa.float64()),
])

CALIBRATION_SPLITS = _schema([
    ("split", pa.string()),
    ("corpus_item_count", pa.int64()),
    ("token_count", pa.int64()),
])


#--------------------------------------------------------------------------
#activations/
#--------------------------------------------------------------------------

ACTIVATION_STATS = _schema([
    ("unit_id", pa.string()),
    ("calibration_run_id", pa.string()),
    ("observed_tokens", pa.int64()),
    ("active_count", pa.int64()),
    ("activation_rate", pa.float64()),
    ("mean", pa.float64()),
    ("std", pa.float64()),
    ("min", pa.float64()),
    ("max", pa.float64()),
    ("quantiles", _FLOAT_LIST),
    ("skewness", pa.float64()),
    ("kurtosis", pa.float64()),
    ("sparsity_score", pa.float64()),
    ("specificity_score", pa.float64()),
    ("burstiness_score", pa.float64()),
    #Per-unit active threshold (Issue 2) and active-bitset saturation (Issue 1).
    ("active_threshold", pa.float64()),
    ("bitset_density", pa.float64()),
])

ACTIVATION_HISTOGRAMS = _schema([
    ("unit_id", pa.string()),
    ("bin_left", _FLOAT_LIST),
    ("bin_right", _FLOAT_LIST),
    ("count", _INT_LIST),
])

ACTIVATION_TOP_EVENTS = _schema([
    ("unit_id", pa.string()),
    ("event_rank", pa.int32()),
    ("corpus_item_id", pa.int64()),
    ("sequence_id", pa.int64()),
    ("token_position", pa.int32()),
    ("token_id", pa.int64()),
    ("token_text", pa.string()),
    ("activation_value", pa.float64()),
])


#--------------------------------------------------------------------------
#probes/
#--------------------------------------------------------------------------

PROBE_INDEX = _schema([
    ("probe_id", pa.string()),
    ("probe_kind", pa.string()),
    ("layer_id", pa.int32()),
    ("source_ref", pa.string()),
    ("norm", pa.float64()),
])

PROBE_RESPONSES = _schema([
    ("probe_id", pa.string()),
    ("unit_id", pa.string()),
    ("response", pa.float64()),
    ("response_rank", pa.int32()),
])


#--------------------------------------------------------------------------
#graphs/
#--------------------------------------------------------------------------

def _unit_edge_schema(score_field: str) -> pa.Schema:
    return _schema([
        ("source_unit_id", pa.string()),
        ("target_unit_id", pa.string()),
        ("source_layer", pa.int32()),
        ("target_layer", pa.int32()),
        (score_field, pa.float64()),
    ])


UNIT_EDGES_ACTIVATION = _unit_edge_schema("activation_score")
UNIT_EDGES_STATIC_ALIGNMENT = _unit_edge_schema("static_alignment_score")
UNIT_EDGES_LAGGED = _unit_edge_schema("lagged_score")
UNIT_EDGES_ATTENTION_ROUTING = _unit_edge_schema("attention_routing_score")
UNIT_EDGES_PROBE_RESPONSE = _unit_edge_schema("probe_response_score")

UNIT_EDGES_COMBINED = _schema([
    ("source_unit_id", pa.string()),
    ("target_unit_id", pa.string()),
    ("source_layer", pa.int32()),
    ("target_layer", pa.int32()),
    ("activation_score", pa.float64()),
    ("lagged_score", pa.float64()),
    ("static_alignment_score", pa.float64()),
    ("attention_routing_score", pa.float64()),
    ("probe_response_score", pa.float64()),
    ("causal_score", pa.float64()),
    ("combined_score", pa.float64()),
    ("edge_confidence", pa.float64()),
])

CLUSTER_EDGES = _schema([
    ("source_cluster_id", pa.string()),
    ("target_cluster_id", pa.string()),
    ("edge_count", pa.int64()),
    ("mean_combined_score", pa.float64()),
    ("sum_combined_score", pa.float64()),
])

GRAPH_PARTITIONS = _schema([
    ("partition_id", pa.int64()),
    ("layer_min", pa.int32()),
    ("layer_max", pa.int32()),
    ("unit_count", pa.int64()),
    ("edge_count", pa.int64()),
])


#--------------------------------------------------------------------------
#clusters/
#--------------------------------------------------------------------------

CLUSTER_INDEX = _schema([
    ("cluster_id", pa.string()),
    ("cluster_level", pa.string()),
    ("cluster_type", pa.string()),
    ("parent_cluster_id", pa.string()),
    ("layer_min", pa.int32()),
    ("layer_max", pa.int32()),
    ("dominant_layer", pa.int32()),
    ("dominant_unit_type", pa.string()),
    ("member_count", pa.int64()),
    ("clustering_algorithm", pa.string()),
    ("resolution", pa.float64()),
    #True when a local cluster covers a suspiciously large share of its
    #(layer, unit_type) population — a sign of bridge-edge over-merging (Issue 6).
    ("giant_component_warning", pa.bool_()),
])

#Per-unit record of whether a unit was included in graph construction (Issue 7).
GRAPH_PARTICIPATION = _schema([
    ("unit_id", pa.string()),
    ("layer_id", pa.int32()),
    ("unit_type", pa.string()),
    ("included_in_graph", pa.bool_()),
])

#Single-row metadata describing how the graph was built (Issue 7).
GRAPH_META = _schema([
    ("graph_sampled", pa.bool_()),
    ("per_layer_cap", pa.int64()),
    ("layer_window", pa.int32()),
    ("candidate_generation_method", pa.string()),
    ("total_units", pa.int64()),
    ("included_units", pa.int64()),
])

CLUSTER_MEMBERSHIP = _schema([
    ("cluster_id", pa.string()),
    ("member_id", pa.string()),
    ("member_type", pa.string()),  # "unit" | "cluster"
    ("membership_weight", pa.float64()),
    ("membership_rank", pa.int32()),
])

CLUSTER_STATS = _schema([
    ("cluster_id", pa.string()),
    ("member_count", pa.int64()),
    ("mean_activation_rate", pa.float64()),
    ("mean_specificity", pa.float64()),
    ("mean_read_norm", pa.float64()),
    ("mean_write_norm", pa.float64()),
    ("fire_coherence", pa.float64()),
    ("structural_coherence", pa.float64()),
    ("source_score", pa.float64()),
    ("sink_score", pa.float64()),
    ("relay_score", pa.float64()),
    ("routing_score", pa.float64()),
    ("causal_confidence", pa.float64()),
])

CLUSTER_HIERARCHY = _schema([
    ("child_cluster_id", pa.string()),
    ("parent_cluster_id", pa.string()),
    ("child_level", pa.string()),
    ("parent_level", pa.string()),
])

CLUSTER_EXEMPLARS = _schema([
    ("cluster_id", pa.string()),
    ("unit_id", pa.string()),
    ("exemplar_rank", pa.int32()),
    ("centrality", pa.float64()),
])


#--------------------------------------------------------------------------
#summaries/
#--------------------------------------------------------------------------

HISTOGRAM_INDEX = _schema([
    ("histogram_id", pa.string()),
    ("entity_type", pa.string()),
    ("entity_id", pa.string()),
    ("metric_name", pa.string()),
    ("bin_count", pa.int32()),
    ("baseline_entity_type", pa.string()),
    ("baseline_entity_id", pa.string()),
])

HISTOGRAM_BINS = _schema([
    ("histogram_id", pa.string()),
    ("entity_type", pa.string()),
    ("entity_id", pa.string()),
    ("metric_name", pa.string()),
    ("bin_left", pa.float64()),
    ("bin_right", pa.float64()),
    ("count", pa.int64()),
    ("density", pa.float64()),
    ("baseline_entity_type", pa.string()),
    ("baseline_entity_id", pa.string()),
])

QUANTILE_SUMMARIES = _schema([
    ("entity_type", pa.string()),
    ("entity_id", pa.string()),
    ("metric_name", pa.string()),
    ("quantile_points", _FLOAT_LIST),
    ("quantile_values", _FLOAT_LIST),
])

TOPK_INDEX = _schema([
    ("entity_type", pa.string()),
    ("entity_id", pa.string()),
    ("relation", pa.string()),
    ("rank", pa.int32()),
    ("target_id", pa.string()),
    ("score", pa.float64()),
])

MATERIALIZED_CLUSTER_VIEWS = _schema([
    ("cluster_id", pa.string()),
    ("cluster_level", pa.string()),
    ("layer_min", pa.int32()),
    ("layer_max", pa.int32()),
    ("dominant_unit_type", pa.string()),
    ("member_count", pa.int64()),
    ("mean_activation_rate", pa.float64()),
    ("mean_write_norm", pa.float64()),
    ("source_score", pa.float64()),
    ("sink_score", pa.float64()),
    ("relay_score", pa.float64()),
    ("upstream_cluster_count", pa.int64()),
    ("downstream_cluster_count", pa.int64()),
])

BASELINE_COMPARISONS = _schema([
    ("entity_type", pa.string()),
    ("entity_id", pa.string()),
    ("metric_name", pa.string()),
    ("entity_value", pa.float64()),
    ("baseline_entity_type", pa.string()),
    ("baseline_entity_id", pa.string()),
    ("baseline_value", pa.float64()),
    ("z_score", pa.float64()),
    ("ratio", pa.float64()),
])


#Map of relative artifact path -> schema, for validation and the artifact index.
SCHEMA_REGISTRY: dict[str, pa.Schema] = {
    "catalog/tensor_index.parquet": TENSOR_INDEX,
    "catalog/logical_tensor_index.parquet": LOGICAL_TENSOR_INDEX,
    "catalog/unit_index.parquet": UNIT_INDEX,
    "catalog/unit_weight_refs.parquet": UNIT_WEIGHT_REFS,
    "catalog/artifact_index.parquet": ARTIFACT_INDEX,
    "tensors/tensor_stats.parquet": TENSOR_STATS,
    "tensors/tensor_histograms.parquet": TENSOR_HISTOGRAMS,
    "tensors/row_col_stats.parquet": ROW_COL_STATS,
    "tensors/singular_summaries.parquet": SINGULAR_SUMMARIES,
    "units/unit_static_stats.parquet": UNIT_STATIC_STATS,
    "units/unit_role_scores.parquet": UNIT_ROLE_SCORES,
    "calibration/corpus_index.parquet": CORPUS_INDEX,
    "calibration/token_index.parquet": TOKEN_INDEX,
    "calibration/token_group_stats.parquet": TOKEN_GROUP_STATS,
    "calibration/calibration_splits.parquet": CALIBRATION_SPLITS,
    "activations/activation_stats.parquet": ACTIVATION_STATS,
    "activations/activation_histograms.parquet": ACTIVATION_HISTOGRAMS,
    "activations/activation_top_events.parquet": ACTIVATION_TOP_EVENTS,
    "probes/probe_index.parquet": PROBE_INDEX,
    "probes/probe_responses.parquet": PROBE_RESPONSES,
    "graphs/unit_edges_activation.parquet": UNIT_EDGES_ACTIVATION,
    "graphs/unit_edges_static_alignment.parquet": UNIT_EDGES_STATIC_ALIGNMENT,
    "graphs/unit_edges_lagged.parquet": UNIT_EDGES_LAGGED,
    "graphs/unit_edges_attention_routing.parquet": UNIT_EDGES_ATTENTION_ROUTING,
    "graphs/unit_edges_probe_response.parquet": UNIT_EDGES_PROBE_RESPONSE,
    "graphs/unit_edges_combined.parquet": UNIT_EDGES_COMBINED,
    "graphs/cluster_edges.parquet": CLUSTER_EDGES,
    "graphs/graph_partitions.parquet": GRAPH_PARTITIONS,
    "graphs/graph_participation.parquet": GRAPH_PARTICIPATION,
    "graphs/graph_meta.parquet": GRAPH_META,
    "clusters/cluster_index.parquet": CLUSTER_INDEX,
    "clusters/cluster_membership.parquet": CLUSTER_MEMBERSHIP,
    "clusters/cluster_stats.parquet": CLUSTER_STATS,
    "clusters/cluster_hierarchy.parquet": CLUSTER_HIERARCHY,
    "clusters/cluster_exemplars.parquet": CLUSTER_EXEMPLARS,
    "summaries/histogram_index.parquet": HISTOGRAM_INDEX,
    "summaries/histogram_bins.parquet": HISTOGRAM_BINS,
    "summaries/quantile_summaries.parquet": QUANTILE_SUMMARIES,
    "summaries/topk_index.parquet": TOPK_INDEX,
    "summaries/materialized_cluster_views.parquet": MATERIALIZED_CLUSTER_VIEWS,
    "summaries/baseline_comparisons.parquet": BASELINE_COMPARISONS,
}

SCHEMA_VERSION = "1.0.0"
