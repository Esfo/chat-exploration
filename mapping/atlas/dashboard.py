"""Streamlit dashboard for exploring a built Atlas library.

This is the interactive "front end" the warehouse is designed to support: it reads
only the committed artifacts (via the DuckDB views and the AtlasQuery helper) and
never touches the model or the extraction runtime. Launch it with:

    streamlit run atlas/dashboard.py -- --library /path/to/atlas_library

or use the ./atlasdashboard convenience script. The library path can also be set
in the sidebar or via ATLAS_LIBRARY_DIR.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt  # ships with streamlit
    _HAS_ALT = True
except Exception:  # noqa: BLE001
    _HAS_ALT = False

#Allow running both as `streamlit run atlas/dashboard.py` and as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.config import DEFAULT_LIBRARY_DIR  # noqa: E402
from atlas.query import AtlasQuery  # noqa: E402


def _default_library() -> str:
    return os.environ.get("ATLAS_LIBRARY_DIR") or str(DEFAULT_LIBRARY_DIR)


def _cli_library() -> str | None:
    #Args after `--` when launched via `streamlit run ... -- --library X`.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--library", default=None)
    known, _ = parser.parse_known_args()
    return known.library


@st.cache_resource(show_spinner=False)
def _connect(library_dir: str):
    """Open the library's DuckDB once and reuse it across reruns."""
    q = AtlasQuery(library_dir)
    con = q.duckdb()
    return q, con


@st.cache_data(show_spinner=False)
def _tables(library_dir: str) -> set[str]:
    _, con = _connect(library_dir)
    rows = con.execute("SELECT table_name FROM information_schema.tables").fetchall()
    return {r[0] for r in rows}


def _sql(library_dir: str, query: str, params: list | None = None) -> pd.DataFrame:
    """Run a query, returning an empty frame instead of raising on missing data."""
    _, con = _connect(library_dir)
    try:
        return con.execute(query, params or []).df()
    except Exception as e:  # noqa: BLE001 — surface as an empty result in the UI
        st.warning(f"Query failed: {e}")
        return pd.DataFrame()


def _has(tables: set[str], *names: str) -> bool:
    return all(n in tables for n in names)


def _hist(df: pd.DataFrame, col: str, title: str, bins: int = 40):
    """Altair histogram of a numeric column, falling back to a bar chart."""
    if df.empty or col not in df.columns:
        return
    if _HAS_ALT:
        chart = (alt.Chart(df.dropna(subset=[col]))
                 .mark_bar()
                 .encode(alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins), title=title),
                         alt.Y("count()", title="count"))
                 .properties(height=240))
        st.altair_chart(chart, use_container_width=True)
    else:
        st.bar_chart(df[col])


#======================================================================
# Pages
#======================================================================

def page_overview(lib: str, tables: set[str]):
    st.header("Overview")

    #Model / architecture card — these live in manifest JSON, not DuckDB views.
    q_obj, _ = _connect(lib)
    cols = st.columns(4)
    try:
        model = q_obj.library.read_json("manifest/model.json")
        cols[0].metric("Model", str(model.get("model_path", "—")).split("/")[-1])
        pc = model.get("approx_param_count")
        if pc:
            cols[1].metric("Parameters", f"{int(pc)/1e9:.1f}B")
    except Exception:  # noqa: BLE001
        pass
    try:
        arch = q_obj.library.read_json("manifest/architecture.json")
        cols[2].metric("Layers", int(arch.get("num_layers", 0)))
        cols[3].metric("Hidden", int(arch.get("hidden_size", 0)))
    except Exception:  # noqa: BLE001
        pass

    #Headline counts.
    st.subheader("Contents")
    counts = {
        "Units": ("unit_index", None),
        "Clusters": ("cluster_index", None),
        "Tokens captured": ("token_index", None),
        "Unit edges": ("unit_edges_combined", None),
        "Top firing events": ("activation_top_events", None),
    }
    mcols = st.columns(len(counts))
    for i, (label, (tbl, _)) in enumerate(counts.items()):
        if tbl in tables:
            n = _sql(lib, f"SELECT count(*) c FROM {tbl}")
            mcols[i].metric(label, int(n.iloc[0]["c"]) if not n.empty else 0)
        else:
            mcols[i].metric(label, "—")

    #Visual breakdowns.
    if "unit_index" in tables:
        st.subheader("Units per layer")
        perlayer = _sql(lib, "SELECT layer_id, count(*) units FROM unit_index "
                             "GROUP BY layer_id ORDER BY layer_id")
        if not perlayer.empty:
            st.bar_chart(perlayer.set_index("layer_id"))

    if _has(tables, "unit_index", "activation_stats"):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Activation-rate distribution")
            ar = _sql(lib, "SELECT activation_rate FROM activation_stats "
                           "WHERE activation_rate IS NOT NULL")
            _hist(ar, "activation_rate", "activation rate")
        with c2:
            st.subheader("Mean activation rate by layer")
            mbl = _sql(lib, "SELECT u.layer_id, avg(a.activation_rate) mean_rate "
                            "FROM unit_index u JOIN activation_stats a USING (unit_id) "
                            "GROUP BY u.layer_id ORDER BY u.layer_id")
            if not mbl.empty:
                st.line_chart(mbl.set_index("layer_id"))

    #Cluster levels breakdown.
    if "cluster_index" in tables:
        st.subheader("Clusters by level")
        df = _sql(lib, "SELECT cluster_level, count(*) n FROM cluster_index "
                       "GROUP BY cluster_level ORDER BY n DESC")
        if not df.empty:
            st.bar_chart(df.set_index("cluster_level"))

    #Recent run log (JSONL on disk, not a DuckDB view).
    log_path = q_obj.library.path("manifest/run_log.jsonl")
    if log_path.exists():
        with st.expander("Run log (latest 25)"):
            try:
                logdf = pd.read_json(log_path, lines=True)
                st.dataframe(logdf.tail(25).iloc[::-1], use_container_width=True)
            except Exception as e:  # noqa: BLE001
                st.caption(f"Could not read run log: {e}")


def page_clusters(lib: str, tables: set[str]):
    st.header("Clusters")
    if "cluster_index" not in tables:
        st.info("No cluster_index in this library.")
        return

    levels = _sql(lib, "SELECT DISTINCT cluster_level FROM cluster_index")
    level = st.selectbox("Level", ["(all)"] + sorted(levels["cluster_level"].tolist())
                         if not levels.empty else ["(all)"])
    where = "" if level == "(all)" else f"WHERE c.cluster_level = '{level}'"

    #Join index with stats (roles) when available.
    if "cluster_stats" in tables:
        q = f"""
            SELECT c.cluster_id, c.cluster_level, c.dominant_unit_type, c.member_count,
                   c.layer_min, c.layer_max,
                   s.mean_activation_rate, s.mean_specificity,
                   s.source_score, s.sink_score, s.relay_score, s.routing_score,
                   s.structural_coherence
            FROM cluster_index c LEFT JOIN cluster_stats s USING (cluster_id)
            {where}
            ORDER BY c.member_count DESC LIMIT 500
        """
    else:
        q = f"SELECT * FROM cluster_index c {where} ORDER BY member_count DESC LIMIT 500"
    df = _sql(lib, q)

    #Role landscape: source vs sink, sized by membership, colored by level.
    if _HAS_ALT and _has(tables, "cluster_stats") and \
            {"source_score", "sink_score"}.issubset(df.columns) and not df.empty:
        st.subheader("Cluster role landscape")
        st.caption("Right = pushes signal out (source) · Up = receives (sink) · "
                   "size = members")
        scatter = (alt.Chart(df.dropna(subset=["source_score", "sink_score"]))
                   .mark_circle(opacity=0.6)
                   .encode(x=alt.X("source_score:Q", title="source score"),
                           y=alt.Y("sink_score:Q", title="sink score"),
                           size=alt.Size("member_count:Q", title="members"),
                           color=alt.Color("cluster_level:N", title="level"),
                           tooltip=["cluster_id", "dominant_unit_type", "member_count",
                                    "source_score", "sink_score", "relay_score"])
                   .interactive().properties(height=360))
        st.altair_chart(scatter, use_container_width=True)

    st.caption(f"{len(df)} clusters (top 500 by size)")
    st.dataframe(df, use_container_width=True, height=300)

    if df.empty:
        return
    cid = st.selectbox("Inspect cluster", df["cluster_id"].tolist())
    if not cid:
        return

    q_obj, _ = _connect(lib)

    #Role profile of the selected cluster.
    if "cluster_stats" in tables:
        roles = _sql(lib, "SELECT source_score, sink_score, relay_score, routing_score "
                          "FROM cluster_stats WHERE cluster_id = ?", [cid])
        if not roles.empty:
            st.subheader("Role profile")
            prof = roles.iloc[0].rename({"source_score": "source", "sink_score": "sink",
                                         "relay_score": "relay",
                                         "routing_score": "routing"})
            st.bar_chart(prof)

    left, right = st.columns(2)
    with left:
        st.subheader("Members")
        members = q_obj.units_in_cluster(cid)
        st.caption(f"{len(members)} units")
        st.dataframe(pd.DataFrame({"unit_id": members}), use_container_width=True, height=260)
    with right:
        st.subheader("Signal flow")
        up = q_obj.upstream_clusters(cid)
        down = q_obj.downstream_clusters(cid)
        st.write("**Upstream (feeds in):**")
        st.dataframe(pd.DataFrame(up, columns=["cluster_id", "score"]).head(15),
                     use_container_width=True)
        st.write("**Downstream (feeds out):**")
        st.dataframe(pd.DataFrame(down, columns=["cluster_id", "score"]).head(15),
                     use_container_width=True)


def page_units(lib: str, tables: set[str]):
    st.header("Units")
    if "unit_index" not in tables:
        st.info("No unit_index in this library.")
        return

    c1, c2, c3 = st.columns(3)
    layers = _sql(lib, "SELECT DISTINCT layer_id FROM unit_index ORDER BY layer_id")
    layer = c1.selectbox("Layer", ["(all)"] + layers["layer_id"].tolist()
                         if not layers.empty else ["(all)"])
    utype = c2.selectbox("Type", ["(all)", "mlp_neuron", "attn_head"])
    sort_by = c3.selectbox("Rank by", ["activation_rate", "specificity_score",
                                       "burstiness_score", "mean", "std"])

    conds = []
    if layer != "(all)":
        conds.append(f"u.layer_id = {layer}")
    if utype != "(all)":
        conds.append(f"u.unit_type = '{utype}'")
    where = ("WHERE " + " AND ".join(conds)) if conds else ""

    if "activation_stats" in tables:
        q = f"""
            SELECT u.unit_id, u.layer_id, u.unit_type,
                   a.activation_rate, a.specificity_score, a.burstiness_score,
                   a.mean, a.std, a.observed_tokens
            FROM unit_index u LEFT JOIN activation_stats a USING (unit_id)
            {where}
            ORDER BY a.{sort_by} DESC NULLS LAST
            LIMIT 500
        """
    else:
        q = f"SELECT unit_id, layer_id, unit_type FROM unit_index u {where} LIMIT 500"
    df = _sql(lib, q)
    st.caption(f"Showing up to 500 units, ranked by {sort_by}")
    if sort_by in df.columns:
        st.subheader(f"Distribution of {sort_by} (shown units)")
        _hist(df, sort_by, sort_by)
    st.dataframe(df, use_container_width=True, height=300)

    if df.empty:
        return
    uid = st.selectbox("Inspect unit", df["unit_id"].tolist())
    if not uid:
        return
    _unit_detail(lib, tables, uid)


def _unit_detail(lib: str, tables: set[str], uid: str):
    st.subheader(f"Unit `{uid}`")
    q_obj, _ = _connect(lib)

    if "activation_stats" in tables:
        stats = _sql(lib, "SELECT * FROM activation_stats WHERE unit_id = ?", [uid])
        if not stats.empty:
            row = stats.iloc[0]
            m = st.columns(4)
            m[0].metric("Activation rate", f"{row.get('activation_rate', 0):.3f}")
            m[1].metric("Specificity", f"{row.get('specificity_score', 0):.3f}")
            m[2].metric("Burstiness", f"{row.get('burstiness_score', 0):.3f}")
            m[3].metric("Observed tokens", int(row.get("observed_tokens", 0)))

    #Per-unit activation histogram (stored as arrays in activation_histograms).
    if "activation_histograms" in tables:
        h = _sql(lib, "SELECT bin_left, bin_right, count FROM activation_histograms "
                      "WHERE unit_id = ?", [uid])
        if not h.empty:
            try:
                row = h.iloc[0]
                centers = [(a + b) / 2 for a, b in zip(row["bin_left"], row["bin_right"])]
                hist_df = pd.DataFrame({"activation": centers, "count": list(row["count"])})
                st.write("**Activation value distribution**")
                if _HAS_ALT:
                    chart = (alt.Chart(hist_df).mark_bar()
                             .encode(x=alt.X("activation:Q", title="activation value"),
                                     y=alt.Y("count:Q"))
                             .properties(height=220))
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.bar_chart(hist_df.set_index("activation"))
            except Exception as e:  # noqa: BLE001
                st.caption(f"Could not render histogram: {e}")

    left, right = st.columns(2)
    with left:
        st.write("**Top firing tokens**")
        if "activation_top_events" in tables:
            ev = _sql(lib, "SELECT event_rank, token_text, activation_value, "
                           "sequence_id, token_position FROM activation_top_events "
                           "WHERE unit_id = ? ORDER BY event_rank", [uid])
            st.dataframe(ev, use_container_width=True, height=260)
    with right:
        st.write("**Similar units (by firing signature)**")
        sim = q_obj.similar_units(uid, k=15)
        st.dataframe(pd.DataFrame(sim, columns=["unit_id", "similarity"]),
                     use_container_width=True, height=260)

    with st.expander("Connections (combined edge graph)"):
        if "unit_edges_combined" in tables:
            edges = _sql(lib,
                         "SELECT source_unit_id, target_unit_id, combined_score, "
                         "edge_confidence FROM unit_edges_combined "
                         "WHERE source_unit_id = ? OR target_unit_id = ? "
                         "ORDER BY combined_score DESC LIMIT 50", [uid, uid])
            st.dataframe(edges, use_container_width=True)

    with st.expander("Checkpoint tensor slices (where this unit lives in the weights)"):
        st.dataframe(pd.DataFrame(q_obj.tensor_slices_for_unit(uid)),
                     use_container_width=True)


def page_graph(lib: str, tables: set[str]):
    st.header("Connection graph")
    if "unit_edges_combined" not in tables:
        st.info("No unit_edges_combined in this library.")
        return
    #Layer-to-layer signal flow heatmap.
    flow = _sql(lib, "SELECT source_layer, target_layer, count(*) edges, "
                     "sum(combined_score) score FROM unit_edges_combined "
                     "GROUP BY source_layer, target_layer")
    if not flow.empty:
        st.subheader("Layer-to-layer signal flow")
        if _HAS_ALT:
            heat = (alt.Chart(flow).mark_rect()
                    .encode(x=alt.X("source_layer:O", title="source layer"),
                            y=alt.Y("target_layer:O", title="target layer"),
                            color=alt.Color("score:Q", title="Σ combined score",
                                            scale=alt.Scale(scheme="magma")),
                            tooltip=["source_layer", "target_layer", "edges", "score"])
                    .properties(height=420))
            st.altair_chart(heat, use_container_width=True)
        else:
            st.dataframe(flow, use_container_width=True)

    st.subheader("Top edges")
    min_conf = st.slider("Minimum edge confidence", 0.0, 1.0, 0.0, 0.05)
    df = _sql(lib, "SELECT source_unit_id, target_unit_id, source_layer, target_layer, "
                   "combined_score, edge_confidence FROM unit_edges_combined "
                   "WHERE edge_confidence >= ? ORDER BY combined_score DESC LIMIT 500",
              [min_conf])
    st.caption(f"Top {len(df)} edges by combined score (confidence ≥ {min_conf})")
    st.dataframe(df, use_container_width=True, height=420)
    if "cluster_edges" in tables:
        with st.expander("Cluster-level edges"):
            st.dataframe(_sql(lib, "SELECT * FROM cluster_edges "
                                   "ORDER BY sum_combined_score DESC LIMIT 200"),
                         use_container_width=True)


#----------------------------------------------------------------------
# Plot Lab: freeform chart builder over any dataset
#----------------------------------------------------------------------

#Pre-joined, denormalized datasets so you can cross fields that live in
#different tables (e.g. unit weights vs. their cluster). Each entry is
#(SQL, required_tables). Raw tables are added on top at runtime.
def _datasets(tables: set[str]) -> dict[str, str]:
    ds: dict[str, str] = {}
    if _has(tables, "unit_index"):
        joins = "FROM unit_index u"
        cols = ["u.unit_id", "u.layer_id", "u.unit_type"]
        if "unit_static_stats" in tables:
            cols += ["s.read_norm", "s.write_norm", "s.read_kurtosis", "s.write_kurtosis",
                     "s.weight_tail_ratio", "s.static_outlier_score"]
            joins += " LEFT JOIN unit_static_stats s USING (unit_id)"
        if "activation_stats" in tables:
            cols += ["a.activation_rate", "a.specificity_score", "a.burstiness_score",
                     "a.mean AS act_mean", "a.std AS act_std", "a.observed_tokens"]
            joins += " LEFT JOIN activation_stats a USING (unit_id)"
        if "cluster_membership" in tables:
            cols += ["cm.cluster_id"]
            joins += (" LEFT JOIN cluster_membership cm "
                      "ON cm.member_id = u.unit_id AND cm.member_type = 'unit'")
        ds["units — catalog + weights + activations + cluster"] = \
            f"SELECT {', '.join(cols)} {joins}"
    if _has(tables, "cluster_index", "cluster_stats"):
        ds["clusters — index + stats"] = (
            "SELECT c.*, s.* EXCLUDE (cluster_id, member_count) "
            "FROM cluster_index c LEFT JOIN cluster_stats s USING (cluster_id)")
    if "unit_edges_combined" in tables:
        ds["edges — combined graph"] = "SELECT * FROM unit_edges_combined"
    if "tensor_stats" in tables:
        ds["tensors — weight statistics"] = "SELECT * FROM tensor_stats"
    return ds


_NUMERIC_HINTS = ("INT", "FLOAT", "DOUBLE", "DECIMAL", "REAL", "HUGEINT")


def _schema(lib: str, sql: str) -> dict[str, bool]:
    """Return {column: is_numeric}, skipping list/struct columns."""
    _, con = _connect(lib)
    try:
        desc = con.execute(f"DESCRIBE SELECT * FROM ({sql}) _t").df()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for _, r in desc.iterrows():
        t = str(r["column_type"]).upper()
        if "[]" in t or "STRUCT" in t or "MAP" in t or "LIST" in t:
            continue  # not plottable directly
        out[r["column_name"]] = any(h in t for h in _NUMERIC_HINTS)
    return out


def page_plotlab(lib: str, tables: set[str]):
    st.header("Plot Lab")
    st.caption("Build any chart from any data: pick a dataset, columns, chart type, "
               "filters, and aggregation.")

    datasets = _datasets(tables)
    #Offer the curated joins first, then every raw table/view.
    options = list(datasets) + [f"raw: {t}" for t in sorted(tables)]
    choice = st.selectbox("Dataset", options)
    base_sql = datasets[choice] if choice in datasets else f"SELECT * FROM {choice[5:]}"
    st.caption("Tip: to histogram per-unit values (e.g. weights) within clusters, "
               "pick **units — …** (one row per unit). The **clusters** dataset has "
               "only one row per cluster, so it can't form a distribution.")

    schema = _schema(lib, base_sql)
    if not schema:
        st.warning("Could not read this dataset's columns.")
        return
    cols = list(schema)
    numeric = [c for c in cols if schema[c]]
    none = "(none)"

    #--- filters --------------------------------------------------------
    with st.expander("Filters", expanded=False):
        n_filters = st.number_input("How many filters", 0, 5, 0)
        wheres = []
        for i in range(int(n_filters)):
            f1, f2, f3 = st.columns([2, 1, 2])
            col = f1.selectbox("Column", cols, key=f"fc{i}")
            op = f2.selectbox("Op", ["=", "!=", ">", ">=", "<", "<=", "contains"],
                              key=f"fo{i}")
            val = f3.text_input("Value", key=f"fv{i}")
            if col and val != "":
                if op == "contains":
                    wheres.append(f"CAST({col} AS VARCHAR) ILIKE '%{val}%'")
                else:
                    lit = val if schema.get(col) and _is_number(val) else f"'{val}'"
                    wheres.append(f"{col} {op} {lit}")
    where_sql = (" WHERE " + " AND ".join(wheres)) if wheres else ""

    #--- chart spec -----------------------------------------------------
    chart_type = st.selectbox(
        "Chart type", ["Histogram", "Scatter", "Bar", "Line", "Box", "Heatmap", "Table"])

    #Histogram manages its own (full, per-group) querying — see _histogram.
    if chart_type == "Histogram":
        _histogram(lib, base_sql, where_sql, schema, none)
        return

    limit = st.slider("Max rows to load", 1000, 200000, 20000, step=1000)

    #--- fetch ----------------------------------------------------------
    sql = f"SELECT * FROM ({base_sql}) _t{where_sql} LIMIT {int(limit)}"
    with st.expander("Generated SQL"):
        st.code(sql, language="sql")
    df = _sql(lib, sql)
    st.caption(f"{len(df)} rows loaded")
    if df.empty:
        return

    #Classify from the actual frame (more reliable than the schema probe).
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    all_cols = list(df.columns)

    #--- render ---------------------------------------------------------
    if chart_type == "Table" or not _HAS_ALT:
        st.dataframe(df, use_container_width=True, height=500)
    else:
        enc: dict[str, str] = {}
        c1, c2, c3, c4 = st.columns(4)
        if chart_type == "Scatter":
            enc["x"] = c1.selectbox("X", numeric_cols or all_cols)
            enc["y"] = c2.selectbox("Y", numeric_cols or all_cols)
            enc["color"] = c3.selectbox("Color", [none] + all_cols)
            enc["size"] = c4.selectbox("Size", [none] + numeric_cols)
        elif chart_type in ("Bar", "Line"):
            enc["x"] = c1.selectbox("X (category/axis)", all_cols)
            enc["agg"] = c2.selectbox("Aggregate",
                                      ["count", "mean", "median", "sum", "min", "max"])
            enc["y"] = c3.selectbox("Y (value)", [none] + numeric_cols)
            enc["color"] = c4.selectbox("Color", [none] + all_cols)
        elif chart_type == "Box":
            enc["x"] = c1.selectbox("Category (X)", all_cols)
            enc["y"] = c2.selectbox("Value (Y)", numeric_cols or all_cols)
        elif chart_type == "Heatmap":
            enc["x"] = c1.selectbox("X", all_cols)
            enc["y"] = c2.selectbox("Y", all_cols)
            enc["agg"] = c3.selectbox("Color = aggregate", ["count", "mean", "sum", "max"])
            enc["cval"] = c4.selectbox("of (value)", [none] + numeric_cols)
        chart = _build_chart(df, chart_type, enc, none)
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)

    st.download_button("Download CSV", df.to_csv(index=False), "atlas_plotlab.csv",
                       "text/csv")


def _histogram(lib, base_sql, where_sql, schema, none):
    """Overlaid per-group histograms that query the FULL data per chosen group.

    Unlike the other chart types (which plot a capped sample), this pulls every
    row for the selected groups so per-cluster distributions are complete, and it
    excludes the unassigned (null) group by default so it doesn't swamp the rest.
    """
    numeric = [c for c, isnum in schema.items() if isnum]
    allcols = list(schema)
    if not numeric:
        st.warning("This dataset has no numeric columns to bin. For weight "
                   "distributions pick the **units — …** dataset and X = "
                   "**write_norm** or **read_norm** (those are the weight magnitudes; "
                   "*_kurtosis is a shape stat, not a weight).")
        return

    c1, c2 = st.columns(2)
    x = c1.selectbox("Value to bin (X axis)", numeric)
    group = c2.selectbox("Separate histogram per … (color)", [none] + allcols)

    o1, o2, o3, o4 = st.columns(4)
    nbins = o1.slider("Bins", 5, 250, 40)
    binwidth = o2.number_input("Exact bin width (0 = auto)", min_value=0.0, value=0.0)
    opacity = o3.slider("Opacity", 0.1, 1.0, 0.6, 0.05)
    log_y = o4.checkbox("Log Y", value=False)
    norm = st.radio("Y axis", ["count", "density (normalized per group)"],
                    horizontal=True)
    show_kde = st.checkbox("Overlay smooth KDE curve", value=False)

    #--- choose which groups (from the FULL data, not a sample) ----------
    chosen: list[str] = []
    include_null = False
    if group != none:
        gv = _sql(lib, f"SELECT CAST({group} AS VARCHAR) g, count(*) n "
                       f"FROM ({base_sql}) _t{where_sql} GROUP BY 1 ORDER BY n DESC "
                       f"LIMIT 3000")
        if gv.empty:
            st.warning("No data for this dataset.")
            return
        if int(gv["n"].max()) <= 1:
            st.warning(
                f"Each value of **{group}** has only one row, so there's no "
                f"within-group distribution to histogram.\n\n"
                f"• To compare clusters by an **aggregate** weight, use the "
                f"**clusters** dataset, set color to **(none)**, and pick "
                f"X = **mean_write_norm** (or mean_read_norm).\n"
                f"• For the weight distribution of the **units inside** clusters, use "
                f"the **units** dataset (X = write_norm, color = cluster_id).")
            return
        opts = [g for g in gv.dropna(subset=["g"])["g"].tolist()
                if g not in ("None", "nan")]
        chosen = st.multiselect(
            f"Which {group} values to overlay ({len(opts)} available, largest first)",
            opts, default=opts[: min(6, len(opts))])
        include_null = st.checkbox("include unassigned (null) group", value=False)
        if not chosen and not include_null:
            st.info("Pick at least one group to plot.")
            return

    #--- fetch full rows for those groups -------------------------------
    conds = [f"{x} IS NOT NULL"]
    params: list = []
    select_cols = x
    if group != none:
        select_cols += f", CAST({group} AS VARCHAR) AS __g"
        gsel = []
        if chosen:
            ph = ",".join(["?"] * len(chosen))
            gsel.append(f"CAST({group} AS VARCHAR) IN ({ph})")
            params += chosen
        if include_null:
            gsel.append(f"{group} IS NULL")
        conds.append("(" + " OR ".join(gsel) + ")")
    extra = (" AND " if where_sql else " WHERE ") + " AND ".join(conds)
    q = f"SELECT {select_cols} FROM ({base_sql}) _t{where_sql}{extra} LIMIT 500000"
    with st.expander("Generated SQL"):
        st.code(q, language="sql")
    data = _sql(lib, q, params)
    if data.empty:
        st.warning("No rows matched.")
        return
    data[x] = pd.to_numeric(data[x], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[x])
    if data.empty:
        st.warning("Selected column has no finite numeric values.")
        return
    gcol = "__g" if group != none else None
    if gcol:
        data["__g"] = data["__g"].fillna("(unassigned)")
    st.caption(f"{len(data):,} rows · "
               f"{data['__g'].nunique() if gcol else 1} distribution(s)")

    #--- render (Altair native binning, overlaid via stack=None) --------
    bin_opts = alt.Bin(step=binwidth) if binwidth and binwidth > 0 \
        else alt.Bin(maxbins=int(nbins))
    yscale = alt.Scale(type="log") if log_y else alt.Scale()
    color_enc = alt.Color("__g:N", title=group) if gcol else alt.value("#4C78A8")
    density = not norm.startswith("count")

    if density:
        gb = ["__bin"] + ([gcol] if gcol else [])
        base = (alt.Chart(data)
                .transform_bin("__bin", field=x, bin=bin_opts)
                .transform_aggregate(__c="count()", groupby=gb)
                .transform_joinaggregate(__t="sum(__c)",
                                         groupby=([gcol] if gcol else []))
                .transform_calculate(__frac="datum.__c / datum.__t"))
        tt = [alt.Tooltip("__bin:Q", title=x), alt.Tooltip("__frac:Q", format=".3f")]
        if gcol:
            tt.append(alt.Tooltip("__g:N", title=group))
        bars = base.mark_bar(opacity=opacity).encode(
            x=alt.X("__bin:Q", title=x), x2="__bin_end:Q",
            y=alt.Y("__frac:Q", stack=None, scale=yscale, title="fraction per group"),
            color=color_enc, tooltip=tt)
    else:
        tt = [alt.Tooltip(f"{x}:Q", bin=bin_opts, title=x),
              alt.Tooltip("count():Q", title="count")]
        if gcol:
            tt.append(alt.Tooltip("__g:N", title=group))
        bars = alt.Chart(data).mark_bar(opacity=opacity).encode(
            x=alt.X(f"{x}:Q", bin=bin_opts, title=x),
            y=alt.Y("count():Q", stack=None, scale=yscale, title="count"),
            color=color_enc, tooltip=tt)

    layers = [bars]
    if show_kde:
        lo, hi = float(data[x].min()), float(data[x].max())
        if hi <= lo:
            hi = lo + 1.0
        width = binwidth if (binwidth and binwidth > 0) else (hi - lo) / max(nbins, 1)
        grid = np.linspace(lo, hi, 256)
        krows = []
        giter = data.groupby("__g") if gcol else [("all", data)]
        for gname, gdf in giter:
            vals = gdf[x].to_numpy()
            if vals.size < 5:
                continue  # too few points for a meaningful curve
            dens = _kde(vals, grid, 1.0)
            if dens is None:
                continue
            scale = width if density else vals.size * width
            for gx, gy in zip(grid, dens * scale):
                krows.append({"x": float(gx), "value": float(gy), "__g": str(gname)})
        if krows:
            kdf = pd.DataFrame(krows)
            if log_y:
                kdf = kdf[kdf["value"] > 0]
            kcolor = alt.Color("__g:N", title=group) if gcol else alt.value("#E45756")
            layers.append(alt.Chart(kdf).mark_line(strokeWidth=2).encode(
                x="x:Q", y=alt.Y("value:Q", scale=yscale), color=kcolor))

    st.altair_chart(alt.layer(*layers).resolve_scale(color="shared")
                    .properties(height=480).interactive(), use_container_width=True)
    st.download_button("Download CSV", data.to_csv(index=False),
                       "atlas_histogram.csv", "text/csv")


def _kde(values, grid, bw_mult):
    """Gaussian KDE on a grid (Silverman bandwidth × bw_mult), no scipy needed."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return None
    std = values.std(ddof=1)
    if std == 0:
        return None
    h = bw_mult * 1.06 * std * n ** (-1 / 5)
    if h <= 0:
        return None
    u = (grid[:, None] - values[None, :]) / h
    return np.exp(-0.5 * u * u).sum(axis=1) / (n * h * np.sqrt(2 * np.pi))


def _build_chart(df, chart_type, enc, none):
    """Translate the chart spec into an Altair chart."""
    try:
        if chart_type == "Histogram":
            e = {"x": alt.X(enc["x"], bin=alt.Bin(maxbins=50), type="quantitative"),
                 "y": alt.Y("count()", title="count")}
            if enc.get("color") and enc["color"] != none:
                e["color"] = alt.Color(enc["color"] + ":N")
            return alt.Chart(df).mark_bar(opacity=0.75).encode(**e).properties(height=380)

        if chart_type == "Scatter":
            e = {"x": alt.X(enc["x"], type="quantitative"),
                 "y": alt.Y(enc["y"], type="quantitative"),
                 "tooltip": list(df.columns[:8])}
            if enc.get("color") and enc["color"] != none:
                e["color"] = alt.Color(enc["color"] + ":N")
            if enc.get("size") and enc["size"] != none:
                e["size"] = alt.Size(enc["size"] + ":Q")
            return alt.Chart(df).mark_circle(opacity=0.6).encode(**e) \
                .interactive().properties(height=420)

        if chart_type in ("Bar", "Line"):
            agg = enc["agg"]
            yfield = "count()" if agg == "count" or enc.get("y") in (none, None) \
                else f"{agg}({enc['y']})"
            e = {"x": alt.X(enc["x"] + ":N" if chart_type == "Bar" else enc["x"]),
                 "y": alt.Y(yfield)}
            if enc.get("color") and enc["color"] != none:
                e["color"] = alt.Color(enc["color"] + ":N")
            mark = alt.Chart(df).mark_bar() if chart_type == "Bar" else alt.Chart(df).mark_line(point=True)
            return mark.encode(**e).properties(height=400)

        if chart_type == "Box":
            return alt.Chart(df).mark_boxplot().encode(
                x=alt.X(enc["x"] + ":N"), y=alt.Y(enc["y"] + ":Q")).properties(height=400)

        if chart_type == "Heatmap":
            agg = enc["agg"]
            cval = "count()" if agg == "count" or enc.get("cval") in (none, None) \
                else f"{agg}({enc['cval']})"
            return alt.Chart(df).mark_rect().encode(
                x=alt.X(enc["x"] + ":O"), y=alt.Y(enc["y"] + ":O"),
                color=alt.Color(cval, scale=alt.Scale(scheme="magma")),
                tooltip=[enc["x"], enc["y"]]).properties(height=420)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not build chart: {e}")
    return None


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def page_sql(lib: str, tables: set[str]):
    st.header("SQL console")
    st.caption("Query any view directly. Tables: " + ", ".join(sorted(tables)))
    default = "SELECT * FROM cluster_stats ORDER BY member_count DESC LIMIT 20"
    query = st.text_area("SQL", value=default, height=120)
    if st.button("Run", type="primary"):
        df = _sql(lib, query)
        st.caption(f"{len(df)} rows")
        st.dataframe(df, use_container_width=True, height=500)
        if not df.empty:
            st.download_button("Download CSV", df.to_csv(index=False),
                               "atlas_query.csv", "text/csv")


#======================================================================
# Main
#======================================================================

def main():
    st.set_page_config(page_title="Atlas Explorer", layout="wide")
    st.title("🧭 Atlas Explorer")

    default_lib = _cli_library() or _default_library()
    lib = st.sidebar.text_input("Library directory", value=default_lib)
    if not lib or not Path(lib).exists():
        st.warning(f"Library not found at `{lib}`. Set the path in the sidebar.")
        st.stop()
    if not (Path(lib) / "indexes" / "atlas.duckdb").exists():
        st.warning(f"No `indexes/atlas.duckdb` under `{lib}`. "
                   f"Run `build-indexes` (or the full pipeline) first.")
        st.stop()

    tables = _tables(lib)
    page = st.sidebar.radio("View", ["Overview", "Plot Lab", "Clusters", "Units",
                                     "Connection graph", "SQL console"])
    st.sidebar.caption(f"{len(tables)} tables available")

    if page == "Overview":
        page_overview(lib, tables)
    elif page == "Plot Lab":
        page_plotlab(lib, tables)
    elif page == "Clusters":
        page_clusters(lib, tables)
    elif page == "Units":
        page_units(lib, tables)
    elif page == "Connection graph":
        page_graph(lib, tables)
    else:
        page_sql(lib, tables)


if __name__ == "__main__":
    main()
