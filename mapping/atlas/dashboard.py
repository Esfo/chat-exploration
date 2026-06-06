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

import pandas as pd
import streamlit as st

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
    st.caption(f"{len(df)} clusters (top 500 by size)")
    st.dataframe(df, use_container_width=True, height=300)

    if df.empty:
        return
    cid = st.selectbox("Inspect cluster", df["cluster_id"].tolist())
    if not cid:
        return

    q_obj, _ = _connect(lib)
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
    page = st.sidebar.radio("View", ["Overview", "Clusters", "Units",
                                     "Connection graph", "SQL console"])
    st.sidebar.caption(f"{len(tables)} tables available")

    if page == "Overview":
        page_overview(lib, tables)
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
