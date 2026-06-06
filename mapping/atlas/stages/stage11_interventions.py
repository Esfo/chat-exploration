"""Stage 11 — targeted causal interventions (V2 scope).

The plan keeps causal evidence separate and treats it as optional, high-value
validation of the strongest inferred edges. The full engine (ablate/patch a
source unit or cluster, re-run the calibration batch, measure the change in a
target's activation) requires live-model forward passes and is V2 work.

These two commands provide the stable contract now:

  * ``plan-interventions`` selects the highest-confidence combined edges as
    intervention candidates and writes the plan.
  * ``run-interventions`` is the placeholder that will execute them and write
    ``causal_score`` back into the combined graph; for now it is a no-op that
    records its invocation so the pipeline stays resumable and auditable.
"""

from __future__ import annotations

from ..manifest import Library
from ..model_backend import ModelBackend
from ..storage import read_parquet
from . import register


@register("plan-interventions", 11,
          requires=["graphs/unit_edges_combined.parquet"],
          produces=["exports/report_inputs/intervention_plan.parquet"],
          version_scope="v2")
def plan(library: Library, backend: ModelBackend, top_n: int = 256, **kwargs):
    combined = read_parquet(library.path("graphs/unit_edges_combined.parquet")).to_pylist()
    combined.sort(key=lambda e: (e["edge_confidence"], e["combined_score"]), reverse=True)
    plan_rows = []
    for rank, e in enumerate(combined[:top_n]):
        plan_rows.append({
            "rank": rank, "source_unit_id": e["source_unit_id"],
            "target_unit_id": e["target_unit_id"],
            "combined_score": e["combined_score"],
            "edge_confidence": e["edge_confidence"],
        })
    #Written as a plain Parquet (no fixed schema in the registry) report input.
    from ..storage import write_parquet
    out = library.path("exports/report_inputs/intervention_plan.parquet")
    write_parquet(plan_rows, out)
    library.register_artifact("exports/report_inputs/intervention_plan.parquet",
                              "exports", "plan-interventions", row_count=len(plan_rows))
    library.log("plan-interventions", "intervention plan written", candidates=len(plan_rows))
    library.update_artifact_versions("plan-interventions")
    return {"candidates": len(plan_rows)}


@register("run-interventions", 11,
          requires=["exports/report_inputs/intervention_plan.parquet"],
          produces=[], version_scope="v2")
def run(library: Library, backend: ModelBackend, **kwargs):
    library.log("run-interventions",
                "no-op placeholder: live causal engine is V2 scope")
    library.update_artifact_versions("run-interventions")
    return {"executed": 0}
