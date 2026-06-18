from __future__ import annotations
import copy, json
from pathlib import Path
from typing import Any, Dict, List, Optional
from .probe_runner import run_privacy_probes
from .schema import ArtifactSpec, init_residual_vector

def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(data: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)

def get_selected_pipeline_id(mediator_result: Dict[str, Any]) -> Optional[str]:
    decision = mediator_result.get("decision") or {}
    return decision.get("selected_pipeline_id")

def find_candidate_by_id(mediator_result: Dict[str, Any], pipeline_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if pipeline_id is None:
        return None
    cands = (mediator_result.get("candidate_generation_result") or {}).get("candidates", [])
    for c in cands:
        if c.get("pipeline_id") == pipeline_id:
            return c
    for c in mediator_result.get("candidates", []) or []:
        if c.get("pipeline_id") == pipeline_id:
            return c
    for ev in mediator_result.get("evaluations", []) or []:
        if ev.get("pipeline_id") == pipeline_id:
            flow = ev.get("flow") or {}
            return {
                "pipeline_id": pipeline_id,
                "residual_disclosure": flow.get("residual_disclosure", {}),
                "matched_output_cap": ev.get("matched_output_cap"),
                "final_output_cap": flow.get("final_output_cap", {}),
            }
    return None

def load_artifact_manifest(path: str | Path) -> List[ArtifactSpec]:
    data = load_json(path)
    artifacts = data.get("artifacts", data if isinstance(data, list) else [])
    return [ArtifactSpec.from_dict(a) for a in artifacts]

def run_probes_for_mediator_result(
    mediator_result: Dict[str, Any],
    artifacts: List[ArtifactSpec] | List[Dict[str, Any]],
    probe_config: Optional[Dict[str, Any]] = None,
    combine_policy: str = "conservative_max",
) -> Dict[str, Any]:
    """Run probes after full_mediator/CI evaluation on the selected pipeline's artifacts."""
    pipeline_id = get_selected_pipeline_id(mediator_result)
    if not pipeline_id:
        return {
            "schema_version": "smartpriv_privacy_probe_stage_v1",
            "status": "skipped",
            "reason": "No selected pipeline in mediator result.",
            "selected_pipeline_id": None,
        }
    candidate = find_candidate_by_id(mediator_result, pipeline_id) or {}
    metadata_residual = candidate.get("residual_disclosure") or init_residual_vector("none")
    report = run_privacy_probes(
        artifacts=artifacts,
        pipeline_id=pipeline_id,
        metadata_residual=metadata_residual,
        probe_config=probe_config,
        combine_policy=combine_policy,
    )
    return {
        "schema_version": "smartpriv_privacy_probe_stage_v1",
        "status": "ok",
        "selected_pipeline_id": pipeline_id,
        "matched_output_cap": candidate.get("matched_output_cap"),
        "probe_report": report.to_dict(),
    }

def attach_probe_report_to_mediator_result(
    mediator_result: Dict[str, Any],
    probe_stage_result: Dict[str, Any],
) -> Dict[str, Any]:
    out = copy.deepcopy(mediator_result)
    out["privacy_probe_stage_result"] = probe_stage_result
    if probe_stage_result.get("status") == "ok":
        pipeline_id = probe_stage_result.get("selected_pipeline_id")
        combined = (probe_stage_result.get("probe_report") or {}).get("combined_residual")
        if combined:
            for ev in (out.get("ci_evaluation_result") or {}).get("evaluations", []):
                if ev.get("pipeline_id") == pipeline_id:
                    ev.setdefault("flow", {})["residual_disclosure_after_probes"] = combined
            for ev in out.get("evaluations", []) or []:
                if ev.get("pipeline_id") == pipeline_id:
                    ev.setdefault("flow", {})["residual_disclosure_after_probes"] = combined
    return out
