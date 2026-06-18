from __future__ import annotations
import argparse, json
from typing import Any, Dict
from .mediator_integration import attach_probe_report_to_mediator_result, load_artifact_manifest, load_json, run_probes_for_mediator_result, write_json
from .probe_runner import run_privacy_probes

def read_optional_json(path: str | None) -> Dict[str, Any]:
    if not path: return {}
    return load_json(path)

def main() -> int:
    parser = argparse.ArgumentParser(description="Run privacy residual-disclosure probes.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("run", help="Run probes directly on an artifact manifest.")
    p1.add_argument("--artifacts", required=True, help="JSON manifest: {'artifacts': [{'path': ..., 'modality': ...}]}")
    p1.add_argument("--pipeline-id", default=None)
    p1.add_argument("--metadata-residual", default=None, help="Optional JSON residual vector from candidate metadata.")
    p1.add_argument("--probe-config", default=None)
    p1.add_argument("--out", required=True)
    p2 = sub.add_parser("mediator", help="Run probes after full_mediator/CI evaluation.")
    p2.add_argument("--mediator-result", required=True, help="full_mediator_result.json or ci_evaluation.json")
    p2.add_argument("--artifacts", required=True, help="Artifact manifest for transformed outputs.")
    p2.add_argument("--probe-config", default=None)
    p2.add_argument("--out", required=True, help="Probe stage output JSON.")
    p2.add_argument("--updated-mediator-result", default=None, help="Optional path to mediator result with probe report attached.")
    args = parser.parse_args()
    if args.cmd == "run":
        manifest = load_json(args.artifacts)
        artifacts = manifest.get("artifacts", manifest if isinstance(manifest, list) else [])
        metadata_residual = read_optional_json(args.metadata_residual) if args.metadata_residual else None
        probe_config = read_optional_json(args.probe_config)
        report = run_privacy_probes(artifacts=artifacts, pipeline_id=args.pipeline_id, metadata_residual=metadata_residual, probe_config=probe_config)
        write_json(report.to_dict(), args.out)
        print(json.dumps({"status": "ok", "pipeline_id": report.pipeline_id, "probe_residual": report.probe_residual, "combined_residual": report.combined_residual, "out": args.out}, indent=2))
        return 0
    if args.cmd == "mediator":
        mediator_result = load_json(args.mediator_result)
        artifacts = load_artifact_manifest(args.artifacts)
        probe_config = read_optional_json(args.probe_config)
        stage_result = run_probes_for_mediator_result(mediator_result=mediator_result, artifacts=artifacts, probe_config=probe_config)
        write_json(stage_result, args.out)
        if args.updated_mediator_result:
            updated = attach_probe_report_to_mediator_result(mediator_result, stage_result)
            write_json(updated, args.updated_mediator_result)
        print(json.dumps({"status": stage_result.get("status"), "selected_pipeline_id": stage_result.get("selected_pipeline_id"), "combined_residual": (stage_result.get("probe_report") or {}).get("combined_residual"), "out": args.out}, indent=2))
        return 0
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
