from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Planner records use contract IDs. Runtime registry uses the same IDs.
EXECUTABLE_OPERATOR_IDS = {
    "op.source",
    "op.sample",
    "op.window",
    "op.trigger_gate",
    "op.join_fuse",
    "op.route_publish",
    "op.schema_adapter",
    "op.person_object_detector",
    "op.pose_extractor",
    "op.ocr_screen_detector",
    "op.audio_level_extractor",
    "op.speech_sound_classifier",
    "op.keyword_intent_extractor",
    "op.region_select_crop",
    "op.region_mask_blur",
    "op.speech_content_removal",
    "op.occupancy_deriver",
    "op.activity_event_classifier",
    "op.aggregate_generalize",
    "op.drop_discard",
}


def executable_stage_from_operator_record(op: Dict[str, Any]) -> Dict[str, Any]:
    operator_id = op.get("operator")
    if operator_id not in EXECUTABLE_OPERATOR_IDS:
        return {"operator_id": operator_id, "unsupported": True, "parameters": op.get("parameters") or {}}
    params = dict(op.get("parameters") or {})
    # The symbolic planner materializes region-mask variants as {target: face}; the
    # runtime expects either target or targets. Keep both for convenience.
    if operator_id == "op.region_mask_blur" and "target" in params and "targets" not in params:
        params["targets"] = [params["target"]]
    if operator_id == "op.schema_adapter":
        out = op.get("output_cap") or {}
        params.setdefault("target_schema", out.get("schema"))
        params.setdefault("target_semantic_type", out.get("semantic_type") or out.get("media_type"))
    return {
        "operator_id": operator_id,
        "variant": op.get("variant"),
        "parameters": params,
        "symbolic_output_cap": op.get("output_cap"),
    }


def build_executable_pipeline_spec(candidate: Dict[str, Any]) -> Dict[str, Any]:
    stages = [executable_stage_from_operator_record(op) for op in candidate.get("operators", [])]
    unsupported = [s for s in stages if s.get("unsupported")]
    return {
        "schema_version": "smartpriv_executable_pipeline_spec_v1",
        "pipeline_id": candidate.get("pipeline_id"),
        "matched_output_cap": candidate.get("matched_output_cap"),
        "residual_score": candidate.get("residual_score"),
        "stages": stages,
        "unsupported_operator_ids": sorted({s.get("operator_id") for s in unsupported}),
        "notes": [
            "The first op.source stage is symbolic. Pass a DataItem to ExecutablePipeline.process().",
            "Generated specs are executable best-effort implementations of the operator contracts; optional ML libraries improve operator quality.",
        ],
    }


def attach_executable_specs(result: Dict[str, Any]) -> Dict[str, Any]:
    for c in result.get("candidates", []) or []:
        c["executable_pipeline_spec"] = build_executable_pipeline_spec(c)
    return result


def write_pipeline_program(candidate: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = build_executable_pipeline_spec(candidate)
    program = f'''#!/usr/bin/env python3
"""Executable SmartPriv pipeline generated from symbolic candidate {candidate.get('pipeline_id')}.

Usage:
  python {path.name} --input frame.jpg --output out.json
  python {path.name} --input audio.wav --output out.json --media-type audio/x-raw
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from smartpriv_runtime.data_model import dump_json_item
from smartpriv_runtime.media_io import item_from_media
from smartpriv_runtime.pipeline import ExecutablePipeline

PIPELINE_SPEC = {json.dumps(spec, indent=2)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run generated SmartPriv pipeline")
    parser.add_argument("--input", required=True, help="Input image/audio/video path")
    parser.add_argument("--media-type", default=None, help="Override media type, e.g. image/x-raw or audio/x-raw")
    parser.add_argument("--output", default="pipeline_output.json", help="Path for JSON summary output")
    parser.add_argument("--include-payload", action="store_true", help="Include JSON-serializable payloads in output when possible")
    args = parser.parse_args()

    item = item_from_media(args.input, media_type=args.media_type)
    pipe = ExecutablePipeline(PIPELINE_SPEC["stages"])
    result = pipe.process(item)
    dump_json_item(result, args.output, include_payload=args.include_payload)
    print(json.dumps({{"pipeline_id": PIPELINE_SPEC.get("pipeline_id"), "output": args.output, "dropped": result is None}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    path.write_text(program, encoding="utf-8")
    path.chmod(0o755)


def emit_programs(result: Dict[str, Any], out_dir: str | Path, top_k: int = 5) -> List[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for i, cand in enumerate((result.get("candidates") or [])[:top_k], start=1):
        filename = f"candidate_{i:02d}_{cand.get('pipeline_id', 'pipeline')}.py"
        path = out / filename
        write_pipeline_program(cand, path)
        written.append(str(path))
    return written
