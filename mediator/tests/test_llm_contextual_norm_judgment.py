#!/usr/bin/env python3
"""
test_llm_contextual_norm_judgment.py

Integration-style test for the LLM-assisted norm-judgment branch of
contextual_integrity_evaluator.py.

This test bypasses candidate generation and creates two synthetic candidate
pipeline outputs for the same home HVAC/occupancy task:

  1. minimized_occupancy_count:
       final output is an occupancy count with aggregate_only/data_minimized terms.
       Expected normative direction: acceptable or acceptable_with_mitigations.

  2. raw_video_for_hvac:
       final output is raw video for low-stakes HVAC automation.
       Expected normative direction: uncertain or inappropriate.

The hard constraints are still loaded and applied first. The scenario is a
private home (not short-term rental, not hospital, not school), so the hard
constraints should generally pass, forcing the evaluator to exercise the LLM
norm-judgment path.

The script prints each LLM decision explicitly.

Usage from project root:
  export OPENAI_API_KEY="sk-..."
  python mediator/tests/test_llm_contextual_norm_judgment.py \
    --evaluator mediator/contextual_integrity_evaluator.py \
    --constraints norms/ci_constraints.json

Optional:
  --strict-expectations
    Fails if the minimized candidate is not accepted or if raw video is accepted.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


VALID_LABELS = {
    "acceptable",
    "acceptable_with_mitigations",
    "uncertain",
    "inappropriate",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def make_request() -> Dict[str, Any]:
    """Structured post-parser app request for a low-stakes home HVAC task."""
    return {
        "schema_version": "smartpriv_application_request_v1",
        "request_identity": {
            "request_id": "req_test_llm_home_hvac_norms",
            "scenario_id": "test_llm_norm_judgment_home_hvac",
            "application_name": "Home HVAC Occupancy Controller",
            "application_category": "energy_management",
            "natural_language_request": (
                "Adjust HVAC based on whether people are present in the living room during a casual evening gathering."
            ),
            "requester": "downstream_application",
        },
        "ci_context": {
            "context": ["home"],
            "space": ["living_room", "common_area", "interior_home_space"],
            "sender": ["owner_controlled_device"],
            "subject": ["resident", "guest", "bystander"],
            "recipient": ["downstream_application", "hvac_controller"],
            "purpose": ["energy_management"],
            "transmissionPrinciple_assumed": ["local_processing", "purpose_limited"],
            "social_context_tags": [
                "guests_present",
                "routine_use",
                "low_stakes_automation",
                "weak_preference_channel",
            ],
        },
        "source_requirements": {
            "allowed_sensing_devices": ["indoor_camera", "motion_sensor"],
            "allowed_content_types": ["video_content", "image_content", "input_data"],
            "allowed_modalities": ["visual", "motion"],
            "forbidden_sensing_devices": ["microphone", "smart_speaker"],
            "forbidden_content_types": ["audio_content"],
        },
        "utility_contract": {
            "requested_capability": "occupancy_estimation",
            "task_description": "Provide room-level occupied/unoccupied state or approximate count for HVAC automation.",
            "quality_requirements": {
                "latency_ms_max": 300000,
                "update_period_ms_max": 300000,
                "minimum_recall": None,
                "minimum_precision": None,
                "minimum_accuracy": None,
                "count_mae_max": 2,
                "confidence_min": None,
            },
            "accepted_output_caps": [
                {
                    "cap_id": "occupancy_count",
                    "semantic_type": "application/x-occupancy-count",
                    "schema": "occupancy_count",
                    "affordance": "WoT.property",
                    "required_informationType": {
                        "sensorPrimitive": [],
                        "interpretedObservation": ["occupancy_count"],
                        "inferredInformationType": [],
                    },
                    "required_transmissionPrinciple": ["aggregate_only", "data_minimized"],
                    "priority": 1,
                    "utility_notes": "Sufficient for HVAC automation.",
                }
            ],
            "explicitly_rejected_output_caps": [
                {"semantic_type": "video/x-raw", "reason": "Raw video is unnecessary for HVAC."}
            ],
        },
    }


def make_environment() -> Dict[str, Any]:
    """Structured environment that would normally be produced by the context parser/LLM."""
    return {
        "context": ["home"],
        "space": ["living_room", "common_area", "interior_home_space"],
        "subject": ["resident", "guest", "bystander"],
        "recipient": ["downstream_application", "hvac_controller"],
        "purpose": ["energy_management"],
        "social_context_tags": [
            "evening",
            "multiple_people_present",
            "unfamiliar_guests_present",
            "casual_social_gathering",
            "weak_preference_channel",
            "not_emergency",
        ],
        "metadata": {
            "time_of_day": "evening",
            "people_count_estimate": "multiple",
            "event_inferred": "casual_social_gathering",
        },
    }


def make_candidate_output() -> Dict[str, Any]:
    """Synthetic candidate output from the pipeline generator."""
    minimized = {
        "pipeline_id": "pipe_test_minimized_occupancy_count",
        "decision": "candidate_pipeline",
        "matched_output_cap": "occupancy_count",
        "matched_output_schema": "occupancy_count",
        "final_output_cap": {
            "semantic_type": "application/x-occupancy-count",
            "schema": "occupancy_count",
            "affordance": "WoT.property",
        },
        "operators": [
            {"operator": "op.source", "variant": "Source(video)", "parameters": {}},
            {"operator": "op.person_object_detector", "variant": "Person / Object Detector", "parameters": {"classes": ["person"]}},
            {"operator": "op.occupancy_deriver", "variant": "Occupancy Deriver", "parameters": {"spatial_scope": "room"}},
            {"operator": "op.aggregate_generalize", "variant": "Aggregate / Generalize", "parameters": {"temporal_granularity_ms": 300000}},
            {"operator": "op.drop_discard", "variant": "Drop / Discard(raw_input_after_successor)", "parameters": {"drop_stage": "raw_input"}},
            {"operator": "op.route_publish", "variant": "Route / Publish(output_to_application)", "parameters": {"recipient": ["hvac_controller"]}},
        ],
        "utility_capabilities": ["occupancy_estimation", "energy_management_support"],
        "quality_status": "requires_runtime_or_benchmark_validation",
        "ci_terms": {
            "pipelineStage": ["output_to_application"],
            "informationType.sensorPrimitive": [],
            "informationType.interpretedObservation": ["occupancy_count"],
            "informationType.inferredInformationType": [],
            "transmissionPrinciple": [
                "local_processing",
                "purpose_limited",
                "aggregate_only",
                "data_minimized",
                "no_raw_data_retention",
                "ephemeral_processing",
            ],
        },
        "transforms": ["aggregate_only", "data_minimized", "no_raw_data_retention"],
        "residual_disclosure": {
            "identity": "none",
            "face": "none",
            "body_shape": "none",
            "clothing": "none",
            "gait": "none",
            "speech_content": "none",
            "speaker_identity": "none",
            "activity": "low",
            "location": "medium",
            "trajectory": "low",
            "co_presence": "medium",
            "visible_text": "none",
            "aggregate_presence": "high",
        },
        "residual_score": 8,
    }

    raw_video = {
        "pipeline_id": "pipe_test_raw_video_for_hvac",
        "decision": "candidate_pipeline",
        "matched_output_cap": "raw_video",
        "matched_output_schema": "video_stream",
        "final_output_cap": {
            "media_type": "video/x-raw",
            "schema": "raw_video_stream",
            "affordance": "WebRTC.stream",
        },
        "operators": [
            {"operator": "op.source", "variant": "Source(video)", "parameters": {}},
            {"operator": "op.route_publish", "variant": "Route / Publish(output_to_application)", "parameters": {"recipient": ["hvac_controller"]}},
        ],
        "utility_capabilities": ["provide_raw_stream", "occupancy_estimation"],
        "quality_status": "requires_runtime_or_benchmark_validation",
        "ci_terms": {
            "pipelineStage": ["output_to_application"],
            "informationType.sensorPrimitive": ["video_stream", "image_frame"],
            "informationType.interpretedObservation": [],
            "informationType.inferredInformationType": [],
            "transmissionPrinciple": ["local_processing", "purpose_limited"],
        },
        "transforms": [],
        "residual_disclosure": {
            "identity": "high",
            "face": "high",
            "body_shape": "high",
            "clothing": "high",
            "gait": "medium",
            "speech_content": "none",
            "speaker_identity": "none",
            "activity": "high",
            "location": "high",
            "trajectory": "high",
            "co_presence": "high",
            "visible_text": "medium",
            "aggregate_presence": "high",
        },
        "residual_score": 45,
    }

    return {
        "schema_version": "smartpriv_pipeline_candidate_output_v1",
        "request_id": "req_test_llm_home_hvac_norms",
        "scenario_id": "test_llm_norm_judgment_home_hvac",
        "planner": {
            "algorithm": "synthetic test fixture",
            "candidate_count": 2,
            "notes": [
                "This fixture bypasses candidate generation to isolate LLM-assisted norm judgment."
            ],
        },
        "decision": {
            "decision": "synthetic_candidates",
            "selected_pipeline_id": None,
            "selected_output_cap": None,
        },
        "candidates": [minimized, raw_video],
        "rejected_goal_examples": [],
    }


def extract_llm_decisions(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ev in result.get("evaluations", []):
        judgment = ev.get("llm_norm_judgment") or {}
        decision = ev.get("ci_decision") or {}
        rows.append({
            "pipeline_id": ev.get("pipeline_id"),
            "matched_output_cap": ev.get("matched_output_cap"),
            "hard_pass": (ev.get("hard_constraint_result") or {}).get("hard_pass"),
            "llm_acceptability_label": judgment.get("acceptability_label"),
            "llm_confidence": judgment.get("confidence"),
            "llm_required_mitigations": judgment.get("required_mitigations", []),
            "llm_violated_expectations": judgment.get("violated_expectations", []),
            "llm_rationale": judgment.get("rationale"),
            "ci_decision": decision.get("decision"),
            "ci_feasible": decision.get("feasible"),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", default="mediator/contextual_integrity_evaluator.py")
    parser.add_argument("--constraints", default="norms/ci_constraints.json")
    parser.add_argument("--out-dir", default="mediator/tests/test_llm_contextual_norm_judgment_run")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--strict-expectations", action="store_true",
                        help="Fail if minimized output is not accepted or raw video is accepted.")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your_api_key_here'"
        )

    evaluator_path = Path(args.evaluator).resolve()
    constraints_path = Path(args.constraints).resolve()
    out_dir = Path(args.out_dir).resolve()

    evaluator = load_module("contextual_integrity_evaluator_under_test", evaluator_path)
    constraints = load_json(constraints_path)
    request = make_request()
    environment = make_environment()
    candidate_output = make_candidate_output()

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(request, out_dir / "request_home_hvac_llm_norm_test.json")
    write_json(environment, out_dir / "environment_home_party_llm_norm_test.json")
    write_json(candidate_output, out_dir / "synthetic_candidate_pipelines.json")

    # Be compatible with slightly different evaluator versions. Older local copies
    # may not support optional kwargs such as unknown_policy or top_k_for_llm.
    eval_kwargs = {
        "candidate_output": candidate_output,
        "request": request,
        "constraints": constraints,
        "environment": environment,
        "sensor_stream": None,
        "use_llm": True,
        "llm_model": args.llm_model,
        "llm_temperature": args.llm_temperature,
        "llm_confidence_threshold": args.llm_confidence_threshold,
        "unknown_policy": "no_match",
        "top_k_for_llm": None,
    }
    sig = inspect.signature(evaluator.evaluate_candidates)
    supported_kwargs = {k: v for k, v in eval_kwargs.items() if k in sig.parameters}
    result = evaluator.evaluate_candidates(**supported_kwargs)
    write_json(result, out_dir / "llm_ci_evaluation.json")

    decisions = extract_llm_decisions(result)
    write_json({"llm_decisions": decisions}, out_dir / "llm_decision_summary.json")

    print("\n=== LLM contextual-integrity norm decisions ===")
    for row in decisions:
        print(json.dumps(row, indent=2))

    print("\n=== Overall evaluator decision ===")
    print(json.dumps(result.get("decision"), indent=2))

    # Always check structural validity.
    for row in decisions:
        label = row.get("llm_acceptability_label")
        assert label in VALID_LABELS, f"Invalid LLM label for {row.get('pipeline_id')}: {label!r}"
        assert row.get("llm_confidence") is not None, f"Missing LLM confidence for {row.get('pipeline_id')}"

    if args.strict_expectations:
        by_id = {r["pipeline_id"]: r for r in decisions}
        minimized = by_id["pipe_test_minimized_occupancy_count"]
        raw = by_id["pipe_test_raw_video_for_hvac"]

        assert minimized["llm_acceptability_label"] in {"acceptable", "acceptable_with_mitigations"}, (
            f"Expected minimized occupancy output to be acceptable-ish, got {minimized['llm_acceptability_label']}"
        )
        assert raw["llm_acceptability_label"] in {"uncertain", "inappropriate", "acceptable_with_mitigations"}, (
            f"Unexpected label for raw video: {raw['llm_acceptability_label']}"
        )
        assert raw["llm_acceptability_label"] != "acceptable", (
            "Expected raw video for HVAC with guests not to be unqualified acceptable."
        )

    print(f"\nPASS: LLM norm-judgment path ran. Outputs written under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
