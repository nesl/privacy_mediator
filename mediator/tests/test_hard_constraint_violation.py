#!/usr/bin/env python3
"""
test_hard_constraint_violation_no_sensor_stream.py

Creates a deliberately CI-violating application request and runs full_mediator.py
end-to-end without requiring a separate sensor_stream interface. The expected behavior is:

  1. candidate generation finds one or more utility-compatible pipelines;
  2. contextual_integrity_evaluator rejects all candidates via hard constraints;
  3. full_mediator returns a no_compromise-style decision.

The violation is intentional and is encoded directly in the application request:
  - context: short_term_rental
  - source: hidden indoor_camera
  - purpose: energy_management / occupancy estimation

This should trigger the hard rule:
  airbnb_hidden_recording_devices_ban

and often also related short-term-rental indoor-camera/surveillance rules.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


RESIDUAL_ATTRS = [
    "identity", "face", "body_shape", "clothing", "gait", "speech_content",
    "speaker_identity", "activity", "location", "trajectory", "co_presence",
    "visible_text", "aggregate_presence"
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def infer_existing(base: Path, *names: str) -> Path:
    for name in names:
        candidate = base / name
        if candidate.exists():
            return candidate
    return base / names[0]


def max_allowed_for_occupancy() -> Dict[str, str]:
    # Loose enough that candidate generation can find utility-compatible pipelines.
    # The CI hard-constraint layer, not this residual bound, should cause rejection.
    d = {a: "none" for a in RESIDUAL_ATTRS}
    d.update({
        "activity": "low",
        "location": "medium",
        "trajectory": "low",
        "co_presence": "medium",
        "aggregate_presence": "high",
    })
    return d


def make_hidden_indoor_camera_request() -> Dict[str, Any]:
    return {
        "schema_version": "smartpriv_application_request_v1",
        "description": "Deliberately violating request for testing hard CI constraints.",
        "request_identity": {
            "request_id": "req_test_hidden_indoor_camera_short_term_rental",
            "scenario_id": "test_hard_constraint_hidden_device",
            "application_name": "Rental HVAC Occupancy Controller using Hidden Indoor Camera",
            "application_category": "energy_management",
            "natural_language_request": (
                "Use a hidden indoor camera in a short-term rental living room to estimate occupancy "
                "for HVAC automation."
            ),
            "requester": "downstream_application",
        },
        "ci_context": {
            "context": ["short_term_rental"],
            "space": ["indoor", "interior_home_space", "living_room", "common_area"],
            "sender": ["host_controlled_device"],
            "subject": ["guest", "bystander"],
            "recipient": ["downstream_application", "host"],
            "purpose": ["energy_management"],
            "transmissionPrinciple_assumed": [
                "hidden",
                "undisclosed",
                "local_processing",
                "data_minimized",
                "aggregate_only",
                "no_raw_data_retention",
            ],
            "social_context_tags": [
                "guests_present",
                "operator_subject_separation",
                "weak_preference_channel",
            ],
        },
        "source_requirements": {
            "allowed_sensing_devices": ["indoor_camera"],
            "allowed_content_types": ["image_content", "video_content"],
            "allowed_modalities": ["visual"],
            "forbidden_sensing_devices": [],
            "forbidden_content_types": [],
            "notes": (
                "This is intentionally invalid: a hidden indoor recording device in a short-term rental "
                "should be rejected by hard CI constraints."
            ),
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
                "count_mae_max": None,
                "confidence_min": None,
            },
            "accepted_output_caps": [
                {
                    "cap_id": "binary_occupancy",
                    "semantic_type": "application/x-binary-occupancy",
                    "schema": "room_occupied",
                    "affordance": "WoT.property",
                    "required_informationType": {
                        "sensorPrimitive": [],
                        "interpretedObservation": ["room_occupied"],
                        "inferredInformationType": [],
                    },
                    "required_transmissionPrinciple": ["aggregate_only", "data_minimized"],
                    "priority": 1,
                    "utility_notes": "Sufficient for HVAC automation.",
                },
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
                    "priority": 2,
                    "utility_notes": "Useful for count-based HVAC control.",
                },
            ],
            "explicitly_rejected_output_caps": [
                {"semantic_type": "audio/x-raw", "reason": "Audio is not needed."}
            ],
        },
        "compatibility_constraints": {
            "output_pipeline_stage_must_be": "output_to_application",
            "accepted_schema_match": "candidate output caps must match at least one accepted_output_caps entry",
            "required_capability_match": "candidate utility_capabilities must include requested_capability or equivalent",
            "adapter_allowed": True,
        },
        "ci_output_constraints": {
            "required_ci_terms": {
                "informationType.sensorPrimitive": [],
                "informationType.interpretedObservation": ["room_occupied OR occupancy_count"],
                "informationType.inferredInformationType": [],
                "transmissionPrinciple": ["aggregate_only", "data_minimized"],
            },
            "forbidden_ci_terms": {
                "informationType.sensorPrimitive": ["audio_waveform", "speech_audio"],
                "informationType.interpretedObservation": ["speech_transcribed"],
                "informationType.inferredInformationType": ["identity", "trajectory"],
                "transmissionPrinciple": ["cloud_processing", "indefinite_retention"],
            },
            "required_transformations_if_media_output": ["identity_removed", "field_of_view_minimized"],
            "preferred_ci_terms": {
                "informationType.interpretedObservation": ["room_occupied", "occupancy_count"],
                "informationType.inferredInformationType": [],
                "transmissionPrinciple": ["aggregate_only", "data_minimized", "no_raw_data_retention"],
            },
        },
        "residual_disclosure_constraints": {
            "risk_order": {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": 4},
            "max_allowed": max_allowed_for_occupancy(),
            "hard_forbidden_attributes": [
                "identity", "face", "speech_content", "speaker_identity",
                "visible_text", "gait", "clothing", "body_shape",
            ],
            "allowed_high_attributes": ["aggregate_presence"],
            "comparison_rule": "candidate residual disclosure vector must be <= max_allowed",
        },
        "no_compromise_conditions": [
            "Return no_compromise if hidden recording-device hard constraints reject the source/flow."
        ],
    }


def make_sensor_stream() -> Dict[str, Any]:
    return {
        "sensor_stream_id": "hidden_living_room_camera_001",
        "sensingDevice": ["indoor_camera"],
        "contentType": ["video_content", "image_content"],
        "sensingModality": ["visual"],
        "space": ["indoor", "interior_home_space", "living_room", "common_area"],
        "sender": ["host_controlled_device"],
        "transmissionPrinciple": ["hidden", "undisclosed"],
        "metadata": {
            "device_location_disclosed": False,
            "coverage_area_disclosed": False,
        },
    }


def find_triggered_rules(result: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ev in result.get("ci_evaluation_result", {}).get("evaluations", []):
        hard = ev.get("hard_constraint_result", {})
        for denial in hard.get("denied_by", []):
            rid = denial.get("rule_id", "<missing>")
            counts[rid] = counts.get(rid, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent),
                        help="Directory containing full_mediator.py and related files.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory. Default: <base-dir>/test_hard_constraint_violation_run")
    parser.add_argument("--operators", default=None)
    parser.add_argument("--constraints", default=None)
    parser.add_argument("--candidate-generator", default=None)
    parser.add_argument("--evaluator", default=None)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--max-states", type=int, default=25000)
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else base / "test_hard_constraint_violation_run"
    fixtures = out_dir / "fixtures"

    full_mediator_path = base / "full_mediator.py"
    operators_path = Path(args.operators) if args.operators else infer_existing(base, "smartpriv_operator_contracts.json", "operator_contracts.json", "norms/operator_contracts.json")
    constraints_path = Path(args.constraints) if args.constraints else infer_existing(base, "ci_constraints(1).json", "ci_constraints.json", "norms/ci_constraints.json")
    generator_path = Path(args.candidate_generator) if args.candidate_generator else infer_existing(base, "generate_pipeline_candidates(2).py", "generate_pipeline_candidates.py")
    evaluator_path = Path(args.evaluator) if args.evaluator else infer_existing(base, "contextual_integrity_evaluator.py")

    if not full_mediator_path.exists():
        raise FileNotFoundError(full_mediator_path)
    if not operators_path.exists():
        raise FileNotFoundError(operators_path)
    if not constraints_path.exists():
        raise FileNotFoundError(constraints_path)
    if not generator_path.exists():
        raise FileNotFoundError(generator_path)
    if not evaluator_path.exists():
        raise FileNotFoundError(evaluator_path)

    request_path = fixtures / "request_hidden_indoor_camera_short_term_rental.json"
    write_json(make_hidden_indoor_camera_request(), request_path)

    mediator = load_module("full_mediator_under_test", full_mediator_path)
    result = mediator.run_mediator(
        operators_path=operators_path,
        request_path=request_path,
        constraints_path=constraints_path,
        sensor_stream_path=None,
        candidate_generator_path=generator_path,
        evaluator_path=evaluator_path,
        max_depth=args.max_depth,
        max_states=args.max_states,
        use_llm=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(result.get("candidate_generation_result", {}), out_dir / "candidate_pipelines.json")
    write_json(result.get("ci_evaluation_result", {}), out_dir / "ci_evaluation.json")
    write_json(result, out_dir / "full_mediator_result.json")

    decision = (result.get("decision") or {}).get("decision")
    candidate_count = (result.get("candidate_generation_result") or {}).get("planner", {}).get("candidate_count", 0)
    triggered = find_triggered_rules(result)

    print(json.dumps({
        "decision": result.get("decision"),
        "candidate_count": candidate_count,
        "triggered_denial_rules": triggered,
        "outputs": {
            "request": str(request_path),
            "sensor_stream": None,
            "candidate_pipelines": str(out_dir / "candidate_pipelines.json"),
            "ci_evaluation": str(out_dir / "ci_evaluation.json"),
            "full_mediator_result": str(out_dir / "full_mediator_result.json"),
        },
    }, indent=2))

    assert candidate_count > 0, "Expected candidate generation to find utility-compatible pipelines before CI rejection."
    assert decision == "no_compromise", f"Expected no_compromise, got {decision!r}."
    assert "airbnb_hidden_recording_devices_ban" in triggered, (
        "Expected airbnb_hidden_recording_devices_ban to reject at least one candidate."
    )

    print("\nPASS: hard-constraint violation was correctly rejected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
