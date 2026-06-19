#!/usr/bin/env python3
"""
generate_pipeline_candidates.py

Symbolic preprocessing-pipeline planner for a privacy-aware smart-space hub.

Inputs:
  --operators  JSON operator-contract catalog, e.g. norms/operator_contracts.json
  --request    JSON application request, e.g. app_requests/request_home_guest_audio_command_safety.json
  --out        Optional JSON output path

The planner does not execute preprocessing code. It performs typed symbolic
composition over operator metadata:
  1. start from allowed source caps;
  2. enumerate caps-compatible operator chains;
  3. propagate residual-disclosure metadata;
  4. match final output caps against the application request;
  5. filter by CI/output/residual constraints;
  6. rank feasible pipelines by least residual disclosure.

This is best understood as a lightweight type-and-effect / abstract-interpretation
pass over a dataflow graph.

Example:
  python generate_pipeline_candidates.py \
      --operators smartpriv_operator_contracts.json \
      --request smartpriv_application_requests/request_home_party_hvac.json \
      --out hvac_candidates.json
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from smartpriv_runtime.codegen import attach_executable_specs, emit_programs
except Exception:  # Runtime package is optional for symbolic-only use.
    attach_executable_specs = None
    emit_programs = None


RISK_ORDER: Dict[str, int] = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "unknown": 4,
}
RISK_LEVELS = {v: k for k, v in RISK_ORDER.items()}

RESIDUAL_ATTRIBUTES: List[str] = [
    "identity",
    "face",
    "body_shape",
    "clothing",
    "gait",
    "speech_content",
    "speaker_identity",
    "activity",
    "location",
    "trajectory",
    "co_presence",
    "visible_text",
    "aggregate_presence",
]

# Lightweight family aliases for caps compatibility.
SEMANTIC_FAMILIES: Dict[str, Set[str]] = {
    "application/x-count": {
        "application/x-count",
        "application/x-occupancy-count",
        "application/x-binary-occupancy",
    },
    "application/x-event": {
        "application/x-event",
        "application/x-safety-event",
        "application/x-security-event",
        "application/x-sound-event-label",
        "application/x-activity-label",
        "application/x-fused-event",
    },
    "application/x-observation": {
        "application/x-observation",
        "application/x-occupancy-count",
        "application/x-binary-occupancy",
        "application/x-decibel-level",
        "application/x-command-intent",
        "application/x-sound-event-label",
        "application/x-safety-event",
        "application/x-security-event",
        "application/x-activity-label",
    },
    "application/x-detections": {
        "application/x-detections",
    },
    "application/x-sensor-reading": {
        "application/x-sensor-reading",
        "application/x-motion-events",
        "application/x-noise-sensor",
    },
}

# Conservative conversions that a schema adapter is allowed to assert symbolically.
# This prevents the generic adapter from converting raw video directly into arbitrary
# semantic events unless a real inference operator has produced a semantically related output.
ADAPTER_ALLOWED_INPUTS: Dict[str, Set[str]] = {
    "application/x-binary-occupancy": {
        "application/x-binary-occupancy",
        "application/x-occupancy-count",
        "application/x-aggregate",
    },
    "application/x-occupancy-count": {
        "application/x-occupancy-count",
        "application/x-binary-occupancy",
        "application/x-aggregate",
    },
    "application/x-safety-event": {
        "application/x-safety-event",
        "application/x-activity-label",
        "application/x-pose-keypoints",
    },
    "application/x-pose-keypoints": {
        "application/x-pose-keypoints",
    },
    "application/x-activity-label": {
        "application/x-activity-label",
        "application/x-pose-keypoints",
    },
    "application/x-command-intent": {
        "application/x-command-intent",
        "application/x-transcript",
    },
    "application/x-sound-event-label": {
        "application/x-sound-event-label",
    },
    "application/x-decibel-level": {
        "application/x-decibel-level",
    },
    "application/x-security-event": {
        "application/x-security-event",
        "application/x-fused-event",
        "application/x-event",
        "application/x-detections",
        "application/x-activity-label",
        "application/x-safety-event",
    },
    "image/x-redacted": {
        "image/x-redacted",
        "image/x-raw",
        "video/x-raw",
    },
    "video/x-redacted": {
        "video/x-redacted",
        "video/x-raw",
    },
}


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def normalize_risk(value: Any) -> str:
    """Map contract phrases like low_to_medium or by_1_level to a conservative level."""
    if value is None:
        return "unknown"
    if isinstance(value, (int, float)):
        # Assume 0-3-ish numeric risk if a probe supplies numeric values.
        if value <= 0:
            return "none"
        if value <= 1:
            return "low"
        if value <= 2:
            return "medium"
        if value <= 3:
            return "high"
        return "unknown"
    s = str(value).strip().lower()
    if s in RISK_ORDER:
        return s
    if "unknown" in s:
        return "unknown"
    # Composite ranges are treated conservatively.
    if "high" in s:
        return "high"
    if "medium" in s:
        return "medium"
    if "low" in s:
        return "low"
    if "none" in s or "removed" in s or "absent" in s:
        return "none"
    return "unknown"


def risk_max(a: str, b: str) -> str:
    return RISK_LEVELS[max(RISK_ORDER[normalize_risk(a)], RISK_ORDER[normalize_risk(b)])]


def risk_min(a: str, b: str) -> str:
    return RISK_LEVELS[min(RISK_ORDER[normalize_risk(a)], RISK_ORDER[normalize_risk(b)])]


def risk_decrement(level: str, steps: int = 1) -> str:
    return RISK_LEVELS[max(0, RISK_ORDER[normalize_risk(level)] - steps)]


def risk_score(vec: Dict[str, str]) -> int:
    # Identity/speech/trajectory get slightly higher weight because they often drive CI conflicts.
    weights = {
        "identity": 3,
        "face": 2,
        "speech_content": 3,
        "speaker_identity": 3,
        "trajectory": 2,
        "visible_text": 2,
        "activity": 1,
        "location": 1,
        "aggregate_presence": 1,
    }
    return sum(weights.get(k, 1) * RISK_ORDER.get(normalize_risk(v), 4) for k, v in vec.items())


def init_residual(default: str = "none") -> Dict[str, str]:
    return {a: default for a in RESIDUAL_ATTRIBUTES}


def stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def cap_type(cap: Dict[str, Any]) -> str:
    return cap.get("semantic_type") or cap.get("media_type") or ""


def cap_schema(cap: Dict[str, Any]) -> str:
    return str(cap.get("schema") or "")


def media_family(media_or_semantic_type: str) -> str:
    """Return coarse media family for compatibility-preserving transforms."""
    t = str(media_or_semantic_type or "")
    if t.startswith("image/"):
        return "image"
    if t.startswith("video/"):
        return "video"
    if t.startswith("audio/"):
        return "audio"
    return ""


def same_media_family(a: str, b: str) -> bool:
    fa = media_family(a)
    fb = media_family(b)
    return bool(fa and fb and fa == fb)


def redacted_media_type_for(upstream_type: str) -> str:
    family = media_family(upstream_type)
    if family == "image":
        return "image/x-redacted"
    if family == "video":
        return "video/x-redacted"
    return upstream_type


def cap_signature(cap: Dict[str, Any]) -> Tuple[str, str]:
    return (cap_type(cap), cap_schema(cap))


def type_matches(upstream: str, downstream: str) -> bool:
    if not downstream or downstream == "*":
        return True
    if not upstream:
        return False
    if downstream == upstream:
        return True

    # Families such as application/x-count accept occupancy count variants.
    if downstream in SEMANTIC_FAMILIES and upstream in SEMANTIC_FAMILIES[downstream]:
        return True
    if upstream in SEMANTIC_FAMILIES and downstream in SEMANTIC_FAMILIES[upstream]:
        return True

    # Treat transformed media as compatible with raw-media consumers only through
    # explicit media-like paths. This allows privacy-preserving transforms such as
    # blurred images/videos or speech-filtered audio to feed legacy media-style apps
    # while still tracking the transformed cap and residual disclosure separately.
    if downstream == "video/x-raw" and upstream in {"video/x-redacted", "video/x-raw"}:
        return True
    if downstream == "image/x-raw" and upstream in {"image/x-redacted", "image/x-raw"}:
        return True
    if downstream == "audio/x-raw" and upstream in {"audio/x-filtered", "audio/x-raw"}:
        return True

    # Redacted media remains media-compatible with same-family media consumers,
    # but the cap records that the output is transformed rather than raw.
    if downstream in {"image/x-redacted", "video/x-redacted"} and same_media_family(upstream, downstream):
        return True

    return False


def cap_matches(up_cap: Dict[str, Any], in_cap: Dict[str, Any]) -> bool:
    # Wildcard semantic type is used by generic adapters.
    if in_cap.get("semantic_type") == "*" or in_cap.get("media_type") == "*":
        return True
    up_t = cap_type(up_cap)
    in_t = cap_type(in_cap)
    if type_matches(up_t, in_t):
        return True
    # If only schemas are supplied, allow exact schema match.
    return bool(cap_schema(up_cap) and cap_schema(up_cap) == cap_schema(in_cap))


def goal_cap_matches(out_cap: Dict[str, Any], goal_cap: Dict[str, Any]) -> bool:
    goal_t = goal_cap.get("semantic_type") or goal_cap.get("media_type")
    out_t = cap_type(out_cap)
    if goal_t and type_matches(out_t, goal_t):
        return True
    goal_schema = goal_cap.get("schema")
    if goal_schema and cap_schema(out_cap) == goal_schema:
        return True
    return False


def parse_or_terms(terms: Sequence[str]) -> List[List[str]]:
    """Convert ["a OR b", "c"] into [["a", "b"], ["c"]]."""
    groups: List[List[str]] = []
    for term in terms:
        if not isinstance(term, str):
            continue
        parts = [p.strip() for p in re.split(r"\s+OR\s+", term) if p.strip()]
        if parts:
            groups.append(parts)
    return groups


def flatten_terms(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, dict):
        out: List[str] = []
        for v in x.values():
            out.extend(flatten_terms(v))
        return out
    if isinstance(x, list):
        out = []
        for i in x:
            out.extend(flatten_terms(i))
        return out
    return [str(x)]


@dataclasses.dataclass(frozen=True)
class OperatorVariant:
    id: str
    label: str
    category: str
    input_caps: Tuple[Tuple[Tuple[str, Any], ...], ...]
    output_cap: Dict[str, Any]
    utility_capabilities: Tuple[str, ...]
    residual_effect: Dict[str, Any]
    ci_additions: Dict[str, List[str]]
    transform_effects: Tuple[str, ...]
    parameters: Dict[str, Any]
    raw_operator_id: str

    def input_caps_dicts(self) -> List[Dict[str, Any]]:
        return [dict(items) for items in self.input_caps]


@dataclasses.dataclass
class State:
    cap: Dict[str, Any]
    residual: Dict[str, str]
    ci_terms: Dict[str, Set[str]]
    utility_caps: Set[str]
    transforms: Set[str]
    pipeline: List[Dict[str, Any]]
    depth: int

    def copy(self) -> "State":
        return State(
            cap=copy.deepcopy(self.cap),
            residual=dict(self.residual),
            ci_terms={k: set(v) for k, v in self.ci_terms.items()},
            utility_caps=set(self.utility_caps),
            transforms=set(self.transforms),
            pipeline=copy.deepcopy(self.pipeline),
            depth=self.depth,
        )


def tupleize_caps(caps: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    # Only simple dicts are needed for matching; stringify nested fields.
    items = []
    for k, v in sorted(caps.items()):
        if isinstance(v, (dict, list)):
            items.append((k, json.dumps(v, sort_keys=True)))
        else:
            items.append((k, v))
    return tuple(items)


def untupleize_caps(items: Tuple[Tuple[str, Any], ...]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in items:
        if isinstance(v, str) and v[:1] in "[{":
            try:
                out[k] = json.loads(v)
                continue
            except Exception:
                pass
        out[k] = v
    return out


def ci_add_from_operator(op: Dict[str, Any]) -> Dict[str, List[str]]:
    additions: Dict[str, List[str]] = {}

    ann = op.get("ci_annotations", {}) or {}
    for k, v in ann.items():
        # Source contracts often list the vocabulary of possible primitives
        # (image_frame, video_stream, audio_waveform, ...). Do not add all of
        # them to every source variant; initial_state_from_source adds only the
        # primitive that corresponds to the selected source cap.
        if op.get("id") == "op.source" and k in {"informationType.sensorPrimitive", "sensingDataMetadata"}:
            continue
        # Preserve stage annotations for the full system. The no_staged_flows
        # ablation intentionally collapses these later in the CI evaluator.
        additions.setdefault(k, [])
        additions[k].extend(flatten_terms(v))

    eff = op.get("residual_disclosure_effect", {}) or {}
    add = eff.get("add", {}) if isinstance(eff, dict) else {}
    for k, v in add.items():
        additions.setdefault(k, [])
        additions[k].extend(flatten_terms(v))

    # Transformation effects often correspond to transmission principles.
    for t in op.get("transformation_effects", []) or []:
        if t in {
            "face_blurred",
            "body_blurred",
            "screen_content_removed",
            "identity_removed",
            "speech_content_removed",
            "data_minimized",
            "aggregate_only",
            "field_of_view_minimized",
            "no_raw_data_retention",
            "ephemeral_processing",
            "event_triggered_collection",
            "raw_audio_removed",
            "raw_pixels_removed",
            "raw_media_removed",
            "event_label_only",
            "audio_waveform_removed",
            "speech_content_minimized",
        }:
            additions.setdefault("transmissionPrinciple", []).append(t)

    return additions


def with_output_cap_effects(out_cap: Dict[str, Any], ci_additions: Dict[str, List[str]]) -> Dict[str, List[str]]:
    additions = copy.deepcopy(ci_additions)
    props = out_cap.get("properties", {}) or {}
    for k in ["sensorPrimitive", "interpretedObservation", "inferredInformationType"]:
        if k in props:
            full_key = f"informationType.{k}"
            additions.setdefault(full_key, [])
            additions[full_key].extend(flatten_terms(props[k]))
    return additions


def materialize_variants(operators: List[Dict[str, Any]], request: Dict[str, Any]) -> Tuple[List[OperatorVariant], List[OperatorVariant]]:
    """Return (source_variants, transform_variants)."""
    source_variants: List[OperatorVariant] = []
    transform_variants: List[OperatorVariant] = []

    accepted_caps = request.get("utility_contract", {}).get("accepted_output_caps", []) or []

    for op in operators:
        op_id = op.get("id", "")
        if op_id in {"op.route_publish", "op.drop_discard"}:
            # Route/publish is represented by output_to_application verification.
            # Drop/discard is represented as a final side-effect when no_raw retention is preferred.
            continue

        input_caps = tuple(tupleize_caps(c) for c in op.get("input_caps", []) or [])
        outputs = op.get("output_caps", []) or [{}]

        # Specialize schema_adapter to each application accepted cap.
        if op_id == "op.schema_adapter":
            outputs = []
            for g in accepted_caps:
                out_cap = {
                    "semantic_type": g.get("semantic_type"),
                    "schema": g.get("schema"),
                    "affordance": g.get("affordance"),
                    "properties": {
                        "required_informationType": g.get("required_informationType", {}),
                        "adapter_target": g.get("cap_id"),
                    },
                }
                outputs.append(out_cap)

        # Specialize region_mask_blur because its residual effects depend on target.
        mask_targets = [None]
        if op_id == "op.region_mask_blur":
            mask_targets = ["face", "body", "screen", "background"]

        for target in mask_targets:
            for idx, out in enumerate(outputs):
                label = op.get("label", op_id)
                params = {}
                residual_effect = copy.deepcopy(op.get("residual_disclosure_effect", {}) or {})
                transform_effects = list(op.get("transformation_effects", []) or [])

                if target:
                    params["target"] = target
                    label = f"{label}({target})"
                    residual_effect = specialize_region_mask_effect(residual_effect, target)
                    if target == "face":
                        transform_effects.extend(["face_blurred", "identity_removed"])
                    elif target == "body":
                        transform_effects.extend(["body_blurred"])
                    elif target == "screen":
                        transform_effects.extend(["screen_content_removed"])
                    elif target == "background":
                        transform_effects.extend(["field_of_view_minimized"])

                # Add a target schema parameter for the schema adapter.
                if op_id == "op.schema_adapter":
                    params["target_schema"] = out.get("schema")
                    params["target_semantic_type"] = out.get("semantic_type")
                    label = f"Schema Adapter({out.get('schema')})"

                ci_add = with_output_cap_effects(out, ci_add_from_operator({**op, "transformation_effects": transform_effects}))
                variant = OperatorVariant(
                    id=f"{op_id}:{idx}:{target or 'default'}",
                    label=label,
                    category=op.get("category", ""),
                    input_caps=input_caps,
                    output_cap=copy.deepcopy(out),
                    utility_capabilities=tuple(op.get("utility_capabilities", []) or []),
                    residual_effect=residual_effect,
                    ci_additions=ci_add,
                    transform_effects=tuple(transform_effects),
                    parameters=params,
                    raw_operator_id=op_id,
                )
                if op_id == "op.source":
                    source_variants.append(variant)
                else:
                    transform_variants.append(variant)

    return source_variants, transform_variants


def specialize_region_mask_effect(effect: Dict[str, Any], target: str) -> Dict[str, Any]:
    """Convert conditional region blur effects into ordinary reduce/preserve effects."""
    conds = effect.get("conditional_effects", {}) or {}
    chosen: Dict[str, Any] = {}
    if target == "face":
        chosen = conds.get("target=face", {})
    elif target == "body":
        chosen = conds.get("target=body", {})
    elif target == "screen":
        chosen = conds.get("target=screen|document", {})
    elif target == "background":
        chosen = conds.get("target=background", {})

    out = {
        "default_policy": effect.get("default_policy", "preserve_unmentioned"),
        "remove": [],
        "set": {},
        "reduce": chosen.get("reduce", {}),
        "preserve": chosen.get("preserve", []),
        "add": effect.get("add", {}),
        "notes": effect.get("notes", ""),
    }
    return out


def allowed_source(v: OperatorVariant, request: Dict[str, Any]) -> bool:
    req = request.get("source_requirements", {}) or {}
    allowed_content = set(req.get("allowed_content_types", []) or [])
    allowed_modality = set(req.get("allowed_modalities", []) or [])
    forbidden_content = set(req.get("forbidden_content_types", []) or [])

    cap = v.output_cap
    content = cap.get("content_type")
    if content in forbidden_content:
        return False
    if allowed_content and content and content in allowed_content:
        return True

    # semantic sensor readings may correspond to input_data, motion, or environmental.
    if cap.get("semantic_type") == "application/x-sensor-reading":
        return ("input_data" in allowed_content) or bool(allowed_modality & {"motion", "environmental"})

    # Some source caps may only have properties.
    props = cap.get("properties", {}) or {}
    prim = props.get("sensorPrimitive")
    if prim == "audio_waveform" and "audio_content" in allowed_content:
        return True
    if prim in {"image_frame", "video_stream"} and ({"image_content", "video_content"} & allowed_content):
        return True

    return not allowed_content


def initial_state_from_source(v: OperatorVariant) -> State:
    cap = copy.deepcopy(v.output_cap)
    content = cap.get("content_type")
    props = cap.get("properties", {}) or {}
    prim = props.get("sensorPrimitive")
    residual = init_residual("none")

    # Source operator's contract contains initial_state_by_modality.
    init_by_mod = v.residual_effect.get("initial_state_by_modality", {}) if isinstance(v.residual_effect, dict) else {}
    if content and content in init_by_mod:
        residual.update({k: normalize_risk(vv) for k, vv in init_by_mod[content].items()})
    elif prim == "decibel_level" and "noise_decibel_monitor" in init_by_mod:
        residual.update({k: normalize_risk(vv) for k, vv in init_by_mod["noise_decibel_monitor"].items()})
    elif prim == "audio_waveform" and "audio_content" in init_by_mod:
        residual.update({k: normalize_risk(vv) for k, vv in init_by_mod["audio_content"].items()})
    elif prim in {"image_frame", "video_stream"}:
        key = "video_content" if prim == "video_stream" else "image_content"
        if key in init_by_mod:
            residual.update({k: normalize_risk(vv) for k, vv in init_by_mod[key].items()})
    else:
        # Conservative defaults for unknown sensor readings.
        residual["location"] = "low"
        residual["aggregate_presence"] = "low"

    ci_terms = {
        "informationType.sensorPrimitive": set(),
        "informationType.interpretedObservation": set(),
        "informationType.inferredInformationType": set(),
        "transmissionPrinciple": set(),
    }
    if prim:
        ci_terms["informationType.sensorPrimitive"].add(prim)
    if content == "video_content":
        ci_terms["informationType.sensorPrimitive"].add("video_stream")
    elif content == "image_content":
        ci_terms["informationType.sensorPrimitive"].add("image_frame")
    elif content == "audio_content":
        ci_terms["informationType.sensorPrimitive"].add("audio_waveform")

    for k, vals in v.ci_additions.items():
        ci_terms.setdefault(k, set()).update(vals)

    return State(
        cap=cap,
        residual=residual,
        ci_terms=ci_terms,
        utility_caps=set(v.utility_capabilities),
        transforms=set(v.transform_effects),
        pipeline=[{
            "operator": v.raw_operator_id,
            "variant": v.label,
            "output_cap": cap,
            "parameters": v.parameters,
        }],
        depth=0,
    )


def apply_reduce_rule(current: str, rule: Any) -> str:
    s = str(rule).lower()
    if "to_none" in s:
        # Conservative interpretation: if rule says to none or low, choose low unless exactly none.
        if "or_low" in s or "or low" in s:
            return "low"
        return "none"
    if "to_low" in s:
        return "low"
    if "by_1_or_2" in s:
        return risk_decrement(current, 1)
    if "by_2" in s:
        return risk_decrement(current, 2)
    if "by_1" in s or "downsampling" in s or "minimization" in s or "generalization" in s or "aggregation" in s:
        return risk_decrement(current, 1)
    if "if_text_regions_excluded" in s:
        return risk_decrement(current, 1)
    if "depends" in s or "may" in s:
        return current
    return risk_decrement(current, 1)


def apply_residual_effect(state: State, v: OperatorVariant) -> State:
    new = state.copy()
    new.depth += 1
    new.cap = copy.deepcopy(v.output_cap)
    upstream_type = cap_type(state.cap)
    upstream_props = state.cap.get("properties", {}) or {}
    # Defensive normalization for media-preserving transforms. If the upstream
    # branch is already redacted, later crop/sample/gate/window operators should
    # not relabel it as raw. A blur/mask transform also produces redacted media.
    if v.raw_operator_id == "op.region_mask_blur" or (upstream_props.get("redacted") and v.raw_operator_id in {"op.sample", "op.window", "op.trigger_gate", "op.region_select_crop"}):
        redacted_type = redacted_media_type_for(upstream_type)
        if redacted_type in {"image/x-redacted", "video/x-redacted"}:
            new.cap["media_type"] = redacted_type
            new.cap["schema"] = "redacted_image_frame" if redacted_type.startswith("image/") else "redacted_video_stream"
            new.cap.setdefault("properties", {})["redacted"] = True
    if upstream_type == "audio/x-filtered" and v.raw_operator_id in {"op.sample", "op.window", "op.trigger_gate"}:
        new.cap["media_type"] = "audio/x-filtered"
        new.cap.setdefault("properties", {})["speech_content_removed"] = True
    new.utility_caps.update(v.utility_capabilities)
    new.transforms.update(v.transform_effects)
    new.pipeline.append({
        "operator": v.raw_operator_id,
        "variant": v.label,
        "output_cap": copy.deepcopy(new.cap),
        "parameters": copy.deepcopy(v.parameters),
    })

    eff = v.residual_effect or {}
    default_policy = eff.get("default_policy", "preserve_unmentioned")

    # Some operators replace raw data with semantic summaries. For these, unmentioned
    # attributes should not silently persist.
    if default_policy == "drop_unmentioned":
        new.residual = init_residual("none")
        # Semantic replacement outputs should not retain raw sensor primitives such as
        # audio_waveform or video_stream at the output-flow level. The raw source may
        # still have existed at raw_input, but this operator's output no longer is that
        # primitive.
        new.ci_terms["informationType.sensorPrimitive"] = set()
    elif default_policy == "remove_all_for_this_branch":
        new.residual = init_residual("none")
    elif default_policy == "attribute_wise_max_over_inputs":
        # In this single-state BFS, join cannot actually see multiple inputs.
        # We preserve current risk and add join-specific risk below.
        pass
    else:
        # preserve_unmentioned and variants: keep current vector.
        pass

    # Explicit set effects.
    for attr, value in (eff.get("set", {}) or {}).items():
        if attr in new.residual:
            new.residual[attr] = normalize_risk(value)

    # Explicit remove effects.
    for attr in (eff.get("remove", []) or []):
        if attr in new.residual:
            new.residual[attr] = "none"

    # Reductions.
    for attr, rule in (eff.get("reduce", {}) or {}).items():
        if attr in new.residual:
            new.residual[attr] = apply_reduce_rule(new.residual[attr], rule)

    # Join/fusion may increase risks through correlation.
    add = eff.get("add", {}) or {}
    for attr, value in add.items():
        if attr in new.residual and value == "may_increase":
            new.residual[attr] = risk_max(new.residual[attr], "medium")

    # CI and transmission additions.
    for k, vals in v.ci_additions.items():
        new.ci_terms.setdefault(k, set()).update(vals)
    if any(t in new.transforms for t in {"raw_audio_removed", "raw_pixels_removed", "raw_media_removed", "event_label_only", "audio_waveform_removed", "speech_content_minimized"}):
        new.ci_terms.setdefault("transmissionPrinciple", set()).add("data_minimized")

    # Some operators internally inspect speech/text but intentionally output only narrow intent.
    # Do not expose speech_transcribed as an output-flow observation for intent-only output.
    if v.raw_operator_id == "op.keyword_intent_extractor":
        new.ci_terms.get("informationType.interpretedObservation", set()).discard("speech_transcribed")

    # Output cap required info from app adapter is nested; do not add directly here.
    props = v.output_cap.get("properties", {}) or {}
    required_info = props.get("required_informationType", {})
    if isinstance(required_info, dict):
        for subkey, vals in required_info.items():
            full = f"informationType.{subkey}"
            new.ci_terms.setdefault(full, set()).update(flatten_terms(vals))

    # If semantic output has a schema, add it as an interpreted observation when useful.
    schema = v.output_cap.get("schema")
    if schema in {
        "occupancy_count", "room_occupied", "pose_keypoints", "activity_label",
        "sound_event_label", "decibel_level_duration", "keyword_or_intent",
        "fall_or_safety_event", "person_at_door_or_intrusion_event"
    }:
        if schema in {"fall_or_safety_event"}:
            new.ci_terms.setdefault("informationType.inferredInformationType", set()).add("fall_event")
        elif schema in {"person_at_door_or_intrusion_event"}:
            new.ci_terms.setdefault("informationType.inferredInformationType", set()).add("presence")
            new.ci_terms.setdefault("informationType.interpretedObservation", set()).add("person_detected")
        elif schema == "keyword_or_intent":
            # Intents intentionally do not add speech_transcribed.
            pass
        elif schema == "decibel_level_duration":
            new.ci_terms.setdefault("informationType.sensorPrimitive", set()).add("decibel_level")
        else:
            new.ci_terms.setdefault("informationType.interpretedObservation", set()).add(schema)

    return new


def adapter_allowed(input_cap: Dict[str, Any], target_cap: Dict[str, Any]) -> bool:
    target_t = target_cap.get("semantic_type") or target_cap.get("media_type")
    if not target_t:
        return False
    up_t = cap_type(input_cap)
    if goal_cap_matches(input_cap, target_cap):
        return True
    allowed = ADAPTER_ALLOWED_INPUTS.get(target_t)
    if not allowed:
        return False
    return up_t in allowed


def passthrough_output_plausible(state_cap: Dict[str, Any], v: OperatorVariant) -> bool:
    """For operators with several output caps, keep output media aligned with input.

    Contract catalogs often list parallel input/output caps (image->image,
    video->video, audio->audio).  The symbolic planner materializes each output
    cap as a variant, so we need this guard to prevent impossible transitions
    such as image/x-raw -> video/x-redacted or audio/x-raw -> video/x-raw.
    """
    in_t = cap_type(state_cap)
    out_t = cap_type(v.output_cap)
    op = v.raw_operator_id

    # Format-preserving media transforms should not change image/video/audio family.
    if op in {"op.sample", "op.window", "op.trigger_gate", "op.region_select_crop", "op.region_mask_blur"}:
        if media_family(in_t):
            if not same_media_family(in_t, out_t):
                return False
            # Backward-compatible operator catalogs may still say image/x-raw;
            # apply_residual_effect normalizes blur/mask outputs to x-redacted.
            return True
        # Non-media event/sensor gates should stay semantic.
        if op == "op.trigger_gate":
            return out_t in {"application/x-event", "application/x-fused-event", "application/x-windowed-events"} or out_t == in_t
        if op == "op.window":
            return out_t == "application/x-windowed-events"

    if op == "op.speech_content_removal":
        if in_t.startswith("audio/"):
            return out_t in {"audio/x-filtered", "application/x-command-intent"}
        return out_t in {"application/x-redacted-transcript", "application/x-command-intent"}

    return True


def can_apply_variant(state: State, v: OperatorVariant) -> bool:
    if v.raw_operator_id == "op.source":
        return False
    # Avoid repeated non-idempotent/loop-prone operators.
    prior = [p["operator"] for p in state.pipeline]
    if v.raw_operator_id in prior and v.raw_operator_id not in {"op.schema_adapter"}:
        return False

    # Special check for schema adapters.
    if v.raw_operator_id == "op.schema_adapter":
        return adapter_allowed(state.cap, v.output_cap)

    if not any(cap_matches(state.cap, in_cap) for in_cap in v.input_caps_dicts()):
        return False
    return passthrough_output_plausible(state.cap, v)


def state_signature(state: State) -> Tuple[Any, ...]:
    return (
        cap_signature(state.cap),
        tuple(sorted((k, normalize_risk(v)) for k, v in state.residual.items())),
        tuple(sorted((k, tuple(sorted(v))) for k, v in state.ci_terms.items())),
        tuple(sorted(state.utility_caps)),
        tuple(p["operator"] for p in state.pipeline[-3:]),  # keep some path context
    )


def output_cap_matches_request(state: State, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for g in request.get("utility_contract", {}).get("accepted_output_caps", []) or []:
        if goal_cap_matches(state.cap, g):
            return g
    return None


def request_quality_status(request: Dict[str, Any]) -> str:
    # Symbolic planner cannot prove numerical metrics. It marks them for validation.
    qr = request.get("utility_contract", {}).get("quality_requirements", {}) or {}
    numeric_requirements = {k: v for k, v in qr.items() if v is not None}
    if numeric_requirements:
        return "requires_runtime_or_benchmark_validation"
    return "not_required"


def utility_capability_satisfied(state: State, request: Dict[str, Any]) -> bool:
    requested = request.get("utility_contract", {}).get("requested_capability")
    if not requested:
        return True
    if requested in state.utility_caps:
        return True

    equivalents = {
        "occupancy_estimation": {"occupancy_estimation", "energy_management_support"},
        "fall_detection": {"fall_detection", "fall_detection_support", "safety_monitoring"},
        "voice_command_or_audio_event_detection": {
            "voice_command", "keyword_detection", "sound_event_detection",
            "safety_audio_event_support", "noise_monitoring"
        },
        "intrusion_or_visitor_detection": {
            "intrusion_detection", "security_support", "person_detection",
            "object_detection", "multi_sensor_confirmation", "complex_event_detection"
        },
        # Downstream-compatible app-request names. These are satisfied when the
        # candidate preserves the input interface expected by the downstream model
        # and/or performs the relevant low-level inference support. The final cap
        # still must match accepted_output_caps separately.
        "visitor_presence_inference": {
            "visitor_presence_inference", "person_detection", "object_detection",
            "security_support", "format_preserving_video", "region_specific_monitoring",
            "field_of_view_minimization", "identity_reduction",
            "provide_raw_stream", "rate_limit_stream", "periodic_monitoring",
        },
        "domestic_sound_event_inference": {
            "domestic_sound_event_inference", "sound_event_detection",
            "audio_event_detection", "audio_safety_without_speech", "voice_privacy",
            "temporal_context", "provide_raw_stream", "rate_limit_stream",
        },
        "adl_recognition": {
            "adl_recognition", "activity_detection", "activity_detection_support",
            "temporal_context", "format_preserving_video", "audio_event_detection",
            "voice_privacy", "provide_raw_stream", "rate_limit_stream",
        },
    }
    return bool(state.utility_caps & equivalents.get(requested, {requested}))


def term_group_satisfied(ci_terms: Dict[str, Set[str]], key: str, group: List[str]) -> bool:
    existing = ci_terms.get(key, set())
    if "none" in group:
        # Treat explicit "none" as satisfied when there is no term in this field,
        # or when any other alternative in the OR group is satisfied.
        if not existing:
            return True
    return any(term in existing for term in group if term != "none")


def ci_constraints_satisfied(state: State, request: Dict[str, Any]) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    constraints = request.get("ci_output_constraints", {}) or {}
    required = constraints.get("required_ci_terms", {}) or {}
    forbidden = constraints.get("forbidden_ci_terms", {}) or {}

    for key, terms in required.items():
        groups = parse_or_terms(as_list(terms))
        for group in groups:
            if group and not term_group_satisfied(state.ci_terms, key, group):
                failures.append(f"missing required CI term for {key}: one of {group}")

    for key, terms in forbidden.items():
        existing = state.ci_terms.get(key, set())
        for term in flatten_terms(terms):
            # Ignore OR in forbidden; split it too.
            for t in [p.strip() for p in re.split(r"\s+OR\s+", term) if p.strip()]:
                if t in existing:
                    failures.append(f"forbidden CI term present for {key}: {t}")

    # If media-like output is returned, required transformations must be present.
    out_t = cap_type(state.cap)
    if out_t in {"video/x-raw", "image/x-raw", "video/x-redacted", "image/x-redacted"} or out_t.startswith("video/") or out_t.startswith("image/"):
        for trans in constraints.get("required_transformations_if_media_output", []) or []:
            if trans not in state.transforms and trans not in state.ci_terms.get("transmissionPrinciple", set()):
                failures.append(f"media output missing required transformation: {trans}")

    return (not failures, failures)


def residual_constraints_satisfied(state: State, request: Dict[str, Any]) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    rc = request.get("residual_disclosure_constraints", {}) or {}
    max_allowed = rc.get("max_allowed", {}) or {}
    hard_forbidden = set(rc.get("hard_forbidden_attributes", []) or [])
    allowed_high = set(rc.get("allowed_high_attributes", []) or [])

    for attr in RESIDUAL_ATTRIBUTES:
        cand = normalize_risk(state.residual.get(attr, "unknown"))
        max_l = normalize_risk(max_allowed.get(attr, "unknown"))

        # Hard-forbidden means: respect the request's max threshold. If max is none,
        # the attribute must be absent. If max is low, narrow semantic leakage is allowed.
        if attr in hard_forbidden:
            if RISK_ORDER[cand] > RISK_ORDER[max_l]:
                failures.append(f"hard-forbidden residual {attr}={cand} exceeds max {max_l}")
            continue

        if attr in allowed_high:
            continue

        if RISK_ORDER[cand] > RISK_ORDER[max_l]:
            failures.append(f"residual {attr}={cand} exceeds max {max_l}")

    return (not failures, failures)


def maybe_add_ephemeral_drop_side_effect(state: State, request: Dict[str, Any]) -> State:
    """Add no_raw_data_retention when requested/preferred and not already present.

    This is modeled as a control/dataflow side effect: raw/intermediate artifacts are
    discarded after deriving the output. It does not change the output cap.
    """
    preferred = request.get("ci_output_constraints", {}).get("preferred_ci_terms", {}).get("transmissionPrinciple", []) or []
    assumed = request.get("ci_context", {}).get("transmissionPrinciple_assumed", []) or []
    wants_no_raw = "no_raw_data_retention" in preferred or "no_raw_data_retention" in assumed
    if wants_no_raw and "no_raw_data_retention" not in state.ci_terms.get("transmissionPrinciple", set()):
        new = state.copy()
        new.ci_terms.setdefault("transmissionPrinciple", set()).add("no_raw_data_retention")
        new.ci_terms.setdefault("transmissionPrinciple", set()).add("ephemeral_processing")
        new.transforms.add("no_raw_data_retention")
        new.transforms.add("ephemeral_processing")
        new.pipeline.append({
            "operator": "op.drop_discard",
            "variant": "Drop / Discard(raw_input_after_successor)",
            "output_cap": copy.deepcopy(new.cap),
            "parameters": {"drop_stage": "raw_input", "after_successor": "derived_output"},
        })
        return new
    return state

def add_matched_cap_required_terms(state: State, matched_cap: Dict[str, Any]) -> State:
    """Attach app/output-cap transmission terms to the final output flow.

    accepted_output_caps can declare output-stage transmission principles such as
    purpose_limited, speech_content_removed, or raw_pixels_removed. These are not
    preprocessing operators by themselves; they are part of the app-facing output
    contract for the selected cap. Adding them here lets CI checks see the complete
    flow while preserving operator residual effects separately.
    """
    required_tp = matched_cap.get("required_transmissionPrinciple", []) or []
    if not required_tp:
        return state
    new = state.copy()
    for tp in flatten_terms(required_tp):
        new.ci_terms.setdefault("transmissionPrinciple", set()).add(str(tp))
        # Keep transforms aligned for rules that check either field.
        if tp in {
            "purpose_limited", "data_minimized", "raw_pixels_removed",
            "speech_content_removed", "limited_retention", "no_raw_data_retention",
        }:
            new.transforms.add(str(tp))
    return new


def pipeline_to_record(
    state: State,
    request: Dict[str, Any],
    matched_cap: Dict[str, Any],
    preserve_staged_flows: bool = True,
) -> Dict[str, Any]:
    # Always include the app-facing output stage. In the full system we also
    # preserve raw/intermediate/retained stages declared by operator contracts;
    # the no_staged_flows ablation collapses this to output_to_application only.
    ci_serialized = {k: sorted(v) for k, v in state.ci_terms.items()}
    if preserve_staged_flows:
        stages = set(ci_serialized.get("pipelineStage", []))
        stages.add("output_to_application")
        ci_serialized["pipelineStage"] = sorted(stages)
    else:
        ci_serialized["pipelineStage"] = ["output_to_application"]

    return {
        "pipeline_id": "pipe_" + stable_hash({
            "pipeline": state.pipeline,
            "cap": state.cap,
            "residual": state.residual,
        }),
        "decision": "candidate_pipeline",
        "matched_output_cap": matched_cap.get("cap_id"),
        "matched_output_schema": matched_cap.get("schema"),
        "final_output_cap": state.cap,
        "operators": state.pipeline + [{
            "operator": "op.route_publish",
            "variant": "Route / Publish(output_to_application)",
            "output_cap": {"semantic_type": "external/application-output", "schema": "output_to_application"},
            "parameters": {
                "recipient": request.get("ci_context", {}).get("recipient", []),
                "purpose": request.get("ci_context", {}).get("purpose", []),
            },
        }],
        "utility_capabilities": sorted(state.utility_caps),
        "quality_status": request_quality_status(request),
        "ci_terms": ci_serialized,
        "transforms": sorted(state.transforms),
        "residual_disclosure": {k: normalize_risk(v) for k, v in state.residual.items()},
        "residual_score": risk_score(state.residual),
    }


def enumerate_candidates(
    operator_catalog: Dict[str, Any],
    request: Dict[str, Any],
    max_depth: int = 7,
    max_states: int = 25000,
    apply_request_ci_constraints: bool = True,
    apply_request_residual_constraints: bool = True,
    preserve_staged_flows: bool = True,
) -> Dict[str, Any]:
    operators = operator_catalog.get("operators", []) or []
    source_variants, transform_variants = materialize_variants(operators, request)

    start_states: List[State] = []
    for v in source_variants:
        if allowed_source(v, request):
            st = initial_state_from_source(v)
            for tp in request.get("ci_context", {}).get("transmissionPrinciple_assumed", []) or []:
                st.ci_terms.setdefault("transmissionPrinciple", set()).add(tp)
            start_states.append(st)

    q: deque[State] = deque(start_states)
    seen: Set[Tuple[Any, ...]] = set()
    raw_candidates: List[Tuple[State, Dict[str, Any]]] = []
    rejected_goals: List[Dict[str, Any]] = []
    states_expanded = 0

    while q and states_expanded < max_states:
        state = q.popleft()
        states_expanded += 1
        sig = state_signature(state)
        if sig in seen:
            continue
        seen.add(sig)

        matched = output_cap_matches_request(state, request)
        if matched and utility_capability_satisfied(state, request):
            final_state = maybe_add_ephemeral_drop_side_effect(state, request)
            final_state = add_matched_cap_required_terms(final_state, matched)
            if apply_request_ci_constraints:
                ci_ok, ci_fail = ci_constraints_satisfied(final_state, request)
            else:
                ci_ok, ci_fail = True, []
            if apply_request_residual_constraints:
                res_ok, res_fail = residual_constraints_satisfied(final_state, request)
            else:
                res_ok, res_fail = True, []

            # In ablation modes we may intentionally keep candidates that would
            # otherwise be removed by request-level CI/residual prefilters. The
            # later CI evaluator and selector still record whether they violate
            # hard constraints or residual bounds unless those stages are also
            # ablated.
            if ci_ok and res_ok:
                raw_candidates.append((final_state, matched))
            else:
                rejected_goals.append({
                    "pipeline": [p["operator"] for p in final_state.pipeline],
                    "final_output_cap": final_state.cap,
                    "matched_output_cap": matched.get("cap_id"),
                    "ci_failures": ci_fail,
                    "residual_failures": res_fail,
                    "residual_disclosure": final_state.residual,
                })

        if state.depth >= max_depth:
            continue

        for v in transform_variants:
            if not can_apply_variant(state, v):
                continue
            nxt = apply_residual_effect(state, v)

            # Light goal-directed pruning: if the state already violates a hard residual
            # threshold and no future semantic replacement is likely, it may still be
            # repairable by summarizers, so we do not prune aggressively here.
            q.append(nxt)

    # Deduplicate final candidate records by pipeline id and sort by least revealing
    # then accepted cap priority.
    records_by_id: Dict[str, Dict[str, Any]] = {}
    cap_priority = {
        c.get("cap_id"): c.get("priority", 999)
        for c in request.get("utility_contract", {}).get("accepted_output_caps", []) or []
    }

    for st, cap in raw_candidates:
        rec = pipeline_to_record(st, request, cap, preserve_staged_flows=preserve_staged_flows)
        records_by_id[rec["pipeline_id"]] = rec

    candidates = list(records_by_id.values())
    candidates.sort(key=lambda r: (r["residual_score"], cap_priority.get(r["matched_output_cap"], 999), len(r["operators"])))

    decision: Dict[str, Any]
    if candidates:
        decision = {
            "decision": "select_pipeline",
            "selected_pipeline_id": candidates[0]["pipeline_id"],
            "selected_output_cap": candidates[0]["matched_output_cap"],
            "reason": "Selected least-revealing feasible candidate under symbolic operator-effect metadata.",
        }
    else:
        # Try to distinguish no compatibility from CI/privacy conflicts.
        any_goal = len(rejected_goals) > 0
        decision = {
            "decision": "no_compromise" if any_goal else "insufficient_utility_evidence",
            "selected_pipeline_id": None,
            "selected_output_cap": None,
            "reason": (
                "Some pipelines matched output caps but failed CI/residual constraints."
                if any_goal else
                "No enumerated pipeline matched an accepted output cap and requested utility capability."
            ),
        }

    return {
        "schema_version": "smartpriv_pipeline_candidate_output_v1",
        "request_id": request.get("request_identity", {}).get("request_id"),
        "scenario_id": request.get("request_identity", {}).get("scenario_id"),
        "planner": {
            "algorithm": "goal-directed BFS over caps-compatible operator-effect contracts",
            "max_depth": max_depth,
            "max_states": max_states,
            "states_expanded": states_expanded,
            "states_seen": len(seen),
            "start_states": len(start_states),
            "operator_variants": len(transform_variants),
            "candidate_count": len(candidates),
            "apply_request_ci_constraints": apply_request_ci_constraints,
            "apply_request_residual_constraints": apply_request_residual_constraints,
            "preserve_staged_flows": preserve_staged_flows,
            "rejected_goal_count": len(rejected_goals),
            "notes": [
                "This planner is symbolic; use --emit-executable-specs or --emit-program-dir to assemble runtime pipelines from smartpriv_runtime.",
                "Numerical utility constraints are marked as requiring benchmark/runtime/app validation.",
                "D_probe can be incorporated later by taking attribute-wise max with residual_disclosure."
            ],
        },
        "decision": decision,
        "candidates": candidates,
        "rejected_goal_examples": rejected_goals[:25],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate symbolic privacy preprocessing pipeline candidates.")
    parser.add_argument("--operators", required=True, help="Path to operator-contract JSON.")
    parser.add_argument("--request", required=True, help="Path to structured application request JSON.")
    parser.add_argument("--out", default=None, help="Optional output JSON path.")
    parser.add_argument("--max-depth", type=int, default=7, help="Maximum number of transformations after source.")
    parser.add_argument("--max-states", type=int, default=25000, help="Maximum BFS states to expand.")
    parser.add_argument("--top-k", type=int, default=20, help="Print top-k candidate summary to stdout.")
    parser.add_argument("--emit-executable-specs", action="store_true", help="Attach runtime executable_pipeline_spec to each candidate.")
    parser.add_argument("--emit-program-dir", default=None, help="Optional directory where runnable Python programs for top candidates are written.")
    parser.add_argument("--emit-program-top-k", type=int, default=5, help="Number of runnable candidate programs to emit.")
    parser.add_argument("--skip-request-ci-constraints", action="store_true", help="Ablation: do not prefilter candidates using request-level CI output constraints.")
    parser.add_argument("--skip-request-residual-constraints", action="store_true", help="Ablation: do not prefilter candidates using request-level residual bounds.")
    parser.add_argument("--collapse-pipeline-stages", action="store_true", help="Ablation: emit only output_to_application instead of staged flow annotations.")
    args = parser.parse_args(argv)

    operators = load_json(args.operators)
    request = load_json(args.request)

    result = enumerate_candidates(
        operators,
        request,
        max_depth=args.max_depth,
        max_states=args.max_states,
        apply_request_ci_constraints=not args.skip_request_ci_constraints,
        apply_request_residual_constraints=not args.skip_request_residual_constraints,
        preserve_staged_flows=not args.collapse_pipeline_stages,
    )

    if args.emit_executable_specs or args.emit_program_dir:
        if attach_executable_specs is None:
            raise RuntimeError("--emit-executable-specs/--emit-program-dir requires the smartpriv_runtime package on PYTHONPATH")
        result = attach_executable_specs(result)

    if args.emit_program_dir:
        if emit_programs is None:
            raise RuntimeError("--emit-program-dir requires the smartpriv_runtime package on PYTHONPATH")
        emitted = emit_programs(result, args.emit_program_dir, top_k=args.emit_program_top_k)
        result.setdefault("planner", {})["emitted_programs"] = emitted

    if args.out:
        write_json(result, args.out)

    decision = result["decision"]
    print(json.dumps({
        "request_id": result["request_id"],
        "scenario_id": result["scenario_id"],
        "decision": decision,
        "candidate_count": result["planner"]["candidate_count"],
        "states_expanded": result["planner"]["states_expanded"],
        "states_seen": result["planner"]["states_seen"],
    }, indent=2))

    for i, c in enumerate(result["candidates"][:args.top_k], start=1):
        print(f"\n#{i} {c['pipeline_id']} score={c['residual_score']} output={c['matched_output_cap']}")
        print("  ops:", " -> ".join(op["operator"] for op in c["operators"]))
        print("  residual:", json.dumps(c["residual_disclosure"], sort_keys=True))

    if not result["candidates"] and result["rejected_goal_examples"]:
        print("\nRejected goal examples:")
        for ex in result["rejected_goal_examples"][:5]:
            print(json.dumps(ex, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
