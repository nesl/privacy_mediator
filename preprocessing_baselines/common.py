#!/usr/bin/env python3
"""Shared helpers for SmartPriv preprocessing baselines.

These helpers intentionally avoid depending on the full mediator driver.  The
baseline scripts can emit records that look like mediator candidate/decision
outputs while making clear which baseline policy produced them.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

SEMANTIC_FAMILIES: Dict[str, set[str]] = {
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
    "application/x-detections": {"application/x-detections"},
    "application/x-sensor-reading": {
        "application/x-sensor-reading",
        "application/x-motion-events",
        "application/x-noise-sensor",
    },
}


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


def stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def normalize_risk(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (int, float)):
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
    if "high" in s:
        return "high"
    if "medium" in s:
        return "medium"
    if "low" in s:
        return "low"
    if "none" in s or "removed" in s or "absent" in s:
        return "none"
    return "unknown"


def init_residual(default: str = "none") -> Dict[str, str]:
    return {a: normalize_risk(default) for a in RESIDUAL_ATTRIBUTES}


def residual_score(residual: Dict[str, Any]) -> int:
    weights = {
        "identity": 3,
        "face": 2,
        "speech_content": 3,
        "speaker_identity": 3,
        "trajectory": 2,
        "visible_text": 2,
    }
    vec = {a: normalize_risk(residual.get(a, "none")) for a in RESIDUAL_ATTRIBUTES}
    return sum(weights.get(a, 1) * RISK_ORDER[vec[a]] for a in RESIDUAL_ATTRIBUTES)


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def flatten_terms(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, dict):
        out: List[str] = []
        for v in x.values():
            out.extend(flatten_terms(v))
        return out
    if isinstance(x, list):
        out: List[str] = []
        for item in x:
            out.extend(flatten_terms(item))
        return out
    return [str(x)]


def cap_type(cap: Dict[str, Any]) -> str:
    return str(cap.get("semantic_type") or cap.get("media_type") or "")


def cap_schema(cap: Dict[str, Any]) -> str:
    return str(cap.get("schema") or "")


def type_matches(upstream: str, downstream: str) -> bool:
    if not downstream or downstream == "*":
        return True
    if not upstream:
        return False
    if upstream == downstream:
        return True
    if downstream in SEMANTIC_FAMILIES and upstream in SEMANTIC_FAMILIES[downstream]:
        return True
    if upstream in SEMANTIC_FAMILIES and downstream in SEMANTIC_FAMILIES[upstream]:
        return True
    if downstream == "video/x-raw" and upstream in {"video/x-raw", "video/x-redacted"}:
        return True
    if downstream == "image/x-raw" and upstream in {"image/x-raw", "image/x-redacted"}:
        return True
    if downstream == "audio/x-raw" and upstream in {"audio/x-raw", "audio/x-filtered"}:
        return True
    return False


def goal_cap_matches(out_cap: Dict[str, Any], goal_cap: Dict[str, Any]) -> bool:
    goal_t = str(goal_cap.get("semantic_type") or goal_cap.get("media_type") or "")
    out_t = cap_type(out_cap)
    if goal_t and type_matches(out_t, goal_t):
        return True
    goal_schema = str(goal_cap.get("schema") or "")
    if goal_schema and cap_schema(out_cap) == goal_schema:
        return True
    return False


def rejected_cap_matches(out_cap: Dict[str, Any], rejected_cap: Dict[str, Any]) -> bool:
    """Return True when an explicitly rejected cap names this exact output.

    Explicit rejections should be stricter than accepted-cap matching: do not
    apply raw/redacted or raw/filtered interface compatibility here, otherwise a
    request that rejects raw audio would accidentally reject speech-filtered
    audio as well.
    """
    r_t = str(rejected_cap.get("semantic_type") or rejected_cap.get("media_type") or "")
    if r_t and cap_type(out_cap) == r_t:
        return True
    r_schema = cap_schema(rejected_cap)
    return bool(r_schema and cap_schema(out_cap) == r_schema)


def request_checkpoint_metadata(request: Dict[str, Any]) -> Dict[str, Any]:
    """Collect checkpoint/runtime metadata used by conditional accepted caps."""
    meta: Dict[str, Any] = {}
    for key in ["checkpoint_metadata", "runtime_metadata"]:
        value = request.get(key)
        if isinstance(value, dict):
            nested = value.get("checkpoint_metadata")
            if isinstance(nested, dict):
                meta.update(nested)
            else:
                meta.update(value)

    downstream = ((request.get("utility_contract", {}) or {}).get("downstream_program", {}) or {})
    if downstream.get("default_checkpoint_modality") and "modality" not in meta:
        meta["modality"] = downstream["default_checkpoint_modality"]
    return meta


def accepted_cap_conditions_satisfied(goal_cap: Dict[str, Any], request: Dict[str, Any]) -> bool:
    """Enforce conditional_on_checkpoint_metadata for accepted caps.

    This prevents fixed-interface baselines from treating conditional audio-only
    or video-only caps as eligible when the request's checkpoint is actually AV.
    """
    cond = goal_cap.get("conditional_on_checkpoint_metadata")
    if not cond:
        return True
    meta = request_checkpoint_metadata(request)
    for key, allowed_values in cond.items():
        actual = meta.get(key)
        if actual is None:
            return False
        allowed = {str(v).strip().lower() for v in as_list(allowed_values)}
        actual_values = {str(v).strip().lower() for v in as_list(actual)}
        if not (actual_values & allowed):
            return False
    return True


def transformations_allowed_by_cap(goal_cap: Dict[str, Any], transforms: Optional[Sequence[str]] = None) -> bool:
    forbidden = {str(t) for t in as_list(goal_cap.get("forbidden_transformations"))}
    if not forbidden:
        return True
    return not (forbidden & {str(t) for t in (transforms or [])})


def _inferred_output_property(out_cap: Dict[str, Any], key: str, transforms: Optional[Sequence[str]] = None) -> Any:
    """Infer important output booleans from cap properties, components, and transforms."""
    props = out_cap.get("properties", {}) or {}
    if key in props:
        return props.get(key)
    transform_set = {str(t) for t in (transforms or [])}
    components = props.get("components") if isinstance(props.get("components"), list) else []
    component_types = {str(c.get("media_type") or c.get("semantic_type") or "") for c in components if isinstance(c, dict)}
    if key == "youhome_av_compatible":
        return cap_type(out_cap) == "application/x-youhome-av-sample" or bool(props.get("av_synchronized"))
    if key in {"visual_redacted", "redacted"}:
        return bool(component_types & {"image/x-redacted", "video/x-redacted"} or transform_set & {"face_blurred", "identity_removed", "visible_text_removed", "screen_content_removed", "body_blurred"})
    if key in {"speech_content_removed", "audio_filtered"}:
        return bool(component_types & {"audio/x-filtered"} or cap_type(out_cap) == "audio/x-filtered" or transform_set & {"speech_content_removed", "speech_content_minimized"})
    return None


def accepted_cap_properties_satisfied(out_cap: Dict[str, Any], goal_cap: Dict[str, Any], transforms: Optional[Sequence[str]] = None) -> bool:
    """Require scalar accepted-cap properties to describe the actual output."""
    goal_props = goal_cap.get("properties", {}) or {}
    for key, expected in goal_props.items():
        if isinstance(expected, (dict, list)):
            continue
        actual = _inferred_output_property(out_cap, str(key), transforms)
        if isinstance(expected, bool):
            if bool(actual) != expected:
                return False
        elif actual != expected:
            return False
    return True


def _accepted_cap_specificity_score(out_cap: Dict[str, Any], goal_cap: Dict[str, Any], transforms: Optional[Sequence[str]] = None) -> Tuple[int, int, int, int]:
    """Score how specifically an accepted cap describes the actual output."""
    goal_props = goal_cap.get("properties", {}) or {}
    matched = 0
    declared = 0
    for k, v in goal_props.items():
        if isinstance(v, (dict, list)):
            continue
        declared += 1
        actual = _inferred_output_property(out_cap, str(k), transforms)
        if isinstance(v, bool):
            if bool(actual) == v:
                matched += 1
        elif actual == v:
            matched += 1
    try:
        priority = int(goal_cap.get("priority", 999))
    except Exception:
        priority = 999
    return (matched, -(declared - matched), len(goal_props), -priority)


def first_matching_accepted_cap(
    out_cap: Dict[str, Any],
    request: Dict[str, Any],
    transforms: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Find the best request accepted cap for a baseline output.

    Mirrors the full mediator's compatibility checks closely enough for raw and
    manual baselines:
      * explicit rejections are respected;
      * conditional caps are skipped unless checkpoint/runtime metadata matches;
      * exact type/schema matches are preferred before loose interface matches;
      * per-cap forbidden transformations are respected.
    """
    utility = request.get("utility_contract", {}) or {}
    for rej in utility.get("explicitly_rejected_output_caps", []) or []:
        if rejected_cap_matches(out_cap, rej):
            return None

    caps = [
        c for c in utility.get("accepted_output_caps", []) or []
        if accepted_cap_conditions_satisfied(c, request)
        and transformations_allowed_by_cap(c, transforms)
        and accepted_cap_properties_satisfied(out_cap, c, transforms)
    ]
    out_t = cap_type(out_cap)
    out_schema = cap_schema(out_cap)

    # Prefer exact advertised caps. This prevents, e.g., audio/x-filtered from
    # matching a broader raw-audio cap before the explicit speech-removed cap.
    # When multiple accepted caps share the same type/schema, prefer the one that
    # is more specific to the actual output's properties.
    exact_type_schema = [
        cap for cap in caps
        if out_t and out_t == cap_type(cap) and (not out_schema or not cap_schema(cap) or out_schema == cap_schema(cap))
    ]
    if exact_type_schema:
        return max(exact_type_schema, key=lambda cap: _accepted_cap_specificity_score(out_cap, cap, transforms))

    exact_schema = [cap for cap in caps if out_schema and out_schema == cap_schema(cap)]
    if exact_schema:
        return max(exact_schema, key=lambda cap: _accepted_cap_specificity_score(out_cap, cap, transforms))

    loose_matches = [cap for cap in caps if goal_cap_matches(out_cap, cap)]
    if loose_matches:
        return max(loose_matches, key=lambda cap: _accepted_cap_specificity_score(out_cap, cap, transforms))
    return None


def request_identity(request: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    ident = request.get("request_identity", {}) or {}
    return ident.get("request_id"), ident.get("scenario_id")


def import_module_from_path(module_name: str, path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def infer_generator_path(explicit_path: Optional[str | Path] = None) -> Optional[Path]:
    if explicit_path:
        p = Path(explicit_path)
        return p if p.exists() else None
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "generate_pipeline_candidates.py",
        here.parent / "mediator" / "generate_pipeline_candidates.py",
        Path.cwd() / "generate_pipeline_candidates.py",
        Path.cwd() / "mediator" / "generate_pipeline_candidates.py",
        Path("/mnt/data/generate_pipeline_candidates_final_cleanup.py"),
        Path("/mnt/data/generate_pipeline_candidates_matched_cap_specificity.py"),
        Path("/mnt/data/generate_pipeline_candidates_av_component_preserve_fixed.py"),
        Path("/mnt/data/generate_pipeline_candidates_av_component_preserve.py"),
        Path("/mnt/data/generate_pipeline_candidates_av_updated.py"),
        Path("/mnt/data/generate_pipeline_candidates_patched.py"),
        Path("/mnt/data/generate_pipeline_candidates.py"),
        Path("/mnt/data/generate_pipeline_candidates(12).py"),
        Path("/mnt/data/generate_pipeline_candidates(4).py"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_generator(explicit_path: Optional[str | Path] = None):
    path = infer_generator_path(explicit_path)
    if not path:
        return None
    return import_module_from_path("smartpriv_baseline_candidate_generator", path)


def route_publish_operator(request: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "operator": "op.route_publish",
        "variant": "Route / Publish(output_to_application)",
        "output_cap": {"semantic_type": "external/application-output", "schema": "output_to_application"},
        "parameters": {
            "recipient": (request.get("ci_context", {}) or {}).get("recipient", []),
            "purpose": (request.get("ci_context", {}) or {}).get("purpose", []),
        },
    }


def quality_status_for_request(request: Dict[str, Any]) -> str:
    qr = (request.get("utility_contract", {}) or {}).get("quality_requirements", {}) or {}
    return "requires_runtime_or_benchmark_validation" if qr else "not_required"


def cap_content_type(cap: Dict[str, Any]) -> Optional[str]:
    val = cap.get("content_type") or cap.get("contentType")
    if val:
        return str(val)
    t = cap_type(cap)
    schema = cap_schema(cap)
    if "video" in t or "clip" in schema:
        return "video_content"
    if "image" in t or "frame" in schema:
        return "image_content"
    if "youhome-av" in t or "youhome_av" in schema or t == "application/x-youhome-av-sample":
        return "av_content"
    if "audio" in t:
        return "audio_content"
    return None


def allowed_source_cap(cap: Dict[str, Any], request: Dict[str, Any]) -> bool:
    req = request.get("source_requirements", {}) or {}
    allowed_content = set(map(str, req.get("allowed_content_types", []) or []))
    forbidden_content = set(map(str, req.get("forbidden_content_types", []) or []))
    content = cap_content_type(cap)
    if content and content in forbidden_content:
        return False
    if allowed_content:
        if content and content in allowed_content:
            return True
        # Synchronized AV samples are compatible only when the request explicitly
        # allows AV content or both visual and audio content.  Generic input_data
        # is intentionally not enough, and a request forbidding audio/visual
        # content rejects the AV container.
        if content == "av_content":
            if "audio_content" in forbidden_content or ({"image_content", "video_content"} & forbidden_content):
                return False
            if (
                "av_content" in allowed_content
                or ({"image_content", "video_content"} & allowed_content and "audio_content" in allowed_content)
            ):
                return True
            return False
        # Sensor readings are generic input/environmental data.
        if cap_type(cap) == "application/x-sensor-reading" and "input_data" in allowed_content:
            return True
        return False
    return True


def infer_source_residual_from_cap(cap: Dict[str, Any], source_operator: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    residual = init_residual("none")
    content = cap_content_type(cap)
    if source_operator:
        init_by_mod = ((source_operator.get("residual_disclosure_effect", {}) or {}).get("initial_state_by_modality", {}) or {})
        if content and content in init_by_mod:
            for k, v in init_by_mod[content].items():
                if k in residual:
                    residual[k] = normalize_risk(v)
            return residual
    if content == "av_content" or cap_type(cap) == "application/x-youhome-av-sample":
        residual.update({
            "identity": "high",
            "face": "high",
            "body_shape": "high",
            "clothing": "high",
            "gait": "medium",
            "speech_content": "high",
            "speaker_identity": "high",
            "activity": "high",
            "location": "medium",
            "trajectory": "medium",
            "co_presence": "high",
            "visible_text": "medium",
            "aggregate_presence": "high",
        })
    elif content in {"video_content", "image_content"} or cap_type(cap) in {"video/x-raw", "image/x-raw"}:
        residual.update({
            "identity": "high",
            "face": "high",
            "body_shape": "high",
            "clothing": "high",
            "gait": "medium" if cap_type(cap).startswith("video") else "low",
            "activity": "high",
            "location": "medium",
            "trajectory": "medium" if cap_type(cap).startswith("video") else "low",
            "co_presence": "medium",
            "visible_text": "medium",
            "aggregate_presence": "high",
        })
    elif content == "audio_content" or cap_type(cap) == "audio/x-raw":
        residual.update({
            "speech_content": "high",
            "speaker_identity": "high",
            "activity": "medium",
            "location": "low",
            "co_presence": "medium",
        })
    elif cap_type(cap) in {"application/x-occupancy-count", "application/x-count", "application/x-binary-occupancy"}:
        residual.update({"aggregate_presence": "medium", "location": "low", "activity": "low"})
    elif cap_type(cap) in {"application/x-detections", "application/x-security-event", "application/x-event"}:
        residual.update({"activity": "medium", "location": "medium", "co_presence": "low", "aggregate_presence": "medium"})
    return residual


def source_candidates_from_catalog(operator_catalog: Dict[str, Any], request: Dict[str, Any], generator: Any = None) -> List[Dict[str, Any]]:
    """Return source-only candidate-like states.

    Each item contains cap, residual, ci_terms, transforms, utility_capabilities, and
    operators.  If the symbolic generator is available, reuse its source materializer;
    otherwise fall back to parsing op.source output caps directly.
    """
    operators = operator_catalog.get("operators", []) or []
    out: List[Dict[str, Any]] = []

    if generator and all(hasattr(generator, name) for name in ["materialize_variants", "allowed_source", "initial_state_from_source"]):
        try:
            source_variants, _ = generator.materialize_variants(operators, request)
            for v in source_variants:
                if not generator.allowed_source(v, request):
                    continue
                st = generator.initial_state_from_source(v)
                ci_terms = {k: sorted(list(vals)) for k, vals in getattr(st, "ci_terms", {}).items()}
                for tp in (request.get("ci_context", {}) or {}).get("transmissionPrinciple_assumed", []) or []:
                    ci_terms.setdefault("transmissionPrinciple", [])
                    if tp not in ci_terms["transmissionPrinciple"]:
                        ci_terms["transmissionPrinciple"].append(tp)
                out.append({
                    "cap": copy.deepcopy(getattr(st, "cap", v.output_cap)),
                    "residual": {k: normalize_risk(val) for k, val in getattr(st, "residual", {}).items()},
                    "ci_terms": ci_terms,
                    "transforms": sorted(list(getattr(st, "transforms", set()))),
                    "utility_capabilities": sorted(list(getattr(st, "utility_caps", set(v.utility_capabilities)))),
                    "operators": [{
                        "operator": "op.source",
                        "variant": getattr(v, "label", "Source"),
                        "output_cap": copy.deepcopy(getattr(st, "cap", v.output_cap)),
                        "parameters": getattr(v, "parameters", {}) or {},
                    }],
                    "raw_operator_id": getattr(v, "raw_operator_id", "op.source"),
                })
            if out:
                return out
        except Exception:
            # Fall through to catalog parsing.
            pass

    for op in operators:
        if op.get("id") != "op.source":
            continue
        for idx, cap in enumerate(op.get("output_caps", []) or []):
            if not allowed_source_cap(cap, request):
                continue
            residual = infer_source_residual_from_cap(cap, op)
            ci_terms: Dict[str, List[str]] = {}
            ann = op.get("ci_annotations", {}) or {}
            for k, v in ann.items():
                # Source annotations list all possible primitives; add only the
                # primitive(s) corresponding to the selected source cap below.
                if k not in {"pipelineStage", "informationType.sensorPrimitive", "sensingDataMetadata"}:
                    ci_terms[k] = sorted(set(flatten_terms(v)))
            prims = flatten_terms((cap.get("properties", {}) or {}).get("sensorPrimitive"))
            if not prims:
                if cap_content_type(cap) == "video_content":
                    prims = ["video_stream"]
                elif cap_content_type(cap) == "image_content":
                    prims = ["image_frame"]
                elif cap_content_type(cap) == "audio_content":
                    prims = ["audio_waveform"]
                elif cap_content_type(cap) == "av_content":
                    prims = ["image_frame", "audio_waveform"]
            if prims:
                ci_terms["informationType.sensorPrimitive"] = sorted(set(map(str, prims)))
            for tp in (request.get("ci_context", {}) or {}).get("transmissionPrinciple_assumed", []) or []:
                ci_terms.setdefault("transmissionPrinciple", [])
                if tp not in ci_terms["transmissionPrinciple"]:
                    ci_terms["transmissionPrinciple"].append(tp)
            out.append({
                "cap": copy.deepcopy(cap),
                "residual": residual,
                "ci_terms": ci_terms,
                "transforms": [],
                "utility_capabilities": sorted(set(flatten_terms(op.get("utility_capabilities", [])))),
                "operators": [{
                    "operator": "op.source",
                    "variant": f"Source({idx})",
                    "output_cap": copy.deepcopy(cap),
                    "parameters": {},
                }],
                "raw_operator_id": "op.source",
            })
    return out


def normalize_final_output_cap_for_reporting(cap: Dict[str, Any], matched_cap: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = copy.deepcopy(cap)
    if matched_cap and not out.get("schema") and matched_cap.get("schema"):
        out["schema"] = matched_cap.get("schema")
    return out


def make_candidate_record(
    *,
    baseline_name: str,
    request: Dict[str, Any],
    operators: List[Dict[str, Any]],
    final_output_cap: Dict[str, Any],
    residual: Dict[str, Any],
    ci_terms: Optional[Dict[str, Sequence[str]]] = None,
    transforms: Optional[Sequence[str]] = None,
    utility_capabilities: Optional[Sequence[str]] = None,
    matched_output_cap: Optional[Dict[str, Any]] = None,
    quality_status: Optional[str] = None,
    notes: Optional[Sequence[str]] = None,
    executable_under_catalog: Optional[bool] = None,
) -> Dict[str, Any]:
    match = matched_output_cap if matched_output_cap is not None else first_matching_accepted_cap(final_output_cap, request, transforms=transforms)
    final_cap_for_record = normalize_final_output_cap_for_reporting(final_output_cap, match)
    ci_serialized: Dict[str, List[str]] = {}
    for k, v in (ci_terms or {}).items():
        ci_serialized[k] = sorted(set(map(str, v)))
    ci_serialized["pipelineStage"] = ["output_to_application"]
    ops = copy.deepcopy(operators)
    if not ops or ops[-1].get("operator") != "op.route_publish":
        ops.append(route_publish_operator(request))
    residual_norm = init_residual("none")
    for a in RESIDUAL_ATTRIBUTES:
        if residual and a in residual:
            residual_norm[a] = normalize_risk(residual[a])
    cid = f"baseline_{baseline_name}_" + stable_hash({
        "operators": ops,
        "cap": final_cap_for_record,
        "residual": residual_norm,
    })
    rec = {
        "pipeline_id": cid,
        "decision": "baseline_candidate",
        "baseline": baseline_name,
        "matched_output_cap": match.get("cap_id") if match else None,
        "matched_output_schema": match.get("schema") if match else None,
        "final_output_cap": final_cap_for_record,
        "operators": ops,
        "utility_capabilities": sorted(set(map(str, utility_capabilities or []))),
        "quality_status": quality_status or quality_status_for_request(request),
        "ci_terms": ci_serialized,
        "transforms": sorted(set(map(str, transforms or []))),
        "residual_disclosure": residual_norm,
        "residual_score": residual_score(residual_norm),
        "baseline_notes": list(notes or []),
    }
    if executable_under_catalog is not None:
        rec["executable_under_catalog"] = bool(executable_under_catalog)
    return rec


def wrap_baseline_output(
    *,
    baseline_name: str,
    request: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    selected_pipeline_id: Optional[str],
    decision: str,
    reason: str,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    request_id, scenario_id = request_identity(request)
    selected = next((c for c in candidates if c.get("pipeline_id") == selected_pipeline_id), None)
    return {
        "schema_version": "smartpriv_preprocessing_baseline_output_v1",
        "baseline": baseline_name,
        "request_id": request_id,
        "scenario_id": scenario_id,
        "decision": {
            "decision": decision,
            "selected_pipeline_id": selected_pipeline_id,
            "selected_output_cap": selected.get("matched_output_cap") if selected else None,
            "reason": reason,
        },
        "candidates": candidates,
        "diagnostics": diagnostics or {},
    }


def relaxed_request_for_baselines(request: Dict[str, Any]) -> Dict[str, Any]:
    """Remove context-sensitive privacy constraints for baselines that should be app-generic.

    This keeps utility contracts and source requirements, but does not let the
    manual/LLM baselines benefit from the system's residual/CI filters.
    """
    r = copy.deepcopy(request)
    r["ci_output_constraints"] = {}
    r["residual_disclosure_constraints"] = {}
    return r


def candidate_operator_ids(candidate: Dict[str, Any]) -> List[str]:
    return [str(op.get("operator")) for op in candidate.get("operators", []) or []]


def strip_json_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"```$", "", s).strip()
    return s
