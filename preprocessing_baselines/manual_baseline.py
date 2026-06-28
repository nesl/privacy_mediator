#!/usr/bin/env python3
"""Manual preprocessing baseline for legacy and flexible SmartPriv requests.

The manual baseline is intentionally a fixed engineering policy, not a full
contextual-integrity mediator.  It is keyed only by task and coarse physical
space.  In flexible mode it may use reusable perceptual primitives, but it does
not collapse to final task decisions unless no richer compatible manual target
exists.  This makes it a useful comparison point against the full mediator:
manual is task/space-aware and utility-aware, but not CI-rule-aware.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .common import (
        as_list,
        candidate_operator_ids,
        cap_schema,
        cap_type,
        first_matching_accepted_cap,
        load_generator,
        load_json,
        make_candidate_record,
        residual_score,
        type_matches,
        wrap_baseline_output,
        write_json,
    )
except ImportError:  # pragma: no cover
    from common import (  # type: ignore
        as_list,
        candidate_operator_ids,
        cap_schema,
        cap_type,
        first_matching_accepted_cap,
        load_generator,
        load_json,
        make_candidate_record,
        residual_score,
        type_matches,
        wrap_baseline_output,
        write_json,
    )

RESIDUAL_ATTRIBUTES = [
    "identity", "face", "body_shape", "clothing", "gait", "speech_content",
    "speaker_identity", "activity", "location", "trajectory", "co_presence",
    "visible_text", "aggregate_presence",
]

SENSITIVE_VISUAL_SPACES = {"bedroom", "bathroom", "patient_room", "living_room", "workspace", "kitchen"}
SENSITIVE_AUDIO_SPACES = {"bedroom", "bathroom", "patient_room", "workspace"}

TASK_ALIASES = {
    "visitor_presence_detection": {"visitor", "presence", "intrusion", "security", "entry", "chokepoint"},
    "fall_detection": {"fall", "elder", "safety", "le2i", "pose"},
    "adl_recognition": {"adl", "activity", "routine", "youhome", "audio_visual", "av"},
    "domestic_sound_monitoring": {"domestic", "sound", "audio", "noise", "voice", "speech", "chime", "chime_home", "chimehome"},
}


@dataclasses.dataclass(frozen=True)
class ManualPolicy:
    name: str
    task: str
    spaces: Tuple[str, ...]
    mode: str
    target_cap_types: Tuple[str, ...]
    target_schema_keywords: Tuple[str, ...]
    preferred_operator_ids: Tuple[str, ...]
    avoid_boundary_roles: Tuple[str, ...]
    fallback_transforms: Tuple[str, ...]
    fallback_residual: Dict[str, str]
    rationale: str


# Legacy/downstream-compatible policies preserve the original downstream model's
# fixed input interface.  Flexible policies may select reusable primitives, but
# avoid final-task-decision outputs by default.
POLICIES: Tuple[ManualPolicy, ...] = (
    ManualPolicy(
        name="legacy_visitor_redacted_frame",
        task="visitor_presence_detection",
        spaces=("bedroom", "bathroom", "common_area", "entrance", "kitchen", "living_room", "outdoor", "patient_room", "workspace"),
        mode="legacy",
        target_cap_types=("image/x-redacted", "image/x-raw", "video/x-redacted", "video/x-raw"),
        target_schema_keywords=("redacted_image_frame", "raw_image_frame", "image_frame", "video"),
        preferred_operator_ids=("op.region_select_crop", "op.region_mask_blur", "op.sample"),
        avoid_boundary_roles=("final_task_decision_boundary",),
        fallback_transforms=("field_of_view_minimized", "face_blurred", "identity_removed", "data_minimized"),
        fallback_residual={"identity": "medium", "face": "low", "activity": "medium", "location": "medium", "aggregate_presence": "high"},
        rationale="Legacy visitor monitoring keeps an image/video interface so YOLO-style downstream inference can run.",
    ),
    ManualPolicy(
        name="flex_visitor_detections_common_space",
        task="visitor_presence_detection",
        spaces=("common_area", "entrance", "living_room", "outdoor"),
        mode="flexible",
        target_cap_types=("application/x-detections", "application/x-occupancy", "application/x-occupancy-count"),
        target_schema_keywords=("object_detections", "person_detections", "occupancy_count"),
        preferred_operator_ids=("op.person_object_detector", "op.region_select_crop", "op.sample"),
        avoid_boundary_roles=("final_task_decision_boundary",),
        fallback_transforms=("object_detection", "field_of_view_minimized", "raw_pixels_removed", "data_minimized"),
        fallback_residual={"identity": "low", "face": "none", "activity": "medium", "location": "low", "aggregate_presence": "medium"},
        rationale="Flexible visitor monitoring in entry/common spaces returns person detections when possible: richer than a binary decision but less revealing than frames.",
    ),
    ManualPolicy(
        name="flex_visitor_occupancy_sensitive_space",
        task="visitor_presence_detection",
        spaces=("bedroom", "bathroom", "kitchen", "patient_room", "workspace"),
        mode="flexible",
        target_cap_types=("application/x-occupancy", "application/x-occupancy-count", "application/x-binary-occupancy"),
        target_schema_keywords=("occupancy_count", "room_occupied", "person_count"),
        preferred_operator_ids=("op.person_object_detector", "op.occupancy_deriver", "op.aggregate_generalize", "op.schema_adapter"),
        avoid_boundary_roles=("final_task_decision_boundary",),
        fallback_transforms=("occupancy_count", "raw_pixels_removed", "aggregate_or_count_only", "data_minimized"),
        fallback_residual={"identity": "none", "face": "none", "activity": "low", "location": "low", "aggregate_presence": "medium"},
        rationale="In sensitive visual spaces the fixed manual flexible policy uses occupancy/count outputs rather than detections or images.",
    ),
    ManualPolicy(
        name="fall_pose_keypoints",
        task="fall_detection",
        spaces=("bedroom", "bathroom", "common_area", "entrance", "kitchen", "living_room", "outdoor", "patient_room", "workspace"),
        mode="any",
        target_cap_types=("application/x-pose-keypoints",),
        target_schema_keywords=("pose_keypoints", "pose"),
        preferred_operator_ids=("op.pose_extractor", "op.sample", "op.window"),
        avoid_boundary_roles=("final_task_decision_boundary",),
        fallback_transforms=("pose_keypoints", "raw_pixels_removed", "data_minimized"),
        fallback_residual={"identity": "low", "face": "none", "body_shape": "medium", "gait": "medium", "activity": "medium", "location": "medium", "aggregate_presence": "high"},
        rationale="Fall detection manual policy returns pose keypoints, preserving downstream fall-model utility without releasing raw video.",
    ),
    ManualPolicy(
        name="legacy_adl_youhome_av_sample",
        task="adl_recognition",
        spaces=("bedroom", "bathroom", "common_area", "entrance", "kitchen", "living_room", "outdoor", "patient_room", "workspace"),
        mode="legacy",
        target_cap_types=("application/x-youhome-av-sample",),
        target_schema_keywords=("youhome_av_manifest_or_sample", "youhome_av"),
        preferred_operator_ids=("op.av_visual_redact", "op.av_audio_speech_filter", "op.youhome_av_adapter", "op.sample", "op.window"),
        avoid_boundary_roles=("final_task_decision_boundary",),
        fallback_transforms=("synchronized_av_window", "face_blurred", "speech_content_removed", "data_minimized"),
        fallback_residual={"identity": "medium", "face": "low", "speech_content": "low", "speaker_identity": "low", "activity": "high", "location": "medium", "trajectory": "medium", "co_presence": "medium", "aggregate_presence": "high"},
        rationale="Legacy YouHome ADL keeps the AV sample interface required by the trained downstream classifier.",
    ),
    ManualPolicy(
        name="flex_adl_multimodal_primitives_sensitive_space",
        task="adl_recognition",
        spaces=("bedroom", "bathroom", "patient_room", "workspace"),
        mode="flexible",
        target_cap_types=("application/x-multimodal-primitives", "application/x-pose-keypoints", "application/x-detections", "application/x-sound-event"),
        target_schema_keywords=("multimodal_primitive", "pose_keypoints", "object_detections", "sound_event_label"),
        preferred_operator_ids=("op.pose_extractor", "op.person_object_detector", "op.speech_sound_classifier", "op.multimodal_primitive_fuser"),
        avoid_boundary_roles=("final_task_decision_boundary",),
        fallback_transforms=("multimodal_primitives", "raw_media_removed", "data_minimized"),
        fallback_residual={"identity": "low", "face": "none", "speech_content": "none", "speaker_identity": "low", "activity": "medium", "location": "medium", "co_presence": "medium", "aggregate_presence": "high"},
        rationale="Flexible ADL in sensitive spaces prefers reusable multimodal primitives rather than a final ADL label or raw AV sample.",
    ),
    ManualPolicy(
        name="flex_adl_youhome_av_or_multimodal_common_space",
        task="adl_recognition",
        spaces=("common_area", "entrance", "kitchen", "living_room", "outdoor"),
        mode="flexible",
        target_cap_types=("application/x-youhome-av-sample", "application/x-multimodal-primitives", "application/x-pose-keypoints", "application/x-detections"),
        target_schema_keywords=("youhome_av", "multimodal_primitive", "pose_keypoints", "object_detections"),
        preferred_operator_ids=("op.youhome_av_adapter", "op.av_visual_redact", "op.pose_extractor", "op.person_object_detector", "op.multimodal_primitive_fuser"),
        avoid_boundary_roles=("final_task_decision_boundary",),
        fallback_transforms=("synchronized_av_window", "optional_visual_redaction", "data_minimized"),
        fallback_residual={"identity": "medium", "face": "medium", "speech_content": "medium", "speaker_identity": "medium", "activity": "high", "location": "medium", "trajectory": "medium", "co_presence": "medium", "aggregate_presence": "high"},
        rationale="Flexible ADL in common spaces prefers richer AV or multimodal evidence over a coarse/final activity label.",
    ),
    ManualPolicy(
        name="legacy_audio_filtered_waveform",
        task="domestic_sound_monitoring",
        spaces=("bedroom", "bathroom", "common_area", "entrance", "kitchen", "living_room", "outdoor", "patient_room", "workspace"),
        mode="legacy",
        target_cap_types=("audio/x-filtered", "audio/x-raw"),
        target_schema_keywords=("speech_removed_audio_waveform", "raw_audio_waveform", "audio"),
        preferred_operator_ids=("op.speech_content_removal", "op.sample", "op.window"),
        avoid_boundary_roles=("final_task_decision_boundary",),
        fallback_transforms=("speech_content_removed", "data_minimized"),
        fallback_residual={"speech_content": "low", "speaker_identity": "low", "activity": "medium", "location": "low", "co_presence": "medium"},
        rationale="Legacy CHiME utility uses waveform-like audio chunks, preferably speech-filtered.",
    ),
    ManualPolicy(
        name="flex_audio_filtered_waveform_or_sound_events",
        task="domestic_sound_monitoring",
        spaces=("bedroom", "bathroom", "common_area", "entrance", "kitchen", "living_room", "outdoor", "patient_room", "workspace"),
        mode="flexible",
        target_cap_types=("audio/x-filtered", "application/x-sound-event", "application/x-sound-event-label", "application/x-decibel-level"),
        target_schema_keywords=("speech_removed_audio_waveform", "sound_event_label", "decibel"),
        preferred_operator_ids=("op.speech_content_removal", "op.speech_sound_classifier", "op.audio_level_extractor", "op.sample", "op.window"),
        avoid_boundary_roles=("final_task_decision_boundary",),
        fallback_transforms=("speech_content_removed", "sound_event_labels_or_filtered_waveform", "data_minimized"),
        fallback_residual={"speech_content": "low", "speaker_identity": "low", "activity": "medium", "location": "low", "co_presence": "medium"},
        rationale="Flexible domestic audio keeps filtered waveform when utility needs richer evidence, otherwise sound-event labels are acceptable primitives.",
    ),
)


def normalize_term(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def is_flexible_request(request: Dict[str, Any]) -> bool:
    uc = request.get("utility_contract", {}) or {}
    ident = request.get("request_identity", {}) or {}
    return bool(
        request.get("flexible_tag") is True
        or request.get("request_family") == "flexible_multi_output_app_request"
        or uc.get("interface_model") == "multi_representation_utility_contract"
        or ident.get("request_family") == "flexible_multi_output_app_request"
        or ident.get("interface_model") == "multi_representation_utility_contract"
        or ident.get("flexible_tag") is True
    )


def infer_task_from_request(request: Dict[str, Any], explicit_task: Optional[str] = None) -> str:
    if explicit_task:
        return normalize_term(explicit_task)
    rid = request.get("request_identity", {}) or {}
    uc = request.get("utility_contract", {}) or {}
    # Trust explicit machine-readable task fields before broad keyword aliases.
    for key in ["application_category", "task", "requested_capability"]:
        raw = rid.get(key) if key != "requested_capability" else uc.get(key)
        norm = normalize_term(raw)
        if norm in TASK_ALIASES:
            return norm
    requested = normalize_term(uc.get("requested_capability"))
    explicit_map = {
        "visitor_presence_inference": "visitor_presence_detection",
        "domestic_sound_event_inference": "domestic_sound_monitoring",
        "sound_event_detection": "domestic_sound_monitoring",
        "fall_detection": "fall_detection",
        "adl_recognition": "adl_recognition",
    }
    if requested in explicit_map:
        return explicit_map[requested]
    ctx = request.get("ci_context", {}) or {}
    parts: List[str] = []
    for key in ["request_id", "scenario_id", "application_name", "natural_language_request"]:
        parts.append(str(rid.get(key, "")))
    for key in ["task_description"]:
        parts.append(str(uc.get(key, "")))
    parts.extend(map(str, as_list(ctx.get("purpose"))))
    text = " ".join(parts).lower()
    # Prefer distinctive dataset/app names before generic terms like activity/safety.
    if "youhome" in text or "adl" in text or "audio_visual" in text:
        return "adl_recognition"
    if "chime" in text or "domestic_sound" in text or "sound event" in text or "audio" in text:
        return "domestic_sound_monitoring"
    if "le2i" in text or "fall" in text:
        return "fall_detection"
    if "chokepoint" in text or "visitor" in text or "presence" in text:
        return "visitor_presence_detection"
    for canonical, aliases in TASK_ALIASES.items():
        if canonical in text or any(alias in text for alias in aliases):
            return canonical
    return normalize_term(uc.get("requested_capability") or rid.get("application_category") or "unknown_task")


def infer_space_from_request(request: Dict[str, Any], explicit_space: Optional[str] = None) -> str:
    if explicit_space:
        return normalize_term(explicit_space)
    spaces = [normalize_term(v) for v in as_list((request.get("ci_context", {}) or {}).get("space")) if v]
    generic = {"indoor", "interior_home_space"}
    for s in spaces:
        if s not in generic:
            return s
    return spaces[0] if spaces else "unknown_space"


def policy_mode_matches(policy: ManualPolicy, flexible: bool) -> bool:
    return policy.mode == "any" or (flexible and policy.mode == "flexible") or ((not flexible) and policy.mode == "legacy")


def choose_policy(request: Dict[str, Any], task: Optional[str] = None, space: Optional[str] = None) -> ManualPolicy:
    resolved_task = infer_task_from_request(request, task)
    resolved_space = infer_space_from_request(request, space)
    flexible = is_flexible_request(request)
    for p in POLICIES:
        if p.task == resolved_task and resolved_space in p.spaces and policy_mode_matches(p, flexible):
            return p
    for p in POLICIES:
        if p.task == resolved_task and policy_mode_matches(p, flexible):
            return p
    for p in POLICIES:
        if p.task == resolved_task:
            return p
    raise ValueError(f"No manual policy for task={resolved_task!r}, space={resolved_space!r}, flexible={flexible}")


def run_symbolic_compiler(operator_catalog: Dict[str, Any], request: Dict[str, Any], generator_path: Optional[str | Path], max_depth: int, max_states: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gen = load_generator(generator_path)
    if not gen or not hasattr(gen, "enumerate_candidates"):
        return [], {"compiler_status": "unavailable", "reason": "Could not import enumerate_candidates."}
    req = copy.deepcopy(request)
    # Manual uses generator as a type/dataflow compiler.  It should not benefit
    # from the full mediator's context-specific hard CI and residual filters.
    req["ci_output_constraints"] = {}
    req["residual_disclosure_constraints"] = {}
    try:
        result = gen.enumerate_candidates(
            operator_catalog,
            req,
            max_depth=max_depth,
            max_states=max_states,
            apply_request_ci_constraints=False,
            apply_request_residual_constraints=False,
        )
    except TypeError:
        result = gen.enumerate_candidates(operator_catalog, req, max_depth=max_depth, max_states=max_states)
    except Exception as exc:
        return [], {"compiler_status": "error", "reason": str(exc)}
    return result.get("candidates", []) or [], {"compiler_status": "ok", "candidate_count": len(result.get("candidates", []) or []), "planner": result.get("planner", {})}


def candidate_boundary_role(candidate: Dict[str, Any]) -> str:
    meta = candidate.get("matched_output_metadata") or {}
    if meta.get("boundary_role"):
        return str(meta.get("boundary_role"))
    props = (candidate.get("final_output_cap", {}) or {}).get("properties", {}) or {}
    return str(props.get("representationRole") or props.get("representation_role") or "")


def candidate_matches_policy(candidate: Dict[str, Any], policy: ManualPolicy, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cap = candidate.get("final_output_cap", {}) or {}
    transforms = candidate.get("transforms", []) or []
    matched = first_matching_accepted_cap(cap, request, transforms=transforms)
    if not matched:
        return None
    role = candidate_boundary_role(candidate)
    if role and role in set(policy.avoid_boundary_roles):
        return None
    t = cap_type(cap)
    schema = cap_schema(cap).lower()
    type_ok = any(t == target or type_matches(t, target) or type_matches(target, t) for target in policy.target_cap_types)
    schema_ok = any(kw.lower() in schema or kw.lower() in str(candidate.get("matched_output_cap", "")).lower() for kw in policy.target_schema_keywords)
    if not (type_ok or schema_ok):
        return None
    # Do not let manual flexible ADL select final labels/sound-only labels when
    # the policy asks for richer multimodal/AV evidence unless that is explicitly
    # the only compatible target.
    if policy.task == "adl_recognition" and "final" in role:
        return None
    return matched


def score_candidate(candidate: Dict[str, Any], policy: ManualPolicy, request: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
    cap = candidate.get("final_output_cap", {}) or {}
    t = cap_type(cap)
    schema = cap_schema(cap).lower()
    matched = first_matching_accepted_cap(cap, request, transforms=candidate.get("transforms", []) or [])
    try:
        priority = int((matched or {}).get("priority", 999))
    except Exception:
        priority = 999
    type_rank = 999
    for i, target in enumerate(policy.target_cap_types):
        if t == target or type_matches(t, target) or type_matches(target, t):
            type_rank = i
            break
    schema_rank = 999
    search = schema + " " + str((matched or {}).get("schema", "")).lower() + " " + str((matched or {}).get("cap_id", "")).lower()
    for i, kw in enumerate(policy.target_schema_keywords):
        if kw.lower() in search:
            schema_rank = i
            break
    op_ids = candidate_operator_ids(candidate)
    preferred_missing = sum(1 for op in policy.preferred_operator_ids if op not in op_ids)
    extra_ops = sum(1 for op in op_ids if op not in {"op.source", "op.route_publish"} and op not in set(policy.preferred_operator_ids))
    return (type_rank, schema_rank, preferred_missing, extra_ops, priority, residual_score(candidate.get("residual_disclosure", {}) or {}), len(op_ids))


def accepted_cap_by_policy(request: Dict[str, Any], policy: ManualPolicy) -> Optional[Dict[str, Any]]:
    caps = (request.get("utility_contract", {}) or {}).get("accepted_output_caps", []) or []
    for target in policy.target_cap_types:
        for cap in caps:
            if cap_type(cap) == cap_type({"semantic_type": target}) or cap_type(cap) == target or type_matches(cap_type(cap), target) or type_matches(target, cap_type(cap)):
                return copy.deepcopy(cap)
    for kw in policy.target_schema_keywords:
        for cap in caps:
            if kw.lower() in cap_schema(cap).lower() or kw.lower() in str(cap.get("cap_id", "")).lower():
                return copy.deepcopy(cap)
    return copy.deepcopy(caps[0]) if caps else None


def fallback_candidate(request: Dict[str, Any], policy: ManualPolicy, task: str, space: str) -> Optional[Dict[str, Any]]:
    cap = accepted_cap_by_policy(request, policy)
    if cap is None:
        return None
    # Make the reported final cap conform to the policy target rather than the
    # first broad accepted cap when possible.
    final_cap = copy.deepcopy(cap)
    if not final_cap.get("media_type") and not final_cap.get("semantic_type") and policy.target_cap_types:
        t = policy.target_cap_types[0]
        key = "media_type" if "/" in t and not t.startswith("application/") else "semantic_type"
        final_cap[key] = t
    if not final_cap.get("schema") and policy.target_schema_keywords:
        final_cap["schema"] = policy.target_schema_keywords[0]

    # Minimal, readable symbolic operator chain for non-catalog fallback.
    op_chain = [{"operator": "op.source", "variant": f"manual_source_for_{task}", "output_cap": {}, "parameters": {"space": space}}]
    for oid in policy.preferred_operator_ids:
        op_chain.append({"operator": oid, "variant": f"manual_policy_{policy.name}", "output_cap": copy.deepcopy(final_cap), "parameters": {"fixed_manual_policy": policy.name}})
    return make_candidate_record(
        baseline_name="manual",
        request=request,
        operators=op_chain,
        final_output_cap=final_cap,
        residual=policy.fallback_residual,
        ci_terms={"transmissionPrinciple": list(policy.fallback_transforms)},
        transforms=policy.fallback_transforms,
        utility_capabilities=[(request.get("utility_contract", {}) or {}).get("requested_capability", task)],
        quality_status="manual_symbolic_fallback_not_catalog_validated",
        notes=[
            "Manual baseline used a fixed task/space policy table.",
            "Flexible-mode manual policies are utility-aware: they prefer richer reusable primitives or compatible media over final task decisions.",
            "This fallback was emitted because no matching generated candidate was found.",
        ],
        executable_under_catalog=False,
    )


def run_baseline(
    operator_catalog: Dict[str, Any],
    request: Dict[str, Any],
    candidate_generator_path: Optional[str | Path] = None,
    task: Optional[str] = None,
    space: Optional[str] = None,
    max_depth: int = 7,
    max_states: int = 25000,
    allow_symbolic_fallback: bool = True,
) -> Dict[str, Any]:
    resolved_task = infer_task_from_request(request, task)
    resolved_space = infer_space_from_request(request, space)
    policy = choose_policy(request, resolved_task, resolved_space)
    generated, diag = run_symbolic_compiler(operator_catalog, request, candidate_generator_path, max_depth, max_states)
    ranked: List[Dict[str, Any]] = []
    for c in generated:
        matched = candidate_matches_policy(c, policy, request)
        if not matched:
            continue
        c2 = copy.deepcopy(c)
        c2["baseline"] = "manual"
        c2["matched_output_cap"] = matched.get("cap_id")
        c2["matched_output_schema"] = matched.get("schema")
        c2["space_task_policy"] = dataclasses.asdict(policy)
        c2["executable_under_catalog"] = True
        c2["baseline_notes"] = [
            "Selected by a fixed manual policy table keyed only by task and coarse physical space.",
            f"Resolved key: task={resolved_task}, space={resolved_space}.",
            f"Matched policy: {policy.name}.",
            "Flexible manual mode avoids final-task-decision collapse and prefers utility-preserving reusable primitives or compatible media.",
            "This baseline does not apply the full mediator's CI hard-rule filtering, residual-risk weighting, or least-revealing search.",
        ]
        ranked.append(c2)
    ranked.sort(key=lambda c: score_candidate(c, policy, request))

    candidates: List[Dict[str, Any]] = []
    if ranked:
        candidates.append(ranked[0])
    elif allow_symbolic_fallback:
        fb = fallback_candidate(request, policy, resolved_task, resolved_space)
        if fb:
            fb["space_task_policy"] = dataclasses.asdict(policy)
            candidates.append(fb)

    diagnostics = {
        "resolved_task": resolved_task,
        "resolved_space": resolved_space,
        "request_is_flexible": is_flexible_request(request),
        "manual_policy": dataclasses.asdict(policy),
        **diag,
    }
    if not candidates:
        return wrap_baseline_output(
            baseline_name="manual",
            request=request,
            candidates=[],
            selected_pipeline_id=None,
            decision="no_valid_pipeline",
            reason=f"No manual policy candidate matched {policy.name}.",
            diagnostics=diagnostics,
        )
    return wrap_baseline_output(
        baseline_name="manual",
        request=request,
        candidates=candidates,
        selected_pipeline_id=candidates[0]["pipeline_id"],
        decision="select_pipeline",
        reason=f"Selected fixed manual task/space policy {policy.name}.",
        diagnostics=diagnostics,
    )


def run_manual_baseline(
    operator_catalog: Dict[str, Any],
    request: Dict[str, Any],
    candidate_generator_path: Optional[str | Path] = None,
    max_depth: int = 7,
    max_states: int = 25000,
    allow_symbolic_fallback: bool = True,
) -> Dict[str, Any]:
    return run_baseline(operator_catalog, request, candidate_generator_path, None, None, max_depth, max_states, allow_symbolic_fallback)


def iter_context_scenarios(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if "context_scenarios" in data:
        return data["context_scenarios"]
    if "generated_information_flows" in data:
        return data["generated_information_flows"]
    if isinstance(data, list):
        return data
    raise ValueError("Could not find context_scenarios or generated_information_flows.")


def summarize_space_task_table(context_scenarios: Dict[str, Any]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for s in iter_context_scenarios(context_scenarios):
        c = s.get("ci_parameters_scalar_context_only") or s.get("ci_parameters_scalar") or s
        dummy = {"request_identity": {"scenario_id": s.get("scenario_id")}, "utility_contract": {}, "ci_context": {"space": [c.get("space")], "purpose": [c.get("purpose")], "context": [c.get("context")]}}
        task = normalize_term(s.get("task") or c.get("task"))
        space = normalize_term(c.get("space"))
        for flexible in [False, True]:
            dd = copy.deepcopy(dummy)
            if flexible:
                dd["flexible_tag"] = True
                dd["utility_contract"]["interface_model"] = "multi_representation_utility_contract"
            try:
                p = choose_policy(dd, task, space)
            except Exception:
                continue
            entries.append({"scenario_id": s.get("scenario_id") or s.get("flow_id"), "task": task, "space": space, "mode": "flexible" if flexible else "legacy", "policy_name": p.name, "target_cap_types": list(p.target_cap_types), "rationale": p.rationale})
    return {"schema_version": "manual_policy_table_summary_v2", "entries": entries}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run the manual preprocessing baseline.")
    p.add_argument("--operators", required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--candidate-generator", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--space", default=None)
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--max-states", type=int, default=25000)
    p.add_argument("--no-symbolic-fallback", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--summarize-contexts", default=None, help="Optional context file for policy-table summary mode.")
    args = p.parse_args(argv)
    if args.summarize_contexts:
        result = summarize_space_task_table(load_json(args.summarize_contexts))
    else:
        result = run_baseline(
            operator_catalog=load_json(args.operators),
            request=load_json(args.request),
            candidate_generator_path=args.candidate_generator,
            task=args.task,
            space=args.space,
            max_depth=args.max_depth,
            max_states=args.max_states,
            allow_symbolic_fallback=not args.no_symbolic_fallback,
        )
    if args.out:
        write_json(result, args.out)
    print(json.dumps(result.get("decision", result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
