#!/usr/bin/env python3
"""Downstream-compatible task/space manual preprocessing baseline.

This baseline is a stronger manual baseline than the original fixed app-level
policy table.  It is keyed only by:

    (task, physical space)

but it additionally enforces that the selected preprocessing output is compatible
with the downstream application request.  In particular, it will not select a
privacy-minimized semantic label/event when the downstream program expects media,
pose sequences, audio waveforms, or a multimodal AV sample.

The candidate generator is used only as a compiler/type checker when possible.
For multimodal YouHome ADL, the current catalog models the synchronized AV
sample as a source/interface-preserving representation and applies component-wise
AV transforms rather than using a manual-only synchronization placeholder.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": 4}
RESIDUAL_ATTRIBUTES = [
    "identity", "face", "body_shape", "clothing", "gait", "speech_content",
    "speaker_identity", "activity", "location", "trajectory", "co_presence",
    "visible_text", "aggregate_presence",
]

SEMANTIC_FAMILIES = {
    "application/x-event": {
        "application/x-event", "application/x-safety-event", "application/x-security-event",
        "application/x-sound-event-label", "application/x-activity-label", "application/x-fused-event",
    },
    "application/x-observation": {
        "application/x-observation", "application/x-occupancy-count", "application/x-binary-occupancy",
        "application/x-decibel-level", "application/x-command-intent", "application/x-sound-event-label",
        "application/x-safety-event", "application/x-security-event", "application/x-activity-label",
    },
    "application/x-detections": {"application/x-detections"},
}

TASK_ALIASES = {
    "visitor_presence_detection": {
        "visitor", "presence", "intrusion", "security", "entry", "person_at_door",
        "visitor_presence_inference", "chokepoint",
    },
    "fall_detection": {"fall", "elder", "safety", "le2i", "pose"},
    "adl_recognition": {"adl", "activity", "routine", "youhome", "audio_visual", "av"},
    "domestic_sound_monitoring": {
        "domestic_sound_event_inference", "audio", "sound", "noise", "voice", "speech",
        "command", "chime", "chime_home", "chimehome",
    },
}

SENSITIVE_VISUAL_SPACES = {"bedroom", "bathroom", "patient_room", "living_room", "common_area", "workspace", "kitchen"}
SENSITIVE_AUDIO_SPACES = {"bedroom", "bathroom", "patient_room", "workspace"}


@dataclasses.dataclass(frozen=True)
class SpaceTaskPolicy:
    name: str
    task: str
    spaces: Tuple[str, ...]
    target_cap_types: Tuple[str, ...]
    target_schema_keywords: Tuple[str, ...]
    preferred_operator_ids: Tuple[str, ...]
    required_downstream_interface: str
    fallback_transforms: Tuple[str, ...]
    fallback_residual: Dict[str, str]
    rationale: str


# The target caps are downstream-interface caps, not privacy-only semantic summaries.
SPACE_TASK_POLICIES: Tuple[SpaceTaskPolicy, ...] = (
    SpaceTaskPolicy(
        name="visitor_format_preserving_redacted_frame_sensitive_space",
        task="visitor_presence_detection",
        spaces=tuple(sorted(SENSITIVE_VISUAL_SPACES)),
        target_cap_types=("image/x-raw", "video/x-raw"),
        target_schema_keywords=("raw_image_frame", "raw_video_stream", "image_frame", "video"),
        preferred_operator_ids=("op.region_mask_blur", "op.region_select_crop", "op.sample"),
        required_downstream_interface="image_or_video_frame_for_chokepoint_yolo",
        fallback_transforms=("face_blurred", "body_blurred", "field_of_view_minimized", "identity_removed", "data_minimized"),
        fallback_residual={"identity": "medium", "face": "low", "visible_text": "low", "activity": "medium", "location": "medium", "aggregate_presence": "high"},
        rationale="Visitor monitoring downstream code expects frames.  In indoor/sensitive spaces the manual policy keeps the image/video interface but applies redaction/cropping.",
    ),
    SpaceTaskPolicy(
        name="visitor_format_preserving_frame_entry_or_outdoor",
        task="visitor_presence_detection",
        spaces=("entrance", "outdoor"),
        target_cap_types=("image/x-raw", "video/x-raw"),
        target_schema_keywords=("raw_image_frame", "raw_video_stream", "image_frame", "video"),
        preferred_operator_ids=("op.region_select_crop", "op.region_mask_blur", "op.sample"),
        required_downstream_interface="image_or_video_frame_for_chokepoint_yolo",
        fallback_transforms=("field_of_view_minimized", "face_blurred", "identity_removed", "data_minimized"),
        fallback_residual={"identity": "medium", "face": "low", "visible_text": "low", "activity": "medium", "location": "medium", "aggregate_presence": "high"},
        rationale="Entry/outdoor visitor monitoring still returns image/video frames so the detector can run; the manual policy prefers crop/redaction when available.",
    ),
    SpaceTaskPolicy(
        name="fall_pose_sequence_for_le2i_all_spaces",
        task="fall_detection",
        spaces=("bathroom", "bedroom", "common_area", "entrance", "kitchen", "living_room", "outdoor", "patient_room", "workspace"),
        target_cap_types=("application/x-pose-keypoints",),
        target_schema_keywords=("pose_keypoints", "pose"),
        preferred_operator_ids=("op.pose_extractor", "op.sample", "op.window"),
        required_downstream_interface="pose_keypoint_sequence_for_le2i_fall_model",
        fallback_transforms=("raw_pixels_removed", "pose_keypoints", "data_minimized", "no_raw_data_retention"),
        fallback_residual={"identity": "low", "face": "none", "body_shape": "medium", "gait": "medium", "activity": "high", "location": "high", "co_presence": "high", "aggregate_presence": "high"},
        rationale="The fall downstream program consumes pose windows/sequences, so the manual baseline must output pose keypoints rather than a fall label or raw video.",
    ),
    SpaceTaskPolicy(
        name="adl_youhome_av_sample_sensitive_space",
        task="adl_recognition",
        spaces=("bedroom", "bathroom", "patient_room", "workspace"),
        target_cap_types=("application/x-youhome-av-sample",),
        target_schema_keywords=("youhome_av_manifest_or_sample", "youhome_av"),
        preferred_operator_ids=("op.av_visual_redact", "op.av_audio_speech_filter", "op.youhome_av_adapter", "op.sample", "op.window"),
        required_downstream_interface="youhome_audio_video_sample_or_manifest",
        fallback_transforms=("face_blurred", "speech_content_removed", "synchronized_av_window", "data_minimized"),
        fallback_residual={"identity": "medium", "face": "low", "speech_content": "low", "speaker_identity": "low", "activity": "high", "location": "medium", "trajectory": "medium", "co_presence": "medium", "aggregate_presence": "high"},
        rationale="YouHome ADL uses audio+video.  In sensitive spaces the manual policy still outputs an AV sample, but with format-preserving visual/audio redaction when available.",
    ),
    SpaceTaskPolicy(
        name="adl_youhome_av_sample_common_space",
        task="adl_recognition",
        spaces=("common_area", "entrance", "kitchen", "living_room", "outdoor"),
        target_cap_types=("application/x-youhome-av-sample",),
        target_schema_keywords=("youhome_av_manifest_or_sample", "youhome_av"),
        preferred_operator_ids=("op.av_visual_redact", "op.youhome_av_adapter", "op.sample", "op.window"),
        required_downstream_interface="youhome_audio_video_sample_or_manifest",
        fallback_transforms=("synchronized_av_window", "data_minimized"),
        fallback_residual={"identity": "high", "face": "high", "speech_content": "medium", "speaker_identity": "medium", "activity": "high", "location": "medium", "trajectory": "medium", "co_presence": "medium", "aggregate_presence": "high"},
        rationale="YouHome ADL requires both modalities; the manual policy packages audio+video rather than substituting an activity label or summary.",
    ),
    SpaceTaskPolicy(
        name="audio_filtered_waveform_sensitive_space",
        task="domestic_sound_monitoring",
        spaces=tuple(sorted(SENSITIVE_AUDIO_SPACES)),
        target_cap_types=("audio/x-filtered", "audio/x-raw"),
        target_schema_keywords=("speech_removed_audio_waveform", "raw_audio_waveform", "audio"),
        preferred_operator_ids=("op.speech_content_removal", "op.sample", "op.window"),
        required_downstream_interface="waveform_chunk_for_chime_home_audio_model",
        fallback_transforms=("speech_content_removed", "data_minimized"),
        fallback_residual={"speech_content": "low", "speaker_identity": "low", "activity": "medium", "location": "low", "co_presence": "medium"},
        rationale="The CHiME-Home downstream program expects waveform-like audio chunks.  In sensitive spaces the manual policy prefers speech-removed waveform, not sound labels or decibel levels.",
    ),
    SpaceTaskPolicy(
        name="audio_waveform_common_space",
        task="domestic_sound_monitoring",
        spaces=("common_area", "entrance", "kitchen", "living_room", "outdoor"),
        target_cap_types=("audio/x-filtered", "audio/x-raw"),
        target_schema_keywords=("speech_removed_audio_waveform", "raw_audio_waveform", "audio"),
        preferred_operator_ids=("op.speech_content_removal", "op.sample", "op.window"),
        required_downstream_interface="waveform_chunk_for_chime_home_audio_model",
        fallback_transforms=("speech_content_removed", "data_minimized"),
        fallback_residual={"speech_content": "low", "speaker_identity": "low", "activity": "medium", "location": "low", "co_presence": "medium"},
        rationale="Even in common spaces, the output remains waveform-like so CHiME-Home inference can run; semantic sound labels are not selected.",
    ),
)


# ----------------------------- basic helpers -----------------------------

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


def normalize_term(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def cap_type(cap: Dict[str, Any]) -> str:
    return str(cap.get("semantic_type") or cap.get("media_type") or "")


def cap_schema(cap: Dict[str, Any]) -> str:
    return str(cap.get("schema") or "")


def normalize_risk(value: Any) -> str:
    s = str(value or "unknown").lower()
    if s in RISK_ORDER:
        return s
    if "high" in s:
        return "high"
    if "medium" in s:
        return "medium"
    if "low" in s:
        return "low"
    if "none" in s or "removed" in s:
        return "none"
    return "unknown"


def init_residual(default: str = "none") -> Dict[str, str]:
    return {a: normalize_risk(default) for a in RESIDUAL_ATTRIBUTES}


def residual_score(residual: Dict[str, Any]) -> int:
    weights = {"identity": 3, "face": 2, "speech_content": 3, "speaker_identity": 3, "trajectory": 2, "visible_text": 2}
    return sum(weights.get(a, 1) * RISK_ORDER[normalize_risk(residual.get(a, "none"))] for a in RESIDUAL_ATTRIBUTES)


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
    # Format-preserving privacy transforms remain interface-compatible.
    if downstream == "video/x-raw" and upstream in {"video/x-raw", "video/x-redacted"}:
        return True
    if downstream == "image/x-raw" and upstream in {"image/x-raw", "image/x-redacted"}:
        return True
    if downstream == "audio/x-raw" and upstream in {"audio/x-raw", "audio/x-filtered"}:
        return True
    return False


def goal_cap_matches(out_cap: Dict[str, Any], accepted_cap: Dict[str, Any]) -> bool:
    goal_t = str(accepted_cap.get("semantic_type") or accepted_cap.get("media_type") or "")
    out_t = cap_type(out_cap)
    if goal_t and type_matches(out_t, goal_t):
        return True
    goal_schema = cap_schema(accepted_cap)
    if goal_schema and cap_schema(out_cap) == goal_schema:
        return True
    return False


def rejected_cap_matches(out_cap: Dict[str, Any], rejected_cap: Dict[str, Any]) -> bool:
    # For explicit rejections, schema or exact type match is enough.  Do not apply
    # redacted/raw compatibility because that would reject filtered/redacted media.
    r_t = str(rejected_cap.get("semantic_type") or rejected_cap.get("media_type") or "")
    if r_t and cap_type(out_cap) == r_t:
        return True
    r_schema = cap_schema(rejected_cap)
    return bool(r_schema and cap_schema(out_cap) == r_schema)


def _lookup_metadata_value(request: Dict[str, Any], key: str) -> Any:
    paths = [
        ("utility_contract", "checkpoint_metadata", key),
        ("utility_contract", "runtime_metadata", key),
        ("utility_contract", "model_metadata", key),
        ("checkpoint_metadata", key),
        ("runtime_metadata", key),
        ("model_metadata", key),
    ]
    for path in paths:
        cur: Any = request
        ok = True
        for part in path:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def accepted_cap_conditions_satisfied(cap: Dict[str, Any], request: Dict[str, Any]) -> bool:
    cond = cap.get("conditional_on_checkpoint_metadata") or cap.get("conditional_on_runtime_metadata") or cap.get("conditional_on_metadata")
    if not cond:
        return True
    if not isinstance(cond, dict):
        return False
    for k, expected in cond.items():
        actual = _lookup_metadata_value(request, str(k))
        expected_vals = set(map(str, as_list(expected)))
        actual_vals = set(map(str, as_list(actual)))
        if not actual_vals or not (actual_vals & expected_vals):
            return False
    return True


def _accepted_cap_specificity_score(out_cap: Dict[str, Any], cap: Dict[str, Any], transforms: Optional[Sequence[str]] = None) -> Tuple[int, int, int, int]:
    """Score how specifically an accepted cap describes the actual output."""
    goal_props = cap.get("properties", {}) or {}
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
    terms = set(map(str, as_list(cap.get("required_transformations"))))
    terms.update(map(str, as_list(cap.get("requiredTransformations"))))
    terms.update(map(str, as_list(cap.get("required_transmissionPrinciple"))))
    try:
        priority = int(cap.get("priority", 999))
    except Exception:
        priority = 999
    return (matched, -(declared - matched), len(terms), -priority)


def transformations_allowed_by_cap(cap: Dict[str, Any], transforms: Optional[Sequence[str]]) -> bool:
    if not transforms:
        return True
    forbidden = set(map(str, as_list(cap.get("forbidden_transformations"))))
    return not (forbidden & set(map(str, transforms)))


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
        return bool(
            component_types & {"image/x-redacted", "video/x-redacted"}
            or transform_set & {"face_blurred", "identity_removed", "visible_text_removed", "screen_content_removed", "body_blurred"}
        )
    if key in {"speech_content_removed", "audio_filtered"}:
        return bool(
            component_types & {"audio/x-filtered"}
            or cap_type(out_cap) == "audio/x-filtered"
            or transform_set & {"speech_content_removed", "speech_content_minimized"}
        )
    return None


def accepted_cap_properties_satisfied(out_cap: Dict[str, Any], cap: Dict[str, Any], transforms: Optional[Sequence[str]] = None) -> bool:
    """Require scalar accepted-cap properties to describe the actual output."""
    goal_props = cap.get("properties", {}) or {}
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


def first_matching_accepted_cap(out_cap: Dict[str, Any], request: Dict[str, Any], transforms: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    for rej in (request.get("utility_contract", {}) or {}).get("explicitly_rejected_output_caps", []) or []:
        if rejected_cap_matches(out_cap, rej):
            return None
    caps = [
        c for c in (request.get("utility_contract", {}) or {}).get("accepted_output_caps", []) or []
        if accepted_cap_conditions_satisfied(c, request)
        and transformations_allowed_by_cap(c, transforms)
        and accepted_cap_properties_satisfied(out_cap, c, transforms)
    ]
    out_t = cap_type(out_cap)
    out_schema = cap_schema(out_cap)
    exact = [cap for cap in caps if out_t and out_t == cap_type(cap) and (not out_schema or not cap_schema(cap) or out_schema == cap_schema(cap))]
    if exact:
        return max(exact, key=lambda cap: _accepted_cap_specificity_score(out_cap, cap, transforms))
    exact_schema = [cap for cap in caps if out_schema and out_schema == cap_schema(cap)]
    if exact_schema:
        return max(exact_schema, key=lambda cap: _accepted_cap_specificity_score(out_cap, cap, transforms))
    loose = [cap for cap in caps if goal_cap_matches(out_cap, cap)]
    if loose:
        return max(loose, key=lambda cap: _accepted_cap_specificity_score(out_cap, cap, transforms))
    return None


def accepted_cap_priority(request: Dict[str, Any], matched: Optional[Dict[str, Any]]) -> int:
    if not matched:
        return 999
    try:
        return int(matched.get("priority", 999))
    except Exception:
        return 999


def import_module_from_path(module_name: str, path: str | Path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_generator(explicit_path: Optional[str | Path] = None):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    here = Path(__file__).resolve().parent
    candidates.extend([
        here.parent / "generate_pipeline_candidates.py",
        here.parent.parent / "generate_pipeline_candidates.py",
        Path.cwd() / "generate_pipeline_candidates.py",
        Path.cwd() / "mediator" / "generate_pipeline_candidates.py",
        Path("/mnt/data/generate_pipeline_candidates_final_cleanup.py"),
        Path("/mnt/data/generate_pipeline_candidates_matched_cap_specificity.py"),
        Path("/mnt/data/generate_pipeline_candidates_av_component_preserve_fixed.py"),
        Path("/mnt/data/generate_pipeline_candidates.py"),
        Path("/mnt/data/generate_pipeline_candidates(13).py"),
        Path("/mnt/data/generate_pipeline_candidates(5).py"),
    ])
    for p in candidates:
        if p.exists():
            return import_module_from_path("smartpriv_candidate_generator_for_manual_downstream", p)
    return None


# ----------------------------- task / space ------------------------------

def infer_task_from_request(request: Dict[str, Any], explicit_task: Optional[str] = None) -> str:
    if explicit_task:
        return normalize_term(explicit_task)
    rid = request.get("request_identity", {}) or {}
    uc = request.get("utility_contract", {}) or {}
    ctx = request.get("ci_context", {}) or {}
    parts: List[str] = []
    for key in ["request_id", "scenario_id", "application_category", "application_name", "natural_language_request"]:
        parts.append(str(rid.get(key, "")))
    for key in ["requested_capability", "task_description"]:
        parts.append(str(uc.get(key, "")))
    parts.extend(map(str, as_list(ctx.get("purpose"))))
    text = " ".join(parts).lower()
    for canonical, aliases in TASK_ALIASES.items():
        if canonical in text or any(alias in text for alias in aliases):
            return canonical
    return normalize_term(uc.get("requested_capability") or rid.get("application_category") or "unknown_task")


def infer_space_from_request(request: Dict[str, Any], explicit_space: Optional[str] = None) -> str:
    if explicit_space:
        return normalize_term(explicit_space)
    spaces = [normalize_term(v) for v in as_list((request.get("ci_context", {}) or {}).get("space")) if v]
    generic = {"indoor", "interior_home_space", "common_area"}
    for s in spaces:
        if s not in generic:
            return s
    return spaces[0] if spaces else "unknown_space"


def choose_policy(request: Dict[str, Any], task: Optional[str] = None, space: Optional[str] = None) -> SpaceTaskPolicy:
    t = infer_task_from_request(request, task)
    s = infer_space_from_request(request, space)
    for p in SPACE_TASK_POLICIES:
        if p.task == t and s in p.spaces:
            return p
    for p in SPACE_TASK_POLICIES:
        if p.task == t:
            return p
    raise ValueError(f"No downstream-compatible manual policy for task={t!r}, space={s!r}")


# --------------------------- candidate selection --------------------------

def relaxed_request_for_manual_compiler(request: Dict[str, Any]) -> Dict[str, Any]:
    """Keep accepted output caps/source requirements; relax utility/CI filters.

    The manual baseline decides utility compatibility by accepted caps and its
    policy table.  The generator is used as a type/dataflow compiler only.
    """
    r = copy.deepcopy(request)
    r.setdefault("utility_contract", {})["requested_capability"] = None
    r["ci_output_constraints"] = {}
    r["residual_disclosure_constraints"] = {}
    return r


def run_symbolic_compiler(operator_catalog: Dict[str, Any], request: Dict[str, Any], generator_path: Optional[str | Path], max_depth: int, max_states: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gen = load_generator(generator_path)
    if not gen or not hasattr(gen, "enumerate_candidates"):
        return [], {"compiler_status": "unavailable", "reason": "Could not import enumerate_candidates."}
    try:
        result = gen.enumerate_candidates(operator_catalog, relaxed_request_for_manual_compiler(request), max_depth=max_depth, max_states=max_states)
    except Exception as exc:
        return [], {"compiler_status": "error", "reason": str(exc)}
    return result.get("candidates", []) or [], {
        "compiler_status": "ok",
        "candidate_count": len(result.get("candidates", []) or []),
        "planner": result.get("planner", {}),
    }


def candidate_matches_policy_and_request(candidate: Dict[str, Any], policy: SpaceTaskPolicy, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cap = candidate.get("final_output_cap", {}) or {}
    matched = first_matching_accepted_cap(cap, request, transforms=candidate.get("transforms", []))
    if not matched:
        return None
    t = cap_type(cap)
    if policy.task == "adl_recognition":
        # YouHome default model is AV. Do not allow the audio-only/video-only caps,
        # even if they are in the request for model-ablation compatibility.
        if t != "application/x-youhome-av-sample" and cap_schema(cap) != "youhome_av_manifest_or_sample":
            return None
    if policy.target_cap_types and not any(type_matches(t, target) or t == target for target in policy.target_cap_types):
        return None
    return matched


def target_score(candidate: Dict[str, Any], policy: SpaceTaskPolicy, request: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
    cap = candidate.get("final_output_cap", {}) or {}
    t = cap_type(cap)
    schema = cap_schema(cap).lower()
    matched = first_matching_accepted_cap(cap, request, transforms=candidate.get("transforms", []))

    type_rank = 999
    for i, target in enumerate(policy.target_cap_types):
        if t == target or type_matches(t, target):
            type_rank = i
            break

    op_ids = [str(op.get("operator")) for op in candidate.get("operators", []) or []]
    preferred_missing = sum(1 for op in policy.preferred_operator_ids if op not in op_ids)
    # Penalize extra transforms that are not part of the fixed manual policy.
    # This keeps the manual baseline keyed to task+space rather than letting it
    # become a least-revealing selector. For example, common-space ADL prefers
    # visual redaction but does not automatically add speech filtering unless
    # that operator is listed in the selected policy.
    ignorable_ops = {"op.source", "op.route_publish"}
    extra_transform_penalty = sum(
        1
        for op in op_ids
        if op not in ignorable_ops and op not in set(policy.preferred_operator_ids)
    )

    schema_rank = 999
    search = (schema + " " + str(candidate.get("matched_output_cap", "")).lower() + " " + str((matched or {}).get("schema", "")).lower())
    for i, kw in enumerate(policy.target_schema_keywords):
        if kw in search:
            schema_rank = i
            break

    return (
        type_rank,
        preferred_missing,
        extra_transform_penalty,
        accepted_cap_priority(request, matched),
        schema_rank,
        int(candidate.get("residual_score", residual_score(candidate.get("residual_disclosure", {})))),
        len(op_ids),
    )


def route_publish_operator(request: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "operator": "op.route_publish",
        "variant": "Route / Publish(output_to_application)",
        "output_cap": {"semantic_type": "external/application-output", "schema": "output_to_application"},
        "parameters": {"recipient": (request.get("ci_context", {}) or {}).get("recipient", []), "purpose": (request.get("ci_context", {}) or {}).get("purpose", [])},
    }


def make_candidate_record(name: str, request: Dict[str, Any], operators: List[Dict[str, Any]], final_cap: Dict[str, Any], residual: Dict[str, Any], transforms: Sequence[str], utility_caps: Sequence[str], notes: Sequence[str], executable_under_catalog: bool) -> Dict[str, Any]:
    matched = first_matching_accepted_cap(final_cap, request, transforms=policy.fallback_transforms)
    residual_norm = init_residual("none")
    for a in RESIDUAL_ATTRIBUTES:
        if a in residual:
            residual_norm[a] = normalize_risk(residual[a])
    ops = copy.deepcopy(operators)
    if not ops or ops[-1].get("operator") != "op.route_publish":
        ops.append(route_publish_operator(request))
    rec = {
        "pipeline_id": f"baseline_{name}_" + stable_hash({"operators": ops, "final_cap": final_cap, "residual": residual_norm}),
        "decision": "baseline_candidate",
        "baseline": name,
        "matched_output_cap": matched.get("cap_id") if matched else None,
        "matched_output_schema": matched.get("schema") if matched else None,
        "final_output_cap": final_cap,
        "operators": ops,
        "utility_capabilities": sorted(set(map(str, utility_caps))),
        "quality_status": "requires_runtime_or_benchmark_validation",
        "ci_terms": {"pipelineStage": ["output_to_application"], "transmissionPrinciple": sorted(set(map(str, transforms)))},
        "transforms": sorted(set(map(str, transforms))),
        "residual_disclosure": residual_norm,
        "residual_score": residual_score(residual_norm),
        "baseline_notes": list(notes),
        "executable_under_catalog": executable_under_catalog,
    }
    return rec


def accepted_cap_by_type_or_schema(request: Dict[str, Any], policy: SpaceTaskPolicy) -> Optional[Dict[str, Any]]:
    caps = (request.get("utility_contract", {}) or {}).get("accepted_output_caps", []) or []
    # Exact policy type/schema first.
    for target in policy.target_cap_types:
        for cap in caps:
            if cap_type(cap) == target:
                return copy.deepcopy(cap)
    for kw in policy.target_schema_keywords:
        for cap in caps:
            if kw in cap_schema(cap).lower() or kw in str(cap.get("cap_id", "")).lower():
                return copy.deepcopy(cap)
    # Fall back to any compatible cap.
    for cap in caps:
        if any(type_matches(cap_type(cap), target) or type_matches(target, cap_type(cap)) for target in policy.target_cap_types):
            return copy.deepcopy(cap)
    return copy.deepcopy(caps[0]) if caps else None


def fallback_candidate(request: Dict[str, Any], policy: SpaceTaskPolicy, resolved_task: str, resolved_space: str) -> Optional[Dict[str, Any]]:
    requested = (request.get("utility_contract", {}) or {}).get("requested_capability") or resolved_task
    target = accepted_cap_by_type_or_schema(request, policy)
    if target is None:
        return None

    # Use the accepted cap as the final downstream interface, with extra properties
    # documenting privacy-preserving substructure when useful.
    final_cap = copy.deepcopy(target)
    ops: List[Dict[str, Any]] = []
    residual = dict(policy.fallback_residual)
    executable = False

    if policy.task == "visitor_presence_detection":
        # Format-preserving visual pipeline.  The catalog can often compile this;
        # fallback is still a clear manual construction.
        source_cap = {"media_type": "image/x-raw", "schema": "raw_image_frame", "content_type": "image_content"}
        redacted_cap = {"media_type": final_cap.get("media_type", "image/x-raw"), "schema": final_cap.get("schema", "raw_image_frame"), "properties": {"redacted": True, "privacy_transform": "crop_or_blur"}}
        final_cap = redacted_cap
        ops = [
            {"operator": "op.source", "variant": "Source(image_frame)", "output_cap": source_cap, "parameters": {}},
            {"operator": "op.region_mask_blur", "variant": "manual_space_task_visual_redaction", "output_cap": redacted_cap, "parameters": {"target": "face_or_background", "space": resolved_space}},
        ]
        executable = True
    elif policy.task == "fall_detection":
        final_cap = {"semantic_type": "application/x-pose-keypoints", "schema": "pose_keypoints"}
        ops = [
            {"operator": "op.source", "variant": "Source(video_or_image)", "output_cap": {"media_type": "video/x-raw", "schema": "raw_video_stream", "content_type": "video_content"}, "parameters": {}},
            {"operator": "op.pose_extractor", "variant": "manual_space_task_pose_extraction", "output_cap": final_cap, "parameters": {"sequence_output": True}},
        ]
        executable = True
    elif policy.task == "adl_recognition":
        # Model YouHome as an already synchronized AV sample, then apply
        # component-wise, format-preserving transforms. This replaces the older
        # manual-only av_sync_packager placeholder and keeps the output compatible
        # with the fixed YouHome AV downstream interface.
        av_source_cap = {
            "semantic_type": "application/x-youhome-av-sample",
            "schema": "youhome_av_manifest_or_sample",
            "content_type": "av_content",
            "properties": {
                "input_interface": "youhome_av_manifest",
                "sensorPrimitive": ["image_frame", "audio_waveform"],
                "components": [
                    {"media_type": "image/x-raw", "schema": "youhome_video_frames", "role": "visual"},
                    {"media_type": "audio/x-raw", "schema": "youhome_audio_waveform", "role": "audio"},
                ],
                "av_synchronized": True,
                "youhome_av_compatible": True,
            },
        }
        visual_redact = resolved_space in SENSITIVE_VISUAL_SPACES
        audio_filter = resolved_space in SENSITIVE_AUDIO_SPACES
        final_cap = copy.deepcopy(av_source_cap)
        ops = [
            {"operator": "op.source", "variant": "Source(youhome_av_sample)", "output_cap": av_source_cap, "parameters": {}},
        ]
        if visual_redact:
            final_cap = {
                "semantic_type": "application/x-youhome-av-sample",
                "schema": "youhome_av_manifest_or_sample",
                "properties": {
                    "input_interface": "youhome_av_manifest",
                    "sensorPrimitive": ["image_frame", "audio_waveform"],
                    "components": [
                        {"media_type": "image/x-redacted", "schema": "redacted_image_frame", "role": "visual"},
                        {"media_type": "audio/x-raw", "schema": "youhome_audio_waveform", "role": "audio"},
                    ],
                    "visual_redacted": True,
                    "redacted": True,
                    "youhome_av_compatible": True,
                },
            }
            ops.append({"operator": "op.av_visual_redact", "variant": "manual_space_task_av_visual_redact", "output_cap": copy.deepcopy(final_cap), "parameters": {"space": resolved_space, "preserve_av_interface": True}})
        if audio_filter:
            final_cap = {
                "semantic_type": "application/x-youhome-av-sample",
                "schema": "youhome_av_manifest_or_sample",
                "properties": {
                    "input_interface": "youhome_av_manifest",
                    "sensorPrimitive": ["image_frame", "audio_waveform"],
                    "components": [
                        {"media_type": "image/x-redacted" if visual_redact else "image/x-raw", "schema": "redacted_image_frame" if visual_redact else "youhome_video_frames", "role": "visual"},
                        {"media_type": "audio/x-filtered", "schema": "speech_removed_audio_waveform", "role": "audio"},
                    ],
                    "visual_redacted": bool(visual_redact),
                    "speech_content_removed": True,
                    "audio_filtered": True,
                    "youhome_av_compatible": True,
                },
            }
            ops.append({"operator": "op.av_audio_speech_filter", "variant": "manual_space_task_av_audio_speech_filter", "output_cap": copy.deepcopy(final_cap), "parameters": {"space": resolved_space, "preserve_av_interface": True}})
        executable = True
    elif policy.task == "domestic_sound_monitoring":
        use_filtered = "audio/x-filtered" in policy.target_cap_types
        final_cap = {"media_type": "audio/x-filtered" if use_filtered else "audio/x-raw", "schema": "speech_removed_audio_waveform" if use_filtered else "raw_audio_waveform", "properties": {"speech_content_removed": use_filtered}}
        # Make sure final cap is accepted; if not, use the accepted cap verbatim.
        if not first_matching_accepted_cap(final_cap, request, transforms=policy.fallback_transforms):
            final_cap = copy.deepcopy(target)
        ops = [
            {"operator": "op.source", "variant": "Source(audio_waveform)", "output_cap": {"media_type": "audio/x-raw", "schema": "raw_audio_waveform", "content_type": "audio_content"}, "parameters": {}},
        ]
        if cap_type(final_cap) == "audio/x-filtered":
            ops.append({"operator": "op.speech_content_removal", "variant": "manual_space_task_speech_content_removal", "output_cap": final_cap, "parameters": {"preserve_waveform_interface": True}})
        executable = True

    matched = first_matching_accepted_cap(final_cap, request, transforms=policy.fallback_transforms)
    if not matched:
        return None
    return make_candidate_record(
        "manual_space_task_downstream",
        request,
        ops,
        final_cap,
        residual,
        policy.fallback_transforms,
        [requested, policy.required_downstream_interface, "downstream_input_compatible"],
        [
            "Manual task/space baseline with downstream-compatible output enforcement.",
            f"Resolved key: task={resolved_task}, space={resolved_space}.",
            f"Policy: {policy.name}.",
            "The selected/fallback output is an input interface accepted by the downstream application request, not merely a minimized semantic label.",
            "For ADL, fallback uses AV-sample component transforms rather than a manual-only AV synchronization placeholder.",
        ],
        executable,
    )


def wrap_output(request: Dict[str, Any], candidates: List[Dict[str, Any]], selected_id: Optional[str], decision: str, reason: str, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    selected = next((c for c in candidates if c.get("pipeline_id") == selected_id), None)
    return {
        "schema_version": "smartpriv_preprocessing_baseline_output_v1",
        "baseline": "manual_space_task_downstream",
        "request_id": (request.get("request_identity", {}) or {}).get("request_id"),
        "scenario_id": (request.get("request_identity", {}) or {}).get("scenario_id"),
        "decision": {"decision": decision, "selected_pipeline_id": selected_id, "selected_output_cap": selected.get("matched_output_cap") if selected else None, "reason": reason},
        "candidates": candidates,
        "diagnostics": diagnostics,
    }


def run_baseline(operator_catalog: Dict[str, Any], request: Dict[str, Any], candidate_generator_path: Optional[str | Path] = None, task: Optional[str] = None, space: Optional[str] = None, max_depth: int = 7, max_states: int = 25000, allow_symbolic_fallback: bool = True) -> Dict[str, Any]:
    resolved_task = infer_task_from_request(request, task)
    resolved_space = infer_space_from_request(request, space)
    policy = choose_policy(request, resolved_task, resolved_space)

    generated, diag = run_symbolic_compiler(operator_catalog, request, candidate_generator_path, max_depth, max_states)
    ranked: List[Dict[str, Any]] = []
    for c in generated:
        matched = candidate_matches_policy_and_request(c, policy, request)
        if not matched:
            continue
        c2 = copy.deepcopy(c)
        c2["matched_output_cap"] = matched.get("cap_id")
        c2["matched_output_schema"] = matched.get("schema")
        ranked.append(c2)
    ranked.sort(key=lambda c: target_score(c, policy, request))

    candidates: List[Dict[str, Any]] = []
    if ranked:
        chosen = copy.deepcopy(ranked[0])
        chosen["baseline"] = "manual_space_task_downstream"
        chosen["space_task_policy"] = dataclasses.asdict(policy)
        chosen["executable_under_catalog"] = True
        chosen["baseline_notes"] = [
            "Selected by a fixed manual policy table keyed only by task and physical space.",
            f"Resolved key: task={resolved_task}, space={resolved_space}.",
            f"Matched policy: {policy.name}.",
            "The candidate was filtered to match the downstream application request's accepted output caps and the task-specific downstream interface.",
            "This baseline still ignores the rest of CI: sender, subject, recipient, disclosure, social context, and hard norms.",
        ]
        candidates.append(chosen)
    elif allow_symbolic_fallback:
        fb = fallback_candidate(request, policy, resolved_task, resolved_space)
        if fb:
            fb["space_task_policy"] = dataclasses.asdict(policy)
            candidates.append(fb)

    diagnostics = {"resolved_task": resolved_task, "resolved_space": resolved_space, "space_task_policy": dataclasses.asdict(policy), **diag}
    if not candidates:
        return wrap_output(request, [], None, "no_valid_pipeline", f"No downstream-compatible pipeline could be selected for policy {policy.name}.", diagnostics)
    return wrap_output(request, candidates, candidates[0]["pipeline_id"], "select_pipeline", f"Selected downstream-compatible fixed task/space policy {policy.name}.", diagnostics)


# Backwards-compatible alias for older evaluation runners that import
# preprocessing_baselines.manual_baseline.run_manual_baseline.  The downstream-
# compatible implementation is run_baseline(...), and it can infer task/space
# from the context-overlaid app request when task/space are not passed.
def run_manual_baseline(
    operator_catalog: Dict[str, Any],
    request: Dict[str, Any],
    candidate_generator_path: Optional[str | Path] = None,
    max_depth: int = 7,
    max_states: int = 25000,
    allow_symbolic_fallback: bool = True,
) -> Dict[str, Any]:
    return run_baseline(
        operator_catalog=operator_catalog,
        request=request,
        candidate_generator_path=candidate_generator_path,
        task=None,
        space=None,
        max_depth=max_depth,
        max_states=max_states,
        allow_symbolic_fallback=allow_symbolic_fallback,
    )


# ------------------------------- table mode -------------------------------

def iter_context_scenarios(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if "context_scenarios" in data:
        return data["context_scenarios"]
    if "generated_information_flows" in data:
        return data["generated_information_flows"]
    if isinstance(data, list):
        return data
    raise ValueError("Could not find context_scenarios or generated_information_flows.")


def scenario_task_space(s: Dict[str, Any]) -> Tuple[str, str]:
    c = s.get("ci_parameters_scalar_context_only") or s.get("ci_parameters_scalar") or {}
    return normalize_term(s.get("task") or c.get("task")), normalize_term(c.get("space"))


def summarize_space_task_table(context_scenarios: Dict[str, Any]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    dummy = {"request_identity": {}, "utility_contract": {}, "ci_context": {}}
    scenarios = list(iter_context_scenarios(context_scenarios))
    for s in scenarios:
        task, space = scenario_task_space(s)
        if not task or not space:
            continue
        key = (task, space)
        if key not in groups:
            policy = choose_policy(dummy, task, space)
            groups[key] = {
                "task": task,
                "space": space,
                "policy_name": policy.name,
                "required_downstream_interface": policy.required_downstream_interface,
                "target_cap_types": list(policy.target_cap_types),
                "target_schema_keywords": list(policy.target_schema_keywords),
                "preferred_operator_ids": list(policy.preferred_operator_ids),
                "rationale": policy.rationale,
                "scenario_ids": [],
            }
        groups[key]["scenario_ids"].append(s.get("scenario_id") or s.get("flow_id"))
    entries = [groups[k] for k in sorted(groups)]
    unique_policy_names = sorted({e["policy_name"] for e in entries})
    unique_interfaces = sorted({e["required_downstream_interface"] for e in entries})
    return {
        "schema_version": "manual_space_task_downstream_policy_table_summary_v1",
        "deduplication_key": ["task", "space"],
        "scenario_count": len(scenarios),
        "unique_task_space_pairs": len(entries),
        "unique_policy_names": len(unique_policy_names),
        "unique_policy_name_list": unique_policy_names,
        "unique_downstream_interfaces": len(unique_interfaces),
        "unique_downstream_interface_list": unique_interfaces,
        "note": "Each observed task-space pair has a manual policy whose output cap is compatible with the corresponding downstream app interface.",
        "entries": entries,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run or summarize the downstream-compatible task/space manual baseline.")
    p.add_argument("--operators", help="Path to operator-contract JSON. Required unless only emitting a table.")
    p.add_argument("--request", help="Path to structured downstream-compatible application request JSON.")
    p.add_argument("--candidate-generator", default=None, help="Path to generate_pipeline_candidates.py.")
    p.add_argument("--task", default=None, help="Override task key from a context scenario, e.g. adl_recognition.")
    p.add_argument("--space", default=None, help="Override space key from a context scenario, e.g. bedroom.")
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--max-states", type=int, default=25000)
    p.add_argument("--no-symbolic-fallback", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--context-scenarios", default=None)
    p.add_argument("--emit-space-task-table", default=None)
    args = p.parse_args(argv)

    if args.emit_space_task_table:
        if not args.context_scenarios:
            raise SystemExit("--context-scenarios is required with --emit-space-task-table")
        summary = summarize_space_task_table(load_json(args.context_scenarios))
        write_json(summary, args.emit_space_task_table)
        print(json.dumps({
            "scenario_count": summary["scenario_count"],
            "unique_task_space_pairs": summary["unique_task_space_pairs"],
            "unique_policy_names": summary["unique_policy_names"],
            "unique_downstream_interfaces": summary["unique_downstream_interfaces"],
        }, indent=2))
        return 0

    if not args.operators or not args.request:
        raise SystemExit("--operators and --request are required to run the baseline")
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
    print(json.dumps(result.get("decision", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
