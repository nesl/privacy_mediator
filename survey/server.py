#!/usr/bin/env python3
"""
Standard-library web survey server for rating smart-space data-sharing scenarios
instantiated with concrete preprocessing outputs. Participant-facing text avoids
contextual-integrity jargon; internal exports still keep method/output metadata.

Default rating unit:
  one (context scenario, unique shared output / information type) survey case.

If multiple baselines or ablations produce the same shared output for the same
context, they are deduplicated into one survey item and the item records all
contributing method IDs/pipeline IDs.

Run from the survey directory or project root, for example:

  python server.py --host 0.0.0.0 --port 5000 --k 25 \
    --pipeline-output-dir ../runs/context_pipeline_generation

Outputs:
  outputs/responses.db
  /admin/export.csv
  /admin/export.json
  /admin/flows.json
  /admin/survey_items.json
"""
from __future__ import annotations

import argparse, csv, hashlib, json, mimetypes, random, sqlite3, time, uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DEFAULT_FLOW_FILE = ROOT / "data" / "ci_focused_user_study_context_only_dedup_32_no_output_readable.json"
DEFAULT_DB = ROOT / "outputs" / "responses.db"
# Runner usually writes this from project root. If server.py lives under survey/,
# ROOT.parent is the project root.
DEFAULT_PIPELINE_OUTPUT_DIR = ROOT.parent / "runs" / "context_pipeline_generation"
OFFLINE_METHODS = ["raw", "manual", "direct_llm", "full_mediator"]

# Participant-facing task labels. These do not change the normalized task values
# stored in the scenario JSON or exported in machine-readable fields.
TASK_LABEL_OVERRIDES = {
    "adl_recognition": "Daily activity recognition",
    "domestic_sound_monitoring": "Sound monitoring",
    "visitor_presence_detection": "Presence detection",
}

@dataclass
class Config:
    flow_file: Path
    db_path: Path
    k: int
    seed: int
    assignment_mode: str
    pipeline_output_dir: Optional[Path]
    include_pipeline_outputs: bool
    include_no_output_variants: bool


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def now_ms() -> int:
    return int(time.time() * 1000)


def stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def stable_hash_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_present(*vals: Any) -> Optional[str]:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return str(v)
    return None


def get_scenario_id(flow: Dict[str, Any], ordinal: int = 0) -> str:
    return str(flow.get("flow_id") or flow.get("scenario_id") or f"S{ordinal:03d}")


def scenario_ci_params(flow: Dict[str, Any]) -> Dict[str, Any]:
    params = (
        flow.get("ci_parameters_scalar_context_only")
        or flow.get("ci_parameters_scalar")
        or flow.get("ci_parameters")
        or {}
    )
    if params:
        return dict(params)
    mf = flow.get("machine_flow_for_ci_constraints_context_only") or flow.get("machine_flow_for_ci_constraints") or {}
    return {
        "context": first_scalar(mf.get("context")),
        "space": first_scalar(mf.get("space")),
        "sender": first_scalar(mf.get("sender")),
        "subject": first_scalar(mf.get("subject")),
        "recipient": first_scalar(mf.get("recipient")),
        "purpose": first_scalar(mf.get("purpose")),
        "transmission_principle": first_scalar(mf.get("transmissionPrinciple")),
    }


def first_scalar(x: Any) -> Optional[str]:
    vals = as_list(x)
    if not vals or vals[0] is None:
        return None
    return str(vals[0])


def cap_type(cap: Optional[Dict[str, Any]]) -> str:
    if not cap:
        return ""
    return str(cap.get("semantic_type") or cap.get("media_type") or "")


def cap_schema(cap: Optional[Dict[str, Any]]) -> str:
    if not cap:
        return ""
    return str(cap.get("schema") or "")


def flatten_strings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        out: List[str] = []
        for v in value.values():
            out.extend(flatten_strings(v))
        return out
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(flatten_strings(item))
        return out
    return [str(value)]


def resolve_existing_path(path_like: Any, pipeline_dir: Optional[Path] = None) -> Optional[Path]:
    if not path_like:
        return None
    p = Path(str(path_like))
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            Path.cwd() / p,
            ROOT / p,
            ROOT.parent / p,
        ])
        if pipeline_dir:
            candidates.extend([
                pipeline_dir / p,
                pipeline_dir.parent / p,
                pipeline_dir.parent.parent / p,
            ])
    for c in candidates:
        try:
            if c.exists():
                return c.resolve()
        except Exception:
            pass
    return None


def selected_pipeline_from_row(row: Dict[str, Any], pipeline_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    for key in ["selected_pipeline_json", "pipeline_spec_json"]:
        p = resolve_existing_path(row.get(key), pipeline_dir)
        if p and p.exists():
            try:
                data = load_json(p)
                # pipeline_spec wraps final output; selected_pipeline has full candidate.
                if key == "pipeline_spec_json":
                    return {
                        "pipeline_id": data.get("pipeline_id"),
                        "final_output_cap": data.get("final_output_cap") or {},
                        "matched_output_cap": data.get("matched_output_cap"),
                        "operators": (data.get("source_candidate_metadata") or {}).get("operators", []),
                        "ci_terms": (data.get("source_candidate_metadata") or {}).get("ci_terms", {}),
                        "transforms": (data.get("source_candidate_metadata") or {}).get("transforms", []),
                        "residual_disclosure": (data.get("source_candidate_metadata") or {}).get("residual_disclosure", {}),
                    }
                return data
            except Exception:
                continue
    return None


def load_pipeline_rows(pipeline_dir: Optional[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not pipeline_dir:
        return [], {"status": "disabled"}
    pipeline_dir = pipeline_dir.resolve() if pipeline_dir.exists() else pipeline_dir
    if not pipeline_dir.exists():
        return [], {"status": "missing", "pipeline_output_dir": str(pipeline_dir)}

    rows: List[Dict[str, Any]] = []
    index_path = pipeline_dir / "index.json"
    summary_path = pipeline_dir / "summary.json"
    source = None
    method_registry: Dict[str, Dict[str, Any]] = {}

    if index_path.exists():
        index = load_json(index_path)
        for m in index.get("methods") or []:
            if isinstance(m, dict) and m.get("method_id"):
                method_registry[str(m["method_id"])] = dict(m)

        for sid, meta in (index.get("contexts") or {}).items():
            method_rows = meta.get("methods") or {}

            # Backward compatibility: older generators only stored baselines;
            # some intermediate versions stored baselines and ablations separately.
            if not method_rows:
                method_rows = {}
                for baseline, row in (meta.get("baselines") or {}).items():
                    if isinstance(row, dict):
                        method_rows[str(row.get("method_id") or baseline)] = row
                for ablation, row in (meta.get("ablations") or {}).items():
                    if isinstance(row, dict):
                        method_rows[str(row.get("method_id") or f"ablation:{ablation}")] = row

            for method_id, row in method_rows.items():
                if not isinstance(row, dict):
                    continue
                r = dict(row)
                registry = method_registry.get(str(method_id), {})
                r.setdefault("scenario_id", sid)
                r.setdefault("method_id", str(method_id))
                r.setdefault("method_kind", registry.get("method_kind") or ("ablation" if str(method_id).startswith("ablation:") else "baseline"))
                r.setdefault("baseline", r.get("method_id"))
                r.setdefault("baseline_id", registry.get("baseline_id"))
                r.setdefault("ablation_mode", registry.get("ablation_mode"))
                r.setdefault("parent_method", registry.get("parent_method"))
                r.setdefault("method_label", registry.get("method_label") or r.get("method_id"))
                rows.append(r)
        source = str(index_path)
    elif summary_path.exists():
        data = load_json(summary_path)
        if isinstance(data, list):
            rows = [dict(r) for r in data if isinstance(r, dict)]
            for r in rows:
                r.setdefault("method_id", r.get("baseline"))
                r.setdefault("method_kind", "ablation" if str(r.get("method_id") or "").startswith("ablation:") else "baseline")
                r.setdefault("method_label", r.get("method_id"))
        source = str(summary_path)

    baseline_methods = sorted({str(r.get("method_id")) for r in rows if r.get("method_kind") == "baseline" and r.get("method_id")})
    ablation_methods = sorted({str(r.get("method_id")) for r in rows if r.get("method_kind") == "ablation" and r.get("method_id")})
    return rows, {
        "status": "ok" if rows else "empty",
        "pipeline_output_dir": str(pipeline_dir),
        "source": source,
        "method_pipeline_row_count": len(rows),
        "baseline_pipeline_row_count": len([r for r in rows if r.get("method_kind") == "baseline"]),
        "ablation_pipeline_row_count": len([r for r in rows if r.get("method_kind") == "ablation"]),
        "method_ids": sorted({str(r.get("method_id")) for r in rows if r.get("method_id")}),
        "baseline_method_ids": baseline_methods,
        "ablation_method_ids": ablation_methods,
    }


def information_types_from_candidate(candidate: Optional[Dict[str, Any]], row: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, Set[str]] = {
        "sensorPrimitive": set(),
        "interpretedObservation": set(),
        "inferredInformationType": set(),
    }
    if candidate:
        ci_terms = candidate.get("ci_terms") or {}
        for full_key, short in [
            ("informationType.sensorPrimitive", "sensorPrimitive"),
            ("informationType.interpretedObservation", "interpretedObservation"),
            ("informationType.inferredInformationType", "inferredInformationType"),
        ]:
            out[short].update(flatten_strings(ci_terms.get(full_key)))
        for source in [candidate.get("final_output_cap") or {}]:
            props = source.get("properties") if isinstance(source.get("properties"), dict) else {}
            req_info = source.get("required_informationType") if isinstance(source.get("required_informationType"), dict) else {}
            for src in [props, req_info]:
                for short in out:
                    out[short].update(flatten_strings(src.get(short)))

    # Fallback from summary row type/schema.
    t = str(row.get("final_output_type") or "")
    schema = str(row.get("final_output_schema") or "")
    if not any(out.values()):
        if t.startswith("image/") or "image" in schema or "frame" in schema:
            out["sensorPrimitive"].add("image_frame")
        if t.startswith("video/") or "video" in schema:
            out["sensorPrimitive"].add("video_stream")
        if t.startswith("audio/") or "audio" in schema or "waveform" in schema:
            out["sensorPrimitive"].add("audio_waveform")
        if "pose" in t or "pose" in schema:
            out["interpretedObservation"].add("pose_keypoints")
        if "occupancy" in t or "occupancy" in schema:
            out["interpretedObservation"].add("occupancy_count")
        if "activity" in t or "activity" in schema:
            out["interpretedObservation"].add("activity_label")
        if "sound" in t or "sound" in schema:
            out["interpretedObservation"].add("sound_event_label")
        if "event" in t or "event" in schema:
            out["inferredInformationType"].add("event")
    return {k: sorted(v) for k, v in out.items()}



def av_component_flags(final_cap: Dict[str, Any]) -> Dict[str, bool]:
    """Detect whether an audio-video sample includes redaction or speech filtering."""
    flags = {
        "has_visual": False,
        "has_audio": False,
        "redacted_visual": False,
        "raw_visual": False,
        "speech_removed_audio": False,
        "raw_audio": False,
    }

    def visit(cap: Any) -> None:
        if not isinstance(cap, dict):
            return
        media = str(cap.get("media_type") or cap.get("semantic_type") or "")
        schema = str(cap.get("schema") or "")
        props = cap.get("properties") if isinstance(cap.get("properties"), dict) else {}
        is_visual = media.startswith(("image/", "video/")) or any(x in schema for x in ["image", "video", "frame"])
        is_audio = media.startswith("audio/") or "audio" in schema or "waveform" in schema
        if is_visual:
            flags["has_visual"] = True
            if props.get("redacted"):
                flags["redacted_visual"] = True
            else:
                flags["raw_visual"] = True
        if is_audio:
            flags["has_audio"] = True
            if props.get("speech_content_removed") or media == "audio/x-filtered":
                flags["speech_removed_audio"] = True
            else:
                flags["raw_audio"] = True
        for component in props.get("components") or []:
            visit(component)

    visit(final_cap)
    return flags


def is_audio_video_sample(final_cap: Dict[str, Any], row: Dict[str, Any]) -> bool:
    t = cap_type(final_cap) or str(row.get("final_output_type") or "")
    schema = cap_schema(final_cap) or str(row.get("final_output_schema") or "")
    flags = av_component_flags(final_cap)
    return (
        t == "application/x-youhome-av-sample"
        or "youhome_av" in schema
        or "av_sample" in schema
        or (flags["has_visual"] and flags["has_audio"])
    )


def audio_video_output_label(final_cap: Dict[str, Any]) -> str:
    flags = av_component_flags(final_cap)
    if flags["redacted_visual"] and flags["speech_removed_audio"]:
        return "Short audio-video clip with identifying details blurred and spoken words removed"
    if flags["redacted_visual"] and flags["raw_audio"]:
        return "Short audio-video clip with identifying details blurred; audio may include speech"
    if flags["redacted_visual"]:
        return "Short audio-video clip with identifying details blurred"
    if flags["speech_removed_audio"] and flags["raw_visual"]:
        return "Short audio-video clip with spoken words removed; video is not blurred"
    if flags["speech_removed_audio"]:
        return "Short audio-video clip with spoken words removed"
    return "Short audio-video clip"


def audio_video_output_description(final_cap: Dict[str, Any]) -> str:
    flags = av_component_flags(final_cap)
    parts = ["The app or recipient would receive a short synchronized clip containing both camera video and audio from the space."]
    if flags["redacted_visual"]:
        parts.append("The video has faces or other identifying details blurred or hidden.")
    elif flags["raw_visual"]:
        parts.append("The video may show the unblurred camera view.")
    if flags["speech_removed_audio"]:
        parts.append("Spoken words or conversation content are removed from the audio; non-speech sounds may remain.")
    elif flags["raw_audio"]:
        parts.append("The audio may include speech or conversation.")
    return " ".join(parts)

def human_output_label(final_cap: Dict[str, Any], info_types: Dict[str, List[str]], row: Dict[str, Any]) -> str:
    """Plain-language description of the shared output for participants."""
    t = cap_type(final_cap) or str(row.get("final_output_type") or "")
    schema = cap_schema(final_cap) or str(row.get("final_output_schema") or "")
    props = final_cap.get("properties") if isinstance(final_cap.get("properties"), dict) else {}
    redacted = bool(props.get("redacted"))
    speech_removed = bool(props.get("speech_content_removed")) or t == "audio/x-filtered"
    fov_minimized = bool(props.get("field_of_view_minimized"))

    if t == "application/x-pose-keypoints" or "pose" in schema:
        return "Body pose points (stick-figure joint locations, not photo/video)"
    if is_audio_video_sample(final_cap, row):
        return audio_video_output_label(final_cap)
    if t.startswith("image/"):
        if redacted:
            return "Camera image with faces or identifying details blurred"
        if fov_minimized:
            return "Cropped camera image showing only the relevant area"
        return "Camera image"
    if t.startswith("video/"):
        if redacted:
            return "Camera video with faces or identifying details blurred"
        if fov_minimized:
            return "Cropped camera video showing only the relevant area"
        return "Camera video footage"
    if speech_removed:
        return "Audio after spoken words are removed"
    if t.startswith("audio/"):
        return "Audio recording"
    if "occupancy" in t or "occupancy" in schema or "room_occupied" in schema:
        return "Presence/occupancy estimate"
    if "decibel" in t or "decibel" in schema:
        return "Sound level only, without an audio recording"
    if "sound" in t or "sound" in schema or "noise_event" in schema:
        return "Sound category, such as alarm, glass breaking, or other noise"
    if "activity" in t or "activity" in schema:
        return "Activity category, such as walking, cooking, or lying down"
    if "event" in t or "event" in schema:
        return "Event alert or label"
    return first_present(row.get("matched_output_cap"), schema, t, "Data from the smart-space system") or "Data from the smart-space system"


def output_description(final_cap: Dict[str, Any], row: Dict[str, Any]) -> str:
    """Short lay explanation shown below the output label."""
    t = cap_type(final_cap) or str(row.get("final_output_type") or "")
    schema = cap_schema(final_cap) or str(row.get("final_output_schema") or "")
    props = final_cap.get("properties") if isinstance(final_cap.get("properties"), dict) else {}
    if t == "application/x-pose-keypoints" or "pose" in schema:
        return "The app or recipient would receive body joint locations, like a stick-figure skeleton. It would not receive the original image or video."
    if is_audio_video_sample(final_cap, row):
        return audio_video_output_description(final_cap)
    if t.startswith(("image/", "video/")) and props.get("redacted"):
        if t.startswith("image/"):
            return "The app or recipient would receive one camera image after faces or other identifying details, such as screens or background details, are blurred or hidden."
        if t.startswith("video/"):
            return "The app or recipient would receive camera video after faces or other identifying details, such as screens or background details, are blurred or hidden."
        return "The app or recipient would receive camera media after faces or other identifying details, such as screens or background details, are blurred or hidden."
    if t.startswith(("image/", "video/")) and props.get("field_of_view_minimized"):
        if t.startswith("image/"):
            return "The app or recipient would receive one cropped camera image showing only the relevant area, rather than the full scene."
        if t.startswith("video/"):
            return "The app or recipient would receive cropped camera video showing only the relevant area, rather than the full scene."
        return "The app or recipient would receive only a cropped part of the camera view, rather than the full scene."
    if t.startswith("image/"):
        return "The app or recipient would receive one camera image from the space, not video footage."
    if t.startswith("video/"):
        return "The app or recipient would receive camera video footage, including motion over time."
    if t == "audio/x-filtered" or props.get("speech_content_removed"):
        return "The app or recipient would receive audio after spoken words or conversation content are removed; non-speech sounds such as footsteps, crashes, alarms, or background noises may remain."
    if t.startswith("audio/"):
        return "The app or recipient would receive an audio recording from the space."
    if "decibel" in t or "decibel" in schema:
        return "The app or recipient would receive a sound-level measurement, not an audio recording."
    if "sound" in t or "sound" in schema:
        return "The app or recipient would receive a label describing the type of sound, not the original audio."
    if "occupancy" in t or "occupancy" in schema or "room_occupied" in schema:
        return "The app or recipient would receive a presence or count estimate, not the original sensor recording."
    if "activity" in t or "activity" in schema:
        return "The app or recipient would receive an activity category, not the original sensor recording."
    if "event" in t or "event" in schema:
        return "The app or recipient would receive an event label or alert, not the original sensor recording."
    return "This is the data that would be sent out of the smart-space system for the stated purpose."


def privacy_class_from_output(final_cap: Dict[str, Any], row: Dict[str, Any]) -> str:
    t = cap_type(final_cap) or str(row.get("final_output_type") or "")
    schema = cap_schema(final_cap) or str(row.get("final_output_schema") or "")
    props = final_cap.get("properties") if isinstance(final_cap.get("properties"), dict) else {}
    if t.startswith(("image/", "video/")) and not props.get("redacted"):
        return "raw_media"
    if t.startswith(("image/", "video/")) and props.get("redacted"):
        return "redacted_media"
    if t == "audio/x-filtered" or props.get("speech_content_removed"):
        return "filtered_audio"
    if t.startswith("audio/"):
        return "raw_audio"
    if "pose" in t or "pose" in schema:
        return "derived_pose"
    if is_audio_video_sample(final_cap, row):
        flags = av_component_flags(final_cap)
        if flags["redacted_visual"] and flags["speech_removed_audio"]:
            return "redacted_filtered_audio_video"
        if flags["redacted_visual"]:
            return "redacted_audio_video"
        if flags["speech_removed_audio"]:
            return "speech_filtered_audio_video"
        return "audio_video_clip"
    if t.startswith("application/"):
        return "derived_or_semantic_output"
    return "output_data"


def compact_final_cap(cap: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize the data shared enough to deduplicate same-output baselines."""
    if not cap:
        return {}
    props = cap.get("properties") if isinstance(cap.get("properties"), dict) else {}
    keep_props = {}
    for k in [
        "redacted", "speech_content_removed", "field_of_view_minimized", "skeleton",
        "backend", "shape", "input_interface", "components", "required_informationType",
    ]:
        if k in props:
            keep_props[k] = props[k]
    out = {
        "media_type": cap.get("media_type"),
        "semantic_type": cap.get("semantic_type"),
        "schema": cap.get("schema"),
        "properties": keep_props,
    }
    return {k: v for k, v in out.items() if v not in (None, {}, [])}


def row_represents_shared_output(row: Dict[str, Any]) -> bool:
    decision = str(row.get("decision") or "").strip().lower()
    return decision in {"select_pipeline", "selected", "accept", "accepted"}


def build_output_variant_from_row(row: Dict[str, Any], pipeline_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    # no_compromise / review / no_candidates rows may contain diagnostics or closest
    # rejected pipeline paths; those must not be treated as shared outputs.
    if not row_represents_shared_output(row):
        return None
    candidate = selected_pipeline_from_row(row, pipeline_dir)
    final_cap = (candidate or {}).get("final_output_cap") or {}
    if not final_cap:
        # Fall back to summary row.
        t = str(row.get("final_output_type") or "")
        schema = str(row.get("final_output_schema") or "")
        if t:
            if t.startswith(("image/", "video/", "audio/")):
                final_cap = {"media_type": t, "schema": schema}
            else:
                final_cap = {"semantic_type": t, "schema": schema}
    if not final_cap:
        return None

    info_types = information_types_from_candidate(candidate, row)
    label = human_output_label(final_cap, info_types, row)
    signature_obj = {
        "final_output_cap": compact_final_cap(final_cap),
        "information_types": info_types,
        "label": label,
    }
    return {
        "output_signature": stable_hash_obj(signature_obj),
        "output_variant_label": label,
        "output_variant_description": output_description(final_cap, row),
        "variant_privacy_class": privacy_class_from_output(final_cap, row),
        "final_output_cap": final_cap,
        "information_types": info_types,
        "matched_output_cap": row.get("matched_output_cap") or (candidate or {}).get("matched_output_cap"),
        "matched_output_schema": row.get("matched_output_schema") or row.get("final_output_schema") or cap_schema(final_cap),
        "candidate": candidate,
        "signature_object": signature_obj,
    }


def no_output_variant_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    decision = row.get("decision") or "no_output"
    signature_obj = {"decision": decision, "kind": "no_output"}
    return {
        "output_signature": stable_hash_obj(signature_obj),
        "output_variant_label": "No data would be shared",
        "output_variant_description": f"The system would not send data for this request ({decision}).",
        "variant_privacy_class": "no_output",
        "final_output_cap": {},
        "information_types": {"sensorPrimitive": [], "interpretedObservation": [], "inferredInformationType": []},
        "matched_output_cap": None,
        "matched_output_schema": None,
        "candidate": None,
        "signature_object": signature_obj,
    }


def build_survey_items(
    flows: List[Dict[str, Any]],
    pipeline_rows: List[Dict[str, Any]],
    pipeline_dir: Optional[Path],
    include_pipeline_outputs: bool,
    include_no_output_variants: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    flows_by_id: Dict[str, Dict[str, Any]] = {}
    for i, f in enumerate(flows, start=1):
        sid = get_scenario_id(f, i)
        flows_by_id[sid] = f

    if not include_pipeline_outputs or not pipeline_rows:
        items = []
        for i, f in enumerate(flows, start=1):
            sid = get_scenario_id(f, i)
            items.append({
                "item_id": f"{sid}__context_only",
                "flow_index": i - 1,
                "flow_id": sid,
                "flow": f,
                "output_variant": None,
                "method_ids": [],
                "method_details": [],
                "baseline_ids": [],
                "baseline_details": [],
                "ablation_modes": [],
            })
        return items, {
            "output_augmented": False,
            "context_scenario_count": len(flows),
            "survey_item_pool_count": len(items),
            "baseline_pipeline_row_count": 0,
            "baseline_pipeline_rows_with_output": 0,
            "baseline_pipeline_rows_without_output": 0,
            "deduplicated_output_variant_count": 0,
        }

    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    rows_with_output = 0
    rows_without_output = 0
    rows_for_known_context = 0

    for row in pipeline_rows:
        sid = str(row.get("scenario_id") or row.get("flow_id") or "")
        if sid not in flows_by_id:
            continue
        rows_for_known_context += 1
        variant = build_output_variant_from_row(row, pipeline_dir)
        if variant is None:
            rows_without_output += 1
            if not include_no_output_variants:
                continue
            variant = no_output_variant_from_row(row)
        else:
            rows_with_output += 1

        key = (sid, variant["output_signature"])
        if key not in grouped:
            flow = flows_by_id[sid]
            item_id = f"{sid}__out_{variant['output_signature']}"
            grouped[key] = {
                "item_id": item_id,
                "flow_index": list(flows_by_id.keys()).index(sid),
                "flow_id": sid,
                "flow": flow,
                "output_variant": {
                    "output_variant_id": f"out_{variant['output_signature']}",
                    "output_variant_label": variant["output_variant_label"],
                    "output_variant_description": variant.get("output_variant_description"),
                    "variant_privacy_class": variant["variant_privacy_class"],
                    "final_output_cap": variant["final_output_cap"],
                    "information_types": variant["information_types"],
                    "matched_output_cap": variant.get("matched_output_cap"),
                    "matched_output_schema": variant.get("matched_output_schema"),
                    "signature_object": variant.get("signature_object"),
                },
                "method_ids": [],
                "method_details": [],
                "baseline_ids": [],
                "baseline_details": [],
                "ablation_modes": [],
            }
        method_id = str(row.get("method_id") or row.get("baseline") or "unknown_method")
        method_kind = str(row.get("method_kind") or ("ablation" if method_id.startswith("ablation:") else "baseline"))
        baseline = str(row.get("baseline") or method_id)
        ablation_mode = row.get("ablation_mode")
        method_detail = {
            "method_id": method_id,
            "method_kind": method_kind,
            "method_label": row.get("method_label") or method_id,
            "baseline": baseline,
            "baseline_id": row.get("baseline_id"),
            "ablation_mode": ablation_mode,
            "parent_method": row.get("parent_method"),
            "decision": row.get("decision"),
            "selected_pipeline_id": row.get("selected_pipeline_id"),
            "matched_output_cap": row.get("matched_output_cap"),
            "matched_output_schema": row.get("matched_output_schema"),
            "final_output_type": row.get("final_output_type"),
            "final_output_schema": row.get("final_output_schema"),
            "operators": row.get("operators"),
            "baseline_output_dir": row.get("baseline_output_dir"),
            "method_output_dir": row.get("method_output_dir") or row.get("baseline_output_dir"),
            "result_json": row.get("result_json"),
            "selected_pipeline_json": row.get("selected_pipeline_json"),
            "pipeline_spec_json": row.get("pipeline_spec_json"),
            "error": row.get("error"),
        }
        if method_id not in grouped[key]["method_ids"]:
            grouped[key]["method_ids"].append(method_id)
        grouped[key]["method_details"].append(method_detail)

        # Backward-compatible aliases: baseline_ids now contain method IDs too,
        # so older export code still links one survey response to every producing
        # baseline/ablation. baseline_details mirrors method_details.
        if method_id not in grouped[key]["baseline_ids"]:
            grouped[key]["baseline_ids"].append(method_id)
        grouped[key]["baseline_details"].append(method_detail)
        if method_kind == "ablation" and ablation_mode and str(ablation_mode) not in grouped[key]["ablation_modes"]:
            grouped[key]["ablation_modes"].append(str(ablation_mode))

    items = list(grouped.values())
    # Stable, readable order.
    items.sort(key=lambda x: (str(x.get("flow_id")), str((x.get("output_variant") or {}).get("output_variant_label"))))

    return items, {
        "output_augmented": True,
        "context_scenario_count": len(flows),
        "survey_item_pool_count": len(items),
        "method_pipeline_row_count": len(pipeline_rows),
        "method_pipeline_rows_for_known_contexts": rows_for_known_context,
        "method_pipeline_rows_with_output": rows_with_output,
        "method_pipeline_rows_without_output": rows_without_output,
        "baseline_pipeline_row_count": len([r for r in pipeline_rows if r.get("method_kind") == "baseline"]),
        "ablation_pipeline_row_count": len([r for r in pipeline_rows if r.get("method_kind") == "ablation"]),
        "baseline_pipeline_rows_with_output": rows_with_output,
        "baseline_pipeline_rows_without_output": rows_without_output,
        "deduplicated_output_variant_count": len(items),
        "mean_output_variants_per_context": (len(items) / len(flows)) if flows else 0,
        "contexts_with_at_least_one_output": len({x[0] for x in grouped.keys()}),
    }


class SurveyState:
    def __init__(self, config: Config):
        self.config = config
        self.flow_data = load_json(config.flow_file)
        self.flows = self.flow_data.get("generated_information_flows") or self.flow_data.get("context_scenarios", [])
        if not self.flows:
            raise ValueError(f"No generated_information_flows or context_scenarios found in {config.flow_file}")

        self.pipeline_rows: List[Dict[str, Any]] = []
        self.pipeline_load_info: Dict[str, Any] = {"status": "disabled"}
        if config.include_pipeline_outputs and config.pipeline_output_dir:
            self.pipeline_rows, self.pipeline_load_info = load_pipeline_rows(config.pipeline_output_dir)

        self.items, self.item_pool_summary = build_survey_items(
            self.flows,
            self.pipeline_rows,
            config.pipeline_output_dir,
            include_pipeline_outputs=config.include_pipeline_outputs,
            include_no_output_variants=config.include_no_output_variants,
        )
        if not self.items:
            raise ValueError(
                "Survey item pool is empty. Check --pipeline-output-dir, or run with --no-pipeline-outputs "
                "to fall back to context-only items."
            )
        init_db(config.db_path)


def ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl_type: str) -> None:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                assignment_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                session_id TEXT NOT NULL,
                participant_id TEXT NOT NULL,
                item_index INTEGER NOT NULL,
                flow_id TEXT NOT NULL,
                method_id TEXT NOT NULL,
                task TEXT,
                context_family TEXT,
                output_variant_id TEXT,
                rating INTEGER NOT NULL,
                confidence INTEGER,
                free_text TEXT,
                elapsed_ms INTEGER,
                created_at_ms INTEGER NOT NULL,
                item_json TEXT NOT NULL,
                raw_payload_json TEXT NOT NULL,
                PRIMARY KEY (session_id, item_index)
            )
        """)
        # Backward-compatible schema extension for output-augmented survey items.
        ensure_column(conn, "responses", "item_id", "TEXT")
        ensure_column(conn, "responses", "baseline_ids_json", "TEXT")
        ensure_column(conn, "responses", "baseline_count", "INTEGER")
        ensure_column(conn, "responses", "baseline_details_json", "TEXT")
        ensure_column(conn, "responses", "method_ids_json", "TEXT")
        ensure_column(conn, "responses", "method_count", "INTEGER")
        ensure_column(conn, "responses", "method_details_json", "TEXT")
        ensure_column(conn, "responses", "ablation_modes_json", "TEXT")
        ensure_column(conn, "responses", "baseline_method_ids_json", "TEXT")
        ensure_column(conn, "responses", "output_variant_label", "TEXT")
        ensure_column(conn, "responses", "output_variant_description", "TEXT")
        ensure_column(conn, "responses", "variant_privacy_class", "TEXT")
        ensure_column(conn, "responses", "final_output_type", "TEXT")
        ensure_column(conn, "responses", "final_output_schema", "TEXT")
        ensure_column(conn, "responses", "information_types_json", "TEXT")
        conn.commit()


def create_session(db_path: Path, session_id: str, participant_id: str, assignment: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?)", (session_id, participant_id, now_ms(), json.dumps(assignment), json.dumps(metadata)))
        conn.commit()


def get_session(db_path: Path, session_id: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if not row:
            return None
        return {"session_id": row["session_id"], "participant_id": row["participant_id"], "created_at_ms": row["created_at_ms"], "assignment": json.loads(row["assignment_json"]), "metadata": json.loads(row["metadata_json"])}


def get_response(db_path: Path, session_id: str, index: int) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT rating, confidence, free_text, elapsed_ms FROM responses WHERE session_id=? AND item_index=?", (session_id, index)).fetchone()
        return dict(row) if row else None


def get_responses_for_session(db_path: Path, session_id: str) -> List[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM responses WHERE session_id=? ORDER BY item_index", (session_id,)).fetchall()]


def save_response(db_path: Path, session_id: str, participant_id: str, index: int, item: Dict[str, Any], rating: int, confidence: Any, free_text: str, elapsed_ms: Any, raw_payload: Dict[str, Any]) -> None:
    flow = item.get("flow", {})
    output = flow.get("output_data_slot", {}) or {}
    final_cap = output.get("final_output_cap") or {}
    try:
        confidence_int = int(confidence) if confidence not in (None, "") else None
    except Exception:
        confidence_int = None
    try:
        elapsed_int = int(elapsed_ms) if elapsed_ms not in (None, "") else None
    except Exception:
        elapsed_int = None
    method_ids = flow.get("method_ids") or flow.get("baseline_ids") or []
    method_details = flow.get("method_details") or flow.get("baseline_details") or []
    baseline_ids = flow.get("baseline_ids") or method_ids
    baseline_details = flow.get("baseline_details") or method_details
    ablation_modes = flow.get("ablation_modes") or []
    baseline_method_ids = [
        d.get("method_id") for d in method_details
        if isinstance(d, dict) and d.get("method_kind") == "baseline" and d.get("method_id")
    ]
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO responses(
                session_id, participant_id, item_index, flow_id, method_id, task,
                context_family, output_variant_id, rating, confidence, free_text,
                elapsed_ms, created_at_ms, item_json, raw_payload_json,
                item_id, baseline_ids_json, baseline_count, baseline_details_json,
                method_ids_json, method_count, method_details_json,
                ablation_modes_json, baseline_method_ids_json,
                output_variant_label, output_variant_description, variant_privacy_class, final_output_type,
                final_output_schema, information_types_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            participant_id,
            index,
            flow.get("flow_id"),
            "context_output_human_rating",
            flow.get("task"),
            flow.get("context_family"),
            flow.get("output_variant_id"),
            rating,
            confidence_int,
            free_text,
            elapsed_int,
            now_ms(),
            json.dumps(item, sort_keys=True),
            json.dumps(raw_payload, sort_keys=True),
            flow.get("item_id"),
            json.dumps(baseline_ids, sort_keys=True),
            len(baseline_ids),
            json.dumps(baseline_details, sort_keys=True),
            json.dumps(method_ids, sort_keys=True),
            len(method_ids),
            json.dumps(method_details, sort_keys=True),
            json.dumps(ablation_modes, sort_keys=True),
            json.dumps(baseline_method_ids, sort_keys=True),
            flow.get("output_variant_label"),
            flow.get("output_variant_description"),
            flow.get("variant_privacy_class"),
            cap_type(final_cap),
            cap_schema(final_cap),
            json.dumps(output.get("information_types") or {}, sort_keys=True),
        ))
        conn.commit()


def assign_items(items: List[Dict[str, Any]], k: int, seed: int, participant_id: str, session_id: str, mode: str, db_path: Path) -> List[Dict[str, Any]]:
    rng = random.Random(seed + stable_int(participant_id) + stable_int(session_id))
    indexed = list(enumerate(items))
    if not indexed:
        return []
    if mode == "sequential":
        offset = stable_int(session_id) % len(indexed)
        return [
            {
                "item_index": indexed[(offset+j) % len(indexed)][0],
                "item_id": indexed[(offset+j) % len(indexed)][1].get("item_id"),
                "flow_id": indexed[(offset+j) % len(indexed)][1].get("flow_id"),
                "output_variant_id": (indexed[(offset+j) % len(indexed)][1].get("output_variant") or {}).get("output_variant_id"),
            }
            for j in range(k)
        ]

    prior_counts: Dict[str, int] = {}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT assignment_json FROM sessions").fetchall():
            try:
                for a in json.loads(row["assignment_json"]):
                    iid = a.get("item_id") or f"{a.get('flow_id')}::{a.get('output_variant_id')}"
                    prior_counts[iid] = prior_counts.get(iid, 0) + 1
            except Exception:
                pass

    by_task: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, item in indexed:
        flow = item.get("flow") or {}
        by_task.setdefault(str(flow.get("task") or "unknown"), []).append((idx, item))

    tasks = sorted(by_task)
    chosen: List[Dict[str, Any]] = []
    used = set()
    target_n = min(k, len(items))
    while len(chosen) < target_n:
        any_added = False
        for task in tasks:
            if len(chosen) >= target_n:
                break
            candidates = [x for x in by_task[task] if x[0] not in used]
            if not candidates:
                continue
            rng.shuffle(candidates)
            candidates.sort(key=lambda x: (prior_counts.get(str(x[1].get("item_id")), 0), rng.random()))
            idx, item = candidates[0]
            chosen.append({
                "item_index": idx,
                "item_id": item.get("item_id"),
                "flow_id": item.get("flow_id"),
                "output_variant_id": (item.get("output_variant") or {}).get("output_variant_id"),
            })
            used.add(idx)
            any_added = True
        if not any_added:
            break
    rng.shuffle(chosen)
    return chosen[:target_n]


def _field_value_by_label(flow: Dict[str, Any], label: str) -> Optional[str]:
    for row in flow.get("participant_display_fields", []) or []:
        if row.get("label") == label:
            val = row.get("value")
            return str(val) if val is not None else None
    return None


def _task_label(flow: Dict[str, Any]) -> str:
    task = str(flow.get("task") or "")
    if task in TASK_LABEL_OVERRIDES:
        return TASK_LABEL_OVERRIDES[task]
    return str(flow.get("task_label") or flow.get("task") or "smart-space task").replace("_", " ")



def _row_value_by_original_label(flow: Optional[Dict[str, Any]], label: str) -> str:
    if not flow:
        return ""
    for row in flow.get("participant_display_fields", []) or []:
        if row.get("label") == label and row.get("value") not in (None, ""):
            return str(row.get("value"))
    return ""


def _split_situation_value(value: Any) -> Tuple[str, str]:
    text = str(value or "").strip()
    if "—" in text:
        left, right = text.split("—", 1)
        return left.strip(), right.strip()
    if " - " in text:
        left, right = text.split(" - ", 1)
        return left.strip(), right.strip()
    return text, ""


def _plain_setting_value(value: Any) -> str:
    """Participant-facing setting text that does not depend on Markdown/HTML rendering."""
    setting, space = _split_situation_value(value)
    if setting and space:
        return f"{setting} — {space}"
    return setting or str(value or "")


def _setting_parts(flow: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    situation = _row_value_by_original_label(flow, "Situation")
    return _split_situation_value(situation)


def _setting_plain(flow: Optional[Dict[str, Any]]) -> str:
    setting, _space = _setting_parts(flow)
    return setting


def _subject_plain(flow: Optional[Dict[str, Any]]) -> str:
    return _row_value_by_original_label(flow, "Data subject") or "person"


def _purpose_plain(flow: Optional[Dict[str, Any]]) -> str:
    return _row_value_by_original_label(flow, "Purpose") or "the stated purpose"


def _task_specific_app_label(flow: Optional[Dict[str, Any]]) -> str:
    purpose = _purpose_plain(flow).lower()
    task = str((flow or {}).get("task") or "").lower()
    if "fall" in purpose or "fall" in task:
        return "the fall-detection app or software service"
    if "sound" in task or "sound" in purpose or "communication" in purpose:
        return "the sound-monitoring app or software service"
    if "adl" in task or "activity" in task:
        if "personalization" in purpose:
            return "the daily-activity recognition/personalization app or software service"
        return "the daily-activity recognition app or software service"
    if "visitor" in task or "presence" in task:
        if "energy" in purpose or "lighting" in purpose:
            return "the visitor/presence-detection app or energy/lighting automation service"
        return "the visitor/presence-detection app or software service"
    if "energy" in purpose or "lighting" in purpose:
        return "the energy or lighting automation app/software service"
    if "personalization" in purpose:
        return "the personalization app or software service"
    return "the task-specific app or software service"


def _event_phrase(flow: Optional[Dict[str, Any]]) -> str:
    purpose = _purpose_plain(flow).lower()
    task = str((flow or {}).get("task") or "").lower()
    if "fall" in purpose or "fall" in task:
        return "a possible fall"
    if "sound" in task or "communication" in purpose:
        return "a relevant sound event, such as an alarm, glass breaking, footsteps, a loud noise, or another non-speech sound pattern"
    if "activity" in task or "adl" in task or "routine" in purpose:
        return "a daily-activity event, such as cooking, sitting, lying down, or room use"
    if "visitor" in task or "security" in purpose:
        return "a visitor, presence, or intrusion event"
    if "energy" in purpose or "lighting" in purpose:
        return "an occupancy or room-use event"
    if "safety" in purpose:
        return "a safety-relevant event, such as a possible fall, alarm, or dangerous sound"
    return "a relevant event for the stated purpose"


def _contextual_actor_value(original_label: str, original_value: str, flow: Optional[Dict[str, Any]]) -> Optional[str]:
    setting_plain = _setting_plain(flow)
    subject = _subject_plain(flow)
    text = original_value
    if not setting_plain:
        return None

    if original_label == "Sender":
        sender_map = {
            "resident-owned device": f"resident-owned device in the {setting_plain}",
            "rental host’s device": f"rental host’s device in or around the {setting_plain}",
            "organization-operated system": f"organization-operated system in the {setting_plain}",
            "hospital monitoring system": f"hospital monitoring system in the {setting_plain}",
            "patient/family device": f"patient/family device in the {setting_plain}",
            "research sensor system": f"research sensor system in the {setting_plain}",
            "school monitoring system": f"school monitoring system in the {setting_plain}",
        }
        return sender_map.get(text)

    if original_label == "Data subject":
        return f"{text} in the {setting_plain}"

    if original_label == "Recipient":
        recipient_map = {
            "homeowner/resident": f"homeowner/resident for the {setting_plain}",
            "rental host": f"rental host for the {setting_plain}",
            "authorized staff": f"authorized staff for the {setting_plain}",
            "caregiver": f"caregiver for the {subject} in the {setting_plain}",
            "clinician": f"clinician caring for the {subject} in the {setting_plain}",
            "researcher": f"researcher for the {setting_plain} study",
            "teacher or school official": f"teacher or school official for the {setting_plain}",
            "the smart-space app": _task_specific_app_label(flow),
            "the app or software service": _task_specific_app_label(flow),
        }
        return recipient_map.get(text)
    return None


def _purpose_display_value(original_value: str, flow: Optional[Dict[str, Any]]) -> str:
    """Make purpose values self-explanatory and include concrete examples in the main text."""
    purpose = str(original_value or "").strip()
    task = str((flow or {}).get("task") or "").lower()
    if purpose == "routine monitoring":
        if "sound" in task:
            return "everyday sound monitoring, such as detecting noise levels, alarms, footsteps, loud sounds, or other non-speech sound patterns over time"
        return "daily activity monitoring, such as detecting cooking, sleeping, movement, or room-use patterns over time"
    if purpose == "safety monitoring":
        if "sound" in task:
            return "sound-based safety monitoring, such as detecting alarms, breaking glass, yelling, or other concerning sounds"
        if "fall" in task:
            return "fall/safety monitoring, such as detecting a possible fall or urgent safety event"
        return "safety-event monitoring, such as detecting a possible fall, alarm, dangerous sound, or other urgent event"
    if purpose == "security monitoring":
        return "visitor/security monitoring, such as detecting someone arriving, entering, or approaching the area"
    if purpose == "energy or lighting automation":
        return "presence-based energy or lighting automation, such as turning lights or HVAC on/off when people enter, leave, or occupy the area"
    if purpose == "work performance monitoring":
        return "employee activity or presence monitoring, such as tracking whether employees are present, moving, or active in the work area"
    if purpose == "clinical care":
        if "sound" in task:
            return "clinical care support, such as alerting clinicians to patient-room sounds or sound patterns that may need attention"
        if "activity" in task or "adl" in task:
            return "clinical care support, such as tracking patient movement, room use, or daily activities relevant to care"
        return "clinical care support, such as helping clinicians monitor patient safety or care needs"
    if purpose == "fall detection":
        return "fall detection, meaning detecting whether a person may have fallen and may need help"
    if purpose == "personalization":
        return "personalization, such as adapting home services based on room use or daily activity patterns"
    if purpose == "research study":
        return "research study, such as studying daily activities or sensor behavior with research participants"
    if purpose == "training or supervision":
        return "training or supervision, such as school staff reviewing activity patterns or events for oversight"
    if purpose == ("voice " + "command or communication"):
        return "audio-event support, such as detecting that a relevant home sound or audio cue occurred without sharing full speech content"
    return purpose


def _purpose_description(original_value: str, flow: Optional[Dict[str, Any]]) -> str:
    value = _purpose_display_value(original_value, flow)
    return f"This describes what the sensing-and-software system is trying to do with the collected data or shared output: {value}."


def _display_value_override(original_label: str, value: Any, flow: Optional[Dict[str, Any]] = None) -> Any:
    if value is None:
        return value
    text = str(value)

    if original_label == "Situation":
        return _plain_setting_value(text)

    if original_label == "Purpose":
        return _purpose_display_value(text, flow)

    contextual = _contextual_actor_value(original_label, text, flow)
    if contextual:
        return contextual

    if text == "the smart-space app":
        return _task_specific_app_label(flow)

    if text == "only when an event occurs":
        return f"data is collected or shared only after {_event_phrase(flow)} is detected"

    replacements = {
        "continuous monitoring": "data collection continues without pause",
        "processed locally": "processed on the device or local hub, not in a third-party cloud",
        "not disclosed to the person": "data collection and sharing are not disclosed to the monitored person",
        "disclosed to people nearby": "people in this setting are told that data is being collected or shared",
        "written notice is provided": "affected people receive written notice about collection and sharing",
        "speech content is removed": "spoken words are removed before any sound-monitoring output is shared",
        "sent to a cloud service": "sensor data is sent to a cloud service for processing",
        "explicit consent is obtained": "the monitored person gives explicit consent for this collection and sharing",
        "only authorized people can access it": "only authorized staff can access the shared output",
        "disclosed in the rental listing": "the rental listing discloses the device, coverage area, and data sharing",
    }
    return replacements.get(text, text)


def _transmission_help_and_example(original_value: str, flow: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    event_phrase = _event_phrase(flow)
    if original_value == "continuous monitoring":
        return (
            "The sensing-and-software system keeps collecting sensor data in this setting without pause, rather than waiting for a specific event before collecting or analyzing data.",
            "The sensor stays on throughout the day or night and keeps collecting data continuously.",
        )
    if original_value == "only when an event occurs":
        return (
            f"The sensing-and-software system waits until it detects {event_phrase} before collecting, analyzing, or sharing the listed output.",
            f"The sensing-and-software system sends the listed output only when it detects {event_phrase}.",
        )
    if original_value == "not disclosed to the person":
        return (
            "The person being monitored is not told that the device is collecting data or that the listed output may be shared.",
            "For example, a hidden rental camera or sensor collects data without telling the guest.",
        )
    if original_value == "processed locally":
        return (
            "Sensor data is analyzed on the device or local hub and is not sent to a third-party cloud service for analysis. The listed output may still be shared with the stated recipient.",
            "A local device or home hub detects the event before any listed output is shared.",
        )
    if original_value == "disclosed to people nearby":
        return (
            "People nearby are told that the device is collecting data and/or that the listed output may be shared.",
            "A sign, notice, or setup screen explains that the sensor is active and what kind of output may be shared.",
        )
    if original_value == "written notice is provided":
        return (
            "Affected people receive written notice explaining the device, collection, recipient, and purpose. For children, this may mean parent/guardian notice or a school-required process.",
            "A written workplace, school, care, or study notice explains the monitoring and sharing.",
        )
    if original_value == "sent to a cloud service":
        return (
            "Sensor data is sent to a remote cloud service for analysis or processing instead of being analyzed only on a local device or hub.",
            "Audio, video, or sensor data is sent to a vendor cloud service before the listed output is produced or shared.",
        )
    if original_value == "explicit consent is obtained":
        return (
            "The monitored person clearly agrees to this data collection and sharing for the stated purpose. For children or dependent adults, the relevant consent process may involve a parent, guardian, or legally authorized representative.",
            "A signed consent form or clear opt-in explains what data is collected, who receives it, and why.",
        )
    if original_value == "only authorized people can access it":
        return (
            "Only approved people for this setting and purpose can access the listed output.",
            "Only approved care staff, clinical staff, security staff, or designated managers can see the alert or output.",
        )
    if original_value == "disclosed in the rental listing":
        return (
            "The short-term rental listing describes the device, where it is located, what area it covers, and what kind of output may be shared.",
            "The listing says an outdoor camera covers the entrance and explains what information the host may receive.",
        )
    if original_value == "speech content is removed":
        return (
            "Spoken words or conversation content are removed before the output is shared; non-speech sounds, sound labels, or sound levels may still remain.",
            "The shared output may identify an alarm, footsteps, a loud sound, or a sound level, but not the words someone said.",
        )
    return None, None


def _plain_field(row: Dict[str, Any], flow: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Rename technical CI labels into lay survey labels without changing normalized CI terms."""
    label = str(row.get("label") or "")
    mapping = {
        "Situation": (
            "Where this happens",
            "The overall setting and specific physical area where the sensing-and-software system is used.",
        ),
        "Sender": (
            "Who/what collects or sends the data",
            "The device, sensing system, software system, or organization that collects data in this setting and/or sends the listed output.",
        ),
        "Data subject": (
            "Person the data is about",
            "The person in this setting whose activity, presence, or surroundings are represented by the data.",
        ),
        "Recipient": (
            "Who receives the shared output",
            "The person, organization, app, or software service that would receive the listed output for this scenario.",
        ),
        "Purpose": (
            "What the sensing-and-software system is trying to do",
            "The concrete task or reason for collecting, using, or sharing the data in this scenario.",
        ),
        "Transmission principle": (
            "Condition for collection, processing, or sharing",
            "What must be true about collection, processing, notice, consent, access, or sharing in this scenario.",
        ),
    }
    new_label, desc = mapping.get(label, (label, str(row.get("description") or row.get("help") or "")))
    original_value = str(row.get("value") or "")
    value = _display_value_override(label, row.get("value"), flow)
    # Scenario-detail rows should not show separate examples. The main value and
    # description now carry the concrete meaning; examples created confusion in
    # the web UI because they looked like additional scenario facts.
    example = None

    if label == "Purpose":
        desc = _purpose_description(original_value, flow)

    if label == "Transmission principle":
        help_override, _example_override = _transmission_help_and_example(original_value, flow)
        if help_override:
            desc = help_override

    field = {
        "label": new_label,
        "value": value,
        "description": desc,
        "help": desc,
        "example": example,
    }
    if label == "Situation":
        setting, space = _split_situation_value(original_value)
        field["value_parts"] = {"general_setting": setting, "specific_place": space}
        field["value_html"] = f"<strong>{setting}</strong> — {space}" if setting and space else (f"<strong>{setting}</strong>" if setting else str(value))
    return field

def output_example(label: Any, description: Any = None) -> str:
    """Short example text for the participant-facing 'What would be shared' row."""
    text = f"{label or ''} {description or ''}".lower()
    if "pose" in text or "stick-figure" in text:
        return "Example: a stick-figure skeleton showing body joint positions."
    if "audio-video" in text or "video and audio" in text:
        return "Example: a short camera-and-audio clip from the space."
    if "camera video" in text or "video footage" in text:
        return "Example: video footage from a camera, showing motion over time."
    if "camera image" in text or "image" in text:
        return "Example: one camera image from the space, not video footage."
    if "spoken words" in text or "speech" in text:
        return "Example: audio where words are removed, but sounds like alarms or footsteps may remain."
    if "audio recording" in text:
        return "Example: recorded sound from the space."
    if "presence" in text or "occupancy" in text:
        return "Example: a yes/no presence signal or a count of people in the room."
    if "sound level" in text or "decibel" in text:
        return "Example: a decibel level, not the original audio."
    if "sound category" in text or "noise" in text:
        return "Example: a label such as alarm, glass breaking, or footsteps."
    if "activity" in text:
        return "Example: a label such as walking, cooking, sitting, or lying down."
    if "event alert" in text or "event label" in text:
        return "Example: an alert saying that a relevant event was detected."
    return "Example: the specific output named in this row."

def _task_overview_text(flow: Optional[Dict[str, Any]]) -> str:
    setting, space = _setting_parts(flow)
    setting_phrase = f"In this {setting}" if setting else "In this setting"
    if space:
        setting_phrase += f" ({space})"
    task = str((flow or {}).get("task") or "").lower()
    purpose_value = _purpose_display_value(_purpose_plain(flow), flow)
    if "visitor" in task or "presence" in task:
        task_text = "a sensing-and-software system is doing presence detection, meaning it detects when people enter, leave, approach, or are present in the area"
    elif "fall" in task:
        task_text = "a sensing-and-software system is doing fall detection, meaning it looks for signs that a person may have fallen and may need help"
    elif "adl" in task or "activity" in task:
        task_text = "a sensing-and-software system is doing daily activity recognition, meaning it detects activity patterns such as cooking, sleeping, movement, sitting, lying down, or room use"
    elif "sound" in task:
        task_text = "a sensing-and-software system is doing sound monitoring, meaning it detects or classifies everyday environmental audio events and sound patterns, such as alarms, glass breaking, footsteps, loud/noise levels, or other non-speech sounds"
    else:
        task_text = "a sensing-and-software system is performing the smart-space task described below"
    return f"{setting_phrase}, {task_text}. In this scenario, the data is collected or shared for {purpose_value}."


def _scenario_overview_field(flow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    overview = _task_overview_text(flow)
    return {
        "label": "Scenario overview",
        "value": overview,
        "description": "",
        "help": "",
        "example": None,
    }


def participant_display_fields(flow: Dict[str, Any], output_variant: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = {"Situation", "Sender", "Data subject", "Recipient", "Purpose", "Transmission principle"}
    fields = [_scenario_overview_field(flow)]
    fields.extend(_plain_field(r, flow) for r in flow.get("participant_display_fields", []) if r.get("label") in keep)
    if output_variant:
        output_desc = output_variant.get("output_variant_description") or "The data or output that would be sent to the stated recipient in this scenario."
        fields.append({
            "label": "What would be shared",
            "value": output_variant.get("output_variant_label"),
            "description": output_desc,
            "help": output_desc,
            "example": output_example(output_variant.get("output_variant_label"), output_desc),
        })
    return fields


def _indefinite_subject_phrase(subject: str) -> str:
    subject = str(subject or "a person").strip()
    if not subject:
        return "a person"
    lower = subject.lower()
    if lower.startswith(("a ", "an ", "the ", "their ")):
        return subject
    if lower in {"child", "guest", "patient", "resident", "roommate", "visitor", "research participant", "caregiver", "clinician"}:
        article = "an" if lower[0] in "aeiou" else "a"
        return f"{article} {subject}"
    if lower == "employee":
        return "an employee"
    return subject


def _display_value(fields: List[Dict[str, Any]], label: str, default: str) -> str:
    for field in fields:
        if field.get("label") == label and field.get("value") not in (None, ""):
            return str(field.get("value"))
    return default


def plain_vignette(flow: Dict[str, Any], output_variant: Optional[Dict[str, Any]]) -> str:
    """Instruction text shown above the scenario details.

    The table below already recaps the scenario. This text explains the rating
    task instead of repeating the same fields in paragraph form.
    """
    return (
        "Review the scenario overview and details below, then rate whether it is appropriate "
        "for this sensing-and-software system to share the listed data or output in this situation. "
        "Here, appropriate means the sharing seems acceptable and justified for "
        "this specific setting, person, recipient, purpose, handling condition, "
        "and type of data shared."
    )


def materialize_survey_item(items: List[Dict[str, Any]], assignment: Dict[str, Any], index: int, total: int) -> Dict[str, Any]:
    base = items[int(assignment["item_index"])]
    flow = base["flow"]
    output_variant = base.get("output_variant")
    method_ids = list(base.get("method_ids") or base.get("baseline_ids") or [])
    method_details = list(base.get("method_details") or base.get("baseline_details") or [])
    baseline_ids = list(base.get("baseline_ids") or method_ids)
    ablation_modes = list(base.get("ablation_modes") or [])
    return {
        "index": index,
        "total": total,
        "item_id": base.get("item_id"),
        "flow_id": base.get("flow_id"),
        "flow": {
            "item_id": base.get("item_id"),
            "flow_id": base.get("flow_id"),
            "task": flow.get("task"),
            "task_label": _task_label(flow),
            "context_family": flow.get("context_family"),
            "context_bundle_id": flow.get("context_bundle_id"),
            "context_bundle_label": flow.get("context_bundle_label"),
            "expected_acceptability_prior_for_stratification": flow.get("expected_acceptability_prior_for_stratification"),
            "output_variant_id": (output_variant or {}).get("output_variant_id"),
            "output_variant_label": (output_variant or {}).get("output_variant_label"),
            "output_variant_description": (output_variant or {}).get("output_variant_description"),
            "variant_privacy_class": (output_variant or {}).get("variant_privacy_class"),
            "method_ids": method_ids,
            "method_count": len(method_ids),
            "method_details": method_details,
            "baseline_ids": baseline_ids,
            "baseline_count": len(baseline_ids),
            "baseline_details": list(base.get("baseline_details") or method_details),
            "ablation_modes": ablation_modes,
        },
        "vignette": plain_vignette(flow, output_variant),
        "display_fields": participant_display_fields(flow, output_variant),
        "output_data_slot": {
            "status": "included_from_generated_pipeline_outputs" if output_variant else "not_included_in_context_only_survey",
            "output_data": (output_variant or {}).get("output_variant_label"),
            "output_variant_id": (output_variant or {}).get("output_variant_id"),
            "output_variant_label": (output_variant or {}).get("output_variant_label"),
            "output_variant_description": (output_variant or {}).get("output_variant_description"),
            "variant_privacy_class": (output_variant or {}).get("variant_privacy_class"),
            "final_output_cap": (output_variant or {}).get("final_output_cap"),
            "information_types": (output_variant or {}).get("information_types"),
            "matched_output_cap": (output_variant or {}).get("matched_output_cap"),
            "matched_output_schema": (output_variant or {}).get("matched_output_schema"),
            "method_ids": method_ids,
            "method_details": method_details,
            "baseline_ids": baseline_ids,
            "baseline_details": list(base.get("baseline_details") or method_details),
            "ablation_modes": ablation_modes,
        },
        "ci_constraint_evaluation_hidden_from_participant": flow.get("ci_constraint_evaluation", {}),
    }

def export_rows(db_path: Path) -> List[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM responses ORDER BY created_at_ms ASC").fetchall()]


def summary(db_path: Path, state: SurveyState) -> Dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        session_count = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
        response_count = conn.execute("SELECT COUNT(*) AS c FROM responses").fetchone()["c"]
        by_task = [dict(r) for r in conn.execute("SELECT task, COUNT(*) AS n, AVG(rating) AS avg_rating FROM responses GROUP BY task").fetchall()]
        by_output = [dict(r) for r in conn.execute("SELECT output_variant_id, output_variant_label, COUNT(*) AS n, AVG(rating) AS avg_rating FROM responses GROUP BY output_variant_id, output_variant_label").fetchall()]
    rated_item_ids = {r.get("item_id") for r in export_rows(db_path) if r.get("item_id")}
    out = {
        **state.item_pool_summary,
        "pipeline_load_info": state.pipeline_load_info,
        "offline_methods": state.pipeline_load_info.get("method_ids") or OFFLINE_METHODS,
        "baseline_method_ids": state.pipeline_load_info.get("baseline_method_ids", []),
        "ablation_method_ids": state.pipeline_load_info.get("ablation_method_ids", []),
        "session_count": session_count,
        "response_count": response_count,
        "by_task": by_task,
        "by_output_variant": by_output,
        "rated_survey_item_count": len(rated_item_ids),
        "mean_ratings_per_rated_item": (response_count / len(rated_item_ids)) if rated_item_ids else 0,
    }
    return out


def build_question_preview(
    state: SurveyState,
    participant_id: str,
    session_id: str,
    k: int,
) -> Dict[str, Any]:
    """Build a JSON preview of the exact questions the web UI would show.

    This does not create a survey session or save any response rows. It uses the
    same assignment and materialization functions as the live web UI so that the
    preview stays aligned with the participant-facing survey pages.
    """
    target_k = max(1, min(int(k), len(state.items))) if state.items else 0
    assignment = assign_items(
        state.items,
        target_k,
        state.config.seed,
        participant_id,
        session_id,
        state.config.assignment_mode,
        state.config.db_path,
    )
    questions: List[Dict[str, Any]] = []
    rating_scale = [
        {"value": 1, "label": "Completely inappropriate"},
        {"value": 2, "label": "Somewhat inappropriate"},
        {"value": 3, "label": "Neutral / unsure"},
        {"value": 4, "label": "Somewhat appropriate"},
        {"value": 5, "label": "Completely appropriate"},
    ]
    for i, a in enumerate(assignment):
        item = materialize_survey_item(state.items, a, i, len(assignment))
        flow = item.get("flow") or {}
        output = item.get("output_data_slot") or {}
        questions.append({
            "question_number": i + 1,
            "item_id": item.get("item_id"),
            "flow_id": item.get("flow_id"),
            "output_variant_id": flow.get("output_variant_id"),
            "participant_visible": {
                "task_title": flow.get("task_label") or flow.get("task"),
                "vignette": item.get("vignette"),
                "display_fields": item.get("display_fields") or [],
                "rating_prompt": "In your judgment, how appropriate is it for this system to share the listed data or output in this situation?",
                "rating_scale": rating_scale,
                "confidence_prompt": "How confident are you in this rating?",
                "free_text_prompt": "Optional: What made this appropriate or inappropriate?",
            },
            "output_summary": {
                "label": output.get("output_variant_label"),
                "description": output.get("output_variant_description"),
                "privacy_class": output.get("variant_privacy_class"),
            },
            "method_linkage_hidden_from_participant": {
                "method_ids": output.get("method_ids") or [],
                "method_details": output.get("method_details") or [],
                "baseline_ids": output.get("baseline_ids") or [],
                "ablation_modes": output.get("ablation_modes") or [],
            },
            "technical_output_hidden_from_participant": {
                "final_output_cap": output.get("final_output_cap"),
                "information_types": output.get("information_types"),
                "matched_output_cap": output.get("matched_output_cap"),
                "matched_output_schema": output.get("matched_output_schema"),
            },
            "source_context_hidden_from_participant": {
                "context_family": flow.get("context_family"),
                "context_bundle_id": flow.get("context_bundle_id"),
                "context_bundle_label": flow.get("context_bundle_label"),
                "expected_acceptability_prior_for_stratification": flow.get("expected_acceptability_prior_for_stratification"),
            },
        })
    return {
        "schema_version": "smart_space_survey_question_preview_v1",
        "generated_at_ms": now_ms(),
        "note": (
            "Preview generated from the same assignment/materialization path used by the web survey. "
            "It is for debugging and IRB/study-review inspection; it does not create a participant session."
        ),
        "assignment": {
            "participant_id": participant_id,
            "session_id": session_id,
            "k": len(assignment),
            "seed": state.config.seed,
            "assignment_mode": state.config.assignment_mode,
        },
        "source_files": {
            "flow_file": str(state.config.flow_file),
            "pipeline_output_dir": str(state.config.pipeline_output_dir) if state.config.pipeline_output_dir else None,
            "db_path_for_prior_assignment_counts": str(state.config.db_path),
        },
        "survey_item_pool_summary": state.item_pool_summary,
        "pipeline_load_info": state.pipeline_load_info,
        "questions": questions,
    }


def write_question_preview(path: Path, state: SurveyState, participant_id: str, session_id: str, k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = build_question_preview(state, participant_id=participant_id, session_id=session_id, k=k)
    with path.open("w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2, sort_keys=False)


def json_response(handler: BaseHTTPRequestHandler, obj: Any, status: int = 200) -> None:
    data = json.dumps(obj, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    try:
        return json.loads(handler.rfile.read(length).decode("utf-8"))
    except Exception:
        return {}


def make_handler(state: SurveyState):
    class SurveyHandler(BaseHTTPRequestHandler):
        server_version = "CISurveyHTTP/4.0-output-augmented"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

        def do_GET(self) -> None:
            try:
                self.handle_get()
            except Exception as exc:
                json_response(self, {"error": str(exc)}, 500)

        def do_POST(self) -> None:
            try:
                self.handle_post()
            except Exception as exc:
                json_response(self, {"error": str(exc)}, 500)

        def handle_get(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                return self.serve_file(ROOT / "templates" / "index.html", "text/html; charset=utf-8")
            if path.startswith("/static/"):
                static_path = (ROOT / "static" / path[len("/static/"):]).resolve()
                if ROOT / "static" not in static_path.parents and static_path != ROOT / "static":
                    return json_response(self, {"error": "invalid static path"}, 400)
                return self.serve_file(static_path)
            if path == "/api/config":
                return json_response(self, {
                    "k": state.config.k,
                    "assignment_mode": state.config.assignment_mode,
                    "scenario_count": len(state.flows),
                    "survey_item_pool_count": len(state.items),
                    "output_augmented": state.item_pool_summary.get("output_augmented"),
                    "method_pipeline_row_count": state.item_pool_summary.get("method_pipeline_row_count"),
                    "baseline_pipeline_row_count": state.item_pool_summary.get("baseline_pipeline_row_count"),
                    "ablation_pipeline_row_count": state.item_pool_summary.get("ablation_pipeline_row_count"),
                    "baseline_pipeline_rows_with_output": state.item_pool_summary.get("baseline_pipeline_rows_with_output"),
                    "deduplicated_output_variant_count": state.item_pool_summary.get("deduplicated_output_variant_count"),
                    "mean_output_variants_per_context": state.item_pool_summary.get("mean_output_variants_per_context"),
                    "offline_methods": state.pipeline_load_info.get("method_ids") or OFFLINE_METHODS,
                    "baseline_method_ids": state.pipeline_load_info.get("baseline_method_ids", []),
                    "ablation_method_ids": state.pipeline_load_info.get("ablation_method_ids", []),
                    "pipeline_load_info": state.pipeline_load_info,
                    "counts": state.flow_data.get("counts", {}),
                })
            if path.startswith("/api/session/"):
                return self.handle_session_get(path)
            if path == "/admin/summary":
                return json_response(self, summary(state.config.db_path, state))
            if path == "/admin/export.json":
                return json_response(self, {"responses": export_rows(state.config.db_path), "summary": summary(state.config.db_path, state)})
            if path == "/admin/export.csv":
                return self.export_csv()
            if path == "/admin/flows.json":
                return json_response(self, state.flow_data)
            if path == "/admin/survey_items.json":
                return json_response(self, {"summary": state.item_pool_summary, "items": state.items})
            if path == "/admin/pipeline_rows.json":
                return json_response(self, {"pipeline_load_info": state.pipeline_load_info, "rows": state.pipeline_rows})
            if path == "/admin/glossary.json":
                return json_response(self, state.flow_data.get("human_readable_glossary", {}))
            return json_response(self, {"error": "not found"}, 404)

        def handle_post(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/start":
                payload = read_json_body(self)
                participant_code = str(payload.get("participant_code") or "").strip()
                k = max(1, min(int(payload.get("k") or state.config.k), len(state.items)))
                participant_id = participant_code or f"P-{uuid.uuid4().hex[:10]}"
                session_id = uuid.uuid4().hex
                assignment = assign_items(state.items, k, state.config.seed, participant_id, session_id, state.config.assignment_mode, state.config.db_path)
                create_session(state.config.db_path, session_id, participant_id, assignment, payload)
                return json_response(self, {"session_id": session_id, "participant_id": participant_id, "k": len(assignment), "first_index": 0})
            if path.startswith("/api/session/") and path.endswith("/submit"):
                parts = path.strip("/").split("/")
                if len(parts) != 4:
                    return json_response(self, {"error": "bad submit path"}, 400)
                return self.handle_submit(parts[2])
            return json_response(self, {"error": "not found"}, 404)

        def handle_session_get(self, path: str) -> None:
            parts = path.strip("/").split("/")
            if len(parts) == 3:
                session = get_session(state.config.db_path, parts[2])
                if not session:
                    return json_response(self, {"error": "unknown session"}, 404)
                responses = get_responses_for_session(state.config.db_path, parts[2])
                return json_response(self, {"session_id": parts[2], "participant_id": session["participant_id"], "k": len(session["assignment"]), "answered": len(responses), "completed": len(responses) >= len(session["assignment"])})
            if len(parts) == 5 and parts[3] == "item":
                session = get_session(state.config.db_path, parts[2])
                if not session:
                    return json_response(self, {"error": "unknown session"}, 404)
                index = int(parts[4])
                assignment = session["assignment"]
                if index < 0 or index >= len(assignment):
                    return json_response(self, {"error": "index out of range"}, 404)
                item = materialize_survey_item(state.items, assignment[index], index, len(assignment))
                item["existing_response"] = get_response(state.config.db_path, parts[2], index)
                return json_response(self, item)
            return json_response(self, {"error": "not found"}, 404)

        def handle_submit(self, session_id: str) -> None:
            payload = read_json_body(self)
            session = get_session(state.config.db_path, session_id)
            if not session:
                return json_response(self, {"error": "unknown session"}, 404)
            index = int(payload.get("index", -1))
            assignment = session["assignment"]
            if index < 0 or index >= len(assignment):
                return json_response(self, {"error": "index out of range"}, 400)
            rating = int(payload.get("rating"))
            if rating < 1 or rating > 5:
                return json_response(self, {"error": "rating must be between 1 and 5"}, 400)
            item = materialize_survey_item(state.items, assignment[index], index, len(assignment))
            save_response(state.config.db_path, session_id, session["participant_id"], index, item, rating, payload.get("confidence"), str(payload.get("free_text") or "").strip(), payload.get("elapsed_ms"), payload)
            answered = len(get_responses_for_session(state.config.db_path, session_id))
            return json_response(self, {"ok": True, "answered": answered, "k": len(assignment), "completed": answered >= len(assignment)})

        def serve_file(self, path: Path, content_type: Optional[str] = None) -> None:
            if not path.exists() or not path.is_file():
                return json_response(self, {"error": "file not found"}, 404)
            data = path.read_bytes()
            ctype = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def export_csv(self) -> None:
            rows = export_rows(state.config.db_path)
            import io
            buf = io.StringIO()
            header = [
                "session_id", "participant_id", "item_index", "item_id", "flow_id", "method_id",
                "task", "context_family", "output_variant_id", "output_variant_label",
                "output_variant_description", "variant_privacy_class", "final_output_type", "final_output_schema",
                "baseline_ids_json", "baseline_count", "baseline_details_json",
                "method_ids_json", "method_count", "method_details_json",
                "ablation_modes_json", "baseline_method_ids_json",
                "rating", "confidence", "free_text",
                "elapsed_ms", "created_at_ms", "information_types_json",
            ]
            writer = csv.DictWriter(buf, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
            data = buf.getvalue().encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="ci_survey_responses.csv"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return SurveyHandler


def main() -> int:
    p = argparse.ArgumentParser(description="Run an output-augmented CI acceptability survey server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--flow-file", default=str(DEFAULT_FLOW_FILE))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--assignment-mode", default="least_rated_balanced", choices=["least_rated_balanced", "sequential"])
    p.add_argument("--pipeline-output-dir", default=str(DEFAULT_PIPELINE_OUTPUT_DIR), help="Directory created by evaluation.generate_pipelines_for_all_contexts, containing index.json/summary.json.")
    p.add_argument("--no-pipeline-outputs", action="store_true", help="Fall back to context-only survey items.")
    p.add_argument("--include-no-output-variants", action="store_true", help="Also create survey cases for baselines that deny or produce no selected output.")
    p.add_argument(
        "--write-question-preview-json",
        default=None,
        help=(
            "Optional path for a JSON preview of the exact k questions that would be shown in the web UI. "
            "Disabled by default. The preview uses the same assignment logic but does not create a session."
        ),
    )
    p.add_argument(
        "--preview-participant-id",
        default="PREVIEW_PARTICIPANT",
        help="Participant id seed used only for --write-question-preview-json.",
    )
    p.add_argument(
        "--preview-session-id",
        default="PREVIEW_SESSION",
        help="Session id seed used only for --write-question-preview-json.",
    )
    args = p.parse_args()

    pipeline_dir = None if args.no_pipeline_outputs else Path(args.pipeline_output_dir)
    state = SurveyState(Config(
        Path(args.flow_file),
        Path(args.db),
        args.k,
        args.seed,
        args.assignment_mode,
        pipeline_dir,
        include_pipeline_outputs=not args.no_pipeline_outputs,
        include_no_output_variants=args.include_no_output_variants,
    ))
    if args.write_question_preview_json:
        preview_path = Path(args.write_question_preview_json)
        write_question_preview(
            preview_path,
            state,
            participant_id=str(args.preview_participant_id),
            session_id=str(args.preview_session_id),
            k=args.k,
        )
        print(f"Wrote survey question preview JSON: {preview_path}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))

    print(f"Serving {len(state.flows)} context scenarios from {state.config.flow_file}")
    print(f"Survey item pool count: {len(state.items)}")
    if state.item_pool_summary.get("output_augmented"):
        print(f"Loaded generated pipeline outputs from: {state.pipeline_load_info.get('source') or state.pipeline_load_info.get('pipeline_output_dir')}")
        print(f"Baseline pipeline rows: {state.item_pool_summary.get('baseline_pipeline_row_count')}")
        print(f"Pipeline rows with shared output: {state.item_pool_summary.get('baseline_pipeline_rows_with_output')}")
        print(f"Unique context-output survey cases after deduplication: {state.item_pool_summary.get('deduplicated_output_variant_count')}")
        print(f"Mean output variants per context: {state.item_pool_summary.get('mean_output_variants_per_context'):.2f}")
    else:
        print("Pipeline outputs disabled or unavailable; using context-only survey items.")
    print(f"Open http://{args.host}:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
