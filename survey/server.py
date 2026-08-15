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

import argparse, csv, hashlib, html, json, mimetypes, os, random, sqlite3, time, uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DEFAULT_FLOW_FILE = ROOT / "data" / "ci_focused_user_study_context.json"
DEFAULT_DB = ROOT / "outputs" / "responses.db"
# Runner usually writes this from project root. If server.py lives under survey/,
# ROOT.parent is the project root.
DEFAULT_PIPELINE_OUTPUT_DIR = ROOT.parent / "runs" / "flexible_context_pipeline_generation"
DEFAULT_PREVIOUS_PIPELINE_OUTPUT_DIR = ROOT.parent / "runs" / "context_pipeline_generation"
OFFLINE_METHODS = ["raw", "manual", "direct_llm", "full_mediator"]

# Server-side only: this Prolific completion URL is intentionally not embedded in
# index.html.  The browser is redirected through /api/session/<id>/complete_redirect
# after the server has recorded that the assigned survey is complete.
PROLIFIC_COMPLETION_URL = os.environ.get(
    "PROLIFIC_COMPLETION_URL",
    "https://app.prolific.com/submissions/complete?cc=C1FYCZFY",
)

# Participant-facing task labels. These do not change the normalized task values
# stored in the scenario JSON or exported in machine-readable fields.
TASK_LABEL_OVERRIDES = {
    "adl_recognition": "Daily activity recognition",
    "domestic_sound_monitoring": "Sound monitoring",
    "visitor_presence_detection": "Presence detection",
}

# Extra participant-facing output vocabulary that is intentionally kept outside
# ci_focused_user_study_context.json, because that context file is used as a
# computational scenario source.  The server merges these terms into the admin
# glossary endpoint and uses the same term IDs in generated survey items.
EXTRA_OUTPUT_DATA_GLOSSARY = [
    {
        "term": "raw_audio_video",
        "label": "Combined video and audio sample",
        "definition": "A short synchronized clip of both video and audio from the room at the same time. It may show people and the surrounding scene, and the sound may include clear voices, spoken words, and background sounds.",
    },
    {
        "term": "redacted_audio_video",
        "label": "Blurred combined video and audio sample",
        "definition": "A short synchronized video and audio clip where faces, text, and sensitive personal details are blurred in the video. The sound may still include voices, spoken words, and background sounds.",
    },
    {
        "term": "speech_filtered_audio_video",
        "label": "Combined video and speech-muted audio sample",
        "definition": "A short synchronized video and audio clip where human words are scrubbed out so conversations cannot be understood. The video itself is not blurred, and background noises may remain.",
    },
    {
        "term": "redacted_filtered_audio_video",
        "label": "Blurred combined video and speech-muted audio sample",
        "definition": "A short synchronized video and audio clip where faces, text, and sensitive personal details are blurred in the video, and human words are scrubbed out so conversations cannot be understood. Background noises may remain.",
    },
]
# Output schemas that were introduced by the flexible app-request run and were
# not present in the earlier fixed/downstream-compatible run.  When the server is
# used for a supplemental survey, it keeps only these schemas so participants are
# not paid to re-rate output types that were already covered by the original
# survey.
SUPPLEMENTAL_FLEXIBLE_OUTPUT_SCHEMAS = {
    # These are genuinely new participant-facing data types introduced by the
    # flexible run.  Media-style outputs such as raw/blurred image, raw/blurred
    # video, audio recordings, speech-muted audio, pose outlines, and combined
    # video+audio samples are intentionally excluded because they were already
    # covered by the original survey vocabulary/items, even if the flexible run
    # uses a different internal schema name such as redacted_video_stream.
    "object_detections",
    "occupancy_count",
    "room_occupied",
    "aggregate_summary",
    "fused_event_record",
}

SUPPLEMENTAL_OUTPUT_DATA_GLOSSARY = [
    {
        "term": "person_detections",
        "label": "Location boxes (no faces/images)",
        "definition": "A text list or grid map showing where detected people are located in the camera view. No original photos or videos are shared.",
    },
    {
        "term": "room_occupied",
        "label": "Yes/no room presence indicator",
        "definition": 'A simple text status showing either "Occupied" or "Empty." No original video, photo, or audio is shared.',
    },
    {
        "term": "occupancy_count",
        "label": "People count log",
        "definition": 'A number showing how many people are in the room over time, such as "3 people present." No original video, photo, or audio is shared.',
    },
    {
        "term": "activity_summary",
        "label": "Activity summary report",
        "definition": "A daily, weekly, or monthly chart showing broad activity patterns, such as occupied versus empty time. It does not show specific moments, exact timestamps, names, video, audio, or detailed movement traces.",
        "example": "Example: a monthly bar graph showing that a room was occupied 35% of the time and empty 65% of the time, with no specific times or names attached.",
    },
    {
        "term": "sound_event_summary",
        "label": "Sound-event summary report",
        "definition": "A weekly chart or written summary showing broad counts or percentages for noise types, such as alarms, footsteps, loud sounds, and quiet time. The original audio recording is never saved or shared, and it does not include words from conversations.",
        "example": "Example: a weekly chart showing: alarms triggered: 2 times; footstep noise: 15% of the day; quiet hours: 80% of the day.",
    },
    {
        "term": "fused_event_record",
        "label": "Combined text log of events",
        "definition": "A short written timeline combining multiple sensor alerts into a task-specific event record. It lists only the alert text and relevant times; no audio, video, photos, or detailed sensor streams are shared.",
        "example": "Example: Safety alert: front door opened at 3:00 AM, followed immediately by motion in the hallway.",
    },
]

OUTPUT_DATA_GLOSSARY_WORDING_OVERRIDES = [
    {
        "term": "raw_image",
        "label": "Original still photo",
        "definition": "A clear, unblurred camera snapshot showing faces, objects, and background details exactly as they look.",
    },
    {
        "term": "redacted_image",
        "label": "Blurred still photo",
        "definition": "A single camera snapshot where faces, text, and sensitive personal details are permanently blurred out.",
    },
    {
        "term": "raw_video",
        "label": "Original live video feed",
        "definition": "A continuous, unblurred video stream showing everything clearly, including faces and background text. Audio is not included unless the survey explicitly says video and audio are shared together.",
    },
    {
        "term": "redacted_video",
        "label": "Blurred live video feed",
        "definition": "A continuous video stream where faces and identifying details are blurred in real time. Audio is not included unless the survey explicitly says video and audio are shared together.",
    },
    {
        "term": "pose_keypoints",
        "label": "Stick-figure movement outline",
        "definition": "A set of moving dots and lines mapping body joints, such as arms, legs, and torso, to track movement. No original photo or video is shared.",
    },
    {
        "term": "filtered_audio",
        "label": "Speech-muted audio clip",
        "definition": "An audio recording where human words are scrubbed out so conversations cannot be understood, but background noises such as alarms, footsteps, or coughing may remain.",
    },
    {
        "term": "raw_audio",
        "label": "Original audio recording",
        "definition": "The original sound file, including clear voices, spoken words, and background sounds.",
    },
    {
        "term": "occupancy",
        "label": "Presence or people count indicator",
        "definition": "A yes/no room presence indicator or a people count log. No original video, photo, or audio is shared.",
    },
    {
        "term": "event_alert",
        "label": "Event alert or text log",
        "definition": "A short written alert or timeline saying that a scenario-relevant event occurred. No original audio or video is shared unless separately stated.",
    },
]


def merged_human_readable_glossary(flow_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the embedded glossary plus server-side output-data additions.

    This avoids editing ci_focused_user_study_context.json while keeping the
    admin/debug endpoint aligned with the output terms emitted by the survey
    item generator.
    """
    raw = flow_data.get("human_readable_glossary", {}) if isinstance(flow_data, dict) else {}
    glossary = json.loads(json.dumps(raw)) if raw else {"fields": {}}
    fields = glossary.setdefault("fields", {})
    output_terms = fields.setdefault("output_data", [])
    by_term = {str(item.get("term")): item for item in output_terms if isinstance(item, dict) and item.get("term")}
    for item in OUTPUT_DATA_GLOSSARY_WORDING_OVERRIDES + EXTRA_OUTPUT_DATA_GLOSSARY + SUPPLEMENTAL_OUTPUT_DATA_GLOSSARY:
        term = str(item["term"])
        if term in by_term:
            by_term[term].update(item)
        else:
            output_terms.append(dict(item))
    notes = glossary.setdefault("revision_notes", [])
    note = "Server merges synchronized video-with-sound output-data terms without modifying the computational context scenario file."
    if note not in notes:
        notes.append(note)
    guidance = glossary.setdefault("display_guidance", {})
    guidance.setdefault(
        "audio_video_description",
        "For video-with-sound outputs, state whether faces/identifying visual details are blurred and whether spoken words are removed. Mention that non-speech sounds may remain after speech filtering.",
    )
    return glossary


@dataclass
class Config:
    flow_file: Path
    db_path: Path
    k: int
    seed: int
    assignment_mode: str
    max_per_scenario_group: int
    pipeline_output_dir: Optional[Path]
    previous_pipeline_output_dir: Optional[Path]
    include_pipeline_outputs: bool
    include_no_output_variants: bool
    survey_output_scope: str
    supplemental_output_schemas: Tuple[str, ...]
    study_contact_name: str
    study_contact_email: str
    rights_contact_name: str
    rights_contact_email: str
    rights_contact_phone: str


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


def _payload_first(payload: Dict[str, Any], *keys: str) -> str:
    """Return the first non-empty string from equivalent payload keys."""
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def extract_prolific_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize Prolific URL parameter metadata from the start payload.

    Prolific appends PROLIFIC_PID, STUDY_ID, and SESSION_ID to external survey
    links. The browser copies those into the /api/start request.  Store both
    canonical lowercase names and the raw query string/param object so exports
    can be reconciled with Prolific later.
    """
    prolific_pid = _payload_first(payload, "prolific_pid", "PROLIFIC_PID", "participant_code")
    prolific_study_id = _payload_first(payload, "prolific_study_id", "STUDY_ID", "study_id")
    prolific_session_id = _payload_first(payload, "prolific_session_id", "SESSION_ID", "submission_id")
    raw_params = payload.get("prolific_url_params")
    if not isinstance(raw_params, dict):
        raw_params = {}
    return {
        "prolific_pid": prolific_pid,
        "prolific_study_id": prolific_study_id,
        "prolific_session_id": prolific_session_id,
        "prolific_url_query": _payload_first(payload, "prolific_url_query", "url_query"),
        "prolific_url_params": raw_params,
    }


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


def is_child_related_flow(flow: Dict[str, Any]) -> bool:
    """Return True for scenarios removed from the participant-facing survey.

    Pilot feedback showed that parent/guardian and school-child scenarios
    introduce a separate perspective-taking problem. This survey version keeps
    the adult/self-perspective scenarios and excludes child/school cases.
    """
    params = scenario_ci_params(flow)
    subject = str(params.get("subject") or "").strip().lower()
    context = str(params.get("context") or "").strip().lower()
    sender = str(params.get("sender") or "").strip().lower()
    recipient = str(params.get("recipient") or "").strip().lower()
    family = str(flow.get("context_family") or "").strip().lower()
    text = json.dumps({
        "family_id": flow.get("family_id"),
        "context_bundle_id": flow.get("context_bundle_id"),
        "context_bundle_label": flow.get("context_bundle_label"),
        "participant_vignette": flow.get("participant_vignette"),
        "participant_display_fields": flow.get("participant_display_fields"),
    }, ensure_ascii=False).lower()
    return (
        subject == "child"
        or context == "school_or_classroom"
        or family == "child_or_school"
        or sender == "school_monitoring_system"
        or recipient == "teacher_or_school_official"
        or "child" in text
        or "school" in text
        or "classroom" in text
    )


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


def row_matched_cap_text(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return ""
    return str(row.get("matched_output_cap") or row.get("matched_output_schema") or row.get("final_output_schema") or "").lower()


def output_kind_from_cap(final_cap: Dict[str, Any], row: Optional[Dict[str, Any]] = None) -> str:
    """Classify newer flexible semantic outputs before generic media fallbacks."""
    t = (cap_type(final_cap) or str((row or {}).get("final_output_type") or "")).lower()
    schema = (cap_schema(final_cap) or str((row or {}).get("final_output_schema") or "")).lower()
    cap = row_matched_cap_text(row)
    joined = " ".join([t, schema, cap])
    if "object_detections" in joined or "person_detections" in joined or "application/x-detections" in joined:
        return "person_detections"
    if "room_occupied" in joined or "binary-occupancy" in joined or "binary_presence" in joined:
        return "room_occupied"
    if "occupancy_count" in joined or "person_count_timeseries" in joined:
        return "occupancy_count"
    if "aggregate_summary" in joined and ("sound" in joined or "chime" in joined):
        return "sound_event_summary"
    if "aggregate_summary" in joined and ("activity" in joined or "youhome" in joined or "adl" in joined):
        return "activity_summary"
    if "fused_event_record" in joined or "fall_adapter" in joined:
        return "fused_event_record"
    return ""


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



def _normalize_schema(value: Any) -> str:
    return str(value or "").strip()


def _parse_csv_terms(value: Any) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple, set)):
        raw: List[str] = []
        for item in value:
            raw.extend(str(item or "").split(","))
    else:
        raw = str(value or "").split(",")
    return tuple(sorted({x.strip() for x in raw if x and x.strip()}))


def _row_output_schema(row: Dict[str, Any], variant: Optional[Dict[str, Any]] = None) -> str:
    schema = _normalize_schema(row.get("final_output_schema") or row.get("matched_output_schema"))
    if schema:
        return schema
    if variant:
        cap = variant.get("final_output_cap") or {}
        schema = _normalize_schema(cap.get("schema") if isinstance(cap, dict) else None)
        if schema:
            return schema
        schema = _normalize_schema(variant.get("matched_output_schema"))
        if schema:
            return schema
    return ""


def selected_output_schemas(rows: Iterable[Dict[str, Any]]) -> Set[str]:
    schemas: Set[str] = set()
    for row in rows:
        if not row_represents_shared_output(row):
            continue
        schema = _row_output_schema(row)
        if schema:
            schemas.add(schema)
    return schemas


def flatten_summary_by_context_doc(data: Any) -> List[Dict[str, Any]]:
    """Flatten a summary_by_context-style JSON document into pipeline rows.

    This lets --pipeline-output-dir and --previous-pipeline-output-dir point
    either to a run directory containing index.json/summary.json or directly to a
    summary_by_context JSON file used for supplemental-survey planning.
    """
    rows: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                rows.append(dict(row))
        return rows
    if not isinstance(data, dict):
        return rows
    # summary_by_context.json shape: {S001: {method_id: row, ...}, ...}
    for sid, methods in data.items():
        if not isinstance(methods, dict):
            continue
        for method_id, row in methods.items():
            if not isinstance(row, dict):
                continue
            r = dict(row)
            r.setdefault("scenario_id", str(sid))
            r.setdefault("method_id", str(method_id))
            r.setdefault("method_kind", "ablation" if str(method_id).startswith("ablation:") else "baseline")
            r.setdefault("baseline", r.get("method_id"))
            rows.append(r)
    return rows


def compute_target_output_schemas(
    current_rows: Iterable[Dict[str, Any]],
    previous_rows: Iterable[Dict[str, Any]],
    supplemental_output_schemas: Iterable[str] = (),
) -> Set[str]:
    explicit = {_normalize_schema(x) for x in supplemental_output_schemas if _normalize_schema(x)}
    if explicit:
        return explicit
    current = selected_output_schemas(current_rows)
    previous = selected_output_schemas(previous_rows)
    allowed = set(SUPPLEMENTAL_FLEXIBLE_OUTPUT_SCHEMAS)
    if previous:
        # Compare run schemas, but only keep schemas whose participant-facing data
        # type is actually new for the supplemental survey.  This prevents internal
        # schema renames such as redacted_video_stream from reintroducing old
        # concepts like blurred live video feed.
        diff = {s for s in current - previous if s and s in allowed}
        if diff:
            return diff
    # Fallback for convenience when the old run is unavailable or the diff is empty.
    # This keeps a supplemental survey from accidentally including every current
    # output schema just because the earlier run path was not provided.
    return allowed


def load_pipeline_rows(pipeline_dir: Optional[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not pipeline_dir:
        return [], {"status": "disabled"}
    pipeline_dir = pipeline_dir.resolve() if pipeline_dir.exists() else pipeline_dir
    if not pipeline_dir.exists():
        return [], {"status": "missing", "pipeline_output_dir": str(pipeline_dir)}

    rows: List[Dict[str, Any]] = []
    if pipeline_dir.is_file():
        data = load_json(pipeline_dir)
        rows = flatten_summary_by_context_doc(data)
        for r in rows:
            r.setdefault("method_id", r.get("baseline"))
            r.setdefault("method_kind", "ablation" if str(r.get("method_id") or "").startswith("ablation:") else "baseline")
            r.setdefault("method_label", r.get("method_id"))
        baseline_methods = sorted({str(r.get("method_id")) for r in rows if r.get("method_kind") == "baseline" and r.get("method_id")})
        ablation_methods = sorted({str(r.get("method_id")) for r in rows if r.get("method_kind") == "ablation" and r.get("method_id")})
        return rows, {
            "status": "ok" if rows else "empty",
            "pipeline_output_dir": str(pipeline_dir),
            "source": str(pipeline_dir),
            "method_pipeline_row_count": len(rows),
            "baseline_pipeline_row_count": len([r for r in rows if r.get("method_kind") == "baseline"]),
            "ablation_pipeline_row_count": len([r for r in rows if r.get("method_kind") == "ablation"]),
            "method_ids": sorted({str(r.get("method_id")) for r in rows if r.get("method_id")}),
            "baseline_method_ids": baseline_methods,
            "ablation_method_ids": ablation_methods,
        }

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



def row_operator_ids(row: Optional[Dict[str, Any]] = None, candidate: Optional[Dict[str, Any]] = None) -> Set[str]:
    """Return operator IDs from either a full candidate or a flattened summary row."""
    ops: Set[str] = set()

    def add_op(value: Any) -> None:
        if isinstance(value, dict):
            op = value.get("operator") or value.get("id") or value.get("op")
            if op:
                ops.add(str(op))
        elif isinstance(value, str):
            for part in value.split("->"):
                part = part.strip()
                if part:
                    ops.add(part)

    if candidate:
        for op in candidate.get("operators") or candidate.get("pipeline") or []:
            add_op(op)
    if row:
        add_op(row.get("operators"))
    return ops


def av_component_flags(final_cap: Dict[str, Any], row: Optional[Dict[str, Any]] = None, candidate: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """Detect whether an audio-video sample includes redaction or speech filtering.

    The generated summary can be flattened, so component-level properties may be
    missing even when the operator path makes the transformation clear.  This
    function therefore combines final-cap properties, matched cap names, schemas,
    and the operator chain.  It intentionally does not modify the computational
    scenario JSON.
    """
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
        prop_values = " ".join(flatten_strings(props)).lower()
        media_schema = f"{media} {schema} {prop_values}".lower()

        is_visual = (
            media.startswith(("image/", "video/"))
            or any(x in schema.lower() for x in ["image", "video", "frame"])
            or "visual" in media_schema
        )
        is_audio = (
            media.startswith("audio/")
            or "audio" in schema.lower()
            or "waveform" in schema.lower()
            or "sound" in media_schema
        )

        if is_visual:
            flags["has_visual"] = True
            if (
                props.get("redacted")
                or props.get("visual_redacted")
                or media in {"image/x-redacted", "video/x-redacted"}
                or "redacted" in schema.lower()
            ):
                flags["redacted_visual"] = True
            else:
                flags["raw_visual"] = True
        if is_audio:
            flags["has_audio"] = True
            if (
                props.get("speech_content_removed")
                or media == "audio/x-filtered"
                or "speech_removed" in schema.lower()
                or "speech_content_removed" in media_schema
            ):
                flags["speech_removed_audio"] = True
            else:
                flags["raw_audio"] = True
        for component in props.get("components") or []:
            visit(component)

    visit(final_cap)

    row = row or {}
    t = cap_type(final_cap) or str(row.get("final_output_type") or "")
    schema = cap_schema(final_cap) or str(row.get("final_output_schema") or "")
    matched = str(row.get("matched_output_cap") or "")
    ops = row_operator_ids(row, candidate)
    joined = " ".join([t, schema, matched, " ".join(sorted(ops))]).lower()

    # YouHome AV samples are synchronized video-with-sound containers even if the
    # flattened summary row does not expose nested image/audio component caps.
    if t == "application/x-youhome-av-sample" or "youhome_av" in schema.lower() or "av_sample" in schema.lower():
        flags["has_visual"] = True
        flags["has_audio"] = True

    if "op.av_visual_redact" in ops or "redacted_visual" in joined or "visual_redacted" in joined:
        flags["redacted_visual"] = True
    if "op.av_audio_speech_filter" in ops or "speech_filtered" in joined or "speech_removed" in joined or "speech_content_removed" in joined:
        flags["speech_removed_audio"] = True

    if flags["has_visual"] and not flags["redacted_visual"]:
        flags["raw_visual"] = True
    if flags["has_audio"] and not flags["speech_removed_audio"]:
        flags["raw_audio"] = True
    if flags["redacted_visual"]:
        flags["raw_visual"] = False
    if flags["speech_removed_audio"]:
        flags["raw_audio"] = False
    return flags


def is_audio_video_sample(final_cap: Dict[str, Any], row: Dict[str, Any]) -> bool:
    t = cap_type(final_cap) or str(row.get("final_output_type") or "")
    schema = cap_schema(final_cap) or str(row.get("final_output_schema") or "")
    flags = av_component_flags(final_cap, row)
    return (
        t == "application/x-youhome-av-sample"
        or "youhome_av" in schema
        or "av_sample" in schema
        or (flags["has_visual"] and flags["has_audio"])
    )



def audio_video_output_label(final_cap: Dict[str, Any], row: Optional[Dict[str, Any]] = None) -> str:
    """Participant-facing sentence for video-with-audio outputs."""
    flags = av_component_flags(final_cap, row)
    if flags["redacted_visual"] and flags["speech_removed_audio"]:
        return "The shared data is a short synchronized video and audio clip. Faces, text, and sensitive personal details are blurred, and human words are scrubbed out so conversations cannot be understood."
    if flags["redacted_visual"]:
        return "The shared data is a short synchronized video and audio clip. Faces, text, and sensitive personal details are blurred in the video."
    if flags["speech_removed_audio"]:
        return "The shared data is a short synchronized video and speech-muted audio clip. Human words are scrubbed out so conversations cannot be understood, but the video is not blurred."
    return "The shared data is a short synchronized video and audio clip taken from the room at the same time."


def audio_video_output_description(final_cap: Dict[str, Any], row: Optional[Dict[str, Any]] = None) -> str:
    """Lay explanation for video-with-audio outputs that avoids repeating the main value."""
    flags = av_component_flags(final_cap, row)
    parts = []
    if flags["redacted_visual"]:
        parts.append("Faces, tattoos, and readable personal text are blurred out. Clothing, posture, movement, and room layout may still be visible.")
    elif flags["raw_visual"]:
        parts.append("The video may show people and the surrounding scene without visual blurring.")
    if flags["speech_removed_audio"]:
        parts.append("Speech-like parts of the audio are silenced, so words should not be understandable. Other sounds may still be heard.")
    elif flags["raw_audio"]:
        parts.append("The sound may include speech or conversation if people are talking.")
    return " ".join(parts)


def human_output_label(final_cap: Dict[str, Any], info_types: Dict[str, List[str]], row: Dict[str, Any]) -> str:
    """Plain-language sentence describing the shared output for participants."""
    t = cap_type(final_cap) or str(row.get("final_output_type") or "")
    schema = cap_schema(final_cap) or str(row.get("final_output_schema") or "")
    props = final_cap.get("properties") if isinstance(final_cap.get("properties"), dict) else {}
    redacted = bool(props.get("redacted")) or t in {"image/x-redacted", "video/x-redacted"} or "redacted" in schema
    speech_removed = bool(props.get("speech_content_removed")) or t == "audio/x-filtered" or "speech_removed" in schema
    fov_minimized = bool(props.get("field_of_view_minimized"))

    if t == "application/x-pose-keypoints" or "pose" in schema:
        return "The shared data is a stick-figure movement outline: moving dots and lines mapping body joints such as arms, legs, and torso. No original photo or video is shared."
    kind = output_kind_from_cap(final_cap, row)
    if kind == "person_detections":
        return "The shared data is location boxes: a text list or grid map showing where detected people are located in the camera view. No original photos or videos are shared."
    if kind == "room_occupied":
        return "The shared data is a yes/no room presence indicator: a simple text status showing either \"Occupied\" or \"Empty.\" No original video, photo, or audio is shared."
    if kind == "occupancy_count":
        return "The shared data is a people count log: a number showing how many people are in the room over time, such as \"3 people present.\" No original video, photo, or audio is shared."
    if kind == "activity_summary":
        return "The shared data is an activity summary report: a daily, weekly, or monthly chart showing broad patterns such as occupied versus empty time. It does not show specific moments, exact timestamps, names, video, audio, or detailed movement traces."
    if kind == "sound_event_summary":
        return "The shared data is a sound-event summary report: a weekly chart or written summary showing broad counts or percentages for noise types such as alarms, footsteps, loud sounds, and quiet time. The original audio recording is never saved or shared, and it does not include words from conversations."
    if kind == "fused_event_record":
        return "The shared data is a combined text log of events: a short written timeline combining multiple sensor alerts into a task-specific event record. It lists only the alert text and relevant times; no audio, video, photos, or detailed sensor streams are shared."
    if is_audio_video_sample(final_cap, row):
        return audio_video_output_label(final_cap, row)
    if t.startswith("image/"):
        if redacted:
            return "The shared data is a blurred still photo: a single camera snapshot where faces, text, and sensitive personal details are permanently blurred out."
        if fov_minimized:
            return "The shared data is a cropped still photo showing only the relevant area."
        return "The shared data is an original still photo: a clear, unblurred camera snapshot showing faces, objects, and background details exactly as they look."
    if t.startswith("video/"):
        if redacted:
            return "The shared data is a blurred live video feed: a continuous video stream where faces and identifying details are blurred in real time."
        if fov_minimized:
            return "The shared data is a cropped live video feed showing only the relevant area."
        return "The shared data is an original live video feed: a continuous, unblurred video stream showing everything clearly, including faces and background text."
    if speech_removed:
        return "The shared data is a speech-muted audio clip: an audio recording where human words are scrubbed out so conversations cannot be understood, but background noises such as alarms, footsteps, or coughing may remain."
    if t.startswith("audio/"):
        return "The shared data is an original audio recording: the original sound file, including clear voices, spoken words, and background sounds."
    if "occupancy" in t or "occupancy" in schema or "room_occupied" in schema:
        return "The shared data is a presence or occupancy estimate, such as whether someone is present or how many people are there."
    if "decibel" in t or "decibel" in schema:
        return "The shared data is a sound-level measurement, not the original audio."
    if "sound" in t or "sound" in schema or "noise_event" in schema:
        return "The shared data is a sound-event label, such as alarm, glass breaking, footsteps, or another noise category."
    if "activity" in t or "activity" in schema:
        return "The shared data is an activity label, such as walking, cooking, sitting, or lying down."
    if "event" in t or "event" in schema:
        return "The shared data is an event alert or event label for the scenario task."
    fallback = first_present(row.get("matched_output_cap"), schema, t, "data from the smart-space monitoring device") or "data from the smart-space monitoring device"
    return f"The shared data is {fallback}."


def output_description(final_cap: Dict[str, Any], row: Dict[str, Any]) -> str:
    """Short lay explanation shown below the output sentence."""
    t = cap_type(final_cap) or str(row.get("final_output_type") or "")
    schema = cap_schema(final_cap) or str(row.get("final_output_schema") or "")
    props = final_cap.get("properties") if isinstance(final_cap.get("properties"), dict) else {}
    if t == "application/x-pose-keypoints" or "pose" in schema:
        return ""
    kind = output_kind_from_cap(final_cap, row)
    if kind == "person_detections":
        return "Only the derived location boxes are shared; the original camera photo or video is not shared."
    if kind == "room_occupied":
        return "Only the occupied/empty status is shared; the original video, photo, audio, and detailed sensor stream are not shared."
    if kind == "occupancy_count":
        return "Only the count log is shared; the original video, photo, audio, and detailed sensor stream are not shared."
    if kind == "activity_summary":
        return "The report uses broad percentages or counts rather than a live minute-by-minute tracker. It does not include names, exact personal routines, original video, audio, images, or the detailed movement trace that produced the summary."
    if kind == "sound_event_summary":
        return "The report uses broad weekly counts or percentages rather than a live transcript or minute-by-minute audio log. The original recording and spoken words are not shared."
    if kind == "fused_event_record":
        return "Only the short written event timeline is shared. It does not include the original audio, video, photos, conversations, or detailed sensor streams."
    if is_audio_video_sample(final_cap, row):
        return audio_video_output_description(final_cap, row)
    if t.startswith(("image/", "video/")) and (props.get("redacted") or t in {"image/x-redacted", "video/x-redacted"} or "redacted" in schema):
        if t.startswith("image/"):
            return "This is a still image, not a live video feed or audio recording. Clothing, posture, objects, and room layout may still be visible after blurring."
        if t.startswith("video/"):
            return "This is video only; audio is not included. Clothing, posture, movement, objects, and room layout may still be visible after blurring."
        return "Faces and other identifying visual details are blurred or obscured."
    if t.startswith(("image/", "video/")) and props.get("field_of_view_minimized"):
        if t.startswith("image/"):
            return "These are separate camera images cropped to the relevant area, rather than the full camera view."
        if t.startswith("video/"):
            return "This is video only; audio is not included. The camera view is cropped to the relevant area."
        return "Only a cropped part of the camera view is shared, rather than the full scene."
    if t.startswith("image/"):
        return "This is a still image, not a live video feed or audio recording. It may show faces, bodies, clothing, activities, objects, and the surrounding room or scene."
    if t.startswith("video/"):
        return "This is video only; audio is not included. It may show faces, bodies, clothing, activities, movement, and the surrounding room or scene."
    if t == "audio/x-filtered" or props.get("speech_content_removed") or "speech_removed" in schema:
        return "Human words are muted or scrubbed out. Other background sounds, such as alarms, footsteps, coughing, or glass breaking, may still be heard."
    if t.startswith("audio/"):
        return "This is the original microphone recording. It may include speech, conversation, and other sounds in the area."
    if "decibel" in t or "decibel" in schema:
        return "This means the named receiver gets a sound-level number, such as a decibel value, rather than the original audio."
    if "sound" in t or "sound" in schema:
        return "This means the named receiver gets a label describing the type of sound, rather than the original audio."
    if "occupancy" in t or "occupancy" in schema or "room_occupied" in schema:
        return "This means the named receiver gets a presence or count estimate, rather than the original audio, video, or sensor data."
    if "activity" in t or "activity" in schema:
        return "This means the named receiver gets a category describing an activity, rather than the original audio, video, or sensor data."
    if "event" in t or "event" in schema:
        return "This means the named receiver gets a short alert or label saying that a relevant event was detected."
    return "This is the data or output that would be sent out of the smart-space monitoring device for the stated purpose."

def privacy_class_from_output(final_cap: Dict[str, Any], row: Dict[str, Any]) -> str:
    t = cap_type(final_cap) or str(row.get("final_output_type") or "")
    schema = cap_schema(final_cap) or str(row.get("final_output_schema") or "")
    props = final_cap.get("properties") if isinstance(final_cap.get("properties"), dict) else {}
    kind = output_kind_from_cap(final_cap, row)
    if kind:
        return kind
    if is_audio_video_sample(final_cap, row):
        flags = av_component_flags(final_cap, row)
        if flags["redacted_visual"] and flags["speech_removed_audio"]:
            return "redacted_filtered_audio_video"
        if flags["redacted_visual"]:
            return "redacted_audio_video"
        if flags["speech_removed_audio"]:
            return "speech_filtered_audio_video"
        return "raw_audio_video"
    if t.startswith(("image/", "video/")) and not (props.get("redacted") or t in {"image/x-redacted", "video/x-redacted"} or "redacted" in schema):
        return "raw_media"
    if t.startswith(("image/", "video/")) and (props.get("redacted") or t in {"image/x-redacted", "video/x-redacted"} or "redacted" in schema):
        return "redacted_media"
    if t == "audio/x-filtered" or props.get("speech_content_removed") or "speech_removed" in schema:
        return "filtered_audio"
    if t.startswith("audio/"):
        return "raw_audio"
    if "pose" in t or "pose" in schema:
        return "derived_pose"
    if t.startswith("application/"):
        return "derived_or_semantic_output"
    return "output_data"


def output_data_term_from_output(final_cap: Dict[str, Any], row: Dict[str, Any]) -> str:
    """Return a stable readable-glossary term for the shared output."""
    t = cap_type(final_cap) or str(row.get("final_output_type") or "")
    schema = cap_schema(final_cap) or str(row.get("final_output_schema") or "")
    props = final_cap.get("properties") if isinstance(final_cap.get("properties"), dict) else {}
    kind = output_kind_from_cap(final_cap, row)
    if kind:
        return kind
    if is_audio_video_sample(final_cap, row):
        flags = av_component_flags(final_cap, row)
        if flags["redacted_visual"] and flags["speech_removed_audio"]:
            return "redacted_filtered_audio_video"
        if flags["redacted_visual"]:
            return "redacted_audio_video"
        if flags["speech_removed_audio"]:
            return "speech_filtered_audio_video"
        return "raw_audio_video"
    if t == "application/x-pose-keypoints" or "pose" in schema:
        return "pose_keypoints"
    if t.startswith("image/"):
        return "redacted_image" if (props.get("redacted") or t == "image/x-redacted" or "redacted" in schema) else "raw_image"
    if t.startswith("video/"):
        return "redacted_video" if (props.get("redacted") or t == "video/x-redacted" or "redacted" in schema) else "raw_video"
    if t == "audio/x-filtered" or props.get("speech_content_removed") or "speech_removed" in schema:
        return "filtered_audio"
    if t.startswith("audio/"):
        return "raw_audio"
    if "occupancy" in t or "occupancy" in schema or "room_occupied" in schema:
        return "occupancy"
    if "sound" in t or "sound" in schema or "noise_event" in schema:
        return "sound_label"
    if "event" in t or "event" in schema or "activity" in t or "activity" in schema:
        return "event_alert"
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
        "output_data_term": output_data_term_from_output(final_cap, row),
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
    survey_output_scope: str = "all",
    target_output_schemas: Optional[Set[str]] = None,
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
    rows_excluded_by_output_schema = 0
    target_output_schemas = set(target_output_schemas or set())
    survey_output_scope = str(survey_output_scope or "all")

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

        if survey_output_scope == "new_flexible_only":
            row_schema = _row_output_schema(row, variant)
            if row_schema not in target_output_schemas:
                rows_excluded_by_output_schema += 1
                continue

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
                    "output_data_term": variant.get("output_data_term"),
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
        "method_pipeline_rows_excluded_by_output_schema": rows_excluded_by_output_schema,
        "survey_output_scope": survey_output_scope,
        "target_output_schemas": sorted(target_output_schemas),
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
        raw_flows = self.flow_data.get("generated_information_flows") or self.flow_data.get("context_scenarios", [])
        self.excluded_child_related_flows = [f for f in raw_flows if is_child_related_flow(f)]
        self.flows = [f for f in raw_flows if not is_child_related_flow(f)]
        if not self.flows:
            raise ValueError(f"No non-child generated_information_flows or context_scenarios found in {config.flow_file}")

        self.pipeline_rows: List[Dict[str, Any]] = []
        self.pipeline_load_info: Dict[str, Any] = {"status": "disabled"}
        if config.include_pipeline_outputs and config.pipeline_output_dir:
            self.pipeline_rows, self.pipeline_load_info = load_pipeline_rows(config.pipeline_output_dir)

        self.previous_pipeline_rows: List[Dict[str, Any]] = []
        self.previous_pipeline_load_info: Dict[str, Any] = {"status": "disabled"}
        if (
            config.include_pipeline_outputs
            and config.survey_output_scope == "new_flexible_only"
            and config.previous_pipeline_output_dir
        ):
            self.previous_pipeline_rows, self.previous_pipeline_load_info = load_pipeline_rows(config.previous_pipeline_output_dir)

        self.target_output_schemas = compute_target_output_schemas(
            self.pipeline_rows,
            self.previous_pipeline_rows,
            config.supplemental_output_schemas,
        ) if config.survey_output_scope == "new_flexible_only" else set()

        self.items, self.item_pool_summary = build_survey_items(
            self.flows,
            self.pipeline_rows,
            config.pipeline_output_dir,
            include_pipeline_outputs=config.include_pipeline_outputs,
            include_no_output_variants=config.include_no_output_variants,
            survey_output_scope=config.survey_output_scope,
            target_output_schemas=self.target_output_schemas,
        )
        self.item_pool_summary["previous_pipeline_load_info"] = self.previous_pipeline_load_info
        self.item_pool_summary["excluded_child_related_context_count"] = len(self.excluded_child_related_flows)
        self.item_pool_summary["excluded_child_related_context_ids"] = [get_scenario_id(f) for f in self.excluded_child_related_flows]
        if not self.items:
            raise ValueError(
                "Survey item pool is empty after child-related scenarios are excluded. Check --pipeline-output-dir, or run with --no-pipeline-outputs "
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
        ensure_column(conn, "sessions", "completed_at_ms", "INTEGER")
        ensure_column(conn, "sessions", "last_activity_at_ms", "INTEGER")
        ensure_column(conn, "sessions", "total_elapsed_ms", "INTEGER")
        ensure_column(conn, "sessions", "total_active_elapsed_ms", "INTEGER")
        ensure_column(conn, "sessions", "answered_count", "INTEGER")
        ensure_column(conn, "sessions", "prolific_pid", "TEXT")
        ensure_column(conn, "sessions", "prolific_study_id", "TEXT")
        ensure_column(conn, "sessions", "prolific_session_id", "TEXT")
        ensure_column(conn, "sessions", "prolific_url_params_json", "TEXT")
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
        ensure_column(conn, "responses", "prolific_pid", "TEXT")
        ensure_column(conn, "responses", "prolific_study_id", "TEXT")
        ensure_column(conn, "responses", "prolific_session_id", "TEXT")
        ensure_column(conn, "responses", "prolific_url_params_json", "TEXT")
        ensure_column(conn, "responses", "attention_check_field", "TEXT")
        ensure_column(conn, "responses", "attention_check_prompt", "TEXT")
        ensure_column(conn, "responses", "attention_check_expected", "TEXT")
        ensure_column(conn, "responses", "attention_check_answer", "TEXT")
        ensure_column(conn, "responses", "attention_check_correct", "INTEGER")
        ensure_column(conn, "responses", "comprehension_check_field", "TEXT")
        ensure_column(conn, "responses", "comprehension_check_prompt", "TEXT")
        ensure_column(conn, "responses", "comprehension_check_expected", "TEXT")
        ensure_column(conn, "responses", "comprehension_check_answer", "TEXT")
        ensure_column(conn, "responses", "comprehension_check_correct", "INTEGER")
        conn.commit()


def create_session(db_path: Path, session_id: str, participant_id: str, assignment: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    created = now_ms()
    prolific = extract_prolific_metadata(metadata)
    # Preserve canonical Prolific fields inside metadata_json as well as explicit
    # database columns, so existing exports and direct SQL inspection both work.
    metadata = dict(metadata)
    metadata.update(prolific)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sessions(
                session_id, participant_id, created_at_ms, assignment_json, metadata_json,
                last_activity_at_ms, total_elapsed_ms, total_active_elapsed_ms, answered_count,
                prolific_pid, prolific_study_id, prolific_session_id, prolific_url_params_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, participant_id, created, json.dumps(assignment), json.dumps(metadata),
                created, 0, 0, 0,
                prolific.get("prolific_pid"), prolific.get("prolific_study_id"),
                prolific.get("prolific_session_id"), json.dumps(prolific.get("prolific_url_params") or {}, sort_keys=True),
            ),
        )
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
        row = conn.execute(
            """
            SELECT rating, confidence, free_text, elapsed_ms,
                   attention_check_answer, attention_check_correct,
                   comprehension_check_answer, comprehension_check_correct
            FROM responses WHERE session_id=? AND item_index=?
            """,
            (session_id, index),
        ).fetchone()
        return dict(row) if row else None


def get_responses_for_session(db_path: Path, session_id: str) -> List[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM responses WHERE session_id=? ORDER BY item_index", (session_id,)).fetchall()]


def normalize_attention_answer(value: Any) -> str:
    """Normalize attention-check strings for exact-but-forgiving comparison."""
    text = " ".join(str(value or "").strip().lower().split())
    return text.rstrip(".")


def attention_check_is_correct(expected: Any, answer: Any) -> bool:
    return normalize_attention_answer(expected) == normalize_attention_answer(answer)


def save_response(db_path: Path, session_id: str, participant_id: str, index: int, item: Dict[str, Any], rating: int, confidence: Any, free_text: str, elapsed_ms: Any, raw_payload: Dict[str, Any]) -> None:
    flow = item.get("flow", {})
    output = flow.get("output_data_slot", {}) or {}
    final_cap = output.get("final_output_cap") or {}
    attention = item.get("attention_check") or {}
    attention_answer = raw_payload.get("attention_check_answer")
    attention_expected = attention.get("expected_value")
    attention_correct = None
    if attention:
        attention_correct = 1 if attention_check_is_correct(attention_expected, attention_answer) else 0

    comprehension = item.get("comprehension_check") or {}
    comprehension_answer = raw_payload.get("comprehension_check_answer")
    comprehension_expected = comprehension.get("expected_value")
    comprehension_correct = None
    if comprehension:
        comprehension_correct = 1 if attention_check_is_correct(comprehension_expected, comprehension_answer) else 0
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
        conn.row_factory = sqlite3.Row
        sess_row = conn.execute(
            "SELECT prolific_pid, prolific_study_id, prolific_session_id, prolific_url_params_json FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        prolific_pid = sess_row["prolific_pid"] if sess_row else None
        prolific_study_id = sess_row["prolific_study_id"] if sess_row else None
        prolific_session_id = sess_row["prolific_session_id"] if sess_row else None
        prolific_url_params_json = sess_row["prolific_url_params_json"] if sess_row else None
        conn.execute("""
            INSERT OR REPLACE INTO responses(
                session_id, participant_id, item_index, flow_id, method_id, task,
                context_family, output_variant_id, rating, confidence, free_text,
                elapsed_ms, created_at_ms, item_json, raw_payload_json,
                item_id, baseline_ids_json, baseline_count, baseline_details_json,
                method_ids_json, method_count, method_details_json,
                ablation_modes_json, baseline_method_ids_json,
                output_variant_label, output_variant_description, variant_privacy_class, final_output_type,
                final_output_schema, information_types_json,
                prolific_pid, prolific_study_id, prolific_session_id, prolific_url_params_json,
                attention_check_field, attention_check_prompt, attention_check_expected,
                attention_check_answer, attention_check_correct,
                comprehension_check_field, comprehension_check_prompt, comprehension_check_expected,
                comprehension_check_answer, comprehension_check_correct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            prolific_pid,
            prolific_study_id,
            prolific_session_id,
            prolific_url_params_json,
            attention.get("field_label"),
            attention.get("question"),
            attention_expected,
            attention_answer,
            attention_correct,
            comprehension.get("field_label"),
            comprehension.get("question"),
            comprehension_expected,
            comprehension_answer,
            comprehension_correct,
        ))
        conn.commit()


def update_session_progress(db_path: Path, session_id: str, total_assigned: int) -> Dict[str, Any]:
    """Update and return session-level timing/progress summary."""
    now = now_ms()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sess = conn.execute("SELECT created_at_ms FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if not sess:
            return {}
        row = conn.execute(
            """
            SELECT COUNT(*) AS answered_count,
                   COALESCE(SUM(COALESCE(elapsed_ms, 0)), 0) AS total_active_elapsed_ms
            FROM responses WHERE session_id=?
            """,
            (session_id,),
        ).fetchone()
        answered = int(row["answered_count"] or 0)
        total_active = int(row["total_active_elapsed_ms"] or 0)
        total_elapsed = max(0, now - int(sess["created_at_ms"]))
        completed_at = now if answered >= total_assigned else None
        conn.execute(
            """
            UPDATE sessions
            SET last_activity_at_ms=?, total_elapsed_ms=?, total_active_elapsed_ms=?,
                answered_count=?, completed_at_ms=COALESCE(completed_at_ms, ?)
            WHERE session_id=?
            """,
            (now, total_elapsed, total_active, answered, completed_at, session_id),
        )
        conn.commit()
    return {
        "answered_count": answered,
        "total_assigned": total_assigned,
        "completed": answered >= total_assigned,
        "total_elapsed_ms": total_elapsed,
        "total_active_elapsed_ms": total_active,
        "completed_at_ms": completed_at,
    }




def session_is_complete(db_path: Path, session_id: str) -> Tuple[bool, Optional[str]]:
    """Return whether the participant has answered every assigned item."""
    session = get_session(db_path, session_id)
    if not session:
        return False, None
    assigned_count = len(session.get("assignment") or [])
    answered_count = len(get_responses_for_session(db_path, session_id))
    return assigned_count > 0 and answered_count >= assigned_count, session.get("participant_id")


def redirect_response(handler: BaseHTTPRequestHandler, location: str, status: int = 302) -> None:
    handler.send_response(status)
    handler.send_header("Location", location)
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()

def assign_items(
    items: List[Dict[str, Any]],
    k: int,
    seed: int,
    participant_id: str,
    session_id: str,
    mode: str,
    db_path: Path,
    max_per_scenario_group: int = 2,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed + stable_int(participant_id) + stable_int(session_id))
    indexed = list(enumerate(items))
    if not indexed:
        return []

    def _assignment_record(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "item_index": idx,
            "item_id": item.get("item_id"),
            "flow_id": item.get("flow_id"),
            "output_variant_id": (item.get("output_variant") or {}).get("output_variant_id"),
        }

    def _participant_visible_signature(item: Dict[str, Any]) -> str:
        """Hash the text a participant would actually see for this item.

        Different backend pipelines can occasionally collapse to the same
        participant-facing wording after plain-language simplification. Without
        this second dedupe pass, a participant may see two word-for-word
        identical pages even though the hidden item IDs differ.
        """
        try:
            visible_fields = participant_display_fields(item.get("flow") or {}, item.get("output_variant"))
            simplified = []
            for f in visible_fields:
                simplified.append({
                    "label": f.get("label") or "",
                    "value": f.get("value") or "",
                    "description": f.get("description") or f.get("help") or "",
                    "example": f.get("example") or "",
                })
            return stable_hash_obj(simplified)
        except Exception:
            return stable_hash_obj({
                "flow_id": item.get("flow_id"),
                "output": (item.get("output_variant") or {}).get("output_variant_label"),
                "privacy": (item.get("output_variant") or {}).get("variant_privacy_class"),
            })

    def _item_scenario_group_key(item: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
        """Group cases so the scenario usually stays constant while parameters vary.

        A flow/context scenario fixes the broad setting, room, task, subject, and
        purpose. The different output variants within that flow then change the
        parameters participants are rating, such as the shared data type. Keeping
        those variants adjacent reduces role/setting whiplash for Prolific users.
        """
        flow = item.get("flow") or {}
        params = scenario_ci_params(flow)
        return (
            str(params.get("context") or ""),
            str(params.get("space") or ""),
            str(flow.get("task") or ""),
            str(params.get("subject") or ""),
            str(params.get("purpose") or ""),
            str(item.get("flow_id") or ""),
        )

    def _item_parameter_sort_key(item: Dict[str, Any]) -> Tuple[str, str, str]:
        flow = item.get("flow") or {}
        params = scenario_ci_params(flow)
        variant = item.get("output_variant") or {}
        return (
            str(variant.get("variant_privacy_class") or ""),
            str(variant.get("output_variant_label") or ""),
            str(params.get("transmission_principle") or ""),
        )

    if mode == "sequential":
        # Sequential mode still shows scenario blocks rather than a fully raw order.
        ordered = sorted(indexed, key=lambda x: (_item_scenario_group_key(x[1]), _item_parameter_sort_key(x[1])))
        offset = stable_int(session_id) % len(ordered)
        return [_assignment_record(*ordered[(offset + j) % len(ordered)]) for j in range(k)]

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

    # Build scenario blocks first, then choose output variants inside each block.
    # This changes the user experience from "25 unrelated vignettes" to "several
    # coherent scenarios where the parameters change."
    by_scenario: Dict[Tuple[str, str, str, str, str, str], List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, item in indexed:
        by_scenario.setdefault(_item_scenario_group_key(item), []).append((idx, item))

    target_n = min(k, len(items))
    scenario_blocks = list(by_scenario.items())

    def _block_score(block: Tuple[Tuple[str, str, str, str, str, str], List[Tuple[int, Dict[str, Any]]]]) -> Tuple[int, float, float]:
        _, group_items = block
        counts = [prior_counts.get(str(item.get("item_id")), 0) for _, item in group_items]
        min_count = min(counts) if counts else 0
        mean_count = sum(counts) / max(1, len(counts))
        return (min_count, mean_count, rng.random())

    scenario_blocks.sort(key=_block_score)

    chosen: List[Dict[str, Any]] = []
    group_order: Dict[str, int] = {}
    group_counts: Dict[Tuple[str, str, str, str, str, str], int] = {}
    used: Set[int] = set()
    selected_visible_keys: Set[str] = set()
    per_group_cap = int(max_per_scenario_group or 0)
    for block_idx, (scenario_key, group_items) in enumerate(scenario_blocks):
        if len(chosen) >= target_n:
            break
        shuffled = list(group_items)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda x: (prior_counts.get(str(x[1].get("item_id")), 0), _item_parameter_sort_key(x[1]), rng.random()))
        for idx, item in shuffled:
            if len(chosen) >= target_n:
                break
            if per_group_cap > 0 and group_counts.get(scenario_key, 0) >= per_group_cap:
                break
            if idx in used:
                continue
            visible_key = _participant_visible_signature(item)
            if visible_key in selected_visible_keys:
                continue
            rec = _assignment_record(idx, item)
            chosen.append(rec)
            selected_visible_keys.add(visible_key)
            used.add(idx)
            group_counts[scenario_key] = group_counts.get(scenario_key, 0) + 1
            group_order[str(item.get("flow_id") or idx)] = block_idx

    # If some unusual item pool left gaps, fill from the least-used leftovers,
    # still respecting the per-scenario cap when possible. If the cap would make
    # the survey too short, relax it only as a last resort.
    if len(chosen) < target_n:
        leftovers = [(idx, item) for idx, item in indexed if idx not in used]
        rng.shuffle(leftovers)
        leftovers.sort(key=lambda x: (prior_counts.get(str(x[1].get("item_id")), 0), _item_scenario_group_key(x[1]), _item_parameter_sort_key(x[1]), rng.random()))
        for relax_cap in [False, True]:
            for idx, item in leftovers:
                if len(chosen) >= target_n:
                    break
                if idx in used:
                    continue
                scenario_key = _item_scenario_group_key(item)
                if not relax_cap and per_group_cap > 0 and group_counts.get(scenario_key, 0) >= per_group_cap:
                    continue
                visible_key = _participant_visible_signature(item)
                if visible_key in selected_visible_keys:
                    continue
                rec = _assignment_record(idx, item)
                chosen.append(rec)
                selected_visible_keys.add(visible_key)
                group_counts[scenario_key] = group_counts.get(scenario_key, 0) + 1
                group_order.setdefault(str(item.get("flow_id") or idx), len(group_order))
                used.add(idx)
            if len(chosen) >= target_n:
                break

    def _persona_order(item: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
        flow = item.get("flow") or {}
        params = scenario_ci_params(flow)
        return (
            str(params.get("subject") or ""),
            str(params.get("context") or ""),
            str(params.get("space") or ""),
            str(flow.get("task") or ""),
            str(params.get("purpose") or ""),
        )

    def _final_order_key(a: Dict[str, Any]) -> Tuple[Tuple[str, str, str, str, str], int, Tuple[str, str, str]]:
        try:
            item = items[int(a["item_index"])]
            return (_persona_order(item), group_order.get(str(item.get("flow_id") or a["item_index"]), 10**9), _item_parameter_sort_key(item))
        except Exception:
            return (("", "", "", "", ""), 10**9, ("", "", ""))

    chosen.sort(key=_final_order_key)

    # Mark rows that actually changed from the immediately previous page.
    # This keeps the explicit “Changed” cue for repeated scenario versions, while
    # avoiding false positives on the first assigned page.
    comparable_labels = {
        "What is the scenario?",
        "What is your role in this scenario?",
        "What data would be shared?",
        "Who controls the monitoring device?",
        "Who receives or uses the shared data?",
        "What notice or permission is given?",
        "When is data collected or shared?",
        "Where is the data processed?",
        "Who is allowed to access the shared data?",
        "What audio filtering happens before sharing?",
        "What rule applies to the data?",
    }

    def _visible_field_map_for_record(rec: Dict[str, Any]) -> Dict[str, str]:
        try:
            item = items[int(rec["item_index"])]
            out: Dict[str, str] = {}
            for f in participant_display_fields(item.get("flow") or {}, item.get("output_variant")):
                label = str(f.get("label") or "")
                if label not in comparable_labels:
                    continue
                out[label] = "\n".join([
                    str(f.get("value") or ""),
                    str(f.get("description") or f.get("help") or ""),
                    str(f.get("example") or ""),
                ]).strip()
            return out
        except Exception:
            return {}

    previous_map: Optional[Dict[str, str]] = None
    for rec in chosen:
        current_map = _visible_field_map_for_record(rec)
        if previous_map is not None:
            changed_labels = sorted(
                label for label, value in current_map.items()
                if previous_map.get(label) != value
            )
            if changed_labels:
                rec["changed_field_labels"] = changed_labels
        previous_map = current_map

    return chosen[:target_n]


def _field_value_by_label(flow: Dict[str, Any], label: str) -> Optional[str]:
    for row in flow.get("participant_display_fields", []) or []:
        if row.get("label") == label:
            val = row.get("value")
            return str(val) if val is not None else None
    return None


def _task_family_phrase(flow: Optional[Dict[str, Any]]) -> str:
    task = str((flow or {}).get("task") or "").lower()
    purpose = _purpose_plain(flow).lower() if flow else ""
    if "fall" in task or "fall" in purpose:
        return "Fall detection"
    if "sound" in task or "sound" in purpose or "communication" in purpose:
        return "Sound monitoring"
    if "adl" in task or "activity" in task or "routine" in purpose:
        return "Daily activity monitoring"
    if "visitor" in task or "presence" in task or "security" in purpose:
        return "Visitor or presence monitoring"
    return str((flow or {}).get("task_label") or (flow or {}).get("task") or "Smart-space scenario").replace("_", " ").title()



def _task_label(flow: Dict[str, Any]) -> str:
    """Readable scenario title for previews/admin exports; the live UI does not show it."""
    subject = _subject_plain(flow)
    family = _task_family_phrase(flow)
    who = _indefinite_subject_phrase(subject) if subject and subject != "person" else "a person"
    location = _location_phrase(flow)
    return f"{family} for {who} in {location}"

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


def _article_for_setting(setting: str) -> str:
    setting = str(setting or "").strip()
    if not setting:
        return ""
    lower = setting.lower()
    if lower.startswith(("a ", "an ", "the ")):
        return setting
    if lower in {"public area", "private home", "short-term rental", "long-term care home", "research smart-home lab", "hospital or clinic", "school or classroom", "workplace"}:
        article = "an" if lower[0] in "aeiou" else "a"
        return f"{article} {setting}"
    return f"a {setting}"


def _location_phrase(flow: Optional[Dict[str, Any]], demonstrative: bool = False) -> str:
    """Natural phrase for the context+space pair, e.g. 'the living room of a private home'."""
    setting, space = _setting_parts(flow)
    try:
        params = scenario_ci_params(flow or {})
    except Exception:
        params = {}
    context = str(params.get("context") or "").strip()
    space_term = str(params.get("space") or "").strip()

    setting_article = _article_for_setting(setting)
    this_setting = f"this {setting}" if setting else "this setting"

    if space_term == "outdoor" and context == "public_space":
        return "this public outdoor area" if demonstrative else "a public outdoor area"
    if context == "workplace" and space_term == "workspace":
        return "the main office floor of this workplace" if demonstrative else "the main office floor of a workplace"
    if context == "research_living_lab":
        if space:
            return f"the {space} of this research smart-home lab" if demonstrative else f"the {space} of a research smart-home lab"
        return "this research smart-home lab" if demonstrative else "a research smart-home lab"
    if space_term == "outdoor" and setting:
        return f"the outdoor area of {this_setting}" if demonstrative else f"the outdoor area of {setting_article}"
    if space and setting:
        return f"the {space} of {this_setting}" if demonstrative else f"the {space} of {setting_article}"
    if setting:
        return this_setting if demonstrative else setting_article
    return "this smart-space setting" if demonstrative else "a smart-space setting"


def _situation_value_sentence(flow: Optional[Dict[str, Any]], fallback: Any = "") -> str:
    phrase = _location_phrase(flow)
    if phrase:
        return f"This happens in {phrase}."
    text = _plain_setting_value(fallback)
    return f"This happens in {text}." if text else "This happens in the setting described in the scenario."



def _subject_plain(flow: Optional[Dict[str, Any]]) -> str:
    return _row_value_by_original_label(flow, "Data subject") or "person"


def _subject_role_for_participant(subject: str, responsible: bool = False) -> str:
    """Return a natural participant-perspective subject phrase."""
    raw = str(subject or "person").strip()
    lower = raw.lower()
    if lower == "employee":
        role = "an employee"
    elif lower == "research participant":
        role = "a research participant"
    elif lower == "roommate":
        role = "a co-resident or roommate"
    elif lower in {"guest", "visitor", "patient", "resident", "child"}:
        role = f"a {lower}"
    elif lower.startswith(("a ", "an ")):
        role = raw
    elif lower.startswith("the "):
        role = "a " + raw[4:]
    else:
        role = f"a {raw}"
    if responsible and lower == "child":
        return f"{role}, or someone responsible for the child"
    return role



def _place_preposition_for_phrase(phrase: str) -> str:
    text = str(phrase or "").lower()
    if "office floor" in text or "shop floor" in text:
        return "on"
    return "in"


def _participant_subject_value(flow: Optional[Dict[str, Any]]) -> str:
    """Value for the displayed subject row, framed as a direct answer."""
    subject = _subject_plain(flow)
    role = _subject_role_for_participant(subject, responsible=False)
    setting_phrase = _location_phrase(flow, demonstrative=True)
    prep = _place_preposition_for_phrase(setting_phrase)
    if str(subject or "").strip().lower() == "child":
        return f"The data is about a child {prep} {setting_phrase}. Answer from the perspective of a parent, guardian, or responsible adult."
    return f"The data is about you as {role} {prep} {setting_phrase}."


def _surveyed_user_sentence(flow: Optional[Dict[str, Any]]) -> str:
    """Short overview sentence making the participant perspective explicit."""
    subject = _subject_plain(flow)
    role = _subject_role_for_participant(subject, responsible=False)
    setting_phrase = _location_phrase(flow, demonstrative=True)
    prep = _place_preposition_for_phrase(setting_phrase)
    if str(subject or "").strip().lower() == "child":
        return f"For this scenario, answer from the perspective of a parent, guardian, or responsible adult for a child {prep} {setting_phrase}."
    return f"For this scenario, answer as if the data is about you as {role} {prep} {setting_phrase}."

def _purpose_plain(flow: Optional[Dict[str, Any]]) -> str:
    return _row_value_by_original_label(flow, "Purpose") or "the stated purpose"


def _task_specific_app_label(flow: Optional[Dict[str, Any]]) -> str:
    purpose = _purpose_plain(flow).lower()
    task = str((flow or {}).get("task") or "").lower()
    if "fall" in purpose or "fall" in task:
        return "fall-detection app"
    if "sound" in task or "sound" in purpose or "communication" in purpose:
        return "sound-monitoring app"
    if "adl" in task or "activity" in task:
        if "personalization" in purpose:
            return "personalization app"
        return "daily-activity recognition app"
    if "visitor" in task or "presence" in task:
        if "energy" in purpose or "lighting" in purpose:
            return "energy or lighting automation service"
        return "presence-detection app"
    if "energy" in purpose or "lighting" in purpose:
        return "energy or lighting automation service"
    if "personalization" in purpose:
        return "personalization app"
    return "task-specific app"


def _responsible_recipient_phrase(flow: Optional[Dict[str, Any]]) -> str:
    """Who would normally use or see output delivered to a task app.

    The normalized recipient may be downstream_application, but participants need
    to know who the app is effectively serving. This keeps the machine-readable
    recipient unchanged while clarifying the participant-facing wording.
    """
    try:
        params = scenario_ci_params(flow or {})
    except Exception:
        params = {}
    context = str(params.get("context") or "").strip()
    sender = str(params.get("sender") or "").strip()
    subject = _subject_plain(flow)
    mapping = {
        "home": "the home resident or homeowner",
        "short_term_rental": "the rental host or property manager",
        "workplace": "the employer or authorized workplace staff",
        "public_space": "the person or organization responsible for the public area",
        "long_term_care": f"a caregiver or care staff member responsible for the {subject}",
        "hospital_or_clinic": f"a clinician or care team member responsible for the {subject}",
        "school_or_classroom": "a teacher or school official",
        "research_living_lab": "the researcher running the study",
    }
    if context == "public_space" and sender == "owner_controlled_device":
        return "nearby shop owner using the device"
    if context in mapping:
        return mapping[context]
    if sender == "host_controlled_device":
        return "the rental host or property manager"
    if sender == "owner_controlled_device":
        return "the device owner or space owner"
    if sender == "data_controller":
        return "the organization responsible for the setting"
    return "the person or organization responsible for this setting"



def _flow_processing_mode(flow: Optional[Dict[str, Any]]) -> str:
    """Return local/cloud/other for participant-facing app storage wording."""
    try:
        params = scenario_ci_params(flow or {})
    except Exception:
        params = {}
    tp = str(params.get("transmission_principle") or "").lower()
    if "cloud" in tp:
        return "cloud"
    if "local" in tp:
        return "local"
    return "other"



def _app_recipient_value(flow: Optional[Dict[str, Any]]) -> str:
    app = _task_specific_app_label(flow)
    responsible = _responsible_recipient_phrase(flow)
    if responsible.startswith(("the ", "a ", "an ")):
        who = responsible[0].upper() + responsible[1:]
    else:
        who = f"The {responsible}"
    who_lower = who.lower()

    # Keep this row about people/organizations, not backend storage. The processing
    # row separately explains whether data stays local or goes to a cloud server.
    if "home resident" in who_lower or "homeowner" in who_lower:
        return f"{who} receives the data through the {app} on their personal phone or home device."

    if _flow_processing_mode(flow) == "cloud":
        return f"{who} receives the data through the {app}."

    return f"{who} receives the data through the {app}. The app creator does not receive or store this data."


def _event_phrase(flow: Optional[Dict[str, Any]]) -> str:
    purpose = _purpose_plain(flow).lower()
    task = str((flow or {}).get("task") or "").lower()
    if "fall" in purpose or "fall" in task:
        return "a possible fall"
    if "sound" in task or "communication" in purpose:
        return "a relevant sound event, such as an alarm, glass breaking, footsteps, a loud noise, or another non-speech sound pattern"
    if "activity" in task or "adl" in task or "routine" in purpose:
        return "a daily-activity event, such as cooking, sitting, walking, or resting"
    if "visitor" in task or "security" in purpose:
        return "a visitor, presence, or intrusion event"
    if "energy" in purpose or "lighting" in purpose:
        return "an occupancy or activity event"
    if "safety" in purpose:
        return "a safety-relevant event, such as a possible fall, alarm, or dangerous sound"
    return "a relevant event for the stated purpose"


def _purpose_for_recipient_phrase(flow: Optional[Dict[str, Any]]) -> str:
    """Short purpose phrase for the receiver row."""
    purpose = _purpose_plain(flow).lower()
    task = str((flow or {}).get("task") or "").lower()
    if "fall" in purpose or "fall" in task:
        return "fall detection or safety response"
    if "work" in purpose or "employee" in purpose:
        return "employee activity or presence monitoring"
    if "clinical" in purpose:
        return "clinical care"
    if "research" in purpose:
        return "the research study"
    if "personalization" in purpose:
        return "home personalization"
    if "energy" in purpose or "lighting" in purpose:
        return "energy or lighting automation"
    if "security" in purpose or "visitor" in purpose:
        return "visitor or security monitoring"
    if "sound" in task or "sound" in purpose or "communication" in purpose:
        return "sound monitoring"
    if "activity" in task or "adl" in task or "routine" in purpose:
        return "daily activity monitoring"
    if "safety" in purpose:
        return "safety monitoring"
    if "training" in purpose or "supervision" in purpose:
        return "training or supervision"
    return "the stated purpose"


def _area_for_recipient_phrase(flow: Optional[Dict[str, Any]]) -> str:
    try:
        params = scenario_ci_params(flow or {})
    except Exception:
        params = {}
    context = str(params.get("context") or "").lower()
    if context == "short_term_rental":
        return "at the rental"
    if context == "workplace":
        return "in the workplace"
    if context == "hospital_or_clinic":
        return "in the hospital or clinic"
    if context == "long_term_care":
        return "in the care home"
    if context == "research_living_lab":
        return "in the study"
    if context == "public_space":
        return "in the public area"
    if context == "home":
        return "in the home"
    return "in this setting"


def _uses_app_recipient(flow: Optional[Dict[str, Any]]) -> bool:
    recipient = (_row_value_by_original_label(flow, "Recipient") or "").lower()
    return "app" in recipient or "software" in recipient or recipient == "the smart-space app" or recipient == "task-specific app"


def _sensor_device_description(flow: Optional[Dict[str, Any]], output_variant: Optional[Dict[str, Any]] = None) -> str:
    """Participant-facing device/modality description.

    The scenario JSON says who controls the sender, but not always whether the
    sensing device is a camera, microphone, or other sensor. Infer this from the
    generated output variant when possible and fall back to the task type.
    """
    label = str((output_variant or {}).get("output_variant_label") or "").lower()
    desc = str((output_variant or {}).get("output_variant_description") or "").lower()
    final_cap = (output_variant or {}).get("final_output_cap") or {}
    media = str(final_cap.get("media_type") or final_cap.get("semantic_type") or "").lower()
    schema = str(final_cap.get("schema") or "").lower()
    combined = " ".join([label, desc, media, schema])
    flags = av_component_flags(final_cap) if isinstance(final_cap, dict) else {"has_visual": False, "has_audio": False}

    # Text such as "camera images, not video or audio" contains the word
    # "audio" only to say audio is absent. Check camera-only phrases before
    # component flags, because generated caps can preserve upstream components
    # even when the released output excludes audio.
    if any(x in combined for x in ["not video or audio", "without sound", "audio is not included", "stick-figure", "skeleton", "camera images"]):
        return "camera"

    # Keep the device row aligned with what is actually shared. If the output is
    # camera-only image/video/pose data, do not mention a microphone merely
    # because another variant in the same task can include audio.
    if flags.get("has_visual") and not flags.get("has_audio"):
        return "camera"
    if flags.get("has_audio") and not flags.get("has_visual"):
        return "microphone"
    if flags.get("has_visual") and flags.get("has_audio"):
        return "camera and microphone"
    if "audio-video" in combined or ("video with sound" in combined) or ("audio" in combined and ("video" in combined or "camera" in combined)):
        return "camera and microphone"
    if any(x in combined for x in ["video", "image", "camera", "pose", "stick-figure", "skeleton"]):
        return "camera"
    if any(x in combined for x in ["audio", "sound", "speech", "decibel", "microphone"]):
        return "microphone"
    if any(x in combined for x in ["occupancy", "presence", "people count"]):
        return "camera or presence sensor"

    task = str((flow or {}).get("task") or "").lower()
    if "sound" in task:
        return "microphone"
    if "adl" in task or "activity" in task:
        return "camera and/or microphone"
    if "fall" in task:
        return "camera or motion sensor"
    if "visitor" in task or "presence" in task:
        return "camera or presence sensor"
    return "sensing device"



def _contextual_actor_value(original_label: str, original_value: str, flow: Optional[Dict[str, Any]], output_variant: Optional[Dict[str, Any]] = None) -> Optional[str]:
    setting_plain = _setting_plain(flow)
    subject = _subject_plain(flow)
    text = original_value
    if not setting_plain:
        return None

    if original_label == "Sender":
        device = _sensor_device_description(flow, output_variant)
        try:
            params = scenario_ci_params(flow or {})
        except Exception:
            params = {}
        context = str(params.get("context") or "").strip()
        if text == "resident-owned device":
            if context == "public_space":
                return f"The device is a {device} owned and controlled by a nearby shop owner, not by the city or local government."
            subject_lower = str(subject or "").lower().strip()
            if subject_lower == "child":
                return f"The device is a {device} controlled by another adult in the household, not by the child or the parent/guardian you are roleplaying as."
            if subject_lower in {"guest", "visitor", "roommate"}:
                role_phrase = "co-resident or roommate" if subject_lower == "roommate" else subject_lower
                return f"The device is a {device} controlled by a household resident, not by you as the {role_phrase}."
            if subject_lower == "resident":
                return f"The device is a {device} controlled by you or another resident of the home."
            return f"The device is a {device} controlled by a household resident."
        sender_map = {
            "rental host’s device": f"The device is a {device} controlled by the rental host.",
            "organization-operated system": f"The device is a {device} controlled by the organization responsible for the {setting_plain}.",
            "hospital monitoring system": f"The device is a {device} controlled by the hospital or clinic.",
            "patient/family device": f"The device is a {device} managed by the patient, resident, or family.",
            "research sensor system": f"The device is a {device} controlled by the researchers conducting the smart-home study.",
            "school monitoring system": f"The device is a {device} controlled by the school.",
        }
        return sender_map.get(text)

    if original_label == "Data subject":
        return _participant_subject_value(flow)

    if original_label == "Recipient":
        if text == "caregiver" and str(subject).lower().strip() == "roommate":
            return f"A caregiver receives the data to support you as a co-resident { _area_for_recipient_phrase(flow) }."
        purpose_phrase = _purpose_for_recipient_phrase(flow)
        area_phrase = _area_for_recipient_phrase(flow)
        recipient_map = {
            "home resident": f"The home resident receives the data for {purpose_phrase} {area_phrase}.",
            "homeowner/resident": f"The home resident receives the data for {purpose_phrase} {area_phrase}.",
            "rental host": f"The rental host receives the data for {purpose_phrase} {area_phrase}.",
            "authorized staff": f"Authorized staff receive the data for {purpose_phrase} {area_phrase}.",
            "caregiver": f"A caregiver receives the data for {purpose_phrase} {area_phrase}.",
            "clinician": f"A clinician or care team receives the data for {purpose_phrase} {area_phrase}.",
            "researcher": f"Researchers receive the data for {purpose_phrase}.",
            "teacher or school official": f"School staff receive the data for {purpose_phrase} {area_phrase}.",
            "the smart-space app": _app_recipient_value(flow),
            "the app or software service": _app_recipient_value(flow),
            "task-specific app": _app_recipient_value(flow),
        }
        return recipient_map.get(text)
    return None

def _technical_task_description(flow: Optional[Dict[str, Any]]) -> str:
    task = str((flow or {}).get("task") or "").lower()
    try:
        params = scenario_ci_params(flow or {})
    except Exception:
        params = {}
    space = str(params.get("space") or "").lower()
    context = str(params.get("context") or "").lower()

    if "visitor" in task or "presence" in task:
        return "detects whether someone enters, leaves, approaches, or is present in the area"
    if "fall" in task:
        return "looks for signs that a person may have fallen and may need help"
    if "adl" in task or "activity" in task:
        if space == "bedroom":
            return "tracks how the bedroom is used, such as sleeping, resting, or getting in and out of bed"
        if space == "kitchen":
            return "tracks how the kitchen is used, such as cooking, preparing food, eating, or moving around"
        if space == "bathroom":
            return "tracks bathroom movement or possible safety-related activity"
        if space == "living_room":
            return "tracks how the living room is used, such as sitting, walking, resting, or moving around"
        if space == "common_area":
            return "tracks how the shared area is used, such as sitting, walking, standing, or moving around"
        return "tracks daily movement and activity patterns, such as sitting, walking, resting, or moving around"
    if "sound" in task:
        return "detects or classifies everyday sound events, such as alarms, glass breaking, footsteps, loud sounds, or other sound patterns"
    return "performs the sensing task described in this scenario"

def _purpose_goal_description(original_value: str, flow: Optional[Dict[str, Any]]) -> str:
    purpose = str(original_value or "").strip()
    task = str((flow or {}).get("task") or "").lower()
    if "routine" in purpose or "activity or environmental sound" in purpose:
        if "sound" in task:
            return "monitor everyday sound patterns over time, such as noise levels, alarms, footsteps, or loud sounds"
        return "see how the room is being used throughout the day, such as sitting, walking, resting, or moving around"
    if "safety" in purpose:
        if "visitor" in task or "presence" in task:
            return "support safety by noticing when someone is present, entering, leaving, or approaching"
        if "sound" in task:
            return "support safety by detecting concerning sounds, such as alarms, breaking glass, yelling, or other urgent audio cues"
        if "fall" in task:
            return "support safety by detecting a possible fall or urgent safety event"
        return "support safety by detecting a relevant urgent event"
    if "security" in purpose or "visitor" in purpose:
        return "support visitor/security monitoring, such as noticing someone arriving, entering, or approaching the area"
    if "energy" in purpose or "lighting" in purpose:
        return "support energy or lighting automation, such as turning lights or HVAC on/off when people enter, leave, or occupy the area"
    if "work performance" in purpose or "employee activity" in purpose or "employee presence" in purpose:
        return "monitor employee activity or presence, such as whether employees are present, moving, or active on the main office floor"
    if "clinical care" in purpose:
        if "sound" in task:
            return "support clinical care by alerting clinicians to patient-room sounds or sound patterns that may need attention"
        if "activity" in task or "adl" in task:
            return "support clinical care by tracking patient movement and daily activity relevant to care"
        return "support clinical care by helping clinicians monitor patient safety or care needs"
    if purpose == "fall detection":
        return "send help or an alert when a possible fall is detected"
    if "personalization" in purpose:
        return "personalize a home service based on how the room is used throughout the day"
    if "research" in purpose:
        return "support a research study, such as studying daily activities or sensor behavior with participants"
    if "training" in purpose or "supervision" in purpose:
        return "support school staff supervision, such as reviewing activity patterns or events for oversight"
    if purpose in {"voice command or communication", "audio-event or sound-cue support"} or "audio-event" in purpose or "sound-cue" in purpose:
        return "support safety or security by detecting relevant sound events"
    return purpose or "serve the stated purpose"


def _purpose_display_value(original_value: str, flow: Optional[Dict[str, Any]]) -> str:
    """Two-sentence purpose description for participant-facing survey text."""
    technical = _technical_task_description(flow)
    goal = _purpose_goal_description(original_value, flow)
    return f"The system {technical}. The overall goal is to {goal}."

def _purpose_description(original_value: str, flow: Optional[Dict[str, Any]]) -> str:
    value = _purpose_display_value(original_value, flow)
    return f"This describes what the monitoring device is trying to do with the collected data or shared output: {value}."



def _display_value_override(original_label: str, value: Any, flow: Optional[Dict[str, Any]] = None, output_variant: Optional[Dict[str, Any]] = None) -> Any:
    if value is None:
        return value
    text = str(value)

    if original_label == "Situation":
        return _situation_value_sentence(flow, text)

    if original_label == "Purpose":
        return _purpose_display_value(text, flow)

    contextual = _contextual_actor_value(original_label, text, flow, output_variant)
    if contextual:
        return contextual

    if text == "the smart-space app":
        return _app_recipient_value(flow)

    if text == "only when an event occurs":
        return f"Data is collected, processed, or shared only after the monitoring device detects {_event_phrase(flow)}."

    local_text = "Locally inside the home or building. The data is not sent over the internet for company analysis."
    if _uses_app_recipient(flow):
        local_text = "Locally inside the home or building. The data is never sent over the internet or shared with the app company."
    # Some generated context rows include small wording variants such as
    # "speech content is removed before sharing"; normalize these before the
    # exact replacement table.
    if original_label == "Transmission principle":
        lowered = text.lower().strip()
        if "speech content is removed" in lowered or "spoken words" in lowered:
            return "Speech-like parts of the audio are silenced before sharing, so words should not be understandable. Other sounds may remain."
        if "only when an event occurs" in lowered or "event-triggered" in lowered:
            return f"Data is collected, processed, or shared only after the monitoring device detects {_event_phrase(flow)}."

    replacements = {
        "continuous monitoring": "The device stays on continuously, rather than turning on only after motion, sound, or another event.",
        "processed locally": local_text,
        "not disclosed to the person": "The monitored person is not told that data collection and sharing are happening.",
        "disclosed to people nearby": "People in this setting are told that data is being collected or shared.",
        "written notice is provided": "Affected people receive written notice about collection and sharing.",
        "speech content is removed": "Speech-like parts of the audio are silenced before sharing, so words should not be understandable. Other sounds may remain.",
        "sent to a cloud server": "The data is sent over the internet to a secure online server (cloud storage) for analysis before the listed output is shared.",
        "sent to a cloud service": "The data is sent over the internet to a secure online server (cloud storage) for analysis before the listed output is shared.",
        "explicit consent is obtained": "The monitored person gives explicit consent for this collection and sharing.",
        "only authorized people can access it": "Only authorized staff can access the shared output.",
        "disclosed in the rental listing": "The rental listing discloses the device, its coverage area, and what data or output may be shared.",
    }
    return replacements.get(text, text)

def _hidden_device_example(flow: Optional[Dict[str, Any]]) -> str:
    """Return a hidden-device example without implying extra sensors."""
    return "a hidden device collects or analyzes the listed data without telling the monitored person"


def _transmission_help_and_example(original_value: str, flow: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    event_phrase = _event_phrase(flow)
    value_lower = str(original_value or "").lower()
    if original_value == "continuous monitoring" or "continuous" in value_lower or "ongoing" in value_lower:
        return (
            "The device stays on continuously, rather than turning on only after motion, sound, or another event.",
            None,
        )
    if original_value == "only when an event occurs" or ("event" in value_lower and "only" in value_lower):
        return (
            f"The monitoring device waits until it detects {event_phrase} before collecting, analyzing, or sharing the listed output.",
            f"the monitoring device sends the listed output only when it detects {event_phrase}",
        )
    if original_value == "not disclosed to the person" or "not disclosed" in value_lower or "hidden" in value_lower:
        return (
            "The person being monitored is not told that the device is collecting data or that the listed output may be shared.",
            _hidden_device_example(flow),
        )
    if original_value == "processed locally" or "local" in value_lower:
        return (
            "Locally inside the home or building. The data is not sent over the internet for company analysis.",
            None,
        )
    if original_value == "disclosed to people nearby" or "people nearby" in value_lower:
        return (
            "People nearby are told that the device is collecting data and/or that the listed output may be shared.",
            "a sign, notice, or setup screen explains that the sensor is active and what kind of output may be shared",
        )
    if original_value == "written notice is provided" or "written notice" in value_lower:
        return (
            "Affected people receive written notice explaining the device, collection, recipient, and purpose. For children, this may mean parent/guardian notice or a school-required process.",
            "a written workplace, school, care, or study notice explains the monitoring and sharing",
        )
    if original_value == "sent to a cloud server" or "cloud" in value_lower:
        return (
            "The data is sent over the internet to a secure online server (cloud storage) for analysis instead of staying on the device or a nearby device in the same home or building.",
            "audio, video, or sensor data is sent over the internet to a secure online server before the listed output is produced or shared",
        )
    if original_value == "explicit consent is obtained" or "explicit consent" in value_lower:
        return (
            "The monitored person clearly agrees to this data collection and sharing for the stated purpose. For children or dependent adults, the relevant consent process may involve a parent, guardian, or legally authorized representative.",
            "a signed consent form or clear opt-in explains what data is collected, who receives it, and why",
        )
    if original_value == "only authorized people can access it" or "authorized" in value_lower:
        return (
            "Only approved people for this setting and purpose can access the listed output.",
            "only approved care staff, clinical staff, security staff, or designated managers can see the alert or output",
        )
    if original_value == "disclosed in the rental listing" or "rental listing" in value_lower:
        return (
            "The short-term rental listing describes the device, where it is located, what area it covers, and what kind of output may be shared.",
            "the listing says an outdoor camera covers the entrance and explains what information the host may receive",
        )
    if original_value == "speech content is removed" or "speech" in value_lower:
        return (
            "Speech-like parts of the audio are silenced before sharing, so words should not be understandable. Other sounds may remain.",
            None,
        )
    return None, None


def _readable_entry(flow: Optional[Dict[str, Any]], field: str) -> Dict[str, Any]:
    readable = (flow or {}).get("ci_parameters_readable_context_only") or {}
    return readable.get(field) if isinstance(readable.get(field), dict) else {}



def _context_space_example(flow: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return a context-specific example for situation rows.

    Some space values, especially "outdoor area", are intentionally reused
    across contexts. A generic outdoor example such as "porch" or "driveway"
    makes sense for a home or short-term rental, but not for a public-space
    scenario. Keep the machine-readable context/space terms unchanged and tailor
    only the participant-facing example to the context+space pair.
    """
    try:
        params = scenario_ci_params(flow or {})
    except Exception:
        params = {}
    context = str(params.get("context") or "").strip()
    space = str(params.get("space") or "").strip()
    pair_examples = {
        ("home", "bathroom"): "a bathroom or restroom in a private home",
        ("home", "bedroom"): "a bedroom in a private home or shared apartment",
        ("home", "common_area"): "a shared hallway, lounge, or common room in a home",
        ("home", "entrance"): "a front door, entryway, or lobby entrance of a home",
        ("home", "kitchen"): "a kitchen or food-preparation area in a private home",
        ("home", "living_room"): "a living room or shared living area in a private home",
        ("home", "outdoor"): "a front porch, driveway, yard, garden, or home entrance",
        ("short_term_rental", "bedroom"): "a bedroom in an Airbnb, Vrbo, or other short-term rental",
        ("short_term_rental", "living_room"): "a living room or shared indoor area in an Airbnb, Vrbo, or other short-term rental",
        ("short_term_rental", "outdoor"): "a rental porch, driveway, yard, exterior walkway, or outside entrance",
        ("public_space", "outdoor"): "a park, sidewalk, public plaza, transit stop, public walkway, or outdoor public entrance",
        ("workplace", "workspace"): "the main office floor, an office cubicle area, shared staff room, desk area, or shop floor",
        ("workplace", "outdoor"): "an outdoor worksite, loading area, parking area, or workplace entrance",
        ("school_or_classroom", "common_area"): "a school hallway, shared common area, classroom-adjacent area, or school lounge",
        ("school_or_classroom", "outdoor"): "a school playground, courtyard, outdoor walkway, or school entrance",
        ("long_term_care", "bedroom"): "a resident bedroom in an assisted-living or long-term care home",
        ("long_term_care", "living_room"): "a shared living room or lounge in a long-term care home",
        ("long_term_care", "outdoor"): "a care-home garden, courtyard, patio, or outdoor entrance",
        ("hospital_or_clinic", "patient_room"): "a hospital or clinic patient room",
        ("hospital_or_clinic", "outdoor"): "a hospital entrance, clinic entrance, ambulance bay, or outdoor walkway",
        ("research_living_lab", "living_room"): "a smart-apartment living room used in a research study",
        ("research_living_lab", "outdoor"): "an outdoor area associated with the research living-lab setting",
    }
    if (context, space) in pair_examples:
        return pair_examples[(context, space)]
    if space == "outdoor":
        return "an outside area near the setting"
    return None

def _participant_example_for_field(original_label: str, row: Dict[str, Any], flow: Optional[Dict[str, Any]], original_value: str) -> Optional[str]:
    """Return only examples that add concrete context for a participant.

    The scenario table already phrases the left-column labels as questions, so
    most definitional helper text and obvious examples create clutter.  Keep
    examples mainly where they make an abstract setting or policy condition more
    concrete.
    """
    if original_label == "Situation":
        pair_example = _context_space_example(flow)
        if pair_example:
            return pair_example
        sit = _readable_entry(flow, "situation")
        context_ex = (((sit.get("context") or {}) if isinstance(sit, dict) else {}).get("example") or "").strip()
        space_ex = (((sit.get("space") or {}) if isinstance(sit, dict) else {}).get("example") or "").strip()
        examples = []
        seen_examples = set()
        for x in [context_ex, space_ex]:
            cleaned = str(x or "").strip().rstrip(".")
            key = cleaned.lower()
            if cleaned and key not in seen_examples:
                seen_examples.add(key)
                examples.append(cleaned)
            # If two examples are effectively the same (for example, both just say
        # "hospital patient room"), keep only the more informative wording.
        filtered = []
        for ex in examples:
            ex_l = ex.lower().replace("a ", "").replace("an ", "")
            redundant = False
            for other in examples:
                if ex == other:
                    continue
                other_l = other.lower().replace("a ", "").replace("an ", "")
                if ex_l in other_l and len(ex_l) < len(other_l):
                    redundant = True
                    break
            if not redundant:
                filtered.append(ex)
        if filtered:
            return "; ".join(filtered)
        return row.get("example")

    if original_label == "Transmission principle":
        # Keep examples only for policy/handling conditions that can be ambiguous
        # without a concrete illustration.  Omit obvious examples for ordinary
        # continuous, event-triggered, cloud, or authorized-access cases.
        value_lower = str(original_value or "").lower()
        keep_for = (
            "hidden",
            "not disclosed",
            "rental listing",
            "written notice",
            "explicit consent",
            "speech",
            "local",
        )
        if any(k in value_lower for k in keep_for):
            _help, example = _transmission_help_and_example(original_value, flow)
            return example
        return None

    # Actor values such as "rental host" or "visitor" are usually already
    # self-explanatory after the label is phrased as a question.
    return None


def _plain_field(row: Dict[str, Any], flow: Optional[Dict[str, Any]] = None, output_variant: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Rename technical CI labels into participant-facing question prompts."""
    label = str(row.get("label") or "")
    mapping = {
        "Situation": "Where does this happen?",
        "Sender": "Who owns or operates the device?",
        "Data subject": "Who is the data about?",
        "Recipient": "Who would receive or use the shared output?",
        "Purpose": "What is the monitoring device trying to do?",
        "Transmission principle": "Under what condition is data collected, processed, or shared?",
    }
    new_label = mapping.get(label, label)
    original_value = str(row.get("value") or "")
    value = _display_value_override(label, row.get("value"), flow, output_variant)
    example = _participant_example_for_field(label, row, flow, original_value)

    # Keep the main cell focused on the value. Longer helper descriptions that only
    # explain the left-column parameter are intentionally omitted.
    desc = ""

    field = {
        "label": new_label,
        "value": value,
        "description": desc,
        "help": desc,
        "example": example,
        "ci_field_label": label,
    }
    if label == "Situation":
        setting, space = _split_situation_value(original_value)
        field["value_parts"] = {"general_setting": setting, "specific_place": space}
        field["value_html"] = _safe_emphasize_terms(str(value), [setting, space]) if (setting or space) else str(value)
    return field

def output_example(label: Any, description: Any = None) -> Optional[str]:
    """Short optional example text for less-obvious shared-output rows.

    Keep examples aligned with the actual data statement. In particular, do not
    show a “words are removed” example for raw audio or audio-video outputs that
    may include speech.
    """
    text = f"{label or ''} {description or ''}".lower()
    if "pose" in text or "stick-figure" in text:
        return None
    # Do not add examples for speech-filtered audio. The value and description
    # already explain this concept, and repeated examples made the page feel
    # redundant.
    if ("spoken words" in text or "conversation content" in text or "speech" in text) and ("removed" in text or "not the words" in text or "cannot be understood" in text):
        return None
    if "may include speech" in text or "may include speech or conversation" in text:
        return None
    if "activity summary" in text or "daily-activity summary" in text or "summary of detected daily activity" in text:
        return "Example: a monthly bar graph showing the percentage of time a room was occupied versus empty, with no specific times or names attached."
    if "sound-event summary" in text or "sound event summary" in text or "summary of detected sound events" in text:
        return "Example: a weekly chart showing: alarms triggered: 2 times; footstep noise: 15% of the day; quiet hours: 80% of the day."
    if "combined text log" in text or "written timeline" in text or "sensor alerts" in text:
        return "Example: Safety alert: front door opened at 3:00 AM, followed immediately by motion in the hallway."
    if "detected people" in text or "person locations" in text or "bounding boxes" in text:
        return "a list of detected person locations, not the camera image"
    if "presence" in text or "occupancy" in text:
        return "a yes/no presence signal or a count of people in the room"
    if "sound-level" in text or "sound level" in text or "decibel" in text:
        return "a decibel level, not the original audio"
    if "sound-event label" in text or "sound category" in text or "noise category" in text:
        return "a label such as alarm, glass breaking, or footsteps"
    if "activity label" in text or "activity category" in text:
        return "a label such as walking, sitting, resting, or moving around"
    if "event alert" in text or "event label" in text:
        return "Example: Safety alert: front door opened at 3:00 AM, followed immediately by motion in the hallway."
    return None



def _escape_html(text: Any) -> str:
    return (str(text or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _safe_emphasize_terms(text: str, terms: List[str]) -> str:
    """Return simple server-controlled HTML with key scenario variables emphasized."""
    escaped = _escape_html(text)
    for term in sorted({str(t or "").strip() for t in terms if str(t or "").strip()}, key=len, reverse=True):
        eterm = _escape_html(term)
        if eterm and eterm in escaped:
            escaped = escaped.replace(eterm, f'<strong class="scenario-var">{eterm}</strong>', 1)
    return escaped

def _task_overview_text(flow: Optional[Dict[str, Any]]) -> str:
    try:
        params = scenario_ci_params(flow or {})
    except Exception:
        params = {}
    context = str(params.get("context") or "").strip()
    location = _location_phrase(flow)
    technical = _technical_task_description(flow)
    goal = _purpose_goal_description(_purpose_plain(flow), flow)
    surveyed_user = _surveyed_user_sentence(flow)

    if context == "research_living_lab":
        intro = f"Researchers are conducting a study on smart-home sensing. In {location}, a monitoring device {technical}."
    else:
        intro = f"In {location}, a monitoring device {technical}."
    return f"{intro} The overall goal is to {goal}."

def _role_anchor_text(flow: Optional[Dict[str, Any]]) -> str:
    subject = _subject_plain(flow).strip().lower()
    setting_phrase = _location_phrase(flow, demonstrative=True)
    if subject == "child":
        return f"You are answering as a parent, guardian, or responsible adult for a child in {setting_phrase}."
    role = _subject_role_for_participant(subject, responsible=False)
    prep = _place_preposition_for_phrase(setting_phrase)
    return f"For this scenario, imagine the data is about you as {role} {prep} {setting_phrase}."


def _role_anchor_field(flow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    text = _role_anchor_text(flow)
    subject = _subject_plain(flow).strip().lower()
    if subject == "child":
        terms = ["parent, guardian, or responsible adult", "child"]
    else:
        terms = [_subject_role_for_participant(subject, responsible=False)]
    value_html = _safe_emphasize_terms(text, terms)
    return {
        "label": "What is your role in this scenario?",
        "value": text,
        "value_html": value_html,
        "description": "",
        "help": "",
        "example": None,
        "ci_field_label": "Persona",
    }



def _scenario_group_title(flow: Optional[Dict[str, Any]]) -> str:
    """Short stable label for a family of similar survey cases."""
    family = _task_family_phrase(flow).lower()
    location = _location_phrase(flow)
    # Avoid awkward capitalization like "fall detection" at sentence start later.
    return f"{family} in {location}"


def _scenario_group_field(flow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    title = _scenario_group_title(flow)
    value = (
        f"This question is part of a group about {title}. "
        "You may see more than one version of this same general scenario; rate this version based only on the details shown on this page."
    )
    value_html = _safe_emphasize_terms(value, [title, "this version"])
    return {
        "label": "What general scenario is this question about?",
        "value": value,
        "value_html": value_html,
        "description": "",
        "help": "",
        "example": None,
        "ci_field_label": "Scenario group",
    }


def _scenario_overview_field(flow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    location = _location_phrase(flow)
    goal = _purpose_goal_description(_purpose_plain(flow), flow)
    task = str((flow or {}).get("task") or "").lower()
    try:
        params = scenario_ci_params(flow or {})
    except Exception:
        params = {}
    context = str(params.get("context") or "").lower()
    space = str(params.get("space") or "").lower()

    # One concise narrative sentence. Avoid repeating the same activity phrase in
    # both the task and the goal.
    if "fall" in task:
        overview = f"A monitoring device in {location} looks for possible falls so the named receiver can respond or send help."
    elif "visitor" in task or "presence" in task:
        if "energy" in goal or "lighting" in goal:
            overview = f"A monitoring device in {location} checks whether someone is present so lights, HVAC, or other automation can respond."
        elif context == "workplace":
            overview = "A workplace device checks whether employees are present on the main office floor to monitor daily activity and presence."
        else:
            overview = f"A monitoring device in {location} checks whether someone is present, entering, leaving, or approaching for visitor or security monitoring."
    elif "sound" in task:
        overview = f"A monitoring device in {location} listens for sound events such as alarms, footsteps, or glass breaking to {goal}."
    elif "adl" in task or "activity" in task:
        if context == "workplace":
            overview = "A workplace device tracks employee movement, such as sitting, walking, or resting, on the main office floor to monitor daily activity and presence."
        elif space == "living_room":
            overview = f"A monitoring device tracks living-room activity throughout the day, such as sitting, walking, or resting, for {_purpose_for_recipient_phrase(flow)}."
        elif space == "kitchen":
            overview = f"A monitoring device tracks kitchen activity, such as cooking or moving around, for {_purpose_for_recipient_phrase(flow)}."
        elif space == "bedroom":
            overview = f"A monitoring device tracks bedroom activity, such as sleeping, resting, or getting in and out of bed, for {_purpose_for_recipient_phrase(flow)}."
        else:
            overview = f"A monitoring device in {location} tracks daily movement, such as sitting, walking, or resting, for {_purpose_for_recipient_phrase(flow)}."
    else:
        technical = _technical_task_description(flow)
        overview = f"A monitoring device in {location} {technical} to {goal}."

    if context == "research_living_lab":
        overview = "Researchers are conducting a smart-home study. " + overview

    group_title = _scenario_group_title(flow)
    value_html = _safe_emphasize_terms(overview, [group_title, location])
    return {
        "label": "What is the scenario?",
        "value": overview,
        "value_html": value_html,
        "description": "",
        "help": "",
        "example": None,
        "ci_field_label": "Scenario",
    }


def _flow_has_speech_removed_condition(flow: Optional[Dict[str, Any]]) -> bool:
    try:
        params = scenario_ci_params(flow or {})
    except Exception:
        params = {}
    tp = str(params.get("transmission_principle") or "").lower()
    if "speech_content_removed" in tp or "speech" in tp:
        return True
    for row in (flow or {}).get("participant_display_fields", []) or []:
        if row.get("label") == "Transmission principle":
            text = " ".join(str(row.get(k) or "") for k in ["value", "help", "example"]).lower()
            if "speech" in text or "spoken word" in text:
                return True
    return False


def _output_text_for_flow(output_variant: Dict[str, Any], flow: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    """Return participant-facing output text after applying scenario-level conditions.

    Output variants are deduplicated before the scenario condition row is rendered.
    This small adjustment prevents contradictions such as an audio output saying it
    may include speech while the scenario's handling condition says spoken words
    are removed before sharing.
    """
    label = str(output_variant.get("output_variant_label") or "")
    desc = str(output_variant.get("output_variant_description") or "")
    lower = f"{label} {desc}".lower()
    if _flow_has_speech_removed_condition(flow) and ("audio" in lower or "microphone" in lower or "speech" in lower):
        if "camera" in lower or "video" in lower:
            label = "The shared data is a combined video and speech-muted audio sample: a short synchronized video and audio clip where human words are scrubbed out so conversations cannot be understood."
            desc = "The video remains visible. Other background sounds, such as alarms, footsteps, or coughing, may still be heard."
        else:
            label = "The shared data is a speech-muted audio clip: an audio recording where human words are scrubbed out so conversations cannot be understood, but background noises such as alarms, footsteps, or coughing may remain."
            desc = "Human words are muted or scrubbed out. Other background sounds may still be heard."
    return label, desc if desc != "" else ""


def _row_for_ci_label(flow: Optional[Dict[str, Any]], label: str) -> Optional[Dict[str, Any]]:
    for row in (flow or {}).get("participant_display_fields", []) or []:
        if row.get("label") == label:
            return row
    return None


def _compact_fragment(text: Any, prefixes: Tuple[str, ...] = ()) -> str:
    fragment = str(text or "").strip()
    for prefix in prefixes:
        if fragment.lower().startswith(prefix.lower()):
            fragment = fragment[len(prefix):].strip()
            break
    return fragment.rstrip(".")


def _device_owner_field(flow: Dict[str, Any], output_variant: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    row = _row_for_ci_label(flow, "Sender") or {}
    value = _display_value_override("Sender", row.get("value"), flow, output_variant)
    if not str(value or "").strip().endswith("."):
        value = f"{value}."
    return {
        "label": "Who controls the monitoring device?",
        "value": value,
        "description": "",
        "help": "",
        "example": None,
        "ci_field_label": "Sender",
    }


def _recipient_access_scope_clause(flow: Optional[Dict[str, Any]]) -> str:
    """Access assumption folded into the existing recipient row."""
    text = _transmission_value_text(flow)
    if "authorized" in text:
        return "Only the named receiver and authorized people for this setting and purpose can access or use the shared data."
    if "local" in text or "processed on the device" in text or "local hub" in text:
        return "The data is processed on the local device or local hub; do not assume the device maker, cloud provider, or other unlisted third parties can access it."
    if "cloud" in text:
        return "A cloud service processes the data before the listed output is shared; apart from that processing and the named receiver, do not assume other unlisted parties can access it."
    return "Only this named receiver or listed app/service can access or use the shared data; do not assume device makers, cloud providers, landlords, managers, or other unlisted parties can access it."


def _recipient_field(flow: Dict[str, Any], output_variant: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    row = _row_for_ci_label(flow, "Recipient") or {}
    value = _display_value_override("Recipient", row.get("value"), flow, output_variant)
    if not str(value or "").strip().endswith("."):
        value = f"{value}."
    access_clause = _recipient_access_scope_clause(flow)
    if access_clause:
        value = f"{value} {access_clause}"
    return {
        "label": "Who receives or uses the shared data?",
        "value": value,
        "description": "",
        "help": "",
        "example": None,
        "ci_field_label": "Recipient",
    }


def _transmission_value_text(flow: Optional[Dict[str, Any]]) -> str:
    """Return both readable and machine transmission-principle text."""
    row = _row_for_ci_label(flow, "Transmission principle") or {}
    try:
        params = scenario_ci_params(flow or {})
    except Exception:
        params = {}
    return " ".join([
        str(row.get("value") or ""),
        str(row.get("help") or ""),
        str(row.get("example") or ""),
        str(params.get("transmission_principle") or ""),
    ]).lower()


def _is_notice_permission_condition(original_value: str) -> bool:
    value_lower = str(original_value or "").lower()
    return any(k in value_lower for k in [
        "hidden",
        "not disclosed",
        "disclosed",
        "notice",
        "consent",
        "listing",
        "people nearby are told",
        "told monitoring",
        "told that monitoring",
    ])


def _is_authorized_access_condition(original_value: str) -> bool:
    return "authorized" in str(original_value or "").lower()


def _notice_permission_value(flow: Optional[Dict[str, Any]]) -> str:
    text = _transmission_value_text(flow)
    if "hidden" in text or "not disclosed" in text:
        return "You are not told that this data collection and sharing are happening."
    if "explicit consent" in text or "clear opt-in" in text or "signed consent" in text:
        return "You explicitly agree to this data collection and sharing for the stated purpose."
    if "written notice" in text:
        return "You receive written notice explaining the device, the data collection, the named receiver, and the stated purpose."
    if "rental listing" in text or "listing_disclosure_required" in text:
        return "The rental listing tells you about the device, its coverage area, and what data or output may be shared."
    if "disclosed" in text or "people nearby are told" in text or "sign" in text or "setup screen" in text:
        return "You are told that monitoring or data sharing is happening, such as through a sign, notice, or setup screen."
    return "No additional notice or permission condition is stated. Use only the details on this page; do not assume extra consent, extra notice, or hidden access."


def _notice_permission_example(flow: Optional[Dict[str, Any]]) -> Optional[str]:
    text = _transmission_value_text(flow)
    if "rental listing" in text or "listing_disclosure_required" in text:
        return "for example, an Airbnb/Vrbo listing says what device is present and what area it covers"
    if "explicit consent" in text:
        return "for example, a clear opt-in or consent form before monitoring begins"
    if "written notice" in text:
        return "for example, a written workplace, care, or research notice"
    if "hidden" in text or "not disclosed" in text:
        return "for example, no sign, listing notice, setup notice, or consent prompt is provided"
    if "disclosed" in text:
        return "for example, a sign, notice, or setup screen tells people monitoring is happening"
    return None


def _condition_label_for_value(original_value: str) -> str:
    value_lower = str(original_value or "").lower()
    if _is_notice_permission_condition(original_value):
        return "What notice or permission is given?"
    if any(k in value_lower for k in ["continuous", "event"]):
        return "When is data collected or shared?"
    if any(k in value_lower for k in ["local", "cloud"]):
        return "Where is the data processed?"
    if "authorized" in value_lower:
        return "Who is allowed to access the shared data?"
    if "speech" in value_lower:
        return "What audio filtering happens before sharing?"
    return "What rule applies to the data?"


def _condition_field(flow: Dict[str, Any], output_variant: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    row = _row_for_ci_label(flow, "Transmission principle") or {}
    original_value = str(row.get("value") or "")
    # If speech filtering is already incorporated into the data-shared row, do
    # not repeat the same fact as a separate condition row.
    if output_variant is not None and "speech" in original_value.lower():
        return None
    # Access-control-only conditions are folded into the recipient row so the
    # survey does not introduce a separate access field.
    if _is_authorized_access_condition(original_value):
        return None
    if _is_notice_permission_condition(original_value):
        value = _notice_permission_value(flow)
        example = _notice_permission_example(flow)
    else:
        value = _display_value_override("Transmission principle", row.get("value"), flow, output_variant)
        example = _participant_example_for_field("Transmission principle", row, flow, original_value) if row else None
    if not str(value or "").strip().endswith("."):
        value = f"{value}."
    label = _condition_label_for_value(original_value)
    return {
        "label": label,
        "value": value,
        "description": "",
        "help": "",
        "example": example,
        "ci_field_label": "Transmission principle",
    }


def participant_display_fields(flow: Dict[str, Any], output_variant: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Put the scenario first so participants see the stable vignette before the
    # role and parameters. Each row label is a complete question in ordinary
    # language rather than a CI/factorial-design term.
    fields = [_scenario_overview_field(flow), _role_anchor_field(flow)]
    if output_variant:
        output_label, output_desc = _output_text_for_flow(output_variant, flow)
        fields.append({
            "label": "What data would be shared?",
            "value": output_label,
            "description": output_desc,
            "help": output_desc,
            "example": output_example(output_label, output_desc),
            "ci_field_label": "Output",
        })
    for extra in [
        _device_owner_field(flow, output_variant),
        _recipient_field(flow, output_variant),
        _condition_field(flow, output_variant),
    ]:
        if extra is not None:
            fields.append(extra)
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
        "Some questions are versions of the same general scenario. "
        "If a row is marked Changed, that row differs from the previous question; rows not marked Changed still apply. "
        "Please rate this page as its own complete situation. Assume only the listed data is saved or shared, and only the parties shown on this page can access it. "
        "Do not judge whether the setup is legal or whether the space owner is allowed to use it; judge whether the data sharing feels appropriate from your perspective as the person the data is about."
    )


def build_attention_check(item_id: Any, flow_id: Any, index: int, display_fields: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Create a Prolific-style attention check.

    This check explicitly tells the participant which answer to select for this
    very question. It is not a memory check and does not depend on scenario
    interpretation.
    """
    # Keep checks infrequent: about two checks in a 25-question assignment,
    # away from the first page and separate from comprehension checks.
    if index not in {6, 17}:
        return None
    expected = "Somewhat agree"
    options = [
        "Strongly disagree",
        "Disagree",
        "Somewhat agree",
        "Agree",
        "Strongly agree",
    ]
    return {
        "field_label": "Attention check",
        "expected_value": expected,
        "question": "To show that you are paying attention, please select ‘Somewhat agree’ for this question.",
        "input_type": "select",
        "options": options,
        "required": True,
        "note": "This check is not asking about the scenario.",
    }


def _comprehension_subject_expected(flow: Optional[Dict[str, Any]]) -> str:
    subject = _subject_plain(flow).strip().lower()
    if subject == "roommate":
        return "You as a co-resident or roommate"
    if subject == "research participant":
        return "You as a research participant"
    if subject == "employee":
        return "You as an employee"
    if subject in {"guest", "patient", "resident", "visitor"}:
        return f"You as a {subject}"
    return "The person named in the scenario"


def build_comprehension_check(item_id: Any, flow_id: Any, index: int, flow: Dict[str, Any], display_fields: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ask a short scenario comprehension question.

    These are scenario-based checks, not attention checks. They appear only a
    couple of times and use short answer choices drawn from the visible role row.
    """
    if index not in {11, 22}:
        return None
    expected = _comprehension_subject_expected(flow)
    pool = [
        "You as a guest",
        "You as a resident",
        "You as a patient",
        "You as an employee",
        "You as a visitor",
        "You as a research participant",
        "You as a co-resident or roommate",
        "A rental host",
        "A clinician",
    ]
    choices = [x for x in pool if normalize_attention_answer(x) != normalize_attention_answer(expected)]
    rng = random.Random(stable_int(f"comprehension::{item_id}::{flow_id}::{index}"))
    rng.shuffle(choices)
    options = [expected] + choices[:4]
    rng.shuffle(options)
    return {
        "field_label": "Scenario check",
        "expected_value": expected,
        "question": "According to this page, whose data is being described?",
        "input_type": "select",
        "options": options,
        "required": True,
        "note": "Please answer based on the scenario details shown above.",
    }

def materialize_survey_item(items: List[Dict[str, Any]], assignment: Dict[str, Any], index: int, total: int) -> Dict[str, Any]:
    base = items[int(assignment["item_index"])]
    flow = base["flow"]
    output_variant = base.get("output_variant")
    method_ids = list(base.get("method_ids") or base.get("baseline_ids") or [])
    method_details = list(base.get("method_details") or base.get("baseline_details") or [])
    baseline_ids = list(base.get("baseline_ids") or method_ids)
    ablation_modes = list(base.get("ablation_modes") or [])
    display_fields = participant_display_fields(flow, output_variant)
    changed_labels = set(assignment.get("changed_field_labels") or [])
    for field in display_fields:
        if field.get("label") in changed_labels:
            field["emphasis"] = "changed"
            field["change_label"] = "Changed"
    attention_check = build_attention_check(base.get("item_id"), base.get("flow_id"), index, display_fields)
    comprehension_check = build_comprehension_check(base.get("item_id"), base.get("flow_id"), index, flow, display_fields)
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
            "output_data_term": (output_variant or {}).get("output_data_term"),
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
        "rating_prompt": "From your perspective as the person the data is about, how appropriate is it for the monitoring device to share the listed data in this situation?",
        "rating_guidance": "Judge appropriateness, not whether the setup is legal or whether the space owner is allowed to use it. Use the notice/permission details in the condition row and the access assumptions in the receiver row.",
        "display_fields": display_fields,
        "attention_check": attention_check,
        "comprehension_check": comprehension_check,
        "output_data_slot": {
            "status": "included_from_generated_pipeline_outputs" if output_variant else "not_included_in_context_only_survey",
            "output_data": (output_variant or {}).get("output_variant_label"),
            "output_variant_id": (output_variant or {}).get("output_variant_id"),
            "output_variant_label": (output_variant or {}).get("output_variant_label"),
            "output_variant_description": (output_variant or {}).get("output_variant_description"),
            "output_data_term": (output_variant or {}).get("output_data_term"),
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


def participant_counts(db_path: Path) -> List[Dict[str, Any]]:
    """Return one row per participant/session with number of answered questions."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sessions = [dict(r) for r in conn.execute("SELECT * FROM sessions ORDER BY created_at_ms ASC").fetchall()]
        counts = {
            r["session_id"]: dict(r)
            for r in conn.execute("""
                SELECT session_id,
                       COUNT(*) AS answered_count,
                       AVG(CASE WHEN attention_check_correct IS NOT NULL THEN attention_check_correct END) AS attention_check_accuracy,
                       AVG(CASE WHEN comprehension_check_correct IS NOT NULL THEN comprehension_check_correct END) AS comprehension_check_accuracy
                FROM responses
                GROUP BY session_id
            """).fetchall()
        }
    out: List[Dict[str, Any]] = []
    for srow in sessions:
        try:
            assigned_count = len(json.loads(srow.get("assignment_json") or "[]"))
        except Exception:
            assigned_count = None
        c = counts.get(srow.get("session_id"), {})
        answered = int(srow.get("answered_count") if srow.get("answered_count") is not None else (c.get("answered_count") or 0))
        completed = bool(assigned_count is not None and answered >= assigned_count)
        out.append({
            "participant_id": srow.get("participant_id"),
            "prolific_pid": srow.get("prolific_pid"),
            "prolific_study_id": srow.get("prolific_study_id"),
            "prolific_session_id": srow.get("prolific_session_id"),
            "session_id": srow.get("session_id"),
            "answered_count": answered,
            "assigned_count": assigned_count,
            "completed": completed,
            "created_at_ms": srow.get("created_at_ms"),
            "completed_at_ms": srow.get("completed_at_ms"),
            "total_elapsed_ms": srow.get("total_elapsed_ms"),
            "total_active_elapsed_ms": srow.get("total_active_elapsed_ms"),
            "attention_check_accuracy": c.get("attention_check_accuracy"),
            "comprehension_check_accuracy": c.get("comprehension_check_accuracy"),
        })
    return out



def summary(db_path: Path, state: SurveyState) -> Dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        session_count = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
        response_count = conn.execute("SELECT COUNT(*) AS c FROM responses").fetchone()["c"]
        by_task = [dict(r) for r in conn.execute("SELECT task, COUNT(*) AS n, AVG(rating) AS avg_rating FROM responses GROUP BY task").fetchall()]
        by_output = [dict(r) for r in conn.execute("SELECT output_variant_id, output_variant_label, COUNT(*) AS n, AVG(rating) AS avg_rating FROM responses GROUP BY output_variant_id, output_variant_label").fetchall()]
        attention = conn.execute("""
            SELECT COUNT(attention_check_correct) AS n,
                   SUM(CASE WHEN attention_check_correct=1 THEN 1 ELSE 0 END) AS correct,
                   AVG(CASE WHEN attention_check_correct IS NOT NULL THEN attention_check_correct END) AS accuracy
            FROM responses
        """).fetchone()
        comprehension = conn.execute("""
            SELECT COUNT(comprehension_check_correct) AS n,
                   SUM(CASE WHEN comprehension_check_correct=1 THEN 1 ELSE 0 END) AS correct,
                   AVG(CASE WHEN comprehension_check_correct IS NOT NULL THEN comprehension_check_correct END) AS accuracy
            FROM responses
        """).fetchone()
        timing = conn.execute("""
            SELECT AVG(total_elapsed_ms) AS avg_total_elapsed_ms,
                   AVG(total_active_elapsed_ms) AS avg_total_active_elapsed_ms,
                   SUM(CASE WHEN completed_at_ms IS NOT NULL THEN 1 ELSE 0 END) AS completed_sessions
            FROM sessions
        """).fetchone()
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
        "attention_check": dict(attention) if attention else {},
        "comprehension_check": dict(comprehension) if comprehension else {},
        "session_timing": dict(timing) if timing else {},
    }
    return out


def _preview_visible_field(field: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the field text that the participant sees in the webpage."""
    out: Dict[str, Any] = {
        "label": field.get("label") or "",
        "value": field.get("value") or "",
    }
    if field.get("value_html"):
        out["value_html"] = field.get("value_html")
    if field.get("emphasis"):
        out["emphasis"] = field.get("emphasis")
    if field.get("change_label"):
        out["change_label"] = field.get("change_label")
    help_text = field.get("help") or field.get("description") or ""
    if help_text:
        out["description"] = help_text
    example = field.get("example")
    if example:
        example_text = str(example).strip()
        if example_text and not example_text.lower().startswith("example:"):
            example_text = f"Example: {example_text}"
        if example_text:
            out["example"] = example_text
    return out


def _preview_visible_attention_check(attention: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Keep only the attention-check text/options visible to participants.

    The expected answer is intentionally omitted from preview.json because it is
    not shown on the webpage.
    """
    if not attention:
        return None
    return {
        "label": attention.get("field_label") or "Attention check",
        "question": attention.get("question") or "Please choose the requested answer.",
        "note": attention.get("note") or "",
        "input_type": attention.get("input_type") or "select",
        "placeholder": "Select an answer",
        "options": list(attention.get("options") or []),
        "required": bool(attention.get("required")),
    }


def _preview_visible_comprehension_check(check: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Keep only the comprehension-check text/options visible to participants."""
    if not check:
        return None
    return {
        "label": check.get("field_label") or "Scenario check",
        "question": check.get("question") or "Please answer based on this scenario.",
        "note": check.get("note") or "",
        "input_type": check.get("input_type") or "select",
        "placeholder": "Select an answer",
        "options": list(check.get("options") or []),
        "required": bool(check.get("required")),
    }


def build_question_preview(
    state: SurveyState,
    participant_id: str,
    session_id: str,
    k: int,
) -> Dict[str, Any]:
    """Build a participant-visible JSON preview of the survey questions.

    This does not create a survey session or save any response rows. It uses the
    same assignment and materialization functions as the live web UI, but it
    deliberately omits hidden metadata such as pipeline IDs, technical output
    caps, method linkage, source files, and attention-check expected answers.
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
        state.config.max_per_scenario_group,
    )
    total = len(assignment)
    rating_scale = [
        {"value": 1, "label": "Completely inappropriate"},
        {"value": 2, "label": "Somewhat inappropriate"},
        {"value": 3, "label": "Neutral / unsure"},
        {"value": 4, "label": "Somewhat appropriate"},
        {"value": 5, "label": "Completely appropriate"},
    ]
    questions: List[Dict[str, Any]] = []
    for i, a in enumerate(assignment):
        item = materialize_survey_item(state.items, a, i, total)
        visible_fields = [_preview_visible_field(f) for f in (item.get("display_fields") or [])]
        q: Dict[str, Any] = {
            "question_number": i + 1,
            "progress_text": f"Question {i + 1} of {total}",
            "vignette": item.get("vignette") or "",
            "scenario_details": visible_fields,
            "rating": {
                "prompt": "From your perspective as the person the data is about, how appropriate is it for the monitoring device to share the listed data in this situation? Please judge appropriateness, not whether the setup is legal or whether the space owner is allowed to use it.",
                "scale": rating_scale,
            },
            "confidence_prompt": "How confident are you in this rating?",
            "optional_note_prompt": "Optional: Explain your reasoning for this scenario",
        }
        visible_attention = _preview_visible_attention_check(item.get("attention_check"))
        if visible_attention:
            q["attention_check"] = visible_attention
        visible_comprehension = _preview_visible_comprehension_check(item.get("comprehension_check"))
        if visible_comprehension:
            q["comprehension_check"] = visible_comprehension
        questions.append(q)
    # Keep preview.json intentionally minimal: only the question content a
    # participant would see on the webpage. Do not include hidden method,
    # pipeline, source, assignment, technical-output, or answer-key metadata.
    return {"questions": questions}

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
                return json_response(self, {
                    "responses": export_rows(state.config.db_path),
                    "participants": participant_counts(state.config.db_path),
                    "summary": summary(state.config.db_path, state),
                })
            if path == "/admin/export.csv":
                return self.export_csv()
            if path == "/admin/participant_counts.json":
                return json_response(self, {"participants": participant_counts(state.config.db_path)})
            if path == "/admin/flows.json":
                return json_response(self, state.flow_data)
            if path == "/admin/survey_items.json":
                return json_response(self, {"summary": state.item_pool_summary, "items": state.items})
            if path == "/admin/pipeline_rows.json":
                return json_response(self, {"pipeline_load_info": state.pipeline_load_info, "rows": state.pipeline_rows})
            if path == "/admin/glossary.json":
                return json_response(self, merged_human_readable_glossary(state.flow_data))
            return json_response(self, {"error": "not found"}, 404)

        def handle_post(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/start":
                payload = read_json_body(self)
                prolific = extract_prolific_metadata(payload)
                participant_code = str(payload.get("participant_code") or prolific.get("prolific_pid") or "").strip()
                # In Prolific, PROLIFIC_PID is the participant identifier.  Keep a
                # manual fallback for local testing or preview links opened outside
                # Prolific.
                if prolific.get("prolific_pid"):
                    participant_code = str(prolific["prolific_pid"]).strip()
                if not participant_code:
                    return json_response(self, {"error": "Prolific ID is required"}, 400)
                k = max(1, min(int(state.config.k), len(state.items)))
                participant_id = participant_code
                session_id = uuid.uuid4().hex
                assignment = assign_items(state.items, k, state.config.seed, participant_id, session_id, state.config.assignment_mode, state.config.db_path, state.config.max_per_scenario_group)
                metadata = dict(payload)
                metadata.update(prolific)
                metadata["requested_k_ignored"] = payload.get("k") if "k" in payload else None
                metadata["assigned_k"] = len(assignment)
                create_session(state.config.db_path, session_id, participant_id, assignment, metadata)
                return json_response(self, {
                    "session_id": session_id,
                    "participant_id": participant_id,
                    "prolific_pid": prolific.get("prolific_pid"),
                    "prolific_study_id": prolific.get("prolific_study_id"),
                    "prolific_session_id": prolific.get("prolific_session_id"),
                    "k": len(assignment),
                    "first_index": 0,
                })
            if path.startswith("/api/session/") and path.endswith("/submit"):
                parts = path.strip("/").split("/")
                if len(parts) != 4:
                    return json_response(self, {"error": "bad submit path"}, 400)
                return self.handle_submit(parts[2])
            return json_response(self, {"error": "not found"}, 404)

        def handle_session_get(self, path: str) -> None:
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[3] == "complete_redirect":
                complete, participant_id = session_is_complete(state.config.db_path, parts[2])
                if not participant_id:
                    return json_response(self, {"error": "unknown session"}, 404)
                if not complete:
                    return json_response(self, {"error": "session is not complete yet"}, 403)
                return redirect_response(self, PROLIFIC_COMPLETION_URL)
            if len(parts) == 4 and parts[3] == "completion_status":
                complete, participant_id = session_is_complete(state.config.db_path, parts[2])
                if not participant_id:
                    return json_response(self, {"error": "unknown session"}, 404)
                return json_response(self, {"completed": complete, "participant_id": participant_id, "redirect_path": f"/api/session/{parts[2]}/complete_redirect" if complete else None})
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
            attention = item.get("attention_check") or {}
            if attention and attention.get("required") and not str(payload.get("attention_check_answer") or "").strip():
                return json_response(self, {"error": "please answer the attention-check question before continuing"}, 400)
            comprehension = item.get("comprehension_check") or {}
            if comprehension and comprehension.get("required") and not str(payload.get("comprehension_check_answer") or "").strip():
                return json_response(self, {"error": "please answer the scenario-check question before continuing"}, 400)
            save_response(state.config.db_path, session_id, session["participant_id"], index, item, rating, payload.get("confidence"), str(payload.get("free_text") or "").strip(), payload.get("elapsed_ms"), payload)
            progress = update_session_progress(state.config.db_path, session_id, len(assignment))
            return json_response(self, {"ok": True, "answered": progress.get("answered_count", 0), "k": len(assignment), "completed": progress.get("completed", False), "total_elapsed_ms": progress.get("total_elapsed_ms"), "total_active_elapsed_ms": progress.get("total_active_elapsed_ms")})

        def serve_file(self, path: Path, content_type: Optional[str] = None) -> None:
            if not path.exists() or not path.is_file():
                return json_response(self, {"error": "file not found"}, 404)
            ctype = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            if path == ROOT / "templates" / "index.html":
                template = path.read_text(encoding="utf-8")
                replacements = {
                    "{{STUDY_CONTACT_NAME}}": state.config.study_contact_name,
                    "{{STUDY_CONTACT_EMAIL}}": state.config.study_contact_email,
                    "{{RIGHTS_CONTACT_NAME}}": state.config.rights_contact_name,
                    "{{RIGHTS_CONTACT_EMAIL}}": state.config.rights_contact_email,
                    "{{RIGHTS_CONTACT_PHONE}}": state.config.rights_contact_phone,
                }
                for token, value in replacements.items():
                    template = template.replace(token, html.escape(str(value)))
                data = template.encode("utf-8")
            else:
                data = path.read_bytes()
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
                "attention_check_field", "attention_check_prompt", "attention_check_expected",
                "attention_check_answer", "attention_check_correct",
                "comprehension_check_field", "comprehension_check_prompt", "comprehension_check_expected",
                "comprehension_check_answer", "comprehension_check_correct",
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
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--flow-file", default=str(DEFAULT_FLOW_FILE))
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--assignment-mode", default="least_rated_balanced", choices=["least_rated_balanced", "sequential"])
    p.add_argument("--max-per-scenario-group", type=int, default=2, help="Maximum number of versions from the same scenario family shown to one participant. Use 0 for no cap.")
    p.add_argument("--pipeline-output-dir", default=str(DEFAULT_PIPELINE_OUTPUT_DIR), help="Flexible run directory created by evaluation.generate_pipelines_for_all_contexts, containing index.json/summary.json, or a summary_by_context JSON file.")
    p.add_argument("--previous-pipeline-output-dir", default=str(DEFAULT_PREVIOUS_PIPELINE_OUTPUT_DIR), help="Earlier fixed-schema run directory or summary_by_context JSON file. Used only when --survey-output-scope=new_flexible_only.")
    p.add_argument("--survey-output-scope", default="new_flexible_only", choices=["new_flexible_only", "all"], help="Use new_flexible_only for a supplemental survey that asks only about genuinely new flexible output data types, not internal schema renames of previously surveyed media outputs.")
    p.add_argument("--supplemental-output-schemas", default="", help="Optional comma-separated override for output schemas to include in new_flexible_only mode. Normally omit; default excludes old media concepts such as redacted_video_stream.")
    p.add_argument("--no-pipeline-outputs", action="store_true", help="Fall back to context-only survey items.")
    p.add_argument("--include-no-output-variants", action="store_true", help="Also create survey cases for baselines that deny or produce no selected output.")
    p.add_argument("--study-contact-name", default="[add name]", help="Name displayed as the study contact; defaults to a non-identifying placeholder.")
    p.add_argument("--study-contact-email", default="[add email]", help="Email displayed for the study contact; defaults to a non-identifying placeholder.")
    p.add_argument("--rights-contact-name", default="[add office or contact]", help="Name displayed for participant-rights questions.")
    p.add_argument("--rights-contact-email", default="[add email]", help="Email displayed for participant-rights questions.")
    p.add_argument("--rights-contact-phone", default="[add phone]", help="Phone number displayed for participant-rights questions.")
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
        args.max_per_scenario_group,
        pipeline_dir,
        None if args.no_pipeline_outputs else Path(args.previous_pipeline_output_dir),
        include_pipeline_outputs=not args.no_pipeline_outputs,
        include_no_output_variants=args.include_no_output_variants,
        survey_output_scope=args.survey_output_scope,
        supplemental_output_schemas=_parse_csv_terms(args.supplemental_output_schemas),
        study_contact_name=args.study_contact_name,
        study_contact_email=args.study_contact_email,
        rights_contact_name=args.rights_contact_name,
        rights_contact_email=args.rights_contact_email,
        rights_contact_phone=args.rights_contact_phone,
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

    print(f"Serving {len(state.flows)} non-child context scenarios from {state.config.flow_file}")
    if state.item_pool_summary.get("excluded_child_related_context_count"):
        print(f"Excluded child-related scenarios: {state.item_pool_summary.get('excluded_child_related_context_ids')}")
    print(f"Survey item pool count: {len(state.items)}")
    if state.item_pool_summary.get("output_augmented"):
        print(f"Loaded generated pipeline outputs from: {state.pipeline_load_info.get('source') or state.pipeline_load_info.get('pipeline_output_dir')}")
        print(f"Baseline pipeline rows: {state.item_pool_summary.get('baseline_pipeline_row_count')}")
        print(f"Pipeline rows with shared output: {state.item_pool_summary.get('baseline_pipeline_rows_with_output')}")
        print(f"Unique context-output survey cases after deduplication: {state.item_pool_summary.get('deduplicated_output_variant_count')}")
        print(f"Survey output scope: {state.item_pool_summary.get('survey_output_scope')}")
        if state.item_pool_summary.get("target_output_schemas"):
            print(f"Target output schemas: {state.item_pool_summary.get('target_output_schemas')}")
            print(f"Pipeline rows excluded because their output schema was already covered: {state.item_pool_summary.get('method_pipeline_rows_excluded_by_output_schema')}")
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
