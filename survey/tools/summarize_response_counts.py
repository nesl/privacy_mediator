#!/usr/bin/env python3
"""Summarize completed survey responses by participant and scenario.

Default behavior:
  * reads survey/outputs/responses.db
  * ignores incomplete sessions
  * writes a structured JSON report to survey/outputs/response_summary_by_scenario.json

Examples:
  python summarize_response_counts.py --db survey/outputs/responses.db
  python summarize_response_counts.py --db survey/outputs/responses.db --out survey/outputs/summary.json
  python summarize_response_counts.py --db survey/outputs/responses.db --include-incomplete --out all_sessions.json
  python summarize_response_counts.py --db survey/outputs/responses.db --context-file survey/data/ci_focused_user_study_context.json

The JSON contains:
  * summary: overall completion and quality-control counts
  * participants: one row per included session
  * scenarios: one row per full survey item, including all ratings, average rating,
    rating counts, and free-text comments from completed participants
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_OUT = "survey/outputs/response_summary_by_scenario.json"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def column_names(conn: sqlite3.Connection, table: str) -> Set[str]:
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return set()


def table_names(conn: sqlite3.Connection) -> Set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}


def parse_json(value: Any, default: Any = None) -> Any:
    if default is None:
        default = {}
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def ms_to_seconds(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return round(float(value) / 1000.0, 3)
    except Exception:
        return None


def format_ms(value: Any) -> str:
    try:
        if value is None:
            return ""
        ms = int(float(value))
    except Exception:
        return ""
    if ms < 0:
        return ""
    total_seconds = int(round(ms / 1000.0))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        x = float(value)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def mean_or_none(values: Sequence[Any], ndigits: int = 3) -> Optional[float]:
    xs = [safe_float(v) for v in values]
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return round(sum(xs) / len(xs), ndigits)


def median_or_none(values: Sequence[Any], ndigits: int = 3) -> Optional[float]:
    xs = [safe_float(v) for v in values]
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return round(float(statistics.median(xs)), ndigits)


def stdev_or_none(values: Sequence[Any], ndigits: int = 3) -> Optional[float]:
    xs = [safe_float(v) for v in values]
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    return round(float(statistics.stdev(xs)), ndigits)


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def json_len(value: Any) -> Optional[int]:
    try:
        return len(parse_json(value, []))
    except Exception:
        return None


def as_bool_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return 1 if int(value) == 1 else 0
    except Exception:
        text = str(value).strip().lower()
        if text in {"true", "yes", "correct", "1"}:
            return 1
        if text in {"false", "no", "incorrect", "0"}:
            return 0
    return None


def pct(numer: int, denom: int) -> Optional[float]:
    if denom <= 0:
        return None
    return round(100.0 * numer / denom, 1)


# ---------------------------------------------------------------------------
# Context metadata / item extraction
# ---------------------------------------------------------------------------


def load_context_map(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Load optional context-scenario metadata keyed by scenario_id and original_flow_id."""
    if not path:
        return {}
    if not path.exists():
        raise SystemExit(f"Context file not found: {path}")
    data = parse_json(path.read_text(encoding="utf-8"), {})
    flows = data.get("context_scenarios") or data.get("generated_information_flows") or []
    out: Dict[str, Dict[str, Any]] = {}
    for f in flows:
        if not isinstance(f, dict):
            continue
        for k in (f.get("scenario_id"), f.get("original_flow_id"), f.get("flow_id")):
            if k:
                out[str(k)] = f
        scalar = f.get("ci_parameters_scalar_context_only") or {}
        for k in (scalar.get("scenario_id"), scalar.get("original_flow_id"), scalar.get("flow_id")):
            if k:
                out[str(k)] = f
    return out


def get_flow_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    flow = item.get("flow")
    if isinstance(flow, dict):
        return flow
    # Some older previews use a top-level flow-like structure.
    return item if isinstance(item, dict) else {}


def nested_get(d: Dict[str, Any], path: Sequence[str]) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def extract_scenario_id(flow: Dict[str, Any], row: Dict[str, Any]) -> Optional[str]:
    candidates = [
        flow.get("scenario_id"),
        flow.get("context_scenario_id"),
        flow.get("flow_id"),
        row.get("flow_id"),
        nested_get(flow, ["ci_parameters_scalar_context_only", "scenario_id"]),
        nested_get(flow, ["ci_parameters_readable_context_only", "scenario_id"]),
    ]
    for c in candidates:
        if c:
            return str(c)
    return None


def visible_fields(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    fields = item.get("scenario_details")
    if isinstance(fields, list):
        return [f for f in fields if isinstance(f, dict)]
    fields = item.get("participant_display_fields")
    if isinstance(fields, list):
        return [f for f in fields if isinstance(f, dict)]
    flow = get_flow_from_item(item)
    fields = flow.get("participant_display_fields")
    if isinstance(fields, list):
        return [f for f in fields if isinstance(f, dict)]
    return []


def field_by_label(item: Dict[str, Any], labels: Iterable[str]) -> Dict[str, Any]:
    label_set = {x.lower() for x in labels}
    for f in visible_fields(item):
        label = str(f.get("label") or "").lower()
        if label in label_set:
            return f
    return {}


def field_value(item: Dict[str, Any], labels: Iterable[str]) -> Optional[str]:
    f = field_by_label(item, labels)
    value = f.get("value") if f else None
    return clean_text(value) or None


def field_description(item: Dict[str, Any], labels: Iterable[str]) -> Optional[str]:
    f = field_by_label(item, labels)
    pieces = []
    for k in ("description", "example"):
        text = clean_text(f.get(k)) if f else ""
        if text and text not in pieces:
            pieces.append(text)
    return " ".join(pieces) if pieces else None


def context_readable(flow: Dict[str, Any], context_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    scenario_id = flow.get("scenario_id") or nested_get(flow, ["ci_parameters_scalar_context_only", "scenario_id"])
    ctx = context_map.get(str(scenario_id)) if scenario_id else None
    source = ctx or flow
    scalar = source.get("ci_parameters_scalar_context_only") or {}
    readable = source.get("ci_parameters_readable_context_only") or {}
    situation = readable.get("situation") or {}
    task = source.get("task") or scalar.get("task") or nested_get(readable, ["task", "term"])
    task_label = source.get("task_label") or nested_get(readable, ["task", "label"])
    return {
        "scenario_id": scenario_id or source.get("scenario_id") or scalar.get("scenario_id"),
        "task": task,
        "task_label": task_label,
        "context_family": source.get("context_family"),
        "family_id": source.get("family_id"),
        "context": scalar.get("context"),
        "space": scalar.get("space"),
        "sender": scalar.get("sender"),
        "subject": scalar.get("subject"),
        "recipient": scalar.get("recipient"),
        "purpose": scalar.get("purpose"),
        "transmission_principle": scalar.get("transmission_principle"),
        "situation_label": situation.get("label"),
    }


def scenario_key_for(row: Dict[str, Any], item: Dict[str, Any], flow: Dict[str, Any], context_map: Dict[str, Dict[str, Any]]) -> str:
    """Return a stable key for a full survey item, not just the context shell."""
    item_id = row.get("item_id") or flow.get("item_id")
    if item_id:
        return str(item_id)
    scenario_id = extract_scenario_id(flow, row)
    output_variant_id = row.get("output_variant_id") or flow.get("output_variant_id") or "no_output_variant"
    if scenario_id:
        return f"{scenario_id}|{output_variant_id}"
    # Last resort: use participant-visible text so identical pages collapse together.
    visible_sig = {
        "scenario": field_value(item, ["What is the scenario?", "Situation"]),
        "role": field_value(item, ["What is your role in this scenario?", "Data subject"]),
        "data": field_value(item, ["What data would be shared?", "Output"]),
        "receiver": field_value(item, ["Who receives or uses the shared data?", "Recipient"]),
    }
    return "visible:" + json.dumps(visible_sig, sort_keys=True)


def scenario_metadata(row: Dict[str, Any], item: Dict[str, Any], context_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    flow = get_flow_from_item(item)
    scenario_id = extract_scenario_id(flow, row)
    ctx_info = context_readable({**flow, "scenario_id": scenario_id}, context_map)
    output_variant_id = row.get("output_variant_id") or flow.get("output_variant_id")
    output_variant_label = row.get("output_variant_label") or flow.get("output_variant_label")
    data_value = field_value(item, ["What data would be shared?", "Output", "Data shared"])
    data_description = field_description(item, ["What data would be shared?", "Output", "Data shared"])
    scenario_text = field_value(item, ["What is the scenario?", "Situation"])
    role_text = field_value(item, ["What is your role in this scenario?", "Data subject"])
    receiver_text = field_value(item, ["Who receives or uses the shared data?", "Recipient"])
    controller_text = field_value(item, ["Who controls the monitoring device?", "Sender"])
    processing_text = field_value(item, ["Where is the data processed?"])
    condition_text = None
    for label in [
        "What notice or permission is given?",
        "When is data collected or shared?",
        "Who is allowed to access the shared data?",
        "What rule applies to the data?",
        "What audio filtering happens before sharing?",
    ]:
        condition_text = field_value(item, [label])
        if condition_text:
            break

    return {
        "scenario_key": scenario_key_for(row, item, flow, context_map),
        "scenario_id": scenario_id or ctx_info.get("scenario_id"),
        "flow_id": row.get("flow_id") or flow.get("flow_id"),
        "item_id": row.get("item_id") or flow.get("item_id"),
        "task": row.get("task") or flow.get("task") or ctx_info.get("task"),
        "task_label": ctx_info.get("task_label"),
        "context_family": row.get("context_family") or flow.get("context_family") or ctx_info.get("context_family"),
        "family_id": flow.get("family_id") or ctx_info.get("family_id"),
        "context": ctx_info.get("context"),
        "space": ctx_info.get("space"),
        "sender": ctx_info.get("sender"),
        "subject": ctx_info.get("subject"),
        "recipient": ctx_info.get("recipient"),
        "purpose": ctx_info.get("purpose"),
        "transmission_principle": ctx_info.get("transmission_principle"),
        "situation_label": ctx_info.get("situation_label"),
        "output_variant_id": output_variant_id,
        "output_variant_label": output_variant_label,
        "variant_privacy_class": row.get("variant_privacy_class") or flow.get("variant_privacy_class"),
        "final_output_type": row.get("final_output_type"),
        "final_output_schema": row.get("final_output_schema"),
        "scenario_text": scenario_text,
        "role_text": role_text,
        "data_shared": data_value,
        "data_description": data_description,
        "controller_text": controller_text,
        "receiver_text": receiver_text,
        "processing_text": processing_text,
        "condition_text": condition_text,
    }


# ---------------------------------------------------------------------------
# Loading sessions and responses
# ---------------------------------------------------------------------------


def load_sessions_and_responses(db_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Set[str], Set[str]]:
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        tables = table_names(conn)
        if "sessions" not in tables or "responses" not in tables:
            raise SystemExit(f"Database does not contain expected sessions/responses tables: {db_path}")
        session_cols = column_names(conn, "sessions")
        response_cols = column_names(conn, "responses")
        sessions = [dict(r) for r in conn.execute("SELECT * FROM sessions ORDER BY created_at_ms ASC").fetchall()]
        responses = [dict(r) for r in conn.execute("SELECT * FROM responses ORDER BY session_id ASC, item_index ASC").fetchall()]
    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in responses:
        by_session[str(r.get("session_id"))].append(r)
    return sessions, by_session, session_cols, response_cols


def summarize_session(srow: Dict[str, Any], responses: List[Dict[str, Any]], session_cols: Set[str]) -> Dict[str, Any]:
    assignment = parse_json(srow.get("assignment_json"), [])
    assigned_count = len(assignment) if isinstance(assignment, list) else json_len(srow.get("assignment_json"))
    response_count = len(responses)
    answered_col = srow.get("answered_count") if "answered_count" in session_cols else None
    answered_count = max(safe_int(answered_col, 0), response_count)
    completed_at = srow.get("completed_at_ms") if "completed_at_ms" in session_cols else None
    completed = bool(completed_at is not None or (assigned_count is not None and answered_count >= assigned_count))

    created_at = srow.get("created_at_ms")
    last_activity = srow.get("last_activity_at_ms") if "last_activity_at_ms" in session_cols else None
    last_response = max([safe_int(r.get("created_at_ms"), 0) for r in responses] or [0]) or None
    total_elapsed_ms = srow.get("total_elapsed_ms") if "total_elapsed_ms" in session_cols else None
    if total_elapsed_ms is None and created_at is not None:
        end = completed_at or last_activity or last_response
        if end is not None:
            total_elapsed_ms = max(0, safe_int(end) - safe_int(created_at))
    total_active_elapsed_ms = srow.get("total_active_elapsed_ms") if "total_active_elapsed_ms" in session_cols else None
    if total_active_elapsed_ms is None:
        vals = [safe_int(r.get("elapsed_ms"), 0) for r in responses if r.get("elapsed_ms") is not None]
        total_active_elapsed_ms = sum(vals) if vals else None

    attention_rows = [r for r in responses if r.get("attention_check_correct") is not None or r.get("attention_check_answer")]
    attention_correct = sum(1 for r in attention_rows if as_bool_int(r.get("attention_check_correct")) == 1)
    attention_failed = sum(1 for r in attention_rows if as_bool_int(r.get("attention_check_correct")) == 0)
    comprehension_rows = [r for r in responses if r.get("comprehension_check_correct") is not None or r.get("comprehension_check_answer")]
    comprehension_correct = sum(1 for r in comprehension_rows if as_bool_int(r.get("comprehension_check_correct")) == 1)
    comprehension_failed = sum(1 for r in comprehension_rows if as_bool_int(r.get("comprehension_check_correct")) == 0)
    optional_response_count = sum(1 for r in responses if clean_text(r.get("free_text")))

    return {
        "participant_id": srow.get("participant_id"),
        "session_id": srow.get("session_id"),
        "answered_count": answered_count,
        "assigned_count": assigned_count,
        "completed": completed,
        "included": None,  # filled after filtering
        "created_at_ms": created_at,
        "completed_at_ms": completed_at,
        "last_activity_at_ms": last_activity,
        "last_response_at_ms": last_response,
        "total_elapsed_ms": total_elapsed_ms,
        "total_elapsed_seconds": ms_to_seconds(total_elapsed_ms),
        "total_elapsed_human": format_ms(total_elapsed_ms),
        "total_active_elapsed_ms": total_active_elapsed_ms,
        "total_active_elapsed_seconds": ms_to_seconds(total_active_elapsed_ms),
        "total_active_elapsed_human": format_ms(total_active_elapsed_ms),
        "attention_checks_answered": len(attention_rows),
        "attention_checks_correct": attention_correct,
        "attention_checks_failed": attention_failed,
        "attention_check_accuracy": None if not attention_rows else round(attention_correct / len(attention_rows), 3),
        "attention_check_passed": None if not attention_rows else attention_failed == 0,
        "attention_check_failed": attention_failed > 0,
        "comprehension_checks_answered": len(comprehension_rows),
        "comprehension_checks_correct": comprehension_correct,
        "comprehension_checks_failed": comprehension_failed,
        "comprehension_check_accuracy": None if not comprehension_rows else round(comprehension_correct / len(comprehension_rows), 3),
        "comprehension_check_passed": None if not comprehension_rows else comprehension_failed == 0,
        "comprehension_check_failed": comprehension_failed > 0,
        "optional_response_count": optional_response_count,
    }


# ---------------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------------


def row_rating(row: Dict[str, Any]) -> Optional[int]:
    try:
        if row.get("rating") is None:
            return None
        return int(row.get("rating"))
    except Exception:
        return None


def build_scenario_groups(
    included_sessions: Set[str],
    by_session: Dict[str, List[Dict[str, Any]]],
    context_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for sid in sorted(included_sessions):
        for row in by_session.get(sid, []):
            rating = row_rating(row)
            if rating is None:
                continue
            item = parse_json(row.get("item_json"), {})
            if not isinstance(item, dict):
                item = {}
            meta = scenario_metadata(row, item, context_map)
            key = meta["scenario_key"]
            if key not in groups:
                groups[key] = {**meta, "ratings": [], "comments": []}
            participant_id = row.get("participant_id")
            record = {
                "participant_id": participant_id,
                "session_id": row.get("session_id"),
                "item_index": row.get("item_index"),
                "question_number": (safe_int(row.get("item_index"), -1) + 1) if row.get("item_index") is not None else None,
                "rating": rating,
                "confidence": row.get("confidence"),
                "elapsed_ms": row.get("elapsed_ms"),
                "created_at_ms": row.get("created_at_ms"),
                "attention_check_correct": as_bool_int(row.get("attention_check_correct")),
                "comprehension_check_correct": as_bool_int(row.get("comprehension_check_correct")),
            }
            groups[key]["ratings"].append(record)
            comment = clean_text(row.get("free_text"))
            if comment:
                groups[key]["comments"].append({
                    "participant_id": participant_id,
                    "session_id": row.get("session_id"),
                    "item_index": row.get("item_index"),
                    "question_number": record["question_number"],
                    "rating": rating,
                    "confidence": row.get("confidence"),
                    "comment": comment,
                })

    out = []
    for key, g in groups.items():
        values = [r["rating"] for r in g["ratings"] if r.get("rating") is not None]
        counts = Counter(values)
        g["rating_count"] = len(values)
        g["average_rating"] = mean_or_none(values)
        g["median_rating"] = median_or_none(values)
        g["stdev_rating"] = stdev_or_none(values)
        g["rating_counts"] = {str(k): counts.get(k, 0) for k in range(1, 6)}
        g["comment_count"] = len(g["comments"])
        out.append(g)

    def sort_key(g: Dict[str, Any]) -> Tuple[str, str, str, str]:
        return (
            str(g.get("task") or ""),
            str(g.get("scenario_id") or ""),
            str(g.get("output_variant_id") or ""),
            str(g.get("scenario_key") or ""),
        )

    out.sort(key=sort_key)
    return out


def build_summary(
    participant_rows_all: List[Dict[str, Any]],
    participant_rows_included: List[Dict[str, Any]],
    scenarios: List[Dict[str, Any]],
    include_incomplete: bool,
) -> Dict[str, Any]:
    completed = [p for p in participant_rows_all if p.get("completed")]
    incomplete = [p for p in participant_rows_all if not p.get("completed")]
    included = participant_rows_included
    all_ratings = [r["rating"] for s in scenarios for r in s.get("ratings", []) if r.get("rating") is not None]

    attn_answered = sum(safe_int(p.get("attention_checks_answered"), 0) for p in included)
    attn_correct = sum(safe_int(p.get("attention_checks_correct"), 0) for p in included)
    attn_failed_items = sum(safe_int(p.get("attention_checks_failed"), 0) for p in included)
    comp_answered = sum(safe_int(p.get("comprehension_checks_answered"), 0) for p in included)
    comp_correct = sum(safe_int(p.get("comprehension_checks_correct"), 0) for p in included)
    comp_failed_items = sum(safe_int(p.get("comprehension_checks_failed"), 0) for p in included)

    unique_participants = {str(p.get("participant_id")) for p in included if p.get("participant_id")}
    completed_unique_participants = {str(p.get("participant_id")) for p in completed if p.get("participant_id")}

    return {
        "generated_at_unix_ms": int(time.time() * 1000),
        "default_filter": "completed_sessions_only" if not include_incomplete else "all_sessions_included",
        "total_sessions_surveyed": len(participant_rows_all),
        "total_unique_participant_ids_surveyed": len({str(p.get("participant_id")) for p in participant_rows_all if p.get("participant_id")}),
        "completed_sessions": len(completed),
        "completed_unique_participant_ids": len(completed_unique_participants),
        "incomplete_sessions": len(incomplete),
        "included_sessions": len(included),
        "included_unique_participant_ids": len(unique_participants),
        "excluded_incomplete_sessions": 0 if include_incomplete else len(incomplete),
        "scenario_count": len(scenarios),
        "rating_count": len(all_ratings),
        "average_rating_over_all_included_responses": mean_or_none(all_ratings),
        "median_rating_over_all_included_responses": median_or_none(all_ratings),
        "rating_counts_over_all_included_responses": {str(k): Counter(all_ratings).get(k, 0) for k in range(1, 6)},
        "optional_comment_count": sum(safe_int(p.get("optional_response_count"), 0) for p in included),
        "average_total_elapsed_seconds_completed": mean_or_none([p.get("total_elapsed_seconds") for p in completed]),
        "average_total_active_elapsed_seconds_completed": mean_or_none([p.get("total_active_elapsed_seconds") for p in completed]),
        "attention_checks": {
            "answered": attn_answered,
            "correct": attn_correct,
            "failed_items": attn_failed_items,
            "item_accuracy_percent": pct(attn_correct, attn_answered),
            "sessions_with_any_failure": sum(1 for p in included if p.get("attention_check_failed")),
            "sessions_with_all_passed": sum(1 for p in included if p.get("attention_check_passed") is True),
            "sessions_with_no_attention_checks": sum(1 for p in included if safe_int(p.get("attention_checks_answered"), 0) == 0),
        },
        "comprehension_checks": {
            "answered": comp_answered,
            "correct": comp_correct,
            "failed_items": comp_failed_items,
            "item_accuracy_percent": pct(comp_correct, comp_answered),
            "sessions_with_any_failure": sum(1 for p in included if p.get("comprehension_check_failed")),
            "sessions_with_all_passed": sum(1 for p in included if p.get("comprehension_check_passed") is True),
            "sessions_with_no_comprehension_checks": sum(1 for p in included if safe_int(p.get("comprehension_checks_answered"), 0) == 0),
        },
    }


def build_report(db_path: Path, context_file: Optional[Path], include_incomplete: bool) -> Dict[str, Any]:
    context_map = load_context_map(context_file)
    sessions, by_session, session_cols, response_cols = load_sessions_and_responses(db_path)
    participants_all: List[Dict[str, Any]] = []
    for s in sessions:
        sid = str(s.get("session_id"))
        participants_all.append(summarize_session(s, by_session.get(sid, []), session_cols))

    included_sessions = {
        str(p.get("session_id")) for p in participants_all
        if p.get("session_id") and (include_incomplete or p.get("completed"))
    }
    participants_included = []
    for p in participants_all:
        p = dict(p)
        p["included"] = str(p.get("session_id")) in included_sessions
        if p["included"]:
            participants_included.append(p)

    scenarios = build_scenario_groups(included_sessions, by_session, context_map)
    summary = build_summary(participants_all, participants_included, scenarios, include_incomplete)
    return {
        "source_db": str(db_path),
        "context_file": str(context_file) if context_file else None,
        "filters": {
            "include_incomplete": include_incomplete,
            "included_sessions": "all" if include_incomplete else "completed_only",
        },
        "summary": summary,
        "participants": participants_included,
        "scenarios": scenarios,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write a JSON summary of survey results. By default, incomplete sessions are ignored, "
            "and ratings/comments are grouped by full scenario item."
        )
    )
    parser.add_argument("--db", default="survey/outputs/responses.db", help="Path to responses.db")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output JSON path. Default: {DEFAULT_OUT}")
    parser.add_argument("--context-file", default=None, help="Optional context JSON to enrich scenario metadata.")
    parser.add_argument("--include-incomplete", action="store_true", help="Include incomplete sessions. Default is completed sessions only.")
    parser.add_argument("--stdout", action="store_true", help="Also print the JSON report to stdout.")
    args = parser.parse_args()

    db_path = Path(args.db)
    out_path = Path(args.out)
    context_file = Path(args.context_file) if args.context_file else None
    report = build_report(db_path, context_file, include_incomplete=bool(args.include_incomplete))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    if args.stdout:
        print(json.dumps(report, indent=2, sort_keys=False))
    else:
        print(f"Wrote JSON summary to {out_path}")
        print(
            f"Included {report['summary']['included_sessions']} completed session(s), "
            f"excluded {report['summary']['excluded_incomplete_sessions']} incomplete session(s), "
            f"summarized {report['summary']['scenario_count']} scenario item(s)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
