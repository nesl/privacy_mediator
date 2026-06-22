#!/usr/bin/env python3
"""Summarize survey response counts, timing, and attention checks by participant/session.

Usage:
  python survey/tools/summarize_response_counts.py --db survey/outputs/responses.db
  python survey/tools/summarize_response_counts.py --db survey/outputs/responses.db --json
  python survey/tools/summarize_response_counts.py --db survey/outputs/responses.db --csv

The text output shows participant/session IDs, number of answered questions,
completion status, wall-clock time spent on the survey, browser-reported active
question time when available, attention-check accuracy when available, and one
sample optional free-text response when a participant provided one.

Timing fields:
  total_elapsed_ms        Wall-clock time from session creation to last submit or completion.
  total_active_elapsed_ms Sum of per-question elapsed_ms values reported by the browser.

For older databases that do not yet have the session-level timing columns, this
script derives approximate timing from response timestamps and per-question
elapsed_ms values when possible.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def column_names(conn: sqlite3.Connection, table: str) -> Set[str]:
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return set()


def ms_to_seconds(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return round(float(value) / 1000.0, 3)
    except Exception:
        return None


def format_ms(value: Any) -> str:
    """Return a compact human-readable duration from milliseconds."""
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


def format_accuracy(value: Any) -> str:
    try:
        if value is None:
            return ""
        return f"{100.0 * float(value):.0f}%"
    except Exception:
        return ""


def _json_len(value: Any) -> Optional[int]:
    try:
        return len(json.loads(value or "[]"))
    except Exception:
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def clean_optional_text(value: Any) -> str:
    """Collapse whitespace in a participant's optional free-text response."""
    return " ".join(str(value or "").strip().split())


def truncate_text(value: Any, limit: int = 220) -> str:
    """Return a compact preview suitable for the text table."""
    text = clean_optional_text(value)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _select_existing_columns(response_cols: Set[str], preferred: List[str]) -> List[str]:
    return [c for c in preferred if c in response_cols]


def load_counts(db_path: Path) -> List[Dict[str, Any]]:
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "sessions" not in tables or "responses" not in tables:
            raise SystemExit(f"Database does not contain expected sessions/responses tables: {db_path}")

        session_cols = column_names(conn, "sessions")
        response_cols = column_names(conn, "responses")
        sessions = [dict(r) for r in conn.execute("SELECT * FROM sessions ORDER BY created_at_ms ASC").fetchall()]

        attention_expr = (
            "AVG(CASE WHEN attention_check_correct IS NOT NULL THEN attention_check_correct END)"
            if "attention_check_correct" in response_cols
            else "NULL"
        )
        response_created_min = "MIN(created_at_ms)" if "created_at_ms" in response_cols else "NULL"
        response_created_max = "MAX(created_at_ms)" if "created_at_ms" in response_cols else "NULL"
        response_elapsed_sum = "SUM(elapsed_ms)" if "elapsed_ms" in response_cols else "NULL"
        attention_answered_expr = (
            "SUM(CASE WHEN attention_check_answer IS NOT NULL AND attention_check_answer != '' THEN 1 ELSE 0 END)"
            if "attention_check_answer" in response_cols
            else "NULL"
        )
        attention_correct_expr = (
            "SUM(CASE WHEN attention_check_correct = 1 THEN 1 ELSE 0 END)"
            if "attention_check_correct" in response_cols
            else "NULL"
        )
        optional_response_count_expr = (
            "SUM(CASE WHEN free_text IS NOT NULL AND TRIM(free_text) != '' THEN 1 ELSE 0 END)"
            if "free_text" in response_cols
            else "NULL"
        )

        counts = {
            r["session_id"]: dict(r)
            for r in conn.execute(f"""
                SELECT session_id,
                       COUNT(*) AS answered_count_from_responses,
                       {response_created_min} AS first_response_at_ms,
                       {response_created_max} AS last_response_at_ms,
                       {response_elapsed_sum} AS summed_response_elapsed_ms,
                       {attention_expr} AS attention_check_accuracy,
                       {attention_answered_expr} AS attention_checks_answered,
                       {attention_correct_expr} AS attention_checks_correct,
                       {optional_response_count_expr} AS optional_response_count
                FROM responses
                GROUP BY session_id
            """).fetchall()
        }

        optional_samples: Dict[str, Dict[str, Any]] = {}
        if "free_text" in response_cols:
            preferred_cols = [
                "session_id", "item_index", "flow_id", "task", "context_family",
                "output_variant_id", "output_variant_label", "variant_privacy_class",
                "rating", "confidence", "free_text", "created_at_ms",
            ]
            sample_cols = _select_existing_columns(response_cols, preferred_cols)
            order_col = "created_at_ms" if "created_at_ms" in response_cols else "item_index"
            sample_sql = f"""
                SELECT {', '.join(sample_cols)}
                FROM responses
                WHERE free_text IS NOT NULL AND TRIM(free_text) != ''
                ORDER BY session_id ASC, {order_col} ASC
            """
            for raw in conn.execute(sample_sql).fetchall():
                sample = dict(raw)
                sid = sample.get("session_id")
                if sid not in optional_samples:
                    sample["free_text"] = clean_optional_text(sample.get("free_text"))
                    optional_samples[str(sid)] = sample

    out: List[Dict[str, Any]] = []
    for srow in sessions:
        session_id = srow.get("session_id")
        c = counts.get(session_id, {})
        assigned_count = _json_len(srow.get("assignment_json"))

        answered = _safe_int(
            srow.get("answered_count") if "answered_count" in session_cols else None,
            default=_safe_int(c.get("answered_count_from_responses"), 0),
        )
        # If the session column is stale or absent, trust the response table count.
        answered = max(answered, _safe_int(c.get("answered_count_from_responses"), 0))
        completed = bool(assigned_count is not None and answered >= assigned_count)

        created_at = srow.get("created_at_ms")
        completed_at = srow.get("completed_at_ms") if "completed_at_ms" in session_cols else None
        last_activity = srow.get("last_activity_at_ms") if "last_activity_at_ms" in session_cols else None
        last_response = c.get("last_response_at_ms")

        total_elapsed_ms = srow.get("total_elapsed_ms") if "total_elapsed_ms" in session_cols else None
        if total_elapsed_ms is None and created_at is not None:
            end = completed_at or last_activity or last_response
            if end is not None:
                total_elapsed_ms = max(0, _safe_int(end) - _safe_int(created_at))

        total_active_elapsed_ms = srow.get("total_active_elapsed_ms") if "total_active_elapsed_ms" in session_cols else None
        if total_active_elapsed_ms is None:
            total_active_elapsed_ms = c.get("summed_response_elapsed_ms")

        attention_accuracy = c.get("attention_check_accuracy")
        attention_answered = c.get("attention_checks_answered")
        attention_correct = c.get("attention_checks_correct")
        optional_response_count = c.get("optional_response_count")
        sample = optional_samples.get(str(session_id)) if session_id is not None else None

        row = {
            "participant_id": srow.get("participant_id"),
            "session_id": session_id,
            "answered_count": answered,
            "assigned_count": assigned_count,
            "completed": completed,
            "created_at_ms": created_at,
            "completed_at_ms": completed_at,
            "last_activity_at_ms": last_activity,
            "first_response_at_ms": c.get("first_response_at_ms"),
            "last_response_at_ms": last_response,
            "total_elapsed_ms": total_elapsed_ms,
            "total_elapsed_seconds": ms_to_seconds(total_elapsed_ms),
            "total_elapsed_human": format_ms(total_elapsed_ms),
            "total_active_elapsed_ms": total_active_elapsed_ms,
            "total_active_elapsed_seconds": ms_to_seconds(total_active_elapsed_ms),
            "total_active_elapsed_human": format_ms(total_active_elapsed_ms),
            "attention_checks_answered": attention_answered,
            "attention_checks_correct": attention_correct,
            "attention_check_accuracy": attention_accuracy,
            "optional_response_count": optional_response_count,
            "sample_optional_item_index": sample.get("item_index") if sample else None,
            "sample_optional_flow_id": sample.get("flow_id") if sample else None,
            "sample_optional_task": sample.get("task") if sample else None,
            "sample_optional_context_family": sample.get("context_family") if sample else None,
            "sample_optional_output_variant_id": sample.get("output_variant_id") if sample else None,
            "sample_optional_output_variant_label": sample.get("output_variant_label") if sample else None,
            "sample_optional_variant_privacy_class": sample.get("variant_privacy_class") if sample else None,
            "sample_optional_rating": sample.get("rating") if sample else None,
            "sample_optional_confidence": sample.get("confidence") if sample else None,
            "sample_optional_created_at_ms": sample.get("created_at_ms") if sample else None,
            "sample_optional_response": sample.get("free_text") if sample else None,
        }
        out.append(row)
    return out


def print_text(rows: List[Dict[str, Any]]) -> None:
    header = (
        f"{'participant_id':<26} "
        f"{'session_id':<34} "
        f"{'answered':>8} "
        f"{'assigned':>8} "
        f"{'done':>5} "
        f"{'elapsed':>12} "
        f"{'active':>12} "
        f"{'attn':>6} "
        f"{'notes':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        assigned = "" if r.get("assigned_count") is None else str(r.get("assigned_count"))
        print(
            f"{str(r.get('participant_id') or ''):<26} "
            f"{str(r.get('session_id') or ''):<34} "
            f"{int(r.get('answered_count') or 0):>8} "
            f"{assigned:>8} "
            f"{str(bool(r.get('completed'))):>5} "
            f"{str(r.get('total_elapsed_human') or ''):>12} "
            f"{str(r.get('total_active_elapsed_human') or ''):>12} "
            f"{format_accuracy(r.get('attention_check_accuracy')):>6} "
            f"{str(r.get('optional_response_count') or 0):>6}"
        )
        sample = r.get("sample_optional_response")
        if sample:
            bits = []
            if r.get("sample_optional_item_index") is not None:
                bits.append(f"Q{int(r.get('sample_optional_item_index')) + 1}")
            if r.get("sample_optional_rating") is not None:
                bits.append(f"rating={r.get('sample_optional_rating')}")
            if r.get("sample_optional_confidence") is not None:
                bits.append(f"confidence={r.get('sample_optional_confidence')}")
            context = f" ({', '.join(bits)})" if bits else ""
            print(f"  optional response{context}: {truncate_text(sample)}")


def print_csv(rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "participant_id",
        "session_id",
        "answered_count",
        "assigned_count",
        "completed",
        "total_elapsed_ms",
        "total_elapsed_seconds",
        "total_elapsed_human",
        "total_active_elapsed_ms",
        "total_active_elapsed_seconds",
        "total_active_elapsed_human",
        "attention_checks_answered",
        "attention_checks_correct",
        "attention_check_accuracy",
        "optional_response_count",
        "sample_optional_item_index",
        "sample_optional_flow_id",
        "sample_optional_task",
        "sample_optional_context_family",
        "sample_optional_output_variant_id",
        "sample_optional_output_variant_label",
        "sample_optional_variant_privacy_class",
        "sample_optional_rating",
        "sample_optional_confidence",
        "sample_optional_created_at_ms",
        "sample_optional_response",
        "created_at_ms",
        "completed_at_ms",
        "last_activity_at_ms",
        "first_response_at_ms",
        "last_response_at_ms",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print participant/session IDs, answered question counts, survey time spent, "
            "and attention-check accuracy from responses.db."
        )
    )
    parser.add_argument("--db", default="survey/outputs/responses.db", help="Path to responses.db")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table.")
    parser.add_argument("--csv", action="store_true", help="Emit CSV instead of a text table.")
    args = parser.parse_args()

    rows = load_counts(Path(args.db))
    if args.json:
        print(json.dumps({"participants": rows}, indent=2, sort_keys=False))
        return 0
    if args.csv:
        print_csv(rows)
        return 0

    print_text(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
