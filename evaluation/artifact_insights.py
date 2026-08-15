#!/usr/bin/env python3
"""
Generate PETS-reviewer-facing analysis artifacts for privMediator.

Run from the project's evaluation/ directory, e.g.:

    python pets_artifact_insights.py \
      --fixed-run ../runs/context_pipeline_generation \
      --flex-run ../runs/flexible_context_pipeline_generation \
      --responses-db ../survey/outputs/responses.db \
      --fixed-contracts ../norms/operator_contracts.json \
      --flex-contracts ../norms/operator_contracts_flexible.json \
      --fixed-vocab ../norms/pipeline_vocabulary.json \
      --flex-vocab ../norms/pipeline_vocabulary_flexible.json \
      --contexts ../survey/data/ci_focused_user_study_context.json \
      --out-dir ../runs/pets_artifact_insights

The script is intentionally read-only with respect to your experiment artifacts.
It writes CSV/JSON/Markdown outputs that help answer likely PETS reviewer questions:

  1. Pipeline decision coverage, selected outputs, operator chains, and residual disclosure.
  2. Survey participant QC and method-linked contextual appropriateness summaries.
  3. Operator-contract validation, vocabulary coverage, and selected-operator usage.
  4. Optional candidate-set sensitivity if full, uncompacted result.json files exist.

Dependencies: Python standard library. If pandas/numpy/statsmodels are installed,
extra summary statistics and optional mixed-effects models are produced. The script
will still run without them.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

RISK_LEVELS = {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": 4}
RISK_LEVELS_REV = {v: k for k, v in RISK_LEVELS.items()}
DEFAULT_RESIDUAL_ATTRS = [
    "identity", "face", "body_shape", "clothing", "gait", "speech_content",
    "speaker_identity", "activity", "location", "trajectory", "co_presence",
    "visible_text", "aggregate_presence", "routine_pattern", "room_usage_pattern",
    "sound_context", "pose_signature", "motion_pattern",
]
BASELINE_METHOD_ORDER = [
    "raw", "manual", "direct_llm", "full_mediator",
    "ablation:utility_only", "ablation:no_ci_filter", "ablation:no_residual_bounds",
    "ablation:no_least_revealing", "ablation:uniform_risk_weights",
    "ablation:metadata_only", "ablation:no_staged_flows", "ablation:first_feasible",
    "ablation:latency_first",
]
SELECT_DECISIONS = {"select_pipeline", "selected", "accept", "accepted"}
NO_OUTPUT_DECISIONS = {"no_compromise", "reject", "rejected", "no_candidates", "consent_or_review", "review", "error", "invalid_or_no_pipeline", "llm_error"}


def log(msg: str) -> None:
    print(f"[pets-insights] {msg}", file=sys.stderr)


def load_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        if default is not None:
            return default
        raise
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in {path}: {e}") from e


def _json_safe(obj: Any) -> Any:
    """Convert objects to JSON-safe values, including tuple-key dictionaries."""
    if isinstance(obj, dict):
        safe: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                key = ":".join(str(part) for part in k)
            elif isinstance(k, (str, int, float, bool)) or k is None:
                key = str(k)
            else:
                key = str(k)
            safe[key] = _json_safe(v)
        return safe
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=2, sort_keys=False)


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r.keys():
                if k not in fields:
                    fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in r.items()})


def read_csv_dicts(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def coerce_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        return default


def coerce_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def safe_json_loads(x: Any, default: Any = None) -> Any:
    if x is None or x == "":
        return default
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(str(x))
    except Exception:
        return default


def infer_project_root(cwd: Path) -> Path:
    # Designed for running from evaluation/. Running from project root also works.
    if cwd.name == "evaluation":
        return cwd.parent
    if (cwd / "evaluation").exists() or (cwd / "runs").exists():
        return cwd
    return cwd.parent


def resolve_path(value: Any, project_root: Path, run_dir: Optional[Path] = None) -> Optional[Path]:
    if value is None or value == "":
        return None
    p = Path(str(value))
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(project_root / p)
        if run_dir is not None:
            candidates.append(run_dir / p)
            # Generated rows sometimes store paths starting with runs/<dir>/...
            # If that path is already under project root, project_root / p above catches it.
    for c in candidates:
        if c.exists():
            return c
    # Return the best candidate for diagnostics even if not present.
    return candidates[0] if candidates else p


def cap_type(cap: Any) -> str:
    if not isinstance(cap, dict):
        return ""
    return str(cap.get("semantic_type") or cap.get("media_type") or "")


def cap_schema(cap: Any) -> str:
    if not isinstance(cap, dict):
        return ""
    return str(cap.get("schema") or "")


def normalize_risk_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        # Round/clamp numeric scores if they are ordinal-like.
        i = int(round(float(v)))
        if i in RISK_LEVELS_REV:
            return RISK_LEVELS_REV[i]
        return "unknown"
    s = str(v).strip().lower()
    if not s:
        return None
    # Sometimes values look like "low/medium"; keep conservative upper bound.
    parts = re.split(r"[/,;| ]+", s)
    vals = [p for p in parts if p in RISK_LEVELS]
    if vals:
        return max(vals, key=lambda p: RISK_LEVELS[p])
    if s in RISK_LEVELS:
        return s
    return "unknown"


def risk_ord(v: Any) -> int:
    return RISK_LEVELS.get(normalize_risk_value(v) or "none", 0)


def flatten_residual(obj: Any) -> Dict[str, str]:
    """Extract a residual disclosure vector from several known candidate/spec shapes."""
    if not isinstance(obj, dict):
        return {}
    candidates = [
        obj.get("residual_disclosure"),
        obj.get("residual_disclosure_vector"),
        obj.get("residual_vector"),
        obj.get("residual"),
    ]
    meta = obj.get("source_candidate_metadata")
    if isinstance(meta, dict):
        candidates.extend([
            meta.get("residual_disclosure"), meta.get("residual_disclosure_vector"), meta.get("residual_vector")
        ])
    out: Dict[str, str] = {}
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        # Common forms: {attr: level}; {"attributes": {attr: level}}; {"final": {attr: level}}
        nested = []
        for key in ["attributes", "final", "final_residual", "d", "levels"]:
            if isinstance(cand.get(key), dict):
                nested.append(cand.get(key))
        nested.append(cand)
        for d in nested:
            for k, v in d.items():
                if isinstance(v, dict):
                    # Handle {attr: {"level":"high"}} or {"risk":"medium"}
                    vv = v.get("level", v.get("risk", v.get("value")))
                    if vv is None:
                        continue
                    v = vv
                nv = normalize_risk_value(v)
                if nv is not None:
                    out[str(k)] = nv
    return out


def operator_ids_from_candidate(cand: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(cand, dict):
        return out
    for op in cand.get("operators", []) or []:
        if isinstance(op, str):
            oid = op
        elif isinstance(op, dict):
            oid = op.get("operator") or op.get("operator_id") or op.get("id")
        else:
            oid = None
        if oid:
            out.append(str(oid))
    if not out and isinstance(cand.get("executable_pipeline_spec"), dict):
        for st in cand["executable_pipeline_spec"].get("stages", []) or []:
            oid = st.get("operator_id") or st.get("operator") if isinstance(st, dict) else None
            if oid:
                out.append(str(oid))
    if not out and isinstance(cand.get("stages"), list):
        for st in cand.get("stages", []) or []:
            oid = st.get("operator_id") or st.get("operator") if isinstance(st, dict) else None
            if oid:
                out.append(str(oid))
    return out


def operator_ids_from_summary_string(x: Any) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    return [p.strip() for p in str(x).split("->") if p.strip()]


def final_cap_from_candidate(cand: Any) -> Dict[str, Any]:
    if not isinstance(cand, dict):
        return {}
    return cand.get("final_output_cap") or cand.get("output_cap") or {}


def selected_candidate_from_result(result: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return None
    for key in ["selected_candidate", "selected"]:
        if isinstance(result.get(key), dict):
            return result[key]
    if isinstance(result.get("decision"), dict):
        pid = result["decision"].get("selected_pipeline_id")
    else:
        pid = result.get("selected_pipeline_id")
    if pid:
        cands = []
        if isinstance(result.get("candidates"), list):
            cands = result["candidates"]
        elif isinstance(result.get("candidate_generation_result"), dict):
            cands = result["candidate_generation_result"].get("candidates", []) or []
        for c in cands:
            if isinstance(c, dict) and str(c.get("pipeline_id")) == str(pid):
                return c
    return None


def _nonempty(v: Any) -> bool:
    return v not in (None, "", [], {})


def _merge_prefer_nonempty(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if _nonempty(v) or k not in out:
            out[k] = v
    return out


def _decision_text_from_result(result: Any, selected_exists: bool = False) -> str:
    if isinstance(result, dict):
        d = result.get("decision") or result.get("outcome") or result.get("status")
        if isinstance(d, dict):
            for key in ["decision", "outcome", "status", "action", "label"]:
                if d.get(key):
                    return str(d.get(key))
            if d.get("selected_pipeline_id") or d.get("selected_candidate"):
                return "select_pipeline"
        elif d:
            return str(d)
        if result.get("selected_pipeline_id") or isinstance(result.get("selected_candidate"), dict) or isinstance(result.get("selected"), dict):
            return "select_pipeline"
        if result.get("no_compromise") is True or result.get("no_selected_pipeline") is True:
            return "no_compromise"
        if result.get("error"):
            return "error"
    return "select_pipeline" if selected_exists else "unknown"


def _safe_load_method_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = load_json(path, {})
        return data if isinstance(data, dict) else {}
    except Exception as e:
        return {"_load_error": str(e)}


def _row_from_method_dir(
    method_dir: Path,
    scenario_id: str,
    method_id: str,
    method_kind: str,
    run_dir: Path,
    context_meta: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Reconstruct a generation-summary row from per-method artifacts.

    This covers the common stale-root-summary case where S*/baselines/* or
    S*/ablations/* contains result/selected/pipeline artifacts, but the run-level
    summary.csv/summary.json did not get updated.
    """
    if not method_dir.exists() or not method_dir.is_dir():
        return None
    summary_path = method_dir / "summary.json"
    result_path = method_dir / "result.json"
    selected_path = None
    for name in ["selected_pipeline.json", "pipeline.json", "selected_candidate.json"]:
        p = method_dir / name
        if p.exists():
            selected_path = p
            break
    spec_path = None
    for name in ["pipeline_spec.json", "spec.json"]:
        p = method_dir / name
        if p.exists():
            spec_path = p
            break
    if not any(p and p.exists() for p in [summary_path, result_path, selected_path, spec_path]):
        return None

    summary = _safe_load_method_json(summary_path)
    result = _safe_load_method_json(result_path)
    cand = None
    if selected_path and selected_path.exists():
        cand = _safe_load_method_json(selected_path)
    if not cand and result:
        cand = selected_candidate_from_result(result)
    spec = _safe_load_method_json(spec_path) if spec_path else {}
    final_cap = final_cap_from_candidate(cand or {}) or final_cap_from_candidate(spec or {})
    op_list = operator_ids_from_candidate(cand or {}) or operator_ids_from_candidate(spec or {})
    if not op_list:
        op_list = operator_ids_from_summary_string(summary.get("operators"))
    decision = str(summary.get("decision") or _decision_text_from_result(result, bool(cand or selected_path))).strip()
    params = context_meta.get("ci_parameters_scalar_context_only") or context_meta.get("ci_parameters_scalar") or {}
    row: Dict[str, Any] = {
        "scenario_id": summary.get("scenario_id") or context_meta.get("scenario_id") or scenario_id,
        "task": summary.get("task") or context_meta.get("task") or params.get("task"),
        "context": summary.get("context") or params.get("context"),
        "space": summary.get("space") or params.get("space"),
        "sender": summary.get("sender") or params.get("sender"),
        "subject": summary.get("subject") or params.get("subject"),
        "recipient": summary.get("recipient") or params.get("recipient"),
        "purpose": summary.get("purpose") or params.get("purpose"),
        "transmission_principle": summary.get("transmission_principle") or params.get("transmission_principle"),
        "context_family": summary.get("context_family") or context_meta.get("context_family") or params.get("context_family"),
        "method_id": summary.get("method_id") or method_id,
        "method_kind": summary.get("method_kind") or method_kind,
        "baseline": summary.get("baseline") or (method_dir.name if method_kind == "baseline" else None),
        "baseline_id": summary.get("baseline_id") or (method_dir.name if method_kind == "baseline" else None),
        "ablation_mode": summary.get("ablation_mode") or (method_dir.name if method_kind == "ablation" else None),
        "parent_method": summary.get("parent_method") or ("full_mediator" if method_kind == "ablation" else None),
        "method_output_dir": str(method_dir),
        "baseline_output_dir": str(method_dir) if method_kind == "baseline" else None,
        "decision": decision,
        "selected_pipeline_id": summary.get("selected_pipeline_id") or (cand or {}).get("pipeline_id") or result.get("selected_pipeline_id"),
        "matched_output_cap": summary.get("matched_output_cap") or (cand or {}).get("matched_output_cap"),
        "matched_output_schema": summary.get("matched_output_schema") or cap_schema((cand or {}).get("matched_output_cap") or {}),
        "final_output_type": summary.get("final_output_type") or cap_type(final_cap),
        "final_output_schema": summary.get("final_output_schema") or cap_schema(final_cap),
        "operators": summary.get("operators") or "->".join(op_list),
        "result_json": str(result_path) if result_path.exists() else summary.get("result_json"),
        "selected_pipeline_json": str(selected_path) if selected_path and selected_path.exists() else summary.get("selected_pipeline_json"),
        "pipeline_spec_json": str(spec_path) if spec_path and spec_path.exists() else summary.get("pipeline_spec_json"),
        "run_pipeline_py": str(method_dir / "run_pipeline.py") if (method_dir / "run_pipeline.py").exists() else summary.get("run_pipeline_py"),
        "error": summary.get("error") or result.get("error") or result.get("_load_error"),
        "no_compromise_reason": summary.get("no_compromise_reason") or result.get("no_compromise_reason"),
        "no_compromise_top_failed_rules": summary.get("no_compromise_top_failed_rules") or result.get("no_compromise_top_failed_rules"),
        "candidate_count": summary.get("candidate_count") or result.get("candidate_count"),
        "selector_num_feasible": summary.get("selector_num_feasible") or result.get("selector_num_feasible"),
        "row_source": "reconstructed_from_method_dir",
    }
    # Preserve any extra fields in method summary, but do not let empty summary
    # values clobber reconstructed paths/decision fields.
    row = _merge_prefer_nonempty(summary, row)
    return row


def _rows_from_context_summary(context_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p_json = context_dir / "context_summary.json"
    p_csv = context_dir / "context_summary.csv"
    if p_json.exists():
        data = load_json(p_json, {})
        scenario_id = data.get("scenario_id") or context_dir.name
        task = data.get("task")
        context_family = data.get("context_family")
        params = data.get("ci_parameters_scalar_context_only") or {}
        for method_id, m in (data.get("methods") or {}).items():
            if isinstance(m, dict):
                r = dict(m)
                r.setdefault("scenario_id", scenario_id)
                r.setdefault("task", task)
                r.setdefault("context_family", context_family)
                for k in ["context", "space", "sender", "subject", "recipient", "purpose", "transmission_principle"]:
                    r.setdefault(k, params.get(k))
                r.setdefault("method_id", method_id)
                r.setdefault("row_source", "context_summary_json")
                rows.append(r)
    elif p_csv.exists():
        for r in read_csv_dicts(p_csv):
            r.setdefault("scenario_id", context_dir.name)
            r.setdefault("row_source", "context_summary_csv")
            rows.append(r)
    return rows


def _scan_run_method_dirs(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for context_dir in sorted(run_dir.glob("S*")):
        if not context_dir.is_dir():
            continue
        context_meta = load_json(context_dir / "context_summary.json", {}) if (context_dir / "context_summary.json").exists() else {}
        scenario_id = str(context_meta.get("scenario_id") or context_dir.name)
        # Standard layout: Sxxx/baselines/<baseline> and Sxxx/ablations/<mode>.
        containers = [(context_dir / "baselines", "baseline"), (context_dir / "ablations", "ablation")]
        for container, kind in containers:
            if not container.exists():
                continue
            for method_dir in sorted([p for p in container.iterdir() if p.is_dir()]):
                mid = method_dir.name if kind == "baseline" else f"ablation:{method_dir.name}"
                row = _row_from_method_dir(method_dir, scenario_id, mid, kind, run_dir, context_meta)
                if row:
                    rows.append(row)
        # Defensive fallback for older layouts: Sxxx/<method>/result.json.
        for method_dir in sorted([p for p in context_dir.iterdir() if p.is_dir()]):
            if method_dir.name in {"baselines", "ablations", "_utility_selected_only"}:
                continue
            if method_dir.name.startswith("."):
                continue
            kind = "ablation" if method_dir.name.startswith("ablation") else "baseline"
            mid = method_dir.name if kind == "baseline" else method_dir.name.replace("ablation_", "ablation:")
            row = _row_from_method_dir(method_dir, scenario_id, mid, kind, run_dir, context_meta)
            if row:
                rows.append(row)
    return rows


def read_summary_rows(run_dir: Path, run_mode: str, project_root: Path) -> List[Dict[str, Any]]:
    if not run_dir or not run_dir.exists():
        log(f"Skipping missing run directory for {run_mode}: {run_dir}")
        return []
    loaded: List[Dict[str, Any]] = []
    source_counts: Counter = Counter()

    # Root summary files are useful when current, but they can be stale. Load
    # them first, then add/overwrite with context summaries and reconstructed
    # filesystem rows.
    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"
    if summary_json.exists():
        data = load_json(summary_json, [])
        if isinstance(data, list):
            for x in data:
                if isinstance(x, dict):
                    r = dict(x)
                    r.setdefault("row_source", "root_summary_json")
                    loaded.append(r)
                    source_counts["root_summary_json"] += 1
    elif summary_csv.exists():
        for r in read_csv_dicts(summary_csv):
            r.setdefault("row_source", "root_summary_csv")
            loaded.append(r)
            source_counts["root_summary_csv"] += 1

    for context_dir in sorted(run_dir.glob("S*")):
        if not context_dir.is_dir():
            continue
        for r in _rows_from_context_summary(context_dir):
            loaded.append(r)
            source_counts[str(r.get("row_source") or "context_summary")] += 1

    fs_rows = _scan_run_method_dirs(run_dir)
    for r in fs_rows:
        loaded.append(r)
        source_counts["reconstructed_from_method_dir"] += 1

    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r0 in loaded:
        r = dict(r0)
        sid = str(r.get("scenario_id") or "")
        mid = str(r.get("method_id") or r.get("baseline") or "")
        if not sid or not mid:
            continue
        key = (sid, mid)
        if key in by_key:
            by_key[key] = _merge_prefer_nonempty(by_key[key], r)
        else:
            by_key[key] = r
    rows = []
    for r in by_key.values():
        r["run_mode"] = run_mode
        r["run_dir"] = str(run_dir)
        rows.append(r)
    rows.sort(key=lambda r: (str(r.get("scenario_id")), str(r.get("method_id"))))
    log(f"Loaded {len(rows)} unique rows from {run_mode}: {run_dir} (sources={dict(source_counts)})")
    return rows


def enrich_pipeline_rows(rows: List[Dict[str, Any]], project_root: Path, out_dir: Path) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    residual_attrs_seen: Counter = Counter()
    for r0 in rows:
        r = dict(r0)
        run_dir = Path(str(r.get("run_dir") or project_root))
        selected_path = resolve_path(r.get("selected_pipeline_json"), project_root, run_dir)
        spec_path = resolve_path(r.get("pipeline_spec_json"), project_root, run_dir)
        result_path = resolve_path(r.get("result_json"), project_root, run_dir)
        cand: Optional[Dict[str, Any]] = None
        spec: Optional[Dict[str, Any]] = None
        result: Optional[Dict[str, Any]] = None
        if selected_path and selected_path.exists():
            try:
                cand = load_json(selected_path)
            except Exception as e:
                r["selected_pipeline_load_error"] = str(e)
        if spec_path and spec_path.exists():
            try:
                spec = load_json(spec_path)
            except Exception as e:
                r["pipeline_spec_load_error"] = str(e)
        if result_path and result_path.exists():
            try:
                result = load_json(result_path)
            except Exception as e:
                r["result_json_load_error"] = str(e)
        if cand is None and result is not None:
            cand = selected_candidate_from_result(result)
        residual = flatten_residual(cand or {})
        if not residual and spec is not None:
            residual = flatten_residual(spec)
        residual_attrs_seen.update(residual.keys())
        operators = operator_ids_from_candidate(cand or {})
        if not operators and spec is not None:
            operators = operator_ids_from_candidate(spec)
        if not operators:
            operators = operator_ids_from_summary_string(r.get("operators"))
        final_cap = final_cap_from_candidate(cand or {})
        if not final_cap and spec is not None:
            final_cap = final_cap_from_candidate(spec)
        final_type = cap_type(final_cap) or str(r.get("final_output_type") or "")
        final_schema = cap_schema(final_cap) or str(r.get("final_output_schema") or "")
        selected = str(r.get("decision") or "").strip().lower() in SELECT_DECISIONS
        is_raw_like = final_type in {"video/x-raw", "image/x-raw", "audio/x-raw"} or "raw" in final_schema.lower()
        is_media = final_type.startswith(("video/", "image/", "audio/"))
        is_semantic = final_type.startswith("application/")
        residual_ord = {k: risk_ord(v) for k, v in residual.items()}
        r.update({
            "selected_pipeline_path_resolved": str(selected_path) if selected_path else None,
            "result_json_path_resolved": str(result_path) if result_path else None,
            "pipeline_spec_path_resolved": str(spec_path) if spec_path else None,
            "selected_pipeline_path_exists": bool(selected_path and selected_path.exists()),
            "result_json_path_exists": bool(result_path and result_path.exists()),
            "pipeline_spec_path_exists": bool(spec_path and spec_path.exists()),
            "operators_list": operators,
            "operator_count": len(operators),
            "final_output_type_enriched": final_type,
            "final_output_schema_enriched": final_schema,
            "is_selected_output": int(selected),
            "is_raw_like_output": int(bool(selected and is_raw_like)),
            "is_media_output": int(bool(selected and is_media)),
            "is_semantic_output": int(bool(selected and is_semantic)),
            "residual_json": residual,
            "residual_risk_sum": sum(residual_ord.values()) if residual_ord else None,
            "residual_high_count": sum(1 for v in residual_ord.values() if v >= 3),
            "residual_med_or_high_count": sum(1 for v in residual_ord.values() if v >= 2),
            "residual_unknown_count": sum(1 for v in residual_ord.values() if v >= 4),
        })
        for attr in DEFAULT_RESIDUAL_ATTRS:
            val = residual.get(attr)
            r[f"residual_{attr}"] = val
            r[f"residual_{attr}_ord"] = risk_ord(val) if val is not None else 0
        enriched.append(r)
    fields = None
    write_csv(out_dir / "pipeline_selected_outputs_long.csv", enriched, fields)
    return enriched


def aggregate_pipeline_summaries(rows: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r.get("run_mode") or ""), str(r.get("method_id") or r.get("baseline") or ""))].append(r)

    decision_rows: List[Dict[str, Any]] = []
    residual_rows: List[Dict[str, Any]] = []
    output_rows: List[Dict[str, Any]] = []
    operator_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []

    for (run_mode, method_id), items in sorted(groups.items()):
        dec_counts = Counter(str(x.get("decision") or "").strip() for x in items)
        selected = sum(int(x.get("is_selected_output") or 0) for x in items)
        rejected = len(items) - selected
        decision_rows.append({
            "run_mode": run_mode,
            "method_id": method_id,
            "n_rows": len(items),
            "selected": selected,
            "not_selected_or_rejected": rejected,
            "selected_rate": selected / len(items) if items else None,
            "decision_counts_json": dict(dec_counts),
            "mean_candidate_count": mean([coerce_float(x.get("candidate_count")) for x in items], skip_nan=True),
            "mean_selector_feasible": mean([coerce_float(x.get("selector_num_feasible")) for x in items], skip_nan=True),
        })
        selected_items = [x for x in items if int(x.get("is_selected_output") or 0)]
        if selected_items:
            rr: Dict[str, Any] = {
                "run_mode": run_mode,
                "method_id": method_id,
                "selected_n": len(selected_items),
                "raw_like_release_n": sum(int(x.get("is_raw_like_output") or 0) for x in selected_items),
                "media_release_n": sum(int(x.get("is_media_output") or 0) for x in selected_items),
                "semantic_release_n": sum(int(x.get("is_semantic_output") or 0) for x in selected_items),
                "mean_residual_risk_sum": mean([coerce_float(x.get("residual_risk_sum")) for x in selected_items], skip_nan=True),
                "mean_high_attr_count": mean([coerce_float(x.get("residual_high_count")) for x in selected_items], skip_nan=True),
                "mean_med_or_high_attr_count": mean([coerce_float(x.get("residual_med_or_high_count")) for x in selected_items], skip_nan=True),
                "mean_unknown_attr_count": mean([coerce_float(x.get("residual_unknown_count")) for x in selected_items], skip_nan=True),
            }
            for attr in DEFAULT_RESIDUAL_ATTRS:
                vals = [coerce_float(x.get(f"residual_{attr}_ord"), 0) for x in selected_items]
                rr[f"avg_{attr}_ord"] = mean(vals, skip_nan=True)
                rr[f"pct_{attr}_medium_or_higher"] = sum(1 for v in vals if v >= 2) / len(vals) if vals else None
                rr[f"pct_{attr}_high_or_unknown"] = sum(1 for v in vals if v >= 3) / len(vals) if vals else None
            residual_rows.append(rr)
        schema_counts = Counter((str(x.get("final_output_type_enriched") or x.get("final_output_type") or ""), str(x.get("final_output_schema_enriched") or x.get("final_output_schema") or "")) for x in selected_items)
        for (t, s), n in schema_counts.most_common():
            output_rows.append({"run_mode": run_mode, "method_id": method_id, "final_output_type": t, "final_output_schema": s, "selected_n": n})
        op_counts: Counter = Counter()
        for x in selected_items:
            for op in x.get("operators_list") or []:
                op_counts[str(op)] += 1
        for op, n in op_counts.most_common():
            operator_rows.append({"run_mode": run_mode, "method_id": method_id, "operator_id": op, "selected_pipeline_count": n})
        for x in items:
            if not int(x.get("is_selected_output") or 0):
                diag_rows.append({
                    "run_mode": run_mode,
                    "scenario_id": x.get("scenario_id"),
                    "task": x.get("task"),
                    "context_family": x.get("context_family"),
                    "method_id": method_id,
                    "decision": x.get("decision"),
                    "no_compromise_reason": x.get("no_compromise_reason"),
                    "no_compromise_top_failed_rules": x.get("no_compromise_top_failed_rules"),
                    "candidate_count": x.get("candidate_count"),
                    "selector_num_feasible": x.get("selector_num_feasible"),
                    "error": x.get("error"),
                })

    write_csv(out_dir / "pipeline_decision_coverage_by_method.csv", decision_rows)
    write_csv(out_dir / "pipeline_residual_exposure_by_method.csv", residual_rows)
    write_csv(out_dir / "pipeline_output_schema_by_method.csv", output_rows)
    write_csv(out_dir / "pipeline_operator_usage_by_method.csv", operator_rows)
    write_csv(out_dir / "pipeline_rejection_diagnostics.csv", diag_rows)
    return {
        "decision_rows": decision_rows,
        "residual_rows": residual_rows,
        "output_rows": output_rows,
        "operator_rows": operator_rows,
        "diagnostic_rows": diag_rows,
    }


def mean(vals: Iterable[float], skip_nan: bool = True) -> Optional[float]:
    xs = []
    for v in vals:
        try:
            fv = float(v)
            if math.isnan(fv) and skip_nan:
                continue
            xs.append(fv)
        except Exception:
            if not skip_nan:
                xs.append(float("nan"))
    if not xs:
        return None
    return sum(xs) / len(xs)


def table_names(conn: sqlite3.Connection) -> List[str]:
    return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]


def load_db_table(db_path: Path, table: str) -> List[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if table not in table_names(conn):
            return []
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}").fetchall()]


def session_key(row: Dict[str, Any]) -> str:
    return str(row.get("prolific_pid") or row.get("participant_id") or row.get("session_id") or "")


def analyze_survey_db(db_path: Any, out_dir: Path, include_incomplete: bool = False, bootstrap_iters: int = 1000, mixed_model: bool = False) -> Dict[str, Any]:
    """Analyze one or more survey SQLite DBs as one combined study.

    `db_path` may be a Path, a string, or a sequence of Paths/strings. When
    multiple DBs are supplied (e.g., responses.db and responses_old.db), session
    IDs are prefixed with the DB stem so collisions cannot merge sessions.
    """
    if isinstance(db_path, (list, tuple, set)):
        db_paths = [Path(str(p)) for p in db_path]
    else:
        db_paths = [Path(str(db_path))]
    db_paths = [p for p in db_paths if str(p)]
    if not db_paths:
        log("Skipping survey analysis because no DB paths were supplied")
        return {"available": False, "paths": []}

    all_sessions: List[Dict[str, Any]] = []
    all_responses: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []

    for idx, path in enumerate(db_paths):
        if not path.exists():
            log(f"Survey DB does not exist, skipping: {path}")
            source_rows.append({"source_db": str(path), "exists": 0, "sessions": 0, "responses": 0})
            continue
        try:
            sessions = load_db_table(path, "sessions")
            responses = load_db_table(path, "responses")
        except Exception as e:
            log(f"Could not read survey DB {path}: {e}")
            source_rows.append({"source_db": str(path), "exists": 1, "read_error": str(e), "sessions": 0, "responses": 0})
            continue
        slug = re.sub(r"[^A-Za-z0-9_]+", "_", path.stem) or f"db{idx}"
        session_lookup = {str(s.get("session_id")): s for s in sessions}
        if not sessions and responses:
            # Minimal fallback for exports/dbs that contain responses but no sessions table.
            for sid in sorted({str(r.get("session_id")) for r in responses}):
                sessions.append({"session_id": sid, "participant_id": sid, "assignment_json": "[]"})
                session_lookup[sid] = sessions[-1]
        for s in sessions:
            orig_sid = str(s.get("session_id"))
            ss = dict(s)
            ss["original_session_id"] = orig_sid
            ss["session_id"] = f"{slug}::{orig_sid}"
            ss["source_db"] = str(path)
            ss["source_db_stem"] = slug
            all_sessions.append(ss)
        for r in responses:
            orig_sid = str(r.get("session_id"))
            rr = dict(r)
            rr["original_session_id"] = orig_sid
            rr["session_id"] = f"{slug}::{orig_sid}"
            rr["source_db"] = str(path)
            rr["source_db_stem"] = slug
            s = session_lookup.get(orig_sid, {})
            # Some response rows do not duplicate participant/prolific metadata.
            for k in ["participant_id", "prolific_pid", "prolific_study_id", "prolific_session_id"]:
                if not rr.get(k) and s.get(k):
                    rr[k] = s.get(k)
            all_responses.append(rr)
        source_rows.append({"source_db": str(path), "exists": 1, "sessions": len(sessions), "responses": len(responses)})

    write_csv(out_dir / "survey_db_sources.csv", source_rows)
    sessions = all_sessions
    responses = all_responses
    if not responses:
        log(f"No responses found in supplied DBs: {[str(p) for p in db_paths]}")

    # Participant/session QC.
    responses_by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in responses:
        responses_by_session[str(r.get("session_id"))].append(r)
    qc_rows: List[Dict[str, Any]] = []
    for s in sessions:
        sid = str(s.get("session_id"))
        assigned = safe_json_loads(s.get("assignment_json"), []) or []
        rs = responses_by_session.get(sid, [])
        att = [r.get("attention_check_correct") for r in rs if r.get("attention_check_correct") is not None]
        comp = [r.get("comprehension_check_correct") for r in rs if r.get("comprehension_check_correct") is not None]
        assigned_count = len(assigned)
        answered_count = coerce_int(s.get("answered_count"), len(rs))
        completed = bool(s.get("completed_at_ms")) or (assigned_count > 0 and answered_count >= assigned_count) or (assigned_count == 0 and len(rs) > 0 and bool(s.get("completed")))
        qc_rows.append({
            "participant_key": session_key(s),
            "session_id": sid,
            "original_session_id": s.get("original_session_id"),
            "source_db": s.get("source_db"),
            "source_db_stem": s.get("source_db_stem"),
            "participant_id": s.get("participant_id"),
            "prolific_pid": s.get("prolific_pid"),
            "assigned_count": assigned_count,
            "answered_count": answered_count,
            "responses_in_db": len(rs),
            "completed": int(completed),
            "attention_checks_n": len(att),
            "attention_checks_failed": sum(1 for x in att if coerce_int(x) == 0),
            "attention_accuracy": mean([coerce_float(x) for x in att], skip_nan=True),
            "comprehension_checks_n": len(comp),
            "comprehension_checks_failed": sum(1 for x in comp if coerce_int(x) == 0),
            "comprehension_accuracy": mean([coerce_float(x) for x in comp], skip_nan=True),
            "total_elapsed_min": coerce_float(s.get("total_elapsed_ms"), 0) / 60000.0 if s.get("total_elapsed_ms") is not None else None,
            "total_active_elapsed_min": coerce_float(s.get("total_active_elapsed_ms"), 0) / 60000.0 if s.get("total_active_elapsed_ms") is not None else None,
            "created_at_ms": s.get("created_at_ms"),
            "completed_at_ms": s.get("completed_at_ms"),
        })
    write_csv(out_dir / "survey_participant_qc.csv", qc_rows)

    completed_sessions = {r["session_id"] for r in qc_rows if int(r.get("completed") or 0)}
    use_responses = [r for r in responses if include_incomplete or str(r.get("session_id")) in completed_sessions]
    # Explode responses by method_details_json so deduplicated scenario-output ratings can be joined back to methods.
    long_rows: List[Dict[str, Any]] = []
    for r in use_responses:
        details = safe_json_loads(r.get("method_details_json"), [])
        if not details:
            ids = safe_json_loads(r.get("method_ids_json"), []) or safe_json_loads(r.get("baseline_ids_json"), []) or []
            details = [{"method_id": mid} for mid in ids]
        seen_method_ids = set()
        for d in details or []:
            if not isinstance(d, dict):
                d = {"method_id": str(d)}
            mid = str(d.get("method_id") or d.get("baseline") or "unknown_method")
            if mid in seen_method_ids:
                continue
            seen_method_ids.add(mid)
            item_key = str(r.get("item_id") or f"{r.get('flow_id')}::{r.get('output_variant_id')}")
            participant_key = str(r.get("prolific_pid") or r.get("participant_id") or r.get("session_id"))
            long_rows.append({
                "participant_key": participant_key,
                "session_id": r.get("session_id"),
                "original_session_id": r.get("original_session_id"),
                "source_db": r.get("source_db"),
                "source_db_stem": r.get("source_db_stem"),
                "participant_id": r.get("participant_id"),
                "item_index": r.get("item_index"),
                "item_id": r.get("item_id"),
                "item_key": item_key,
                "flow_id": r.get("flow_id"),
                "task": r.get("task"),
                "context_family": r.get("context_family"),
                "output_variant_id": r.get("output_variant_id"),
                "output_variant_label": r.get("output_variant_label"),
                "variant_privacy_class": r.get("variant_privacy_class"),
                "final_output_type": r.get("final_output_type"),
                "final_output_schema": r.get("final_output_schema"),
                "rating": coerce_int(r.get("rating"), 0),
                "confidence": coerce_int(r.get("confidence"), 0) if r.get("confidence") is not None else None,
                "appropriate_4_or_5": int(coerce_int(r.get("rating"), 0) >= 4),
                "free_text": r.get("free_text"),
                "elapsed_ms": r.get("elapsed_ms"),
                "attention_check_correct": r.get("attention_check_correct"),
                "comprehension_check_correct": r.get("comprehension_check_correct"),
                "method_id": mid,
                "method_kind": d.get("method_kind"),
                "method_label": d.get("method_label") or mid,
                "baseline": d.get("baseline"),
                "baseline_id": d.get("baseline_id"),
                "ablation_mode": d.get("ablation_mode"),
                "parent_method": d.get("parent_method"),
                "selected_pipeline_id": d.get("selected_pipeline_id"),
                "operators": d.get("operators"),
            })
    write_csv(out_dir / "survey_method_ratings_long.csv", long_rows)

    method_summary = summarize_ratings(long_rows, group_cols=["method_id"], bootstrap_iters=bootstrap_iters)
    write_csv(out_dir / "survey_method_rating_summary.csv", method_summary)
    output_summary = summarize_ratings(long_rows, group_cols=["variant_privacy_class", "output_variant_label", "final_output_schema"], bootstrap_iters=bootstrap_iters)
    write_csv(out_dir / "survey_output_variant_rating_summary.csv", output_summary)
    context_summary = summarize_ratings(long_rows, group_cols=["context_family"], bootstrap_iters=bootstrap_iters)
    write_csv(out_dir / "survey_context_family_rating_summary.csv", context_summary)
    task_summary = summarize_ratings(long_rows, group_cols=["task"], bootstrap_iters=bootstrap_iters)
    write_csv(out_dir / "survey_task_rating_summary.csv", task_summary)
    source_summary = summarize_ratings(long_rows, group_cols=["source_db_stem", "method_id"], bootstrap_iters=bootstrap_iters)
    write_csv(out_dir / "survey_method_rating_summary_by_db.csv", source_summary)

    comments: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in use_responses:
        txt = str(r.get("free_text") or "").strip()
        if txt:
            comments[str(r.get("flow_id"))].append({
                "participant_key": str(r.get("prolific_pid") or r.get("participant_id") or r.get("session_id")),
                "source_db": r.get("source_db"),
                "item_id": r.get("item_id"),
                "output_variant_label": r.get("output_variant_label"),
                "rating": r.get("rating"),
                "comment": txt,
            })
    write_json(out_dir / "survey_free_text_by_scenario.json", comments)

    mixed_info: Dict[str, Any] = {"requested": mixed_model, "ran": False}
    if mixed_model:
        mixed_info = run_optional_mixed_model(long_rows, out_dir)

    overview = {
        "available": True,
        "db_paths": [str(p) for p in db_paths],
        "db_sources": source_rows,
        "session_count": len(sessions),
        "completed_session_count": sum(1 for r in qc_rows if int(r.get("completed") or 0)),
        "response_count_total": len(responses),
        "response_count_analyzed": len(use_responses),
        "method_linked_rating_rows": len(long_rows),
        "attention_failed_sessions": sum(1 for r in qc_rows if coerce_int(r.get("attention_checks_failed"), 0) > 0),
        "comprehension_failed_sessions": sum(1 for r in qc_rows if coerce_int(r.get("comprehension_checks_failed"), 0) > 0),
        "mixed_model": mixed_info,
        "include_incomplete": include_incomplete,
    }
    write_json(out_dir / "survey_overview.json", overview)
    return overview


def summarize_ratings(rows: List[Dict[str, Any]], group_cols: List[str], bootstrap_iters: int = 1000) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = tuple(r.get(c) for c in group_cols)
        groups[key].append(r)
    out: List[Dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        ratings = [coerce_float(x.get("rating")) for x in items if coerce_float(x.get("rating"), float("nan")) == coerce_float(x.get("rating"), float("nan"))]
        apps = [1 if coerce_float(x.get("rating"), 0) >= 4 else 0 for x in items]
        participants = {x.get("participant_key") for x in items}
        item_keys = {x.get("item_key") for x in items}
        row: Dict[str, Any] = {c: key[i] for i, c in enumerate(group_cols)}
        row.update({
            "n_method_linked_rows": len(items),
            "n_unique_participants": len(participants),
            "n_unique_items": len(item_keys),
            "mean_rating": mean(ratings, skip_nan=True),
            "median_rating": statistics.median(ratings) if ratings else None,
            "sd_rating_naive": statistics.stdev(ratings) if len(ratings) > 1 else None,
            "appropriate_4_or_5_rate": mean([float(x) for x in apps], skip_nan=True),
        })
        ci = cluster_bootstrap_mean(items, value_key="rating", cluster_key="participant_key", iters=bootstrap_iters)
        row.update({
            "cluster_bootstrap_mean": ci.get("mean"),
            "cluster_bootstrap_ci_low": ci.get("ci_low"),
            "cluster_bootstrap_ci_high": ci.get("ci_high"),
            "cluster_bootstrap_iters": ci.get("iters"),
        })
        out.append(row)
    return out


def cluster_bootstrap_mean(rows: List[Dict[str, Any]], value_key: str, cluster_key: str, iters: int = 1000) -> Dict[str, Any]:
    vals_by_cluster: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        v = coerce_float(r.get(value_key), float("nan"))
        if math.isnan(v):
            continue
        vals_by_cluster[str(r.get(cluster_key))].append(v)
    clusters = list(vals_by_cluster.keys())
    observed = mean([v for vs in vals_by_cluster.values() for v in vs], skip_nan=True)
    if not clusters or iters <= 0 or np is None:
        return {"mean": observed, "ci_low": None, "ci_high": None, "iters": 0}
    rng = np.random.default_rng(13)
    boots: List[float] = []
    for _ in range(iters):
        sample_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sample_values: List[float] = []
        for c in sample_clusters:
            sample_values.extend(vals_by_cluster[str(c)])
        if sample_values:
            boots.append(float(np.mean(sample_values)))
    if not boots:
        return {"mean": observed, "ci_low": None, "ci_high": None, "iters": 0}
    return {
        "mean": observed,
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "iters": iters,
    }


def sanitize_factor(s: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s or "unknown")).strip("_") or "unknown"


def run_optional_mixed_model(rows: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    if pd is None:
        return {"requested": True, "ran": False, "reason": "pandas not installed"}
    try:
        import statsmodels.formula.api as smf  # type: ignore
    except Exception as e:
        return {"requested": True, "ran": False, "reason": f"statsmodels unavailable: {e}"}
    df = pd.DataFrame(rows)
    if df.empty:
        return {"requested": True, "ran": False, "reason": "no rows"}
    # Keep principal methods first to avoid an enormous design matrix if many ablations exist.
    principal = {"raw", "manual", "direct_llm", "full_mediator"}
    df = df[df["method_id"].isin(principal)].copy()
    if df.empty or df["method_id"].nunique() < 2:
        return {"requested": True, "ran": False, "reason": "not enough principal methods"}
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["method_factor"] = df["method_id"].map(sanitize_factor)
    df["participant_factor"] = df["participant_key"].map(sanitize_factor)
    df["item_factor"] = df["item_key"].map(sanitize_factor)
    try:
        # MixedLM with participant as grouping factor and item as variance component.
        model = smf.mixedlm(
            "rating ~ C(method_factor)",
            data=df,
            groups=df["participant_factor"],
            vc_formula={"item": "0 + C(item_factor)"},
        )
        fit = model.fit(method="lbfgs", maxiter=500, disp=False)
        text = str(fit.summary())
        (out_dir / "survey_mixedlm_principal_methods.txt").write_text(text, encoding="utf-8")
        params = []
        for name, coef in fit.params.items():
            params.append({
                "term": name,
                "coef": float(coef),
                "se": float(fit.bse.get(name, float("nan"))) if hasattr(fit, "bse") else None,
                "pvalue": float(fit.pvalues.get(name, float("nan"))) if hasattr(fit, "pvalues") else None,
            })
        write_csv(out_dir / "survey_mixedlm_principal_methods_params.csv", params)
        return {"requested": True, "ran": True, "model": "rating ~ C(method_factor) + (1|participant) + (1|item variance component)", "n_rows": int(len(df))}
    except Exception as e:
        return {"requested": True, "ran": False, "reason": repr(e)}


def load_contract(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        log(f"Contract not found for {label}: {path}")
        return {"operators": [], "abstract_domain": {}, "_missing": True, "_path": str(path), "_label": label}
    data = load_json(path, {})
    data["_path"] = str(path)
    data["_label"] = label
    return data


def extract_cap_strings(caps: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(caps, list):
        return out
    for cap in caps:
        if not isinstance(cap, dict):
            continue
        t = cap.get("media_type") or cap.get("semantic_type") or cap.get("content_type")
        s = cap.get("schema")
        if t or s:
            out.append(f"{t or ''}:{s or ''}")
    return out


def walk_risk_values(obj: Any, prefix: str = "") -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            nv = normalize_risk_value(v)
            if nv is not None and not isinstance(v, (dict, list)):
                out.append((path, nv))
            else:
                out.extend(walk_risk_values(v, path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(walk_risk_values(v, f"{prefix}[{i}]"))
    return out


def analyze_contracts(
    fixed_contract_path: Path,
    flex_contract_path: Path,
    fixed_vocab_path: Optional[Path],
    flex_vocab_path: Optional[Path],
    pipeline_rows: List[Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Any]:
    contracts = [load_contract(fixed_contract_path, "fixed"), load_contract(flex_contract_path, "flexible")]
    operator_validation: List[Dict[str, Any]] = []
    effect_rows: List[Dict[str, Any]] = []
    contract_ids_by_mode: Dict[str, set] = {}
    all_contract_ids: set = set()
    for data in contracts:
        label = data.get("_label")
        operators = data.get("operators") if isinstance(data.get("operators"), list) else []
        required = data.get("operator_interface_schema", {}).get("required_fields") or [
            "id", "label", "category", "operator_kind", "input_caps", "output_caps", "utility_capabilities",
            "transformation_effects", "residual_disclosure_effect", "ci_annotations", "composition_notes",
        ]
        risk_levels = set((data.get("abstract_domain") or {}).get("risk_levels", RISK_LEVELS).keys())
        ids = [str(op.get("id")) for op in operators if isinstance(op, dict) and op.get("id")]
        duplicates = {x for x, n in Counter(ids).items() if n > 1}
        contract_ids_by_mode[str(label)] = set(ids)
        all_contract_ids.update(ids)
        for op in operators:
            if not isinstance(op, dict):
                continue
            oid = str(op.get("id") or "")
            missing_required = [f for f in required if f not in op or op.get(f) in (None, "", [])]
            risk_vals = walk_risk_values(op.get("residual_disclosure_effect"))
            invalid_risks = sorted({v for _, v in risk_vals if v not in risk_levels})
            operator_validation.append({
                "contract_mode": label,
                "operator_id": oid,
                "label": op.get("label"),
                "category": op.get("category"),
                "operator_kind": op.get("operator_kind"),
                "missing_required_fields": ";".join(missing_required),
                "missing_required_count": len(missing_required),
                "input_caps_count": len(op.get("input_caps") or []),
                "output_caps_count": len(op.get("output_caps") or []),
                "input_caps": extract_cap_strings(op.get("input_caps")),
                "output_caps": extract_cap_strings(op.get("output_caps")),
                "utility_capabilities_count": len(op.get("utility_capabilities") or []),
                "transformation_effects_count": len(op.get("transformation_effects") or []),
                "residual_effect_risk_assignments_count": len(risk_vals),
                "invalid_risk_values": ";".join(invalid_risks),
                "duplicate_operator_id_in_contract": int(oid in duplicates),
                "has_ci_annotations": int(bool(op.get("ci_annotations"))),
                "composition_notes_present": int(bool(op.get("composition_notes"))),
            })
            for path, val in risk_vals:
                effect_rows.append({
                    "contract_mode": label,
                    "operator_id": oid,
                    "effect_path": path,
                    "risk_level": val,
                    "risk_ord": RISK_LEVELS.get(val),
                })

    write_csv(out_dir / "contract_operator_validation.csv", operator_validation)
    write_csv(out_dir / "contract_residual_effect_assignments.csv", effect_rows)

    # Vocabulary coverage checks.
    vocab_rows: List[Dict[str, Any]] = []
    for mode, p in [("fixed", fixed_vocab_path), ("flexible", flex_vocab_path)]:
        if not p or not p.exists():
            continue
        vocab = load_json(p, {})
        attr_terms: List[str] = []
        attrs = vocab.get("residual_disclosure_attributes")
        if isinstance(attrs, dict) and isinstance(attrs.get("attributes"), list):
            attr_terms = [str(a.get("term")) for a in attrs["attributes"] if isinstance(a, dict) and a.get("term")]
        elif isinstance(attrs, list):
            attr_terms = [str(x) for x in attrs]
        contract = contracts[0] if mode == "fixed" else contracts[1]
        contract_attrs = list((contract.get("abstract_domain") or {}).get("residual_disclosure_attributes") or [])
        for a in sorted(set(attr_terms) | set(contract_attrs)):
            vocab_rows.append({
                "mode": mode,
                "residual_attribute": a,
                "in_vocabulary": int(a in attr_terms),
                "in_contract_abstract_domain": int(a in contract_attrs),
            })
    write_csv(out_dir / "contract_vocabulary_residual_attribute_coverage.csv", vocab_rows)

    # Usage vs catalog.
    used_counts: Counter = Counter()
    used_by_mode: Dict[Tuple[str, str], int] = Counter()
    for r in pipeline_rows:
        if not int(r.get("is_selected_output") or 0):
            continue
        for op in r.get("operators_list") or []:
            op = str(op)
            used_counts[op] += 1
            used_by_mode[(str(r.get("run_mode") or ""), op)] += 1
    usage_rows: List[Dict[str, Any]] = []
    for op in sorted(set(used_counts.keys()) | all_contract_ids):
        usage_rows.append({
            "operator_id": op,
            "used_in_selected_pipelines_n": used_counts.get(op, 0),
            "in_fixed_contract": int(op in contract_ids_by_mode.get("fixed", set())),
            "in_flexible_contract": int(op in contract_ids_by_mode.get("flexible", set())),
            "missing_from_all_contracts": int(op not in all_contract_ids),
            "fixed_selected_usage_n": used_by_mode.get(("fixed", op), 0) + used_by_mode.get(("downstream_compatible", op), 0),
            "flexible_selected_usage_n": used_by_mode.get(("flexible", op), 0),
        })
    write_csv(out_dir / "contract_vs_selected_operator_usage.csv", usage_rows)

    # Candidate sensitivity if full candidate lists exist in result.json.
    sensitivity_rows = analyze_candidate_sensitivity(pipeline_rows, out_dir)

    summary = {
        "fixed_contract_path": str(fixed_contract_path),
        "flex_contract_path": str(flex_contract_path),
        "fixed_operator_count": len(contract_ids_by_mode.get("fixed", set())),
        "flexible_operator_count": len(contract_ids_by_mode.get("flexible", set())),
        "used_operator_count": len(used_counts),
        "used_operators_missing_from_all_contracts": sorted([op for op in used_counts if op not in all_contract_ids]),
        "operators_with_missing_required_fields": sum(1 for r in operator_validation if coerce_int(r.get("missing_required_count"), 0) > 0),
        "candidate_sensitivity_rows": len(sensitivity_rows),
    }
    write_json(out_dir / "contract_validation_overview.json", summary)
    return summary


def candidate_list_from_result(result: Any) -> List[Dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    if isinstance(result.get("candidates"), list):
        return [c for c in result["candidates"] if isinstance(c, dict)]
    cgr = result.get("candidate_generation_result")
    if isinstance(cgr, dict) and isinstance(cgr.get("candidates"), list):
        return [c for c in cgr["candidates"] if isinstance(c, dict)]
    ps = result.get("pipeline_selection_result")
    if isinstance(ps, dict):
        for key in ["candidates", "feasible_candidates"]:
            if isinstance(ps.get(key), list):
                return [c for c in ps[key] if isinstance(c, dict)]
        sel = ps.get("selector")
        if isinstance(sel, dict):
            for key in ["candidates", "feasible_candidates"]:
                if isinstance(sel.get(key), list):
                    return [c for c in sel[key] if isinstance(c, dict)]
    return []


def candidate_feasible(c: Dict[str, Any]) -> bool:
    if c.get("feasible") is True or c.get("ci_decision") == "accept" or c.get("decision") == "accept":
        return True
    if c.get("ci_decision") in {"reject", "deny"} or c.get("decision") in {"reject", "deny"}:
        return False
    # If no feasibility marker is available but candidate exists in selected result list, include for descriptive sensitivity.
    return True


def analyze_candidate_sensitivity(pipeline_rows: List[Dict[str, Any]], out_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in pipeline_rows:
        if str(r.get("method_id")) != "full_mediator":
            continue
        if not int(r.get("is_selected_output") or 0):
            continue
        result_path = Path(str(r.get("result_json_path_resolved") or r.get("result_json") or "")) if (r.get("result_json_path_resolved") or r.get("result_json")) else None
        if not result_path or not result_path.exists():
            continue
        result = load_json(result_path, {})
        cands_all = candidate_list_from_result(result)
        cands = [c for c in cands_all if candidate_feasible(c)]
        if len(cands) < 2:
            continue
        selected_id = str(r.get("selected_pipeline_id") or "")
        if not selected_id:
            selected = selected_candidate_from_result(result)
            selected_id = str((selected or {}).get("pipeline_id") or "")
        scored: List[Tuple[str, float, int, Dict[str, str]]] = []
        for c in cands:
            residual = flatten_residual(c)
            if not residual:
                continue
            score = sum(risk_ord(v) for v in residual.values())
            high = sum(1 for v in residual.values() if risk_ord(v) >= 3)
            scored.append((str(c.get("pipeline_id") or ""), float(score), high, residual))
        if len(scored) < 2:
            continue
        uniform_best = sorted(scored, key=lambda x: (x[1], x[2], x[0]))[0]
        high_avoid_best = sorted(scored, key=lambda x: (x[2], x[1], x[0]))[0]
        rows.append({
            "run_mode": r.get("run_mode"),
            "scenario_id": r.get("scenario_id"),
            "selected_pipeline_id": selected_id,
            "candidate_count_available": len(cands),
            "candidate_count_total_in_result": len(cands_all),
            "sensitivity_scope": "selected full_mediator rows; feasible candidates only when markers are available",
            "uniform_risk_best_pipeline_id": uniform_best[0],
            "uniform_risk_best_score": uniform_best[1],
            "uniform_risk_changes_selection": int(bool(selected_id) and uniform_best[0] != selected_id),
            "high_attr_avoid_best_pipeline_id": high_avoid_best[0],
            "high_attr_avoid_best_high_count": high_avoid_best[2],
            "high_attr_avoid_changes_selection": int(bool(selected_id) and high_avoid_best[0] != selected_id),
        })
    write_csv(out_dir / "contract_candidate_selection_sensitivity_if_full_results_available.csv", rows)
    return rows


def load_contexts(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = load_json(path, {})
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    # Current scenario file uses context_scenarios; keep broader aliases for old exports.
    for key in ["context_scenarios", "scenarios", "flows", "items", "contexts", "survey_items"]:
        if isinstance(data.get(key), list):
            return [x for x in data[key] if isinstance(x, dict)]
    return []


def context_scenario_id(x: Dict[str, Any], i: int) -> str:
    params = x.get("ci_parameters_scalar_context_only") or x.get("ci_parameters_scalar") or x.get("ci_parameters") or {}
    mf = x.get("machine_flow_for_ci_constraints_context_only") or x.get("machine_flow_for_ci_constraints") or {}
    return str(x.get("scenario_id") or x.get("flow_id") or params.get("scenario_id") or mf.get("pipeline_id") or f"S{i:03d}")


def _first_scalar(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _context_params(c: Dict[str, Any]) -> Dict[str, Any]:
    params = c.get("ci_parameters_scalar_context_only") or c.get("ci_parameters_scalar") or c.get("ci_parameters") or {}
    if params:
        return dict(params)
    mf = c.get("machine_flow_for_ci_constraints_context_only") or c.get("machine_flow_for_ci_constraints") or {}
    return {
        "task": c.get("task"),
        "context": _first_scalar(mf.get("context")),
        "space": _first_scalar(mf.get("space")),
        "sender": _first_scalar(mf.get("sender")),
        "subject": _first_scalar(mf.get("subject")),
        "recipient": _first_scalar(mf.get("recipient")),
        "purpose": _first_scalar(mf.get("purpose")),
        "transmission_principle": _first_scalar(mf.get("transmissionPrinciple") or mf.get("transmission_principle")),
    }


def analyze_context_coverage(context_path: Path, pipeline_rows: List[Dict[str, Any]], survey_overview: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    contexts = load_contexts(context_path)
    rows_by_scenario = defaultdict(list)
    missing_scenario_rows: List[Dict[str, Any]] = []
    for r in pipeline_rows:
        sid = str(r.get("scenario_id") or "")
        if sid:
            rows_by_scenario[sid].append(r)
        else:
            missing_scenario_rows.append(r)
    context_rows: List[Dict[str, Any]] = []
    context_ids: List[str] = []
    for i, c in enumerate(contexts, start=1):
        sid = context_scenario_id(c, i)
        context_ids.append(sid)
        params = _context_params(c)
        rs = rows_by_scenario.get(sid, [])
        selected_by_method = {str(r.get("run_mode")) + ":" + str(r.get("method_id")): int(r.get("is_selected_output") or 0) for r in rs}
        rows_by_run = Counter(str(r.get("run_mode") or "") for r in rs)
        rows_by_method = Counter(str(r.get("method_id") or "") for r in rs)
        context_rows.append({
            "scenario_id": sid,
            "task": c.get("task") or params.get("task"),
            "context_family": c.get("context_family") or params.get("context_family"),
            "context": params.get("context"),
            "space": params.get("space"),
            "sender": params.get("sender"),
            "subject": params.get("subject"),
            "recipient": params.get("recipient"),
            "purpose": params.get("purpose"),
            "transmission_principle": params.get("transmission_principle"),
            "pipeline_rows_found": len(rs),
            "pipeline_rows_by_run_json": dict(rows_by_run),
            "pipeline_rows_by_method_json": dict(rows_by_method),
            "selected_by_run_method_json": selected_by_method,
        })
    write_csv(out_dir / "context_pool_pipeline_coverage.csv", context_rows)
    pipeline_ids = sorted(rows_by_scenario.keys())
    context_id_set = set(context_ids)
    debug = {
        "context_path": str(context_path),
        "context_file_exists": context_path.exists(),
        "context_ids_sample": context_ids[:10],
        "pipeline_scenario_ids_sample": pipeline_ids[:10],
        "context_ids_missing_pipeline_rows": sorted([sid for sid in context_ids if sid not in rows_by_scenario]),
        "pipeline_sids_not_in_context_file": sorted([sid for sid in pipeline_ids if sid not in context_id_set]),
        "pipeline_rows_missing_scenario_id_count": len(missing_scenario_rows),
        "pipeline_run_method_counts": {
            f"{run_mode}:{method_id}": count
            for (run_mode, method_id), count in Counter((str(r.get("run_mode")), str(r.get("method_id"))) for r in pipeline_rows).items()
        },
    }
    try:
        root_data = load_json(context_path, {}) if context_path.exists() else {}
        if isinstance(root_data, dict):
            debug["context_root_keys"] = list(root_data.keys())
            debug["context_list_keys"] = {k: len(v) for k, v in root_data.items() if isinstance(v, list)}
    except Exception as e:
        debug["context_load_error"] = str(e)
    write_json(out_dir / "context_coverage_debug.json", debug)
    summary = {
        "context_path": str(context_path),
        "context_count": len(contexts),
        "contexts_with_pipeline_rows": sum(1 for r in context_rows if coerce_int(r.get("pipeline_rows_found"), 0) > 0),
        "context_ids_missing_pipeline_rows_count": len(debug["context_ids_missing_pipeline_rows"]),
        "pipeline_sids_not_in_context_file_count": len(debug["pipeline_sids_not_in_context_file"]),
    }
    write_json(out_dir / "context_coverage_overview.json", summary)
    return summary


def write_markdown_report(out_dir: Path, pipeline_summary: Dict[str, Any], survey_overview: Dict[str, Any], contract_summary: Dict[str, Any], context_summary: Dict[str, Any]) -> None:
    dec = pipeline_summary.get("decision_rows", [])
    resid = pipeline_summary.get("residual_rows", [])
    lines: List[str] = []
    lines.append("# PETS Artifact Insights Report\n")
    lines.append("This report is generated by `pets_artifact_insights.py`. It is intended to create reviewer-facing tables, not to replace the paper's methodological caveats. In particular, configured-rule compliance should be described as a configured-rule audit, while residual-disclosure tables are a separate descriptive signal.\n")
    lines.append("## Generated files\n")
    for name in [
        "pipeline_selected_outputs_long.csv",
        "pipeline_decision_coverage_by_method.csv",
        "pipeline_residual_exposure_by_method.csv",
        "pipeline_output_schema_by_method.csv",
        "pipeline_operator_usage_by_method.csv",
        "pipeline_rejection_diagnostics.csv",
        "survey_participant_qc.csv",
        "survey_method_ratings_long.csv",
        "survey_method_rating_summary.csv",
        "survey_output_variant_rating_summary.csv",
        "survey_context_family_rating_summary.csv",
        "survey_free_text_by_scenario.json",
        "contract_operator_validation.csv",
        "contract_vs_selected_operator_usage.csv",
        "contract_vocabulary_residual_attribute_coverage.csv",
        "contract_candidate_selection_sensitivity_if_full_results_available.csv",
        "context_pool_pipeline_coverage.csv",
    ]:
        lines.append(f"- `{name}`")
    lines.append("\n## Pipeline decision coverage snapshot\n")
    if dec:
        lines.append("| run_mode | method_id | rows | selected | not selected/rejected | selected rate |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in dec:
            lines.append(f"| {r.get('run_mode')} | {r.get('method_id')} | {r.get('n_rows')} | {r.get('selected')} | {r.get('not_selected_or_rejected')} | {fmt_float(r.get('selected_rate'))} |")
    else:
        lines.append("No pipeline rows were loaded.\n")
    lines.append("\n## Residual exposure snapshot for selected outputs\n")
    if resid:
        lines.append("| run_mode | method_id | selected_n | raw-like releases | mean risk sum | mean high attrs | mean med/high attrs |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in resid:
            lines.append(f"| {r.get('run_mode')} | {r.get('method_id')} | {r.get('selected_n')} | {r.get('raw_like_release_n')} | {fmt_float(r.get('mean_residual_risk_sum'))} | {fmt_float(r.get('mean_high_attr_count'))} | {fmt_float(r.get('mean_med_or_high_attr_count'))} |")
    lines.append("\n## Survey overview\n")
    if survey_overview.get("available"):
        lines.append(f"- Sessions: {survey_overview.get('session_count')}; completed: {survey_overview.get('completed_session_count')}")
        lines.append(f"- Responses analyzed: {survey_overview.get('response_count_analyzed')} / total {survey_overview.get('response_count_total')}")
        lines.append(f"- Sessions with attention-check failures: {survey_overview.get('attention_failed_sessions')}")
        lines.append(f"- Sessions with comprehension-check failures: {survey_overview.get('comprehension_failed_sessions')}")
        lines.append(f"- Mixed model: {survey_overview.get('mixed_model')}")
    else:
        lines.append("Survey DB not found or empty.")
    lines.append("\n## Contract validation overview\n")
    lines.append(f"- Fixed contract operators: {contract_summary.get('fixed_operator_count')}")
    lines.append(f"- Flexible contract operators: {contract_summary.get('flexible_operator_count')}")
    lines.append(f"- Used operators in selected pipelines: {contract_summary.get('used_operator_count')}")
    lines.append(f"- Used operators missing from all contracts: {contract_summary.get('used_operators_missing_from_all_contracts')}")
    lines.append(f"- Operators with missing required fields: {contract_summary.get('operators_with_missing_required_fields')}")
    lines.append("\n## Context coverage\n")
    lines.append(f"- Context scenarios loaded: {context_summary.get('context_count')}")
    lines.append(f"- Context scenarios with pipeline rows: {context_summary.get('contexts_with_pipeline_rows')}")
    (out_dir / "README_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_float(x: Any) -> str:
    try:
        if x is None:
            return ""
        f = float(x)
        if math.isnan(f):
            return ""
        return f"{f:.3f}"
    except Exception:
        return str(x)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate PETS reviewer-facing insights from privMediator artifacts.")
    cwd = Path.cwd().resolve()
    root = infer_project_root(cwd)
    parser.add_argument("--project-root", default=str(root), help="Project root. Defaults to parent if run from evaluation/.")
    parser.add_argument("--fixed-run", default="../runs/context_pipeline_generation", help="Fixed/downstream-compatible pipeline-generation run dir.")
    parser.add_argument("--flex-run", default="../runs/flexible_context_pipeline_generation", help="Flexible pipeline-generation run dir.")
    parser.add_argument("--responses-db", action="append", default=None, help="Survey SQLite DB path. Can be repeated, e.g., --responses-db responses.db --responses-db responses_old.db. Comma-separated values are also accepted.")
    parser.add_argument("--responses-dbs", default=None, help="Optional comma-separated survey DB paths; merged with --responses-db.")
    parser.add_argument("--fixed-contracts", default="../norms/operator_contracts.json")
    parser.add_argument("--flex-contracts", default="../norms/operator_contracts_flexible.json")
    parser.add_argument("--fixed-vocab", default="../norms/pipeline_vocabulary.json")
    parser.add_argument("--flex-vocab", default="../norms/pipeline_vocabulary_flexible.json")
    parser.add_argument("--contexts", default="../survey/data/ci_focused_user_study_context.json")
    parser.add_argument("--out-dir", default="../runs/pets_artifact_insights")
    parser.add_argument("--include-incomplete-survey", action="store_true", help="Include incomplete survey sessions in rating summaries.")
    parser.add_argument("--bootstrap-iters", type=int, default=1000, help="Cluster bootstrap iterations for survey CI summaries; set 0 to disable.")
    parser.add_argument("--mixed-model", action="store_true", help="Try statsmodels MixedLM for principal methods.")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()

    def to_abs(p: str) -> Path:
        pp = Path(p)
        if pp.is_absolute():
            return pp
        # If run from evaluation/, '../runs/...' should resolve relative to cwd.
        return (cwd / pp).resolve()

    out_dir = to_abs(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_run = to_abs(args.fixed_run)
    flex_run = to_abs(args.flex_run)
    fixed_rows = read_summary_rows(fixed_run, "fixed", project_root)
    flex_rows = read_summary_rows(flex_run, "flexible", project_root)
    write_csv(out_dir / "run_fixed_reconstructed_summary.csv", fixed_rows)
    write_csv(out_dir / "run_flexible_reconstructed_summary.csv", flex_rows)
    write_json(out_dir / "run_reconstructed_summary_overview.json", {
        "fixed_rows": len(fixed_rows),
        "flexible_rows": len(flex_rows),
        "fixed_methods": sorted({str(r.get("method_id")) for r in fixed_rows}),
        "flexible_methods": sorted({str(r.get("method_id")) for r in flex_rows}),
    })
    pipeline_rows = enrich_pipeline_rows(fixed_rows + flex_rows, project_root, out_dir)
    pipeline_summary = aggregate_pipeline_summaries(pipeline_rows, out_dir)

    response_db_args: List[str] = []
    if args.responses_db:
        for item in args.responses_db:
            response_db_args.extend([x.strip() for x in str(item).split(",") if x.strip()])
    if args.responses_dbs:
        response_db_args.extend([x.strip() for x in str(args.responses_dbs).split(",") if x.strip()])
    if not response_db_args:
        response_db_args = ["../survey/outputs/responses.db"]
    response_db_paths = [to_abs(p) for p in response_db_args]

    survey_overview = analyze_survey_db(
        response_db_paths, out_dir,
        include_incomplete=args.include_incomplete_survey,
        bootstrap_iters=args.bootstrap_iters,
        mixed_model=args.mixed_model,
    )
    contract_summary = analyze_contracts(
        to_abs(args.fixed_contracts), to_abs(args.flex_contracts),
        to_abs(args.fixed_vocab), to_abs(args.flex_vocab),
        pipeline_rows, out_dir,
    )
    context_summary = analyze_context_coverage(to_abs(args.contexts), pipeline_rows, survey_overview, out_dir)
    write_markdown_report(out_dir, pipeline_summary, survey_overview, contract_summary, context_summary)
    write_json(out_dir / "run_config.json", {
        "project_root": str(project_root),
        "cwd": str(cwd),
        "fixed_run": str(fixed_run),
        "flex_run": str(flex_run),
        "responses_db": [str(p) for p in response_db_paths],
        "fixed_contracts": str(to_abs(args.fixed_contracts)),
        "flex_contracts": str(to_abs(args.flex_contracts)),
        "fixed_vocab": str(to_abs(args.fixed_vocab)),
        "flex_vocab": str(to_abs(args.flex_vocab)),
        "contexts": str(to_abs(args.contexts)),
        "out_dir": str(out_dir),
    })
    log(f"Wrote analysis artifacts to {out_dir}")
    log(f"Start with: {out_dir / 'README_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
