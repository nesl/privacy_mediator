#!/usr/bin/env python3
"""Compact SmartPriv pipeline-generation artifacts for long experiment sweeps.

The planner/evaluator/selector need rich candidate structures in memory, but the
full structures are usually unnecessary on disk.  These helpers preserve the
selected pipeline, small summaries, counts, and diagnostics needed by survey and
utility evaluation while dropping per-candidate CI flows, full candidate lists,
and large rejected/feasible rankings.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # tqdm is optional; fall back to stderr counters.
    _tqdm = None


LARGE_STAGE_FILENAMES = [
    "candidate_pipelines.json",
    "ci_evaluation.json",
    "pipeline_selection.json",
    "privacy_probe_stage_result.json",
    "full_mediator_result.json",
    "result.json",
]


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def cap_type(cap: Optional[Dict[str, Any]]) -> str:
    if not cap:
        return ""
    return str(cap.get("semantic_type") or cap.get("media_type") or "")


def cap_schema(cap: Optional[Dict[str, Any]]) -> str:
    if not cap:
        return ""
    return str(cap.get("schema") or "")


def decision_text(result: Dict[str, Any]) -> str:
    d = result.get("decision")
    if isinstance(d, dict):
        return str(d.get("decision") or d.get("status") or "")
    return str(d or "")


def selected_pipeline_id(result: Dict[str, Any]) -> Optional[str]:
    d = result.get("decision")
    if isinstance(d, dict) and d.get("selected_pipeline_id"):
        return str(d.get("selected_pipeline_id"))
    if result.get("selected_pipeline_id"):
        return str(result.get("selected_pipeline_id"))
    sel = result.get("selected") or result.get("selected_candidate")
    if isinstance(sel, dict) and sel.get("pipeline_id"):
        return str(sel.get("pipeline_id"))
    return None


def operator_ids(candidate: Optional[Dict[str, Any]]) -> List[str]:
    if not candidate:
        return []
    out: List[str] = []
    for op in candidate.get("operators", []) or []:
        if isinstance(op, str):
            out.append(op)
        elif isinstance(op, dict):
            oid = op.get("operator") or op.get("operator_id") or op.get("id")
            if oid:
                out.append(str(oid))
    return out


def slim_operator(op: Any) -> Dict[str, Any]:
    if isinstance(op, str):
        return {"operator": op}
    if not isinstance(op, dict):
        return {"operator": str(op)}
    out = {"operator": op.get("operator") or op.get("operator_id") or op.get("id")}
    params = op.get("parameters") or op.get("params") or {}
    if params:
        out["parameters"] = params
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def compact_candidate(candidate: Optional[Dict[str, Any]], *, include_executable_spec: bool = True) -> Optional[Dict[str, Any]]:
    if not isinstance(candidate, dict):
        return None
    final_cap = candidate.get("final_output_cap") or {}
    out: Dict[str, Any] = {
        "pipeline_id": candidate.get("pipeline_id"),
        "matched_output_cap": candidate.get("matched_output_cap"),
        "matched_output_schema": candidate.get("matched_output_schema"),
        "matched_output_metadata": candidate.get("matched_output_metadata"),
        "final_output_cap": final_cap,
        "final_output_type": cap_type(final_cap),
        "final_output_schema": cap_schema(final_cap),
        "residual_score": candidate.get("residual_score"),
        "residual_disclosure": candidate.get("residual_disclosure"),
        "utility_score": candidate.get("utility_score"),
        "utility_margin": candidate.get("utility_margin"),
        "quality_status": candidate.get("quality_status"),
        "executable_under_catalog": candidate.get("executable_under_catalog"),
        "operators": [slim_operator(op) for op in (candidate.get("operators", []) or [])],
        "operator_chain": operator_ids(candidate),
        "transforms": candidate.get("transforms"),
    }
    # Keep CI terms for the selected candidate only; this is useful for auditing
    # without storing every candidate's full constructed CI flow.
    if candidate.get("ci_terms") is not None:
        out["ci_terms"] = candidate.get("ci_terms")
    if include_executable_spec and candidate.get("executable_pipeline_spec") is not None:
        spec = candidate.get("executable_pipeline_spec") or {}
        out["executable_pipeline_spec"] = {
            "pipeline_id": spec.get("pipeline_id", candidate.get("pipeline_id")),
            "stages": spec.get("stages", []),
            "final_output_cap": spec.get("final_output_cap") or final_cap,
        }
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def compact_candidate_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
    final_cap = candidate.get("final_output_cap") or {}
    return {
        "pipeline_id": candidate.get("pipeline_id"),
        "matched_output_cap": candidate.get("matched_output_cap"),
        "matched_output_schema": candidate.get("matched_output_schema"),
        "final_output_type": cap_type(final_cap),
        "final_output_schema": cap_schema(final_cap),
        "residual_score": candidate.get("residual_score"),
        "residual_disclosure": candidate.get("residual_disclosure"),
        "operator_chain": operator_ids(candidate),
        "quality_status": candidate.get("quality_status"),
        "executable_under_catalog": candidate.get("executable_under_catalog"),
    }


def compact_candidate_generation_result(result: Optional[Dict[str, Any]], *, selected_id: Optional[str] = None, keep_top_k: int = 0) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return result
    candidates = list(result.get("candidates", []) or [])
    selected = None
    if selected_id:
        for cand in candidates:
            if str(cand.get("pipeline_id")) == str(selected_id):
                selected = cand
                break
    top: List[Dict[str, Any]] = []
    if keep_top_k > 0:
        try:
            top = sorted(candidates, key=lambda c: (c.get("residual_score", 10**9), len(c.get("operators", []) or [])))[:keep_top_k]
        except Exception:
            top = candidates[:keep_top_k]
    out = {
        "schema_version": result.get("schema_version", "smartpriv_candidate_generation_output_v1"),
        "request_id": result.get("request_id"),
        "scenario_id": result.get("scenario_id"),
        "planner": result.get("planner") or {},
        "decision": result.get("decision"),
        "candidate_count": len(candidates),
        "selected_candidate": compact_candidate(selected) if selected else None,
        "top_candidate_summaries": [compact_candidate_summary(c) for c in top],
        "note": "Compact artifact: full candidate list omitted. Re-run with --artifact-detail full for debugging.",
    }
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def compact_hard_failure_summary(summary: Any) -> Any:
    if not isinstance(summary, dict):
        return summary
    out = dict(summary)
    # Keep counts and rule ids, but drop verbose per-rule flow snapshots if present.
    for key in ["matched_rules", "rule_evaluations", "flow", "raw_flow"]:
        out.pop(key, None)
    return out


def compact_ci_evaluation(ev: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pipeline_id": ev.get("pipeline_id"),
        "matched_output_cap": ev.get("matched_output_cap"),
        "residual_score": ev.get("residual_score"),
        "ci_decision": ev.get("ci_decision"),
        "hard_failure_summary": compact_hard_failure_summary(ev.get("hard_failure_summary")),
        "ci_mode": ev.get("ci_mode"),
    }


def compact_no_compromise_diagnostics(diag: Any, *, max_closest: int = 3, max_rules: int = 10) -> Any:
    if not isinstance(diag, dict):
        return diag
    out = dict(diag)
    closest = out.get("closest_rejected_candidates")
    if isinstance(closest, list):
        out["closest_rejected_candidates"] = closest[:max_closest]
    rules = out.get("top_failed_rules")
    if isinstance(rules, list):
        out["top_failed_rules"] = rules[:max_rules]
    return out


def compact_ci_evaluation_result(result: Optional[Dict[str, Any]], *, keep_top_k: int = 0) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return result
    evals = list(result.get("evaluations", []) or [])
    if keep_top_k > 0:
        evals_to_keep = sorted(evals, key=lambda e: e.get("residual_score", 10**9))[:keep_top_k]
    else:
        evals_to_keep = []
    feasible_count = sum(1 for ev in evals if ((ev.get("ci_decision") or {}).get("feasible")))
    hard_reject_count = sum(1 for ev in evals if ((ev.get("ci_decision") or {}).get("decision")) == "reject_hard_constraint")
    out = {
        "schema_version": result.get("schema_version", "smartpriv_ci_evaluation_output_v1"),
        "request_id": result.get("request_id"),
        "scenario_id": result.get("scenario_id"),
        "evaluator": result.get("evaluator") or {},
        "decision": result.get("decision"),
        "no_compromise_diagnostics": compact_no_compromise_diagnostics(result.get("no_compromise_diagnostics")),
        "evaluation_count": len(evals),
        "feasible_count": feasible_count,
        "hard_rejection_count": hard_reject_count,
        "evaluation_summaries": [compact_ci_evaluation(ev) for ev in evals_to_keep],
        "note": "Compact artifact: per-candidate CI flows/rule traces omitted. Re-run with --artifact-detail full for debugging.",
    }
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def compact_selection_record(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None
    keys = [
        "pipeline_id", "matched_output_cap", "matched_output_metadata", "boundary_role", "disclosure_tier",
        "adapter", "execution_mode", "ci_feasible", "raw_ci_feasible", "residual_feasible",
        "raw_residual_feasible", "risk_score", "utility_margin", "latency_ms", "implementation_cost",
        "accepted_cap_priority", "operators", "ci_decision", "hard_failure_summary", "final_residual",
    ]
    return {k: record.get(k) for k in keys if record.get(k) not in (None, "", [], {})}


def compact_pipeline_selection_result(result: Optional[Dict[str, Any]], *, keep_ranked_k: int = 5, keep_rejected_k: int = 3) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return result
    feasible = list(result.get("feasible_ranked", []) or [])
    rejected = list(result.get("rejected", []) or [])
    out = {
        "schema_version": result.get("schema_version", "smartpriv_pipeline_selection_output_v1"),
        "request_id": result.get("request_id"),
        "scenario_id": result.get("scenario_id"),
        "selector": result.get("selector") or {},
        "decision": result.get("decision"),
        "selected": compact_selection_record(result.get("selected")),
        "feasible_ranked_top": [compact_selection_record(r) for r in feasible[:keep_ranked_k]],
        "rejected_top": [compact_selection_record(r) for r in rejected[:keep_rejected_k]],
        "note": "Compact artifact: full feasible/rejected rankings omitted. Re-run with --artifact-detail full for debugging.",
    }
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def compact_probe_stage_result(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return result
    # Probe results can become large if they include per-artifact raw evidence.
    keep = ["schema_version", "status", "reason", "selected_pipeline_id", "pipeline_probe_results", "summary"]
    out = {k: result.get(k) for k in keep if k in result}
    # If per-artifact details are present, keep only counts/status summaries.
    if isinstance(out.get("pipeline_probe_results"), dict):
        pr = out["pipeline_probe_results"]
        out["pipeline_probe_results"] = {
            pid: {
                "status": val.get("status") if isinstance(val, dict) else None,
                "combined_residual": val.get("combined_residual") if isinstance(val, dict) else None,
                "probe_residual": val.get("probe_residual") if isinstance(val, dict) else None,
            }
            for pid, val in pr.items()
        }
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def compact_mediator_result(result: Optional[Dict[str, Any]], *, keep_top_k: int = 0) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return result
    sid = selected_pipeline_id(result)
    out = {
        "schema_version": result.get("schema_version", "smartpriv_full_mediator_output_v2_compact"),
        "request_id": result.get("request_id"),
        "scenario_id": result.get("scenario_id"),
        "request_mode": result.get("request_mode"),
        "ablation_modes": result.get("ablation_modes"),
        "stages": result.get("stages"),
        "decision": result.get("decision"),
        "no_compromise_diagnostics": compact_no_compromise_diagnostics(result.get("no_compromise_diagnostics")),
        "selected_candidate": compact_candidate(result.get("selected_candidate")),
        "candidate_generation_result": compact_candidate_generation_result(result.get("candidate_generation_result"), selected_id=sid, keep_top_k=keep_top_k),
        "ci_evaluation_result": compact_ci_evaluation_result(result.get("ci_evaluation_result"), keep_top_k=keep_top_k),
        "privacy_probe_stage_result": compact_probe_stage_result(result.get("privacy_probe_stage_result")),
        "pipeline_selection_result": compact_pipeline_selection_result(result.get("pipeline_selection_result")),
        "preliminary_pipeline_selection_result": compact_pipeline_selection_result(result.get("preliminary_pipeline_selection_result")),
        "note": "Compact artifact: full candidate/evaluation traces omitted. Re-run with --artifact-detail full for debugging.",
    }
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def compact_stage_file(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return False
    name = path.name
    try:
        data = load_json(path)
    except Exception:
        return False
    if name in {"result.json", "full_mediator_result.json"}:
        compacted = compact_mediator_result(data)
    elif name == "candidate_pipelines.json":
        compacted = compact_candidate_generation_result(data)
    elif name == "ci_evaluation.json":
        compacted = compact_ci_evaluation_result(data)
    elif name == "pipeline_selection.json":
        compacted = compact_pipeline_selection_result(data)
    elif name == "privacy_probe_stage_result.json":
        compacted = compact_probe_stage_result(data)
    else:
        return False
    if compacted is None:
        return False
    try:
        before = path.stat().st_size
        serialized = json.dumps(compacted, indent=2, sort_keys=False) + "\n"
        # Avoid rewriting already-compact or tiny diagnostic files into a larger
        # pretty-printed artifact.  Large full-debug files should still shrink
        # dramatically and will be rewritten.
        if len(serialized.encode("utf-8")) >= before:
            return False
        path.write_text(serialized, encoding="utf-8")
        return True
    except Exception:
        # Fall back to the simple writer if stat/encoding comparison fails.
        write_json(compacted, path)
        return True


def human_bytes(n: int | float) -> str:
    value = float(n or 0)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TB"


def discover_large_stage_files(root: str | Path, *, progress: bool = False) -> List[Path]:
    """Return candidate JSON artifact paths under ``root``.

    Discovery is separated from compaction so the condense command can show a
    determinate progress bar rather than appearing stuck on a recursive rglob.
    """
    root = Path(root)
    if not root.exists():
        return []
    files: List[Path] = []
    for name in LARGE_STAGE_FILENAMES:
        if progress:
            print(f"[condense] scanning for {name} under {root}", file=sys.stderr, flush=True)
        files.extend(sorted(root.rglob(name)))
    # Deduplicate while preserving deterministic order.
    seen = set()
    out: List[Path] = []
    for p in sorted(files):
        key = str(p)
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out


def _iter_with_progress(paths: Sequence[Path], *, enabled: bool, desc: str, progress_every: int) -> Iterator[Path]:
    total = len(paths)
    if not enabled:
        yield from paths
        return
    if _tqdm is not None:
        with _tqdm(total=total, desc=desc, unit="file", dynamic_ncols=True) as bar:
            for path in paths:
                try:
                    bar.set_postfix_str(path.name, refresh=False)
                except Exception:
                    pass
                yield path
                bar.update(1)
        return

    # Fallback for environments without tqdm.
    progress_every = max(1, int(progress_every or 25))
    started = time.time()
    for i, path in enumerate(paths, start=1):
        if i == 1 or i == total or i % progress_every == 0:
            elapsed = time.time() - started
            print(f"[condense] {i}/{total} files ({elapsed:.1f}s): {path.name}", file=sys.stderr, flush=True)
        yield path


def condense_run_tree(
    root: str | Path,
    *,
    progress: bool = False,
    progress_every: int = 25,
    min_bytes: int = 0,
) -> Dict[str, Any]:
    """Condense large generation artifacts under ``root`` with optional progress.

    Parameters
    ----------
    root:
        Run tree to scan.
    progress:
        If true, show scanning messages and a tqdm/simple progress indicator.
    progress_every:
        Fallback counter frequency when tqdm is unavailable.
    min_bytes:
        Skip files smaller than this threshold.  The default, 0, preserves the
        previous behavior and rewrites every recognized artifact.
    """
    root = Path(root)
    report: Dict[str, Any] = {
        "root": str(root),
        "files_seen": 0,
        "files_compacted": 0,
        "files_skipped_small": 0,
        "files_unchanged_or_already_compact": 0,
        "bytes_before": 0,
        "bytes_after": 0,
        "bytes_saved": 0,
        "errors": [],
    }
    if not root.exists():
        return report

    paths = discover_large_stage_files(root, progress=progress)
    report["files_discovered"] = len(paths)
    if progress:
        print(f"[condense] discovered {len(paths)} candidate artifact files", file=sys.stderr, flush=True)

    for path in _iter_with_progress(paths, enabled=progress, desc="Condensing artifacts", progress_every=progress_every):
        try:
            before = path.stat().st_size
            report["files_seen"] += 1
            if min_bytes and before < int(min_bytes):
                report["files_skipped_small"] += 1
                continue
            report["bytes_before"] += before
            if compact_stage_file(path):
                after = path.stat().st_size
                report["bytes_after"] += after
                if after < before:
                    report["files_compacted"] += 1
                else:
                    report["files_unchanged_or_already_compact"] += 1
            else:
                report["bytes_after"] += before
                report["files_unchanged_or_already_compact"] += 1
        except Exception as exc:  # pragma: no cover - defensive cleanup path
            report["errors"].append({"path": str(path), "error": repr(exc)})
    report["bytes_saved"] = int(report["bytes_before"] - report["bytes_after"])
    report["bytes_before_human"] = human_bytes(report["bytes_before"])
    report["bytes_after_human"] = human_bytes(report["bytes_after"])
    report["bytes_saved_human"] = human_bytes(report["bytes_saved"])
    if progress:
        print(
            "[condense] done: "
            f"compacted={report['files_compacted']} "
            f"seen={report['files_seen']} "
            f"saved={report['bytes_saved_human']} "
            f"errors={len(report['errors'])}",
            file=sys.stderr,
            flush=True,
        )
    return report
