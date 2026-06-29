#!/usr/bin/env python3
"""Timing-focused dynamic-context overhead evaluation for SmartPriv/Prism.

This script complements ``evaluation.dynamic_context_experiments``.  It uses the
same dynamic context traces, app requests, baselines, and mediator components,
but records latency/overhead rather than primarily reporting utility.  It is
intended for an experimental section about context-switch overhead and the cost
of mediator components/ablations.

What it measures
----------------
1. Context materialization time: create the context-specific app request and
   environment for a dynamic phase.
2. Decision / pipeline-generation time: run raw/manual/direct LLM/full mediator
   or a full-mediator ablation for the current context.
3. Full-mediator stage breakdown: candidate enumeration, CI evaluation,
   preliminary selection, optional probes, and final selection.
4. Optional tiny utility pass: run ``evaluation.evaluate_utility`` on selected
   outputs only, with cache disabled, and derive an approximate
   preprocessing/wrapper time as ``elapsed_ms - downstream_elapsed_ms``.

Typical use from the project root:

    python -m evaluation.dynamic_context_overhead \
      --out-dir runs/dynamic_context_overhead \
      --trace-set all \
      --methods raw,manual,direct_llm,full_mediator \
      --ablation-modes utility_only,no_ci_filter,no_least_revealing,latency_first \
      --repeats 3 \
      --utility-max-samples 1 \
      --utility-max-frames-per-sample 24 \
      --yes

For decision/generation timing only:

    python -m evaluation.dynamic_context_overhead --out-dir runs/overhead_gen_only --no-utility --repeats 5 --yes

Outputs under --out-dir:
  overhead_timing.csv                 per-run timing rows
  overhead_summary_by_method.csv      aggregate latency by method/ablation
  overhead_summary_by_trace.json      per-trace aggregate latency
  pipeline_generation/summary.csv     selected rows from one replicate for utility
  utility_eval/                       optional one-sample utility timing pass
  runtime_timing_summary.csv          utility timings plus approximate preprocessing time
  overhead_run_summary.json           run metadata and file index
"""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import inspect
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

_PROGRESS_ENABLED = True

DEFAULT_BASE_METHODS = ["raw", "manual", "direct_llm", "full_mediator"]
DEFAULT_ABLATION_MODES = [
    "utility_only",
    "no_ci_filter",
    "no_residual_bounds",
    "no_least_revealing",
    "uniform_risk_weights",
    "metadata_only",
    "no_staged_flows",
    "first_feasible",
    "latency_first",
]


def set_progress_enabled(enabled: bool) -> None:
    global _PROGRESS_ENABLED
    _PROGRESS_ENABLED = bool(enabled)


def progress_write(msg: str) -> None:
    if not _PROGRESS_ENABLED:
        return
    if _tqdm is not None:
        try:
            _tqdm.write(str(msg))
            return
        except Exception:
            pass
    print(str(msg), file=sys.stderr, flush=True)


def progress_iter(iterable, *, total: int, desc: str, unit: str = "item"):
    if _PROGRESS_ENABLED and _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)
    return iterable


def now_ns() -> int:
    return time.perf_counter_ns()


def elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def parse_csv_list(text: Optional[str], default: Optional[Sequence[str]] = None) -> List[str]:
    if text is None or str(text).strip() == "":
        return list(default or [])
    return [x.strip() for x in str(text).split(",") if x.strip()]


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


def write_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(rows: Sequence[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def import_module_from_path(module_name: str, path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not import {module_name}: {path} does not exist")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def import_module_or_path(module_name: str, optional_path: Optional[str | Path] = None):
    if optional_path:
        p = Path(optional_path)
        if p.exists():
            return import_module_from_path(module_name.replace(".", "_"), p)
    return importlib.import_module(module_name)


def call_with_supported_kwargs(fn, kwargs: Dict[str, Any]):
    sig = inspect.signature(fn)
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def load_dynamic_module(args: argparse.Namespace):
    if args.dynamic_context_module_path:
        return import_module_from_path("dynamic_context_experiments_for_overhead", args.dynamic_context_module_path)
    return importlib.import_module(args.dynamic_context_module)


def method_slug(method_id: str) -> str:
    return method_id.replace(":", "__").replace("/", "_")


def method_label(method_id: str) -> str:
    if method_id.startswith("ablation:"):
        return "Ablation: " + method_id.split(":", 1)[1]
    return {
        "raw": "Raw",
        "manual": "Manual",
        "direct_llm": "Direct LLM",
        "full_mediator": "Full mediator",
    }.get(method_id, method_id)


def p50(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return float(statistics.median(vals))


def mean(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return float(statistics.mean(vals))


def percentile(vals: Sequence[float], q: float) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(float(v) for v in vals)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def float_or_none(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def int_bool(x: Any) -> int:
    return 1 if bool(x) else 0


def boolish(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    text = str(x).strip().lower()
    return text in {"1", "true", "yes", "y", "warmup"}


def is_warmup_row(row: Dict[str, Any]) -> bool:
    return boolish(row.get("is_warmup"))


def overhead_row_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("method_id") or ""),
        str(row.get("replicate_label") or ("warmup" + str(row.get("replicate")) if boolish(row.get("is_warmup")) else "rep" + str(row.get("replicate")))),
        str(row.get("dynamic_trace_id") or ""),
    )


def existing_overhead_row_is_complete(row: Dict[str, Any]) -> bool:
    decision = str(row.get("decision") or "")
    if not decision:
        return False
    result_json = str(row.get("result_json") or "")
    if result_json and not Path(result_json).exists():
        return False
    if decision == "select_pipeline":
        spec = str(row.get("pipeline_spec_json") or "")
        if spec and not Path(spec).exists():
            return False
    return True


def load_existing_overhead_rows(out_root: Path) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    path = out_root / "overhead_timing.csv"
    if not path.exists():
        return {}
    rows: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    try:
        for r in read_csv(path):
            if existing_overhead_row_is_complete(r):
                rows[overhead_row_key(r)] = r
    except Exception as exc:
        progress_write(f"[resume] could not read existing overhead timing rows from {path}: {exc!r}")
    return rows


def existing_runtime_summary_paths(out_root: Path, utility_out: Path) -> Dict[str, Path]:
    return {
        "runtime_csv": out_root / "runtime_timing_summary.csv",
        "runtime_json": out_root / "runtime_timing_summary.json",
        "utility_summary_csv": utility_out / "utility_summary.csv",
        "utility_log": utility_out / "overhead_utility_eval.log",
    }


class TimingContext:
    """Small container for preloaded modules and static data."""

    def __init__(self, args: argparse.Namespace, dyn):
        self.args = args
        self.dyn = dyn
        self.operator_catalog = load_json(args.operators)
        self.raw_module = import_module_or_path("preprocessing_baselines.raw_baseline", args.raw_module)
        self.manual_module = import_module_or_path("preprocessing_baselines.manual_baseline", args.manual_module)
        self.direct_module = import_module_or_path("preprocessing_baselines.direct_llm_baseline", args.direct_llm_module)
        self.full_module = import_module_from_path("overhead_full_mediator", args.full_mediator_module)
        self.generator = import_module_from_path("overhead_candidate_generator", args.candidate_generator)
        self.evaluator = import_module_from_path("overhead_ci_evaluator", args.evaluator)
        self.selector = import_module_from_path("overhead_pipeline_selector", args.selector)
        self.constraints = load_json(args.constraints)
        self.selection_config_base = self.full_module.read_optional_json(args.selection_config) if args.selection_config else None

    def explicit_requests(self) -> Dict[str, Optional[str]]:
        return {
            "visitor_presence_detection": self.args.visitor_request,
            "fall_detection": self.args.fall_request,
            "adl_recognition": self.args.adl_request,
            "domestic_sound_monitoring": self.args.audio_request,
        }


def candidate_summary_fields(dyn, cand: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if hasattr(dyn, "candidate_summary_fields"):
        return dyn.candidate_summary_fields(cand)
    if not cand:
        return {
            "selected_pipeline_id": None,
            "matched_output_cap": None,
            "matched_output_schema": None,
            "final_output_type": None,
            "final_output_schema": None,
            "operators": "",
            "residual_score": None,
            "quality_status": None,
            "executable_under_catalog": None,
        }
    final_cap = cand.get("final_output_cap") or {}
    op_ids = []
    for op in cand.get("operators", []) or []:
        oid = op.get("operator") or op.get("operator_id")
        if oid:
            op_ids.append(str(oid))
    return {
        "selected_pipeline_id": cand.get("pipeline_id"),
        "matched_output_cap": cand.get("matched_output_cap"),
        "matched_output_schema": cand.get("matched_output_schema"),
        "final_output_type": str(final_cap.get("semantic_type") or final_cap.get("media_type") or ""),
        "final_output_schema": str(final_cap.get("schema") or ""),
        "operators": " -> ".join(op_ids),
        "residual_score": cand.get("residual_score"),
        "quality_status": cand.get("quality_status"),
        "executable_under_catalog": cand.get("executable_under_catalog"),
    }


def diagnostic_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    diag = result.get("no_compromise_diagnostics") or {}
    decision = result.get("decision") if isinstance(result.get("decision"), dict) else {}
    top_rules = diag.get("top_failed_rules") or (decision.get("selection_failure_diagnostics") or {}).get("top_failed_rules") or []
    stages = result.get("stages") if isinstance(result.get("stages"), dict) else {}
    cand_gen = stages.get("candidate_generation") if isinstance(stages.get("candidate_generation"), dict) else {}
    ci_eval = stages.get("ci_evaluation") if isinstance(stages.get("ci_evaluation"), dict) else {}
    selector = stages.get("pipeline_selection") if isinstance(stages.get("pipeline_selection"), dict) else {}
    ablation = stages.get("ablation") if isinstance(stages.get("ablation"), dict) else {}
    return {
        "candidate_count": cand_gen.get("candidate_count") or diag.get("candidate_count"),
        "states_expanded": cand_gen.get("states_expanded"),
        "states_seen": cand_gen.get("states_seen"),
        "ci_eval_count": ci_eval.get("candidate_count") or ci_eval.get("evaluation_count"),
        "selector_num_candidates": selector.get("num_candidates"),
        "selector_num_feasible": selector.get("num_feasible"),
        "selector_num_rejected": selector.get("num_rejected"),
        "no_compromise_reason": diag.get("reason") or decision.get("reason"),
        "no_compromise_candidate_count": diag.get("candidate_count"),
        "no_compromise_ci_feasible_count": diag.get("ci_feasible_count"),
        "no_compromise_hard_rejection_count": diag.get("hard_rejection_count"),
        "no_compromise_top_failed_rules": json.dumps(top_rules, sort_keys=False),
        "ablation_apply_request_ci_constraints": ablation.get("apply_request_ci_constraints"),
        "ablation_apply_request_residual_constraints": ablation.get("apply_request_residual_constraints"),
        "ablation_preserve_staged_flows": ablation.get("preserve_staged_flows"),
        "ablation_ci_mode": ablation.get("ci_mode"),
    }


def compact_pipeline_spec(dyn, cand: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cand:
        return None
    if hasattr(dyn, "make_stage_specs_from_candidate"):
        stages = dyn.make_stage_specs_from_candidate(cand)
    else:
        stages = []
        for op in cand.get("operators", []) or []:
            oid = op.get("operator") or op.get("operator_id")
            if not oid or oid in {"op.source", "op.route_publish"}:
                continue
            stages.append({"operator_id": oid, "parameters": op.get("parameters") or {}})
    return {
        "schema_version": "smartpriv_saved_pipeline_spec_v1",
        "pipeline_id": cand.get("pipeline_id"),
        "executable_under_catalog": cand.get("executable_under_catalog"),
        "final_output_cap": cand.get("final_output_cap"),
        "matched_output_cap": cand.get("matched_output_cap"),
        "stages": stages,
        "source_candidate_metadata": {
            "operators": cand.get("operators", []),
            "residual_disclosure": cand.get("residual_disclosure"),
            "ci_terms": cand.get("ci_terms"),
            "transforms": cand.get("transforms"),
            "quality_status": cand.get("quality_status"),
        },
    }


def write_compact_artifacts(dyn, result: Dict[str, Any], cand: Optional[Dict[str, Any]], out_dir: Path, save_full_artifacts: bool) -> Dict[str, Optional[str]]:
    """Write enough artifacts for utility evaluation without saving huge stage JSON unless requested."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_full_artifacts:
        # Reuse the dynamic script writer, which saves full candidate/CI/selector JSON.
        return dyn.write_pipeline_code_and_metadata(cand, result, out_dir)

    compact_result = {
        "schema_version": "smartpriv_overhead_compact_result_v1",
        "request_id": result.get("request_id"),
        "scenario_id": result.get("scenario_id"),
        "ablation_modes": result.get("ablation_modes", []),
        "decision": result.get("decision"),
        "stages": result.get("stages", {}),
        "no_compromise_diagnostics": result.get("no_compromise_diagnostics"),
        "selected_candidate": cand,
    }
    write_json(compact_result, out_dir / "result.json")
    paths: Dict[str, Optional[str]] = {"result_json": str(out_dir / "result.json")}
    if not cand:
        write_text("No selected candidate was available for this timing run.\n", out_dir / "NO_SELECTED_PIPELINE.txt")
        paths.update({"selected_pipeline_json": None, "pipeline_spec_json": None})
        return paths

    write_json(cand, out_dir / "selected_pipeline.json")
    spec = compact_pipeline_spec(dyn, cand)
    write_json(spec, out_dir / "pipeline_spec.json")
    paths.update({
        "selected_pipeline_json": str(out_dir / "selected_pipeline.json"),
        "pipeline_spec_json": str(out_dir / "pipeline_spec.json"),
    })
    return paths


def decision_text(dyn, result: Dict[str, Any]) -> str:
    if hasattr(dyn, "decision_text"):
        return dyn.decision_text(result)
    d = result.get("decision")
    if isinstance(d, dict):
        return str(d.get("decision") or d.get("status") or "")
    return str(d or "")


def selected_candidate(dyn, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if hasattr(dyn, "selected_candidate"):
        return dyn.selected_candidate(result)
    if decision_text(dyn, result) != "select_pipeline":
        return None
    sel = result.get("selected_candidate") or result.get("selected")
    return sel if isinstance(sel, dict) else None


def run_timed_full_mediator(ctx: TimingContext, request_path: Path, environment_path: Path, ablation_modes: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Run the full mediator stages directly and return stage timings.

    This mirrors full_mediator.run_mediator but adds wall-clock timing around
    each stage.  Module imports and static JSON loads are done before timing, so
    rows reflect steady-state context-switch latency rather than Python import
    overhead.
    """
    args = ctx.args
    fm = ctx.full_module
    dyn = ctx.dyn
    timings: Dict[str, float] = {}
    total_start = now_ns()

    stage_start = now_ns()
    request = fm.require_structured_request(request_path)
    environment = fm.read_optional_json(environment_path)
    sensor_stream = fm.read_optional_json(args.sensor_stream) if args.sensor_stream else None
    ablation_modes = fm.normalize_ablation_modes(ablation_modes)
    selection_config = fm.merge_selection_config_with_ablations(ctx.selection_config_base, ablation_modes)
    timings["stage_request_load_elapsed_ms"] = elapsed_ms(stage_start)

    ablation_set = set(ablation_modes)
    apply_request_ci_constraints = not bool({"utility_only", "no_ci_filter", "llm_only"} & ablation_set)
    apply_request_residual_constraints = not bool({"utility_only", "no_residual_bounds"} & ablation_set)
    preserve_staged_flows = "no_staged_flows" not in ablation_set

    ci_mode = "full"
    use_llm = bool(args.full_mediator_use_llm)
    if "utility_only" in ablation_set or "no_ci_filter" in ablation_set:
        ci_mode = "no_hard_rules"
    elif "llm_only" in ablation_set:
        ci_mode = "llm_only"
        use_llm = True
    elif "no_staged_flows" in ablation_set:
        ci_mode = "no_staged_flows"

    stage_start = now_ns()
    candidate_result = fm.run_candidate_generation(
        generator=ctx.generator,
        operators=ctx.operator_catalog,
        request=request,
        max_depth=args.max_depth,
        max_states=args.max_states,
        apply_request_ci_constraints=apply_request_ci_constraints,
        apply_request_residual_constraints=apply_request_residual_constraints,
        preserve_staged_flows=preserve_staged_flows,
    )
    timings["stage_candidate_generation_elapsed_ms"] = elapsed_ms(stage_start)

    stage_start = now_ns()
    ci_result = fm.run_ci_evaluation(
        evaluator=ctx.evaluator,
        candidate_result=candidate_result,
        request=request,
        constraints=ctx.constraints,
        environment=environment,
        sensor_stream=sensor_stream,
        use_llm=use_llm,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_confidence_threshold=args.llm_confidence_threshold,
        top_k_for_llm=args.top_k_for_llm,
        ci_mode=ci_mode,
        collapse_stages="no_staged_flows" in ablation_set,
    )
    timings["stage_ci_evaluation_elapsed_ms"] = elapsed_ms(stage_start)

    stage_start = now_ns()
    preliminary_selection = fm.run_pipeline_selection(
        selector=ctx.selector,
        candidate_result=candidate_result,
        ci_result=ci_result,
        request=request,
        probe_stage_result=None,
        selection_config=selection_config,
    )
    timings["stage_preliminary_selection_elapsed_ms"] = elapsed_ms(stage_start)

    preliminary_result = {
        "schema_version": "smartpriv_full_mediator_preliminary_output_v1",
        "request_id": request.get("request_identity", {}).get("request_id"),
        "scenario_id": request.get("request_identity", {}).get("scenario_id"),
        "decision": preliminary_selection.get("decision"),
        "selected_candidate": fm.selected_candidate_from_selection(candidate_result, preliminary_selection),
        "candidate_generation_result": candidate_result,
        "ci_evaluation_result": ci_result,
        "pipeline_selection_result": preliminary_selection,
    }

    effective_probe_artifacts_path = None if "metadata_only" in ablation_set else args.probe_artifacts
    stage_start = now_ns()
    probe_stage_result = fm.run_optional_privacy_probes(
        preliminary_mediator_result=preliminary_result,
        probe_artifacts_path=effective_probe_artifacts_path,
        probe_config_path=args.probe_config,
        probe_package_dir=args.probe_package_dir,
    )
    if "metadata_only" in ablation_set and args.probe_artifacts:
        probe_stage_result["reason"] = "Ablation metadata_only: privacy probes intentionally skipped."
    timings["stage_privacy_probe_elapsed_ms"] = elapsed_ms(stage_start)

    stage_start = now_ns()
    final_selection = fm.run_pipeline_selection(
        selector=ctx.selector,
        candidate_result=candidate_result,
        ci_result=ci_result,
        request=request,
        probe_stage_result=probe_stage_result,
        selection_config=selection_config,
    )
    timings["stage_final_selection_elapsed_ms"] = elapsed_ms(stage_start)

    selected = fm.selected_candidate_from_selection(candidate_result, final_selection)
    no_comp = fm.no_compromise_diagnostics_from_results(ci_result, final_selection)
    result = {
        "schema_version": "smartpriv_full_mediator_output_v2_timed",
        "request_id": request.get("request_identity", {}).get("request_id"),
        "scenario_id": request.get("request_identity", {}).get("scenario_id"),
        "ablation_modes": list(ablation_modes),
        "timing": dict(timings),
        "stages": {
            "structured_request_loaded": True,
            "environment_loaded": environment is not None,
            "sensor_stream_loaded": sensor_stream is not None,
            "candidate_generation": candidate_result.get("planner", {}),
            "ci_evaluation": ci_result.get("evaluator", {}),
            "privacy_probes": {
                "status": probe_stage_result.get("status"),
                "selected_pipeline_id": probe_stage_result.get("selected_pipeline_id"),
                "reason": probe_stage_result.get("reason"),
            },
            "pipeline_selection": final_selection.get("selector", {}),
            "no_compromise_diagnostics_available": no_comp is not None,
            "ablation": {
                "ablation_modes": list(ablation_modes),
                "apply_request_ci_constraints": apply_request_ci_constraints,
                "apply_request_residual_constraints": apply_request_residual_constraints,
                "preserve_staged_flows": preserve_staged_flows,
                "ci_mode": ci_mode,
            },
        },
        "decision": final_selection.get("decision"),
        "no_compromise_diagnostics": no_comp,
        "selected_candidate": selected,
        "candidate_generation_result": candidate_result,
        "ci_evaluation_result": ci_result,
        "privacy_probe_stage_result": probe_stage_result,
        "pipeline_selection_result": final_selection,
        "preliminary_pipeline_selection_result": preliminary_selection,
    }
    timings["stage_total_full_mediator_elapsed_ms"] = elapsed_ms(total_start)
    result["timing"] = dict(timings)
    return result, timings


def prepare_request_for_scenario(ctx: TimingContext, scenario: Dict[str, Any], sid: str, request_dir: Path) -> Tuple[Dict[str, Any], Path, Path, Path, Dict[str, Any], float]:
    dyn = ctx.dyn
    args = ctx.args
    start = now_ns()
    task = dyn.scenario_task(scenario)
    app_request_path = dyn.resolve_request_path(task, args.app_request_dir, ctx.explicit_requests())
    app_request = load_json(app_request_path)
    request = dyn.overlay_context_on_app_request(app_request, scenario, sid)
    environment = dyn.environment_from_dynamic_scenario(scenario)
    request_dir.mkdir(parents=True, exist_ok=True)
    request_path = request_dir / "context_app_request.json"
    scenario_path = request_dir / "context_scenario.json"
    environment_path = request_dir / "environment.json"
    write_json(request, request_path)
    write_json(scenario, scenario_path)
    write_json(environment, environment_path)
    return request, request_path, scenario_path, environment_path, app_request_path, elapsed_ms(start)


def method_specs(base_methods: Sequence[str], ablation_modes: Sequence[str]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for m in base_methods:
        specs.append({"method_id": m, "method_kind": "baseline", "baseline": m, "ablation_mode": None, "parent_method": None})
    for mode in ablation_modes:
        mode = str(mode).strip()
        if not mode:
            continue
        if mode == "full":
            continue
        specs.append({"method_id": f"ablation:{mode}", "method_kind": "ablation", "baseline": f"ablation:{mode}", "ablation_mode": mode, "parent_method": "full_mediator"})
    # Preserve order while removing duplicates.
    seen = set()
    out = []
    for s in specs:
        if s["method_id"] in seen:
            continue
        seen.add(s["method_id"])
        out.append(s)
    return out


def run_one_method(ctx: TimingContext, scenario: Dict[str, Any], method: Dict[str, Any], request: Dict[str, Any], request_path: Path, environment_path: Path, method_dir: Path) -> Tuple[Dict[str, Any], Dict[str, float], Optional[str], Optional[str]]:
    dyn = ctx.dyn
    args = ctx.args
    method_id = method["method_id"]
    timings: Dict[str, float] = {}
    error = None
    tb = None
    start = now_ns()
    try:
        task = dyn.scenario_task(scenario)
        params = dyn.scenario_ci_params(scenario)
        if method_id == "raw":
            result = dyn.run_raw_baseline(ctx.raw_module, ctx.operator_catalog, request, args.candidate_generator)
        elif method_id == "manual":
            result = dyn.run_manual_baseline(
                ctx.manual_module,
                ctx.operator_catalog,
                request,
                args.candidate_generator,
                task=task,
                space=dyn.display_values(params.get("space")),
                max_depth=args.max_depth,
                max_states=args.max_states,
            )
        elif method_id == "direct_llm":
            result = dyn.run_direct_llm_baseline(
                ctx.direct_module,
                ctx.operator_catalog,
                request,
                environment=dyn.environment_from_dynamic_scenario(scenario),
                candidate_generator=args.candidate_generator,
                max_depth=args.max_depth,
                max_states=args.max_states,
                llm_model=args.llm_model,
                llm_temperature=args.llm_temperature,
                openai_api_key=args.openai_api_key,
            )
        elif method_id == "full_mediator":
            result, stage_timings = run_timed_full_mediator(ctx, request_path, environment_path, [])
            timings.update(stage_timings)
        elif method_id.startswith("ablation:"):
            result, stage_timings = run_timed_full_mediator(ctx, request_path, environment_path, [method["ablation_mode"]])
            timings.update(stage_timings)
        else:
            raise ValueError(f"Unknown method {method_id!r}")
    except Exception as exc:
        error = repr(exc)
        tb = traceback.format_exc()
        result = {
            "schema_version": "smartpriv_overhead_method_error_v1",
            "request_id": request.get("request_identity", {}).get("request_id"),
            "scenario_id": request.get("request_identity", {}).get("scenario_id"),
            "method_id": method_id,
            "decision": {"decision": "error", "selected_pipeline_id": None, "reason": error},
            "error": error,
            "traceback": tb,
        }
        write_text(tb, method_dir / "traceback.txt")
        if args.fail_fast:
            raise
    timings["decision_elapsed_ms"] = elapsed_ms(start)
    return result, timings, error, tb


def run_generation_timing(args: argparse.Namespace, dyn, scenarios: Sequence[Dict[str, Any]], trace_meta: Dict[str, Any]) -> Tuple[Path, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    out_root = Path(args.out_dir)
    pipeline_root = out_root / "pipeline_generation"
    pipeline_root.mkdir(parents=True, exist_ok=True)
    ctx = TimingContext(args, dyn)

    methods = method_specs(parse_csv_list(args.methods, DEFAULT_BASE_METHODS), parse_csv_list(args.ablation_modes, DEFAULT_ABLATION_MODES))
    wanted_sids = set(parse_csv_list(args.scenario_ids)) if args.scenario_ids else set()
    active_scenarios = [sc for sc in scenarios if not wanted_sids or str(sc.get("scenario_id") or "") in wanted_sids]

    context_doc = {
        "schema_version": "smartpriv_dynamic_context_overhead_contexts_v1",
        "trace_set": args.trace_set,
        "source_dynamic_contexts": str(args.dynamic_contexts) if args.dynamic_contexts else None,
        "methods": [m["method_id"] for m in methods],
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "traces": trace_meta,
        "context_scenarios": list(active_scenarios),
        "timing_note": "Decision times exclude Python module import and static operator/constraint JSON load. Context materialization includes request overlay and small request/environment writes.",
    }
    write_json(context_doc, out_root / "dynamic_context_overhead_scenarios.json")
    write_json(context_doc, pipeline_root / "dynamic_context_overhead_scenarios.json")

    timing_rows: List[Dict[str, Any]] = []
    utility_rows: List[Dict[str, Any]] = []
    index: Dict[str, Any] = {"schema_version": "dynamic_context_overhead_index_v1", "methods": [m["method_id"] for m in methods], "traces": {}, "contexts": {}}

    existing_rows = {} if args.rerun_existing_overhead else load_existing_overhead_rows(out_root)
    if existing_rows:
        progress_write(f"[resume] loaded {len(existing_rows)} completed overhead timing rows from {out_root / 'overhead_timing.csv'}")

    total = len(active_scenarios) * len(methods) * (args.warmup_runs + args.repeats)
    progress_write(f"[overhead] timing {len(active_scenarios)} dynamic phases x {len(methods)} methods/ablations x {args.warmup_runs}+{args.repeats} runs = {total} method-runs")

    run_counter = 0
    run_items: List[Tuple[Dict[str, Any], Dict[str, Any], int, bool]] = []
    for sc in active_scenarios:
        for method in methods:
            for w in range(args.warmup_runs):
                run_items.append((sc, method, w + 1, True))
            for r in range(args.repeats):
                run_items.append((sc, method, r + 1, False))

    for scenario, method, replicate, is_warmup in progress_iter(run_items, total=len(run_items), desc="overhead timing", unit="run"):
        run_counter += 1
        sid = str(scenario.get("scenario_id") or f"DYN_{run_counter:03d}")
        task = dyn.scenario_task(scenario)
        params = dyn.scenario_ci_params(scenario)
        trace_id = str(scenario.get("dynamic_trace_id") or "dynamic_trace")
        method_id = method["method_id"]
        rep_label = f"warmup{replicate}" if is_warmup else f"rep{replicate}"
        context_dir = pipeline_root / sid
        request_dir = context_dir / "request"
        method_dir = context_dir / "methods" / method_slug(method_id) / rep_label

        resume_key = (sid, method_id, rep_label, trace_id)
        existing_row = existing_rows.get(resume_key)
        if existing_row is not None:
            existing_row = dict(existing_row)
            existing_row["resumed_from_existing"] = True
            timing_rows.append(existing_row)
            if (not is_warmup_row(existing_row)) and int(str(existing_row.get("replicate") or replicate)) == args.utility_replicate and str(existing_row.get("decision") or "") == "select_pipeline":
                utility_rows.append(dict(existing_row))

            index["contexts"].setdefault(sid, {"scenario_id": sid, "task": task, "trace_id": trace_id, "methods": {}})
            index["contexts"][sid]["methods"].setdefault(method_id, []).append(existing_row)
            index["traces"].setdefault(trace_id, {"trace_id": trace_id, "phases": {}, "methods": {}})
            index["traces"][trace_id]["phases"].setdefault(sid, {"scenario_id": sid, "phase_index": scenario.get("dynamic_phase_index"), "methods": {}})["methods"].setdefault(method_id, []).append(existing_row)
            index["traces"][trace_id]["methods"].setdefault(method_id, []).append(existing_row)
            progress_write(f"[resume] {sid}/{method_id}/{rep_label}: using existing decision={existing_row.get('decision')} gen={existing_row.get('generation_total_elapsed_ms')}ms")
            continue

        request, request_path, scenario_path, environment_path, app_request_path, context_elapsed = prepare_request_for_scenario(ctx, scenario, sid, request_dir)
        result, timings, error, tb = run_one_method(ctx, scenario, method, request, request_path, environment_path, method_dir)
        cand = selected_candidate(dyn, result)
        paths = write_compact_artifacts(dyn, result, cand, method_dir, save_full_artifacts=args.save_full_artifacts)
        fields = candidate_summary_fields(dyn, cand)
        diag = diagnostic_fields(result)
        decision = decision_text(dyn, result)

        row: Dict[str, Any] = {
            "run_index": run_counter,
            "is_warmup": is_warmup,
            "replicate": replicate,
            "replicate_label": rep_label,
            "scenario_id": sid,
            "task": task,
            "dynamic_trace_id": trace_id,
            "dynamic_phase_index": scenario.get("dynamic_phase_index"),
            "dynamic_phase_label": scenario.get("dynamic_phase_label"),
            "context": dyn.display_values(params.get("context")),
            "space": dyn.display_values(params.get("space")),
            "sender": dyn.display_values(params.get("sender")),
            "subject": dyn.display_values(params.get("subject")),
            "recipient": dyn.display_values(params.get("recipient")),
            "purpose": dyn.display_values(params.get("purpose")),
            "transmission_principle": dyn.display_values(params.get("transmission_principle") or params.get("transmissionPrinciple")),
            "context_family": scenario.get("context_family"),
            "method_id": method_id,
            "method_label": method_label(method_id),
            "method_kind": method["method_kind"],
            "baseline": method["baseline"],
            "baseline_id": method["baseline"],
            "parent_method": method.get("parent_method"),
            "ablation_mode": method.get("ablation_mode"),
            "decision": decision,
            **fields,
            **diag,
            "context_materialization_elapsed_ms": context_elapsed,
            **timings,
            "generation_total_elapsed_ms": context_elapsed + float(timings.get("decision_elapsed_ms") or 0.0),
            "app_request_path": str(app_request_path),
            "context_request_path": str(request_path),
            "environment_path": str(environment_path),
            "method_output_dir": str(method_dir),
            "result_json": paths.get("result_json"),
            "selected_pipeline_json": paths.get("selected_pipeline_json"),
            "pipeline_spec_json": paths.get("pipeline_spec_json"),
            "error": error,
        }
        timing_rows.append(row)
        if (not is_warmup) and replicate == args.utility_replicate and decision == "select_pipeline":
            utility_rows.append(dict(row))

        index["contexts"].setdefault(sid, {"scenario_id": sid, "task": task, "trace_id": trace_id, "methods": {}})
        index["contexts"][sid]["methods"].setdefault(method_id, []).append(row)
        index["traces"].setdefault(trace_id, {"trace_id": trace_id, "phases": {}, "methods": {}})
        index["traces"][trace_id]["phases"].setdefault(sid, {"scenario_id": sid, "phase_index": scenario.get("dynamic_phase_index"), "methods": {}})["methods"].setdefault(method_id, []).append(row)
        index["traces"][trace_id]["methods"].setdefault(method_id, []).append(row)

        progress_write(
            f"[overhead] {sid}/{method_id}/{rep_label}: decision={decision} "
            f"gen={row['generation_total_elapsed_ms']:.1f}ms decision={row['decision_elapsed_ms']:.1f}ms"
        )

    write_json(index, pipeline_root / "index.json")
    write_json(timing_rows, out_root / "overhead_timing.json")
    write_csv(timing_rows, out_root / "overhead_timing.csv")

    # Utility root with selected outputs only from the configured replicate.
    utility_root = pipeline_root / "_utility_selected_only"
    utility_root.mkdir(parents=True, exist_ok=True)
    write_json(utility_rows, utility_root / "summary.json")
    write_csv(utility_rows, utility_root / "summary.csv")
    write_json({f"{r['scenario_id']}::{r['method_id']}": r for r in utility_rows}, utility_root / "summary_by_context.json")

    summary_by_trace: Dict[str, Any] = {}
    for trace_id, tr in index["traces"].items():
        summary_by_trace[trace_id] = {
            "trace_id": trace_id,
            "methods": {},
            "phase_count": len(tr.get("phases", {})),
        }
        for method_id, mrows in tr.get("methods", {}).items():
            measured = [r for r in mrows if not is_warmup_row(r)]
            vals = [float(r.get("generation_total_elapsed_ms") or 0.0) for r in measured]
            summary_by_trace[trace_id]["methods"][method_id] = {
                "method_id": method_id,
                "run_count": len(measured),
                "phase_count": len({r.get("scenario_id") for r in measured}),
                "select_count": sum(1 for r in measured if r.get("decision") == "select_pipeline"),
                "no_compromise_count": sum(1 for r in measured if r.get("decision") == "no_compromise"),
                "review_count": sum(1 for r in measured if r.get("decision") in {"consent_or_review_required", "review"}),
                "error_count": sum(1 for r in measured if r.get("decision") == "error" or r.get("error")),
                "hard_violation_select_count": sum(1 for r in measured if r.get("context_family") == "hard_violation" and r.get("decision") == "select_pipeline"),
                "generation_total_ms_mean": mean(vals),
                "generation_total_ms_median": p50(vals),
                "generation_total_ms_p95": percentile(vals, 0.95),
            }
    write_json(summary_by_trace, out_root / "overhead_summary_by_trace.json")

    return pipeline_root, timing_rows, utility_rows, summary_by_trace


def summarize_by_method(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if is_warmup_row(r):
            continue
        groups.setdefault(str(r.get("method_id")), []).append(r)

    numeric_cols = [
        "context_materialization_elapsed_ms",
        "decision_elapsed_ms",
        "generation_total_elapsed_ms",
        "stage_request_load_elapsed_ms",
        "stage_candidate_generation_elapsed_ms",
        "stage_ci_evaluation_elapsed_ms",
        "stage_preliminary_selection_elapsed_ms",
        "stage_privacy_probe_elapsed_ms",
        "stage_final_selection_elapsed_ms",
        "stage_total_full_mediator_elapsed_ms",
    ]
    out: List[Dict[str, Any]] = []
    for method_id, mrows in sorted(groups.items()):
        row: Dict[str, Any] = {
            "method_id": method_id,
            "method_label": method_label(method_id),
            "method_kind": mrows[0].get("method_kind"),
            "ablation_mode": mrows[0].get("ablation_mode"),
            "run_count": len(mrows),
            "phase_count": len({r.get("scenario_id") for r in mrows}),
            "select_count": sum(1 for r in mrows if r.get("decision") == "select_pipeline"),
            "no_compromise_count": sum(1 for r in mrows if r.get("decision") == "no_compromise"),
            "review_count": sum(1 for r in mrows if r.get("decision") in {"consent_or_review_required", "review"}),
            "error_count": sum(1 for r in mrows if r.get("decision") == "error" or r.get("error")),
            "hard_violation_select_count": sum(1 for r in mrows if r.get("context_family") == "hard_violation" and r.get("decision") == "select_pipeline"),
        }
        for col in numeric_cols:
            vals = [float(v) for v in (float_or_none(r.get(col)) for r in mrows) if v is not None]
            row[f"{col}_mean"] = mean(vals)
            row[f"{col}_median"] = p50(vals)
            row[f"{col}_p95"] = percentile(vals, 0.95)
            row[f"{col}_min"] = min(vals) if vals else None
            row[f"{col}_max"] = max(vals) if vals else None
        out.append(row)
    return out


def run_utility_timing(args: argparse.Namespace, utility_root: Path, utility_out: Path, utility_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if args.no_utility:
        return {"status": "skipped", "reason": "--no-utility"}
    if not utility_rows:
        return {"status": "skipped", "reason": "no selected-output rows"}

    existing_paths = existing_runtime_summary_paths(Path(args.out_dir), utility_out)
    if (not args.rerun_utility) and existing_paths["runtime_csv"].exists():
        runtime_rows = read_csv(existing_paths["runtime_csv"])
        progress_write(f"[resume] runtime timing already exists; skipping utility rerun: {existing_paths['runtime_csv']}")
        return {
            "status": "ok",
            "resumed_from_existing": True,
            "reason": "runtime_timing_summary.csv already exists; use --rerun-utility to recompute",
            "runtime_timing_summary_csv": str(existing_paths["runtime_csv"]),
            "runtime_timing_summary_json": str(existing_paths["runtime_json"]) if existing_paths["runtime_json"].exists() else None,
            "utility_summary_csv": str(existing_paths["utility_summary_csv"]) if existing_paths["utility_summary_csv"].exists() else None,
            "log_path": str(existing_paths["utility_log"]) if existing_paths["utility_log"].exists() else None,
            "runtime_row_count": len(runtime_rows),
            "selected_output_rows": len(utility_rows),
        }
    if (not args.rerun_utility) and existing_paths["utility_summary_csv"].exists():
        progress_write(f"[resume] utility_summary.csv already exists; deriving runtime_timing_summary without rerunning utility: {existing_paths['utility_summary_csv']}")
        runtime_rows = []
        for r in read_csv(existing_paths["utility_summary_csv"]):
            total = float_or_none(r.get("elapsed_ms"))
            downstream = float_or_none(r.get("downstream_elapsed_ms"))
            rr = dict(r)
            rr["approx_preprocessing_plus_wrapper_elapsed_ms"] = max(0.0, total - downstream) if total is not None and downstream is not None else None
            runtime_rows.append(rr)
        write_csv(runtime_rows, existing_paths["runtime_csv"])
        write_json(runtime_rows, existing_paths["runtime_json"])
        return {
            "status": "ok",
            "resumed_from_existing": True,
            "reason": "utility_summary.csv already exists; use --rerun-utility to recompute",
            "runtime_timing_summary_csv": str(existing_paths["runtime_csv"]),
            "runtime_timing_summary_json": str(existing_paths["runtime_json"]),
            "utility_summary_csv": str(existing_paths["utility_summary_csv"]),
            "runtime_row_count": len(runtime_rows),
            "selected_output_rows": len(utility_rows),
        }

    all_tasks = sorted({str(r.get("task")) for r in utility_rows if r.get("task")})
    all_methods = sorted({str(r.get("method_id")) for r in utility_rows if r.get("method_id")})
    tasks = parse_csv_list(getattr(args, "utility_tasks", None), default=all_tasks)
    methods = parse_csv_list(getattr(args, "utility_methods", None), default=all_methods)
    utility_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        args.evaluate_utility_module,
        "--pipeline-root", str(utility_root),
        "--out-dir", str(utility_out),
        "--project-root", str(Path(args.project_root).resolve()),
        "--runtime-package", args.runtime_package,
        "--tasks", ",".join(tasks) if tasks else "auto",
        "--methods", ",".join(methods),
        "--ablation-policy", "none",
        "--max-samples", str(args.utility_max_samples),
        "--yes",
        "--no-preflight-confirm",
        "--keep-intermediate-data",
    ]
    if args.rerun_utility:
        cmd.append("--rerun-existing")
    if args.no_task_pipeline_cache:
        cmd.append("--no-task-pipeline-cache")
    if args.utility_max_frames_per_sample is not None:
        cmd += ["--max-frames-per-sample", str(args.utility_max_frames_per_sample)]
    if args.device:
        cmd += ["--device", str(args.device)]
    if args.prefer_gpu_name:
        cmd += ["--prefer-gpu-name", str(args.prefer_gpu_name)]
    if args.utility_extra_args:
        cmd += shlex.split(args.utility_extra_args)

    log_path = utility_out / "overhead_utility_eval.log"
    progress_write(f"[utility] timing selected outputs only: {shlex.join(cmd)}")
    progress_write(f"[utility] streaming output to terminal and log: {log_path}")
    start = now_ns()

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    returncode = 0
    captured_tail: List[str] = []
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write("$ " + shlex.join(cmd) + "\n\n")
        log_f.flush()
        if getattr(args, "no_stream_utility_output", False):
            proc = subprocess.run(
                cmd,
                cwd=str(Path(args.project_root).resolve()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            out = proc.stdout or ""
            log_f.write(out)
            captured_tail = out.splitlines()[-200:]
            returncode = int(proc.returncode)
        else:
            proc = subprocess.Popen(
                cmd,
                cwd=str(Path(args.project_root).resolve()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                print(line, end="", file=sys.stderr, flush=True)
                captured_tail.append(line.rstrip("\n"))
                if len(captured_tail) > 200:
                    captured_tail = captured_tail[-200:]
            returncode = int(proc.wait())
    elapsed = elapsed_ms(start)

    result: Dict[str, Any] = {
        "status": "ok" if returncode == 0 else "error",
        "returncode": returncode,
        "cmd": cmd,
        "elapsed_ms": elapsed,
        "log_path": str(log_path),
        "utility_out_dir": str(utility_out),
        "selected_output_rows": len(utility_rows),
        "tasks": tasks,
        "methods": methods,
        "all_tasks": all_tasks,
        "all_methods": all_methods,
        "streamed_output": not bool(getattr(args, "no_stream_utility_output", False)),
    }
    summary_path = utility_out / "utility_summary.csv"
    runtime_rows: List[Dict[str, Any]] = []
    if summary_path.exists():
        for r in read_csv(summary_path):
            total = float_or_none(r.get("elapsed_ms"))
            downstream = float_or_none(r.get("downstream_elapsed_ms"))
            approx = None
            if total is not None and downstream is not None:
                approx = max(0.0, total - downstream)
            rr = dict(r)
            rr["approx_preprocessing_plus_wrapper_elapsed_ms"] = approx
            runtime_rows.append(rr)
        write_csv(runtime_rows, Path(args.out_dir) / "runtime_timing_summary.csv")
        write_json(runtime_rows, Path(args.out_dir) / "runtime_timing_summary.json")
        result["runtime_timing_summary_csv"] = str(Path(args.out_dir) / "runtime_timing_summary.csv")
        result["runtime_row_count"] = len(runtime_rows)
    else:
        result["runtime_row_count"] = 0
        result["warning"] = f"utility_summary.csv not found at {summary_path}"
    if returncode != 0:
        result["stdout_tail"] = "\n".join(captured_tail[-120:])
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Measure dynamic context-switch and pipeline-generation overhead.")
    # Dynamic context integration.
    p.add_argument("--dynamic-context-module", default="evaluation.dynamic_context_experiments", help="Import path for the dynamic context experiment helper module.")
    p.add_argument("--dynamic-context-module-path", default=None, help="Optional file path to dynamic_context_experiments.py; overrides --dynamic-context-module.")
    p.add_argument("--dynamic-contexts", default=None, help="Optional custom dynamic context JSON. If omitted, use built-in traces from dynamic_context_experiments.py.")
    p.add_argument("--trace-set", default="default", help="default, all, or aliases supported by dynamic_context_experiments.py.")
    p.add_argument("--scenario-ids", default="", help="Optional comma list of dynamic phase IDs.")

    # Core paths.
    p.add_argument("--operators", default="norms/operator_contracts.json")
    p.add_argument("--contexts", default="survey/data/ci_focused_user_study_context.json")
    p.add_argument("--app-request-dir", default="app_requests/templates")
    p.add_argument("--candidate-generator", default="mediator/generate_pipeline_candidates.py")
    p.add_argument("--constraints", default="norms/ci_constraints.json")
    p.add_argument("--evaluator", default="mediator/contextual_integrity_evaluator.py")
    p.add_argument("--selector", default="mediator/pipeline_selection.py")
    p.add_argument("--full-mediator-module", default="mediator/full_mediator.py")
    p.add_argument("--raw-module", default=None)
    p.add_argument("--manual-module", default=None)
    p.add_argument("--direct-llm-module", default=None)
    p.add_argument("--out-dir", default="runs/dynamic_context_overhead")
    p.add_argument("--project-root", default=".")

    # Methods / ablations.
    p.add_argument("--methods", default=",".join(DEFAULT_BASE_METHODS), help="Comma list of baselines: raw,manual,direct_llm,full_mediator.")
    p.add_argument("--ablation-modes", default=",".join(DEFAULT_ABLATION_MODES), help="Comma list of full-mediator ablations to time. Use '' to disable.")
    p.add_argument("--repeats", type=int, default=3, help="Measured repeats per phase/method.")
    p.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs excluded from aggregate summaries.")
    p.add_argument("--utility-replicate", type=int, default=1, help="Which measured repeat to expose to optional utility timing.")
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--max-states", type=int, default=25000)
    p.add_argument("--save-full-artifacts", action="store_true", help="Save full candidate/CI/selection JSON for every timing run. Default saves compact artifacts only.")
    p.add_argument("--rerun-existing-overhead", action="store_true", help="Recompute overhead timing rows even if overhead_timing.csv already contains completed rows. Default resumes/skips completed rows.")

    # Full mediator options.
    p.add_argument("--full-mediator-use-llm", action="store_true")
    p.add_argument("--llm-model", default="gpt-4o")
    p.add_argument("--llm-temperature", type=float, default=0.0)
    p.add_argument("--llm-confidence-threshold", type=float, default=0.75)
    p.add_argument("--top-k-for-llm", type=int, default=None)
    p.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--probe-artifacts", default=None)
    p.add_argument("--probe-config", default=None)
    p.add_argument("--probe-package-dir", default=None)
    p.add_argument("--selection-config", default=None)
    p.add_argument("--sensor-stream", default=None)

    # Explicit app request overrides.
    p.add_argument("--visitor-request", default=None)
    p.add_argument("--fall-request", default=None)
    p.add_argument("--adl-request", default=None)
    p.add_argument("--audio-request", default=None)

    # Optional tiny utility pass.
    p.add_argument("--no-utility", action="store_true", help="Skip the optional utility/preprocessing timing pass.")
    p.add_argument("--evaluate-utility-module", default="evaluation.evaluate_utility")
    p.add_argument("--runtime-package", default="mediator.smartpriv_runtime")
    p.add_argument("--utility-max-samples", type=int, default=1)
    p.add_argument("--utility-max-frames-per-sample", type=int, default=24)
    p.add_argument("--utility-extra-args", default="")
    p.add_argument("--utility-tasks", default="", help="Optional comma-separated subset of tasks for optional utility timing.")
    p.add_argument("--utility-methods", default="", help="Optional comma-separated subset of methods/ablations for optional utility timing.")
    p.add_argument("--no-stream-utility-output", action="store_true", help="Capture optional utility timing output silently instead of streaming it live.")
    p.add_argument("--rerun-utility", action="store_true", help="Force optional utility timing to rerun even if utility/runtime timing outputs already exist. Default resumes/skips existing utility results.")
    p.add_argument("--no-task-pipeline-cache", action="store_true", help="Pass --no-task-pipeline-cache to evaluate_utility. Leave off for resume-friendly runs.")
    p.add_argument("--device", default="auto")
    p.add_argument("--prefer-gpu-name", default="RTX 2070")

    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--yes", "-y", action="store_true", help="Noninteractive symmetry flag.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    set_progress_enabled(not args.no_progress)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.utility_replicate < 1 or args.utility_replicate > args.repeats:
        raise ValueError("--utility-replicate must be between 1 and --repeats")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    project_root = Path(args.project_root).resolve()
    for c in [project_root, project_root / "mediator"]:
        if c.exists() and str(c) not in sys.path:
            sys.path.insert(0, str(c))

    dyn = load_dynamic_module(args)
    if args.dynamic_contexts:
        scenarios, trace_meta = dyn.load_custom_dynamic_contexts(args.dynamic_contexts)
    else:
        scenarios, trace_meta = dyn.built_in_dynamic_scenarios(args.trace_set)
    progress_write(f"[overhead] loaded trace-set={args.trace_set}: {len(scenarios)} phases across {len(trace_meta)} traces")

    run_start = now_ns()
    pipeline_root, timing_rows, utility_rows, summary_by_trace = run_generation_timing(args, dyn, scenarios, trace_meta)
    measured_rows = [r for r in timing_rows if not is_warmup_row(r)]
    method_summary = summarize_by_method(timing_rows)
    write_csv(method_summary, out_root / "overhead_summary_by_method.csv")
    write_json(method_summary, out_root / "overhead_summary_by_method.json")

    utility_root = pipeline_root / "_utility_selected_only"
    utility_result = run_utility_timing(args, utility_root, out_root / "utility_eval", utility_rows)

    run_summary = {
        "schema_version": "smartpriv_dynamic_context_overhead_run_v1",
        "out_dir": str(out_root),
        "pipeline_root": str(pipeline_root),
        "trace_set": args.trace_set,
        "trace_count": len(trace_meta),
        "phase_count": len({r.get("scenario_id") for r in measured_rows}),
        "methods": sorted({r.get("method_id") for r in timing_rows if r.get("method_id")}),
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "row_count_total": len(timing_rows),
        "row_count_measured": len(measured_rows),
        "overhead_timing_csv": str(out_root / "overhead_timing.csv"),
        "overhead_summary_by_method_csv": str(out_root / "overhead_summary_by_method.csv"),
        "overhead_summary_by_trace_json": str(out_root / "overhead_summary_by_trace.json"),
        "utility_selected_only_pipeline_root": str(utility_root),
        "utility_selected_output_rows": len(utility_rows),
        "utility_eval_result": utility_result,
        "run_elapsed_ms": elapsed_ms(run_start),
        "method_decision_counts": {
            m: {
                "select": sum(1 for r in measured_rows if r.get("method_id") == m and r.get("decision") == "select_pipeline"),
                "no_compromise": sum(1 for r in measured_rows if r.get("method_id") == m and r.get("decision") == "no_compromise"),
                "review": sum(1 for r in measured_rows if r.get("method_id") == m and r.get("decision") in {"consent_or_review_required", "review"}),
                "error": sum(1 for r in measured_rows if r.get("method_id") == m and (r.get("decision") == "error" or r.get("error"))),
                "hard_violation_select": sum(1 for r in measured_rows if r.get("method_id") == m and r.get("context_family") == "hard_violation" and r.get("decision") == "select_pipeline"),
            }
            for m in sorted({r.get("method_id") for r in measured_rows if r.get("method_id")})
        },
    }
    write_json(run_summary, out_root / "overhead_run_summary.json")

    print(json.dumps({
        "status": "ok" if utility_result.get("status") in {"ok", "skipped"} else "error",
        "out_dir": str(out_root),
        "overhead_timing_csv": str(out_root / "overhead_timing.csv"),
        "overhead_summary_by_method_csv": str(out_root / "overhead_summary_by_method.csv"),
        "runtime_timing_summary_csv": utility_result.get("runtime_timing_summary_csv"),
        "overhead_run_summary": str(out_root / "overhead_run_summary.json"),
        "utility_eval_status": utility_result.get("status"),
    }, indent=2))
    return 0 if utility_result.get("status") in {"ok", "skipped"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
