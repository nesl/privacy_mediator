#!/usr/bin/env python3
"""Generate preprocessing-pipeline outputs for all context scenarios.

This module is intended to be run from the project root:

    python -m evaluation.generate_pipelines_for_all_contexts \
      --operators norms/operator_contracts.json \
      --contexts survey/data/ci_focused_user_study_context.json \
      --app-request-dir app_requests/templates \
      --candidate-generator mediator/generate_pipeline_candidates.py \
      --constraints norms/ci_constraints.json \
      --evaluator mediator/contextual_integrity_evaluator.py \
      --selector mediator/pipeline_selection.py \
      --out-dir runs/context_pipeline_generation

For each context scenario, the script:
  1. loads the task-specific downstream app request;
  2. overlays the context's CI fields into that request;
  3. runs selected baselines/full mediator;
  4. writes per-context/per-baseline results and pipeline specs;
  5. writes simple summary CSV/JSON files and an index for retrieval.

The generated context-specific request is deliberately separate from the original
context-only survey item.  The context supplies CI fields; the app request supplies
utility and downstream output-format compatibility constraints.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib
import inspect
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # tqdm is optional; fall back to a simple stderr counter.
    _tqdm = None


class _SimpleProgress:
    def __init__(self, total: int, enabled: bool = True):
        self.total = int(total)
        self.enabled = enabled
        self.n = 0

    def __enter__(self):
        if self.enabled:
            print(f"Generating pipelines: 0/{self.total}", end="", file=sys.stderr, flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            print(file=sys.stderr, flush=True)

    def update(self, n: int = 1) -> None:
        self.n += n
        if self.enabled:
            print(f"\rGenerating pipelines: {self.n}/{self.total}", end="", file=sys.stderr, flush=True)

    def set_postfix_str(self, value: str) -> None:
        # Kept for API compatibility with tqdm; the fallback counter stays compact.
        return None


def make_progress(total: int, enabled: bool = True):
    if not enabled:
        return _SimpleProgress(total, enabled=False)
    if _tqdm is not None:
        return _tqdm(total=total, desc="Generating pipelines", unit="run", dynamic_ncols=True)
    return _SimpleProgress(total, enabled=True)


TASK_TO_REQUEST_FILENAME: Dict[str, str] = {
    "visitor_presence_detection": "request_app_visitor_chokepoint_downstream_compatible.json",
    "fall_detection": "request_app_fall_le2i_pose_downstream_compatible.json",
    "adl_recognition": "request_app_adl_youhome_av_downstream_compatible.json",
    "domestic_sound_monitoring": "request_app_domestic_audio_chimehome_downstream_compatible.json",
}

DEFAULT_BASELINES = ["raw", "manual", "direct_llm", "full_mediator"]

# Default ablations are chosen to be runnable without requiring an LLM API key.
# Add llm_only explicitly with --ablations if you want to test unconstrained LLM decisions.
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



def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def write_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def import_module_from_path(module_name: str, path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not import {module_name}: file does not exist: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_module_or_path(module_name: str, optional_path: Optional[str | Path] = None):
    if optional_path:
        p = Path(optional_path)
        if p.exists():
            return import_module_from_path(module_name.replace(".", "_"), p)
    return importlib.import_module(module_name)


def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def first_scalar(x: Any) -> Optional[str]:
    vals = as_list(x)
    if not vals:
        return None
    if vals[0] is None:
        return None
    return str(vals[0])


def normalize_term(x: Any) -> str:
    return str(x or "").strip()


def iter_context_scenarios(context_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "context_scenarios" in context_doc:
        return list(context_doc["context_scenarios"])
    if "generated_information_flows" in context_doc:
        return list(context_doc["generated_information_flows"])
    if isinstance(context_doc, list):
        return list(context_doc)
    raise ValueError("Could not find context_scenarios or generated_information_flows in contexts file.")


def scenario_ci_params(scenario: Dict[str, Any]) -> Dict[str, Any]:
    params = (
        scenario.get("ci_parameters_scalar_context_only")
        or scenario.get("ci_parameters_scalar")
        or scenario.get("ci_parameters")
        or {}
    )
    if not params and "machine_flow_for_ci_constraints_context_only" in scenario:
        mf = scenario["machine_flow_for_ci_constraints_context_only"]
        params = {
            "context": first_scalar(mf.get("context")),
            "space": first_scalar(mf.get("space")),
            "sender": first_scalar(mf.get("sender")),
            "subject": first_scalar(mf.get("subject")),
            "recipient": first_scalar(mf.get("recipient")),
            "purpose": first_scalar(mf.get("purpose")),
            "transmission_principle": first_scalar(mf.get("transmissionPrinciple")),
        }
    return dict(params)


def scenario_id(scenario: Dict[str, Any], ordinal: int) -> str:
    return str(
        scenario.get("scenario_id")
        or scenario.get("flow_id")
        or scenario_ci_params(scenario).get("scenario_id")
        or f"S{ordinal:03d}"
    )


def scenario_task(scenario: Dict[str, Any]) -> str:
    params = scenario_ci_params(scenario)
    return normalize_term(scenario.get("task") or params.get("task"))


def scenario_space(scenario: Dict[str, Any]) -> str:
    return normalize_term(scenario_ci_params(scenario).get("space"))


def resolve_request_path(task: str, app_request_dir: str | Path, explicit: Dict[str, Optional[str]]) -> Path:
    if explicit.get(task):
        p = Path(str(explicit[task]))
        if p.exists():
            return p
        raise FileNotFoundError(f"Explicit request path for task {task} does not exist: {p}")

    fname = TASK_TO_REQUEST_FILENAME.get(task)
    if not fname:
        raise KeyError(f"No app-request filename mapping for task {task!r}")

    candidates = [
        Path(app_request_dir) / fname,
        Path("app_requests/templates") / fname,
        Path("app_requests_downstream_compatible") / fname,
        Path("/mnt/data/app_requests_downstream_compatible") / fname,
        Path("/mnt/data") / fname,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find app request for task {task!r}. Tried: " + ", ".join(str(p) for p in candidates)
    )


def overlay_context_on_app_request(app_request: Dict[str, Any], scenario: Dict[str, Any], sid: str) -> Dict[str, Any]:
    """Return a context-specific request.

    The app request keeps the downstream utility/output-format contract.  The
    context scenario supplies the CI tuple fields for this evaluation case.
    """
    params = scenario_ci_params(scenario)
    req = copy.deepcopy(app_request)

    identity = req.setdefault("request_identity", {})
    base_request_id = identity.get("request_id") or "app_request"
    base_scenario_id = identity.get("scenario_id") or "app_scenario"
    identity["source_app_request_id"] = base_request_id
    identity["source_app_scenario_id"] = base_scenario_id
    identity["request_id"] = f"{base_request_id}__{sid}"
    identity["scenario_id"] = sid
    identity["context_scenario_id"] = sid
    identity["context_family"] = scenario.get("context_family")
    identity["context_bundle_id"] = scenario.get("context_bundle_id")
    identity["generated_by"] = "evaluation.generate_pipelines_for_all_contexts"

    ci = req.setdefault("ci_context", {})
    scalar_to_ci_key = {
        "context": "context",
        "space": "space",
        "sender": "sender",
        "subject": "subject",
        "recipient": "recipient",
        "purpose": "purpose",
    }
    for src, dst in scalar_to_ci_key.items():
        value = params.get(src)
        ci[dst] = [value] if value else []

    tp = params.get("transmission_principle") or params.get("transmissionPrinciple")
    ci["transmissionPrinciple_assumed"] = [tp] if tp else []

    tags = list(ci.get("social_context_tags", []) or [])
    for tag in ["context_study_case", scenario.get("context_family"), scenario.get("study_group")]:
        if tag and tag not in tags:
            tags.append(str(tag))
    ci["social_context_tags"] = tags

    req["evaluation_context_scenario"] = {
        "scenario_id": sid,
        "task": scenario_task(scenario),
        "task_label": scenario.get("task_label"),
        "context_family": scenario.get("context_family"),
        "context_bundle_id": scenario.get("context_bundle_id"),
        "context_bundle_label": scenario.get("context_bundle_label"),
        "ci_parameters_scalar_context_only": params,
        "participant_vignette": scenario.get("participant_vignette"),
    }

    return req


def decision_text(result: Dict[str, Any]) -> str:
    d = result.get("decision")
    if isinstance(d, dict):
        return str(d.get("decision") or d.get("status") or "")
    if isinstance(d, str):
        return d
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


def all_candidates(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(result.get("candidates"), list):
        return result["candidates"]
    if isinstance(result.get("candidate_generation_result"), dict):
        return result["candidate_generation_result"].get("candidates", []) or []
    return []


def selected_candidate(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Only an actual selection should be materialized as a survey output.
    # no_compromise / consent_or_review / no_candidates may include closest rejected
    # candidates for diagnostics, but those are not shared outputs.
    if decision_text(result) != "select_pipeline":
        return None
    sel = result.get("selected_candidate") or result.get("selected")
    if isinstance(sel, dict):
        return sel
    pid = selected_pipeline_id(result)
    if pid:
        for cand in all_candidates(result):
            if str(cand.get("pipeline_id")) == pid:
                return cand
    # Baseline wrappers usually have candidates but no nested selected candidate.
    cands = all_candidates(result)
    if cands and pid and str(cands[0].get("pipeline_id")) == pid:
        return cands[0]
    return None


def cap_type(cap: Optional[Dict[str, Any]]) -> str:
    if not cap:
        return ""
    return str(cap.get("semantic_type") or cap.get("media_type") or "")


def cap_schema(cap: Optional[Dict[str, Any]]) -> str:
    if not cap:
        return ""
    return str(cap.get("schema") or "")


def operator_ids(cand: Optional[Dict[str, Any]]) -> List[str]:
    if not cand:
        return []
    out = []
    for op in cand.get("operators", []) or []:
        oid = op.get("operator") or op.get("operator_id")
        if oid:
            out.append(str(oid))
    return out


def candidate_summary_fields(cand: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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
    return {
        "selected_pipeline_id": cand.get("pipeline_id"),
        "matched_output_cap": cand.get("matched_output_cap"),
        "matched_output_schema": cand.get("matched_output_schema"),
        "final_output_type": cap_type(final_cap),
        "final_output_schema": cap_schema(final_cap),
        "operators": " -> ".join(operator_ids(cand)),
        "residual_score": cand.get("residual_score"),
        "quality_status": cand.get("quality_status"),
        "executable_under_catalog": cand.get("executable_under_catalog"),
    }



def result_diagnostic_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten generator/CI/selector diagnostics into the summary row.

    This makes cases like "full_mediator selected null" diagnosable from
    summary.csv without opening nested result.json files.
    """
    cand_gen = result.get("candidate_generation_result") or {}
    planner = cand_gen.get("planner") or {}
    gen_decision = cand_gen.get("decision") or {}
    ci = result.get("ci_evaluation_result") or {}
    selector = (result.get("pipeline_selection_result") or {}).get("selector") or {}
    return {
        "candidate_generation_decision": gen_decision.get("decision") if isinstance(gen_decision, dict) else gen_decision,
        "candidate_generation_reason": gen_decision.get("reason") if isinstance(gen_decision, dict) else None,
        "candidate_count": planner.get("candidate_count"),
        "rejected_goal_count": planner.get("rejected_goal_count"),
        "states_expanded": planner.get("states_expanded"),
        "states_seen": planner.get("states_seen"),
        "ci_evaluation_count": len(ci.get("evaluations", []) or []),
        "selector_num_candidates": selector.get("num_candidates"),
        "selector_num_feasible": selector.get("num_feasible"),
        "selector_num_rejected": selector.get("num_rejected"),
        "no_compromise_reason": ((result.get("no_compromise_diagnostics") or {}).get("reason") or ((result.get("decision") or {}) if isinstance(result.get("decision"), dict) else {}).get("reason")),
        "no_compromise_candidate_count": (result.get("no_compromise_diagnostics") or {}).get("candidate_count"),
        "no_compromise_hard_rejection_count": (result.get("no_compromise_diagnostics") or {}).get("hard_rejection_count"),
        "no_compromise_ci_feasible_count": (result.get("no_compromise_diagnostics") or {}).get("ci_feasible_count"),
        "no_compromise_top_failed_rules": json.dumps((result.get("no_compromise_diagnostics") or {}).get("top_failed_rules") or []),
        "no_compromise_diagnostics_json": json.dumps(result.get("no_compromise_diagnostics") or {}),
    }

def make_stage_specs_from_candidate(cand: Dict[str, Any]) -> List[Dict[str, Any]]:
    if cand.get("executable_pipeline_spec") and cand["executable_pipeline_spec"].get("stages"):
        return list(cand["executable_pipeline_spec"]["stages"])
    stages: List[Dict[str, Any]] = []
    for op in cand.get("operators", []) or []:
        oid = op.get("operator") or op.get("operator_id")
        if not oid or oid in {"op.source", "op.route_publish"}:
            continue
        stages.append({"operator_id": oid, "parameters": op.get("parameters") or {}})
    return stages


def write_pipeline_code_and_metadata(cand: Optional[Dict[str, Any]], result: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    """Write selected candidate metadata plus a simple executable wrapper when possible."""
    paths: Dict[str, str] = {}
    write_json(result, out_dir / "result.json")
    paths["result_json"] = str(out_dir / "result.json")

    if not cand:
        write_text("No selected candidate was available for this baseline/context.\n", out_dir / "NO_SELECTED_PIPELINE.txt")
        paths["note"] = str(out_dir / "NO_SELECTED_PIPELINE.txt")
        return paths

    write_json(cand, out_dir / "selected_pipeline.json")
    paths["selected_pipeline_json"] = str(out_dir / "selected_pipeline.json")

    stages = make_stage_specs_from_candidate(cand)
    spec = {
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
    write_json(spec, out_dir / "pipeline_spec.json")
    paths["pipeline_spec_json"] = str(out_dir / "pipeline_spec.json")

    runnable = f'''#!/usr/bin/env python3
"""Minimal runner for saved preprocessing pipeline {cand.get("pipeline_id")}.

This wrapper assumes your project exposes smartpriv_runtime.pipeline.ExecutablePipeline
and smartpriv_runtime.media_io.item_from_media.  It is generated from symbolic
operator metadata; if executable_under_catalog is false or the operator ids are
manual placeholders, the script is documentation rather than a runnable pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from smartpriv_runtime.media_io import item_from_media
from smartpriv_runtime.pipeline import ExecutablePipeline


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input media path")
    p.add_argument("--media-type", default=None, help="Optional media type override, e.g. image/x-raw")
    p.add_argument("--spec", default=str(Path(__file__).with_name("pipeline_spec.json")))
    p.add_argument("--out", default=None, help="Optional output JSON path")
    args = p.parse_args()

    pipe = ExecutablePipeline.from_spec_file(args.spec)
    item = item_from_media(args.input, media_type=args.media_type)
    out = pipe.process(item)
    obj = None if out is None else out.to_jsonable(include_payload=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
    else:
        print(json.dumps(obj, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    write_text(runnable, out_dir / "run_pipeline.py")
    paths["run_pipeline_py"] = str(out_dir / "run_pipeline.py")
    return paths


def run_raw_baseline(raw_module, operator_catalog: Dict[str, Any], request: Dict[str, Any], candidate_generator: Optional[str]) -> Dict[str, Any]:
    return raw_module.run_raw_baseline(
        operator_catalog=operator_catalog,
        request=request,
        candidate_generator_path=candidate_generator,
    )


def run_manual_baseline(manual_module, operator_catalog: Dict[str, Any], request: Dict[str, Any], candidate_generator: Optional[str], task: str, space: str, max_depth: int, max_states: int) -> Dict[str, Any]:
    if hasattr(manual_module, "run_baseline"):
        return manual_module.run_baseline(
            operator_catalog=operator_catalog,
            request=request,
            candidate_generator_path=candidate_generator,
            task=task,
            space=space,
            max_depth=max_depth,
            max_states=max_states,
        )
    if hasattr(manual_module, "run_manual_baseline"):
        # Fallback for the older manual baseline.  It is not task/space-specific.
        return manual_module.run_manual_baseline(
            operator_catalog=operator_catalog,
            request=request,
            candidate_generator_path=candidate_generator,
            max_depth=max_depth,
            max_states=max_states,
        )
    raise AttributeError("Manual baseline module exposes neither run_baseline nor run_manual_baseline.")


def run_direct_llm_baseline(direct_module, operator_catalog: Dict[str, Any], request: Dict[str, Any], environment: Dict[str, Any], candidate_generator: Optional[str], max_depth: int, max_states: int, llm_model: str, llm_temperature: float, openai_api_key: Optional[str]) -> Dict[str, Any]:
    return direct_module.run_direct_llm_baseline(
        operator_catalog=operator_catalog,
        request=request,
        environment=environment,
        candidate_generator_path=candidate_generator,
        max_depth=max_depth,
        max_states=max_states,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        openai_api_key=openai_api_key,
    )


def run_full_mediator(
    full_module,
    args: argparse.Namespace,
    request_path: Path,
    out_dir: Path,
    ablation_mode: Optional[str] = None,
) -> Dict[str, Any]:
    if not args.constraints:
        raise ValueError("--constraints is required when running full_mediator")

    kwargs: Dict[str, Any] = {
        "operators_path": args.operators,
        "request_path": request_path,
        "constraints_path": args.constraints,
        "environment_path": None,
        "sensor_stream_path": None,
        "candidate_generator_path": args.candidate_generator,
        "evaluator_path": args.evaluator,
        "selector_path": args.selector,
        "max_depth": args.max_depth,
        "max_states": args.max_states,
        "use_llm": args.full_mediator_use_llm,
        "llm_model": args.llm_model,
        "llm_temperature": args.llm_temperature,
        "llm_confidence_threshold": args.llm_confidence_threshold,
        "top_k_for_llm": args.top_k_for_llm,
        "probe_artifacts_path": args.probe_artifacts,
        "probe_config_path": args.probe_config,
        "probe_package_dir": args.probe_package_dir,
        "selection_config_path": args.selection_config,
    }

    sig = inspect.signature(full_module.run_mediator)
    if ablation_mode:
        if "ablation_modes" in sig.parameters:
            kwargs["ablation_modes"] = [ablation_mode]
        elif "ablation_mode" in sig.parameters:
            kwargs["ablation_mode"] = ablation_mode
        else:
            raise TypeError(
                "Requested ablation mode "
                f"{ablation_mode!r}, but the selected full_mediator module does not "
                "expose an ablation_modes/ablation_mode parameter. Use the "
                "ablation-supported full_mediator.py generated with this patch."
            )
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return full_module.run_mediator(**filtered)


def method_summary_entry(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method_id": r.get("method_id") or r.get("baseline"),
        "method_kind": r.get("method_kind"),
        "baseline": r.get("baseline"),
        "baseline_id": r.get("baseline_id"),
        "ablation_mode": r.get("ablation_mode"),
        "parent_method": r.get("parent_method"),
        "decision": r.get("decision"),
        "selected_pipeline_id": r.get("selected_pipeline_id"),
        "matched_output_cap": r.get("matched_output_cap"),
        "matched_output_schema": r.get("matched_output_schema"),
        "final_output_type": r.get("final_output_type"),
        "final_output_schema": r.get("final_output_schema"),
        "operators": r.get("operators"),
        "output_dir": r.get("method_output_dir") or r.get("baseline_output_dir"),
        "result_json": r.get("result_json"),
        "selected_pipeline_json": r.get("selected_pipeline_json"),
        "pipeline_spec_json": r.get("pipeline_spec_json"),
        "error": r.get("error"),
        "no_compromise_reason": r.get("no_compromise_reason"),
        "no_compromise_top_failed_rules": r.get("no_compromise_top_failed_rules"),
    }


def write_context_summary(context_dir: Path, scenario: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    methods = {str(r.get("method_id") or r.get("baseline")): method_summary_entry(r) for r in rows}
    baselines = {str(r.get("baseline_id") or r.get("baseline")): method_summary_entry(r) for r in rows if r.get("method_kind") == "baseline"}
    ablations = {str(r.get("ablation_mode")): method_summary_entry(r) for r in rows if r.get("method_kind") == "ablation"}
    simple = {
        "scenario_id": rows[0]["scenario_id"] if rows else scenario.get("scenario_id"),
        "task": rows[0]["task"] if rows else scenario_task(scenario),
        "context_family": scenario.get("context_family"),
        "ci_parameters_scalar_context_only": scenario_ci_params(scenario),
        "methods": methods,
        "baselines": baselines,
        "ablations": ablations,
    }
    write_json(simple, context_dir / "context_summary.json")


def write_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        write_text("", path)
        return
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_list_arg(value: Optional[str], default: Sequence[str]) -> List[str]:
    if value is None or str(value).strip() == "":
        return list(default)
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_ablation_modes(value: Optional[str]) -> List[str]:
    if value is None:
        return list(DEFAULT_ABLATION_MODES)
    raw = str(value).strip()
    if raw == "" or raw.lower() in {"none", "false", "off", "0"}:
        return []
    if raw.lower() == "all":
        return list(DEFAULT_ABLATION_MODES) + ["llm_only"]
    return [x.strip() for x in raw.split(",") if x.strip()]


def make_run_specs(requested_baselines: Sequence[str], requested_ablations: Sequence[str]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for baseline in requested_baselines:
        specs.append({
            "method_id": baseline,
            "method_kind": "baseline",
            "baseline": baseline,
            "baseline_id": baseline,
            "ablation_mode": None,
            "parent_method": None,
            "method_label": baseline,
            "dir_parts": ("baselines", baseline),
        })
    for mode in requested_ablations:
        method_id = f"ablation:{mode}"
        specs.append({
            "method_id": method_id,
            "method_kind": "ablation",
            "baseline": method_id,  # Backward-compatible column used by older survey loaders.
            "baseline_id": None,
            "ablation_mode": mode,
            "parent_method": "full_mediator",
            "method_label": f"full_mediator ablation: {mode}",
            "dir_parts": ("ablations", mode),
        })
    return specs


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run all preprocessing baselines/full mediator over all context scenarios.")
    p.add_argument("--contexts", default="survey/data/ci_focused_user_study_context.json")
    p.add_argument("--app-request-dir", default="app_requests/templates")
    p.add_argument("--operators", required=True)
    p.add_argument("--candidate-generator", default="mediator/generate_pipeline_candidates.py")
    p.add_argument("--constraints", default="norms/ci_constraints.json")
    p.add_argument("--evaluator", default="mediator/contextual_integrity_evaluator.py")
    p.add_argument("--selector", default="mediator/pipeline_selection.py")
    p.add_argument("--selection-config", default=None)

    p.add_argument("--raw-module", default=None, help="Optional path to raw_baseline.py; defaults to preprocessing_baselines.raw_baseline")
    p.add_argument("--manual-module", default=None, help="Optional path to manual_baseline.py; defaults to preprocessing_baselines.manual_baseline")
    p.add_argument("--direct-llm-module", default=None, help="Optional path to direct_llm_baseline.py")
    p.add_argument("--full-mediator-module", default="mediator/full_mediator.py", help="Path to full_mediator.py")

    p.add_argument("--visitor-request", default=None)
    p.add_argument("--fall-request", default=None)
    p.add_argument("--adl-request", default=None)
    p.add_argument("--audio-request", default=None)

    p.add_argument("--baselines", default=",".join(DEFAULT_BASELINES), help="Comma list: raw,manual,direct_llm,full_mediator")
    p.add_argument(
        "--ablations",
        default=",".join(DEFAULT_ABLATION_MODES),
        help=(
            "Comma list of full-mediator ablation modes to run for every context. "
            "Use an empty string or 'none' to disable; use 'all' to include llm_only."
        ),
    )
    p.add_argument("--scenario-ids", default=None, help="Optional comma-separated subset of scenario ids")
    p.add_argument("--out-dir", default="runs/context_pipeline_generation")
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--max-states", type=int, default=25000)

    p.add_argument("--llm-model", default="gpt-4o-mini")
    p.add_argument("--llm-temperature", type=float, default=0.0)
    p.add_argument("--openai-api-key", default=None)
    p.add_argument("--full-mediator-use-llm", action="store_true")
    p.add_argument("--llm-confidence-threshold", type=float, default=0.75)
    p.add_argument("--top-k-for-llm", type=int, default=None)

    p.add_argument("--probe-artifacts", default=None)
    p.add_argument("--probe-config", default=None)
    p.add_argument("--probe-package-dir", default=None)

    p.add_argument("--continue-on-error", action="store_true", default=True)
    p.add_argument("--fail-fast", action="store_true", help="Stop at first baseline/context error")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm/simple progress display")
    args = p.parse_args(argv)

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    requested_baselines = parse_list_arg(args.baselines, DEFAULT_BASELINES)
    # Backwards-compatible alias from the older runner/package.
    requested_baselines = ["manual" if b == "manual_space_task" else b for b in requested_baselines]
    requested_ablations = parse_ablation_modes(args.ablations)
    run_specs = make_run_specs(requested_baselines, requested_ablations)
    wanted_sids = set(parse_list_arg(args.scenario_ids, [])) if args.scenario_ids else None

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    context_doc = load_json(args.contexts)
    scenarios_all = iter_context_scenarios(context_doc)
    scenarios: List[Tuple[int, str, Dict[str, Any]]] = []
    for idx, sc in enumerate(scenarios_all, start=1):
        sid = scenario_id(sc, idx)
        if wanted_sids and sid not in wanted_sids:
            continue
        scenarios.append((idx, sid, sc))

    operator_catalog = load_json(args.operators)

    # Import baseline modules only if requested. Full mediator is also required for ablations.
    raw_module = import_module_or_path("preprocessing_baselines.raw_baseline", args.raw_module) if "raw" in requested_baselines else None
    manual_module = import_module_or_path("preprocessing_baselines.manual_baseline", args.manual_module) if "manual" in requested_baselines else None
    direct_module = import_module_or_path("preprocessing_baselines.direct_llm_baseline", args.direct_llm_module) if "direct_llm" in requested_baselines else None
    full_module = import_module_from_path("evaluation_full_mediator", args.full_mediator_module) if ("full_mediator" in requested_baselines or requested_ablations) else None

    explicit_requests = {
        "visitor_presence_detection": args.visitor_request,
        "fall_detection": args.fall_request,
        "adl_recognition": args.adl_request,
        "domestic_sound_monitoring": args.audio_request,
    }

    all_rows: List[Dict[str, Any]] = []
    index: Dict[str, Any] = {
        "schema_version": "context_pipeline_generation_index_v1",
        "contexts_file": str(args.contexts),
        "out_dir": str(out_root),
        "baselines": requested_baselines,
        "ablations": requested_ablations,
        "methods": [
            {
                "method_id": s["method_id"],
                "method_kind": s["method_kind"],
                "baseline_id": s.get("baseline_id"),
                "ablation_mode": s.get("ablation_mode"),
                "parent_method": s.get("parent_method"),
                "method_label": s.get("method_label"),
            }
            for s in run_specs
        ],
        "contexts": {},
    }

    total_runs = len(scenarios) * len(run_specs)
    with make_progress(total_runs, enabled=not args.no_progress) as progress:
        for ordinal, sid, scenario in scenarios:
            task = scenario_task(scenario)
            space = scenario_space(scenario)
            params = scenario_ci_params(scenario)
            context_dir = out_root / sid
            request_dir = context_dir / "request"
            request_dir.mkdir(parents=True, exist_ok=True)

            app_request_path = resolve_request_path(task, args.app_request_dir, explicit_requests)
            app_request = load_json(app_request_path)
            context_request = overlay_context_on_app_request(app_request, scenario, sid)
            request_path = request_dir / "context_app_request.json"
            write_json(context_request, request_path)
            write_json(scenario, request_dir / "context_scenario.json")

            context_rows: List[Dict[str, Any]] = []
            index["contexts"][sid] = {
                "task": task,
                "space": space,
                "context_family": scenario.get("context_family"),
                "request_path": str(request_path),
                "source_app_request_path": str(app_request_path),
                "methods": {},
                "baselines": {},
                "ablations": {},
            }

            for spec in run_specs:
                method_id = str(spec["method_id"])
                method_kind = str(spec["method_kind"])
                baseline = str(spec["baseline"])
                ablation_mode = spec.get("ablation_mode")
                method_dir = context_dir.joinpath(*spec["dir_parts"])
                baseline_dir = method_dir  # Backward-compatible local variable used by writer code.
                baseline_dir.mkdir(parents=True, exist_ok=True)
                progress.set_postfix_str(f"{sid} {method_id}")
                result: Optional[Dict[str, Any]] = None
                error: Optional[str] = None
                tb: Optional[str] = None

                try:
                    if baseline == "raw":
                        result = run_raw_baseline(raw_module, operator_catalog, context_request, args.candidate_generator)
                    elif baseline == "manual":
                        result = run_manual_baseline(
                            manual_module,
                            operator_catalog,
                            context_request,
                            args.candidate_generator,
                            task=task,
                            space=space,
                            max_depth=args.max_depth,
                            max_states=args.max_states,
                        )
                    elif baseline == "direct_llm":
                        result = run_direct_llm_baseline(
                            direct_module,
                            operator_catalog,
                            context_request,
                            environment=scenario,
                            candidate_generator=args.candidate_generator,
                            max_depth=args.max_depth,
                            max_states=args.max_states,
                            llm_model=args.llm_model,
                            llm_temperature=args.llm_temperature,
                            openai_api_key=args.openai_api_key,
                        )
                    elif baseline == "full_mediator":
                        result = run_full_mediator(full_module, args, request_path, baseline_dir)
                    elif method_kind == "ablation":
                        result = run_full_mediator(full_module, args, request_path, baseline_dir, ablation_mode=str(ablation_mode))
                        # Mirror full_mediator's usual stage files for easier inspection.
                        if result:
                            if result.get("candidate_generation_result") is not None:
                                write_json(result["candidate_generation_result"], baseline_dir / "candidate_pipelines.json")
                            if result.get("ci_evaluation_result") is not None:
                                write_json(result["ci_evaluation_result"], baseline_dir / "ci_evaluation.json")
                            if result.get("pipeline_selection_result") is not None:
                                write_json(result["pipeline_selection_result"], baseline_dir / "pipeline_selection.json")
                            if result.get("privacy_probe_stage_result") is not None:
                                write_json(result["privacy_probe_stage_result"], baseline_dir / "privacy_probe_stage_result.json")
                    else:
                        raise ValueError(f"Unknown baseline {baseline!r}")
                except Exception as exc:  # Keep sweeping through all contexts unless fail-fast.
                    error = repr(exc)
                    tb = traceback.format_exc()
                    result = {
                        "schema_version": "method_error_v1",
                        "baseline": baseline,
                        "method_id": method_id,
                        "method_kind": method_kind,
                        "baseline_id": spec.get("baseline_id"),
                        "ablation_mode": ablation_mode,
                        "parent_method": spec.get("parent_method"),
                        "request_id": context_request.get("request_identity", {}).get("request_id"),
                        "scenario_id": sid,
                        "decision": {"decision": "error", "selected_pipeline_id": None, "reason": error},
                        "error": error,
                        "traceback": tb,
                    }
                    write_text(tb, baseline_dir / "traceback.txt")
                    if args.fail_fast:
                        raise

                cand = selected_candidate(result or {})
                paths = write_pipeline_code_and_metadata(cand, result or {}, baseline_dir)
                fields = candidate_summary_fields(cand)
                row: Dict[str, Any] = {
                    "scenario_id": sid,
                    "task": task,
                    "context": params.get("context"),
                    "space": params.get("space"),
                    "sender": params.get("sender"),
                    "subject": params.get("subject"),
                    "recipient": params.get("recipient"),
                    "purpose": params.get("purpose"),
                    "transmission_principle": params.get("transmission_principle"),
                    "context_family": scenario.get("context_family"),
                    "method_id": method_id,
                    "method_kind": method_kind,
                    "method_label": spec.get("method_label"),
                    "baseline": baseline,
                    "baseline_id": spec.get("baseline_id"),
                    "ablation_mode": ablation_mode,
                    "parent_method": spec.get("parent_method"),
                    "app_request_path": str(app_request_path),
                    "context_request_path": str(request_path),
                    "decision": decision_text(result or {}),
                    **fields,
                    **result_diagnostic_fields(result or {}),
                    "baseline_output_dir": str(baseline_dir),  # Backward-compatible alias.
                    "method_output_dir": str(method_dir),
                    "result_json": paths.get("result_json"),
                    "selected_pipeline_json": paths.get("selected_pipeline_json"),
                    "pipeline_spec_json": paths.get("pipeline_spec_json"),
                    "run_pipeline_py": paths.get("run_pipeline_py"),
                    "error": error,
                }
                context_rows.append(row)
                all_rows.append(row)
                index["contexts"][sid]["methods"][method_id] = row
                if method_kind == "baseline":
                    index["contexts"][sid]["baselines"][str(spec.get("baseline_id") or baseline)] = row
                elif method_kind == "ablation":
                    index["contexts"][sid]["ablations"][str(ablation_mode)] = row
                progress.update(1)

            write_context_summary(context_dir, scenario, context_rows)
            write_csv(context_rows, context_dir / "context_summary.csv")

    write_json(index, out_root / "index.json")
    write_json(all_rows, out_root / "summary.json")
    write_csv(all_rows, out_root / "summary.csv")

    by_context_simple = {
        sid: {method_id: method_summary_entry(entry) for method_id, entry in meta.get("methods", {}).items()}
        for sid, meta in index["contexts"].items()
    }
    write_json(by_context_simple, out_root / "summary_by_context.json")

    print(json.dumps({
        "contexts": len(scenarios),
        "baselines": requested_baselines,
        "ablations": requested_ablations,
        "methods": [s["method_id"] for s in run_specs],
        "runs": len(all_rows),
        "out_dir": str(out_root),
        "summary_csv": str(out_root / "summary.csv"),
        "summary_json": str(out_root / "summary.json"),
        "index_json": str(out_root / "index.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
