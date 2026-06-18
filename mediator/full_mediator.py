#!/usr/bin/env python3
"""
full_mediator.py

End-to-end driver for the SmartPriv/Prism-style privacy mediator.

Implemented stages:
  1. Load structured application request and optional environment/sensor metadata.
     Natural-language -> structured parsing is intentionally a future front-end.
  2. Generate candidate preprocessing pipelines using symbolic operator contracts.
  3. Evaluate candidate residual CI flows using hard constraints and optional LLM norm judgment.
  4. Optionally run empirical privacy probes after CI evaluation, if transformed artifacts exist.
  5. Select the least-revealing feasible pipeline using a context-specific residual-risk function.
  6. Emit one final decision: selected preprocessing pipeline, consent/review, or no-compromise.

Important note about probes:
  The mediator cannot run privacy probes until a preprocessing pipeline has produced
  transformed artifacts. Therefore probes are optional in this driver. If an artifact
  manifest is supplied, the mediator attaches empirical probe results and the selector
  uses the combined residual D(p)=max(D_meta,D_probe). If no artifacts are supplied,
  selection falls back to metadata residuals from operator contracts.

Example:
  python mediator/full_mediator.py \
    --operators norms/operator_contracts.json \
    --request app_requests/request_home_guest_audio_command_safety.json \
    --constraints norms/ci_constraints.json \
    --candidate-generator mediator/generate_pipeline_candidates.py \
    --evaluator mediator/contextual_integrity_evaluator.py \
    --selector mediator/pipeline_selection.py \
    --out-dir runs/audio_demo

With LLM norm judgment:
  export OPENAI_API_KEY=...
  python mediator/full_mediator.py ... --use-llm --llm-model gpt-4o-mini --top-k-for-llm 5

With privacy probes after transformed artifacts exist:
  python mediator/full_mediator.py ... \
    --probe-artifacts runs/audio_demo/artifact_manifest.json \
    --probe-config probe_config.example.json
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def import_module_from_path(module_name: str, path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def infer_default_path(name: str) -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / name,
        Path.cwd() / name,
        Path.cwd() / "mediator" / name,
        Path("/mnt/data") / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path(name)


def read_optional_json(path: Optional[str | Path]) -> Optional[Dict[str, Any]]:
    return load_json(path) if path else None


def require_structured_request(request_path: str | Path) -> Dict[str, Any]:
    path = Path(request_path)
    if path.suffix.lower() != ".json":
        raise ValueError(
            "This implementation expects a post-LLM structured application request JSON. "
            "Natural-language parsing should run before full_mediator.py."
        )
    request = load_json(path)
    if "utility_contract" not in request or "ci_context" not in request:
        raise ValueError("Request JSON is missing utility_contract or ci_context; expected smartpriv_application_request_v1 format.")
    return request


def call_with_supported_kwargs(fn, kwargs: Dict[str, Any]):
    """Call fn while filtering optional kwargs for older local module versions."""
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def run_candidate_generation(
    generator,
    operators: Dict[str, Any],
    request: Dict[str, Any],
    max_depth: int,
    max_states: int,
) -> Dict[str, Any]:
    if not hasattr(generator, "enumerate_candidates"):
        raise AttributeError("Candidate generator does not expose enumerate_candidates(...)")
    return call_with_supported_kwargs(generator.enumerate_candidates, {
        "operator_catalog": operators,
        "request": request,
        "max_depth": max_depth,
        "max_states": max_states,
    })


def run_ci_evaluation(
    evaluator,
    candidate_result: Dict[str, Any],
    request: Dict[str, Any],
    constraints: Dict[str, Any],
    environment: Optional[Dict[str, Any]],
    sensor_stream: Optional[Dict[str, Any]],
    use_llm: bool,
    llm_model: str,
    llm_temperature: float,
    llm_confidence_threshold: float,
    top_k_for_llm: Optional[int],
) -> Dict[str, Any]:
    if not hasattr(evaluator, "evaluate_candidates"):
        raise AttributeError("CI evaluator does not expose evaluate_candidates(...)")
    return call_with_supported_kwargs(evaluator.evaluate_candidates, {
        "candidate_output": candidate_result,
        "request": request,
        "constraints": constraints,
        "environment": environment,
        "sensor_stream": sensor_stream,
        "use_llm": use_llm,
        "llm_model": llm_model,
        "llm_temperature": llm_temperature,
        "llm_confidence_threshold": llm_confidence_threshold,
        "top_k_for_llm": top_k_for_llm,
        "unknown_policy": "no_match",
    })


def run_pipeline_selection(
    selector,
    candidate_result: Dict[str, Any],
    ci_result: Dict[str, Any],
    request: Dict[str, Any],
    probe_stage_result: Optional[Dict[str, Any]],
    selection_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not hasattr(selector, "select_pipeline"):
        raise AttributeError("Pipeline selector does not expose select_pipeline(...)")
    return call_with_supported_kwargs(selector.select_pipeline, {
        "candidate_generation_result": candidate_result,
        "ci_evaluation_result": ci_result,
        "request": request,
        "probe_stage_result": probe_stage_result,
        "selection_config": selection_config,
    })


def run_optional_privacy_probes(
    preliminary_mediator_result: Dict[str, Any],
    probe_artifacts_path: Optional[str | Path],
    probe_config_path: Optional[str | Path] = None,
    probe_package_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run empirical probes if an artifact manifest is supplied.

    Returns a stage result with status skipped/ok/error.
    """
    if not probe_artifacts_path:
        return {
            "schema_version": "smartpriv_privacy_probe_stage_v1",
            "status": "skipped",
            "reason": "No --probe-artifacts manifest supplied.",
            "selected_pipeline_id": None,
        }

    if probe_package_dir:
        sys.path.insert(0, str(Path(probe_package_dir).resolve()))

    # Also allow mediator/privacy_probes or project-root/privacy_probes.
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, Path.cwd(), Path.cwd() / "mediator"]:
        if (candidate / "privacy_probes").exists():
            sys.path.insert(0, str(candidate))

    try:
        from privacy_probes.mediator_integration import load_artifact_manifest, run_probes_for_mediator_result  # type: ignore
    except Exception as exc:
        return {
            "schema_version": "smartpriv_privacy_probe_stage_v1",
            "status": "error",
            "reason": f"Could not import privacy_probes package: {exc}",
            "selected_pipeline_id": None,
        }

    try:
        artifacts = load_artifact_manifest(probe_artifacts_path)
        probe_config = load_json(probe_config_path) if probe_config_path else None
        return run_probes_for_mediator_result(
            mediator_result=preliminary_mediator_result,
            artifacts=artifacts,
            probe_config=probe_config,
        )
    except Exception as exc:
        return {
            "schema_version": "smartpriv_privacy_probe_stage_v1",
            "status": "error",
            "reason": f"Privacy probe stage failed: {exc}",
            "selected_pipeline_id": None,
        }


def selected_candidate_from_selection(candidate_result: Dict[str, Any], selection_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    selected_id = (selection_result.get("decision") or {}).get("selected_pipeline_id")
    if not selected_id:
        return None
    for cand in candidate_result.get("candidates", []) or []:
        if cand.get("pipeline_id") == selected_id:
            return cand
    return None


def run_mediator(
    operators_path: str | Path,
    request_path: str | Path,
    constraints_path: str | Path,
    environment_path: Optional[str | Path] = None,
    sensor_stream_path: Optional[str | Path] = None,
    candidate_generator_path: Optional[str | Path] = None,
    evaluator_path: Optional[str | Path] = None,
    selector_path: Optional[str | Path] = None,
    max_depth: int = 7,
    max_states: int = 25000,
    use_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
    llm_temperature: float = 0.0,
    llm_confidence_threshold: float = 0.75,
    top_k_for_llm: Optional[int] = None,
    probe_artifacts_path: Optional[str | Path] = None,
    probe_config_path: Optional[str | Path] = None,
    probe_package_dir: Optional[str | Path] = None,
    selection_config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    request = require_structured_request(request_path)
    operators = load_json(operators_path)
    constraints = load_json(constraints_path)
    environment = read_optional_json(environment_path)
    sensor_stream = read_optional_json(sensor_stream_path)
    selection_config = read_optional_json(selection_config_path)

    gen_path = Path(candidate_generator_path) if candidate_generator_path else infer_default_path("generate_pipeline_candidates.py")
    eval_path = Path(evaluator_path) if evaluator_path else infer_default_path("contextual_integrity_evaluator.py")
    sel_path = Path(selector_path) if selector_path else infer_default_path("pipeline_selection.py")

    generator = import_module_from_path("smartpriv_candidate_generator", gen_path)
    evaluator = import_module_from_path("smartpriv_ci_evaluator", eval_path)
    selector = import_module_from_path("smartpriv_pipeline_selector", sel_path)

    candidate_result = run_candidate_generation(
        generator=generator,
        operators=operators,
        request=request,
        max_depth=max_depth,
        max_states=max_states,
    )

    ci_result = run_ci_evaluation(
        evaluator=evaluator,
        candidate_result=candidate_result,
        request=request,
        constraints=constraints,
        environment=environment,
        sensor_stream=sensor_stream,
        use_llm=use_llm,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        llm_confidence_threshold=llm_confidence_threshold,
        top_k_for_llm=top_k_for_llm,
    )

    # Preliminary selection based on metadata+CI only. This supplies a selected
    # pipeline ID for the optional probe stage if transformed artifacts exist.
    preliminary_selection = run_pipeline_selection(
        selector=selector,
        candidate_result=candidate_result,
        ci_result=ci_result,
        request=request,
        probe_stage_result=None,
        selection_config=selection_config,
    )

    preliminary_result = {
        "schema_version": "smartpriv_full_mediator_preliminary_output_v1",
        "request_id": request.get("request_identity", {}).get("request_id"),
        "scenario_id": request.get("request_identity", {}).get("scenario_id"),
        "decision": preliminary_selection.get("decision"),
        "selected_candidate": selected_candidate_from_selection(candidate_result, preliminary_selection),
        "candidate_generation_result": candidate_result,
        "ci_evaluation_result": ci_result,
        "pipeline_selection_result": preliminary_selection,
    }

    probe_stage_result = run_optional_privacy_probes(
        preliminary_mediator_result=preliminary_result,
        probe_artifacts_path=probe_artifacts_path,
        probe_config_path=probe_config_path,
        probe_package_dir=probe_package_dir,
    )

    # Final selection can use empirical probe residuals if available.
    final_selection = run_pipeline_selection(
        selector=selector,
        candidate_result=candidate_result,
        ci_result=ci_result,
        request=request,
        probe_stage_result=probe_stage_result,
        selection_config=selection_config,
    )

    selected_candidate = selected_candidate_from_selection(candidate_result, final_selection)

    return {
        "schema_version": "smartpriv_full_mediator_output_v2",
        "request_id": request.get("request_identity", {}).get("request_id"),
        "scenario_id": request.get("request_identity", {}).get("scenario_id"),
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
        },
        "decision": final_selection.get("decision"),
        "selected_candidate": selected_candidate,
        "candidate_generation_result": candidate_result,
        "ci_evaluation_result": ci_result,
        "privacy_probe_stage_result": probe_stage_result,
        "pipeline_selection_result": final_selection,
        "preliminary_pipeline_selection_result": preliminary_selection,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run candidate generation, CI evaluation, optional probes, and least-revealing selection.")
    p.add_argument("--operators", required=True, help="Path to norms/operator_contracts.json")
    p.add_argument("--request", required=True, help="Path to structured application request JSON")
    p.add_argument("--constraints", required=True, help="Path to norms/ci_constraints.json")
    p.add_argument("--environment", help="Optional structured environment JSON")
    p.add_argument("--sensor-stream", help="Optional concrete sensor stream metadata JSON")
    p.add_argument("--candidate-generator", help="Path to generate_pipeline_candidates.py; default: same directory/CWD")
    p.add_argument("--evaluator", help="Path to contextual_integrity_evaluator.py; default: same directory/CWD")
    p.add_argument("--selector", help="Path to pipeline_selection.py; default: same directory/CWD")
    p.add_argument("--out-dir", default="mediator_run", help="Directory for outputs")
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--max-states", type=int, default=25000)

    p.add_argument("--use-llm", action="store_true", help="Use LLM-assisted norm judgment after hard constraints pass")
    p.add_argument("--openai-api-key", help="Optional API key; otherwise use OPENAI_API_KEY env var")
    p.add_argument("--llm-model", default="gpt-4o-mini")
    p.add_argument("--llm-temperature", type=float, default=0.0)
    p.add_argument("--llm-confidence-threshold", type=float, default=0.75)
    p.add_argument("--top-k-for-llm", type=int, help="Only call LLM on top-k candidates by residual score")

    p.add_argument("--probe-artifacts", help="Optional artifact manifest from executed preprocessing pipeline")
    p.add_argument("--probe-config", help="Optional privacy probe config JSON")
    p.add_argument("--probe-package-dir", help="Optional directory containing privacy_probes package")
    p.add_argument("--selection-config", help="Optional selector config JSON with weights/method")

    args = p.parse_args(argv)

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    result = run_mediator(
        operators_path=args.operators,
        request_path=args.request,
        constraints_path=args.constraints,
        environment_path=args.environment,
        sensor_stream_path=args.sensor_stream,
        candidate_generator_path=args.candidate_generator,
        evaluator_path=args.evaluator,
        selector_path=args.selector,
        max_depth=args.max_depth,
        max_states=args.max_states,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_confidence_threshold=args.llm_confidence_threshold,
        top_k_for_llm=args.top_k_for_llm,
        probe_artifacts_path=args.probe_artifacts,
        probe_config_path=args.probe_config,
        probe_package_dir=args.probe_package_dir,
        selection_config_path=args.selection_config,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(result["candidate_generation_result"], out_dir / "candidate_pipelines.json")
    write_json(result["ci_evaluation_result"], out_dir / "ci_evaluation.json")
    write_json(result["privacy_probe_stage_result"], out_dir / "privacy_probe_stage_result.json")
    write_json(result["pipeline_selection_result"], out_dir / "pipeline_selection.json")
    write_json(result, out_dir / "full_mediator_result.json")

    print(json.dumps({
        "request_id": result.get("request_id"),
        "scenario_id": result.get("scenario_id"),
        "decision": result.get("decision"),
        "outputs": {
            "candidate_pipelines": str(out_dir / "candidate_pipelines.json"),
            "ci_evaluation": str(out_dir / "ci_evaluation.json"),
            "privacy_probe_stage_result": str(out_dir / "privacy_probe_stage_result.json"),
            "pipeline_selection": str(out_dir / "pipeline_selection.json"),
            "full_mediator_result": str(out_dir / "full_mediator_result.json"),
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
