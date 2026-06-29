#!/usr/bin/env python3
"""Run dynamic-context mediation experiments for SmartPriv/Prism.

This script is intentionally separate from evaluation.generate_pipelines_for_all_contexts.
It simulates *runtime context changes* for a fixed downstream task, runs the
full mediator and optional comparison baselines for each phase, writes the
selected/no-compromise decisions and pipelines, then optionally runs the
downstream utility evaluator on a small number of held-out samples and retains
transformed-output examples.

Typical use from the project root:

    python -m evaluation.dynamic_context_experiments \
      --operators norms/operator_contracts.json \
      --contexts survey/data/ci_focused_user_study_context.json \
      --app-request-dir app_requests/templates \
      --candidate-generator mediator/generate_pipeline_candidates.py \
      --constraints norms/ci_constraints.json \
      --evaluator mediator/contextual_integrity_evaluator.py \
      --selector mediator/pipeline_selection.py \
      --full-mediator-module mediator/full_mediator.py \
      --evaluate-utility-module evaluation.evaluate_utility \
      --out-dir runs/dynamic_contexts \
      --trace-set default \
      --utility-max-samples 2 \
      --utility-max-frames-per-sample 48 \
      --yes

The default trace-set includes two lightweight traces: ADL runtime context changes
and visitor/rental monitoring context changes. Use --trace-set all to include
fall and domestic sound traces too.

Progress is displayed for trace/phase generation, mediator decisions, utility evaluation, and example collection unless --no-progress is passed. Dynamic context switches are trace-driven: each phase is a structured context update ordered by dynamic_phase_index, not a real wall-clock timer. Use --phase-delay-seconds only for demos that should pause between phases.

Outputs under --out-dir:
  pipeline_generation/              context requests, mediator outputs, saved specs
  pipeline_generation/summary.json  flat rows compatible with evaluate_utility.py
  pipeline_generation/summary.csv   human-readable dynamic-decision summary
  pipeline_generation/summary_by_trace.json
  dynamic_context_scenarios.json    generated trace phases
  utility_eval/                     small-sample downstream utility results
  output_examples/                  copied examples of transformed artifacts/predictions
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


def set_progress_enabled(enabled: bool) -> None:
    global _PROGRESS_ENABLED
    _PROGRESS_ENABLED = bool(enabled)


def progress_enabled() -> bool:
    return bool(_PROGRESS_ENABLED)


def progress_write(msg: str) -> None:
    if not progress_enabled():
        return
    if _tqdm is not None:
        try:
            _tqdm.write(str(msg))
            return
        except Exception:
            pass
    print(str(msg), file=sys.stderr, flush=True)


def progress_iter(iterable, *, total: int, desc: str, unit: str = "item"):
    if progress_enabled() and _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)
    return iterable


TASK_TO_REQUEST_FILENAME: Dict[str, str] = {
    "visitor_presence_detection": "request_app_visitor_chokepoint_downstream_compatible.json",
    "fall_detection": "request_app_fall_le2i_pose_downstream_compatible.json",
    "adl_recognition": "request_app_adl_youhome_av_downstream_compatible.json",
    "domestic_sound_monitoring": "request_app_domestic_audio_chimehome_downstream_compatible.json",
}

TRACE_SET_DEFAULT = "default"
TRACE_SET_ALL = "all"
DEFAULT_METHODS = ["raw", "manual", "direct_llm", "full_mediator"]



# ---------------------------------------------------------------------------
# Basic IO / import helpers
# ---------------------------------------------------------------------------


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


def prepare_selected_only_utility_root(pipeline_root: Path, rows: Sequence[Dict[str, Any]]) -> Tuple[Path, List[Dict[str, Any]]]:
    """Write a filtered summary root containing only selected-output rows.

    evaluate_utility.py consumes the flat summary files under --pipeline-root.
    Dynamic experiments also produce review/no-compromise rows for baselines such
    as Direct LLM. Those rows are real decision outcomes, but they are not shared
    outputs and should not be scored as if they had utility.  This helper writes
    a small selected-only summary root whose rows still point at the original
    result/pipeline_spec paths, so the evaluator only materializes selected
    outputs.
    """
    selected = [dict(r) for r in rows if r.get("decision") == "select_pipeline"]
    utility_root = pipeline_root / "_utility_selected_only"
    utility_root.mkdir(parents=True, exist_ok=True)
    write_json(selected, utility_root / "summary.json")
    write_csv(selected, utility_root / "summary.csv")
    write_json({f"{r.get('scenario_id')}::{r.get('method_id')}": r for r in selected}, utility_root / "summary_by_context.json")
    index: Dict[str, Any] = {"schema_version": "dynamic_utility_selected_only_index_v1", "contexts": {}, "traces": {}}
    for r in selected:
        sid = str(r.get("scenario_id"))
        tid = str(r.get("dynamic_trace_id") or "")
        mid = str(r.get("method_id") or "")
        index["contexts"].setdefault(sid, {"scenario_id": sid, "task": r.get("task"), "trace_id": tid, "methods": {}})["methods"][mid] = r
        index["traces"].setdefault(tid, {"trace_id": tid, "phases": {}, "methods": {}})
        index["traces"][tid]["phases"].setdefault(sid, {"scenario_id": sid, "phase_index": r.get("dynamic_phase_index"), "methods": {}})["methods"][mid] = r
        index["traces"][tid]["methods"].setdefault(mid, []).append(r)
    write_json(index, utility_root / "index.json")
    return utility_root, selected


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


def call_with_supported_kwargs(fn, kwargs: Dict[str, Any]):
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def parse_csv_list(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Context/request construction
# ---------------------------------------------------------------------------


def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def as_str_list(x: Any) -> List[str]:
    out: List[str] = []
    for item in as_list(x):
        if item is None:
            continue
        if isinstance(item, list):
            out.extend(as_str_list(item))
        else:
            text = str(item).strip()
            if text:
                out.append(text)
    # Preserve order while removing duplicates.
    seen: set[str] = set()
    deduped: List[str] = []
    for item in out:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def scalar_or_list(x: Any) -> Any:
    vals = as_str_list(x)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    return vals


def display_values(x: Any) -> str:
    vals = as_str_list(x)
    return ", ".join(vals) if vals else ""


def first_scalar(x: Any) -> Optional[str]:
    vals = as_list(x)
    if not vals or vals[0] is None:
        return None
    return str(vals[0])


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


def scenario_task(scenario: Dict[str, Any]) -> str:
    params = scenario_ci_params(scenario)
    return str(scenario.get("task") or params.get("task") or "").strip()


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
    raise FileNotFoundError(f"Could not find app request for task {task!r}. Tried: " + ", ".join(str(p) for p in candidates))


def overlay_context_on_app_request(app_request: Dict[str, Any], scenario: Dict[str, Any], sid: str) -> Dict[str, Any]:
    """Return a context-specific request.

    The app request keeps the downstream utility/output-format contract. The
    dynamic phase supplies CI fields. This mirrors the static context-generation
    evaluator but marks request_identity as a dynamic trace phase.
    """
    import copy

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
    identity["dynamic_trace_id"] = scenario.get("dynamic_trace_id")
    identity["dynamic_phase_index"] = scenario.get("dynamic_phase_index")
    identity["dynamic_phase_label"] = scenario.get("dynamic_phase_label")
    identity["context_family"] = scenario.get("context_family")
    identity["generated_by"] = "evaluation.dynamic_context_experiments"

    ci = req.setdefault("ci_context", {})
    for src, dst in {
        "context": "context",
        "space": "space",
        "sender": "sender",
        "subject": "subject",
        "recipient": "recipient",
        "purpose": "purpose",
    }.items():
        ci[dst] = as_str_list(params.get(src))

    tp = params.get("transmission_principle") or params.get("transmissionPrinciple")
    ci["transmissionPrinciple_assumed"] = as_str_list(tp)

    # Dynamic phases may include explicit policy-condition metadata. This is
    # useful for scenarios such as disclosed outdoor rental cameras, where hard
    # rules may require coverage disclosure, security-only purpose, access
    # limits, and deletion/retention conditions in addition to a generic listing
    # disclosure term.
    if isinstance(scenario.get("metadata"), dict):
        md = ci.setdefault("metadata", {})
        if isinstance(md, dict):
            md.update(scenario["metadata"])

    tags = list(ci.get("social_context_tags", []) or [])
    for tag in [
        "dynamic_context_trace",
        scenario.get("dynamic_trace_id"),
        scenario.get("context_family"),
        *list(scenario.get("social_context_tags") or []),
    ]:
        if tag and tag not in tags:
            tags.append(str(tag))
    ci["social_context_tags"] = tags

    req["dynamic_context_phase"] = {
        "trace_id": scenario.get("dynamic_trace_id"),
        "phase_index": scenario.get("dynamic_phase_index"),
        "phase_label": scenario.get("dynamic_phase_label"),
        "phase_description": scenario.get("dynamic_phase_description"),
    }
    req["evaluation_context_scenario"] = {
        "scenario_id": sid,
        "task": scenario_task(scenario),
        "task_label": scenario.get("task_label"),
        "context_family": scenario.get("context_family"),
        "ci_parameters_scalar_context_only": params,
        "participant_vignette": scenario.get("participant_vignette"),
    }
    return req


def environment_from_dynamic_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    params = scenario_ci_params(scenario)
    env = {
        "schema_version": "smartpriv_dynamic_environment_v1",
        "dynamic_trace_id": scenario.get("dynamic_trace_id"),
        "dynamic_phase_index": scenario.get("dynamic_phase_index"),
        "dynamic_phase_label": scenario.get("dynamic_phase_label"),
        "dynamic_phase_description": scenario.get("dynamic_phase_description"),
        "context": as_str_list(params.get("context")),
        "space": as_str_list(params.get("space")),
        "sender": as_str_list(params.get("sender")),
        "subject": as_str_list(params.get("subject")),
        "recipient": as_str_list(params.get("recipient")),
        "purpose": as_str_list(params.get("purpose")),
        "transmissionPrinciple": as_str_list(params.get("transmission_principle")),
        "social_context_tags": list(scenario.get("social_context_tags") or []),
        "metadata": {
            "dynamic_context": True,
            "trace_id": scenario.get("dynamic_trace_id"),
            "phase_index": scenario.get("dynamic_phase_index"),
        },
    }
    if isinstance(scenario.get("metadata"), dict):
        env["metadata"].update(scenario["metadata"])
    return env


def make_phase(
    *,
    sid: str,
    trace_id: str,
    phase_index: int,
    phase_label: str,
    task: str,
    context_family: str,
    context: str,
    space: str,
    sender: str,
    subject: str,
    recipient: str,
    purpose: str,
    transmission_principle: Any,
    description: str,
    tags: Optional[Sequence[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    params = {
        "scenario_id": sid,
        "pipeline_stage": "output_to_application",
        "task": task,
        "context": context,
        "space": space,
        "sender": sender,
        "subject": subject,
        "recipient": recipient,
        "purpose": purpose,
        "transmission_principle": scalar_or_list(transmission_principle),
    }
    tp_values = as_str_list(transmission_principle)
    tp_display = display_values(transmission_principle)
    return {
        "schema_version": "smartpriv_dynamic_context_phase_v1",
        "scenario_id": sid,
        "task": task,
        "task_label": task.replace("_", " ").title(),
        "dynamic_trace_id": trace_id,
        "dynamic_phase_index": phase_index,
        "dynamic_phase_label": phase_label,
        "dynamic_phase_description": description,
        "context_family": context_family,
        "social_context_tags": list(tags or []),
        "ci_parameters_scalar_context_only": params,
        "participant_vignette": (
            f"Dynamic trace {trace_id}, phase {phase_index}: {description}. "
            f"Task={task}; context={context}; space={space}; sender={sender}; "
            f"subject={subject}; recipient={recipient}; purpose={purpose}; "
            f"transmission principle={tp_display}."
        ),
        "machine_flow_for_ci_constraints_context_only": {
            "pipeline_id": sid,
            "pipelineStage": ["output_to_application"],
            "context": [context],
            "space": [space],
            "sender": [sender],
            "subject": [subject],
            "recipient": [recipient],
            "purpose": [purpose],
            "transmissionPrinciple": tp_values,
            "informationType": {"sensorPrimitive": [], "interpretedObservation": [], "inferredInformationType": []},
            "sensingDataMetadata": {"sensingDevice": [], "contentType": [], "sensingModality": []},
            "metadata": {"dynamic_trace_id": trace_id, "dynamic_phase_index": phase_index, **dict(metadata or {})},
            "tags": list(tags or []),
            "attribute": [],
        },
        "metadata": dict(metadata or {}),
        "output_data_slot": {
            "status": "not_included_in_context_phase",
            "note": "Output/data is supplied by the selected mediator pipeline.",
        },
    }


def built_in_dynamic_scenarios(trace_set: str = TRACE_SET_DEFAULT) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return built-in dynamic trace phases.

    These are not context-recognition labels inferred from the data. They are
    structured runtime context updates supplied to the mediator to isolate
    adaptation behavior.
    """
    traces: Dict[str, List[Dict[str, Any]]] = {}

    traces["DT_ADL_HOME_SHARED_PRIVATE"] = [
        make_phase(
            sid="D_ADL_001", trace_id="DT_ADL_HOME_SHARED_PRIVATE", phase_index=1,
            phase_label="resident_local_personalization", task="adl_recognition",
            context_family="expected_use", context="home", space="living_room",
            sender="owner_controlled_device", subject="resident", recipient="downstream_application",
            purpose="personalization", transmission_principle="local_processing",
            description="resident uses an ADL/personalization app locally in their own living room",
            tags=["routine_home_use"],
        ),
        make_phase(
            sid="D_ADL_002", trace_id="DT_ADL_HOME_SHARED_PRIVATE", phase_index=2,
            phase_label="party_guests_present", task="adl_recognition",
            context_family="shared_space_gray", context="home", space="living_room",
            sender="owner_controlled_device", subject="guest", recipient="downstream_application",
            purpose="routine_monitoring", transmission_principle="continuous_collection",
            description="the same ADL app is active while guests are present in a shared living room",
            tags=["guests_present", "weak_preference_channel", "bystanders_present"],
        ),
        make_phase(
            sid="D_ADL_003", trace_id="DT_ADL_HOME_SHARED_PRIVATE", phase_index=3,
            phase_label="roommate_kitchen_routine", task="adl_recognition",
            context_family="shared_space_gray", context="home", space="kitchen",
            sender="owner_controlled_device", subject="roommate", recipient="downstream_application",
            purpose="routine_monitoring", transmission_principle="continuous_collection",
            description="routine monitoring covers a roommate in a shared kitchen",
            tags=["co_resident_present", "weak_preference_channel"],
        ),
        make_phase(
            sid="D_ADL_004", trace_id="DT_ADL_HOME_SHARED_PRIVATE", phase_index=4,
            phase_label="rental_hidden_monitoring", task="adl_recognition",
            context_family="hard_violation", context="short_term_rental", space="living_room",
            sender="host_controlled_device", subject="guest", recipient="host",
            purpose="routine_monitoring", transmission_principle="hidden",
            description="a rental host attempts hidden routine monitoring in a rental living room",
            tags=["hidden_monitoring", "short_term_rental"],
        ),
        make_phase(
            sid="D_ADL_005", trace_id="DT_ADL_HOME_SHARED_PRIVATE", phase_index=5,
            phase_label="care_home_authorized_routine", task="adl_recognition",
            context_family="sensitive_care", context="long_term_care", space="common_area",
            sender="patient_or_family_device", subject="resident", recipient="caregiver",
            purpose="routine_monitoring", transmission_principle="authorized_personnel_only",
            description="authorized care staff monitor ADLs in a long-term-care common area",
            tags=["care_setting", "authorized_care"],
        ),
    ]

    traces["DT_VISITOR_RENTAL_ENTRY"] = [
        make_phase(
            sid="D_VIS_001", trace_id="DT_VISITOR_RENTAL_ENTRY", phase_index=1,
            phase_label="home_entry_event_triggered", task="visitor_presence_detection",
            context_family="expected_use", context="home", space="entrance",
            sender="owner_controlled_device", subject="visitor", recipient="homeowner",
            purpose="security_monitoring", transmission_principle="event_triggered_collection",
            description="resident-owned entryway monitoring detects a visitor only when an event occurs",
            tags=["expected_security_use"],
        ),
        make_phase(
            sid="D_VIS_002", trace_id="DT_VISITOR_RENTAL_ENTRY", phase_index=2,
            phase_label="rental_outdoor_disclosed_policy_compliant", task="visitor_presence_detection",
            context_family="policy_condition", context="short_term_rental", space="outdoor",
            sender="host_controlled_device", subject="guest", recipient="host",
            purpose="security_monitoring",
            transmission_principle=[
                # Generic disclosure terms.
                "disclosed",
                "listing_disclosure_required",
                "written_notice_required",
                # Outdoor/location/coverage terms.  The rule set has evolved over
                # time, so include stable aliases used by different policy-rule
                # encodings rather than relying on one spelling.
                "coverage_disclosed",
                "coverage_area_disclosed",
                "camera_coverage_disclosed",
                "device_coverage_disclosed",
                "surveillance_coverage_disclosed",
                "location_disclosed",
                "outdoor_location_disclosed",
                "camera_location_disclosed",
                "device_location_disclosed",
                "surveillance_location_disclosed",
                "outdoor_only",
                "only_outdoor_area",
                "no_indoor_coverage",
                "no_private_indoor_spaces",
                # Security-only terms.
                "security_only",
                "security_monitoring_only",
                "surveillance_purpose_security_only",
                # Exact token required by vrbo_outdoor_surveillance_security_only.
                "security_purpose_only",
                # Access / retention / deletion terms.
                "limited_access",
                "authorized_access_only",
                "limited_to_authorized_personnel",
                # Exact token required by vrbo_surveillance_data_limited_access_delete.
                "authorized_personnel_only",
                "retention_limited",
                "limited_retention",
                "delete_after_stay",
                "delete_after_checkout",
                "deleted_after_stay",
                "data_limited_access_delete",
                "surveillance_data_limited_access_delete",
            ],
            description=(
                "a rental host monitors only an outdoor entrance for security, with the "
                "device location/coverage disclosed in the listing and access/retention limits"
            ),
            tags=["short_term_rental", "disclosed_device", "outdoor_only", "policy_compliant_rental_camera"],
            metadata={
                "disclosed": True,
                "listing_disclosure_required": True,
                "written_notice_required": True,
                "outdoor_only": True,
                "only_outdoor_area": True,
                "indoor_coverage": False,
                "no_indoor_coverage": True,
                "no_private_indoor_spaces": True,
                "coverage_disclosed": True,
                "coverage_area_disclosed": True,
                "camera_coverage_disclosed": True,
                "device_coverage_disclosed": True,
                "surveillance_coverage_disclosed": True,
                "location_disclosed": True,
                "outdoor_location_disclosed": True,
                "camera_location_disclosed": True,
                "device_location_disclosed": True,
                "surveillance_location_disclosed": True,
                "security_only": True,
                "security_monitoring_only": True,
                "surveillance_purpose_security_only": True,
                "security_purpose_only": True,
                "limited_access": True,
                "authorized_access_only": True,
                "limited_to_authorized_personnel": True,
                "authorized_personnel_only": True,
                "access_limited_to_host": True,
                "retention_limited": True,
                "limited_retention": True,
                "retention_period_limited": True,
                "delete_after_stay": True,
                "delete_after_checkout": True,
                "deleted_after_stay": True,
                "data_limited_access_delete": True,
                "surveillance_data_limited_access_delete": True,
                # The CI evaluator's requiredConditions for metadata check values,
                # not just boolean field names. Keep exact rule-required tokens
                # as string values so metadata=[...] predicates can match.
                "policy_condition_terms": [
                    "device_location_disclosed",
                    "coverage_area_disclosed",
                    "security_purpose_only",
                    "authorized_personnel_only",
                    "limited_retention",
                ],
                # Some rule encodings check nested metadata.
                "disclosure": {
                    "listing": True,
                    "written_notice": True,
                    "location": True,
                    "coverage": True,
                },
                "surveillance": {
                    "outdoor_only": True,
                    "security_only": True,
                    "indoor_coverage": False,
                },
                "access": {
                    "limited": True,
                    "authorized_only": True,
                },
                "retention": {
                    "limited": True,
                    "delete_after_stay": True,
                    "delete_after_checkout": True,
                },
            },
        ),
        make_phase(
            sid="D_VIS_003", trace_id="DT_VISITOR_RENTAL_ENTRY", phase_index=3,
            phase_label="rental_indoor_hidden", task="visitor_presence_detection",
            context_family="hard_violation", context="short_term_rental", space="living_room",
            sender="host_controlled_device", subject="guest", recipient="host",
            purpose="energy_management", transmission_principle="hidden",
            description="a rental host attempts hidden indoor monitoring for energy management",
            tags=["hidden_monitoring", "short_term_rental"],
        ),
        make_phase(
            sid="D_VIS_004", trace_id="DT_VISITOR_RENTAL_ENTRY", phase_index=4,
            phase_label="workplace_continuous_performance", task="visitor_presence_detection",
            context_family="power_asymmetry", context="workplace", space="workspace",
            sender="data_controller", subject="employee", recipient="authorized_personnel",
            purpose="work_performance_monitoring", transmission_principle="continuous_collection",
            description="workplace monitoring shifts from security to continuous performance monitoring",
            tags=["workplace", "power_asymmetry"],
        ),
    ]

    traces["DT_FALL_CARE_HOME"] = [
        make_phase(
            sid="D_FALL_001", trace_id="DT_FALL_CARE_HOME", phase_index=1,
            phase_label="care_home_living_room_local", task="fall_detection",
            context_family="expected_use", context="long_term_care", space="living_room",
            sender="patient_or_family_device", subject="resident", recipient="caregiver",
            purpose="fall_detection", transmission_principle="local_processing",
            description="local fall detection in a long-term-care living room",
            tags=["safety_monitoring", "care_setting"],
        ),
        make_phase(
            sid="D_FALL_002", trace_id="DT_FALL_CARE_HOME", phase_index=2,
            phase_label="home_shared_roommate", task="fall_detection",
            context_family="shared_space_gray", context="home", space="living_room",
            sender="owner_controlled_device", subject="roommate", recipient="caregiver",
            purpose="fall_detection", transmission_principle="event_triggered_collection",
            description="fall monitoring in a shared living room may incidentally cover a roommate",
            tags=["co_resident_present", "event_triggered"],
        ),
        make_phase(
            sid="D_FALL_003", trace_id="DT_FALL_CARE_HOME", phase_index=3,
            phase_label="home_bedroom_cloud", task="fall_detection",
            context_family="sensitive_space", context="home", space="bedroom",
            sender="patient_or_family_device", subject="resident", recipient="caregiver",
            purpose="fall_detection", transmission_principle="cloud_processing",
            description="fall monitoring in a bedroom switches from local processing to cloud processing",
            tags=["sensitive_space", "cloud_processing"],
        ),
        make_phase(
            sid="D_FALL_004", trace_id="DT_FALL_CARE_HOME", phase_index=4,
            phase_label="rental_hidden_safety", task="fall_detection",
            context_family="hard_violation", context="short_term_rental", space="bedroom",
            sender="host_controlled_device", subject="guest", recipient="host",
            purpose="safety_monitoring", transmission_principle="hidden",
            description="a rental host attempts hidden fall/safety monitoring in a bedroom",
            tags=["hidden_monitoring", "sensitive_space"],
        ),
    ]

    traces["DT_AUDIO_HOME_RENTAL_WORK"] = [
        make_phase(
            sid="D_AUD_001", trace_id="DT_AUDIO_HOME_RENTAL_WORK", phase_index=1,
            phase_label="home_kitchen_speech_removed", task="domestic_sound_monitoring",
            context_family="expected_use", context="home", space="kitchen",
            sender="owner_controlled_device", subject="resident", recipient="downstream_application",
            purpose="communication", transmission_principle="speech_content_removed",
            description="home audio monitoring keeps non-speech cues while removing speech content",
            tags=["speech_content_removed"],
        ),
        make_phase(
            sid="D_AUD_002", trace_id="DT_AUDIO_HOME_RENTAL_WORK", phase_index=2,
            phase_label="home_party_guest_safety", task="domestic_sound_monitoring",
            context_family="shared_space_gray", context="home", space="living_room",
            sender="owner_controlled_device", subject="guest", recipient="downstream_application",
            purpose="safety_monitoring", transmission_principle="disclosed",
            description="sound monitoring runs during a party with guests present",
            tags=["guests_present", "bystanders_present"],
        ),
        make_phase(
            sid="D_AUD_003", trace_id="DT_AUDIO_HOME_RENTAL_WORK", phase_index=3,
            phase_label="rental_bedroom_audio", task="domestic_sound_monitoring",
            context_family="hard_violation", context="short_term_rental", space="bedroom",
            sender="host_controlled_device", subject="guest", recipient="host",
            purpose="routine_monitoring", transmission_principle="disclosed",
            description="a rental host attempts bedroom sound monitoring for routine monitoring",
            tags=["short_term_rental", "sensitive_space"],
        ),
        make_phase(
            sid="D_AUD_004", trace_id="DT_AUDIO_HOME_RENTAL_WORK", phase_index=4,
            phase_label="workplace_audio_safety", task="domestic_sound_monitoring",
            context_family="power_asymmetry", context="workplace", space="workspace",
            sender="data_controller", subject="employee", recipient="authorized_personnel",
            purpose="safety_monitoring", transmission_principle="disclosed",
            description="workplace audio monitoring is disclosed but covers employees",
            tags=["workplace", "power_asymmetry"],
        ),
    ]

    include_trace_ids: List[str]
    if trace_set in {TRACE_SET_ALL, "all_tasks"}:
        include_trace_ids = list(traces.keys())
    elif trace_set in {TRACE_SET_DEFAULT, "light", "smoke"}:
        include_trace_ids = ["DT_ADL_HOME_SHARED_PRIVATE", "DT_VISITOR_RENTAL_ENTRY"]
    else:
        requested = set(parse_csv_list(trace_set))
        aliases = {
            "adl": "DT_ADL_HOME_SHARED_PRIVATE",
            "visitor": "DT_VISITOR_RENTAL_ENTRY",
            "fall": "DT_FALL_CARE_HOME",
            "audio": "DT_AUDIO_HOME_RENTAL_WORK",
            "sound": "DT_AUDIO_HOME_RENTAL_WORK",
        }
        include_trace_ids = [aliases.get(x, x) for x in requested]
        missing = [x for x in include_trace_ids if x not in traces]
        if missing:
            raise ValueError(f"Unknown trace-set entries: {missing}. Valid aliases: default, all, adl, visitor, fall, audio.")

    scenarios: List[Dict[str, Any]] = []
    trace_meta: Dict[str, Any] = {}
    for tid in include_trace_ids:
        phases = traces[tid]
        scenarios.extend(phases)
        trace_meta[tid] = {
            "trace_id": tid,
            "task": phases[0]["task"] if phases else None,
            "phase_count": len(phases),
            "phase_ids": [p["scenario_id"] for p in phases],
            "description": "Runtime context changes supplied as structured metadata; sensor data is held to the same task dataset during utility checks.",
        }
    return scenarios, trace_meta


# ---------------------------------------------------------------------------
# Mediator output helpers
# ---------------------------------------------------------------------------


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
    # no_compromise/review decisions can include closest rejected candidates, but
    # those are diagnostics rather than outputs shared with the app.
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
    paths: Dict[str, str] = {}
    write_json(result, out_dir / "result.json")
    paths["result_json"] = str(out_dir / "result.json")

    if result.get("candidate_generation_result") is not None:
        write_json(result["candidate_generation_result"], out_dir / "candidate_pipelines.json")
        paths["candidate_pipelines_json"] = str(out_dir / "candidate_pipelines.json")
    if result.get("ci_evaluation_result") is not None:
        write_json(result["ci_evaluation_result"], out_dir / "ci_evaluation.json")
        paths["ci_evaluation_json"] = str(out_dir / "ci_evaluation.json")
    if result.get("pipeline_selection_result") is not None:
        write_json(result["pipeline_selection_result"], out_dir / "pipeline_selection.json")
        paths["pipeline_selection_json"] = str(out_dir / "pipeline_selection.json")
    if result.get("privacy_probe_stage_result") is not None:
        write_json(result["privacy_probe_stage_result"], out_dir / "privacy_probe_stage_result.json")
        paths["privacy_probe_stage_result_json"] = str(out_dir / "privacy_probe_stage_result.json")

    if not cand:
        write_text("No selected candidate was available for this dynamic phase.\n", out_dir / "NO_SELECTED_PIPELINE.txt")
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

    runnable = '''#!/usr/bin/env python3
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
    diag = result.get("no_compromise_diagnostics") or {}
    decision = result.get("decision") if isinstance(result.get("decision"), dict) else {}
    top_rules = diag.get("top_failed_rules") or (decision.get("selection_failure_diagnostics") or {}).get("top_failed_rules") or []
    return {
        "no_compromise_reason": diag.get("reason") or decision.get("reason"),
        "no_compromise_candidate_count": diag.get("candidate_count"),
        "no_compromise_ci_feasible_count": diag.get("ci_feasible_count"),
        "no_compromise_hard_rejection_count": diag.get("hard_rejection_count"),
        "no_compromise_top_failed_rules": json.dumps(top_rules, sort_keys=False),
    }


def run_full_mediator(full_module, args: argparse.Namespace, request_path: Path, environment_path: Path, out_dir: Path) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "operators_path": args.operators,
        "request_path": request_path,
        "constraints_path": args.constraints,
        "environment_path": environment_path,
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
        "ablation_modes": [],
    }
    return call_with_supported_kwargs(full_module.run_mediator, kwargs)


def run_raw_baseline(raw_module, operator_catalog: Dict[str, Any], request: Dict[str, Any], candidate_generator: Optional[str]) -> Dict[str, Any]:
    return raw_module.run_raw_baseline(
        operator_catalog=operator_catalog,
        request=request,
        candidate_generator_path=candidate_generator,
    )


def run_manual_baseline(
    manual_module,
    operator_catalog: Dict[str, Any],
    request: Dict[str, Any],
    candidate_generator: Optional[str],
    task: str,
    space: str,
    max_depth: int,
    max_states: int,
) -> Dict[str, Any]:
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
        return manual_module.run_manual_baseline(
            operator_catalog=operator_catalog,
            request=request,
            candidate_generator_path=candidate_generator,
            max_depth=max_depth,
            max_states=max_states,
        )
    raise AttributeError("Manual baseline module exposes neither run_baseline nor run_manual_baseline.")


def run_direct_llm_baseline(
    direct_module,
    operator_catalog: Dict[str, Any],
    request: Dict[str, Any],
    environment: Dict[str, Any],
    candidate_generator: Optional[str],
    max_depth: int,
    max_states: int,
    llm_model: str,
    llm_temperature: float,
    openai_api_key: Optional[str],
) -> Dict[str, Any]:
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


def method_label(method_id: str) -> str:
    return {
        "raw": "Raw",
        "manual": "Manual",
        "direct_llm": "Direct LLM",
        "full_mediator": "Full mediator",
    }.get(method_id, method_id)


# ---------------------------------------------------------------------------
# Utility evaluator integration / examples
# ---------------------------------------------------------------------------


def command_exists_python_module(module: str, cwd: Optional[str | Path]) -> bool:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        return proc.returncode == 0
    except Exception:
        return False


def run_utility_evaluator(args: argparse.Namespace, pipeline_root: Path, utility_out: Path, tasks: Sequence[str]) -> Dict[str, Any]:
    if args.no_utility:
        progress_write("[utility] skipped because --no-utility was passed")
        return {"status": "skipped", "reason": "--no-utility"}

    utility_out.mkdir(parents=True, exist_ok=True)
    task_arg = ",".join(sorted(set(tasks))) if tasks else "auto"
    cmd = [
        sys.executable,
        "-m",
        args.evaluate_utility_module,
        "--pipeline-root", str(pipeline_root),
        "--out-dir", str(utility_out),
        "--project-root", str(Path(args.project_root).resolve()),
        "--runtime-package", args.runtime_package,
        "--tasks", task_arg,
        "--methods", ",".join(sorted(set(args.methods_to_evaluate or ["full_mediator"]))),
        "--ablation-policy", "none",
        "--max-samples", str(args.utility_max_samples),
        "--yes",
        "--no-preflight-confirm",
        "--keep-intermediate-data",
    ]
    if args.utility_max_frames_per_sample is not None:
        cmd += ["--max-frames-per-sample", str(args.utility_max_frames_per_sample)]
    if args.device:
        cmd += ["--device", str(args.device)]
    if args.prefer_gpu_name:
        cmd += ["--prefer-gpu-name", str(args.prefer_gpu_name)]
    if args.utility_extra_args:
        cmd += shlex.split(args.utility_extra_args)

    log_path = utility_out / "dynamic_utility_eval.log"
    write_text("$ " + shlex.join(cmd) + "\n\n", log_path)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    root = Path(args.project_root).resolve()
    py_parts = [str(root), str(root / "mediator"), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join([p for p in py_parts if p])
    progress_write(f"[utility] running small-sample utility evaluator for tasks={task_arg}; max_samples={args.utility_max_samples}; log={log_path}")
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(proc.stdout or "")
    progress_write(f"[utility] finished small-sample utility evaluator rc={proc.returncode} status={'ok' if proc.returncode == 0 else 'error'}")
    result = {
        "status": "ok" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "cmd": cmd,
        "log_path": str(log_path),
        "utility_out_dir": str(utility_out),
        "tasks": sorted(set(tasks)),
    }
    if proc.returncode != 0:
        result["stdout_tail"] = (proc.stdout or "")[-5000:]
    return result


def collect_output_examples(utility_out: Path, example_root: Path, limit_per_result: int = 8) -> Dict[str, Any]:
    """Copy a small sample of retained transformed artifacts and predictions.

    The utility evaluator stores retained intermediate artifacts under each
    utility row's work directory when --keep-intermediate-data is used. This
    function gathers a few files so they are easy to inspect in one location.
    """
    example_root.mkdir(parents=True, exist_ok=True)
    progress_write(f"[examples] collecting retained output examples from {utility_out} into {example_root}")
    result_jsons = sorted(utility_out.glob("*/*/utility_result.json"))
    copied: List[Dict[str, Any]] = []
    allowed_suffixes = {".jpg", ".jpeg", ".png", ".wav", ".flac", ".npz", ".json", ".csv"}

    for res_path in progress_iter(result_jsons, total=len(result_jsons), desc="copy examples", unit="result"):
        try:
            res = load_json(res_path)
        except Exception:
            continue
        sid = str(res.get("scenario_id") or res_path.parents[1].name)
        method_id = str(res.get("method_id") or res_path.parent.name)
        dest = example_root / sid / method_id
        dest.mkdir(parents=True, exist_ok=True)

        # Always copy the compact result, predictions, and metrics if present.
        for key_path in [
            res_path,
            Path(str((res.get("downstream") or {}).get("output_csv", ""))),
            Path(str((res.get("downstream") or {}).get("metrics_json", ""))),
            Path(str(res.get("prepared_manifest", ""))),
        ]:
            if key_path and str(key_path) and key_path.exists() and key_path.is_file():
                target = dest / key_path.name
                if not target.exists():
                    shutil.copy2(key_path, target)
                    copied.append({"scenario_id": sid, "method_id": method_id, "source": str(key_path), "copy": str(target), "kind": "summary_or_prediction"})

        inter = Path(str(res.get("intermediate_artifact_dir") or (res.get("preprocessing") or {}).get("intermediate_artifact_dir") or ""))
        if not inter.exists():
            continue
        n = 0
        for p in sorted(inter.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in allowed_suffixes:
                continue
            # Skip huge transform record files after copying a few media examples.
            if p.name.endswith(".transforms.json") and n >= 1:
                continue
            rel = p.relative_to(inter)
            target = dest / "intermediate_artifacts" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            copied.append({"scenario_id": sid, "method_id": method_id, "source": str(p), "copy": str(target), "kind": "intermediate_artifact"})
            n += 1
            if n >= limit_per_result:
                break

    copy_paths = [item.get("copy") for item in copied]
    duplicate_copy_count = len(copy_paths) - len(set(copy_paths))
    manifest = {
        "schema_version": "dynamic_context_output_examples_v1",
        "utility_out_dir": str(utility_out),
        "example_root": str(example_root),
        "layout": "<scenario_id>/<method_id>/...",
        "copied_count": len(copied),
        "unique_copy_count": len(set(copy_paths)),
        "duplicate_copy_count": duplicate_copy_count,
        "examples": copied,
    }
    write_json(manifest, example_root / "output_examples_manifest.json")
    progress_write(f"[examples] copied {len(copied)} files; manifest={example_root / 'output_examples_manifest.json'}")
    return manifest


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run dynamic-context mediation/baseline experiments and small-sample utility checks.")
    p.add_argument("--operators", default="norms/operator_contracts.json")
    p.add_argument("--contexts", default="survey/data/ci_focused_user_study_context.json", help="Optional source context file; used for provenance only by built-in traces.")
    p.add_argument("--dynamic-contexts", default=None, help="Optional JSON file with custom dynamic context_scenarios. If omitted, built-in traces are generated.")
    p.add_argument("--trace-set", default=TRACE_SET_DEFAULT, help="default, all, or comma list/aliases: adl,visitor,fall,audio")
    p.add_argument("--app-request-dir", default="app_requests/templates")
    p.add_argument("--candidate-generator", default="mediator/generate_pipeline_candidates.py")
    p.add_argument("--constraints", default="norms/ci_constraints.json")
    p.add_argument("--evaluator", default="mediator/contextual_integrity_evaluator.py")
    p.add_argument("--selector", default="mediator/pipeline_selection.py")
    p.add_argument("--full-mediator-module", default="mediator/full_mediator.py")
    p.add_argument("--raw-module", default=None, help="Optional path to raw_baseline.py; defaults to preprocessing_baselines.raw_baseline")
    p.add_argument("--manual-module", default=None, help="Optional path to manual_baseline.py; defaults to preprocessing_baselines.manual_baseline")
    p.add_argument("--direct-llm-module", default=None, help="Optional path to direct_llm_baseline.py; defaults to preprocessing_baselines.direct_llm_baseline")
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="Comma list of methods to run: raw,manual,direct_llm,full_mediator. Use --methods full_mediator to reproduce the original dynamic-only run.")
    p.add_argument("--out-dir", default="runs/dynamic_contexts")
    p.add_argument("--project-root", default=".")
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--max-states", type=int, default=25000)
    p.add_argument("--scenario-ids", default="", help="Optional comma list of dynamic phase IDs to run.")

    # Full mediator options.
    p.add_argument("--full-mediator-use-llm", action="store_true")
    p.add_argument("--llm-model", default="gpt-4o")
    p.add_argument("--llm-temperature", type=float, default=0.0)
    p.add_argument("--llm-confidence-threshold", type=float, default=0.75)
    p.add_argument("--top-k-for-llm", type=int, default=None)
    p.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY"), help="API key passed to the direct LLM baseline, if it needs one.")
    p.add_argument("--probe-artifacts", default=None)
    p.add_argument("--probe-config", default=None)
    p.add_argument("--probe-package-dir", default=None)
    p.add_argument("--selection-config", default=None)

    # Explicit app requests if your layout differs.
    p.add_argument("--visitor-request", default=None)
    p.add_argument("--fall-request", default=None)
    p.add_argument("--adl-request", default=None)
    p.add_argument("--audio-request", default=None)

    # Utility evaluator options.
    p.add_argument("--no-utility", action="store_true", help="Only run dynamic mediator decisions; skip downstream utility examples.")
    p.add_argument("--evaluate-utility-module", default="evaluation.evaluate_utility")
    p.add_argument("--runtime-package", default="mediator.smartpriv_runtime")
    p.add_argument("--utility-max-samples", type=int, default=2, help="Number of held-out examples per selected dynamic phase. Default 2.")
    p.add_argument("--utility-max-frames-per-sample", type=int, default=48, help="Frame cap for visual tasks. Default 48.")
    p.add_argument("--utility-extra-args", default="", help="Extra arguments appended to evaluate_utility, parsed with shlex.")
    p.add_argument("--device", default="auto")
    p.add_argument("--prefer-gpu-name", default="RTX 2070")
    p.add_argument("--example-copy-limit", type=int, default=8)

    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bars/status messages.")
    p.add_argument("--phase-delay-seconds", type=float, default=0.0, help="Optional demo-only pause between dynamic phases. Default 0: trace phases advance immediately, not by wall clock time.")
    p.add_argument("--yes", "-y", action="store_true", help="Kept for symmetry with other evaluators; this script is noninteractive.")
    return p


def run_dynamic_generation(args: argparse.Namespace, scenarios: Sequence[Dict[str, Any]], trace_meta: Dict[str, Any]) -> Tuple[Path, List[Dict[str, Any]], Dict[str, Any]]:
    out_root = Path(args.out_dir)
    pipeline_root = out_root / "pipeline_generation"
    pipeline_root.mkdir(parents=True, exist_ok=True)

    requested_methods = parse_csv_list(args.methods) or ["full_mediator"]
    requested_methods = ["manual" if m == "manual_space_task" else m for m in requested_methods]
    valid_methods = {"raw", "manual", "direct_llm", "full_mediator"}
    bad = [m for m in requested_methods if m not in valid_methods]
    if bad:
        raise ValueError(f"Unknown methods {bad}. Valid methods: {sorted(valid_methods)}")
    args.methods_to_evaluate = list(requested_methods)

    # Persist the dynamic context file for reproducibility.
    context_doc = {
        "schema_version": "smartpriv_dynamic_context_scenarios_v1",
        "source_context_file": str(args.contexts) if args.contexts else None,
        "trace_set": args.trace_set,
        "note": "These phases simulate runtime context updates supplied to the mediator; context recognition itself is not evaluated here. Phase changes are trace-driven by dynamic_phase_index, not real wall-clock time.",
        "switching_semantics": {"mode": "trace_order", "ordered_by": "dynamic_phase_index", "wall_clock_time_used": False, "optional_demo_delay_seconds": args.phase_delay_seconds},
        "methods": requested_methods,
        "traces": trace_meta,
        "context_scenarios": list(scenarios),
    }
    write_json(context_doc, out_root / "dynamic_context_scenarios.json")
    write_json(context_doc, pipeline_root / "dynamic_context_scenarios.json")

    operator_catalog: Dict[str, Any] = load_json(args.operators)
    full_module = import_module_from_path("dynamic_context_full_mediator", args.full_mediator_module) if "full_mediator" in requested_methods else None
    raw_module = import_module_or_path("preprocessing_baselines.raw_baseline", args.raw_module) if "raw" in requested_methods else None
    manual_module = import_module_or_path("preprocessing_baselines.manual_baseline", args.manual_module) if "manual" in requested_methods else None
    direct_module = import_module_or_path("preprocessing_baselines.direct_llm_baseline", args.direct_llm_module) if "direct_llm" in requested_methods else None

    explicit_requests = {
        "visitor_presence_detection": args.visitor_request,
        "fall_detection": args.fall_request,
        "adl_recognition": args.adl_request,
        "domestic_sound_monitoring": args.audio_request,
    }

    wanted_sids = set(parse_csv_list(args.scenario_ids)) if args.scenario_ids else set()
    rows: List[Dict[str, Any]] = []
    index: Dict[str, Any] = {
        "schema_version": "dynamic_context_pipeline_generation_index_v2",
        "out_dir": str(pipeline_root),
        "methods": requested_methods,
        "traces": {},
        "contexts": {},
    }

    active_scenarios = [sc for sc in scenarios if not wanted_sids or str(sc.get("scenario_id") or "") in wanted_sids]
    total_runs = len(active_scenarios) * len(requested_methods)
    progress_write(f"[dynamic] running methods={requested_methods} for {len(active_scenarios)} dynamic context phases across {len(trace_meta)} traces ({total_runs} method-runs)")

    run_items: List[Tuple[int, Dict[str, Any], str]] = []
    for i, scenario in enumerate(active_scenarios, start=1):
        for method in requested_methods:
            run_items.append((i, scenario, method))

    for run_idx, (phase_ordinal, scenario, method_id) in enumerate(progress_iter(run_items, total=len(run_items), desc="dynamic methods", unit="run"), start=1):
        sid = str(scenario.get("scenario_id") or f"DYN_{phase_ordinal:03d}")
        task = scenario_task(scenario)
        params = scenario_ci_params(scenario)
        trace_id = str(scenario.get("dynamic_trace_id") or "dynamic_trace")
        progress_write(
            f"[dynamic] run {run_idx}/{len(run_items)} phase={sid} method={method_id} "
            f"trace={trace_id} task={task} context={display_values(params.get('context'))} "
            f"space={display_values(params.get('space'))} purpose={display_values(params.get('purpose'))}"
        )
        if args.phase_delay_seconds and args.phase_delay_seconds > 0 and run_idx > 1 and method_id == requested_methods[0]:
            progress_write(f"[dynamic] demo delay before next context phase: {args.phase_delay_seconds:.2f}s")
            time.sleep(float(args.phase_delay_seconds))

        context_dir = pipeline_root / sid
        method_dir = context_dir / "baselines" / method_id
        request_dir = context_dir / "request"
        method_dir.mkdir(parents=True, exist_ok=True)
        request_dir.mkdir(parents=True, exist_ok=True)

        app_request_path = resolve_request_path(task, args.app_request_dir, explicit_requests)
        app_request = load_json(app_request_path)
        request = overlay_context_on_app_request(app_request, scenario, sid)
        request_path = request_dir / "context_app_request.json"
        scenario_path = request_dir / "context_scenario.json"
        environment_path = request_dir / "environment.json"
        if not request_path.exists():
            write_json(request, request_path)
            write_json(scenario, scenario_path)
            write_json(environment_from_dynamic_scenario(scenario), environment_path)

        result: Dict[str, Any]
        error: Optional[str] = None
        tb: Optional[str] = None
        try:
            if method_id == "raw":
                result = run_raw_baseline(raw_module, operator_catalog, request, args.candidate_generator)
            elif method_id == "manual":
                result = run_manual_baseline(
                    manual_module,
                    operator_catalog,
                    request,
                    args.candidate_generator,
                    task=task,
                    space=display_values(params.get("space")),
                    max_depth=args.max_depth,
                    max_states=args.max_states,
                )
            elif method_id == "direct_llm":
                result = run_direct_llm_baseline(
                    direct_module,
                    operator_catalog,
                    request,
                    environment=environment_from_dynamic_scenario(scenario),
                    candidate_generator=args.candidate_generator,
                    max_depth=args.max_depth,
                    max_states=args.max_states,
                    llm_model=args.llm_model,
                    llm_temperature=args.llm_temperature,
                    openai_api_key=args.openai_api_key,
                )
            elif method_id == "full_mediator":
                result = run_full_mediator(full_module, args, request_path, environment_path, method_dir)
            else:  # guarded above
                raise ValueError(f"Unknown method {method_id!r}")
        except Exception as exc:
            error = repr(exc)
            tb = traceback.format_exc()
            result = {
                "schema_version": "dynamic_method_error_v1",
                "request_id": request.get("request_identity", {}).get("request_id"),
                "scenario_id": sid,
                "method_id": method_id,
                "decision": {"decision": "error", "selected_pipeline_id": None, "reason": error},
                "error": error,
                "traceback": tb,
            }
            write_text(tb, method_dir / "traceback.txt")
            if args.fail_fast:
                raise

        cand = selected_candidate(result)
        paths = write_pipeline_code_and_metadata(cand, result, method_dir)
        fields = candidate_summary_fields(cand)
        row = {
            "scenario_id": sid,
            "task": task,
            "dynamic_trace_id": trace_id,
            "dynamic_phase_index": scenario.get("dynamic_phase_index"),
            "dynamic_phase_label": scenario.get("dynamic_phase_label"),
            "context": display_values(params.get("context")),
            "space": display_values(params.get("space")),
            "sender": display_values(params.get("sender")),
            "subject": display_values(params.get("subject")),
            "recipient": display_values(params.get("recipient")),
            "purpose": display_values(params.get("purpose")),
            "transmission_principle": display_values(params.get("transmission_principle") or params.get("transmissionPrinciple")),
            "context_family": scenario.get("context_family"),
            "method_id": method_id,
            "method_kind": "baseline",
            "method_label": method_label(method_id),
            "baseline": method_id,
            "baseline_id": method_id,
            "ablation_mode": None,
            "parent_method": None,
            "app_request_path": str(app_request_path),
            "context_request_path": str(request_path),
            "environment_path": str(environment_path),
            "decision": decision_text(result),
            **fields,
            **result_diagnostic_fields(result),
            "method_output_dir": str(method_dir),
            "result_json": paths.get("result_json"),
            "selected_pipeline_json": paths.get("selected_pipeline_json"),
            "pipeline_spec_json": paths.get("pipeline_spec_json"),
            "candidate_pipelines_json": paths.get("candidate_pipelines_json"),
            "ci_evaluation_json": paths.get("ci_evaluation_json"),
            "pipeline_selection_json": paths.get("pipeline_selection_json"),
            "error": error,
        }
        out_type = row.get("final_output_type") or "none"
        out_schema = row.get("final_output_schema") or "none"
        ops = row.get("operators") or "none"
        progress_write(f"[dynamic] decision {sid}/{method_id}: {row.get('decision')} output={out_type}/{out_schema} operators={ops}")
        rows.append(row)
        ctx = index["contexts"].setdefault(sid, {"scenario_id": sid, "task": task, "trace_id": trace_id, "methods": {}})
        ctx["methods"][method_id] = row
        index["traces"].setdefault(trace_id, {"trace_id": trace_id, "phases": {}, "methods": {}})
        index["traces"][trace_id]["phases"].setdefault(sid, {"scenario_id": sid, "phase_index": scenario.get("dynamic_phase_index"), "methods": {}})["methods"][method_id] = row
        index["traces"][trace_id]["methods"].setdefault(method_id, []).append(row)

    write_json(rows, pipeline_root / "summary.json")
    write_csv(rows, pipeline_root / "summary.csv")
    write_json(index, pipeline_root / "index.json")

    # Trace-centric summary, with per-method switch/no-compromise counts.
    summary_by_trace: Dict[str, Any] = {}
    for trace_id, info in index["traces"].items():
        method_summaries: Dict[str, Any] = {}
        for method_id, m_rows in info.get("methods", {}).items():
            phases = sorted(m_rows, key=lambda r: int(r.get("dynamic_phase_index") or 0))
            switches = 0
            prev_sig = None
            for r in phases:
                sig = (r.get("decision"), r.get("final_output_type"), r.get("final_output_schema"), r.get("operators"))
                if prev_sig is not None and sig != prev_sig:
                    switches += 1
                prev_sig = sig
            method_summaries[method_id] = {
                "method_id": method_id,
                "phase_count": len(phases),
                "select_count": sum(1 for r in phases if r.get("decision") == "select_pipeline"),
                "no_compromise_count": sum(1 for r in phases if r.get("decision") == "no_compromise"),
                "review_count": sum(1 for r in phases if r.get("decision") in {"consent_or_review_required", "review"}),
                "error_count": sum(1 for r in phases if r.get("decision") == "error" or r.get("error")),
                "hard_violation_select_count": sum(1 for r in phases if r.get("context_family") == "hard_violation" and r.get("decision") == "select_pipeline"),
                "switches": switches,
                "phases": phases,
            }
        full_rows = [r for rows_ in info.get("methods", {}).values() for r in rows_]
        summary_by_trace[trace_id] = {
            "trace_id": trace_id,
            "task": full_rows[0].get("task") if full_rows else None,
            "phase_count": len(info.get("phases", {})),
            "methods": method_summaries,
            "phases": info.get("phases", {}),
        }
    write_json(summary_by_trace, pipeline_root / "summary_by_trace.json")
    write_json({f"{r['scenario_id']}::{r['method_id']}": r for r in rows}, pipeline_root / "summary_by_context.json")
    return pipeline_root, rows, summary_by_trace

def load_custom_dynamic_contexts(path: str | Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    doc = load_json(path)
    scenarios = doc.get("context_scenarios") if isinstance(doc, dict) else doc
    if not isinstance(scenarios, list):
        raise ValueError("Custom dynamic context file must be a list or contain context_scenarios.")
    trace_meta: Dict[str, Any] = {}
    for sc in scenarios:
        tid = str(sc.get("dynamic_trace_id") or "custom_dynamic_trace")
        trace_meta.setdefault(tid, {"trace_id": tid, "phase_ids": [], "phase_count": 0, "task": sc.get("task")})
        trace_meta[tid]["phase_ids"].append(sc.get("scenario_id"))
        trace_meta[tid]["phase_count"] += 1
    return scenarios, trace_meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    set_progress_enabled(not args.no_progress)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Make project modules importable for mediator and utility subprocesses.
    project_root = Path(args.project_root).resolve()
    for candidate in [project_root, project_root / "mediator"]:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    if args.dynamic_contexts:
        scenarios, trace_meta = load_custom_dynamic_contexts(args.dynamic_contexts)
        progress_write(f"[dynamic] loaded custom dynamic contexts from {args.dynamic_contexts}: {len(scenarios)} phases")
    else:
        scenarios, trace_meta = built_in_dynamic_scenarios(args.trace_set)
        progress_write(f"[dynamic] built trace-set={args.trace_set}: {len(scenarios)} phases across {len(trace_meta)} traces")

    pipeline_root, rows, summary_by_trace = run_dynamic_generation(args, scenarios, trace_meta)
    progress_write(f"[dynamic] wrote pipeline summary to {pipeline_root / 'summary.csv'}")

    tasks_to_eval = sorted({r["task"] for r in rows if r.get("decision") == "select_pipeline"})
    args.methods_to_evaluate = sorted({r["method_id"] for r in rows if r.get("decision") == "select_pipeline"})
    skipped_for_utility = [r for r in rows if r.get("decision") != "select_pipeline"]
    utility_pipeline_root, selected_utility_rows = prepare_selected_only_utility_root(pipeline_root, rows)
    progress_write(
        f"[dynamic] selected outputs cover utility tasks: {tasks_to_eval if tasks_to_eval else 'none'}; "
        f"utility rows={len(selected_utility_rows)}, skipped non-output decisions={len(skipped_for_utility)}"
    )
    utility_out = out_root / "utility_eval"
    utility_result = run_utility_evaluator(args, utility_pipeline_root, utility_out, tasks_to_eval)
    utility_result["selected_only_pipeline_root"] = str(utility_pipeline_root)
    utility_result["selected_output_rows"] = len(selected_utility_rows)
    utility_result["skipped_non_output_decision_rows"] = len(skipped_for_utility)
    utility_result["skipped_non_output_decisions"] = [
        {
            "scenario_id": r.get("scenario_id"),
            "method_id": r.get("method_id"),
            "decision": r.get("decision"),
            "context_family": r.get("context_family"),
        }
        for r in skipped_for_utility
    ]
    write_json(utility_result, out_root / "utility_eval_result.json")

    examples_manifest = {"status": "skipped", "reason": "utility not run"}
    if utility_result.get("status") == "ok":
        examples_manifest = collect_output_examples(
            utility_out=utility_out,
            example_root=out_root / "output_examples",
            limit_per_result=args.example_copy_limit,
        )

    run_summary = {
        "schema_version": "smartpriv_dynamic_context_experiment_run_v1",
        "out_dir": str(out_root),
        "pipeline_root": str(pipeline_root),
        "dynamic_contexts_json": str(out_root / "dynamic_context_scenarios.json"),
        "summary_json": str(pipeline_root / "summary.json"),
        "summary_csv": str(pipeline_root / "summary.csv"),
        "summary_by_trace_json": str(pipeline_root / "summary_by_trace.json"),
        "utility_eval_result": utility_result,
        "output_examples_manifest": examples_manifest,
        "row_count": len(rows),
        "selected_count": sum(1 for r in rows if r.get("decision") == "select_pipeline"),
        "no_compromise_count": sum(1 for r in rows if r.get("decision") == "no_compromise"),
        "methods": sorted({r.get("method_id") for r in rows if r.get("method_id")}),
        "method_decision_counts": {
            m: {
                "select": sum(1 for r in rows if r.get("method_id") == m and r.get("decision") == "select_pipeline"),
                "no_compromise": sum(1 for r in rows if r.get("method_id") == m and r.get("decision") == "no_compromise"),
                "review": sum(1 for r in rows if r.get("method_id") == m and r.get("decision") in {"consent_or_review_required", "review"}),
                "error": sum(1 for r in rows if r.get("method_id") == m and (r.get("decision") == "error" or r.get("error"))),
                "hard_violation_select": sum(1 for r in rows if r.get("method_id") == m and r.get("context_family") == "hard_violation" and r.get("decision") == "select_pipeline"),
            }
            for m in sorted({r.get("method_id") for r in rows if r.get("method_id")})
        },
        "utility_selected_only_pipeline_root": utility_result.get("selected_only_pipeline_root"),
        "utility_selected_output_rows": utility_result.get("selected_output_rows"),
        "utility_skipped_non_output_decision_rows": utility_result.get("skipped_non_output_decision_rows"),
        "utility_skipped_non_output_decisions": utility_result.get("skipped_non_output_decisions", []),
        "trace_count": len(summary_by_trace),
        "traces": summary_by_trace,
    }
    write_json(run_summary, out_root / "dynamic_context_run_summary.json")
    progress_write(f"[dynamic] complete. run summary={out_root / 'dynamic_context_run_summary.json'}")

    print(json.dumps({
        "status": "ok" if utility_result.get("status") in {"ok", "skipped"} else "error",
        "out_dir": str(out_root),
        "pipeline_summary": str(pipeline_root / "summary.csv"),
        "summary_by_trace": str(pipeline_root / "summary_by_trace.json"),
        "utility_eval": utility_result,
        "output_examples_manifest": str(out_root / "output_examples" / "output_examples_manifest.json") if utility_result.get("status") == "ok" else None,
    }, indent=2))
    return 0 if utility_result.get("status") in {"ok", "skipped"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
