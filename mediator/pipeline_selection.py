#!/usr/bin/env python3
"""
pipeline_selection.py

Least-revealing feasible pipeline selector for SmartPriv/Prism.

This component runs after:
  1. candidate pipeline generation;
  2. contextual-integrity evaluation;
  3. optional empirical privacy probes.

It does not generate pipelines and does not call an LLM. It consumes structured
outputs from the earlier stages and chooses the least-revealing feasible pipeline:

  F = {p in P | U(p,A) >= tau_U and CI(f_p,C) acceptable}

  p* = argmin_{p in F} R_C(D(p))

Where D(p) may be metadata-only or the conservative max(metadata, probe).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


RISK_ORDER: Dict[str, int] = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "unknown": 4,
}
RISK_LEVELS = {v: k for k, v in RISK_ORDER.items()}

RESIDUAL_ATTRIBUTES: List[str] = [
    "identity",
    "face",
    "body_shape",
    "clothing",
    "gait",
    "speech_content",
    "speaker_identity",
    "activity",
    "location",
    "trajectory",
    "co_presence",
    "visible_text",
    "aggregate_presence",
]


DEFAULT_WEIGHTS: Dict[str, float] = {
    "identity": 5.0,
    "face": 4.0,
    "body_shape": 2.0,
    "clothing": 2.0,
    "gait": 3.0,
    "speech_content": 5.0,
    "speaker_identity": 5.0,
    "activity": 2.0,
    "location": 2.0,
    "trajectory": 4.0,
    "co_presence": 3.0,
    "visible_text": 4.0,
    "aggregate_presence": 1.0,
}


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def normalize_risk(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (int, float)):
        if value <= 0:
            return "none"
        if value <= 1:
            return "low"
        if value <= 2:
            return "medium"
        if value <= 3:
            return "high"
        return "unknown"
    s = str(value).strip().lower()
    if s in RISK_ORDER:
        return s
    if "unknown" in s:
        return "unknown"
    if "high" in s:
        return "high"
    if "medium" in s:
        return "medium"
    if "low" in s:
        return "low"
    if "none" in s or "removed" in s or "absent" in s:
        return "none"
    return "unknown"


def init_residual(default: str = "none") -> Dict[str, str]:
    return {a: normalize_risk(default) for a in RESIDUAL_ATTRIBUTES}


def max_risk(a: Any, b: Any) -> str:
    a_s = normalize_risk(a)
    b_s = normalize_risk(b)
    return RISK_LEVELS[max(RISK_ORDER[a_s], RISK_ORDER[b_s])]


def residual_vector(raw: Optional[Dict[str, Any]]) -> Dict[str, str]:
    out = init_residual("none")
    for a in RESIDUAL_ATTRIBUTES:
        if raw and a in raw:
            out[a] = normalize_risk(raw[a])
    return out


def combine_residuals(metadata: Dict[str, Any], probe: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    meta = residual_vector(metadata)
    if not probe:
        return meta
    p = residual_vector(probe)
    return {a: max_risk(meta[a], p[a]) for a in RESIDUAL_ATTRIBUTES}


def request_purposes(request: Dict[str, Any]) -> List[str]:
    ctx = request.get("ci_context", {}) or {}
    vals = ctx.get("purpose", [])
    if isinstance(vals, str):
        return [vals]
    return [str(v) for v in vals]


def request_contexts(request: Dict[str, Any]) -> List[str]:
    ctx = request.get("ci_context", {}) or {}
    vals = ctx.get("context", [])
    if isinstance(vals, str):
        return [vals]
    return [str(v) for v in vals]


def context_specific_weights(request: Dict[str, Any], selection_config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Return context-specific residual-risk weights.

    This is intentionally transparent and configurable. The defaults are not a
    universal privacy ordering; they only instantiate R_C for a given request.
    """
    weights = dict(DEFAULT_WEIGHTS)
    purposes = set(request_purposes(request))
    contexts = set(request_contexts(request))
    social = set((request.get("ci_context", {}) or {}).get("social_context_tags", []) or [])

    # Low-stakes automation: raw identity/speech/trajectory are heavily penalized;
    # aggregate presence is expected and lower-weighted.
    if purposes & {"energy_management", "comfort_automation", "occupancy_estimation"}:
        weights.update({
            "identity": 7.0,
            "face": 6.0,
            "speech_content": 7.0,
            "speaker_identity": 7.0,
            "trajectory": 6.0,
            "visible_text": 5.0,
            "aggregate_presence": 0.5,
            "activity": 1.5,
            "location": 1.5,
        })

    # Safety/fall detection may need activity/body/pose, but identity/speech/text
    # remain high-risk unless explicitly requested.
    if purposes & {"safety_monitoring", "fall_detection", "elder_care"}:
        weights.update({
            "identity": 6.0,
            "face": 5.0,
            "speech_content": 6.0,
            "speaker_identity": 6.0,
            "visible_text": 5.0,
            "activity": 1.5,
            "gait": 2.5,
            "body_shape": 2.0,
        })

    # Security use may legitimately need event evidence, but routine trajectories
    # and non-security raw identity still remain costly.
    if purposes & {"security_monitoring", "intrusion_detection", "incident_investigation"}:
        weights.update({
            "identity": 4.0,
            "face": 3.5,
            "activity": 1.5,
            "location": 1.5,
            "trajectory": 4.0,
            "speech_content": 6.0,
            "visible_text": 4.0,
        })

    # Contexts with guests/bystanders/weak preference channels raise the cost of
    # identity, speech, trajectory, and co-presence.
    if social & {"guests_present", "bystanders_present", "weak_preference_channel"}:
        for a in ["identity", "face", "speech_content", "speaker_identity", "trajectory", "co_presence", "visible_text"]:
            weights[a] = weights.get(a, DEFAULT_WEIGHTS[a]) + 1.0

    # High-privacy contexts.
    if contexts & {"home", "private_home", "short_term_rental", "hospital_or_clinic", "school_or_classroom"}:
        for a in ["identity", "speech_content", "speaker_identity", "visible_text"]:
            weights[a] = weights.get(a, DEFAULT_WEIGHTS[a]) + 1.0

    # User override.
    if selection_config:
        for k, v in (selection_config.get("weights", {}) or {}).items():
            if k in weights:
                weights[k] = float(v)

    return weights


def residual_risk_score(residual: Dict[str, Any], weights: Dict[str, float]) -> float:
    vec = residual_vector(residual)
    return sum(weights.get(a, 1.0) * RISK_ORDER[vec[a]] for a in RESIDUAL_ATTRIBUTES)


def lexicographic_key(residual: Dict[str, Any], priority_order: Optional[List[str]] = None) -> Tuple[int, ...]:
    order = priority_order or [
        "identity",
        "speech_content",
        "speaker_identity",
        "face",
        "visible_text",
        "trajectory",
        "gait",
        "body_shape",
        "clothing",
        "co_presence",
        "activity",
        "location",
        "aggregate_presence",
    ]
    vec = residual_vector(residual)
    return tuple(RISK_ORDER[vec[a]] for a in order if a in vec)


def accepted_cap_priority(request: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for cap in (request.get("utility_contract", {}) or {}).get("accepted_output_caps", []) or []:
        cid = cap.get("cap_id")
        if cid is not None:
            out[str(cid)] = int(cap.get("priority", 999))
    return out


def pipeline_cost(candidate: Dict[str, Any]) -> float:
    if "implementation_cost" in candidate:
        try:
            return float(candidate["implementation_cost"])
        except Exception:
            pass
    ops = candidate.get("operators", []) or []
    # Route/publish and source are cheap bookkeeping; transformations cost more.
    cost = 0.0
    for op in ops:
        oid = str(op.get("operator", ""))
        if oid in {"op.source", "op.route_publish"}:
            cost += 0.1
        elif oid in {"op.schema_adapter", "op.drop_discard"}:
            cost += 0.25
        else:
            cost += 1.0
    return cost


def utility_margin(candidate: Dict[str, Any], request: Dict[str, Any]) -> float:
    """Best-effort utility margin proxy.

    If the system later reports quantitative margins, this function will use them.
    Until then, lower accepted-output priority and exact requested capability serve
    as symbolic proxies.
    """
    if "utility_margin" in candidate:
        try:
            return float(candidate["utility_margin"])
        except Exception:
            pass

    requested = (request.get("utility_contract", {}) or {}).get("requested_capability")
    caps = set(candidate.get("utility_capabilities", []) or [])
    margin = 0.0
    if requested and requested in caps:
        margin += 1.0
    if candidate.get("quality_status") == "validated":
        margin += 1.0
    elif candidate.get("quality_status") == "requires_runtime_or_benchmark_validation":
        margin += 0.25
    return margin


def latency_ms(candidate: Dict[str, Any]) -> float:
    for key in ["estimated_latency_ms", "latency_ms", "runtime_latency_ms"]:
        if key in candidate:
            try:
                return float(candidate[key])
            except Exception:
                pass
    # Unknown latency. Use operator count as a mild proxy.
    return 100.0 * pipeline_cost(candidate)


def residual_satisfies_request(residual: Dict[str, Any], request: Dict[str, Any]) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    rc = request.get("residual_disclosure_constraints", {}) or {}
    max_allowed = rc.get("max_allowed", {}) or {}
    hard_forbidden = set(rc.get("hard_forbidden_attributes", []) or [])
    allowed_high = set(rc.get("allowed_high_attributes", []) or [])

    vec = residual_vector(residual)
    for attr in RESIDUAL_ATTRIBUTES:
        cand = vec[attr]
        max_l = normalize_risk(max_allowed.get(attr, "unknown"))

        if attr in allowed_high:
            continue

        if attr in hard_forbidden and RISK_ORDER[cand] > RISK_ORDER[max_l]:
            failures.append(f"hard-forbidden residual {attr}={cand} exceeds max {max_l}")
        elif RISK_ORDER[cand] > RISK_ORDER[max_l]:
            failures.append(f"residual {attr}={cand} exceeds max {max_l}")

    return not failures, failures


def ci_evaluations_by_pipeline(ci_evaluation_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for ev in ci_evaluation_result.get("evaluations", []) or []:
        pid = ev.get("pipeline_id")
        if pid:
            out[str(pid)] = ev
    return out


def candidates_by_pipeline(candidate_generation_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for c in candidate_generation_result.get("candidates", []) or []:
        pid = c.get("pipeline_id")
        if pid:
            out[str(pid)] = c
    return out


def probe_residuals_by_pipeline(probe_stage_result: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Support both current selected-pipeline probe output and future all-candidate reports."""
    out: Dict[str, Dict[str, Any]] = {}
    if not probe_stage_result:
        return out

    # Current privacy_probes mediator output.
    pid = probe_stage_result.get("selected_pipeline_id")
    report = probe_stage_result.get("probe_report") or {}
    if pid and report.get("combined_residual"):
        out[str(pid)] = report.get("combined_residual")
    elif pid and report.get("probe_residual"):
        out[str(pid)] = report.get("probe_residual")

    # Future format: candidate_probe_reports list.
    for item in probe_stage_result.get("candidate_probe_reports", []) or []:
        ipid = item.get("pipeline_id") or item.get("selected_pipeline_id")
        ireport = item.get("probe_report") or item
        if ipid and ireport.get("combined_residual"):
            out[str(ipid)] = ireport.get("combined_residual")
        elif ipid and ireport.get("probe_residual"):
            out[str(ipid)] = ireport.get("probe_residual")

    return out


def is_ci_feasible(ev: Optional[Dict[str, Any]]) -> bool:
    if not ev:
        return False
    decision = ev.get("ci_decision") or {}
    if decision.get("feasible") is True:
        return True
    # Be compatible with older outputs.
    d = str(decision.get("decision", "")).lower()
    return d in {"accept_hard_constraints_only", "accept_llm_norm", "acceptable"}


def select_pipeline(
    candidate_generation_result: Dict[str, Any],
    ci_evaluation_result: Dict[str, Any],
    request: Dict[str, Any],
    probe_stage_result: Optional[Dict[str, Any]] = None,
    selection_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Select least-revealing feasible pipeline after CI filtering.

    The selector is conservative:
      - CI must mark the candidate feasible.
      - final residual must satisfy request residual constraints.
      - probe residual overrides/combines with metadata residual when available.
    """
    selection_config = selection_config or {}
    weights = context_specific_weights(request, selection_config)
    cap_prio = accepted_cap_priority(request)
    cands = candidates_by_pipeline(candidate_generation_result)
    evals = ci_evaluations_by_pipeline(ci_evaluation_result)
    probe_by_id = probe_residuals_by_pipeline(probe_stage_result)

    feasible: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for pid, cand in cands.items():
        ev = evals.get(pid)
        meta_res = cand.get("residual_disclosure") or {}
        # The probe stage may already contain combined residuals; still passing it
        # through combine is safe and conservative.
        final_res = combine_residuals(meta_res, probe_by_id.get(pid)) if pid in probe_by_id else residual_vector(meta_res)

        ci_ok = is_ci_feasible(ev)
        residual_ok, residual_failures = residual_satisfies_request(final_res, request)

        record = {
            "pipeline_id": pid,
            "matched_output_cap": cand.get("matched_output_cap"),
            "ci_feasible": ci_ok,
            "residual_feasible": residual_ok,
            "residual_failures": residual_failures,
            "metadata_residual": residual_vector(meta_res),
            "probe_or_combined_residual_available": pid in probe_by_id,
            "final_residual": final_res,
            "risk_score": residual_risk_score(final_res, weights),
            "lexicographic_key": list(lexicographic_key(final_res, selection_config.get("lexicographic_order"))),
            "utility_margin": utility_margin(cand, request),
            "latency_ms": latency_ms(cand),
            "implementation_cost": pipeline_cost(cand),
            "accepted_cap_priority": cap_prio.get(str(cand.get("matched_output_cap")), 999),
            "operators": [op.get("operator") for op in cand.get("operators", []) or []],
            "ci_decision": (ev or {}).get("ci_decision"),
        }

        if ci_ok and residual_ok:
            feasible.append(record)
        else:
            rejected.append(record)

    # Ranking method.
    method = selection_config.get("method", "weighted")
    if method == "lexicographic":
        feasible.sort(key=lambda r: (
            tuple(r["lexicographic_key"]),
            -r["utility_margin"],
            r["latency_ms"],
            r["implementation_cost"],
            r["accepted_cap_priority"],
        ))
    elif method == "pareto_then_weighted":
        pareto = pareto_front(feasible)
        pareto.sort(key=lambda r: (
            r["risk_score"],
            -r["utility_margin"],
            r["latency_ms"],
            r["implementation_cost"],
            r["accepted_cap_priority"],
        ))
        feasible = pareto + [r for r in feasible if r not in pareto]
    else:
        feasible.sort(key=lambda r: (
            r["risk_score"],
            tuple(r["lexicographic_key"]),
            -r["utility_margin"],
            r["latency_ms"],
            r["implementation_cost"],
            r["accepted_cap_priority"],
        ))

    if feasible:
        selected = feasible[0]
        decision = {
            "decision": "select_pipeline",
            "selected_pipeline_id": selected["pipeline_id"],
            "selected_output_cap": selected["matched_output_cap"],
            "reason": "Selected least-revealing feasible pipeline after utility, CI, residual, and optional probe evaluation.",
        }
    else:
        # Diagnose why no pipeline survived.
        any_ci_feasible = any(r["ci_feasible"] for r in rejected)
        if not cands:
            d = "no_candidates"
            reason = "No generated candidate pipelines were available."
        elif not any_ci_feasible:
            d = "no_compromise"
            reason = "No candidate pipeline passed contextual-integrity evaluation."
        else:
            d = "no_compromise"
            reason = "Some candidates passed CI evaluation, but none satisfied final residual-disclosure constraints."
        decision = {
            "decision": d,
            "selected_pipeline_id": None,
            "selected_output_cap": None,
            "reason": reason,
        }

    return {
        "schema_version": "smartpriv_pipeline_selection_output_v1",
        "request_id": candidate_generation_result.get("request_id") or request.get("request_identity", {}).get("request_id"),
        "scenario_id": candidate_generation_result.get("scenario_id") or request.get("request_identity", {}).get("scenario_id"),
        "selector": {
            "method": method,
            "weights": weights,
            "probe_stage_used": bool(probe_by_id),
            "num_candidates": len(cands),
            "num_ci_evaluations": len(evals),
            "num_probe_residuals": len(probe_by_id),
            "num_feasible": len(feasible),
            "num_rejected": len(rejected),
            "tie_breakers": [
                "risk_score",
                "lexicographic_key",
                "utility_margin_desc",
                "latency_ms",
                "implementation_cost",
                "accepted_output_cap_priority",
            ],
        },
        "decision": decision,
        "selected": feasible[0] if feasible else None,
        "feasible_ranked": feasible,
        "rejected": rejected[:100],
    }


def dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Return True if a's final residual is no worse than b and better somewhere."""
    av = residual_vector(a.get("final_residual", {}))
    bv = residual_vector(b.get("final_residual", {}))
    no_worse = all(RISK_ORDER[av[x]] <= RISK_ORDER[bv[x]] for x in RESIDUAL_ATTRIBUTES)
    better = any(RISK_ORDER[av[x]] < RISK_ORDER[bv[x]] for x in RESIDUAL_ATTRIBUTES)
    return no_worse and better


def pareto_front(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    front: List[Dict[str, Any]] = []
    for r in records:
        if not any(dominates(other, r) for other in records if other is not r):
            front.append(r)
    return front


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Select least-revealing feasible pipeline after CI evaluation.")
    p.add_argument("--request", required=True)
    p.add_argument("--candidates", required=True, help="candidate_pipelines.json")
    p.add_argument("--ci-evaluation", required=True, help="ci_evaluation.json")
    p.add_argument("--probe-stage", help="Optional privacy_probe_stage_result.json")
    p.add_argument("--selection-config", help="Optional JSON config with weights/method")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    request = load_json(args.request)
    candidates = load_json(args.candidates)
    ci_eval = load_json(args.ci_evaluation)
    probe = load_json(args.probe_stage) if args.probe_stage else None
    config = load_json(args.selection_config) if args.selection_config else None

    result = select_pipeline(
        candidate_generation_result=candidates,
        ci_evaluation_result=ci_eval,
        request=request,
        probe_stage_result=probe,
        selection_config=config,
    )
    write_json(result, args.out)

    print(json.dumps({
        "decision": result["decision"],
        "num_feasible": result["selector"]["num_feasible"],
        "probe_stage_used": result["selector"]["probe_stage_used"],
        "out": args.out,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
