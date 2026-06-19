#!/usr/bin/env python3
"""
contextual_integrity_evaluator.py

Layered contextual-integrity evaluator for symbolic preprocessing pipelines.

Inputs:
  --constraints  norms/ci_constraints.json
  --request      structured application request JSON
  --candidates   output JSON from generate_pipeline_candidates.py
  --environment optional JSON with structured context or raw description
  --sensor-stream optional JSON describing the concrete sensor stream/source
  --out          optional output JSON

The evaluator:
  1. constructs a residual CI information flow for each candidate pipeline;
  2. applies hard policy/rule constraints from ci_constraints.json;
  3. optionally calls an LLM for structured norm judgment when hard constraints pass;
  4. selects the least-revealing candidate that passes the selected CI mode.

Norm templates are intentionally ignored. Rules whose provenance source_type is
created_template are skipped, and PREFER-only rules are not used as hard decisions.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": 4}
RESIDUAL_ATTRS = [
    "identity", "face", "body_shape", "clothing", "gait", "speech_content",
    "speaker_identity", "activity", "location", "trajectory", "co_presence",
    "visible_text", "aggregate_presence",
]

ATTRIBUTE_TO_INFORMATION_TYPE = {
    "identity": "identity",
    "face": "face_detected",
    "body_shape": "body_shape",
    "clothing": "clothing",
    "gait": "gait",
    "speech_content": "speech_transcribed",
    "speaker_identity": "speaker_identity",
    "activity": "activity_label",
    "location": "current_location",
    "trajectory": "trajectory",
    "co_presence": "presence",
    "visible_text": "screen_content_detected",
    "aggregate_presence": "occupancy_count",
}


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def as_str_list(value: Any) -> List[str]:
    out: List[str] = []
    for item in as_list(value):
        if item is None:
            continue
        if isinstance(item, list):
            out.extend(as_str_list(item))
        elif isinstance(item, dict):
            out.extend(flatten_values(item))
        else:
            out.append(str(item))
    return out


def normalize_risk(value: Any) -> str:
    if value is None:
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
    if "none" in s or "absent" in s or "removed" in s:
        return "none"
    return "unknown"


def risk_score(residual: Dict[str, Any]) -> int:
    weights = {
        "identity": 3, "speech_content": 3, "speaker_identity": 3,
        "face": 2, "trajectory": 2, "visible_text": 2,
    }
    return sum(weights.get(k, 1) * RISK_ORDER.get(normalize_risk(v), 4)
               for k, v in residual.items())


def deep_get(obj: Dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def flatten_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        out: List[str] = []
        for v in value.values():
            out.extend(flatten_values(v))
        return out
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(flatten_values(item))
        return out
    return [str(value)]


def value_set(flow: Dict[str, Any], dotted: str) -> Set[str]:
    return set(flatten_values(deep_get(flow, dotted)))


def any_present(flow: Dict[str, Any], dotted: str, required: Iterable[str]) -> bool:
    return bool(value_set(flow, dotted) & set(map(str, required)))


def all_present(flow: Dict[str, Any], dotted: str, required: Iterable[str]) -> bool:
    return set(map(str, required)).issubset(value_set(flow, dotted))


def detect_content_type_from_cap(cap: Dict[str, Any]) -> List[str]:
    t = cap.get("semantic_type") or cap.get("media_type") or ""
    schema = cap.get("schema") or ""
    out: Set[str] = set()
    if t.startswith("video/") or "video" in t or "clip" in schema:
        out.add("video_content")
    if t.startswith("image/") or "image" in t or "frame" in schema:
        out.add("image_content")
    if t.startswith("audio/") or "audio" in t:
        out.add("audio_content")
    if "transcript" in schema or "intent" in schema:
        out.add("textual_content")
    if not out and t:
        out.add("output_data")
    return sorted(out)


def infer_source_metadata(candidate: Dict[str, Any], request: Dict[str, Any], sensor_stream: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    req_src = request.get("source_requirements", {}) or {}
    final_cap = candidate.get("final_output_cap", {}) or {}
    sensing_device: List[str] = []
    content_type: List[str] = []
    modality: List[str] = []
    if sensor_stream:
        sensing_device += as_str_list(sensor_stream.get("sensingDevice") or sensor_stream.get("sensing_device"))
        content_type += as_str_list(sensor_stream.get("contentType") or sensor_stream.get("content_type"))
        modality += as_str_list(sensor_stream.get("sensingModality") or sensor_stream.get("sensing_modality"))
    if not sensing_device:
        sensing_device += as_str_list(req_src.get("allowed_sensing_devices"))
    if not content_type:
        content_type += as_str_list(req_src.get("allowed_content_types"))
    if not modality:
        modality += as_str_list(req_src.get("allowed_modalities"))
    content_type += detect_content_type_from_cap(final_cap)
    return {
        "sensingDevice": sorted(set(sensing_device)),
        "contentType": sorted(set(content_type)),
        "sensingModality": sorted(set(modality)),
    }


def residual_attributes_to_information_types(residual: Dict[str, Any]) -> Dict[str, List[str]]:
    primitive: Set[str] = set()
    interpreted: Set[str] = set()
    inferred: Set[str] = set()
    for attr, level in residual.items():
        if RISK_ORDER.get(normalize_risk(level), 4) <= 0:
            continue
        term = ATTRIBUTE_TO_INFORMATION_TYPE.get(attr, attr)
        if attr in {"identity", "trajectory", "location", "co_presence"}:
            inferred.add(term)
        elif attr == "speech_content":
            interpreted.add("speech_transcribed")
            primitive.add("speech_audio")
        elif attr == "aggregate_presence":
            interpreted.add("occupancy_count")
        else:
            interpreted.add(term)
    return {
        "sensorPrimitive": sorted(primitive),
        "interpretedObservation": sorted(interpreted),
        "inferredInformationType": sorted(inferred),
    }


def merge_information_types(candidate: Dict[str, Any]) -> Dict[str, List[str]]:
    ci_terms = candidate.get("ci_terms", {}) or {}
    result = {"sensorPrimitive": set(), "interpretedObservation": set(), "inferredInformationType": set()}
    mapping = {
        "informationType.sensorPrimitive": "sensorPrimitive",
        "informationType.interpretedObservation": "interpretedObservation",
        "informationType.inferredInformationType": "inferredInformationType",
    }
    for src, dst in mapping.items():
        result[dst].update(as_str_list(ci_terms.get(src)))
    for k, vals in residual_attributes_to_information_types(candidate.get("residual_disclosure", {}) or {}).items():
        result[k].update(vals)
    return {k: sorted(v) for k, v in result.items()}


def construct_ci_flow(candidate: Dict[str, Any], request: Dict[str, Any], environment: Optional[Dict[str, Any]] = None, sensor_stream: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    env = environment or {}
    req_ctx = request.get("ci_context", {}) or {}
    ci_terms = candidate.get("ci_terms", {}) or {}

    def choose(field: str) -> List[str]:
        vals: List[str] = []
        vals.extend(as_str_list(env.get(field)))
        vals.extend(as_str_list(req_ctx.get(field)))
        if sensor_stream:
            vals.extend(as_str_list(sensor_stream.get(field)))
        return sorted(set(v for v in vals if v))

    transmission = set(as_str_list(req_ctx.get("transmissionPrinciple_assumed")))
    transmission.update(as_str_list(ci_terms.get("transmissionPrinciple")))
    transmission.update(as_str_list(env.get("transmissionPrinciple")))
    if sensor_stream:
        transmission.update(as_str_list(sensor_stream.get("transmissionPrinciple")))

    information_type = merge_information_types(candidate)
    attributes = sorted(set(information_type["sensorPrimitive"] + information_type["interpretedObservation"] + information_type["inferredInformationType"]))

    metadata: Dict[str, Any] = {}
    for source in [req_ctx.get("metadata"), env.get("metadata"), sensor_stream.get("metadata") if sensor_stream else None]:
        if isinstance(source, dict):
            metadata.update(source)
        elif isinstance(source, list):
            metadata.setdefault("tags", []).extend(source)

    social_tags = set(as_str_list(req_ctx.get("social_context_tags")))
    social_tags.update(as_str_list(env.get("social_context_tags")))

    return {
        "pipeline_id": candidate.get("pipeline_id"),
        "pipelineStage": as_str_list(ci_terms.get("pipelineStage")) or ["output_to_application"],
        "context": choose("context"),
        "space": choose("space"),
        "sender": choose("sender"),
        "subject": choose("subject"),
        "recipient": choose("recipient"),
        "purpose": choose("purpose"),
        "transmissionPrinciple": sorted(transmission),
        "attribute": attributes,
        "informationType": information_type,
        "sensingDataMetadata": infer_source_metadata(candidate, request, sensor_stream),
        "metadata": metadata,
        "social_context_tags": sorted(social_tags),
        "residual_disclosure": candidate.get("residual_disclosure", {}) or {},
        "final_output_cap": candidate.get("final_output_cap", {}) or {},
        "transforms": candidate.get("transforms", []) or [],
        "tags": sorted(set(as_str_list(candidate.get("tags")))),
    }


def match_condition(flow: Dict[str, Any], field: str, condition: Any) -> bool:
    if isinstance(condition, dict):
        if "anyOf" in condition:
            return any_present(flow, field, as_str_list(condition.get("anyOf")))
        if "allOf" in condition:
            return all_present(flow, field, as_str_list(condition.get("allOf")))
        if "noneOf" in condition:
            return not any_present(flow, field, as_str_list(condition.get("noneOf")))
        return all(match_condition(flow, f"{field}.{sub}", subcond) for sub, subcond in condition.items())
    return any_present(flow, field, as_str_list(condition))


def rule_matches(flow: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    return all(match_condition(flow, field, cond) for field, cond in (rule.get("match", {}) or {}).items())


def unless_satisfied(flow: Dict[str, Any], unless: Any) -> bool:
    if not isinstance(unless, dict) or not unless:
        return False
    for field, expected in unless.items():
        if isinstance(expected, list):
            if not any_present(flow, field, [str(v) for v in expected]):
                return False
        elif isinstance(expected, bool):
            if bool(deep_get(flow, field)) != expected:
                return False
        else:
            if not any_present(flow, field, [str(expected)]):
                return False
    return True


def stages_apply(flow: Dict[str, Any], effect: Dict[str, Any]) -> bool:
    stages = set(as_str_list(effect.get("appliesToPipelineStages"))) or {"raw_input", "intermediate_representation", "output_to_application", "retained_artifact"}
    return bool(set(as_str_list(flow.get("pipelineStage"))) & stages)


def check_required(flow: Dict[str, Any], required: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    for field, vals in (required or {}).items():
        vals_list = as_str_list(vals)
        if vals_list and not all_present(flow, field, vals_list):
            missing.append(f"{field} missing one or more of {vals_list}")
    return missing


def check_forbidden(flow: Dict[str, Any], forbidden: Dict[str, Any]) -> List[str]:
    present: List[str] = []
    for field, vals in (forbidden or {}).items():
        overlap = value_set(flow, field) & set(as_str_list(vals))
        if overlap:
            present.append(f"{field} has forbidden {sorted(overlap)}")
    return present


def is_template_rule(rule: Dict[str, Any]) -> bool:
    prov = rule.get("provenance", {}) or {}
    return prov.get("source_type") == "created_template" or str(rule.get("id", "")).startswith("template_")


def apply_classification_rules(flow: Dict[str, Any], rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    tags = set(as_str_list(flow.get("tags")))
    for rule in rules:
        effect = rule.get("effect", {}) or {}
        if is_template_rule(rule) or effect.get("action") != "CLASSIFY_AS":
            continue
        if rule_matches(flow, rule):
            new_tags = as_str_list(effect.get("tags"))
            tags.update(new_tags)
            out.append({"rule_id": rule.get("id"), "action": "CLASSIFY_AS", "status": "tagged", "tags_added": new_tags, "explanation": effect.get("explanation", "")})
    flow["tags"] = sorted(tags)
    return out


def evaluate_hard_constraints(flow: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
    rules = constraints.get("rules", []) or []
    rule_evals: List[Dict[str, Any]] = []
    rule_evals.extend(apply_classification_rules(flow, rules))
    denied: List[Dict[str, Any]] = []
    missing_conditions: List[Dict[str, Any]] = []
    missing_transformations: List[Dict[str, Any]] = []
    allow_if_failures: List[Dict[str, Any]] = []
    obligations_satisfied: List[Dict[str, Any]] = []

    for rule in rules:
        if is_template_rule(rule):
            continue
        effect = rule.get("effect", {}) or {}
        action = effect.get("action")
        if action in {"CLASSIFY_AS", "PREFER"}:
            continue
        if not rule_matches(flow, rule):
            continue
        if unless_satisfied(flow, effect.get("unless") or rule.get("unless")):
            rule_evals.append({"rule_id": rule.get("id"), "action": action, "status": "skipped_by_unless", "explanation": effect.get("explanation", "")})
            continue
        if not stages_apply(flow, effect):
            continue

        rec = {
            "rule_id": rule.get("id"),
            "ruleType": rule.get("ruleType"),
            "priority": rule.get("priority"),
            "action": action,
            "explanation": effect.get("explanation", ""),
            "natural_language_description": (rule.get("provenance", {}) or {}).get("natural_language_description", ""),
            "source_name": (rule.get("provenance", {}) or {}).get("source_name", ""),
        }
        if action == "DENY_FLOW":
            denied.append(rec)
            rule_evals.append({**rec, "status": "denied"})
        elif action == "REQUIRE_CONDITION":
            missing = check_required(flow, effect.get("requiredConditions", {}) or {})
            forbidden = check_forbidden(flow, effect.get("forbiddenConditions", {}) or {})
            if missing or forbidden:
                rec.update({"missing_required": missing, "forbidden_present": forbidden})
                missing_conditions.append(rec)
                rule_evals.append({**rec, "status": "condition_unsatisfied"})
            else:
                obligations_satisfied.append(rec)
                rule_evals.append({**rec, "status": "condition_satisfied"})
        elif action == "REQUIRE_TRANSFORMATION":
            required = set(as_str_list(effect.get("requiredTransformations")))
            have = set(as_str_list(flow.get("transforms"))) | set(as_str_list(flow.get("transmissionPrinciple")))
            missing = sorted(required - have)
            if missing:
                rec.update({"missing_transformations": missing})
                missing_transformations.append(rec)
                rule_evals.append({**rec, "status": "transformation_missing"})
            else:
                obligations_satisfied.append(rec)
                rule_evals.append({**rec, "status": "transformation_satisfied"})
        elif action == "ALLOW_IF":
            missing = check_required(flow, effect.get("requiredConditions", {}) or {})
            forbidden = check_forbidden(flow, effect.get("forbiddenConditions", {}) or {})
            if missing or forbidden:
                rec.update({"missing_required": missing, "forbidden_present": forbidden})
                allow_if_failures.append(rec)
                rule_evals.append({**rec, "status": "allow_if_failed"})
            else:
                rule_evals.append({**rec, "status": "allow_if_satisfied"})
        else:
            rec.update({"status": "unknown_action"})
            missing_conditions.append(rec)
            rule_evals.append(rec)

    hard_pass = not (denied or missing_conditions or missing_transformations or allow_if_failures)
    return {
        "hard_pass": hard_pass,
        "denied_by": denied,
        "missing_conditions": missing_conditions,
        "missing_transformations": missing_transformations,
        "allow_if_failures": allow_if_failures,
        "obligations_satisfied": obligations_satisfied,
        "rule_evaluations": rule_evals,
    }



def summarize_hard_constraint_result(hard: Dict[str, Any]) -> Dict[str, Any]:
    """Compact rule-level diagnostics for one candidate's hard-CI outcome."""
    denied = hard.get("denied_by", []) or []
    missing_conditions = hard.get("missing_conditions", []) or []
    missing_transformations = hard.get("missing_transformations", []) or []
    allow_if_failures = hard.get("allow_if_failures", []) or []

    failed_rules: List[Dict[str, Any]] = []
    for category, rows in [
        ("denied_by", denied),
        ("missing_conditions", missing_conditions),
        ("missing_transformations", missing_transformations),
        ("allow_if_failures", allow_if_failures),
    ]:
        for r in rows:
            failed_rules.append({
                "category": category,
                "rule_id": r.get("rule_id"),
                "ruleType": r.get("ruleType"),
                "action": r.get("action"),
                "priority": r.get("priority"),
                "source_name": r.get("source_name"),
                "explanation": r.get("explanation"),
                "natural_language_description": r.get("natural_language_description"),
                "missing_required": r.get("missing_required", []),
                "missing_transformations": r.get("missing_transformations", []),
                "forbidden_present": r.get("forbidden_present", []),
            })

    counts_by_action: Dict[str, int] = {}
    counts_by_category: Dict[str, int] = {}
    rule_ids: List[str] = []
    for r in failed_rules:
        action = str(r.get("action") or "unknown")
        category = str(r.get("category") or "unknown")
        counts_by_action[action] = counts_by_action.get(action, 0) + 1
        counts_by_category[category] = counts_by_category.get(category, 0) + 1
        if r.get("rule_id"):
            rule_ids.append(str(r["rule_id"]))

    return {
        "hard_pass": bool(hard.get("hard_pass")),
        "failure_count": len(failed_rules),
        "counts_by_action": counts_by_action,
        "counts_by_category": counts_by_category,
        "failed_rule_ids": sorted(set(rule_ids)),
        "failed_rules": failed_rules,
        "obligations_satisfied_count": len(hard.get("obligations_satisfied", []) or []),
    }


def compact_candidate_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
    cap = candidate.get("final_output_cap", {}) or {}
    return {
        "pipeline_id": candidate.get("pipeline_id"),
        "matched_output_cap": candidate.get("matched_output_cap"),
        "matched_output_schema": candidate.get("matched_output_schema"),
        "final_output_type": cap.get("semantic_type") or cap.get("media_type"),
        "final_output_schema": cap.get("schema"),
        "operators": [op.get("operator") for op in candidate.get("operators", []) or []],
        "residual_score": candidate.get("residual_score", risk_score(candidate.get("residual_disclosure", {}) or {})),
    }


def build_no_compromise_diagnostics(evaluations: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Explain why no candidate passed CI and which rejected candidates were closest."""
    candidate_by_id = {str(c.get("pipeline_id")): c for c in candidates if c.get("pipeline_id")}
    diagnostics: Dict[str, Any] = {
        "candidate_count": len(candidates),
        "evaluation_count": len(evaluations),
        "ci_feasible_count": 0,
        "hard_rejection_count": 0,
        "llm_rejection_count": 0,
        "other_rejection_count": 0,
        "rule_failure_counts": {},
        "action_failure_counts": {},
        "category_failure_counts": {},
        "primary_reason": "No candidate pipeline passed the CI evaluator.",
        "closest_rejected_candidates": [],
    }

    rejected_records: List[Dict[str, Any]] = []
    for ev in evaluations:
        decision = ev.get("ci_decision") or {}
        if decision.get("feasible"):
            diagnostics["ci_feasible_count"] += 1
            continue
        d = str(decision.get("decision") or "unknown")
        if d == "reject_hard_constraint":
            diagnostics["hard_rejection_count"] += 1
        elif "llm" in d:
            diagnostics["llm_rejection_count"] += 1
        else:
            diagnostics["other_rejection_count"] += 1

        hard_summary = summarize_hard_constraint_result(ev.get("hard_constraint_result", {}) or {})
        for rid in hard_summary.get("failed_rule_ids", []):
            diagnostics["rule_failure_counts"][rid] = diagnostics["rule_failure_counts"].get(rid, 0) + 1
        for action, count in (hard_summary.get("counts_by_action") or {}).items():
            diagnostics["action_failure_counts"][action] = diagnostics["action_failure_counts"].get(action, 0) + count
        for cat, count in (hard_summary.get("counts_by_category") or {}).items():
            diagnostics["category_failure_counts"][cat] = diagnostics["category_failure_counts"].get(cat, 0) + count

        cand = candidate_by_id.get(str(ev.get("pipeline_id")), {})
        rejected_records.append({
            "pipeline_id": ev.get("pipeline_id"),
            "ci_decision": decision,
            "candidate": compact_candidate_summary(cand) if cand else {"pipeline_id": ev.get("pipeline_id")},
            "hard_failure_summary": hard_summary,
            "flow_context": {
                "context": (ev.get("flow") or {}).get("context"),
                "space": (ev.get("flow") or {}).get("space"),
                "purpose": (ev.get("flow") or {}).get("purpose"),
                "transmissionPrinciple": (ev.get("flow") or {}).get("transmissionPrinciple"),
                "pipelineStage": (ev.get("flow") or {}).get("pipelineStage"),
                "attribute": (ev.get("flow") or {}).get("attribute"),
            },
            "residual_score": ev.get("residual_score", 10**9),
        })

    if not candidates:
        diagnostics["primary_reason"] = "No candidate pipelines were generated."
    elif diagnostics["hard_rejection_count"] == len(evaluations) and evaluations:
        diagnostics["primary_reason"] = "All utility-compatible candidates failed hard contextual-integrity rules or required conditions/transformations."
    elif diagnostics["hard_rejection_count"] > 0:
        diagnostics["primary_reason"] = "No candidate passed CI; at least one failed hard contextual-integrity rules or required conditions/transformations."
    elif diagnostics["llm_rejection_count"] > 0:
        diagnostics["primary_reason"] = "Hard rules passed for some candidates, but no candidate passed LLM norm judgment."

    rejected_records.sort(key=lambda r: (r.get("residual_score", 10**9), len((r.get("hard_failure_summary") or {}).get("failed_rule_ids", []))))
    diagnostics["closest_rejected_candidates"] = rejected_records[:10]
    diagnostics["top_failed_rules"] = sorted(
        [{"rule_id": k, "count": v} for k, v in diagnostics["rule_failure_counts"].items()],
        key=lambda x: (-x["count"], x["rule_id"]),
    )[:20]
    return diagnostics


def make_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """Create a LangChain ChatOpenAI backend. Set os.environ['OPENAI_API_KEY'] before calling."""
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError("LangChain backend requires `pip install langchain-openai langchain-core`.") from exc
    return ChatOpenAI(model=model, temperature=temperature)


def strip_json_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"```$", "", s).strip()
    return s


def call_llm_norm_judgment(flow: Dict[str, Any], candidate: Dict[str, Any], request: Dict[str, Any], environment: Optional[Dict[str, Any]] = None, model: str = "gpt-4o-mini", temperature: float = 0.0) -> Dict[str, Any]:
    llm = make_llm(model=model, temperature=temperature)
    prompt = f"""
You are assisting a contextual-integrity privacy mediator. You do NOT select pipelines.
You only produce a structured norm judgment for one already-generated residual information flow.

Return JSON only with this schema:
{{
  "acceptability_label": "acceptable | acceptable_with_mitigations | uncertain | inappropriate",
  "confidence": 0.0,
  "relevant_ci_dimensions": {{"sender": [], "subject": [], "recipient": [], "attribute": [], "purpose": [], "transmission": [], "context": []}},
  "satisfied_expectations": [],
  "violated_expectations": [],
  "required_mitigations": [],
  "rationale": "brief explanation"
}}

Rules:
- Do not override hard constraints. Assume hard constraints have already passed.
- Be conservative for guests, bystanders, children, patients, workers, and weak preference channels.
- Prefer minimized semantic outputs over raw media when utility allows.
- Mark uncertain if social context is underspecified or if the judgment depends on consent/disclosure not present in the flow.
- Do not invent facts not present in the flow, request, or environment.

Application request:
{json.dumps(request, indent=2)}

Structured environment:
{json.dumps(environment or {}, indent=2)}

Residual CI flow:
{json.dumps(flow, indent=2)}

Candidate pipeline summary:
{json.dumps({k: candidate.get(k) for k in ["pipeline_id", "matched_output_cap", "final_output_cap", "residual_disclosure", "operators"]}, indent=2)}
""".strip()
    response = llm.invoke(prompt)
    raw = str(getattr(response, "content", response))
    try:
        parsed = json.loads(strip_json_fence(raw))
    except Exception:
        parsed = {
            "acceptability_label": "uncertain",
            "confidence": 0.0,
            "relevant_ci_dimensions": {},
            "satisfied_expectations": [],
            "violated_expectations": ["LLM returned unparsable judgment"],
            "required_mitigations": ["human_review"],
            "rationale": raw[:1000],
        }
    parsed.setdefault("raw_response", raw)
    return parsed




def normalize_ci_mode(ci_mode: str) -> str:
    mode = str(ci_mode or "full").strip().lower()
    aliases = {
        "hard_only": "full",
        "hard_rules_only": "full",
        "no_ci": "no_hard_rules",
        "skip_hard_rules": "no_hard_rules",
        "utility_only": "no_hard_rules",
        "collapse_stages": "no_staged_flows",
    }
    return aliases.get(mode, mode)


def maybe_collapse_flow_stages(flow: Dict[str, Any], collapse_stages: bool) -> Dict[str, Any]:
    if not collapse_stages:
        return flow
    out = dict(flow)
    out["original_pipelineStage"] = as_str_list(flow.get("pipelineStage"))
    out["pipelineStage"] = ["output_to_application"]
    return out

def llm_judgment_passes(judgment: Dict[str, Any], threshold: float) -> bool:
    label = str(judgment.get("acceptability_label", "")).strip().lower()
    try:
        conf = float(judgment.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    return label in {"acceptable", "acceptable_with_mitigations"} and conf >= threshold


def evaluate_candidate(
    candidate: Dict[str, Any],
    request: Dict[str, Any],
    constraints: Dict[str, Any],
    environment: Optional[Dict[str, Any]],
    sensor_stream: Optional[Dict[str, Any]],
    use_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
    llm_temperature: float = 0.0,
    llm_confidence_threshold: float = 0.75,
    ci_mode: str = "full",
    collapse_stages: bool = False,
) -> Dict[str, Any]:
    ci_mode = normalize_ci_mode(ci_mode)
    if ci_mode == "no_staged_flows":
        collapse_stages = True

    flow = construct_ci_flow(candidate, request, environment, sensor_stream)
    flow = maybe_collapse_flow_stages(flow, collapse_stages)
    hard = evaluate_hard_constraints(flow, constraints)
    result = {
        "pipeline_id": candidate.get("pipeline_id"),
        "matched_output_cap": candidate.get("matched_output_cap"),
        "residual_score": candidate.get("residual_score", risk_score(candidate.get("residual_disclosure", {}))),
        "flow": flow,
        "hard_constraint_result": hard,
        "hard_failure_summary": summarize_hard_constraint_result(hard),
        "llm_norm_judgment": None,
        "ci_decision": None,
        "ci_mode": ci_mode,
    }

    if ci_mode == "no_hard_rules":
        # Keep diagnostics in hard_constraint_result, but do not gate feasibility on them.
        result["ci_decision"] = {
            "decision": "accept_ablation_no_hard_rules",
            "feasible": True,
            "reason": "Ablation: hard CI rules were evaluated for diagnostics but not used as a feasibility gate.",
        }
        return result

    if ci_mode == "llm_only":
        if not use_llm:
            result["ci_decision"] = {
                "decision": "reject_llm_only_no_llm",
                "feasible": False,
                "reason": "LLM-only CI ablation requires --use-llm.",
            }
            return result
        judgment = call_llm_norm_judgment(flow, candidate, request, environment, model=llm_model, temperature=llm_temperature)
        result["llm_norm_judgment"] = judgment
        if llm_judgment_passes(judgment, llm_confidence_threshold):
            result["ci_decision"] = {
                "decision": "accept_ablation_llm_only",
                "feasible": True,
                "reason": "Ablation: LLM norm judgment used without hard-rule gating.",
            }
        else:
            result["ci_decision"] = {
                "decision": "uncertain_or_reject_ablation_llm_only",
                "feasible": False,
                "reason": "Ablation: LLM-only judgment was low-confidence, uncertain, or inappropriate.",
            }
        return result

    if not hard["hard_pass"]:
        result["ci_decision"] = {
            "decision": "reject_hard_constraint",
            "feasible": False,
            "reason": "Hard CI constraint denied the flow or required unmet conditions/transformations.",
            "hard_failure_summary": result["hard_failure_summary"],
        }
        return result
    if use_llm:
        judgment = call_llm_norm_judgment(flow, candidate, request, environment, model=llm_model, temperature=llm_temperature)
        result["llm_norm_judgment"] = judgment
        if llm_judgment_passes(judgment, llm_confidence_threshold):
            result["ci_decision"] = {"decision": "accept_llm_norm", "feasible": True, "reason": "Hard constraints passed and LLM norm judgment met confidence threshold."}
        else:
            result["ci_decision"] = {"decision": "uncertain_or_reject_llm_norm", "feasible": False, "reason": "Hard constraints passed, but LLM judgment was low-confidence, uncertain, or inappropriate."}
    else:
        result["ci_decision"] = {"decision": "accept_hard_constraints_only", "feasible": True, "reason": "Hard constraints passed; LLM norm judgment disabled."}
    return result


def select_best(evaluations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    feasible = [e for e in evaluations if (e.get("ci_decision") or {}).get("feasible")]
    if not feasible:
        return None
    feasible.sort(key=lambda e: (e.get("residual_score", 10**9), len((e.get("flow") or {}).get("attribute", []))))
    return feasible[0]


def evaluate_candidates(
    candidate_output: Dict[str, Any],
    request: Dict[str, Any],
    constraints: Dict[str, Any],
    environment: Optional[Dict[str, Any]] = None,
    sensor_stream: Optional[Dict[str, Any]] = None,
    use_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
    llm_temperature: float = 0.0,
    llm_confidence_threshold: float = 0.75,
    top_k_for_llm: Optional[int] = None,
    ci_mode: str = "full",
    collapse_stages: bool = False,
) -> Dict[str, Any]:
    candidates = candidate_output.get("candidates", []) or []
    candidates_for_llm = candidates
    if use_llm and top_k_for_llm is not None:
        candidates_for_llm = sorted(candidates, key=lambda c: c.get("residual_score", risk_score(c.get("residual_disclosure", {}))))[:top_k_for_llm]
    llm_ids = {id(c) for c in candidates_for_llm}
    ci_mode = normalize_ci_mode(ci_mode)
    if ci_mode == "llm_only":
        use_llm = True
    evals = [
        evaluate_candidate(
            c, request, constraints, environment, sensor_stream,
            use_llm=(use_llm and id(c) in llm_ids),
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_confidence_threshold=llm_confidence_threshold,
            ci_mode=ci_mode,
            collapse_stages=collapse_stages,
        )
        for c in candidates
    ]
    best = select_best(evals)
    no_compromise_diagnostics = None
    if best:
        decision = {"decision": "select_pipeline", "selected_pipeline_id": best.get("pipeline_id"), "selected_output_cap": best.get("matched_output_cap"), "reason": (best.get("ci_decision") or {}).get("reason")}
    else:
        hard_rejections = [e for e in evals if (e.get("ci_decision") or {}).get("decision") == "reject_hard_constraint"]
        no_compromise_diagnostics = build_no_compromise_diagnostics(evals, candidates)
        decision_kind = "no_compromise" if hard_rejections else ("consent_or_review_required" if candidates else "no_candidates")
        decision = {
            "decision": decision_kind,
            "selected_pipeline_id": None,
            "selected_output_cap": None,
            "reason": no_compromise_diagnostics.get("primary_reason", "No candidate pipeline passed the CI evaluator."),
            "no_compromise_diagnostics": no_compromise_diagnostics,
        }
    return {
        "schema_version": "smartpriv_ci_evaluation_output_v1",
        "request_id": candidate_output.get("request_id") or request.get("request_identity", {}).get("request_id"),
        "scenario_id": candidate_output.get("scenario_id") or request.get("request_identity", {}).get("scenario_id"),
        "evaluator": {
            "hard_constraints_file_schema": constraints.get("schema_version"),
            "use_llm": use_llm,
            "llm_model": llm_model if use_llm else None,
            "llm_confidence_threshold": llm_confidence_threshold if use_llm else None,
            "templates_ignored": True,
            "ci_mode": ci_mode,
            "collapse_stages": collapse_stages or ci_mode == "no_staged_flows",
        },
        "decision": decision,
        "no_compromise_diagnostics": no_compromise_diagnostics,
        "evaluations": evals,
    }


def read_optional_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    return load_json(path) if path else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate candidate pipelines under CI hard constraints and optional LLM norm judgment.")
    p.add_argument("--constraints", required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--environment")
    p.add_argument("--sensor-stream")
    p.add_argument("--out")
    p.add_argument("--use-llm", action="store_true")
    p.add_argument("--llm-model", default="gpt-4o-mini")
    p.add_argument("--llm-temperature", type=float, default=0.0)
    p.add_argument("--llm-confidence-threshold", type=float, default=0.75)
    p.add_argument("--top-k-for-llm", type=int)
    p.add_argument("--ci-mode", default="full", choices=["full", "no_hard_rules", "llm_only", "no_staged_flows"], help="Ablation mode for CI evaluation.")
    p.add_argument("--collapse-stages", action="store_true", help="Ablation: collapse staged flows to output_to_application before rule evaluation.")
    args = p.parse_args(argv)
    result = evaluate_candidates(
        candidate_output=load_json(args.candidates),
        request=load_json(args.request),
        constraints=load_json(args.constraints),
        environment=read_optional_json(args.environment),
        sensor_stream=read_optional_json(args.sensor_stream),
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_confidence_threshold=args.llm_confidence_threshold,
        top_k_for_llm=args.top_k_for_llm,
        ci_mode=args.ci_mode,
        collapse_stages=args.collapse_stages,
    )
    if args.out:
        write_json(result, args.out)
    print(json.dumps({"request_id": result["request_id"], "scenario_id": result["scenario_id"], "decision": result["decision"], "evaluated": len(result["evaluations"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
