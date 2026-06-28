#!/usr/bin/env python3
"""Direct LLM preprocessing baseline with downstream-output awareness.

This baseline asks an LLM to directly choose exactly one preprocessing pipeline
from the operator catalog. Unlike the full mediator, the LLM is the planner and
selector. Unlike earlier versions of this baseline, it is explicitly prompted to
satisfy the application request's accepted output caps / downstream input
interface.

Important: this baseline intentionally does NOT enumerate all symbolic pipeline
candidates. It performs only lightweight post-hoc checks:
  1. Did the LLM return parseable JSON?
  2. Did it select a pipeline, no-compromise, or review outcome?
  3. If it selected a pipeline, did it use only operator ids from the catalog?
  4. If it selected a pipeline, did it declare a final output cap that appears
     compatible with one accepted output cap in the application request?

If the chosen chain would fail at runtime or is not actually type-valid when
implemented, that is recorded as baseline weakness rather than repaired. If the
LLM decides that no pipeline can both preserve utility and be contextually
appropriate, the baseline now returns a first-class no_compromise/needs_review
result rather than treating that as an invalid pipeline.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .common import (
        load_json,
        stable_hash,
        strip_json_fence,
        wrap_baseline_output,
        write_json,
    )
except ImportError:
    from common import (  # type: ignore
        load_json,
        stable_hash,
        strip_json_fence,
        wrap_baseline_output,
        write_json,
    )


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


CAP_TYPE_ALIASES: Dict[str, str] = {
    "application/x-sound-event-label": "application/x-sound-event",
    "application/x-occupancy-count": "application/x-occupancy",
    "application/x-binary-occupancy": "application/x-occupancy",
    "application/x-activity-label": "application/x-activity-event",
    "application/x-safety-event": "application/x-activity-event",
    "application/x-security-event": "application/x-activity-event",
}


def normalize_cap_type(value: Any) -> str:
    t = str(value or "").strip()
    return CAP_TYPE_ALIASES.get(t, t)


SEMANTIC_FAMILIES: Dict[str, set[str]] = {
    "application/x-count": {
        "application/x-count",
        "application/x-occupancy-count",
        "application/x-binary-occupancy",
    },
    "application/x-event": {
        "application/x-event",
        "application/x-safety-event",
        "application/x-security-event",
        "application/x-sound-event-label",
        "application/x-activity-label",
        "application/x-fused-event",
    },
    "application/x-observation": {
        "application/x-observation",
        "application/x-occupancy-count",
        "application/x-binary-occupancy",
        "application/x-decibel-level",
        "application/x-command-intent",
        "application/x-sound-event-label",
        "application/x-safety-event",
        "application/x-security-event",
        "application/x-activity-label",
    },
    "application/x-detections": {"application/x-detections"},
    "application/x-sensor-reading": {
        "application/x-sensor-reading",
        "application/x-motion-events",
        "application/x-noise-sensor",
    },
    "application/x-occupancy": {"application/x-occupancy", "application/x-occupancy-count", "application/x-binary-occupancy"},
    "application/x-sound-event": {"application/x-sound-event", "application/x-sound-event-label"},
    "application/x-activity-event": {"application/x-activity-event", "application/x-activity-label", "application/x-safety-event", "application/x-security-event"},
}


def cap_type(cap: Optional[Dict[str, Any]]) -> str:
    if not cap:
        return ""
    return normalize_cap_type(cap.get("semantic_type") or cap.get("media_type") or "")


def cap_schema(cap: Optional[Dict[str, Any]]) -> str:
    if not cap:
        return ""
    return str(cap.get("schema") or "")


def type_matches(upstream: str, downstream: str) -> bool:
    """Lightweight compatibility relation, mirroring the symbolic generator.

    upstream is the LLM-declared final output type; downstream is an accepted
    application cap. Redacted/filtered media are treated as interface-compatible
    with raw-media consumers, while still being distinct disclosure types.
    """
    upstream = normalize_cap_type(upstream)
    downstream = normalize_cap_type(downstream)
    if not downstream or downstream == "*":
        return True
    if not upstream:
        return False
    if upstream == downstream:
        return True
    if downstream in SEMANTIC_FAMILIES and upstream in SEMANTIC_FAMILIES[downstream]:
        return True
    if upstream in SEMANTIC_FAMILIES and downstream in SEMANTIC_FAMILIES[upstream]:
        return True
    if downstream == "video/x-raw" and upstream in {"video/x-redacted", "video/x-raw"}:
        return True
    if downstream == "image/x-raw" and upstream in {"image/x-redacted", "image/x-raw"}:
        return True
    if downstream == "audio/x-raw" and upstream in {"audio/x-filtered", "audio/x-raw"}:
        return True
    return False


def cap_matches_goal(out_cap: Dict[str, Any], accepted_cap: Dict[str, Any]) -> bool:
    goal_t = accepted_cap.get("semantic_type") or accepted_cap.get("media_type")
    out_t = cap_type(out_cap)
    if goal_t and type_matches(out_t, str(goal_t)):
        return True
    goal_schema = accepted_cap.get("schema")
    if goal_schema and cap_schema(out_cap) == str(goal_schema):
        return True
    return False


TASK_RELEVANT_TERMS: Dict[str, set[str]] = {
    "visitor_presence_detection": {
        "visitor", "presence", "person", "people", "occupancy", "count", "detection",
        "detect", "chokepoint", "entry", "image", "video", "visual", "frame",
        "crop", "blur", "region", "silhouette", "motion", "aggregate",
    },
    "fall_detection": {
        "fall", "safety", "pose", "keypoint", "skeleton", "motion", "silhouette",
        "le2i", "activity", "event", "video", "image", "frame", "window",
    },
    "adl_recognition": {
        "adl", "activity", "youhome", "audio", "visual", "video", "image", "av",
        "pose", "detection", "sound", "speech", "multimodal", "primitive", "redact",
        "filter", "window", "sample",
    },
    "domestic_sound_monitoring": {
        "domestic", "sound", "audio", "chime", "chime_home", "waveform", "speech",
        "speaker", "filtered", "event", "decibel", "noise", "aggregate", "window",
    },
}

ALWAYS_INCLUDE_OPERATOR_IDS = {
    "op.source", "op.route_publish", "op.schema_adapter", "op.sample", "op.window",
    "op.trigger_gate", "op.join_fuse", "op.youhome_av_adapter",
}


def infer_task_key_for_prompt(request: Dict[str, Any]) -> str:
    rid = request.get("request_identity", {}) or {}
    uc = request.get("utility_contract", {}) or {}
    ctx = request.get("ci_context", {}) or {}
    haystack = " ".join(
        str(x)
        for x in [
            rid.get("request_id"), rid.get("scenario_id"), rid.get("application_category"), rid.get("application_name"),
            uc.get("requested_capability"), uc.get("task_description"),
            ctx.get("purpose"), ctx.get("context"),
        ]
    ).lower()
    if any(x in haystack for x in ["chokepoint", "visitor", "presence", "entry", "security_monitoring"]):
        return "visitor_presence_detection"
    if any(x in haystack for x in ["fall", "le2i", "elder", "safety"]):
        return "fall_detection"
    if any(x in haystack for x in ["youhome", "adl", "activity", "routine"]):
        return "adl_recognition"
    if any(x in haystack for x in ["chime", "domestic", "sound", "audio", "speech"]):
        return "domestic_sound_monitoring"
    return "unknown"


def _cap_terms(cap: Any) -> set[str]:
    terms: set[str] = set()
    if isinstance(cap, dict):
        for key in ["media_type", "semantic_type", "schema", "cap_id", "content_type"]:
            if cap.get(key):
                terms.update(str(cap.get(key)).lower().replace("/", " ").replace("-", "_").split())
        props = cap.get("properties") if isinstance(cap.get("properties"), dict) else {}
        for v in props.values():
            if isinstance(v, str):
                terms.update(v.lower().replace("/", " ").replace("-", "_").split())
            elif isinstance(v, list):
                for item in v:
                    terms.update(str(item).lower().replace("/", " ").replace("-", "_").split())
    elif isinstance(cap, list):
        for item in cap:
            terms.update(_cap_terms(item))
    return terms


def _operator_role(op: Dict[str, Any]) -> str:
    oid = str(op.get("id") or "").lower()
    cat = str(op.get("category") or "").lower()
    text = " ".join([oid, cat, str(op.get("label") or "")]).lower()
    if "source" in oid or "source" in cat:
        return "source"
    if "route" in oid or "publish" in oid:
        return "publish"
    if "adapter" in oid or "adapter" in cat:
        return "adapter"
    if any(x in text for x in ["blur", "redact", "filter", "crop", "mask", "downscale", "speech_content_removal"]):
        return "format_preserving_transform"
    if any(x in text for x in ["detector", "extractor", "classifier", "spotting", "pose", "ocr", "silhouette"]):
        return "semantic_extractor"
    if any(x in text for x in ["aggregate", "generalize", "occupancy_deriver", "fuser"]):
        return "derive_or_aggregate"
    return "other"


def _compact_operator(op: Dict[str, Any]) -> Dict[str, Any]:
    # Keep enough structure for the LLM to compose a plausible chain, but not the
    # full metadata for every unrelated operator.
    return {
        "id": op.get("id"),
        "label": op.get("label"),
        "role": _operator_role(op),
        "category": op.get("category"),
        "input_caps": op.get("input_caps", []),
        "output_caps": op.get("output_caps", []),
        "utility_capabilities": op.get("utility_capabilities", []),
        "transformation_effects": op.get("transformation_effects", []),
        "ci_annotations": op.get("ci_annotations", {}),
    }


def compact_operator_catalog(operator_catalog: Dict[str, Any], request: Optional[Dict[str, Any]] = None, max_per_role: int = 10) -> Dict[str, Any]:
    """Return a task-relevant catalog for the direct-LLM baseline prompt.

    The direct baseline should not receive thousands of generated candidates, but
    a full operator catalog can still be noisy.  This compactor keeps globally
    necessary operators plus operators whose ids/caps/capabilities overlap the
    request task and accepted output interfaces.  Operators are grouped by role
    so the LLM sees a menu of building blocks rather than a flat wall of JSON.
    """
    request = request or {}
    task = infer_task_key_for_prompt(request)
    accepted_caps = (request.get("utility_contract", {}) or {}).get("accepted_output_caps", []) or []
    output_terms = _cap_terms(accepted_caps)
    task_terms = set(TASK_RELEVANT_TERMS.get(task, set())) | output_terms

    groups: Dict[str, List[Dict[str, Any]]] = {}
    kept_ids: set[str] = set()
    dropped = 0
    for op in operator_catalog.get("operators", []) or []:
        oid = str(op.get("id") or "")
        op_text = json.dumps({
            "id": op.get("id"),
            "label": op.get("label"),
            "category": op.get("category"),
            "input_caps": op.get("input_caps", []),
            "output_caps": op.get("output_caps", []),
            "utility_capabilities": op.get("utility_capabilities", []),
        }, default=str).lower().replace("/", " ").replace("-", "_")
        relevant = oid in ALWAYS_INCLUDE_OPERATOR_IDS or any(term and term in op_text for term in task_terms)
        if not relevant:
            dropped += 1
            continue
        role = _operator_role(op)
        groups.setdefault(role, [])
        if len(groups[role]) < max_per_role or oid in ALWAYS_INCLUDE_OPERATOR_IDS:
            groups[role].append(_compact_operator(op))
            kept_ids.add(oid)
        else:
            dropped += 1

    # Ensure source/publish/schema_adapter are present if in the original catalog.
    if kept_ids:
        by_id = {str(op.get("id")): op for op in operator_catalog.get("operators", []) or []}
        for oid in sorted(ALWAYS_INCLUDE_OPERATOR_IDS & set(by_id)):
            if oid not in kept_ids:
                groups.setdefault(_operator_role(by_id[oid]), []).append(_compact_operator(by_id[oid]))
                kept_ids.add(oid)

    role_order = ["source", "format_preserving_transform", "semantic_extractor", "derive_or_aggregate", "adapter", "publish", "other"]
    return {
        "task_key": task,
        "compaction_policy": "task_relevant_catalog_not_generated_candidates",
        "operator_count_total": len(operator_catalog.get("operators", []) or []),
        "operator_count_kept": sum(len(v) for v in groups.values()),
        "operator_count_dropped": dropped,
        "roles": {role: groups.get(role, []) for role in role_order if groups.get(role)},
        "notes": [
            "This is not a generated candidate list; it is a task-relevant menu of operator building blocks.",
            "Prefer richer accepted-output interfaces when contextually acceptable; do not blindly pick the lowest-information output.",
            "Generic schema_adapter is not an inference model and may be used at most once.",
        ],
    }


def compact_application_output_contract(request: Dict[str, Any]) -> Dict[str, Any]:
    uc = request.get("utility_contract", {}) or {}
    return {
        "requested_capability": uc.get("requested_capability"),
        "task_description": uc.get("task_description"),
        "accepted_output_caps": uc.get("accepted_output_caps", []) or [],
        "explicitly_rejected_output_caps": uc.get("explicitly_rejected_output_caps", []) or [],
        "compatibility_constraints": request.get("compatibility_constraints", {}) or {},
        "source_requirements": request.get("source_requirements", {}) or {},
    }


def make_direct_llm_prompt(
    operator_catalog: Dict[str, Any],
    request: Dict[str, Any],
    environment: Optional[Dict[str, Any]] = None,
) -> str:
    output_contract = compact_application_output_contract(request)
    ci_context = request.get("ci_context", {}) or {}
    contextual_integrity_context = {
        "note": (
            "The following fields are contextual-integrity parameters. Treat them as the "
            "social/normative context for judging whether a preprocessing output is acceptable."
        ),
        "context": ci_context.get("context", []),
        "space": ci_context.get("space", []),
        "sender": ci_context.get("sender", []),
        "subject": ci_context.get("subject", []),
        "recipient": ci_context.get("recipient", []),
        "purpose": ci_context.get("purpose", []),
        "transmission_principle": ci_context.get("transmissionPrinciple_assumed", []),
        "social_context_tags": ci_context.get("social_context_tags", []),
    }
    return f"""
You are a direct-LLM privacy mediator baseline. Your job is to choose exactly ONE preprocessing pipeline for the application request.

This is a direct-LLM baseline, not the full mediator:
- Do NOT enumerate alternatives.
- Do NOT ask for another planner, verifier, CI evaluator, or selector to repair your answer.
- Choose exactly one operator sequence from the catalog, OR explicitly choose no_compromise / needs_review.
- Do not use repeated generic schema_adapter steps. At most one op.schema_adapter may appear unless the catalog names a specific task adapter.
- Do not use op.schema_adapter as an inference model: it may repackage equivalent records but may not convert motion features into sound events, sound events into ADL labels, or detections into fall labels.
- You may make a mistake; if your proposed chain is not actually executable, that is still the baseline result.
- The catalog below is task-relevant and grouped by operator role. Use it as a building-block menu, not as an exhaustive proof that no other chain exists.

The request contains TWO different kinds of constraints that you must consider together, plus a utility/privacy tradeoff rule:

1. DOWNSTREAM UTILITY / FORMAT COMPATIBILITY
The final output must be usable by the downstream application described in accepted_output_caps.
Do not choose a privacy-minimized semantic label/event if the downstream application expects media, pose, or audio-waveform input.
Utility is usually higher when the app receives richer app-relevant evidence. Do not automatically collapse to the least revealing semantic label. Choose the highest-utility accepted output that is still contextually appropriate and no more revealing than needed.
Examples:
- If the app accepts image/x-raw image frames, you may output image/x-raw or image/x-redacted only if it is still an image-frame interface.
- If the app accepts video/x-raw video frames, you may output video/x-raw or video/x-redacted only if it is still a video-frame interface.
- If the app accepts audio/x-raw waveform chunks, you may output audio/x-raw or audio/x-filtered only if it is still waveform-like audio.
- If the app accepts application/x-pose-keypoints, output pose keypoints rather than a fall label.
- If the app expects an audio-visual sample, do not output only an activity label.

2. CONTEXTUAL-INTEGRITY / ACCEPTABILITY GUIDANCE
The context is given in contextual-integrity form: context, physical space, sender, subject, recipient, purpose, and transmission principle.
Use these CI parameters to choose the most acceptable utility-preserving preprocessing pipeline, not merely the most technically direct one.

When several pipelines could satisfy the downstream format, choose a balanced option:
- First preserve enough app-relevant information to plausibly maintain downstream utility.
- Then minimize disclosure beyond what is needed for the stated purpose.
- Do not rank outputs solely by information reduction; a very coarse output may be privacy-preserving but may fail the intended utility contract.
- Prefer transformations that preserve the downstream input interface while reducing sensitive information, such as face/background blur, field-of-view cropping, pose extraction, speech-content removal, aggregation, local/ephemeral processing, or event-triggered collection when appropriate.
- In sensitive spaces such as bedrooms, bathrooms, patient rooms, schools, care settings, workplaces, and short-term rentals, avoid raw identifiable images/video/audio unless the task cannot work otherwise.
- For guests, visitors, children, patients, employees, bystanders, or people with weak preference channels, be more conservative about identity, face, speech content, speaker identity, trajectory, visible text, and co-presence.
- Respect the sender/recipient relationship. A host, employer, school, or organization receiving information about guests/employees/children should usually receive less revealing output than a local downstream app or a caregiver receiving a safety signal.
- Respect the stated purpose. Do not preserve details useful for other purposes, such as work monitoring, routine tracking, identification, or behavioral inference, unless those details are necessary for the requested task.
- Respect the transmission principle. Hidden or continuous collection should push you toward stronger minimization or denial; disclosed, local, event-triggered, explicit-consent, or authorized-personnel-only flows may still require minimization.
- If no operator chain can both satisfy the downstream app format and be plausibly acceptable for the CI context, return decision = "no_compromise".
- If a pipeline may be acceptable only with missing consent, disclosure, approval, or human judgment, return decision = "needs_review".
- Do not use no_compromise simply because a less revealing output would have lower utility; use it only when you believe every utility-compatible output would be contextually inappropriate or policy-prohibited.

You are NOT allowed to invent operators. Choose only operator ids from the catalog. Use op.source at the beginning and op.route_publish at the end if available.

Return exactly one valid JSON object with these keys. Do not include Markdown fences or text outside the JSON object.
{{
  "decision": "select_pipeline",
  "operator_ids": ["op.source", "op.route_publish"],
  "operator_parameters": {{}},
  "matched_accepted_cap_id": null,
  "final_output_cap": {{
    "media_type": null,
    "semantic_type": null,
    "schema": null,
    "properties": {{}}
  }},
  "downstream_compatibility_rationale": "one sentence",
  "information_sufficiency_rationale": "why this output preserves enough utility rather than over-minimizing",
  "contextual_integrity_rationale": "one sentence",
  "privacy_rationale": "one sentence",
  "utility_privacy_tradeoff_rationale": "why this is the highest-utility contextually acceptable output, not merely the least revealing output",
  "rationale": "brief overall explanation",
  "no_compromise_rationale": null,
  "needs_review_rationale": null
}}
Allowed decision values are exactly: "select_pipeline", "no_compromise", "needs_review", or "deny".

Application request output contract:
{json.dumps(output_contract, indent=2)}

Contextual-integrity context to use for acceptability reasoning:
{json.dumps(contextual_integrity_context, indent=2)}

Full application request:
{json.dumps(request, indent=2)}

Environment/context:
{json.dumps(environment or {}, indent=2)}

Task-relevant operator catalog grouped by role:
{json.dumps(compact_operator_catalog(operator_catalog, request), indent=2)}
""".strip()


def extract_json_object(text: str) -> str:
    """Return the first parseable JSON object from an LLM response.

    Models sometimes wrap valid JSON in prose, Markdown, or trailing notes.  This
    keeps the direct-LLM baseline from reporting llm_error merely because the
    response had a preface.  It does not repair semantic mistakes; it only
    extracts a syntactically valid object.
    """
    s = strip_json_fence(text)
    try:
        json.loads(s)
        return s
    except Exception:
        pass
    start = s.find("{")
    if start < 0:
        return s
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return s[start:]


def call_direct_llm(prompt: str, model: str, temperature: float = 0.0) -> Dict[str, Any]:
    system_msg = (
        "Return one valid JSON object only. Do not use Markdown fences, prose, "
        "comments, or trailing explanation."
    )
    # Prefer the same LangChain backend used by contextual_integrity_evaluator.py,
    # but request JSON-object mode when supported.
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        try:
            llm = ChatOpenAI(model=model, temperature=temperature, model_kwargs={"response_format": {"type": "json_object"}})
        except TypeError:
            llm = ChatOpenAI(model=model, temperature=temperature)
        resp = llm.invoke([("system", system_msg), ("user", prompt)])
        raw = str(getattr(resp, "content", resp))
    except Exception as first_exc:
        # Fall back to the OpenAI Python SDK if available.
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
                )
            except Exception:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
                )
            raw = resp.choices[0].message.content or ""
        except Exception as second_exc:
            return {
                "_call_status": "error",
                "error": f"Could not call LLM backend. langchain_openai error: {first_exc}; openai SDK error: {second_exc}",
                "raw_response": "",
            }
    try:
        parsed = json.loads(extract_json_object(raw))
        parsed["_call_status"] = "ok"
        parsed["raw_response"] = raw
        return parsed
    except Exception as exc:
        return {
            "_call_status": "unparsable",
            "error": str(exc),
            "raw_response": raw,
        }


def canonicalize_operator_sequence(seq: Sequence[str]) -> List[str]:
    out = [str(x).strip() for x in seq if str(x).strip()]
    if out and out[0] != "op.source":
        out.insert(0, "op.source")
    if out and out[-1] != "op.route_publish":
        out.append("op.route_publish")
    return out




def validate_direct_llm_operator_sequence(op_ids: Sequence[str]) -> List[str]:
    issues: List[str] = []
    if list(op_ids).count("op.schema_adapter") > 1:
        issues.append("op.schema_adapter appears more than once; generic schema adaptation is limited to one step.")
    for a, b in zip(op_ids, list(op_ids)[1:]):
        if a == "op.schema_adapter" and b == "op.schema_adapter":
            issues.append("Consecutive op.schema_adapter steps are not allowed.")
            break
    return issues

def catalog_operator_ids(operator_catalog: Dict[str, Any]) -> set[str]:
    ids = {str(op.get("id")) for op in operator_catalog.get("operators", []) or [] if op.get("id")}
    # route_publish/source are sometimes treated as special bookkeeping in planners.
    ids.update({"op.source", "op.route_publish"})
    return ids


def parameters_for_operator(operator_id: str, params: Any) -> Dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    val = params.get(operator_id)
    return copy.deepcopy(val) if isinstance(val, dict) else {}


def find_declared_output_match(
    llm_choice: Dict[str, Any],
    request: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    accepted = (request.get("utility_contract", {}) or {}).get("accepted_output_caps", []) or []
    requested_id = llm_choice.get("matched_accepted_cap_id")
    final_cap = llm_choice.get("final_output_cap") if isinstance(llm_choice.get("final_output_cap"), dict) else {}

    # If the LLM names an accepted cap, prefer that as the matched app contract.
    if requested_id:
        for cap in accepted:
            if str(cap.get("cap_id")) == str(requested_id):
                compatible = cap_matches_goal(final_cap, cap) if final_cap else True
                return cap, {
                    "declared_output_compatibility": compatible,
                    "declared_output_compatibility_reason": "LLM named an accepted cap id; final_output_cap was checked when present.",
                    "llm_declared_final_output_cap": final_cap or None,
                }

    # Otherwise infer a match from the declared final output cap.
    if final_cap:
        for cap in accepted:
            if cap_matches_goal(final_cap, cap):
                return cap, {
                    "declared_output_compatibility": True,
                    "declared_output_compatibility_reason": "LLM final_output_cap is exact or interface-compatible with an accepted cap.",
                    "llm_declared_final_output_cap": final_cap,
                }
        return None, {
            "declared_output_compatibility": False,
            "declared_output_compatibility_reason": "LLM final_output_cap did not match any accepted_output_caps entry.",
            "llm_declared_final_output_cap": final_cap,
        }

    return None, {
        "declared_output_compatibility": False,
        "declared_output_compatibility_reason": "LLM did not provide matched_accepted_cap_id or final_output_cap.",
        "llm_declared_final_output_cap": None,
    }


def ci_terms_from_request_and_output(request: Dict[str, Any], matched_cap: Optional[Dict[str, Any]], final_cap: Dict[str, Any]) -> Dict[str, List[str]]:
    ci = {
        "informationType.sensorPrimitive": [],
        "informationType.interpretedObservation": [],
        "informationType.inferredInformationType": [],
        "transmissionPrinciple": [],
        "pipelineStage": ["output_to_application"],
    }
    req_ci = request.get("ci_output_constraints", {}) or {}
    preferred = (req_ci.get("preferred_ci_terms", {}) or {}).get("transmissionPrinciple", []) or []
    required = (req_ci.get("required_ci_terms", {}) or {}).get("transmissionPrinciple", []) or []
    assumed = (request.get("ci_context", {}) or {}).get("transmissionPrinciple_assumed", []) or []
    ci["transmissionPrinciple"] = sorted({str(x) for x in list(preferred) + list(required) + list(assumed)})

    for source in [matched_cap or {}, final_cap or {}]:
        info = source.get("required_informationType") if isinstance(source.get("required_informationType"), dict) else None
        if info:
            for subkey, vals in info.items():
                key = f"informationType.{subkey}"
                ci.setdefault(key, [])
                if isinstance(vals, list):
                    ci[key].extend(str(v) for v in vals)
                elif vals:
                    ci[key].append(str(vals))
        props = source.get("properties") if isinstance(source.get("properties"), dict) else None
        if props:
            for subkey in ["sensorPrimitive", "interpretedObservation", "inferredInformationType"]:
                if subkey in props:
                    key = f"informationType.{subkey}"
                    vals = props.get(subkey)
                    if isinstance(vals, list):
                        ci.setdefault(key, []).extend(str(v) for v in vals)
                    elif vals:
                        ci.setdefault(key, []).append(str(vals))

    return {k: sorted(set(v)) for k, v in ci.items()}


def conservative_unknown_residual() -> Dict[str, str]:
    return {a: "unknown" for a in RESIDUAL_ATTRIBUTES}



NO_COMPROMISE_DECISIONS = {"no_compromise", "deny", "denied", "reject", "reject_flow"}
NEEDS_REVIEW_DECISIONS = {"needs_review", "review", "consent_or_review_required", "human_review"}
SELECT_DECISIONS = {"select_pipeline", "select", "pipeline"}


def normalize_llm_decision(decision: Any) -> str:
    """Normalize the LLM's free-form decision into the baseline result space."""
    d = str(decision or "").strip().lower().replace("-", "_").replace(" ", "_")
    if d in SELECT_DECISIONS:
        return "select_pipeline"
    if d in NO_COMPROMISE_DECISIONS:
        return "no_compromise"
    if d in NEEDS_REVIEW_DECISIONS:
        return "consent_or_review_required"
    return "invalid_or_no_pipeline"


def make_non_selection_diagnostics(llm_choice: Dict[str, Any], normalized_decision: str) -> Dict[str, Any]:
    """Diagnostics for LLM no-compromise/review outcomes.

    These fields mirror the full mediator's no-compromise output enough for
    context-generation and survey code to treat the result as no shared output,
    while still preserving why the LLM denied or escalated the scenario.
    """
    rationale = (
        llm_choice.get("no_compromise_rationale")
        or llm_choice.get("needs_review_rationale")
        or llm_choice.get("contextual_integrity_rationale")
        or llm_choice.get("privacy_rationale")
        or llm_choice.get("rationale")
    )
    return {
        "validation_status": "non_selection_decision",
        "validation_reason": f"LLM returned {llm_choice.get('decision')!r}, normalized to {normalized_decision}.",
        "llm_normalized_decision": normalized_decision,
        "llm_no_compromise_rationale": llm_choice.get("no_compromise_rationale"),
        "llm_needs_review_rationale": llm_choice.get("needs_review_rationale"),
        "llm_contextual_integrity_rationale": llm_choice.get("contextual_integrity_rationale"),
        "llm_privacy_rationale": llm_choice.get("privacy_rationale"),
        "llm_downstream_compatibility_rationale": llm_choice.get("downstream_compatibility_rationale"),
        "llm_overall_rationale": llm_choice.get("rationale"),
        "no_compromise_diagnostics": {
            "source": "direct_llm_baseline",
            "decision": normalized_decision,
            "reason": rationale,
            "candidate_enumeration_used": False,
            "hard_ci_rules_evaluated": False,
            "note": (
                "This is the direct-LLM baseline's own judgment. Unlike the full mediator, "
                "it does not enumerate all candidates or prove that every utility-compatible "
                "pipeline failed hard CI constraints."
            ),
        } if normalized_decision == "no_compromise" else None,
        "review_diagnostics": {
            "source": "direct_llm_baseline",
            "decision": normalized_decision,
            "reason": rationale,
            "candidate_enumeration_used": False,
            "hard_ci_rules_evaluated": False,
        } if normalized_decision == "consent_or_review_required" else None,
        "full_candidate_enumeration_used": False,
    }

def make_llm_candidate(
    operator_catalog: Dict[str, Any],
    request: Dict[str, Any],
    llm_choice: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    normalized_decision = normalize_llm_decision(llm_choice.get("decision"))
    if normalized_decision != "select_pipeline":
        return None, make_non_selection_diagnostics(llm_choice, normalized_decision)
    if not isinstance(llm_choice.get("operator_ids"), list):
        return None, {
            "validation_status": "invalid",
            "reason": "LLM did not return operator_ids as a list.",
        }

    op_ids = canonicalize_operator_sequence(llm_choice.get("operator_ids", []))
    known_ids = catalog_operator_ids(operator_catalog)
    unknown_ops = [op for op in op_ids if op not in known_ids]
    sequence_issues = validate_direct_llm_operator_sequence(op_ids)

    matched_cap, output_diag = find_declared_output_match(llm_choice, request)
    final_cap = copy.deepcopy(llm_choice.get("final_output_cap") if isinstance(llm_choice.get("final_output_cap"), dict) else {})
    if not final_cap and matched_cap:
        final_cap = copy.deepcopy(matched_cap)

    params = llm_choice.get("operator_parameters") if isinstance(llm_choice.get("operator_parameters"), dict) else {}
    operators: List[Dict[str, Any]] = []
    for op_id in op_ids:
        output_cap = final_cap if op_id == op_ids[-1] or op_id == "op.route_publish" else {}
        operators.append({
            "operator": op_id,
            "variant": "llm_direct_choice",
            "output_cap": copy.deepcopy(output_cap),
            "parameters": parameters_for_operator(op_id, params),
        })

    requested_capability = (request.get("utility_contract", {}) or {}).get("requested_capability")
    ci_terms = ci_terms_from_request_and_output(request, matched_cap, final_cap)

    candidate = {
        "pipeline_id": "baseline_direct_llm_" + stable_hash({
            "operator_ids": op_ids,
            "final_output_cap": final_cap,
            "request_id": (request.get("request_identity", {}) or {}).get("request_id"),
        }),
        "decision": "candidate_pipeline",
        "baseline": "direct_llm",
        "matched_output_cap": (matched_cap or {}).get("cap_id"),
        "matched_output_schema": (matched_cap or {}).get("schema"),
        "final_output_cap": final_cap,
        "operators": operators,
        "utility_capabilities": [requested_capability] if requested_capability else [],
        "quality_status": "llm_declared_requires_runtime_or_benchmark_validation",
        "ci_terms": ci_terms,
        "transforms": [],
        "residual_disclosure": conservative_unknown_residual(),
        "residual_score": 999,
        "executable_under_catalog": not unknown_ops,
        "operator_ids_exist_in_catalog": not unknown_ops,
        "unknown_operator_ids": unknown_ops,
        "operator_sequence_issues": sequence_issues,
        "baseline_notes": [
            "Direct LLM baseline selected one pipeline directly; no full candidate enumeration was used.",
            "The prompt required the LLM to respect downstream accepted_output_caps.",
            "Only lightweight validation was performed: operator ids and declared output-cap compatibility.",
            "If the chain is not actually executable or fails utility at runtime, that is recorded as a baseline failure rather than repaired.",
        ],
        "llm_choice": {k: v for k, v in llm_choice.items() if k != "raw_response"},
        "downstream_compatibility_rationale": llm_choice.get("downstream_compatibility_rationale"),
        "privacy_rationale": llm_choice.get("privacy_rationale"),
    }

    repeated_schema_adapters = op_ids.count("op.schema_adapter") > 1
    schema_adapter_inference_warning = False
    if "op.schema_adapter" in op_ids:
        out_t = cap_type(final_cap)
        out_schema = cap_schema(final_cap)
        # The direct baseline is allowed to make mistakes, but we flag cases
        # where the LLM appears to rely on a generic adapter to emit a semantic
        # task output.  The full symbolic generator enforces this more strictly.
        schema_adapter_inference_warning = out_t.startswith("application/x-") and out_schema not in {"", "occupancy_count", "room_occupied", "object_detections", "pose_keypoints", "sound_event_label", "aggregate_summary"}

    validation_status = "ok"
    validation_reason = "LLM returned a single declared pipeline."
    if unknown_ops:
        validation_status = "invalid_operator_ids"
        validation_reason = "LLM used operators not present in the catalog."
    elif repeated_schema_adapters:
        validation_status = "repeated_schema_adapter"
        validation_reason = "LLM used repeated generic schema_adapter steps, which are not allowed in the stricter flexible compatibility model."
    elif not output_diag.get("declared_output_compatibility"):
        # Keep the selected baseline result, but mark compatibility as questionable.
        validation_status = "declared_output_incompatible_or_unknown"
        validation_reason = str(output_diag.get("declared_output_compatibility_reason"))
    elif schema_adapter_inference_warning:
        validation_status = "schema_adapter_may_be_doing_inference"
        validation_reason = "LLM used a generic schema_adapter before a semantic task output; this is flagged as questionable rather than repaired."

    diagnostics = {
        "validation_status": validation_status,
        "validation_reason": validation_reason,
        "llm_operator_ids": op_ids,
        "unknown_operator_ids": unknown_ops,
        "operator_sequence_issues": sequence_issues,
        "repeated_schema_adapters": repeated_schema_adapters,
        "schema_adapter_inference_warning": schema_adapter_inference_warning,
        "matched_accepted_output_cap": matched_cap,
        **output_diag,
        "full_candidate_enumeration_used": False,
    }
    return candidate, diagnostics


def run_direct_llm_baseline(
    operator_catalog: Dict[str, Any],
    request: Dict[str, Any],
    environment: Optional[Dict[str, Any]] = None,
    candidate_generator_path: Optional[str | Path] = None,
    max_depth: int = 7,
    max_states: int = 25000,
    llm_model: str = "gpt-4o",
    llm_temperature: float = 0.0,
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    # candidate_generator_path/max_depth/max_states are intentionally accepted for
    # CLI compatibility with older scripts, but are not used by this baseline.
    _ = (candidate_generator_path, max_depth, max_states)

    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    prompt = make_direct_llm_prompt(operator_catalog, request, environment=environment)
    llm_choice = call_direct_llm(prompt, model=llm_model, temperature=llm_temperature)

    diagnostics: Dict[str, Any] = {
        "llm_call_status": llm_choice.get("_call_status"),
        "llm_error": llm_choice.get("error"),
        "llm_decision": llm_choice.get("decision"),
        "llm_operator_ids": llm_choice.get("operator_ids"),
        "llm_target_output_cap": llm_choice.get("target_output_cap"),
        "llm_matched_accepted_cap_id": llm_choice.get("matched_accepted_cap_id"),
        "llm_final_output_cap": llm_choice.get("final_output_cap"),
        "llm_rationale": llm_choice.get("rationale"),
        "full_candidate_enumeration_used": False,
        "candidate_generator_ignored": bool(candidate_generator_path),
    }

    if llm_choice.get("_call_status") != "ok":
        return wrap_baseline_output(
            baseline_name="direct_llm",
            request=request,
            candidates=[],
            selected_pipeline_id=None,
            decision="llm_error",
            reason="Direct LLM baseline could not produce parseable JSON.",
            diagnostics=diagnostics,
        )

    selected, validation = make_llm_candidate(operator_catalog, request, llm_choice)
    diagnostics.update(validation)
    normalized_decision = str(validation.get("llm_normalized_decision") or normalize_llm_decision(llm_choice.get("decision")))

    if selected:
        return wrap_baseline_output(
            baseline_name="direct_llm",
            request=request,
            candidates=[selected],
            selected_pipeline_id=selected["pipeline_id"],
            decision="select_pipeline",
            reason=(
                "LLM directly selected a single pipeline with downstream-output awareness; "
                "no full candidate enumeration was used."
            ),
            diagnostics=diagnostics,
        )

    if normalized_decision == "no_compromise":
        return wrap_baseline_output(
            baseline_name="direct_llm",
            request=request,
            candidates=[],
            selected_pipeline_id=None,
            decision="no_compromise",
            reason=(
                diagnostics.get("llm_no_compromise_rationale")
                or diagnostics.get("llm_contextual_integrity_rationale")
                or diagnostics.get("llm_privacy_rationale")
                or diagnostics.get("llm_rationale")
                or "Direct LLM baseline judged that no utility-compatible output is contextually appropriate."
            ),
            diagnostics=diagnostics,
        )

    if normalized_decision == "consent_or_review_required":
        return wrap_baseline_output(
            baseline_name="direct_llm",
            request=request,
            candidates=[],
            selected_pipeline_id=None,
            decision="consent_or_review_required",
            reason=(
                diagnostics.get("llm_needs_review_rationale")
                or diagnostics.get("llm_contextual_integrity_rationale")
                or diagnostics.get("llm_privacy_rationale")
                or diagnostics.get("llm_rationale")
                or "Direct LLM baseline judged that consent, disclosure, or human review is required."
            ),
            diagnostics=diagnostics,
        )

    return wrap_baseline_output(
        baseline_name="direct_llm",
        request=request,
        candidates=[],
        selected_pipeline_id=None,
        decision="invalid_or_no_pipeline",
        reason="LLM did not select a pipeline, no-compromise, or review outcome.",
        diagnostics=diagnostics,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run the direct LLM preprocessing baseline with downstream-output awareness.")
    p.add_argument("--operators", required=True, help="Path to operator-contract JSON.")
    p.add_argument("--request", required=True, help="Path to structured application request JSON.")
    p.add_argument("--environment", default=None, help="Optional environment/context JSON.")
    p.add_argument("--candidate-generator", default=None, help="Accepted for backwards compatibility, but ignored by this baseline.")
    p.add_argument("--max-depth", type=int, default=7, help="Accepted for backwards compatibility, but ignored.")
    p.add_argument("--max-states", type=int, default=25000, help="Accepted for backwards compatibility, but ignored.")
    p.add_argument("--llm-model", default="gpt-4o-mini")
    p.add_argument("--llm-temperature", type=float, default=0.0)
    p.add_argument("--openai-api-key", default=None)
    p.add_argument("--out", default=None, help="Optional output JSON path.")
    args = p.parse_args(argv)

    result = run_direct_llm_baseline(
        operator_catalog=load_json(args.operators),
        request=load_json(args.request),
        environment=load_json(args.environment) if args.environment else None,
        candidate_generator_path=args.candidate_generator,
        max_depth=args.max_depth,
        max_states=args.max_states,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        openai_api_key=args.openai_api_key,
    )
    if args.out:
        write_json(result, args.out)
    print(json.dumps(result["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
