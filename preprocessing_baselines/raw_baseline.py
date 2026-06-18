#!/usr/bin/env python3
"""Raw-data preprocessing baseline.

This baseline represents the no-mediation/default-access comparison: choose an
allowed source and publish the raw stream/reading to the application.  It does
not use CI rules, residual bounds, or least-revealing selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from .common import (
        first_matching_accepted_cap,
        load_generator,
        load_json,
        make_candidate_record,
        residual_score,
        source_candidates_from_catalog,
        wrap_baseline_output,
        write_json,
    )
except ImportError:  # Allow running as a script from inside the folder.
    from common import (  # type: ignore
        first_matching_accepted_cap,
        load_generator,
        load_json,
        make_candidate_record,
        residual_score,
        source_candidates_from_catalog,
        wrap_baseline_output,
        write_json,
    )


def run_raw_baseline(
    operator_catalog: Dict[str, Any],
    request: Dict[str, Any],
    candidate_generator_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    generator = load_generator(candidate_generator_path)
    source_states = source_candidates_from_catalog(operator_catalog, request, generator=generator)

    candidates: List[Dict[str, Any]] = []
    for st in source_states:
        cap = st["cap"]
        match = first_matching_accepted_cap(cap, request)
        cand = make_candidate_record(
            baseline_name="raw",
            request=request,
            operators=st["operators"],
            final_output_cap=cap,
            residual=st.get("residual", {}),
            ci_terms=st.get("ci_terms", {}),
            transforms=st.get("transforms", []),
            utility_capabilities=st.get("utility_capabilities", []),
            matched_output_cap=match,
            quality_status="raw_forwarded_not_contract_checked" if not match else "raw_forwarded",
            notes=[
                "Raw baseline forwards the source representation without preprocessing.",
                "It intentionally ignores contextual-integrity constraints and residual-disclosure bounds.",
                "If matched_output_cap is null, the raw stream was forwarded even though it was not one of the request's accepted output schemas.",
            ],
            executable_under_catalog=True,
        )
        candidates.append(cand)

    # Prefer a raw source that already matches an accepted cap; otherwise choose the
    # highest-disclosure source as the canonical no-mediation comparison.
    candidates.sort(key=lambda c: (
        0 if c.get("matched_output_cap") else 1,
        -int(c.get("residual_score", 0)),
        len(c.get("operators", [])),
    ))

    if not candidates:
        return wrap_baseline_output(
            baseline_name="raw",
            request=request,
            candidates=[],
            selected_pipeline_id=None,
            decision="no_source",
            reason="No allowed source cap was found in the operator catalog for the request.",
            diagnostics={"source_states": 0},
        )

    selected = candidates[0]
    return wrap_baseline_output(
        baseline_name="raw",
        request=request,
        candidates=candidates,
        selected_pipeline_id=selected["pipeline_id"],
        decision="select_pipeline",
        reason="Selected raw source forwarding baseline; no preprocessing or privacy mediation applied.",
        diagnostics={"source_states": len(source_states)},
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run the raw-data preprocessing baseline.")
    p.add_argument("--operators", required=True, help="Path to operator-contract JSON.")
    p.add_argument("--request", required=True, help="Path to structured application request JSON.")
    p.add_argument("--candidate-generator", default=None, help="Optional path to generate_pipeline_candidates.py for source materialization.")
    p.add_argument("--out", default=None, help="Optional output JSON path.")
    args = p.parse_args(argv)

    result = run_raw_baseline(
        operator_catalog=load_json(args.operators),
        request=load_json(args.request),
        candidate_generator_path=args.candidate_generator,
    )
    if args.out:
        write_json(result, args.out)
    print(json.dumps(result["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
