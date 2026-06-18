#!/usr/bin/env python3
"""Run all preprocessing baselines for one request."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

try:
    from .common import load_json, write_json
    from .direct_llm_baseline import run_direct_llm_baseline
    from .manual_baseline import run_manual_baseline
    from .raw_baseline import run_raw_baseline
except ImportError:
    from common import load_json, write_json  # type: ignore
    from direct_llm_baseline import run_direct_llm_baseline  # type: ignore
    from manual_baseline import run_manual_baseline  # type: ignore
    from raw_baseline import run_raw_baseline  # type: ignore


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run raw, manual, and direct-LLM preprocessing baselines.")
    p.add_argument("--operators", required=True, help="Path to operator-contract JSON.")
    p.add_argument("--request", required=True, help="Path to structured application request JSON.")
    p.add_argument("--environment", default=None, help="Optional environment/context JSON for direct LLM prompt.")
    p.add_argument("--candidate-generator", default=None, help="Optional path to generate_pipeline_candidates.py.")
    p.add_argument("--out-dir", default="baseline_runs", help="Output directory.")
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--max-states", type=int, default=25000)
    p.add_argument("--skip-direct-llm", action="store_true", help="Skip the LLM baseline, useful for offline runs.")
    p.add_argument("--llm-model", default="gpt-4o-mini")
    p.add_argument("--llm-temperature", type=float, default=0.0)
    p.add_argument("--openai-api-key", default=None)
    p.add_argument("--no-manual-symbolic-fallback", action="store_true")
    args = p.parse_args(argv)

    operator_catalog = load_json(args.operators)
    request = load_json(args.request)
    environment = load_json(args.environment) if args.environment else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    raw = run_raw_baseline(operator_catalog, request, candidate_generator_path=args.candidate_generator)
    write_json(raw, out_dir / "raw_baseline.json")
    results["raw"] = raw["decision"]

    manual = run_manual_baseline(
        operator_catalog,
        request,
        candidate_generator_path=args.candidate_generator,
        max_depth=args.max_depth,
        max_states=args.max_states,
        allow_symbolic_fallback=not args.no_manual_symbolic_fallback,
    )
    write_json(manual, out_dir / "manual_baseline.json")
    results["manual"] = manual["decision"]

    if not args.skip_direct_llm:
        direct = run_direct_llm_baseline(
            operator_catalog,
            request,
            environment=environment,
            candidate_generator_path=args.candidate_generator,
            max_depth=args.max_depth,
            max_states=args.max_states,
            llm_model=args.llm_model,
            llm_temperature=args.llm_temperature,
            openai_api_key=args.openai_api_key,
        )
        write_json(direct, out_dir / "direct_llm_baseline.json")
        results["direct_llm"] = direct["decision"]
    else:
        results["direct_llm"] = {"decision": "skipped", "selected_pipeline_id": None, "selected_output_cap": None, "reason": "--skip-direct-llm was set."}

    summary = {
        "schema_version": "smartpriv_preprocessing_baseline_summary_v1",
        "outputs": {
            "raw": str(out_dir / "raw_baseline.json"),
            "manual": str(out_dir / "manual_baseline.json"),
            "direct_llm": None if args.skip_direct_llm else str(out_dir / "direct_llm_baseline.json"),
        },
        "decisions": results,
    }
    write_json(summary, out_dir / "baseline_summary.json")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
