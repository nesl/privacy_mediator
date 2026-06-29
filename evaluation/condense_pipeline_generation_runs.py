#!/usr/bin/env python3
"""Condense existing SmartPriv context-pipeline generation run directories.

This rewrites large result/candidate/CI/selector JSON artifacts in-place into
compact summaries. It preserves selected_pipeline.json, pipeline_spec.json,
summary.csv, and other small files used by utility evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .pipeline_artifact_compaction import condense_run_tree
except Exception:  # Allow direct script execution.
    from pipeline_artifact_compaction import condense_run_tree  # type: ignore


def main() -> int:
    p = argparse.ArgumentParser(description="Condense existing SmartPriv pipeline-generation artifacts in-place.")
    p.add_argument("--root", default="runs/flexible_context_pipeline_generation", help="Run root to condense.")
    p.add_argument("--report", default=None, help="Optional JSON report path.")
    p.add_argument("--no-progress", action="store_true", help="Disable progress display.")
    p.add_argument("--progress-every", type=int, default=25, help="Fallback counter frequency when tqdm is unavailable.")
    p.add_argument("--min-bytes", type=int, default=0, help="Skip recognized artifacts smaller than this many bytes. Default: compact all.")
    args = p.parse_args()
    report = condense_run_tree(
        Path(args.root),
        progress=not args.no_progress,
        progress_every=args.progress_every,
        min_bytes=args.min_bytes,
    )
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
