"""Semantic utility adapter for ChokePoint visitor presence.

This module does not run YOLO.  It assumes the privacy pipeline already emitted
semantic outputs such as detections, occupancy counts, or binary presence and
converts them into the prediction CSV shape expected by the ChokePoint evaluator.
"""
from __future__ import annotations

import shlex
import subprocess
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import (
    json_paths_for_semantic_row,
    load_json,
    load_semantic_json,
    read_csv_rows,
    semantic_labels_and_count,
    write_csv_rows,
    write_json,
)


def _run_metric_command(cmd: List[str], cwd: Optional[str | Path], log_path: Path) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    stdout = proc.stdout or ""
    log_path.write_text("$ " + shlex.join(cmd) + "\n\n" + stdout, encoding="utf-8")
    return {"cmd": cmd, "returncode": proc.returncode, "log_path": str(log_path), "stdout_tail": stdout[-4000:]}


def run(args: Any, row: Any, manifest: Path, data_root: Optional[str | Path], work_dir: Path) -> Dict[str, Any]:
    dry_run = bool(getattr(args, "dry_run", False))
    output_csv = work_dir / "predictions.csv"
    metrics_json = work_dir / "metrics.json"
    rows, _ = read_csv_rows(manifest)
    pred_rows: List[Dict[str, Any]] = []

    if not dry_run:
        for i, r in enumerate(rows):
            sample_id = str(r.get("sample_id") or r.get("chunk_id") or i)
            json_paths = json_paths_for_semantic_row(r)
            if not json_paths:
                pred_rows.append({
                    "sample_id": sample_id,
                    "split": r.get("split", ""),
                    "frame_index": 0,
                    "person_count": 0,
                    "person_present": 0,
                    "boxes_json": "[]",
                    "frame_path": "",
                    "semantic_adapter_error": "missing_semantic_json",
                })
                continue
            for j, jp in enumerate(json_paths):
                obj = load_semantic_json(jp)
                labels, count, occupied, conf = semantic_labels_and_count(obj)
                m = re.search(r"(\d+)(?!.*\d)", jp.stem)
                frame_index = int(m.group(1)) if m else j
                pred_rows.append({
                    "sample_id": sample_id,
                    "split": r.get("split", ""),
                    "frame_index": frame_index,
                    "person_count": int(count),
                    "person_present": 1 if occupied else 0,
                    "boxes_json": "[]",
                    "frame_path": "",
                    "confidence": conf,
                    "semantic_labels": ";".join(labels[:8]),
                    "semantic_json_path": str(jp),
                })

    write_csv_rows(pred_rows, output_csv)
    metrics: Dict[str, Any] = {
        "status": "pending",
        "task": getattr(row, "task", "visitor_presence_detection"),
        "level": "semantic_presence_adapter",
        "adapter_note": "Converted detections/counts/occupancy into ChokePoint prediction CSV columns without running YOLO.",
        "n_prediction_rows": len(pred_rows),
    }

    # If the caller has supplied the same evaluator template used by legacy YOLO
    # runs, reuse it for metrics.  This calls the metric evaluator, not infer_yolo.
    template = getattr(args, "chokepoint_eval_cmd_template", None)
    if template and not dry_run:
        try:
            template_text = str(template).format(
                predictions=str(output_csv),
                manifest=str(manifest),
                data_root=str(data_root or ""),
                out_json=str(metrics_json),
            )
            eval_cmd = shlex.split(template_text)
            eval_run = _run_metric_command(eval_cmd, cwd=getattr(args, "project_root", None), log_path=work_dir / "evaluate_semantic.log")
            metrics["eval_command"] = eval_run
            if metrics_json.exists():
                loaded = load_json(metrics_json)
                if isinstance(loaded, dict):
                    loaded.update({
                        "semantic_adapter": "chokepoint_presence",
                        "adapter_note": metrics["adapter_note"],
                    })
                    metrics = loaded
        except Exception as exc:
            metrics["status"] = "metric_error"
            metrics["metric_error"] = repr(exc)

    write_json(metrics, metrics_json)
    return {
        "status": "ok",
        "inference": {"semantic_adapter": "chokepoint_presence", "returncode": 0, "dry_run": dry_run, "calls_legacy_inference": False},
        "output_csv": str(output_csv),
        "metrics_json": str(metrics_json),
        "metrics": metrics,
    }
