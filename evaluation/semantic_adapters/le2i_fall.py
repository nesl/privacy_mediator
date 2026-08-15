"""Semantic utility adapter for LE2I fall outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import (
    binary_metrics,
    json_paths_for_semantic_row,
    label_to_binary,
    load_semantic_json,
    manifest_truth_label,
    read_csv_rows,
    semantic_labels_and_count,
    write_csv_rows,
    write_json,
)


def run(args: Any, row: Any, manifest: Path, data_root: Optional[str | Path], work_dir: Path) -> Dict[str, Any]:
    dry_run = bool(getattr(args, "dry_run", False))
    output_csv = work_dir / "predictions.csv"
    metrics_json = work_dir / "metrics.json"
    rows, _ = read_csv_rows(manifest)
    pred_rows: List[Dict[str, Any]] = []
    y_true_bin: List[int] = []
    y_pred_bin: List[int] = []

    if not dry_run:
        for i, r in enumerate(rows):
            sample_id = str(r.get("sample_id") or i)
            truth = manifest_truth_label(r, "fall_detection")
            json_paths = json_paths_for_semantic_row(r)
            labels: List[str] = []
            confidence = 0.0
            for jp in json_paths[:64]:
                obj = load_semantic_json(jp)
                labs, _count, _occ, conf = semantic_labels_and_count(obj)
                labels.extend(labs)
                confidence = max(confidence, conf)
            pred_label = labels[0] if labels else "none"
            true_bin = label_to_binary(truth, ["fall"])
            pred_bin = label_to_binary(pred_label, ["fall"])
            pred_rows.append({
                "sample_id": sample_id,
                "true_label": truth,
                "pred_label": "fall" if pred_bin else "nonfall",
                "y_true": true_bin,
                "y_pred": pred_bin,
                "confidence": confidence,
                "semantic_labels": ";".join(labels[:32]),
                "semantic_json_count": len(json_paths),
                "split": r.get("split", ""),
            })
            if truth:
                y_true_bin.append(true_bin)
                y_pred_bin.append(pred_bin)

    write_csv_rows(pred_rows, output_csv)
    if y_true_bin:
        metrics = binary_metrics(y_true_bin, y_pred_bin)
        metrics.update({
            "status": "ok",
            "task": "fall_detection",
            "level": "semantic_fall_binary_adapter",
            "semantic_adapter": "le2i_fall",
            "calls_legacy_inference": False,
            "boundary_note": "If this cap is fall_or_safety_event, the hub is emitting a final or near-final app decision.",
        })
    else:
        metrics = {"status": "skipped", "task": "fall_detection", "level": "semantic_fall_binary_adapter", "error": "No truth labels found."}
    write_json(metrics, metrics_json)
    return {
        "status": "ok",
        "inference": {"semantic_adapter": "le2i_fall", "returncode": 0, "dry_run": dry_run, "calls_legacy_inference": False},
        "output_csv": str(output_csv),
        "metrics_json": str(metrics_json),
        "metrics": metrics,
    }
