"""Semantic utility adapter for YouHome ADL outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import (
    json_paths_for_semantic_row,
    load_semantic_json,
    manifest_truth_label,
    multiclass_metrics,
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
    y_true_cls: List[str] = []
    y_pred_cls: List[str] = []

    if not dry_run:
        for i, r in enumerate(rows):
            sample_id = str(r.get("sample_id") or i)
            truth = manifest_truth_label(r, "adl_recognition")
            json_paths = json_paths_for_semantic_row(r)
            labels: List[str] = []
            confidence = 0.0
            for jp in json_paths[:64]:
                obj = load_semantic_json(jp)
                labs, _count, _occ, conf = semantic_labels_and_count(obj)
                labels.extend(labs)
                confidence = max(confidence, conf)
            pred_label = labels[0] if labels else "none"
            pred_rows.append({
                "sample_id": sample_id,
                "true_label": truth,
                "pred_label": pred_label,
                "confidence": confidence,
                "semantic_labels": ";".join(labels[:32]),
                "semantic_json_count": len(json_paths),
                "split": r.get("split", ""),
            })
            if truth:
                y_true_cls.append(truth)
                y_pred_cls.append(pred_label)

    write_csv_rows(pred_rows, output_csv)
    if y_true_cls:
        metrics = multiclass_metrics(y_true_cls, y_pred_cls)
        metrics.update({
            "task": "adl_recognition",
            "level": "semantic_adl_label_adapter",
            "semantic_adapter": "youhome_adl",
            "calls_legacy_inference": False,
            "warning": "Semantic adapter metrics are proxy metrics unless the app contract declares this representation as the final accepted interface.",
        })
    else:
        metrics = {"status": "skipped", "task": "adl_recognition", "level": "semantic_adl_label_adapter", "error": "No truth labels found."}
    write_json(metrics, metrics_json)
    return {
        "status": "ok",
        "inference": {"semantic_adapter": "youhome_adl", "returncode": 0, "dry_run": dry_run, "calls_legacy_inference": False},
        "output_csv": str(output_csv),
        "metrics_json": str(metrics_json),
        "metrics": metrics,
    }
