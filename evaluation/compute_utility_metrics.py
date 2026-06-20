#!/usr/bin/env python3
"""Post-process utility evaluator outputs into task performance metrics.

This script is intentionally independent of evaluation/evaluate_utility.py so you
can run it after a utility run.  It currently has a robust ChokePoint visitor
presence evaluator and a generic binary-classification fallback for other tasks
when prediction files already contain truth/prediction columns.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

TRUE_COL_CANDIDATES = ["y_true", "true", "target", "label", "gt", "ground_truth", "person_present_gt"]
PRED_COL_CANDIDATES = ["y_pred", "pred", "prediction", "pred_label", "person_present", "person_present_pred"]
PERSON_LABELS = {"person", "pedestrian", "human", "visitor"}


def _is_blank(x: Any) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except TypeError:
        pass
    return str(x).strip() == ""


def _as_int(x: Any, default: int | None = None) -> int | None:
    if _is_blank(x):
        return default
    text = str(x).strip()
    stem = Path(text).stem
    for candidate in (text, stem):
        try:
            return int(float(candidate))
        except Exception:
            continue
    m = re.search(r"(\d+)", stem)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return default
    return default


def _candidate_paths(value: Any, roots: Iterable[Path | None]) -> list[Path]:
    if _is_blank(value):
        return []
    raw = Path(str(value))
    out: list[Path] = []
    if raw.is_absolute():
        out.append(raw)
    else:
        out.append(raw)
        for root in roots:
            if root is not None:
                out.append(root / raw)
    dedup: list[Path] = []
    seen: set[str] = set()
    for p in out:
        key = str(p)
        if key not in seen:
            dedup.append(p)
            seen.add(key)
    return dedup


def _infer_data_root(source_manifest: Path | None, explicit: str | None = None) -> Path | None:
    if explicit:
        return Path(explicit)
    if source_manifest is None:
        return None
    # data/chokepoint/chokepoint_manifest.csv -> data/chokepoint
    return source_manifest.parent


def _resolve_path(value: Any, source_manifest: Path | None = None, data_root: Path | None = None) -> Path | None:
    roots = [data_root, source_manifest.parent if source_manifest is not None else None, Path.cwd()]
    for p in _candidate_paths(value, roots):
        if p.exists():
            return p
    return None


def parse_chokepoint_xml(xml_path: Path) -> dict[int, int]:
    """Return frame_index -> person_present from common video-annotation XMLs.

    Supports several common structures: CVAT track/image XML, frame/object XML,
    and generic frame-indexed object/box annotations. The goal is conservative
    binary presence evaluation, not box IoU.
    """
    root = ET.parse(xml_path).getroot()
    gt: dict[int, int] = {}

    # CVAT interpolation format: <track label="person"><box frame="123" outside="0" .../></track>
    for track in root.iter():
        if track.tag.lower().endswith("track"):
            label = str(track.attrib.get("label", "")).strip().lower()
            label_is_person = (not label) or label in PERSON_LABELS
            for box in track.iter():
                if not box.tag.lower().endswith("box"):
                    continue
                frame = _as_int(box.attrib.get("frame"))
                if frame is None:
                    continue
                outside = str(box.attrib.get("outside", "0")).strip().lower() in {"1", "true", "yes"}
                if not outside and label_is_person:
                    gt[frame] = 1
                else:
                    gt.setdefault(frame, 0)

    # CVAT image format: <image id="123" name="00000123.jpg"><box label="person" .../></image>
    for image in root.iter():
        if not image.tag.lower().endswith("image"):
            continue
        frame = _as_int(image.attrib.get("id"), None)
        if frame is None:
            frame = _as_int(image.attrib.get("name"), None)
        if frame is None:
            continue
        present = 0
        for child in image.iter():
            tag = child.tag.lower().split("}")[-1]
            if tag in {"box", "polygon", "points", "object"}:
                label = str(child.attrib.get("label", child.attrib.get("name", ""))).strip().lower()
                if (not label) or label in PERSON_LABELS:
                    outside = str(child.attrib.get("outside", "0")).strip().lower() in {"1", "true", "yes"}
                    if not outside:
                        present = 1
                        break
        gt[frame] = max(gt.get(frame, 0), present)

    # Generic ChokePoint-ish frame format: <frame number="..."> ... objects ... </frame>
    for frame_el in root.iter():
        tag = frame_el.tag.lower().split("}")[-1]
        if tag not in {"frame", "img", "image"}:
            continue
        frame = None
        for key in ("number", "num", "frame", "id", "index", "name"):
            frame = _as_int(frame_el.attrib.get(key), None)
            if frame is not None:
                break
        if frame is None:
            continue
        present = 0
        for child in frame_el.iter():
            ctag = child.tag.lower().split("}")[-1]
            if ctag in {"object", "person", "box", "bndbox", "bbox"}:
                label = str(child.attrib.get("label", child.attrib.get("name", child.tag))).strip().lower().split("}")[-1]
                if label in PERSON_LABELS or ctag in {"person", "object", "box", "bndbox", "bbox"}:
                    present = 1
                    break
        gt[frame] = max(gt.get(frame, 0), present)

    return gt


def _binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    assert len(y_true) == len(y_pred)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    beta2 = 2.0
    f2 = (1 + beta2**2) * precision * recall / (beta2**2 * precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "n": len(y_true),
        "support_pos": sum(y_true),
        "support_neg": len(y_true) - sum(y_true),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "accuracy": accuracy,
    }


def evaluate_chokepoint(row: pd.Series, data_root: str | None = None) -> dict[str, Any]:
    pred_path = Path(str(row["downstream_output_csv"]))
    pred = pd.read_csv(pred_path)
    if pred.empty:
        return {"metric_status": "error", "metric_error": "prediction CSV is empty"}
    if "person_present" not in pred.columns or "sample_id" not in pred.columns or "frame_index" not in pred.columns:
        return {"metric_status": "error", "metric_error": "prediction CSV lacks sample_id/frame_index/person_present columns"}

    source_manifest = Path(str(row.get("source_manifest"))) if not _is_blank(row.get("source_manifest")) else None
    root = _infer_data_root(source_manifest, data_root)
    if source_manifest is None or not source_manifest.exists():
        return {"metric_status": "error", "metric_error": f"source manifest not found: {source_manifest}"}
    manifest = pd.read_csv(source_manifest)
    if "split" in manifest.columns and not _is_blank(row.get("task")):
        # Keep same split as the predictions if available.
        split_values = set(str(x) for x in pred.get("split", pd.Series(dtype=str)).dropna().unique())
        if len(split_values) == 1:
            manifest = manifest[manifest["split"].astype(str) == next(iter(split_values))]
    if "sample_id" not in manifest.columns or "xml_path" not in manifest.columns:
        return {"metric_status": "error", "metric_error": "source manifest lacks sample_id or xml_path columns"}

    pred_samples = set(str(x) for x in pred["sample_id"].astype(str).unique())
    manifest = manifest[manifest["sample_id"].astype(str).isin(pred_samples)]

    y_true: list[int] = []
    y_pred: list[int] = []
    missing_xml: list[str] = []
    empty_xml: list[str] = []

    gt_by_sample: dict[str, dict[int, int]] = {}
    for _, mrow in manifest.iterrows():
        sample_id = str(mrow["sample_id"])
        xml_path = _resolve_path(mrow.get("xml_path"), source_manifest=source_manifest, data_root=root)
        if xml_path is None:
            missing_xml.append(sample_id)
            continue
        try:
            gt_map = parse_chokepoint_xml(xml_path)
        except Exception as e:
            return {"metric_status": "error", "metric_error": f"failed parsing XML for sample {sample_id}: {e}"}
        if not gt_map:
            empty_xml.append(sample_id)
        gt_by_sample[sample_id] = gt_map

    for _, prow in pred.iterrows():
        sample_id = str(prow["sample_id"])
        frame = _as_int(prow["frame_index"])
        if frame is None:
            continue
        gt_map = gt_by_sample.get(sample_id)
        if not gt_map:
            continue
        y_true.append(int(gt_map.get(frame, 0)))
        y_pred.append(1 if int(prow["person_present"]) > 0 else 0)

    if not y_true:
        return {
            "metric_status": "error",
            "metric_error": "no prediction rows could be aligned to XML ground truth",
            "missing_xml_samples": missing_xml[:10],
            "empty_xml_samples": empty_xml[:10],
            "n_missing_xml_samples": len(missing_xml),
            "n_empty_xml_samples": len(empty_xml),
        }
    out = _binary_metrics(y_true, y_pred)
    out.update({
        "metric_status": "ok",
        "metric_task": "visitor_presence_detection",
        "metric_level": "frame_binary_presence",
        "n_prediction_rows": int(len(pred)),
        "n_aligned_rows": int(len(y_true)),
        "n_missing_xml_samples": len(missing_xml),
        "n_empty_xml_samples": len(empty_xml),
    })
    return out


def evaluate_generic_binary(row: pd.Series) -> dict[str, Any]:
    pred_path = Path(str(row["downstream_output_csv"]))
    df = pd.read_csv(pred_path)
    true_col = next((c for c in TRUE_COL_CANDIDATES if c in df.columns), None)
    pred_col = next((c for c in PRED_COL_CANDIDATES if c in df.columns and c != true_col), None)
    if true_col is None or pred_col is None:
        return {"metric_status": "skipped", "metric_error": "no generic true/pred columns found"}
    y_true = [1 if int(x) > 0 else 0 for x in df[true_col].fillna(0)]
    y_pred = [1 if int(x) > 0 else 0 for x in df[pred_col].fillna(0)]
    out = _binary_metrics(y_true, y_pred)
    out.update({"metric_status": "ok", "metric_task": row.get("task", "unknown"), "metric_level": "generic_binary", "true_col": true_col, "pred_col": pred_col})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="runs/utility_eval/utility_summary.csv")
    ap.add_argument("--out-csv", default="runs/utility_eval/utility_metrics_summary.csv")
    ap.add_argument("--out-json", default="runs/utility_eval/utility_metrics.json")
    ap.add_argument("--chokepoint-data-root", default=None)
    args = ap.parse_args()

    summary = pd.read_csv(args.summary)
    records: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        base = row.to_dict()
        if str(row.get("status")) != "ok" or str(row.get("downstream_status")) != "ok" or _is_blank(row.get("downstream_output_csv")):
            metrics = {"metric_status": "skipped", "metric_error": "method or downstream status is not ok"}
        elif not Path(str(row.get("downstream_output_csv"))).exists():
            metrics = {"metric_status": "skipped", "metric_error": "prediction CSV does not exist"}
        elif str(row.get("task")) == "visitor_presence_detection":
            metrics = evaluate_chokepoint(row, data_root=args.chokepoint_data_root)
        else:
            metrics = evaluate_generic_binary(row)
        rec = {**base, **metrics}
        # Keep JSON clean of NaN values.
        for k, v in list(rec.items()):
            if isinstance(v, float) and math.isnan(v):
                rec[k] = None
        records.append(rec)

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")

    compact_cols = [
        "scenario_id", "task", "method_id", "status", "downstream_status",
        "metric_status", "n", "precision", "recall", "f1", "f2", "accuracy", "tp", "fp", "fn", "tn",
        "metric_error",
    ]
    available = [c for c in compact_cols if c in pd.DataFrame(records).columns]
    print(pd.DataFrame(records)[available].to_string(index=False))
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
