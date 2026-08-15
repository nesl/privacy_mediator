"""Common utilities for SmartPriv flexible semantic utility adapters.

These helpers intentionally do not import the legacy downstream inference
modules.  They operate on the semantic JSON artifacts materialized by
``evaluation.evaluate_utility`` and produce task-level prediction CSV/metrics.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


def read_csv_rows(path: str | Path, max_rows: Optional[int] = None) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = []
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(dict(row))
        return rows, list(reader.fieldnames or [])


def write_csv_rows(rows: Sequence[Dict[str, Any]], path: str | Path, fieldnames: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in fields:
                    fields.append(k)
    else:
        fields = list(fieldnames)
        for r in rows:
            for k in r.keys():
                if k not in fields:
                    fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def json_paths_for_semantic_row(row: Dict[str, Any]) -> List[Path]:
    paths: List[Path] = []
    for key in ["preprocessed_json_path", "semantic_json_path"]:
        val = row.get(key)
        if val and Path(str(val)).exists():
            paths.append(Path(str(val)))
    for key in ["preprocessed_json_dir", "semantic_dir", "frame_dir"]:
        val = row.get(key)
        p = Path(str(val)) if val else None
        if p and p.exists() and p.is_dir():
            paths.extend(sorted(p.glob("*.json")))
    return paths


def load_semantic_json(path: Path) -> Dict[str, Any]:
    try:
        obj = load_json(path)
    except Exception:
        return {"_error": f"could_not_read:{path}"}
    if isinstance(obj, dict):
        return obj
    return {"data": obj}


def semantic_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    data = obj.get("data") if isinstance(obj.get("data"), dict) else None
    if data is not None:
        return data
    for key in ["payload", "value", "result"]:
        if isinstance(obj.get(key), dict):
            return obj[key]
    return obj


def semantic_labels_and_count(obj: Dict[str, Any]) -> Tuple[List[str], int, bool, float]:
    data = semantic_payload(obj)
    labels: List[str] = []
    confidence = 0.0
    count = 0
    occupied = False

    if isinstance(data.get("labels"), list):
        for l in data.get("labels") or []:
            if isinstance(l, dict):
                label = str(l.get("label") or l.get("class") or l.get("event_type") or l.get("activity") or "").strip()
                if label:
                    labels.append(label)
                try:
                    confidence = max(confidence, float(l.get("confidence", 0.0)))
                except Exception:
                    pass
            elif l:
                labels.append(str(l))

    if isinstance(data.get("detections"), list):
        detections = data.get("detections") or []
        for d in detections:
            if isinstance(d, dict):
                label = str(d.get("label") or d.get("class") or "").strip()
                if label:
                    labels.append(label)
                try:
                    confidence = max(confidence, float(d.get("confidence", 0.0)))
                except Exception:
                    pass
        count = sum(
            1 for d in detections
            if isinstance(d, dict) and str(d.get("label") or d.get("class") or "").lower() in {"person", "face", "human", "visitor"}
        )

    cbl = data.get("count_by_label")
    if isinstance(cbl, dict):
        for k, v in cbl.items():
            if k:
                try:
                    n = int(float(v or 0))
                except Exception:
                    n = 1
                labels.extend([str(k)] * max(1, n))
        for k in ["person", "face", "human", "visitor"]:
            if k in cbl:
                try:
                    count += int(float(cbl.get(k) or 0))
                except Exception:
                    pass

    for key in ["top_label", "label", "event_type", "activity", "class", "pred_label"]:
        if data.get(key):
            labels.append(str(data.get(key)))
    for key in ["count", "occupancy_count", "person_count"]:
        if data.get(key) not in (None, ""):
            try:
                count = max(count, int(float(data.get(key))))
            except Exception:
                pass
    if data.get("occupied") not in (None, ""):
        occupied = bool(data.get("occupied"))
        count = max(count, int(occupied))
    if not occupied:
        occupied = count > 0 or any(
            str(l).lower() in {"person", "face", "human", "visitor", "room_occupied", "occupied", "presence"}
            for l in labels
        )
    return labels, count, occupied, confidence


def normalize_label_text(label: Any) -> str:
    text = str(label or "").strip().lower()
    text = re.sub(r"[^a-z0-9_.]+", "_", text)
    return text.strip("_")


def manifest_truth_label(row: Dict[str, Any], task: str) -> str:
    candidates = [
        "label", "labels", "class", "target", "activity", "activity_label", "adl_label",
        "event", "event_type", "sound_label", "y_true", "gt", "ground_truth", "fall_label",
    ]
    for c in candidates:
        val = row.get(c)
        if val not in (None, ""):
            return str(val)
    if task == "fall_detection":
        text = " ".join(str(v) for v in row.values()).lower()
        if "fall" in text:
            return "fall"
        if "non" in text or "normal" in text:
            return "nonfall"
    return ""


def label_to_binary(label: str, positive_terms: Sequence[str]) -> int:
    text = str(label or "").strip().lower()
    if text in {"1", "true", "yes", "positive", "pos"}:
        return 1
    if text in {"0", "false", "no", "negative", "neg", "none", "normal", "nonfall", "non_fall"}:
        return 0
    return 1 if any(term in text for term in positive_terms) else 0


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    yt = [int(x) for x in y_true]
    yp = [int(x) for x in y_pred]
    n = len(yt)
    tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) else 0.0
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
    }


def multiclass_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n if n else 0.0
    f1s: List[float] = []
    per_label: Dict[str, Any] = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        per_label[lab] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}
    return {
        "status": "ok",
        "n": n,
        "accuracy": accuracy,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "labels": labels,
        "per_label": per_label,
    }


def multilabel_binary_metrics(y_true: List[List[int]], y_pred: List[List[int]], labels: Sequence[str]) -> Dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {"status": "skipped", "error": "no multilabel rows", "n": 0}
    L = len(labels)
    tp = fp = tn = fn = 0
    per_label: Dict[str, Any] = {}
    f1s: List[float] = []
    for j, lab in enumerate(labels):
        lt = [int(row[j]) for row in y_true]
        lp = [int(row[j]) for row in y_pred]
        ltp = sum(1 for t, p in zip(lt, lp) if t == 1 and p == 1)
        ltn = sum(1 for t, p in zip(lt, lp) if t == 0 and p == 0)
        lfp = sum(1 for t, p in zip(lt, lp) if t == 0 and p == 1)
        lfn = sum(1 for t, p in zip(lt, lp) if t == 1 and p == 0)
        tp += ltp; tn += ltn; fp += lfp; fn += lfn
        prec = ltp / (ltp + lfp) if (ltp + lfp) else 0.0
        rec = ltp / (ltp + lfn) if (ltp + lfn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        per_label[lab] = {"tp": ltp, "fp": lfp, "tn": ltn, "fn": lfn, "precision": prec, "recall": rec, "f1": f1}
    micro_precision = tp / (tp + fp) if (tp + fp) else 0.0
    micro_recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0
    exact_match = sum(1 for t, p in zip(y_true, y_pred) if list(map(int, t)) == list(map(int, p))) / n
    hamming_accuracy = (tp + tn) / (n * L) if L else 0.0
    return {
        "status": "ok",
        "n": n,
        "num_labels": L,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "hamming_accuracy": hamming_accuracy,
        "exact_match": exact_match,
        "per_label": per_label,
    }
