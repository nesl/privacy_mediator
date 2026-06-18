"""Metrics shared by the baseline scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int] | None = None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0,
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if len(y_true) else 0.0,
    }
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    metrics["per_class"] = {
        str(label): {
            "precision": float(pi),
            "recall": float(ri),
            "f1": float(fi),
            "support": int(si),
        }
        for label, pi, ri, fi, si in zip(labels if labels is not None else sorted(set(y_true) | set(y_pred)), p, r, f1, support)
    }
    return metrics


def binary_event_metrics(y_true: Sequence[int], y_pred: Sequence[int], beta: float = 2.0) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta2 = beta * beta
    fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall) if (beta2 * precision + recall) else 0.0
    return {
        "n": int(len(y_true)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        f"f{beta:g}": float(fbeta),
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) else 0.0,
    }


def write_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_classification_report(y_true, y_pred, target_names, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    path.write_text(report, encoding="utf-8")
