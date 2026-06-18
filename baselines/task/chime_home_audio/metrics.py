from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, average_precision_score, roc_auc_score

from .labels import LABELS


def compute_multilabel_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "n": int(y_true.shape[0]),
        "threshold": float(threshold),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "label_cardinality_true": float(y_true.sum(axis=1).mean()),
        "label_cardinality_pred": float(y_pred.sum(axis=1).mean()),
    }
    per_label = {}
    for i, lab in enumerate(LABELS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        ys = y_score[:, i]
        d = {
            "support": int(yt.sum()),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
        }
        try:
            if len(np.unique(yt)) > 1:
                d["average_precision"] = float(average_precision_score(yt, ys))
                d["roc_auc"] = float(roc_auc_score(yt, ys))
        except Exception:
            pass
        per_label[lab] = d
    metrics["per_label"] = per_label
    try:
        # average_precision_score supports multilabel matrices.
        metrics["macro_average_precision"] = float(average_precision_score(y_true, y_score, average="macro"))
        metrics["micro_average_precision"] = float(average_precision_score(y_true, y_score, average="micro"))
    except Exception:
        pass
    return metrics


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
