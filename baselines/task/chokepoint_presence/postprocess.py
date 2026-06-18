"""Post-processing helpers for the ChokePoint visitor-monitoring baseline.

The ChokePoint XML files annotate visible face/eye frames, while the utility task
for this project is threshold visitor presence.  These helpers keep that
separation explicit: labels can be dilated from face/eye frames into presence
intervals, and YOLO frame predictions can be smoothed into event-like outputs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventMetrics:
    n_true_events: int
    n_pred_events: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    f2: float

    def as_dict(self) -> dict:
        return {
            "n_true_events": self.n_true_events,
            "n_pred_events": self.n_pred_events,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "f2": self.f2,
        }


def _safe_load_boxes(value: object) -> list[dict]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _box_area(box: dict) -> float:
    xyxy = box.get("xyxy") or []
    if len(xyxy) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(x) for x in xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def recompute_from_boxes(
    pred: pd.DataFrame,
    conf_threshold: float | None = None,
    min_box_area: float = 0.0,
) -> pd.DataFrame:
    """Recompute person_count/person_present from boxes_json.

    This lets you run YOLO once with a permissive confidence such as 0.05, then
    sweep stricter thresholds without rerunning inference. If boxes_json is not
    present, the input columns are left unchanged.
    """
    pred = pred.copy()
    if "boxes_json" not in pred.columns:
        return pred
    counts: list[int] = []
    max_confs: list[float] = []
    for value in pred["boxes_json"]:
        boxes = []
        for b in _safe_load_boxes(value):
            conf = float(b.get("conf", 0.0))
            if conf_threshold is not None and conf < conf_threshold:
                continue
            if min_box_area > 0 and _box_area(b) < min_box_area:
                continue
            boxes.append(b)
        counts.append(len(boxes))
        max_confs.append(max([float(b.get("conf", 0.0)) for b in boxes], default=0.0))
    pred["person_count"] = counts
    pred["person_present"] = [int(c > 0) for c in counts]
    pred["max_person_conf"] = max_confs
    return pred


def dilate_labels(
    labels: pd.DataFrame,
    pre_frames: int = 30,
    post_frames: int = 30,
    label_col: str = "person_present",
    count_col: str = "person_count",
) -> pd.DataFrame:
    """Dilate XML face/eye annotations into visitor-presence labels.

    The raw XML label means "a person node/eye annotation exists on this frame".
    For visitor monitoring, a person may be present shortly before eyes are
    annotated or shortly after eyes leave the annotation region. Dilation makes
    this semantic conversion explicit while preserving raw columns.
    """
    labels = labels.copy()
    if label_col not in labels.columns:
        return labels
    raw_label_col = f"raw_{label_col}"
    raw_count_col = f"raw_{count_col}"
    if raw_label_col not in labels.columns:
        labels[raw_label_col] = labels[label_col].astype(int)
    if count_col in labels.columns and raw_count_col not in labels.columns:
        labels[raw_count_col] = labels[count_col].astype(int)
    if pre_frames <= 0 and post_frames <= 0:
        return labels

    out_groups: list[pd.DataFrame] = []
    for _, g in labels.groupby("sample_id", sort=False):
        g = g.sort_values("frame_index").copy()
        idx = g["frame_index"].astype(int).to_numpy()
        raw = g[raw_label_col].astype(int).to_numpy()
        positive_frames = idx[raw > 0]
        dilated = np.zeros(len(g), dtype=int)
        for p in positive_frames:
            dilated |= ((idx >= p - pre_frames) & (idx <= p + post_frames)).astype(int)
        g[label_col] = dilated.astype(int)
        if count_col in g.columns:
            # For dilated-only frames, count at least one visible visitor. Keep
            # true raw counts where available.
            counts = g[count_col].astype(int).to_numpy()
            counts = np.where(dilated > 0, np.maximum(counts, 1), 0)
            g[count_col] = counts.astype(int)
        out_groups.append(g)
    return pd.concat(out_groups, ignore_index=True) if out_groups else labels


def smooth_binary_predictions(
    pred: pd.DataFrame,
    min_positive_frames: int = 2,
    max_gap_frames: int = 15,
    present_col: str = "person_present",
    count_col: str = "person_count",
) -> pd.DataFrame:
    """Remove isolated YOLO spikes and fill small gaps within events.

    ``min_positive_frames`` is counted in sampled rows, not original video frames.
    ``max_gap_frames`` uses the original numeric frame_index spacing.
    """
    pred = pred.copy()
    if present_col not in pred.columns:
        return pred
    out_groups: list[pd.DataFrame] = []
    for _, g in pred.groupby("sample_id", sort=False):
        g = g.sort_values("frame_index").copy()
        y = g[present_col].astype(int).to_numpy()
        frames = g["frame_index"].astype(int).to_numpy()
        # Fill small negative gaps between positives.
        if max_gap_frames > 0 and y.sum() > 1:
            pos_idx = np.where(y > 0)[0]
            for a, b in zip(pos_idx[:-1], pos_idx[1:]):
                if frames[b] - frames[a] <= max_gap_frames:
                    y[a : b + 1] = 1
        # Remove short positive runs.
        if min_positive_frames > 1 and len(y):
            i = 0
            while i < len(y):
                if y[i] == 0:
                    i += 1
                    continue
                j = i
                while j + 1 < len(y) and y[j + 1] == 1:
                    j += 1
                if (j - i + 1) < min_positive_frames:
                    y[i : j + 1] = 0
                i = j + 1
        g[present_col] = y.astype(int)
        if count_col in g.columns:
            counts = g[count_col].astype(int).to_numpy()
            counts = np.where(y > 0, np.maximum(counts, 1), 0)
            g[count_col] = counts.astype(int)
        out_groups.append(g)
    return pd.concat(out_groups, ignore_index=True) if out_groups else pred


def binary_events_from_rows(
    df: pd.DataFrame,
    present_col: str = "person_present",
    count_col: str = "person_count",
    max_gap_frames: int = 30,
) -> pd.DataFrame:
    """Convert frame rows into presence intervals per sample."""
    rows: list[dict] = []
    if df.empty or present_col not in df.columns:
        return pd.DataFrame(columns=["sample_id", "event_start", "event_end", "duration_frames", "max_person_count"])
    for sample_id, g in df.groupby("sample_id", sort=False):
        g = g.sort_values("frame_index")
        pos = g[g[present_col].astype(int) > 0]
        if pos.empty:
            continue
        start = prev = int(pos.iloc[0]["frame_index"])
        max_count = int(pos.iloc[0][count_col]) if count_col in pos.columns else 1
        n_rows = 1
        for _, row in pos.iloc[1:].iterrows():
            idx = int(row["frame_index"])
            if idx <= prev + max_gap_frames:
                prev = idx
                n_rows += 1
                if count_col in pos.columns:
                    max_count = max(max_count, int(row[count_col]))
            else:
                rows.append({
                    "sample_id": sample_id,
                    "event_start": start,
                    "event_end": prev,
                    "duration_frames": prev - start + 1,
                    "positive_rows": n_rows,
                    "max_person_count": max_count,
                })
                start = prev = idx
                max_count = int(row[count_col]) if count_col in pos.columns else 1
                n_rows = 1
        rows.append({
            "sample_id": sample_id,
            "event_start": start,
            "event_end": prev,
            "duration_frames": prev - start + 1,
            "positive_rows": n_rows,
            "max_person_count": max_count,
        })
    return pd.DataFrame(rows)


def event_metrics(
    true_events: pd.DataFrame,
    pred_events: pd.DataFrame,
    tolerance_frames: int = 30,
) -> EventMetrics:
    """Greedily match predicted events to true events by interval overlap."""
    true_events = true_events.copy()
    pred_events = pred_events.copy()
    if true_events.empty:
        tp = 0
        fp = int(len(pred_events))
        fn = 0
    else:
        matched_true: set[int] = set()
        matched_pred: set[int] = set()
        true_events = true_events.reset_index(drop=True)
        pred_events = pred_events.reset_index(drop=True)
        for pi, prow in pred_events.iterrows():
            ps, pe = int(prow["event_start"]), int(prow["event_end"])
            sid = str(prow["sample_id"])
            candidates = true_events[true_events["sample_id"].astype(str) == sid]
            best_i = None
            best_overlap = -1
            for ti, trow in candidates.iterrows():
                if ti in matched_true:
                    continue
                ts = int(trow["event_start"]) - tolerance_frames
                te = int(trow["event_end"]) + tolerance_frames
                overlap = min(pe, te) - max(ps, ts) + 1
                if overlap > best_overlap and overlap > 0:
                    best_overlap = overlap
                    best_i = ti
            if best_i is not None:
                matched_true.add(int(best_i))
                matched_pred.add(int(pi))
        tp = len(matched_pred)
        fp = int(len(pred_events) - tp)
        fn = int(len(true_events) - len(matched_true))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta2 = 4.0
    f2 = (1 + beta2) * precision * recall / (beta2 * precision + recall) if (beta2 * precision + recall) else 0.0
    return EventMetrics(
        n_true_events=int(len(true_events)),
        n_pred_events=int(len(pred_events)),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        f2=float(f2),
    )
