"""Evaluate ChokePoint visitor-monitoring predictions.

This evaluator reports both frame-level and event-level metrics.  By default it
uses two post-processing choices that better match the visitor-monitoring task:

1. XML label dilation: the XML marks frames with person/eye annotations, not the
   entire interval during which a visitor is visible. We dilate positives by a
   small window to approximate threshold-level presence.
2. Prediction smoothing: isolated YOLO positives are removed and short gaps
   within a crossing are filled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.metrics import mean_absolute_error

from baselines.task.common.manifest import filter_split, load_manifest
from baselines.task.common.metrics import binary_event_metrics, write_json
from baselines.task.chokepoint_presence.postprocess import (
    binary_events_from_rows,
    dilate_labels,
    event_metrics,
    recompute_from_boxes,
    smooth_binary_predictions,
)


def _read_nonempty_csv(path: str, name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except EmptyDataError as e:
        raise SystemExit(f"{name} CSV is empty or has no header: {path}") from e
    if df.empty:
        raise SystemExit(f"{name} CSV has a header but zero rows: {path}")
    return df


def _filter_to_manifest_split(df: pd.DataFrame, manifest_path: str | None, split: str | None) -> pd.DataFrame:
    if not manifest_path or not split:
        return df
    manifest = filter_split(load_manifest(manifest_path), split)
    keep = set(manifest["sample_id"].astype(str))
    return df[df["sample_id"].astype(str).isin(keep)].reset_index(drop=True)


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_predictions(
    pred: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    pred_conf_threshold: float | None = None,
    min_box_area: float = 0.0,
    label_dilation_pre: int = 10,
    label_dilation_post: int = 10,
    min_positive_frames: int = 2,
    max_gap_frames: int = 15,
    event_max_gap_frames: int = 30,
    event_tolerance_frames: int = 30,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = pred.copy()
    labels = labels.copy()
    if pred.empty:
        raise ValueError("Predictions dataframe is empty")
    if labels.empty:
        raise ValueError("Labels dataframe is empty")
    pred["sample_id"] = pred["sample_id"].astype(str)
    labels["sample_id"] = labels["sample_id"].astype(str)

    pred = recompute_from_boxes(pred, conf_threshold=pred_conf_threshold, min_box_area=min_box_area)
    pred_raw = pred.copy()
    pred = smooth_binary_predictions(pred, min_positive_frames=min_positive_frames, max_gap_frames=max_gap_frames)

    raw_labels = labels.copy()
    labels = dilate_labels(labels, pre_frames=label_dilation_pre, post_frames=label_dilation_post)

    keys = ["sample_id", "frame_index"]
    merged = pred.merge(labels, on=keys, suffixes=("_pred", "_true"))
    if merged.empty:
        pred_samples = sorted(pred["sample_id"].astype(str).unique())[:5]
        label_samples = sorted(labels["sample_id"].astype(str).unique())[:5]
        raise ValueError(
            "No matching sample_id/frame_index rows between predictions and labels. "
            f"Prediction sample examples={pred_samples}; label sample examples={label_samples}. "
            "Check frame indexes, --stride, --manifest, and --split."
        )

    # Also compute metrics against raw XML face/eye annotations as a diagnostic.
    raw_merged = pred_raw.merge(raw_labels, on=keys, suffixes=("_pred", "_true"))

    y_true = merged["person_present_true"].astype(int)
    y_pred = merged["person_present_pred"].astype(int)
    metrics: dict = {
        "settings": {
            "pred_conf_threshold": pred_conf_threshold,
            "min_box_area": min_box_area,
            "label_dilation_pre": label_dilation_pre,
            "label_dilation_post": label_dilation_post,
            "min_positive_frames": min_positive_frames,
            "max_gap_frames": max_gap_frames,
            "event_max_gap_frames": event_max_gap_frames,
            "event_tolerance_frames": event_tolerance_frames,
        },
        "n_prediction_rows": int(len(pred)),
        "n_prediction_samples": int(pred["sample_id"].nunique()),
        "n_matched_rows": int(len(merged)),
        "n_matched_samples": int(merged["sample_id"].nunique()),
        "positive_label_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "predicted_positive_rate": float(y_pred.mean()) if len(y_pred) else 0.0,
        "presence": binary_event_metrics(y_true, y_pred),
    }

    if not raw_merged.empty:
        metrics["raw_xml_face_frame_presence"] = binary_event_metrics(
            raw_merged["person_present_true"].astype(int),
            raw_merged["person_present_pred"].astype(int),
        )
        metrics["raw_positive_label_rate"] = float(raw_merged["person_present_true"].astype(int).mean())

    if "person_count_true" in merged.columns and "person_count_pred" in merged.columns:
        metrics["count_mae"] = float(mean_absolute_error(merged["person_count_true"], merged["person_count_pred"]))
        metrics["count_exact_match"] = float((merged["person_count_true"].astype(int) == merged["person_count_pred"].astype(int)).mean())

    neg = merged[merged["person_present_true"].astype(int) == 0]
    pos = merged[merged["person_present_true"].astype(int) == 1]
    metrics["negative_frames"] = int(len(neg))
    metrics["positive_frames"] = int(len(pos))
    metrics["false_positive_rate_on_negative_frames"] = float(neg["person_present_pred"].astype(int).mean()) if len(neg) else 0.0
    metrics["recall_on_positive_frames"] = float(pos["person_present_pred"].astype(int).mean()) if len(pos) else 0.0

    true_events = binary_events_from_rows(labels, max_gap_frames=event_max_gap_frames)
    pred_events = binary_events_from_rows(pred, max_gap_frames=event_max_gap_frames)
    metrics["events"] = event_metrics(true_events, pred_events, tolerance_frames=event_tolerance_frames).as_dict()
    return metrics, merged, true_events, pred_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", default=None, help="Frame labels CSV from make_manifest")
    parser.add_argument("--manifest", default=None, help="Optional sequence manifest for split filtering")
    parser.add_argument("--split", default=None, help="Optional split filter, e.g. test")
    parser.add_argument("--background-prefix", type=int, default=None, help="Treat frame_index < N as negative if labels are not provided")
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--matched-csv", default=None, help="Optional merged prediction/label rows")
    parser.add_argument("--true-events-csv", default=None)
    parser.add_argument("--pred-events-csv", default=None)
    parser.add_argument("--config-json", default=None, help="Optional config emitted by tune_postprocess.py")
    parser.add_argument("--pred-conf-threshold", type=float, default=None, help="Post-hoc confidence threshold using boxes_json")
    parser.add_argument("--min-box-area", type=float, default=0.0)
    parser.add_argument("--label-dilation-pre", type=int, default=10)
    parser.add_argument("--label-dilation-post", type=int, default=10)
    parser.add_argument("--min-positive-frames", type=int, default=2)
    parser.add_argument("--max-gap-frames", type=int, default=15)
    parser.add_argument("--event-max-gap-frames", type=int, default=30)
    parser.add_argument("--event-tolerance-frames", type=int, default=30)
    args = parser.parse_args()

    cfg = _load_config(args.config_json)
    # Explicit CLI args win only when different from parser defaults is hard to
    # know, so config is intended for normal use without overriding these flags.
    pred_conf_threshold = cfg.get("pred_conf_threshold", args.pred_conf_threshold)
    min_box_area = cfg.get("min_box_area", args.min_box_area)
    label_dilation_pre = cfg.get("label_dilation_pre", args.label_dilation_pre)
    label_dilation_post = cfg.get("label_dilation_post", args.label_dilation_post)
    min_positive_frames = cfg.get("min_positive_frames", args.min_positive_frames)
    max_gap_frames = cfg.get("max_gap_frames", args.max_gap_frames)
    event_max_gap_frames = cfg.get("event_max_gap_frames", args.event_max_gap_frames)
    event_tolerance_frames = cfg.get("event_tolerance_frames", args.event_tolerance_frames)

    pred = _read_nonempty_csv(args.predictions, "Predictions")
    pred["sample_id"] = pred["sample_id"].astype(str)
    pred = _filter_to_manifest_split(pred, args.manifest, args.split)
    if pred.empty:
        raise SystemExit(f"No prediction rows remain after filtering to split={args.split!r}")
    if args.labels:
        labels = _read_nonempty_csv(args.labels, "Labels")
        labels["sample_id"] = labels["sample_id"].astype(str)
        labels = _filter_to_manifest_split(labels, args.manifest, args.split)
        if labels.empty:
            raise SystemExit(f"No label rows remain after filtering to split={args.split!r}")
        metrics, merged, true_events, pred_events = evaluate_predictions(
            pred,
            labels,
            pred_conf_threshold=pred_conf_threshold,
            min_box_area=min_box_area,
            label_dilation_pre=label_dilation_pre,
            label_dilation_post=label_dilation_post,
            min_positive_frames=min_positive_frames,
            max_gap_frames=max_gap_frames,
            event_max_gap_frames=event_max_gap_frames,
            event_tolerance_frames=event_tolerance_frames,
        )
        if args.matched_csv:
            Path(args.matched_csv).parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(args.matched_csv, index=False)
        if args.true_events_csv:
            Path(args.true_events_csv).parent.mkdir(parents=True, exist_ok=True)
            true_events.to_csv(args.true_events_csv, index=False)
        if args.pred_events_csv:
            Path(args.pred_events_csv).parent.mkdir(parents=True, exist_ok=True)
            pred_events.to_csv(args.pred_events_csv, index=False)
    elif args.background_prefix is not None:
        bg = pred[pred["frame_index"] < args.background_prefix]
        metrics = {
            "n_prediction_rows": int(len(pred)),
            "n_prediction_samples": int(pred["sample_id"].nunique()) if "sample_id" in pred.columns else 0,
            "background_frames": int(len(bg)),
            "background_false_positive_rate": float(bg["person_present"].astype(int).mean()) if len(bg) else 0.0,
            "background_false_positives": int(bg["person_present"].astype(int).sum()),
        }
    else:
        raise ValueError("Provide --labels or --background-prefix")

    write_json(metrics, args.metrics_json)
    print(metrics)


if __name__ == "__main__":
    main()
