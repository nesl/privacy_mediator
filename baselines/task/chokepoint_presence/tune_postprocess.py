"""Tune ChokePoint visitor-monitoring post-processing on train/val split.

This script does not train YOLO. It searches lightweight evaluation-time
parameters: confidence threshold, label dilation, and temporal smoothing. Use it
on train or val, then pass the emitted config to evaluate.py on test.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError
from tqdm import tqdm

from baselines.task.common.manifest import filter_split, load_manifest
from baselines.task.chokepoint_presence.evaluate import evaluate_predictions


def _parse_list(text: str, cast):
    return [cast(x) for x in str(text).split(",") if str(x).strip() != ""]


def _read_nonempty_csv(path: str, name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except EmptyDataError as e:
        raise SystemExit(
            f"{name} CSV is empty or has no header: {path}. "
            "Rerun infer_yolo.py and inspect its --error-log; no post-processing can be tuned without prediction rows."
        ) from e
    if df.empty:
        raise SystemExit(
            f"{name} CSV has a header but zero rows: {path}. "
            "Rerun infer_yolo.py and inspect its --error-log; no post-processing can be tuned without prediction rows."
        )
    return df


def _filter(df: pd.DataFrame, manifest_path: str | None, split: str | None) -> pd.DataFrame:
    if not manifest_path or not split:
        return df
    manifest = filter_split(load_manifest(manifest_path), split)
    keep = set(manifest["sample_id"].astype(str))
    return df[df["sample_id"].astype(str).isin(keep)].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--sweep-csv", default=None)
    parser.add_argument("--optimize", choices=["f1", "f2", "precision", "recall", "event_f1", "event_f2"], default="f2")
    parser.add_argument("--conf-thresholds", default="0.25,0.35,0.45,0.55,0.65")
    parser.add_argument("--label-dilations", default="0,5,10,15,30")
    parser.add_argument("--min-positive-frames", default="1,2,3")
    parser.add_argument("--max-gap-frames", default="0,10,15,30")
    parser.add_argument("--event-max-gap-frames", type=int, default=30)
    parser.add_argument("--event-tolerance-frames", type=int, default=30)
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar for the tuning sweep.")
    args = parser.parse_args()

    pred = _read_nonempty_csv(args.predictions, "Predictions")
    labels = _read_nonempty_csv(args.labels, "Labels")
    pred["sample_id"] = pred["sample_id"].astype(str)
    labels["sample_id"] = labels["sample_id"].astype(str)
    pred = _filter(pred, args.manifest, args.split)
    labels = _filter(labels, args.manifest, args.split)

    if pred.empty:
        raise SystemExit(f"No prediction rows remain after filtering to split={args.split!r}. Check --manifest/--split and the infer_yolo output.")
    if labels.empty:
        raise SystemExit(f"No label rows remain after filtering to split={args.split!r}. Check --manifest/--split and the labels CSV.")

    confs = _parse_list(args.conf_thresholds, float) if "boxes_json" in pred.columns else [None]
    dilations = _parse_list(args.label_dilations, int)
    min_runs = _parse_list(args.min_positive_frames, int)
    gaps = _parse_list(args.max_gap_frames, int)

    combos = list(itertools.product(confs, dilations, min_runs, gaps))
    iterator = combos if args.no_progress else tqdm(combos, desc="tune_postprocess", unit="config")

    rows = []
    best = None
    best_score = -1.0
    for conf, dil, min_run, gap in iterator:
        try:
            metrics, _, _, _ = evaluate_predictions(
                pred,
                labels,
                pred_conf_threshold=conf,
                min_box_area=0.0,
                label_dilation_pre=dil,
                label_dilation_post=dil,
                min_positive_frames=min_run,
                max_gap_frames=gap,
                event_max_gap_frames=args.event_max_gap_frames,
                event_tolerance_frames=args.event_tolerance_frames,
            )
        except Exception as e:
            rows.append({"error": str(e), "pred_conf_threshold": conf, "label_dilation": dil, "min_positive_frames": min_run, "max_gap_frames": gap})
            continue
        presence = metrics["presence"]
        events = metrics.get("events", {})
        scores = {
            "f1": presence.get("f1", 0.0),
            "f2": presence.get("f2", 0.0),
            "precision": presence.get("precision", 0.0),
            "recall": presence.get("recall", 0.0),
            "event_f1": events.get("f1", 0.0),
            "event_f2": events.get("f2", 0.0),
        }
        row = {
            "pred_conf_threshold": conf,
            "label_dilation_pre": dil,
            "label_dilation_post": dil,
            "min_positive_frames": min_run,
            "max_gap_frames": gap,
            "event_max_gap_frames": args.event_max_gap_frames,
            "event_tolerance_frames": args.event_tolerance_frames,
            "presence_precision": presence.get("precision", 0.0),
            "presence_recall": presence.get("recall", 0.0),
            "presence_f1": presence.get("f1", 0.0),
            "presence_f2": presence.get("f2", 0.0),
            "event_precision": events.get("precision", 0.0),
            "event_recall": events.get("recall", 0.0),
            "event_f1": events.get("f1", 0.0),
            "event_f2": events.get("f2", 0.0),
            "score": scores[args.optimize],
        }
        rows.append(row)
        if row["score"] > best_score:
            best_score = row["score"]
            best = row
            if not args.no_progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(best=f"{best_score:.4f}", conf=conf, dil=dil, min_run=min_run, gap=gap)

    if args.sweep_csv:
        Path(args.sweep_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.sweep_csv, index=False)

    if best is None:
        raise RuntimeError(
            "No valid configuration found. The sweep CSV contains the per-setting errors. "
            "Common causes: prediction frame indexes do not overlap labels, wrong --split, or predictions are from a different manifest."
        )

    config = {
        "pred_conf_threshold": best["pred_conf_threshold"],
        "min_box_area": 0.0,
        "label_dilation_pre": int(best["label_dilation_pre"]),
        "label_dilation_post": int(best["label_dilation_post"]),
        "min_positive_frames": int(best["min_positive_frames"]),
        "max_gap_frames": int(best["max_gap_frames"]),
        "event_max_gap_frames": int(best["event_max_gap_frames"]),
        "event_tolerance_frames": int(best["event_tolerance_frames"]),
        "optimized_on": args.split,
        "optimized_metric": args.optimize,
        "optimized_score": float(best["score"]),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_json).open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
