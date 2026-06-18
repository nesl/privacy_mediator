from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from baselines.task.common.manifest import load_label_map
from baselines.task.common.metrics import classification_metrics, write_json
from baselines.task.common.torch_utils import get_device, load_checkpoint
from baselines.task.le2i_fall.dataset import Le2iPoseWindowDataset, PoseSequenceDataset
from baselines.task.le2i_fall.model import PoseGRUFallClassifier


def _load_model(args, label_map: dict[str, int]):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}
    sequence_len = int(meta.get("sequence_len", args.sequence_len or 64))
    input_dim = int(meta.get("input_dim", 17 * 3))
    hidden_dim = int(meta.get("hidden_dim", args.hidden_dim or 128))
    model = PoseGRUFallClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=len(label_map))
    load_checkpoint(args.checkpoint, model)
    return model, meta, sequence_len


def infer_video_level(args, model, label_map, sequence_len, device):
    ds = PoseSequenceDataset(args.manifest, args.data_root, split=args.split, label_map=label_map, sequence_len=sequence_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    inv = {v: k for k, v in label_map.items()}
    rows, y_true, y_pred = [], [], []
    with torch.no_grad():
        offset = 0
        for x, y in loader:
            x = x.to(device).float()
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu()
            pred = probs.argmax(dim=1)
            for j in range(len(pred)):
                src_row = ds.df.iloc[offset + j]
                rows.append({
                    "sample_id": src_row.get("sample_id", offset + j),
                    "true_label": inv[int(y[j])],
                    "pred_label": inv[int(pred[j])],
                    "confidence": float(probs[j, int(pred[j])]),
                    "fall_probability": float(probs[j, label_map.get("fall", int(pred[j]))]),
                })
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
            offset += len(pred)
    return pd.DataFrame(rows), y_true, y_pred, None


def infer_window_level(args, model, label_map, sequence_len, meta, device):
    include_unlabeled = args.split.lower() == "unlabeled"
    ds = Le2iPoseWindowDataset(
        args.manifest,
        args.data_root,
        split=args.split,
        label_map=label_map,
        sequence_len=sequence_len,
        window_stride=args.window_stride or int(meta.get("window_stride", 32)),
        extraction_stride=args.extraction_stride if args.extraction_stride is not None else meta.get("extraction_stride", None),
        min_fall_overlap=args.min_fall_overlap if args.min_fall_overlap is not None else float(meta.get("min_fall_overlap", 0.10)),
        include_unlabeled=include_unlabeled,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    inv = {v: k for k, v in label_map.items()}
    fall_idx = label_map.get("fall", 1 if len(label_map) > 1 else 0)
    rows, y_true, y_pred = [], [], []
    with torch.no_grad():
        offset = 0
        for x, y in loader:
            x = x.to(device).float()
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu()
            pred = probs.argmax(dim=1)
            for j in range(len(pred)):
                md = ds.get_window_metadata(offset + j)
                true_label = md["label"]
                pred_label = inv[int(pred[j])]
                rows.append({
                    **md,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": float(probs[j, int(pred[j])]),
                    "fall_probability": float(probs[j, fall_idx]),
                })
                if true_label != "unlabeled":
                    y_true.append(label_map[true_label])
                    y_pred.append(int(pred[j]))
            offset += len(pred)

    win_df = pd.DataFrame(rows)
    video_rows = []
    if not win_df.empty:
        for sample_id, group in win_df.groupby("sample_id"):
            fall_prob = float(group["fall_probability"].max())
            pred_label = "fall" if fall_prob >= args.video_threshold else "nonfall"
            true_vals = [x for x in group["true_label"].unique().tolist() if x != "unlabeled"]
            true_label = "fall" if "fall" in true_vals else (true_vals[0] if true_vals else "unlabeled")
            video_rows.append({
                "sample_id": sample_id,
                "true_label": true_label,
                "pred_label": pred_label,
                "fall_probability_max": fall_prob,
                "fall_probability_mean": float(group["fall_probability"].mean()),
                "num_windows": int(len(group)),
            })
    return win_df, y_true, y_pred, pd.DataFrame(video_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference/evaluation for Le2i fall baseline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--sample-mode", choices=["window", "video"], default=None, help="Defaults to checkpoint metadata.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-csv", required=True, help="Window-level or video-level prediction CSV.")
    parser.add_argument("--video-output-csv", default=None, help="For window mode, aggregated video-level prediction CSV.")
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--sequence-len", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--extraction-stride", type=int, default=None)
    parser.add_argument("--min-fall-overlap", type=float, default=None)
    parser.add_argument("--video-threshold", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    label_map = load_label_map(args.label_map)
    model, meta, sequence_len = _load_model(args, label_map)
    sample_mode = args.sample_mode or str(meta.get("sample_mode", "window"))
    device = get_device(args.device)
    model.to(device).eval()

    if sample_mode == "window":
        pred_df, y_true, y_pred, video_df = infer_window_level(args, model, label_map, sequence_len, meta, device)
    else:
        pred_df, y_true, y_pred, video_df = infer_video_level(args, model, label_map, sequence_len, device)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.output_csv, index=False)
    metrics = classification_metrics(y_true, y_pred, labels=list(range(len(label_map)))) if y_true else {"n": 0}
    if video_df is not None:
        video_path = args.video_output_csv or str(Path(args.output_csv).with_name(Path(args.output_csv).stem + "_video.csv"))
        video_df.to_csv(video_path, index=False)
        if not video_df.empty and (video_df["true_label"] != "unlabeled").any():
            ytv, ypv = [], []
            for _, r in video_df[video_df["true_label"] != "unlabeled"].iterrows():
                ytv.append(label_map[str(r["true_label"])])
                ypv.append(label_map[str(r["pred_label"])])
            metrics["video_level"] = classification_metrics(ytv, ypv, labels=list(range(len(label_map))))
    if args.metrics_json:
        write_json(metrics, args.metrics_json)
    print(f"Wrote {args.output_csv}")
    if args.metrics_json:
        print(f"Wrote {args.metrics_json}")


if __name__ == "__main__":
    main()
