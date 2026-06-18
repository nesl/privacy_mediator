from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ChimeHomeDataset, collate_batch
from .labels import LABELS
from .metrics import compute_multilabel_metrics, write_json
from .model import build_model


def load_trusted_checkpoint(path: Path) -> dict:
    """Load a locally-created checkpoint across PyTorch 2.6+ defaults.

    PyTorch 2.6 changed torch.load's default to weights_only=True, which blocks
    non-tensor objects such as pathlib.PosixPath in checkpoint configs.  The
    checkpoints produced by this project are local/trusted training artifacts,
    so if the safe allowlist path fails, fall back to weights_only=False.
    """
    try:
        import pathlib
        safe = [pathlib.PosixPath, pathlib.WindowsPath, pathlib.Path]
        with torch.serialization.safe_globals(safe):
            return torch.load(path, map_location="cpu")
    except Exception as first_exc:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as second_exc:
            raise RuntimeError(
                f"Failed to load checkpoint {path}. First error: {first_exc}. "
                f"Fallback error: {second_exc}."
            ) from second_exc


def main() -> None:
    p = argparse.ArgumentParser(description="Run CHiME-Home audio tagging inference/evaluation")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--metrics-json", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample-rate", type=int, default=None)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--backbone", choices=["logmel_cnn", "ast"], default=None)
    p.add_argument("--ast-model-name", default=None)
    args = p.parse_args()

    ckpt = load_trusted_checkpoint(args.checkpoint)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    backbone = args.backbone or cfg.get("backbone", "logmel_cnn")
    sample_rate = int(args.sample_rate or cfg.get("sample_rate", 16000))
    duration = float(args.duration or cfg.get("duration", 4.0))
    ast_model_name = args.ast_model_name or cfg.get("ast_model_name", "MIT/ast-finetuned-audioset-10-10-0.4593")
    feature_mode = "ast" if backbone == "ast" else "waveform"

    ds = ChimeHomeDataset(
        args.manifest,
        split=args.split,
        sample_rate=sample_rate,
        duration=duration,
        feature_mode=feature_mode,
        ast_model_name=ast_model_name,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(args.device)
    model = build_model(
        backbone,
        num_labels=len(LABELS),
        sample_rate=sample_rate,
        ast_model_name=ast_model_name,
        freeze_encoder=not bool(cfg.get("unfreeze_encoder", False)),
    ).to(device)

    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []
    all_true = []
    all_score = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["x"].to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y = batch["y"].numpy()
            all_true.append(y)
            all_score.append(probs)
            pred = (probs >= args.threshold).astype(int)
            for i, cid in enumerate(batch["chunk_id"]):
                row = {
                    "chunk_id": cid,
                    "audio_path": batch["audio_path"][i],
                    "majorityvote": batch["majorityvote"][i],
                }
                true_labels = []
                pred_labels = []
                for j, lab in enumerate(LABELS):
                    row[f"true_{lab}"] = int(y[i, j])
                    row[f"score_{lab}"] = float(probs[i, j])
                    row[f"pred_{lab}"] = int(pred[i, j])
                    if y[i, j] > 0.5:
                        true_labels.append(lab)
                    if pred[i, j] > 0:
                        pred_labels.append(lab)
                row["true_label_string"] = "".join(true_labels)
                row["pred_label_string"] = "".join(pred_labels)
                rows.append(row)

    y_true = np.concatenate(all_true, axis=0)
    y_score = np.concatenate(all_score, axis=0)
    metrics = compute_multilabel_metrics(y_true, y_score, threshold=args.threshold)
    metrics["split"] = args.split
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["backbone"] = backbone

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    write_json(args.metrics_json, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
