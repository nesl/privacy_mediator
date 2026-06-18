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
from .labels import LABELS, LABEL_MEANINGS
from .metrics import compute_multilabel_metrics, write_json
from .model import build_model


def run_epoch(model, loader, device, optimizer=None, amp=False):
    train = optimizer is not None
    model.train(train)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    losses = []
    all_true = []
    all_score = []
    scaler = torch.cuda.amp.GradScaler(enabled=amp and train)
    for batch in tqdm(loader, leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu()))
        all_true.append(y.detach().cpu().numpy())
        all_score.append(torch.sigmoid(logits).detach().cpu().numpy())
    y_true = np.concatenate(all_true, axis=0)
    y_score = np.concatenate(all_score, axis=0)
    metrics = compute_multilabel_metrics(y_true, y_score, threshold=0.5)
    metrics["loss"] = float(np.mean(losses)) if losses else None
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train CHiME-Home audio tagging baseline")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--backbone", choices=["logmel_cnn", "ast"], default="logmel_cnn")
    p.add_argument("--ast-model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    p.add_argument("--unfreeze-encoder", action="store_true", help="Only applies to AST; default freezes encoder")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--patience", type=int, default=8)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_mode = "ast" if args.backbone == "ast" else "waveform"
    train_ds = ChimeHomeDataset(args.manifest, split="train", sample_rate=args.sample_rate,
                                duration=args.duration, feature_mode=feature_mode,
                                ast_model_name=args.ast_model_name)
    val_ds = ChimeHomeDataset(args.manifest, split="val", sample_rate=args.sample_rate,
                              duration=args.duration, feature_mode=feature_mode,
                              ast_model_name=args.ast_model_name)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_batch,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_batch,
                            pin_memory=True)

    device = torch.device(args.device)
    model = build_model(args.backbone, num_labels=len(LABELS), sample_rate=args.sample_rate,
                        ast_model_name=args.ast_model_name,
                        freeze_encoder=not args.unfreeze_encoder).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config["labels"] = LABELS
    config["label_meanings"] = LABEL_MEANINGS
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "label_map.json").write_text(json.dumps({lab: i for i, lab in enumerate(LABELS)}, indent=2), encoding="utf-8")

    best = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer, amp=args.amp)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None, amp=False)
        rec = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(rec)
        score = val_metrics["macro_f1"]
        print(f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} val_macro_f1={score:.4f} val_micro_f1={val_metrics['micro_f1']:.4f}")
        torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "val_metrics": val_metrics}, args.output_dir / "last.pt")
        if score > best:
            best = score
            stale = 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "val_metrics": val_metrics}, args.output_dir / "best.pt")
            write_json(args.output_dir / "best_val_metrics.json", val_metrics)
        else:
            stale += 1
        write_json(args.output_dir / "history.json", history)
        if args.patience > 0 and stale >= args.patience:
            print(f"Early stopping after {stale} stale epochs")
            break


if __name__ == "__main__":
    main()
