from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.task.common.manifest import save_label_map
from baselines.task.common.metrics import classification_metrics, write_json
from baselines.task.common.torch_utils import get_device, save_checkpoint, set_seed
from baselines.task.le2i_fall.dataset import Le2iPoseWindowDataset, PoseSequenceDataset
from baselines.task.le2i_fall.le2i_layout import make_manifest
from baselines.task.le2i_fall.model import PoseGRUFallClassifier


def make_dataset(args, split: str, label_map=None):
    if args.sample_mode == "window":
        return Le2iPoseWindowDataset(
            args.manifest,
            args.data_root,
            split=split,
            label_map=label_map,
            sequence_len=args.sequence_len,
            window_stride=args.window_stride,
            extraction_stride=args.extraction_stride,
            min_fall_overlap=args.min_fall_overlap,
        )
    return PoseSequenceDataset(
        args.manifest,
        args.data_root,
        split=split,
        label_map=label_map,
        sequence_len=args.sequence_len,
    )


def run_epoch(model, loader, optimizer, device, num_classes: int):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    y_true, y_pred = [], []
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device).float(), y.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = classification_metrics(y_true, y_pred, labels=list(range(num_classes)))
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Le2i fall-detection baseline on extracted pose keypoints.")
    parser.add_argument("--manifest", default=None, help="Manifest with keypoints_path. If omitted, --data-dir is scanned to create a video manifest only.")
    parser.add_argument("--data-dir", default=None, help="Le2i root, used only to auto-create a manifest if --manifest is omitted.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-mode", choices=["window", "video"], default="window")
    parser.add_argument("--sequence-len", type=int, default=64)
    parser.add_argument("--window-stride", type=int, default=32, help="Pose-index stride between training windows.")
    parser.add_argument("--extraction-stride", type=int, default=None, help="Original frame stride used during pose extraction; normally read from manifest.")
    parser.add_argument("--min-fall-overlap", type=float, default=0.10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.manifest is None:
        if args.data_dir is None:
            raise SystemExit("Provide --manifest with keypoints_path, or --data-dir to create a manifest first.")
        args.manifest = str(out / "le2i_manifest.csv")
        make_manifest(args.data_dir, args.manifest, seed=args.seed)
        raise SystemExit(
            f"Created {args.manifest}. Now run extract_pose.py to add keypoints_path, then rerun train.py with --manifest."
        )

    train_ds = make_dataset(args, split="train")
    if len(train_ds) == 0:
        raise SystemExit("No train samples/windows found. Check split labels and keypoints_path in the manifest.")
    val_ds = make_dataset(args, split="val", label_map=train_ds.label_map)
    if len(val_ds) == 0:
        print("Warning: no val split found; using test split for model selection.")
        val_ds = make_dataset(args, split="test", label_map=train_ds.label_map)
    if len(val_ds) == 0:
        print("Warning: no val/test split found; using train split for model selection.")
        val_ds = train_ds

    save_label_map(train_ds.label_map, out / "label_map.json")
    num_classes = len(train_ds.label_map)
    input_dim = train_ds[0][0].shape[-1]
    model = PoseGRUFallClassifier(input_dim=input_dim, hidden_dim=args.hidden_dim, num_classes=num_classes)
    device = get_device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    best = -1.0
    history = []
    for epoch in range(args.epochs):
        train_m = run_epoch(model, train_loader, optimizer, device, num_classes)
        val_m = run_epoch(model, val_loader, None, device, num_classes)
        history.append({"epoch": epoch + 1, "train": train_m, "val": val_m})
        print(f"epoch={epoch+1} train_f1={train_m['macro_f1']:.3f} val_f1={val_m['macro_f1']:.3f}")
        if val_m["macro_f1"] > best:
            best = val_m["macro_f1"]
            save_checkpoint(
                out / "best.pt",
                model,
                label_map=train_ds.label_map,
                sequence_len=args.sequence_len,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                sample_mode=args.sample_mode,
                window_stride=args.window_stride,
                extraction_stride=args.extraction_stride,
                min_fall_overlap=args.min_fall_overlap,
            )
    # Save final separately so failed validation does not leave only an old best.
    save_checkpoint(
        out / "last.pt",
        model,
        label_map=train_ds.label_map,
        sequence_len=args.sequence_len,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        sample_mode=args.sample_mode,
        window_stride=args.window_stride,
        extraction_stride=args.extraction_stride,
        min_fall_overlap=args.min_fall_overlap,
    )
    write_json({"history": history, "best_val_macro_f1": best, "label_map": train_ds.label_map}, out / "metrics.json")
    print(f"Saved {out / 'best.pt'}, {out / 'last.pt'}, and {out / 'label_map.json'}")


if __name__ == "__main__":
    main()
