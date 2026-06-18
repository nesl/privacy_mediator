from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from baselines.task.common.manifest import resolve_path, save_label_map
from baselines.task.common.metrics import classification_metrics, write_json
from baselines.task.common.torch_utils import get_device, save_checkpoint, set_seed
from baselines.task.youhome_adl.dataset import YouHomeAVDataset
from baselines.task.youhome_adl.model import AVADLClassifier


def _num_classes(model: nn.Module) -> int:
    return int(model.classifier[-1].out_features)


def compute_class_weights(dataset: YouHomeAVDataset, num_classes: int, mode: str = "balanced") -> torch.Tensor | None:
    """Return inverse-frequency class weights from the training split."""
    if mode == "none":
        return None
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for label_name in dataset.df["label"].astype(str).tolist():
        counts[int(dataset.label_map[label_name])] += 1
    counts = counts.clamp_min(1.0)
    if mode == "balanced":
        weights = counts.sum() / (num_classes * counts)
    elif mode == "sqrt":
        weights = torch.sqrt(counts.sum() / (num_classes * counts))
    else:
        raise ValueError(f"Unsupported class weighting mode: {mode}")
    return weights / weights.mean().clamp_min(1e-8)


def make_weighted_sampler(dataset: YouHomeAVDataset, class_weights: torch.Tensor) -> WeightedRandomSampler:
    sample_weights = []
    for label_name in dataset.df["label"].astype(str).tolist():
        sample_weights.append(float(class_weights[int(dataset.label_map[label_name])]))
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def audit_dataset(dataset: YouHomeAVDataset, name: str) -> dict[str, int]:
    """Cheap path audit so missing files do not silently become zero tensors."""
    missing_frame_dirs = 0
    missing_audio = 0
    no_frame_col = "frame_dir" not in dataset.df.columns
    no_audio_col = "audio_path" not in dataset.df.columns

    for _, row in dataset.df.iterrows():
        if no_frame_col:
            missing_frame_dirs += 1
        else:
            p = resolve_path(row.get("frame_dir", None), dataset.data_root)
            if p is None or not p.exists():
                missing_frame_dirs += 1
        p = dataset._resolve_audio_path(row)
        if p is None or not p.exists():
            missing_audio += 1

    info = {
        "n": len(dataset),
        "missing_frame_dirs": missing_frame_dirs,
        "missing_audio": missing_audio,
        "num_labels": int(dataset.df["label"].nunique()) if len(dataset) else 0,
    }
    print(f"[{name} audit] {info}")
    if len(dataset):
        print(f"[{name} label counts]\n{dataset.df['label'].value_counts().sort_index().to_string()}")
    return info


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    criterion: nn.Module,
    grad_clip: float = 0.0,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true, y_pred = [], []
    missing_image = 0
    missing_audio = 0

    for batch in tqdm(loader, leave=False):
        image = batch["image"].to(device, non_blocking=True).float()
        audio = batch["audio"].to(device, non_blocking=True).float()
        label = batch["label"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(image=image, audio=audio)
        loss = criterion(logits, label)
        if training:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item()) * label.size(0)
        y_true.extend(label.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
        if "image_missing" in batch:
            missing_image += int(batch["image_missing"].sum().item())
        if "audio_missing" in batch:
            missing_audio += int(batch["audio_missing"].sum().item())

    metrics = classification_metrics(y_true, y_pred, labels=list(range(_num_classes(model))))
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    metrics["missing_image_samples"] = missing_image
    metrics["missing_audio_samples"] = missing_audio
    return metrics, y_true, y_pred


def save_model_checkpoint(path: Path, model: nn.Module, label_map: dict[str, int], args: argparse.Namespace) -> None:
    save_checkpoint(
        path,
        model,
        label_map=label_map,
        modality=args.modality,
        image_mode=args.image_mode,
        num_frames=args.num_frames,
        image_size=args.image_size,
        image_arch=args.image_arch,
        image_pretrained=not args.no_image_pretrained,
        audio_backbone=args.audio_backbone,
        audio_duration=args.audio_duration,
        sample_rate=args.sample_rate,
        freeze_backbones=args.freeze_backbones,
        freeze_wav2vec2=not args.finetune_wav2vec2,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        temporal_pool=args.temporal_pool,
        temporal_hidden_dim=args.temporal_hidden_dim,
        temporal_layers=args.temporal_layers,
        logmel_dim=args.logmel_dim,
        logmel_spec_augment=not args.no_logmel_spec_augment,
        class_weighting=args.class_weighting,
        label_smoothing=args.label_smoothing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YouHome image/audio/audio-visual ADL classifier.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--modality", choices=["image", "audio", "av"], default="av")
    parser.add_argument("--image-mode", choices=["full", "crop", "blur", "lowres", "none"], default="full")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-arch", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--no-image-pretrained", action="store_true")
    parser.add_argument("--temporal-pool", choices=["mean", "gru"], default="gru")
    parser.add_argument("--temporal-hidden-dim", type=int, default=256)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--train-frame-sampling", choices=["random", "uniform"], default="random")

    parser.add_argument("--audio-backbone", choices=["wav2vec2", "logmel_cnn"], default="logmel_cnn")
    parser.add_argument("--finetune-wav2vec2", action="store_true")
    parser.add_argument("--audio-duration", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--logmel-dim", type=int, default=256)
    parser.add_argument("--no-logmel-spec-augment", action="store_true")

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--freeze-backbones", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weighting", choices=["none", "balanced", "sqrt"], default="balanced")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default="plateau")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-audit-data", action="store_true")
    parser.add_argument("--strict-missing", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = YouHomeAVDataset(
        args.manifest,
        args.data_root,
        split="train",
        training=True,
        image_mode=args.image_mode,
        image_size=args.image_size,
        num_frames=args.num_frames,
        audio_duration=args.audio_duration,
        sample_rate=args.sample_rate,
        random_audio_crop=True,
        train_frame_sampling=args.train_frame_sampling,
        strict_missing=args.strict_missing,
    )
    val_ds = YouHomeAVDataset(
        args.manifest,
        args.data_root,
        split="val",
        label_map=train_ds.label_map,
        training=False,
        image_mode=args.image_mode,
        image_size=args.image_size,
        num_frames=args.num_frames,
        audio_duration=args.audio_duration,
        sample_rate=args.sample_rate,
        strict_missing=args.strict_missing,
    )
    if len(train_ds) == 0:
        raise ValueError("No training samples found. Check manifest split and paths.")
    save_label_map(train_ds.label_map, out / "label_map.json")

    if not args.no_audit_data:
        audit = {"train": audit_dataset(train_ds, "train"), "val": audit_dataset(val_ds, "val")}
        write_json(audit, out / "data_audit.json")

    model = AVADLClassifier(
        num_classes=len(train_ds.label_map),
        modality=args.modality,
        image_arch=args.image_arch,
        image_pretrained=not args.no_image_pretrained,
        audio_backbone=args.audio_backbone,
        freeze_backbones=args.freeze_backbones,
        freeze_wav2vec2=not args.finetune_wav2vec2,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        temporal_pool=args.temporal_pool,
        temporal_hidden_dim=args.temporal_hidden_dim,
        temporal_layers=args.temporal_layers,
        logmel_dim=args.logmel_dim,
        logmel_spec_augment=not args.no_logmel_spec_augment,
        sample_rate=args.sample_rate,
    )
    device = get_device(args.device)
    model.to(device)

    class_weights = compute_class_weights(train_ds, len(train_ds.label_map), mode=args.class_weighting)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.weighted_sampler and class_weights is not None:
        sampler = make_weighted_sampler(train_ds, class_weights)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available()) if len(val_ds) else None

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(1, args.patience // 3))
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    else:
        scheduler = None

    best = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(args.epochs):
        train_m, _, _ = run_epoch(model, train_loader, optimizer, device, criterion, grad_clip=args.grad_clip)
        if val_loader is not None:
            val_m, _, _ = run_epoch(model, val_loader, None, device, criterion)
            score = val_m["macro_f1"]
        else:
            val_m = {}
            score = train_m["macro_f1"]

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(score)
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch + 1, "train": train_m, "val": val_m, "lr": lr})
        print(
            f"epoch={epoch+1} "
            f"train_loss={train_m['loss']:.4f} train_acc={train_m['accuracy']:.3f} train_f1={train_m['macro_f1']:.3f} "
            f"val_loss={val_m.get('loss', float('nan')):.4f} val_f1={val_m.get('macro_f1', float('nan')):.3f} "
            f"lr={lr:.2e}"
        )

        if score > best + args.min_delta:
            best = score
            best_epoch = epoch + 1
            stale_epochs = 0
            save_model_checkpoint(out / "best.pt", model, train_ds.label_map, args)
        else:
            stale_epochs += 1

        save_model_checkpoint(out / "last.pt", model, train_ds.label_map, args)

        if val_loader is not None and args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}; best_val_macro_f1={best:.4f} at epoch {best_epoch}.")
            break

    write_json({"history": history, "best_val_macro_f1": best, "best_epoch": best_epoch}, out / "metrics.json")


if __name__ == "__main__":
    main()
