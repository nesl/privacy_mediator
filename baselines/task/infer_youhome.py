from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from baselines.task.common.manifest import load_label_map
from baselines.task.common.metrics import classification_metrics, write_json
from sklearn.metrics import classification_report, confusion_matrix
from baselines.task.common.torch_utils import get_device, load_checkpoint, set_seed
from baselines.task.youhome_adl.dataset import YouHomeAVDataset
from baselines.task.youhome_adl.model import AVADLClassifier


def build_model(label_map: dict[str, int], meta: dict) -> AVADLClassifier:
    return AVADLClassifier(
        num_classes=len(label_map),
        modality=meta.get("modality", "av"),
        image_arch=meta.get("image_arch", "resnet18"),
        image_pretrained=bool(meta.get("image_pretrained", True)),
        audio_backbone=meta.get("audio_backbone", "logmel_cnn"),
        freeze_backbones=bool(meta.get("freeze_backbones", False)),
        freeze_wav2vec2=bool(meta.get("freeze_wav2vec2", True)),
        hidden_dim=int(meta.get("hidden_dim", 256)),
        dropout=float(meta.get("dropout", 0.4)),
        temporal_pool=meta.get("temporal_pool", "gru"),
        temporal_hidden_dim=int(meta.get("temporal_hidden_dim", 256)),
        temporal_layers=int(meta.get("temporal_layers", 1)),
        logmel_dim=int(meta.get("logmel_dim", 256)),
        # Disable stochastic spec augmentation in eval even if the training metadata says it was used.
        logmel_spec_augment=False,
        sample_rate=int(meta.get("sample_rate", 16000)),
    )


def build_loader(args: argparse.Namespace, label_map: dict[str, int], meta: dict, random_audio_crop: bool) -> DataLoader:
    ds = YouHomeAVDataset(
        args.manifest,
        args.data_root,
        split=args.split,
        label_map=label_map,
        training=False,
        image_mode=meta.get("image_mode", "full"),
        image_size=int(meta.get("image_size", 224)),
        num_frames=int(meta.get("num_frames", 8)),
        audio_duration=float(meta.get("audio_duration", 10.0)),
        sample_rate=int(meta.get("sample_rate", 16000)),
        random_audio_crop=random_audio_crop,
        random_audio_crop_eval=random_audio_crop,
        strict_missing=args.strict_missing,
    )
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate/infer with a trained YouHome ADL classifier.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--report-txt", default=None)
    parser.add_argument("--confusion-csv", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tta-runs", type=int, default=1, help="Average predictions over this many inference passes.")
    parser.add_argument("--tta-random-audio-crop", action="store_true", help="Use random audio crops during TTA passes.")
    parser.add_argument("--strict-missing", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    set_seed(args.seed)
    label_map = load_label_map(args.label_map)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}
    model = build_model(label_map, meta)
    load_checkpoint(args.checkpoint, model)
    device = get_device(args.device)
    model.to(device).eval()

    inv = {v: k for k, v in label_map.items()}
    prob_sums: OrderedDict[str, torch.Tensor] = OrderedDict()
    label_by_sid: dict[str, int] = {}
    missing_image_by_sid: dict[str, int] = {}
    missing_audio_by_sid: dict[str, int] = {}

    num_runs = max(1, int(args.tta_runs))
    with torch.no_grad():
        for run_idx in range(num_runs):
            # First pass should be deterministic. Extra TTA passes can random-crop audio.
            random_audio_crop = bool(args.tta_random_audio_crop and num_runs > 1 and run_idx > 0)
            loader = build_loader(args, label_map, meta, random_audio_crop=random_audio_crop)
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True).float()
                audio = batch["audio"].to(device, non_blocking=True).float()
                logits = model(image=image, audio=audio)
                probs = torch.softmax(logits, dim=1).cpu()
                for i, sid in enumerate(batch["sample_id"]):
                    sid = str(sid)
                    if sid not in prob_sums:
                        prob_sums[sid] = torch.zeros_like(probs[i])
                    prob_sums[sid] += probs[i]
                    label_by_sid[sid] = int(batch["label"][i])
                    if "image_missing" in batch:
                        missing_image_by_sid[sid] = int(batch["image_missing"][i].item())
                    if "audio_missing" in batch:
                        missing_audio_by_sid[sid] = int(batch["audio_missing"][i].item())

    rows, y_true, y_pred = [], [], []
    for sid, prob_sum in prob_sums.items():
        prob = prob_sum / num_runs
        true = label_by_sid[sid]
        pred = int(prob.argmax().item())
        top3 = torch.topk(prob, k=min(3, prob.size(0)))
        rec = {
            "sample_id": sid,
            "true_label": inv[int(true)],
            "pred_label": inv[int(pred)],
            "confidence": float(prob[int(pred)]),
            "tta_runs": num_runs,
            "image_missing": missing_image_by_sid.get(sid, 0),
            "audio_missing": missing_audio_by_sid.get(sid, 0),
        }
        for rank, (cls_idx, cls_prob) in enumerate(zip(top3.indices.tolist(), top3.values.tolist()), start=1):
            rec[f"top{rank}_label"] = inv[int(cls_idx)]
            rec[f"top{rank}_prob"] = float(cls_prob)
        rows.append(rec)
        y_true.append(true)
        y_pred.append(pred)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    labels = list(range(len(label_map)))
    if args.metrics_json:
        metrics = classification_metrics(y_true, y_pred, labels=labels)
        metrics["tta_runs"] = num_runs
        metrics["missing_image_samples"] = int(sum(missing_image_by_sid.values()))
        metrics["missing_audio_samples"] = int(sum(missing_audio_by_sid.values()))
        write_json(metrics, args.metrics_json)

    if args.report_txt:
        names = [inv[i] for i in labels]
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=names,
            zero_division=0,
        )
        Path(args.report_txt).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_txt).write_text(report, encoding="utf-8")

    if args.confusion_csv:
        names = [inv[i] for i in labels]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        Path(args.confusion_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cm, index=names, columns=names).to_csv(args.confusion_csv)


if __name__ == "__main__":
    main()
