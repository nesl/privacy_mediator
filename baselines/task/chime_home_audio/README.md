# CHiME-Home domestic audio-tagging baseline

This baseline supports the CHiME-Home / DCASE 2016 domestic audio-tagging layout:

```text
data/chime_home/
  development_chunks_refined.csv
  evaluation_chunks_refined.csv
  chunks/
    <chunkname>.csv
    <chunkname>.16kHz.wav
    <chunkname>.48kHz.wav
```

The task is **multi-label sound-source tagging** over 4-second domestic audio chunks. The label set is:

| Code | Meaning |
|---|---|
| `c` | child speech |
| `m` | adult male speech |
| `f` | adult female speech |
| `v` | video game / TV |
| `p` | percussive sounds such as knocks, bangs, crashes, footsteps |
| `b` | broadband noise such as appliances |
| `o` | other identifiable sounds |

The manifest builder reads the majority-vote label string from each chunk-level CSV.

## Baselines

Two backbones are implemented:

1. `logmel_cnn` — reliable default; small log-mel CNN trained from scratch.
2. `ast` — stronger optional baseline; pretrained Audio Spectrogram Transformer from HuggingFace, frozen by default, with a new multi-label head.

The old DCASE baseline used MFCC/GMM classifiers. This package uses a more modern log-mel CNN by default and includes AST for a pretrained AudioSet-style feature baseline.

## Requirements

Minimum requirements are in `requirements_chime_home.txt` at the package root.

For the AST baseline, also install `transformers` and make sure the model can be downloaded on first use:

```bash
pip install transformers
```

## 1. Build manifest

```bash
python -m baselines.task.chime_home_audio.make_manifest \
  --root data/chime_home \
  --output-csv data/chime_home/chime_home_manifest.csv \
  --summary-json data/chime_home/chime_home_manifest_summary.json \
  --sample-rate-khz 16 \
  --val-frac 0.1 \
  --seed 13
```

This uses development chunks for train/val and evaluation chunks for test.

## 2. Train default log-mel CNN

```bash
CUDA_VISIBLE_DEVICES=1 python -m baselines.task.chime_home_audio.train \
  --manifest data/chime_home/chime_home_manifest.csv \
  --output-dir outputs/chime_home_logmel \
  --backbone logmel_cnn \
  --batch-size 32 \
  --epochs 30 \
  --lr 1e-3 \
  --device cuda:0
```

With `CUDA_VISIBLE_DEVICES=1`, `cuda:0` maps to physical GPU 1.

## 3. Evaluate on test

```bash
CUDA_VISIBLE_DEVICES=1 python -m baselines.task.chime_home_audio.infer \
  --manifest data/chime_home/chime_home_manifest.csv \
  --checkpoint outputs/chime_home_logmel/best.pt \
  --split test \
  --output-csv outputs/chime_home_logmel/test_predictions.csv \
  --metrics-json outputs/chime_home_logmel/test_metrics.json \
  --device cuda:0
```

## Optional: tune threshold on validation predictions

First run validation inference:

```bash
CUDA_VISIBLE_DEVICES=1 python -m baselines.task.chime_home_audio.infer \
  --manifest data/chime_home/chime_home_manifest.csv \
  --checkpoint outputs/chime_home_logmel/best.pt \
  --split val \
  --output-csv outputs/chime_home_logmel/val_predictions.csv \
  --metrics-json outputs/chime_home_logmel/val_metrics.json \
  --device cuda:0
```

Tune a scalar threshold:

```bash
python -m baselines.task.chime_home_audio.tune_threshold \
  --predictions outputs/chime_home_logmel/val_predictions.csv \
  --output-json outputs/chime_home_logmel/best_threshold.json \
  --metric macro_f1
```

Then pass the chosen threshold to test inference:

```bash
CUDA_VISIBLE_DEVICES=1 python -m baselines.task.chime_home_audio.infer \
  --manifest data/chime_home/chime_home_manifest.csv \
  --checkpoint outputs/chime_home_logmel/best.pt \
  --split test \
  --threshold 0.35 \
  --output-csv outputs/chime_home_logmel/test_predictions_tuned.csv \
  --metrics-json outputs/chime_home_logmel/test_metrics_tuned.json \
  --device cuda:0
```

## Optional stronger pretrained AST baseline

```bash
CUDA_VISIBLE_DEVICES=1 python -m baselines.task.chime_home_audio.train \
  --manifest data/chime_home/chime_home_manifest.csv \
  --output-dir outputs/chime_home_ast \
  --backbone ast \
  --batch-size 8 \
  --epochs 15 \
  --lr 1e-4 \
  --device cuda:0
```

Evaluate:

```bash
CUDA_VISIBLE_DEVICES=1 python -m baselines.task.chime_home_audio.infer \
  --manifest data/chime_home/chime_home_manifest.csv \
  --checkpoint outputs/chime_home_ast/best.pt \
  --split test \
  --output-csv outputs/chime_home_ast/test_predictions.csv \
  --metrics-json outputs/chime_home_ast/test_metrics.json \
  --device cuda:0
```

## Main metrics

The evaluation reports:

- macro-F1, micro-F1, samples-F1,
- per-label precision/recall/F1,
- hamming loss,
- subset accuracy,
- optional average precision / ROC-AUC when defined.

For the privacy-mediation paper, macro-F1 or mean per-label F1 is the best primary utility metric because the task is multi-label and label imbalance is expected.
