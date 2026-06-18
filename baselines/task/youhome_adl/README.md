# YouHome ADL baseline

This baseline trains an activity classifier for the nested YouHome layout:

```text
data/youhome/
  data/<participant>/<activity>/<session>/
      *.mp4
      *_1.jpg, *_2.jpg, ...
      audio.wav
  label/<participant>/<activity>/<session>/
      *_1.txt, *_2.txt, ...
```

The activity class is the folder name, e.g. `Write`, `Nap`, `Cook.Cut`.
The label `.txt` files are optional YOLO-style person/user boxes:

```text
class_id x_center y_center width height
```

Those labels are used only for crop-based/privacy variants, not as the activity label.

## 1. Build the manifest

Recommended default: split by participant to avoid subject leakage.

```bash
CUDA_VISIBLE_DEVICES=1 python -m baselines.task.youhome_adl.make_manifest \
  --root data/youhome \
  --output-csv data/youhome/youhome_manifest.csv \
  --summary-json data/youhome/youhome_manifest_summary.json \
  --split-by participant \
  --seed 13
```

If too many labels are missing from train because some participants have only a subset of activities, use the stratified sample split:

```bash
CUDA_VISIBLE_DEVICES=1 python -m baselines.task.youhome_adl.make_manifest \
  --root data/youhome \
  --output-csv data/youhome/youhome_manifest.csv \
  --summary-json data/youhome/youhome_manifest_summary.json \
  --split-by activity_stratified \
  --seed 13
```

## 2. Train the audio-visual baseline

This is the main baseline: ResNet image encoder + pretrained Wav2Vec2 audio encoder + fusion MLP.

```bash
CUDA_VISIBLE_DEVICES=1 python -m baselines.task.youhome_adl.train \
  --manifest data/youhome/youhome_manifest.csv \
  --data-root data/youhome \
  --output-dir outputs/youhome_adl_av \
  --modality av \
  --image-mode full \
  --num-frames 4 \
  --audio-backbone wav2vec2 \
  --audio-duration 10 \
  --batch-size 8 \
  --epochs 30 \
  --device cuda:0
```

With `CUDA_VISIBLE_DEVICES=1`, `--device cuda:0` maps to physical GPU 1.

If you want to avoid downloading Wav2Vec2 weights or the audio is not speech-like, use a trainable log-mel CNN:

```bash
CUDA_VISIBLE_DEVICES=1 python -m baselines.task.youhome_adl.train \
  --manifest data/youhome/youhome_manifest.csv \
  --data-root data/youhome \
  --output-dir outputs/youhome_adl_av_logmel \
  --modality av \
  --image-mode full \
  --num-frames 4 \
  --audio-backbone logmel_cnn \
  --audio-duration 10 \
  --batch-size 8 \
  --epochs 30 \
  --device cuda:0
```

## 3. Evaluate on test split

```bash
CUDA_VISIBLE_DEVICES=1 python -m baselines.task.youhome_adl.infer \
  --manifest data/youhome/youhome_manifest.csv \
  --data-root data/youhome \
  --checkpoint outputs/youhome_adl_av/best.pt \
  --label-map outputs/youhome_adl_av/label_map.json \
  --split test \
  --output-csv outputs/youhome_adl_av/test_predictions.csv \
  --metrics-json outputs/youhome_adl_av/test_metrics.json \
  --report-txt outputs/youhome_adl_av/test_report.txt \
  --device cuda:0
```

## 4. Privacy/preprocessing variants

Use the same training/evaluation code with different `--image-mode` values:

```text
full    raw/released frames
crop    crop to the largest annotated person/user bbox from label txt
blur    Gaussian-blurred frame
lowres  downsample + upsample frame
none    black image; useful for audio-only ablations with modality av
```

You can also train image-only and audio-only baselines:

```bash
# Image only
CUDA_VISIBLE_DEVICES=1 python -m baselines.task.youhome_adl.train \
  --manifest data/youhome/youhome_manifest.csv \
  --data-root data/youhome \
  --output-dir outputs/youhome_adl_image \
  --modality image \
  --image-mode full \
  --num-frames 4 \
  --batch-size 16 \
  --epochs 30 \
  --device cuda:0

# Audio only
CUDA_VISIBLE_DEVICES=1 python -m baselines.task.youhome_adl.train \
  --manifest data/youhome/youhome_manifest.csv \
  --data-root data/youhome \
  --output-dir outputs/youhome_adl_audio \
  --modality audio \
  --audio-backbone wav2vec2 \
  --audio-duration 10 \
  --batch-size 8 \
  --epochs 30 \
  --device cuda:0
```

## Metrics

`infer.py` reports accuracy, macro-F1, weighted-F1, and per-class precision/recall/F1. For the privacy mediator paper, macro-F1 and per-class recall are usually the most useful ADL metrics.
