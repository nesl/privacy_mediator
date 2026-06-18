# ChokePoint entryway / visitor-monitoring baseline

Primary baseline: pretrained YOLO person detector plus threshold-presence logic.
There is **no supervised YOLO training step** for this baseline; YOLO is pretrained.
The train/val/test split is only a sequence-level evaluation split, so you can tune thresholds/smoothing on train or val and report held-out performance on test.

## Expected layout

```text
data/chokepoint/
  groundtruth/
    P1E_S1_C1.xml
    ...
  P1E_S1/
    P1E_S1_C1/
      00000000.jpg
      00000001.jpg
      ...
  P1E_S1.tar.xz
  groundtruth.tar.xz
```

The XML files contain frame elements. A frame with one or more `<person>` nodes is a raw XML positive frame. The XML annotations are face/eye-oriented, so the evaluator distinguishes:

- `raw_xml_face_frame_presence`: exact XML person/eye-frame matching.
- `presence`: visitor-monitoring labels after optional dilation around XML positives.
- `events`: interval-level visitor crossing metrics.

This is important because YOLO detects full visible people, while the XML labels mark frames with annotated eyes/faces.

## GPU note

The current `infer_yolo.py` auto-selects the RTX 2070 by GPU name/UUID before importing Torch/Ultralytics. Use `env -u CUDA_VISIBLE_DEVICES` so any old shell setting does not force the unsupported TITAN V.

If auto-selection fails, inspect GPU UUIDs:

```bash
nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader
```

Then run inference manually with the RTX 2070 UUID:

```bash
CUDA_VISIBLE_DEVICES=GPU-PASTE_RTX_2070_UUID_HERE python -m baselines.task.chokepoint_presence.infer_yolo ... --device 0
```

## 1. Create manifest and labels

Archive extraction happens first by default. The script scans `data/chokepoint/*.tar.xz`, extracts archives as needed, and writes the sequence manifest plus frame/event labels.

```bash
python -m baselines.task.chokepoint_presence.make_manifest \
  --data-dir data/chokepoint \
  --output-manifest data/chokepoint/chokepoint_manifest.csv \
  --output-labels data/chokepoint/chokepoint_labels.csv \
  --output-events data/chokepoint/chokepoint_events.csv \
  --seed 13 \
  --test-frac 0.25 \
  --overwrite-extract
```

Use `--no-extract-archives` only if you explicitly want to skip extraction.

## 2. Run YOLO inference

Run with a low detector confidence so the CSV keeps candidate boxes. You can then tune the actual confidence threshold without rerunning YOLO.

### Train split

```bash
env -u CUDA_VISIBLE_DEVICES python -m baselines.task.chokepoint_presence.infer_yolo \
  --manifest data/chokepoint/chokepoint_manifest.csv \
  --data-root data/chokepoint \
  --split train \
  --output-csv outputs/chokepoint/yolo_train_frames.csv \
  --model yolo11s.pt \
  --conf 0.05 \
  --stride 5 \
  --imgsz 960 \
  --safe-cuda \
  --retry-cpu-on-error \
  --error-log outputs/chokepoint/yolo_train_errors.csv
```

### Test split

```bash
env -u CUDA_VISIBLE_DEVICES python -m baselines.task.chokepoint_presence.infer_yolo \
  --manifest data/chokepoint/chokepoint_manifest.csv \
  --data-root data/chokepoint \
  --split test \
  --output-csv outputs/chokepoint/yolo_test_frames.csv \
  --model yolo11s.pt \
  --conf 0.05 \
  --stride 5 \
  --imgsz 960 \
  --safe-cuda \
  --retry-cpu-on-error \
  --error-log outputs/chokepoint/yolo_test_errors.csv
```

Notes:

- `yolo11s.pt` or `yolo11m.pt` usually works better than `yolo11n.pt` if GPU time is acceptable.
- `--safe-cuda` applies conservative CUDA/cuDNN settings.
- `--retry-cpu-on-error` retries a failed frame on CPU instead of losing the full run.
- `--augment` may improve detection slightly but is slower.

## 3. Tune post-processing on train or val

This is the “fine-tuning” step for this baseline. It does **not** fine-tune YOLO weights. It searches confidence threshold, label dilation, and temporal smoothing.

```bash
python -m baselines.task.chokepoint_presence.tune_postprocess \
  --predictions outputs/chokepoint/yolo_train_frames.csv \
  --labels data/chokepoint/chokepoint_labels.csv \
  --manifest data/chokepoint/chokepoint_manifest.csv \
  --split train \
  --output-json outputs/chokepoint/postprocess_config.json \
  --sweep-csv outputs/chokepoint/postprocess_sweep.csv \
  --optimize f2
```

`tune_postprocess.py` now shows a `tqdm` progress bar over the parameter sweep. With the default sweep it evaluates:

```text
5 confidence thresholds × 5 label dilations × 3 min-positive-frame settings × 4 max-gap settings = 300 configs
```

To make tuning faster while debugging, reduce the sweep:

```bash
python -m baselines.task.chokepoint_presence.tune_postprocess \
  --predictions outputs/chokepoint/yolo_train_frames.csv \
  --labels data/chokepoint/chokepoint_labels.csv \
  --manifest data/chokepoint/chokepoint_manifest.csv \
  --split train \
  --output-json outputs/chokepoint/postprocess_config.json \
  --sweep-csv outputs/chokepoint/postprocess_sweep.csv \
  --optimize f2 \
  --conf-thresholds 0.25,0.45,0.65 \
  --label-dilations 0,10,30 \
  --min-positive-frames 1,2 \
  --max-gap-frames 0,15
```

## 4. Evaluate on the held-out test split

```bash
python -m baselines.task.chokepoint_presence.evaluate \
  --predictions outputs/chokepoint/yolo_test_frames.csv \
  --labels data/chokepoint/chokepoint_labels.csv \
  --manifest data/chokepoint/chokepoint_manifest.csv \
  --split test \
  --config-json outputs/chokepoint/postprocess_config.json \
  --metrics-json outputs/chokepoint/test_metrics.json \
  --matched-csv outputs/chokepoint/test_matched_frames.csv \
  --true-events-csv outputs/chokepoint/test_true_events.csv \
  --pred-events-csv outputs/chokepoint/test_pred_events.csv
```

If you want to evaluate without a tuned config, the improved defaults are:

```text
label_dilation_pre/post = 10 frames
min_positive_frames = 2 sampled frames
max_gap_frames = 15 frames
event_tolerance_frames = 30 frames
```

## 5. Optional: clip-level predictions

```bash
python -m baselines.task.chokepoint_presence.eventize \
  --predictions outputs/chokepoint/yolo_test_frames.csv \
  --output-csv outputs/chokepoint/yolo_test_clip_predictions.csv \
  --min-positive-frames 1 \
  --count-agg max
```

This converts frame predictions into one row per `sample_id`.

## Quick sanity checks

After inference:

```bash
wc -l outputs/chokepoint/yolo_train_frames.csv
wc -l outputs/chokepoint/yolo_test_frames.csv
head -5 outputs/chokepoint/yolo_train_frames.csv
head -5 outputs/chokepoint/yolo_test_frames.csv
```

After tuning/evaluation:

```bash
cat outputs/chokepoint/postprocess_config.json
cat outputs/chokepoint/test_metrics.json
```

If train/test inference produces no rows, inspect the error logs:

```bash
head -20 outputs/chokepoint/yolo_train_errors.csv
head -20 outputs/chokepoint/yolo_test_errors.csv
```

## Why this should improve the earlier result

An earlier result may show perfect recall but many false positives against raw XML-negative frames. That often happens because XML negatives can still contain a visible body before/after the face/eyes are annotated. The improved evaluator reports the raw XML diagnostic, but uses dilated labels and event-level metrics for the visitor-monitoring task. Temporal smoothing also removes isolated YOLO spikes.

## Evaluating privacy-preprocessed outputs

To evaluate a mediator output, create a manifest whose `frame_dir` or `video_path` points to the transformed frames/videos, then run `infer_yolo` and `evaluate` the same way. For box-only or event-label-only outputs, skip YOLO and write a prediction CSV with `sample_id,frame_index,person_present,person_count`.
