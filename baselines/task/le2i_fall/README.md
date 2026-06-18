# Le2i / ImViA fall-detection baseline

Primary utility baseline: **YOLO-pose keypoints + GRU temporal classifier**.

This implementation supports the nested dataset layout:

```text
DATA_ROOT/
  Home_01/Home_01/Videos/video (1).avi
  Home_01/Home_01/Annotation_files/video (1).txt
  Coffee_room_01/Coffee_room_01/...
  Office/Office/Videos/video (1).avi       # may be unlabeled
  Lecture_room/Lecture_room/Videos/...     # may be unlabeled
```

The Le2i README says each annotation file contains the fall start frame, fall
end frame, then per-frame body localization values. The scanner uses the first
two rows to label temporal windows. Folders with videos but no `Annotation_files`
are kept as `split=unlabeled` and are excluded from supervised training/eval.

## 1. Create manifest

From the repository root that contains `baselines/` and `data/`:

```bash
python -m baselines.task.le2i_fall.make_manifest \
  --data-dir data/le2i \
  --output-csv data/le2i/le2i_manifest.csv \
  --seed 13
```

This writes one row per video and assigns train/val/test splits for labeled
videos. Unlabeled Office/Lecture-room videos remain in the manifest with
`split=unlabeled`.

## 2. Extract pose keypoints

```bash
python -m baselines.task.le2i_fall.extract_pose \
  --manifest data/le2i/le2i_manifest.csv \
  --output-dir outputs/le2i_pose \
  --updated-manifest outputs/le2i_manifest_with_keypoints.csv \
  --model yolo11n-pose.pt \
  --stride 3
```

The output `.npz` files contain `keypoints` with shape `[T,17,3]`. The stride is
stored in the `.npz` and manifest so the window labels can be mapped back to the
original video frame indices.

## 3. Train

Window-level training is the recommended default. It creates positive windows
that overlap the annotated fall interval and negative windows outside it.

```bash
python -m baselines.task.le2i_fall.train \
  --manifest outputs/le2i_manifest_with_keypoints.csv \
  --output-dir outputs/le2i_fall_model \
  --sample-mode window \
  --sequence-len 64 \
  --window-stride 32 \
  --epochs 30 \
  --batch-size 16
```

The training script saves:

```text
outputs/le2i_fall_model/best.pt
outputs/le2i_fall_model/last.pt
outputs/le2i_fall_model/label_map.json
outputs/le2i_fall_model/metrics.json
```

## 4. Test / accuracy

```bash
python -m baselines.task.le2i_fall.infer \
  --manifest outputs/le2i_manifest_with_keypoints.csv \
  --checkpoint outputs/le2i_fall_model/best.pt \
  --label-map outputs/le2i_fall_model/label_map.json \
  --split test \
  --output-csv outputs/le2i_test_windows.csv \
  --video-output-csv outputs/le2i_test_videos.csv \
  --metrics-json outputs/le2i_test_metrics.json
```

`outputs/le2i_test_windows.csv` contains window-level predictions. The video CSV
aggregates windows by taking the maximum fall probability per video.

## 5. Inference on unlabeled Office / Lecture-room videos

```bash
python -m baselines.task.le2i_fall.infer \
  --manifest outputs/le2i_manifest_with_keypoints.csv \
  --checkpoint outputs/le2i_fall_model/best.pt \
  --label-map outputs/le2i_fall_model/label_map.json \
  --split unlabeled \
  --output-csv outputs/le2i_unlabeled_windows.csv \
  --video-output-csv outputs/le2i_unlabeled_videos.csv
```

No accuracy is computed for unlabeled videos.
