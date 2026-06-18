# Task utility baselines for privacy-preserving preprocessing

This package contains one primary downstream utility evaluator per dataset/case
study. The goal is not to win a model benchmark; it is to measure how much task
utility changes when your mediator replaces raw sensor data with lower-disclosure
representations.

## Baselines included

| Case | Dataset | Primary evaluator | Folder |
|---|---|---|---|
| Fall detection | Le2i / ImViA | YOLO-pose keypoints + GRU temporal classifier | `le2i_fall/` |
| Threshold presence | ChokePoint | YOLO person detector + event/count logic | `chokepoint_presence/` |
| ADL recognition | YouHome | ResNet image stream + pretrained/fallback audio stream + fusion MLP | `youhome_adl/` |

## Install

```bash
pip install -r requirements.txt
```

If you use CUDA, install the PyTorch build matching your CUDA version from the
official PyTorch instructions before installing the rest of the requirements.

## Data ingestion pattern

Every baseline uses a CSV/JSONL manifest. This keeps the code independent of the
exact dataset layout and makes it easy to add your mediator outputs later.

Common columns:

```csv
sample_id,split,label,video_path,frame_dir,image_path,audio_path,keypoints_path
```

Only the columns needed by each task are required. Paths may be absolute or
relative to `--data-root`.

## How to evaluate mediator outputs

Create a manifest that points to the mediator-preprocessed artifacts instead of
raw data, then run the same utility evaluator. For example, for ChokePoint, run
YOLO over raw frames and then over blurred/low-res/cropped frames. For YouHome,
train or evaluate the same audio-visual classifier with modalities removed or
transformed.
