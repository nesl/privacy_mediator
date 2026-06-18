# Privacy mediator task baselines

See `task/README.md` for the runnable baselines.

Quick examples:

```bash
# Le2i: extract pose, then train classifier
python -m baselines.task.le2i_fall.extract_pose --manifest le2i.csv --output-dir features/le2i_pose
python -m baselines.task.le2i_fall.train --manifest le2i_with_keypoints.csv --output-dir runs/le2i

# ChokePoint: YOLO person presence
python -m baselines.task.chokepoint_presence.infer_yolo --manifest chokepoint.csv --output-csv runs/chokepoint/yolo.csv

# YouHome: audio-visual ADL
python -m baselines.task.youhome_adl.train --manifest youhome_av.csv --output-dir runs/youhome_av --modality av
```
