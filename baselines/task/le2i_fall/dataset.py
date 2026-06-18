"""Le2i/ImViA pose datasets.

The recommended pipeline is:

1. Create a video manifest from ``data/le2i`` with ``make_manifest.py``.
2. Extract YOLO-pose keypoints with ``extract_pose.py``.
3. Train either a video-level or window-level GRU classifier.

The window-level dataset uses the first two lines of each annotation file as the
fall start/end frames and creates positive windows that overlap the fall
interval and negative windows outside it. This is usually better than labeling
an entire video as fall/nonfall.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from baselines.task.common.manifest import build_label_map, filter_split, load_manifest, resolve_path


def default_le2i_label_map(labels) -> dict[str, int]:
    labels = {str(x) for x in labels if str(x)}
    if labels.issubset({"nonfall", "fall"}) and labels:
        return {"nonfall": 0, "fall": 1}
    return build_label_map(labels)


def load_keypoints_npz(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    if "keypoints" not in data:
        raise ValueError(f"{path} does not contain 'keypoints'")
    kpts = data["keypoints"].astype(np.float32)  # [T,J,3]
    if kpts.ndim != 3 or kpts.shape[-1] < 2:
        raise ValueError(f"Expected [T,J,3] keypoints, got {kpts.shape} from {path}")
    if kpts.shape[-1] == 2:
        conf = np.ones((*kpts.shape[:2], 1), dtype=np.float32)
        kpts = np.concatenate([kpts, conf], axis=-1)
    meta = {k: data[k].item() if getattr(data[k], "ndim", 1) == 0 else data[k] for k in data.files if k != "keypoints"}
    return kpts[:, :, :3], meta


def resample_keypoints(kpts: np.ndarray, sequence_len: int) -> np.ndarray:
    if len(kpts) == 0:
        return np.zeros((sequence_len, 17, 3), dtype=np.float32)
    idx = np.linspace(0, len(kpts) - 1, sequence_len).round().astype(int)
    return kpts[idx]


def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    xy = kpts[:, :, :2].copy()
    conf = kpts[:, :, 2:3]
    valid = conf[..., 0] > 0.05
    for t in range(xy.shape[0]):
        if not valid[t].any():
            continue
        pts = xy[t, valid[t]]
        center = pts.mean(axis=0, keepdims=True)
        scale = max(float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))), 1.0)
        xy[t] = (xy[t] - center) / scale
    return np.concatenate([xy, conf], axis=-1)


class PoseSequenceDataset(Dataset):
    """One training sample per video."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path | None = None,
        split: str | None = None,
        label_map: dict[str, int] | None = None,
        sequence_len: int = 64,
        normalize: bool = True,
    ) -> None:
        self.df = filter_split(load_manifest(manifest_path), split)
        self.df = self.df[self.df["label"].astype(str).str.len().gt(0)].reset_index(drop=True)
        if "keypoints_path" not in self.df.columns:
            raise ValueError("Manifest must contain keypoints_path for Le2i pose baseline")
        if "label" not in self.df.columns:
            raise ValueError("Manifest must contain label")
        self.data_root = Path(data_root) if data_root is not None else None
        self.sequence_len = sequence_len
        self.normalize = normalize
        self.label_map = label_map or default_le2i_label_map(self.df["label"].tolist())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        path = resolve_path(row["keypoints_path"], self.data_root)
        if path is None:
            raise ValueError(f"Missing keypoints_path for row {index}")
        kpts, _ = load_keypoints_npz(path)
        kpts = resample_keypoints(kpts, self.sequence_len)
        if self.normalize:
            kpts = normalize_keypoints(kpts)
        label = self.label_map[str(row["label"])]
        return torch.from_numpy(kpts.reshape(self.sequence_len, -1)), torch.tensor(label, dtype=torch.long)


@dataclass(frozen=True)
class WindowRow:
    video_index: int
    sample_id: str
    window_id: str
    start_pose: int
    end_pose: int
    start_frame: int
    end_frame: int
    label: str


class Le2iPoseWindowDataset(Dataset):
    """Window-level Le2i dataset built from per-video keypoint files.

    Each row in the manifest is a video. The dataset expands rows into temporal
    windows. A window is labeled ``fall`` if it overlaps the annotated fall
    interval by at least ``min_fall_overlap`` of the window length; otherwise it
    is labeled ``nonfall``. Videos without annotations are ignored for supervised
    splits and can be used with ``split='unlabeled'`` for prediction only.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path | None = None,
        split: str | None = None,
        label_map: dict[str, int] | None = None,
        sequence_len: int = 64,
        window_stride: int = 32,
        extraction_stride: int | None = None,
        min_fall_overlap: float = 0.10,
        normalize: bool = True,
        include_unlabeled: bool = False,
    ) -> None:
        self.video_df = filter_split(load_manifest(manifest_path), split)
        if not include_unlabeled:
            self.video_df = self.video_df[self.video_df["label"].astype(str).str.len().gt(0)].reset_index(drop=True)
        if "keypoints_path" not in self.video_df.columns:
            raise ValueError("Manifest must contain keypoints_path. Run extract_pose.py first.")
        self.data_root = Path(data_root) if data_root is not None else None
        self.sequence_len = int(sequence_len)
        self.window_stride = int(window_stride)
        self.extraction_stride = extraction_stride
        self.min_fall_overlap = float(min_fall_overlap)
        self.normalize = normalize
        self.include_unlabeled = include_unlabeled
        self.label_map = label_map or {"nonfall": 0, "fall": 1}
        self._cache: dict[int, np.ndarray] = {}
        self.windows = self._build_windows()

    def _row_extraction_stride(self, row: pd.Series, meta: dict[str, Any]) -> int:
        if self.extraction_stride is not None:
            return max(1, int(self.extraction_stride))
        if "extraction_stride" in row and not pd.isna(row["extraction_stride"]):
            return max(1, int(row["extraction_stride"]))
        if "stride" in meta:
            try:
                return max(1, int(meta["stride"]))
            except Exception:
                pass
        return 1

    def _window_label(self, start_frame: int, end_frame: int, fall_start: int, fall_end: int) -> str:
        if fall_start <= 0 or fall_end <= 0 or fall_end < fall_start:
            return "nonfall"
        overlap = max(0, min(end_frame, fall_end) - max(start_frame, fall_start) + 1)
        return "fall" if overlap / max(1, end_frame - start_frame + 1) >= self.min_fall_overlap else "nonfall"

    def _build_windows(self) -> list[WindowRow]:
        windows: list[WindowRow] = []
        for i, row in self.video_df.iterrows():
            path = resolve_path(row["keypoints_path"], self.data_root)
            if path is None or not path.exists():
                raise FileNotFoundError(f"Missing keypoints file for row {i}: {path}")
            kpts, meta = load_keypoints_npz(path)
            stride_frames = self._row_extraction_stride(row, meta)
            total = int(kpts.shape[0])
            if total <= 0:
                continue
            fall_start = int(float(row.get("fall_start", 0) or 0)) if not pd.isna(row.get("fall_start", 0)) else 0
            fall_end = int(float(row.get("fall_end", 0) or 0)) if not pd.isna(row.get("fall_end", 0)) else 0
            sample_id = str(row.get("sample_id", i))
            if total <= self.sequence_len:
                starts = [0]
            else:
                starts = list(range(0, total - self.sequence_len + 1, self.window_stride))
                if starts[-1] != total - self.sequence_len:
                    starts.append(total - self.sequence_len)
            for start_pose in starts:
                end_pose = min(start_pose + self.sequence_len - 1, total - 1)
                start_frame = start_pose * stride_frames + 1
                end_frame = end_pose * stride_frames + 1
                label = "unlabeled" if str(row.get("split", "")).lower() == "unlabeled" else self._window_label(start_frame, end_frame, fall_start, fall_end)
                if label == "unlabeled" and not self.include_unlabeled:
                    continue
                window_id = f"{sample_id}_w{start_pose:06d}_{end_pose:06d}"
                windows.append(WindowRow(i, sample_id, window_id, start_pose, end_pose, start_frame, end_frame, label))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def get_window_metadata(self, index: int) -> dict[str, Any]:
        w = self.windows[index]
        row = self.video_df.iloc[w.video_index]
        return {
            "sample_id": w.sample_id,
            "window_id": w.window_id,
            "start_pose": w.start_pose,
            "end_pose": w.end_pose,
            "start_frame": w.start_frame,
            "end_frame": w.end_frame,
            "label": w.label,
            "video_path": row.get("video_path", ""),
            "annotation_path": row.get("annotation_path", ""),
            "scenario": row.get("scenario", ""),
        }

    def _load_video_keypoints(self, video_index: int) -> np.ndarray:
        if video_index not in self._cache:
            row = self.video_df.iloc[video_index]
            path = resolve_path(row["keypoints_path"], self.data_root)
            kpts, _ = load_keypoints_npz(path)
            self._cache[video_index] = kpts
        return self._cache[video_index]

    def __getitem__(self, index: int):
        w = self.windows[index]
        kpts = self._load_video_keypoints(w.video_index)
        clip = kpts[w.start_pose : w.end_pose + 1]
        clip = resample_keypoints(clip, self.sequence_len)
        if self.normalize:
            clip = normalize_keypoints(clip)
        label_name = "nonfall" if w.label == "unlabeled" else w.label
        label = self.label_map[label_name]
        return torch.from_numpy(clip.reshape(self.sequence_len, -1)), torch.tensor(label, dtype=torch.long)
