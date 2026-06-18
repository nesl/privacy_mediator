"""Video/frame loading helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def sorted_images(frame_dir: str | Path) -> list[Path]:
    frame_dir = Path(frame_dir)
    if not frame_dir.exists():
        raise FileNotFoundError(frame_dir)
    return sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def sample_indices(n: int, num_samples: int) -> list[int]:
    if n <= 0:
        return []
    if num_samples <= 1:
        return [n // 2]
    return np.linspace(0, n - 1, num_samples).round().astype(int).tolist()


def load_frame_sequence(path: str | Path, num_frames: int = 8) -> list[Image.Image]:
    """Load a small sequence from either a video file or a frame directory."""
    path = Path(path)
    frames: list[Image.Image] = []
    if path.is_dir():
        imgs = sorted_images(path)
        for idx in sample_indices(len(imgs), num_frames):
            frames.append(Image.open(imgs[idx]).convert("RGB"))
        return frames
    if path.suffix.lower() not in VID_EXTS:
        return [Image.open(path).convert("RGB")]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in sample_indices(total, num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames


def iter_video_frames(path: str | Path, stride: int = 1, max_frames: int | None = None):
    path = Path(path)
    if path.is_dir():
        count = 0
        for p in sorted_images(path):
            if count % stride == 0:
                img = cv2.imread(str(p))
                if img is not None:
                    yield count, img
            count += 1
            if max_frames is not None and count >= max_frames:
                break
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            yield idx, frame
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break
        idx += 1
    cap.release()
