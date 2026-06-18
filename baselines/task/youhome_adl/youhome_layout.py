"""Utilities for the nested YouHome ADL data/label layout.

Expected layout:

    data/youhome/
      data/<participant>/<activity>/<session>/
          <sample>.mp4
          <sample>_1.jpg ...
          audio.wav
      label/<participant>/<activity>/<session>/
          <sample>_1.txt ...   # YOLO-style person boxes/user ids

Activity labels are taken from the <activity> folder name. The txt labels are
not used as activity labels; they are optional frame-level person/user boxes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import json
import random
import re

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class YouHomeSample:
    sample_id: str
    participant: str
    activity: str
    session: str
    split: str
    label: str
    frame_dir: str
    label_dir: str
    audio_path: str
    video_path: str
    n_images: int
    n_label_files: int
    has_audio: bool
    has_video: bool
    has_labels: bool


def _safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "unknown"


def infer_data_label_dirs(root: str | Path) -> tuple[Path, Path]:
    root = Path(root)
    data_dir = root / "data"
    label_dir = root / "label"
    if not data_dir.exists():
        raise FileNotFoundError(f"Expected YouHome data directory at {data_dir}")
    if not label_dir.exists():
        # Labels are optional for ADL training; keep path for relative matching.
        label_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, label_dir


def find_session_dirs(data_dir: Path) -> list[tuple[str, str, str, Path]]:
    """Return (participant, activity, session, session_dir) rows."""
    rows: list[tuple[str, str, str, Path]] = []
    for participant_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        participant = participant_dir.name
        for activity_dir in sorted(p for p in participant_dir.iterdir() if p.is_dir()):
            activity = activity_dir.name
            for session_dir in sorted(p for p in activity_dir.iterdir() if p.is_dir()):
                # A session dir should contain images/video/audio. If there are
                # extra nesting levels, the recursive manifest builder can be
                # extended later, but this matches p101/Write/c1s0.
                rows.append((participant, activity, session_dir.name, session_dir))
    return rows


def list_images(session_dir: str | Path) -> list[Path]:
    p = Path(session_dir)
    return sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS)


def find_video(session_dir: str | Path) -> Path | None:
    p = Path(session_dir)
    videos = sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in VIDEO_EXTS)
    return videos[0] if videos else None


def find_audio(session_dir: str | Path) -> Path | None:
    p = Path(session_dir)
    preferred = p / "audio.wav"
    if preferred.exists():
        return preferred
    wavs = sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a"})
    return wavs[0] if wavs else None


def relative_or_abs(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def assign_splits(
    items: list[tuple[str, str, str, Path]],
    split_by: str = "participant",
    seed: int = 13,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> dict[tuple[str, str, str], str]:
    if not items:
        return {}
    rng = random.Random(seed)
    train_frac = float(train_frac)
    val_frac = float(val_frac)
    test_frac = float(test_frac)
    total = train_frac + val_frac + test_frac
    if total <= 0:
        raise ValueError("train/val/test fractions must sum to positive value")
    train_cut = train_frac / total
    val_cut = (train_frac + val_frac) / total

    split_by = split_by.lower()
    key_to_items: dict[str, list[tuple[str, str, str, Path]]] = {}
    if split_by == "participant":
        for it in items:
            key_to_items.setdefault(it[0], []).append(it)
    elif split_by == "session":
        for it in items:
            key_to_items.setdefault(f"{it[0]}/{it[2]}", []).append(it)
    elif split_by == "sample":
        for it in items:
            key_to_items.setdefault(f"{it[0]}/{it[1]}/{it[2]}", []).append(it)
    elif split_by == "activity_stratified":
        # Split within each activity. This maximizes train coverage of labels,
        # at the cost of participant/session leakage.
        out: dict[tuple[str, str, str], str] = {}
        by_act: dict[str, list[tuple[str, str, str, Path]]] = {}
        for it in items:
            by_act.setdefault(it[1], []).append(it)
        for act_items in by_act.values():
            shuffled = list(act_items)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_train = max(1, int(round(n * train_cut))) if n >= 3 else max(1, n - 1)
            n_val = int(round(n * (val_frac / total))) if n >= 5 else 0
            for i, it in enumerate(shuffled):
                if i < n_train:
                    split = "train"
                elif i < n_train + n_val:
                    split = "val"
                else:
                    split = "test"
                out[(it[0], it[1], it[2])] = split
        return out
    else:
        raise ValueError(f"Unsupported split_by={split_by!r}")

    keys = list(key_to_items.keys())
    rng.shuffle(keys)
    n_keys = len(keys)
    n_train = max(1, int(round(n_keys * train_cut))) if n_keys >= 3 else max(1, n_keys - 1)
    n_val = int(round(n_keys * (val_frac / total))) if n_keys >= 5 else 0
    out: dict[tuple[str, str, str], str] = {}
    for idx, key in enumerate(keys):
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "val"
        else:
            split = "test"
        for it in key_to_items[key]:
            out[(it[0], it[1], it[2])] = split
    return out


def build_manifest(
    root: str | Path,
    split_by: str = "participant",
    seed: int = 13,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    min_images: int = 1,
) -> pd.DataFrame:
    root = Path(root)
    data_dir, label_root = infer_data_label_dirs(root)
    session_rows = find_session_dirs(data_dir)
    split_map = assign_splits(session_rows, split_by, seed, train_frac, val_frac, test_frac)
    out: list[dict] = []
    for participant, activity, session, session_dir in session_rows:
        images = list_images(session_dir)
        if len(images) < min_images:
            continue
        rel = session_dir.relative_to(data_dir)
        label_dir = label_root / rel
        video = find_video(session_dir)
        audio = find_audio(session_dir)
        label_files = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        sample_id = _safe_id(f"{participant}_{activity}_{session}")
        row = YouHomeSample(
            sample_id=sample_id,
            participant=participant,
            activity=activity,
            session=session,
            split=split_map.get((participant, activity, session), "train"),
            label=activity,
            frame_dir=relative_or_abs(session_dir, root),
            label_dir=relative_or_abs(label_dir, root),
            audio_path=relative_or_abs(audio, root),
            video_path=relative_or_abs(video, root),
            n_images=len(images),
            n_label_files=len(label_files),
            has_audio=audio is not None,
            has_video=video is not None,
            has_labels=bool(label_files),
        )
        out.append(asdict(row))
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values(["participant", "activity", "session"]).reset_index(drop=True)
    return df


def parse_yolo_label_file(path: str | Path) -> list[dict]:
    """Parse YOLO-style txt rows: class x_center y_center width height.

    Coordinates are normalized to [0,1]. The class id is a user/person id for
    YouHome, not the ADL activity label.
    """
    path = Path(path)
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            xc, yc, w, h = [float(x) for x in parts[1:5]]
        except ValueError:
            continue
        rows.append({"class_id": class_id, "x_center": xc, "y_center": yc, "width": w, "height": h, "line_no": line_no})
    return rows


def label_path_for_image(image_path: Path, label_dir: Path) -> Path:
    return label_dir / f"{image_path.stem}.txt"


def write_manifest_summary(df: pd.DataFrame, path: str | Path) -> None:
    summary: dict = {"n_samples": int(len(df))}
    if len(df):
        summary["split_counts"] = {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()}
        summary["n_classes"] = int(df["label"].nunique())
        summary["class_counts"] = {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}
        summary["participants"] = sorted(df["participant"].astype(str).unique().tolist())
        summary["has_audio_rate"] = float(df["has_audio"].mean())
        summary["has_labels_rate"] = float(df["has_labels"].mean())
        train_labels = set(df[df["split"] == "train"]["label"].astype(str))
        other_labels = set(df[df["split"] != "train"]["label"].astype(str))
        summary["labels_missing_from_train"] = sorted(other_labels - train_labels)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
