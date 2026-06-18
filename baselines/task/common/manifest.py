"""Manifest helpers for task baselines.

All baselines use a CSV/JSONL manifest so that dataset-specific ingestion can be
added later without rewriting the training code.  A typical CSV has columns such
as:

    sample_id,split,label,video_path,frame_dir,image_path,audio_path,keypoints_path

Paths can be absolute or relative to --data-root.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import json

import pandas as pd


@dataclass(frozen=True)
class ManifestSpec:
    label_col: str = "label"
    split_col: str = "split"
    id_col: str = "sample_id"


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def resolve_path(value: object, data_root: str | Path | None = None) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    path = Path(text)
    if not path.is_absolute() and data_root is not None:
        path = Path(data_root) / path
    return path


def filter_split(df: pd.DataFrame, split: str | None, split_col: str = "split") -> pd.DataFrame:
    if split is None or split_col not in df.columns:
        return df.reset_index(drop=True)
    return df[df[split_col].astype(str).str.lower() == split.lower()].reset_index(drop=True)


def build_label_map(labels: Iterable[object]) -> dict[str, int]:
    unique = sorted({str(x) for x in labels})
    return {label: i for i, label in enumerate(unique)}


def save_label_map(label_map: dict[str, int], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)


def load_label_map(path: str | Path) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}
