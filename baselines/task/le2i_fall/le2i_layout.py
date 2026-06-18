"""Utilities for discovering and parsing the Le2i/ImViA fall dataset layout.

Expected user layout::

    data/le2i/Home_01/Home_01/Videos/video (1).avi
    data/le2i/Home_01/Home_01/Annotation_files/video (1).txt

Some subsets, such as Office/Lecture_room in some copies, may only have Videos.
Those videos are kept in the manifest with split='unlabeled' and are excluded
from supervised training/evaluation unless labels are added later.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg"}


def _safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "sample"


def parse_annotation_file(path: str | Path) -> dict:
    """Parse a Le2i annotation file.

    The README describes the first two lines as the fall start and fall end
    frames. Remaining rows contain per-frame localization values. We only need
    start/end for clip/window labels, but we also count annotation rows for
    sanity-checking.
    """
    path = Path(path)
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"Annotation file has fewer than two nonempty lines: {path}")

    def to_int(value: str) -> int:
        try:
            return int(float(value.split(",")[0].strip()))
        except Exception:
            return 0

    fall_start = to_int(lines[0])
    fall_end = to_int(lines[1])
    if fall_start <= 0 or fall_end <= 0 or fall_end < fall_start:
        label = "nonfall"
        fall_start = 0
        fall_end = 0
    else:
        label = "fall"
    return {
        "fall_start": fall_start,
        "fall_end": fall_end,
        "label": label,
        "annotation_rows": max(0, len(lines) - 2),
    }


def find_video_dirs(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    return sorted([p for p in data_dir.rglob("Videos") if p.is_dir()])


def scan_le2i_dataset(data_dir: str | Path, include_unlabeled: bool = True) -> pd.DataFrame:
    """Return one manifest row per video discovered under data_dir."""
    data_dir = Path(data_dir).resolve()
    rows: list[dict] = []
    for videos_dir in find_video_dirs(data_dir):
        subset_dir = videos_dir.parent
        ann_dir = subset_dir / "Annotation_files"
        # Example scenario: data/le2i/Home_01/Home_01 -> scenario Home_01.
        try:
            rel_subset = subset_dir.relative_to(data_dir)
        except ValueError:
            rel_subset = subset_dir
        scenario = rel_subset.parts[0] if rel_subset.parts else subset_dir.name

        for video_path in sorted(videos_dir.iterdir()):
            if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTS:
                continue
            ann_path = ann_dir / f"{video_path.stem}.txt"
            has_ann = ann_path.exists()
            if not has_ann and not include_unlabeled:
                continue
            parsed = {"fall_start": np.nan, "fall_end": np.nan, "label": "", "annotation_rows": 0}
            if has_ann:
                try:
                    parsed = parse_annotation_file(ann_path)
                except Exception as exc:
                    parsed = {
                        "fall_start": np.nan,
                        "fall_end": np.nan,
                        "label": "",
                        "annotation_rows": 0,
                        "annotation_error": str(exc),
                    }
            sample_id = _safe_id(f"{scenario}_{video_path.stem}")
            rows.append({
                "sample_id": sample_id,
                "scenario": scenario,
                "subset_path": str(rel_subset),
                "video_path": str(video_path),
                "annotation_path": str(ann_path) if has_ann else "",
                "has_annotation": bool(has_ann),
                "label": parsed.get("label", ""),
                "fall_start": parsed.get("fall_start", np.nan),
                "fall_end": parsed.get("fall_end", np.nan),
                "annotation_rows": parsed.get("annotation_rows", 0),
                "split": "unassigned" if has_ann and parsed.get("label", "") else "unlabeled",
            })
    if not rows:
        raise FileNotFoundError(f"No videos found under {data_dir}; expected nested */*/Videos/*.avi")
    return pd.DataFrame(rows)


def assign_labeled_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 13,
) -> pd.DataFrame:
    """Assign train/val/test splits to labeled videos and keep unlabeled videos separate."""
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")
    out = df.copy()
    labeled_mask = out["label"].astype(str).str.len().gt(0) & out["has_annotation"].astype(bool)
    labeled_idx = out.index[labeled_mask].to_numpy()
    if len(labeled_idx) == 0:
        out.loc[:, "split"] = "unlabeled"
        return out

    labels = out.loc[labeled_idx, "label"].astype(str).to_numpy()
    can_stratify = len(set(labels)) > 1 and min(pd.Series(labels).value_counts()) >= 2

    if len(labeled_idx) < 4:
        out.loc[labeled_idx, "split"] = "train"
        out.loc[~labeled_mask, "split"] = "unlabeled"
        return out

    strat = labels if can_stratify else None
    train_idx, tmp_idx = train_test_split(
        labeled_idx,
        train_size=train_frac,
        random_state=seed,
        shuffle=True,
        stratify=strat,
    )
    tmp_labels = out.loc[tmp_idx, "label"].astype(str).to_numpy()
    can_stratify_tmp = len(set(tmp_labels)) > 1 and min(pd.Series(tmp_labels).value_counts()) >= 2
    rel_val_frac = val_frac / (val_frac + test_frac)
    if len(tmp_idx) < 2:
        val_idx, test_idx = tmp_idx, np.array([], dtype=tmp_idx.dtype)
    else:
        val_idx, test_idx = train_test_split(
            tmp_idx,
            train_size=rel_val_frac,
            random_state=seed,
            shuffle=True,
            stratify=tmp_labels if can_stratify_tmp else None,
        )

    out.loc[train_idx, "split"] = "train"
    out.loc[val_idx, "split"] = "val"
    out.loc[test_idx, "split"] = "test"
    out.loc[~labeled_mask, "split"] = "unlabeled"
    return out.reset_index(drop=True)


def make_manifest(
    data_dir: str | Path,
    output_csv: str | Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 13,
    include_unlabeled: bool = True,
) -> pd.DataFrame:
    df = scan_le2i_dataset(data_dir, include_unlabeled=include_unlabeled)
    df = assign_labeled_splits(df, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=seed)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a manifest for the nested Le2i/ImViA dataset layout.")
    parser.add_argument("--data-dir", default="data/le2i")
    parser.add_argument("--output-csv", default="data/le2i/le2i_manifest.csv")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--exclude-unlabeled", action="store_true")
    args = parser.parse_args()
    df = make_manifest(
        args.data_dir,
        args.output_csv,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        include_unlabeled=not args.exclude_unlabeled,
    )
    print(f"Wrote {args.output_csv}")
    print(df.groupby(["split", "label"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
