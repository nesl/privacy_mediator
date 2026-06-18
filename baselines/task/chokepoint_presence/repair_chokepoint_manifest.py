"""Repair a ChokePoint manifest by rediscovering frame directories on disk.

This is useful when the manifest has labels/XML rows but blank ``frame_dir`` or
incorrect ``has_images`` values. It scans common ChokePoint layouts such as:

    DATA_ROOT/GROUP_ID/SAMPLE_ID/*.jpg
    DATA_ROOT/SAMPLE_ID/*.jpg
    DATA_ROOT/GROUP_ID/XML_STEM/*.jpg
    DATA_ROOT/XML_STEM/*.jpg

Example:
    python repair_chokepoint_manifest.py \
      --manifest data/chokepoint/chokepoint_manifest.csv \
      --data-root data/chokepoint \
      --output data/chokepoint/chokepoint_manifest_repaired.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def count_images(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def rel_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def candidate_dirs(row: pd.Series, root: Path, recursive_search: bool = False) -> Iterable[Path]:
    # Existing explicit frame_dir first.
    if "frame_dir" in row and not is_missing(row.get("frame_dir")):
        frame_dir = Path(str(row.get("frame_dir")))
        yield frame_dir if frame_dir.is_absolute() else root / frame_dir
        yield frame_dir

    sample_id = None if is_missing(row.get("sample_id")) else str(row.get("sample_id")).strip()
    group_id = None if is_missing(row.get("group_id")) else str(row.get("group_id")).strip()
    xml_stem = None
    if not is_missing(row.get("xml_path")):
        xml_stem = Path(str(row.get("xml_path"))).stem

    names = []
    for value in (sample_id, xml_stem):
        if value and value not in names:
            names.append(value)

    groups = []
    if group_id:
        groups.append(group_id)

    for name in names:
        for group in groups:
            yield root / group / name
        yield root / name

    if recursive_search:
        for name in names:
            yield from (p for p in root.rglob(name) if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recursive-search", action="store_true", help="Slower fallback: recursively search for sample_id/xml_stem directories")
    args = parser.parse_args()

    root = Path(args.data_root)
    df = pd.read_csv(args.manifest)

    if "frame_dir" not in df.columns:
        df["frame_dir"] = pd.NA
    if "n_images" not in df.columns:
        df["n_images"] = 0

    found = 0
    missing = []

    for i, row in df.iterrows():
        best_path = None
        best_count = 0
        seen = set()
        for candidate in candidate_dirs(row, root, recursive_search=args.recursive_search):
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            n = count_images(candidate)
            if n > best_count:
                best_path = candidate
                best_count = n

        if best_path is not None and best_count > 0:
            df.at[i, "frame_dir"] = rel_to_root(best_path, root)
            df.at[i, "n_images"] = int(best_count)
            found += 1
        else:
            df.at[i, "frame_dir"] = pd.NA
            df.at[i, "n_images"] = 0
            missing.append(row.get("sample_id", i))

    df["has_images"] = (df["n_images"].fillna(0).astype(int) > 0).astype(int)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {out}")
    print(f"Rows with images: {found}/{len(df)}")
    if "split" in df.columns:
        print(df.groupby("split")["has_images"].sum().to_string())
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        suffix = "..." if len(missing) > 10 else ""
        print(f"Rows still missing images: {len(missing)} ({preview}{suffix})")


if __name__ == "__main__":
    main()
