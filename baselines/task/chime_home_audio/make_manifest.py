"""Create a manifest for CHiME-Home domestic audio tagging.

Expected dataset layout:

    data/chime_home/
      development_chunks_refined.csv
      evaluation_chunks_refined.csv
      chunks/
        <chunkname>.csv
        <chunkname>.16kHz.wav
        <chunkname>.48kHz.wav

The top-level CSVs have two columns: numeric id and chunk name. Per-chunk CSVs
contain metadata and the majority-vote annotation string.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .labels import LABELS, LABEL_MEANINGS, encode_label_string


def read_chunk_list(path: Path) -> list[str]:
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}, found {df.shape[1]}")
    return [str(x).strip() for x in df.iloc[:, 1].tolist()]


def read_metadata(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                key = row[0].strip()
                val = ",".join(row[1:]).strip()
                meta[key] = val
    return meta


def make_row(root: Path, chunkname: str, split: str, sample_rate_khz: int) -> dict[str, Any]:
    chunks_dir = root / "chunks"
    meta_path = chunks_dir / f"{chunkname}.csv"
    wav16 = chunks_dir / f"{chunkname}.16kHz.wav"
    wav48 = chunks_dir / f"{chunkname}.48kHz.wav"
    selected = wav16 if sample_rate_khz == 16 else wav48
    meta = read_metadata(meta_path)
    majority = meta.get("majorityvote", "")
    row: dict[str, Any] = {
        "chunk_id": chunkname,
        "split": split,
        "audio_path": str(selected),
        "audio_16k_path": str(wav16),
        "audio_48k_path": str(wav48),
        "metadata_path": str(meta_path),
        "majorityvote": majority,
        "segmentname": meta.get("segmentname", ""),
        "chunknumber": meta.get("chunknumber", ""),
        "framestart": meta.get("framestart", ""),
        "annotation_a1": meta.get("annotation_a1", ""),
        "annotation_a2": meta.get("annotation_a2", ""),
        "annotation_a3": meta.get("annotation_a3", ""),
    }
    row.update(encode_label_string(majority))
    row["audio_exists"] = selected.exists()
    row["metadata_exists"] = meta_path.exists()
    return row


def split_development(chunks: list[str], val_frac: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = list(chunks)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_frac)) if val_frac > 0 else 0
    val_set = set(shuffled[:n_val])
    return {c: ("val" if c in val_set else "train") for c in chunks}


def main() -> None:
    p = argparse.ArgumentParser(description="Build CHiME-Home manifest")
    p.add_argument("--root", type=Path, default=Path("data/chime_home"))
    p.add_argument("--dev-csv", type=Path, default=None,
                   help="Defaults to <root>/development_chunks_refined.csv")
    p.add_argument("--eval-csv", type=Path, default=None,
                   help="Defaults to <root>/evaluation_chunks_refined.csv")
    p.add_argument("--output-csv", type=Path, default=Path("data/chime_home/chime_home_manifest.csv"))
    p.add_argument("--summary-json", type=Path, default=Path("data/chime_home/chime_home_manifest_summary.json"))
    p.add_argument("--sample-rate-khz", type=int, choices=[16, 48], default=16)
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Fraction of development chunks used as val; evaluation chunks are test")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--drop-missing-audio", action="store_true")
    args = p.parse_args()

    root = args.root
    dev_csv = args.dev_csv or root / "development_chunks_refined.csv"
    eval_csv = args.eval_csv or root / "evaluation_chunks_refined.csv"
    if not dev_csv.exists():
        raise FileNotFoundError(f"Missing development CSV: {dev_csv}")
    if not eval_csv.exists():
        raise FileNotFoundError(f"Missing evaluation CSV: {eval_csv}")

    dev_chunks = read_chunk_list(dev_csv)
    eval_chunks = read_chunk_list(eval_csv)
    dev_splits = split_development(dev_chunks, args.val_frac, args.seed)

    rows: list[dict[str, Any]] = []
    for chunk in dev_chunks:
        rows.append(make_row(root, chunk, dev_splits[chunk], args.sample_rate_khz))
    for chunk in eval_chunks:
        rows.append(make_row(root, chunk, "test", args.sample_rate_khz))

    df = pd.DataFrame(rows)
    if args.drop_missing_audio:
        df = df[df["audio_exists"].astype(bool)].copy()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    summary = {
        "root": str(root),
        "sample_rate_khz": args.sample_rate_khz,
        "n_rows": int(len(df)),
        "splits": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "missing_audio": int((~df["audio_exists"].astype(bool)).sum()),
        "missing_metadata": int((~df["metadata_exists"].astype(bool)).sum()),
        "labels": LABELS,
        "label_meanings": LABEL_MEANINGS,
        "label_counts": {lab: int(df[f"label_{lab}"].sum()) for lab in LABELS},
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote manifest: {args.output_csv}")


if __name__ == "__main__":
    main()
