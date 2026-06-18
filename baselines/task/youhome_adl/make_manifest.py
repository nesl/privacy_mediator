from __future__ import annotations

import argparse
from pathlib import Path

from baselines.task.youhome_adl.youhome_layout import build_manifest, write_manifest_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manifest for the nested YouHome ADL data/label layout.")
    parser.add_argument("--root", default="data/youhome", help="Path containing data/ and label/ subdirectories")
    parser.add_argument("--output-csv", default="data/youhome/youhome_manifest.csv")
    parser.add_argument("--summary-json", default="data/youhome/youhome_manifest_summary.json")
    parser.add_argument("--split-by", choices=["participant", "session", "sample", "activity_stratified"], default="participant")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-images", type=int, default=1)
    args = parser.parse_args()

    df = build_manifest(
        args.root,
        split_by=args.split_by,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        min_images=args.min_images,
    )
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    write_manifest_summary(df, args.summary_json)
    print(f"Wrote {len(df)} samples to {out}")
    print(f"Wrote summary to {args.summary_json}")
    if len(df):
        print(df["split"].value_counts().to_string())
        missing = set(df[df["split"] != "train"]["label"].astype(str)) - set(df[df["split"] == "train"]["label"].astype(str))
        if missing:
            print("WARNING: labels present outside train but missing in train:", sorted(missing))
            print("Consider --split-by activity_stratified if you need every activity in train.")


if __name__ == "__main__":
    main()
