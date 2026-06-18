"""Convert frame-level YOLO predictions to clip-level presence events."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

OUT_COLUMNS = ["sample_id", "person_present", "positive_frames", "person_count"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--min-positive-frames", type=int, default=1)
    parser.add_argument("--count-agg", choices=["max", "mean"], default="max")
    parser.add_argument("--allow-empty-output", action="store_true")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.predictions)
    except EmptyDataError as e:
        raise SystemExit(f"Predictions CSV is empty or has no header: {args.predictions}") from e

    rows = []
    if not df.empty:
        for sample_id, g in df.groupby("sample_id"):
            positives = int(g["person_present"].astype(int).sum())
            if args.count_agg == "max":
                count = int(g["person_count"].max())
            else:
                count = float(g["person_count"].mean())
            rows.append(
                {
                    "sample_id": sample_id,
                    "person_present": int(positives >= args.min_positive_frames),
                    "positive_frames": positives,
                    "person_count": count,
                }
            )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUT_COLUMNS).to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv} ({len(rows)} rows)")
    if not rows and not args.allow_empty_output:
        raise SystemExit("No events were produced because the predictions CSV has zero rows.")


if __name__ == "__main__":
    main()
