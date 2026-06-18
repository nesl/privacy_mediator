from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import LABELS
from .metrics import compute_multilabel_metrics, write_json


def main() -> None:
    p = argparse.ArgumentParser(description="Tune scalar threshold from an inference CSV")
    p.add_argument("--predictions", type=Path, required=True,
                   help="CSV produced by infer.py on the val split")
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--metric", choices=["macro_f1", "micro_f1", "samples_f1"], default="macro_f1")
    p.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    args = p.parse_args()
    df = pd.read_csv(args.predictions)
    y_true = df[[f"true_{lab}" for lab in LABELS]].to_numpy(dtype=int)
    y_score = df[[f"score_{lab}" for lab in LABELS]].to_numpy(dtype=float)
    best = None
    for t in [float(x) for x in args.thresholds.split(",") if x.strip()]:
        m = compute_multilabel_metrics(y_true, y_score, threshold=t)
        rec = {"threshold": t, "metrics": m}
        if best is None or m[args.metric] > best["metrics"][args.metric]:
            best = rec
    write_json(args.output_json, best)
    print(best)


if __name__ == "__main__":
    main()
