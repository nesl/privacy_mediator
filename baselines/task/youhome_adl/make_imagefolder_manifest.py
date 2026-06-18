"""Create a manifest from an ImageFolder-style YouHome activity directory.

Expected layout:
    root/train/<activity_id>/*.jpg
    root/val/<activity_id>/*.jpg
    root/test/<activity_id>/*.jpg

Optional audio lookup: if --audio-root is provided, this script tries to match an
audio file by sample stem: <audio-root>/<split>/<activity_id>/<stem>.wav.
You can replace/repair audio_path later once the exact data layout is known.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio-root", default=None)
    args = parser.parse_args()
    image_root = Path(args.image_root)
    audio_root = Path(args.audio_root) if args.audio_root else None
    rows = []
    for split_dir in image_root.iterdir():
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for img in sorted(class_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                audio_path = ""
                if audio_root:
                    cand = audio_root / split / label / f"{img.stem}.wav"
                    if cand.exists():
                        audio_path = str(cand)
                rows.append({
                    "sample_id": f"{split}_{label}_{img.stem}",
                    "split": split,
                    "label": label,
                    "image_path": str(img),
                    "audio_path": audio_path,
                })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
