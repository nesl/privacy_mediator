"""Dataset ingestion helpers for the ChokePoint portal/visitor-monitoring baseline.

Expected extracted layout::

    data/chokepoint/
      groundtruth/
        P1E_S1_C1.xml
        ...
      P1E_S1/
        P1E_S1_C1/
          00000000.jpg
          00000001.jpg
          ...
      P1E_S1.tar.xz
      groundtruth.tar.xz

The XML ground-truth files annotate frames with zero or more ``person`` nodes.
For the visitor-monitoring task we convert this into frame-level labels:
``person_present`` and ``person_count``.
"""
from __future__ import annotations

import argparse
import json
import random
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ChokePointSample:
    sample_id: str
    frame_dir: Path | None
    xml_path: Path
    group_id: str
    n_images: int
    n_xml_frames: int
    n_positive_frames: int
    n_negative_frames: int


def _rel(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def numeric_frame_index(path: Path, fallback: int) -> int:
    """Use numeric image stems such as 00000233.jpg as the frame index."""
    try:
        return int(path.stem)
    except ValueError:
        return int(fallback)


def sorted_images(frame_dir: Path | None) -> list[Path]:
    if frame_dir is None or not frame_dir.exists() or not frame_dir.is_dir():
        return []
    return sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def parse_groundtruth_xml(xml_path: str | Path) -> pd.DataFrame:
    """Parse one ChokePoint XML file into frame-level visitor labels.

    Returns columns:
      sample_id, frame_index, person_present, person_count, person_ids_json,
      eye_points_json

    ``eye_points_json`` stores available left/right-eye points because those are
    useful for optional face/identity leakage probes, though the primary utility
    evaluation only uses presence/count.
    """
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    sample_id = root.attrib.get("name") or xml_path.stem
    rows: list[dict] = []
    for frame in root.findall(".//frame"):
        number = frame.attrib.get("number", "0")
        try:
            frame_index = int(number)
        except ValueError:
            frame_index = len(rows)
        person_ids: list[str] = []
        eye_points: list[dict] = []
        for person in frame.findall("person"):
            pid = str(person.attrib.get("id", ""))
            if pid:
                person_ids.append(pid)
            points: dict[str, object] = {"id": pid}
            for tag in ("leftEye", "rightEye"):
                node = person.find(tag)
                if node is not None:
                    try:
                        points[tag] = {"x": int(node.attrib.get("x", "0")), "y": int(node.attrib.get("y", "0"))}
                    except ValueError:
                        points[tag] = dict(node.attrib)
            if len(points) > 1:
                eye_points.append(points)
        rows.append(
            {
                "sample_id": sample_id,
                "frame_index": frame_index,
                "person_present": int(len(person_ids) > 0),
                "person_count": int(len(person_ids)),
                "person_ids_json": json.dumps(person_ids),
                "eye_points_json": json.dumps(eye_points),
            }
        )
    return pd.DataFrame(rows)


def contiguous_positive_events(labels: pd.DataFrame, max_gap: int = 1) -> pd.DataFrame:
    """Convert frame labels into positive presence intervals per sample."""
    rows: list[dict] = []
    if labels.empty:
        return pd.DataFrame(columns=["sample_id", "event_start", "event_end", "duration_frames", "positive_labeled_frames", "max_person_count"])
    for sample_id, g in labels.groupby("sample_id"):
        pos = g[g["person_present"].astype(int) > 0].sort_values("frame_index")
        if pos.empty:
            continue
        start = prev = int(pos.iloc[0]["frame_index"])
        max_count = int(pos.iloc[0]["person_count"])
        n_frames = 1
        for _, row in pos.iloc[1:].iterrows():
            idx = int(row["frame_index"])
            if idx <= prev + max_gap:
                prev = idx
                max_count = max(max_count, int(row["person_count"]))
                n_frames += 1
            else:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "event_start": start,
                        "event_end": prev,
                        "duration_frames": prev - start + 1,
                        "positive_labeled_frames": n_frames,
                        "max_person_count": max_count,
                    }
                )
                start = prev = idx
                max_count = int(row["person_count"])
                n_frames = 1
        rows.append(
            {
                "sample_id": sample_id,
                "event_start": start,
                "event_end": prev,
                "duration_frames": prev - start + 1,
                "positive_labeled_frames": n_frames,
                "max_person_count": max_count,
            }
        )
    return pd.DataFrame(rows)


def derive_group_id(sample_id: str) -> str:
    """Group cameras from the same sequence together for splitting.

    Examples:
      P1E_S1_C1 -> P1E_S1
      P2L_S3_C2 -> P2L_S3
    """
    parts = sample_id.split("_")
    if len(parts) >= 3 and parts[-1].startswith("C"):
        return "_".join(parts[:-1])
    return "_".join(parts[:2]) if len(parts) >= 2 else sample_id


def _safe_extract_tar(tf: tarfile.TarFile, target_dir: Path) -> None:
    """Extract a tar archive while preventing path traversal entries."""
    target_dir = target_dir.resolve()
    for member in tf.getmembers():
        member_path = (target_dir / member.name).resolve()
        if member_path != target_dir and not str(member_path).startswith(str(target_dir) + str(Path("/"))):
            raise RuntimeError(f"Unsafe archive member path: {member.name}")
    tf.extractall(target_dir)


def _archive_image_parent_names(archive: Path) -> set[str]:
    """Return leaf directory names in an archive that contain image files.

    This lets us detect partial extractions. The previous implementation skipped
    an archive whenever ``data_dir/<archive-name>`` existed, which misses cases
    where only one camera directory was extracted from a group archive.
    """
    names: set[str] = set()
    try:
        with tarfile.open(archive, "r:xz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                p = Path(member.name)
                if p.suffix.lower() in IMG_EXTS and p.parent.name:
                    names.add(p.parent.name)
    except tarfile.TarError:
        return set()
    return names


def _archive_contains_xml(archive: Path) -> bool:
    try:
        with tarfile.open(archive, "r:xz") as tf:
            return any(member.isfile() and Path(member.name).suffix.lower() == ".xml" for member in tf.getmembers())
    except tarfile.TarError:
        return False


def maybe_extract_archives(data_dir: str | Path, overwrite: bool = False, skip_existing_complete: bool = True) -> None:
    """Extract .tar.xz archives under ``data_dir`` before scanning.

    Archives are extracted into the directory that contains the archive. Unlike
    the older version, this does not blindly skip a group archive only because
    its top-level folder already exists. It inspects the archive and extracts it
    if any expected camera/image directory is missing, which fixes partial
    extraction states such as having only ``P1E_S1_C1`` present.
    """
    data_dir = Path(data_dir)
    archives = sorted(data_dir.rglob("*.tar.xz"))
    if not archives:
        print(f"No .tar.xz archives found under {data_dir}; scanning extracted folders.")
        return

    frame_dirs = find_frame_dirs(data_dir)
    gt_dirs = find_groundtruth_dirs(data_dir)
    has_xml = any(any(p.glob("*.xml")) for p in gt_dirs)

    for archive in archives:
        target_parent = archive.parent
        if not overwrite and skip_existing_complete:
            image_dir_names = _archive_image_parent_names(archive)
            if image_dir_names:
                missing = sorted(name for name in image_dir_names if name not in frame_dirs)
                if not missing:
                    print(f"Skipping extraction for {archive}: all {len(image_dir_names)} image directories already exist")
                    continue
                print(f"Extracting {archive}: missing {len(missing)} image directories, e.g. {missing[:5]}")
            elif _archive_contains_xml(archive) and has_xml:
                print(f"Skipping extraction for {archive}: XML groundtruth already exists")
                continue
            else:
                target_name = archive.name[: -len(".tar.xz")]
                target = target_parent / target_name
                if target.exists():
                    print(f"Skipping extraction for {archive}: {target} already exists")
                    continue
        else:
            print(f"Extracting {archive} -> {target_parent}")

        with tarfile.open(archive, "r:xz") as tf:
            _safe_extract_tar(tf, target_parent)
        # Refresh indexes after each extraction so later archives can be skipped.
        frame_dirs = find_frame_dirs(data_dir)
        gt_dirs = find_groundtruth_dirs(data_dir)
        has_xml = any(any(p.glob("*.xml")) for p in gt_dirs)


def find_groundtruth_dirs(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    preferred = data_dir / "groundtruth"
    dirs: list[Path] = []
    if preferred.exists() and preferred.is_dir():
        dirs.append(preferred)
    for p in sorted(data_dir.rglob("groundtruth")):
        if p.is_dir() and p not in dirs:
            dirs.append(p)
    return dirs


def find_frame_dirs(data_dir: str | Path) -> dict[str, Path]:
    """Find leaf image directories and map directory name -> path.

    If duplicate directory names are found, keep the one with more image files.
    """
    data_dir = Path(data_dir)
    frame_dirs: dict[str, Path] = {}
    image_counts: dict[str, int] = {}
    for p in data_dir.rglob("*"):
        if not p.is_dir():
            continue
        if p.name == "groundtruth" or "groundtruth" in p.parts:
            continue
        imgs = sorted_images(p)
        if imgs and len(imgs) > image_counts.get(p.name, -1):
            frame_dirs[p.name] = p
            image_counts[p.name] = len(imgs)
    return frame_dirs


def discover_samples(data_dir: str | Path) -> tuple[list[ChokePointSample], pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    gt_dirs = find_groundtruth_dirs(data_dir)
    if not gt_dirs:
        raise FileNotFoundError(f"Could not find a groundtruth directory under: {data_dir}")
    xml_paths: list[Path] = []
    for gt_dir in gt_dirs:
        xml_paths.extend(sorted(gt_dir.rglob("*.xml")))
    if not xml_paths:
        raise FileNotFoundError(f"Could not find any XML files under groundtruth directories: {gt_dirs}")

    frame_dirs = find_frame_dirs(data_dir)
    samples: list[ChokePointSample] = []
    all_labels: list[pd.DataFrame] = []
    for xml_path in sorted(set(xml_paths)):
        labels = parse_groundtruth_xml(xml_path)
        if labels.empty:
            continue
        sample_id = str(labels.iloc[0]["sample_id"])
        frame_dir = frame_dirs.get(sample_id)
        n_pos = int(labels["person_present"].astype(int).sum())
        sample = ChokePointSample(
            sample_id=sample_id,
            frame_dir=frame_dir,
            xml_path=xml_path,
            group_id=derive_group_id(sample_id),
            n_images=len(sorted_images(frame_dir)) if frame_dir is not None else 0,
            n_xml_frames=int(len(labels)),
            n_positive_frames=n_pos,
            n_negative_frames=int(len(labels) - n_pos),
        )
        samples.append(sample)
        all_labels.append(labels)
    labels_df = pd.concat(all_labels, ignore_index=True) if all_labels else pd.DataFrame()
    events_df = contiguous_positive_events(labels_df) if not labels_df.empty else pd.DataFrame()
    return samples, labels_df, events_df


def assign_splits(samples: list[ChokePointSample], seed: int = 13, val_frac: float = 0.0, test_frac: float = 0.25) -> dict[str, str]:
    """Assign splits at group/sequence level so cameras from a sequence stay together."""
    groups = sorted({s.group_id for s in samples})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    if n == 0:
        return {}
    n_test = max(1, round(n * test_frac)) if test_frac > 0 else 0
    n_val = max(1, round(n * val_frac)) if val_frac > 0 else 0
    # Keep at least one train group when possible.
    if n_test + n_val >= n and n > 1:
        overflow = n_test + n_val - (n - 1)
        n_test = max(0, n_test - overflow)
    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test : n_test + n_val])
    split_by_group = {}
    for g in groups:
        if g in test_groups:
            split_by_group[g] = "test"
        elif g in val_groups:
            split_by_group[g] = "val"
        else:
            split_by_group[g] = "train"
    return split_by_group


def build_manifest(
    data_dir: str | Path,
    seed: int = 13,
    val_frac: float = 0.0,
    test_frac: float = 0.25,
    drop_missing_images: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    samples, labels_df, events_df = discover_samples(data_dir)
    split_samples = [s for s in samples if (s.frame_dir is not None and s.n_images > 0)] if drop_missing_images else samples
    split_by_group = assign_splits(split_samples, seed=seed, val_frac=val_frac, test_frac=test_frac)
    rows: list[dict] = []
    for s in samples:
        has_images = int(s.frame_dir is not None and s.n_images > 0)
        if drop_missing_images and not has_images:
            continue
        rows.append(
            {
                "sample_id": s.sample_id,
                "group_id": s.group_id,
                "split": split_by_group.get(s.group_id, "test"),
                "frame_dir": _rel(s.frame_dir, data_dir),
                "xml_path": _rel(s.xml_path, data_dir),
                "n_images": s.n_images,
                "n_xml_frames": s.n_xml_frames,
                "n_positive_frames": s.n_positive_frames,
                "n_negative_frames": s.n_negative_frames,
                "has_images": has_images,
            }
        )
    manifest = pd.DataFrame(rows)
    if drop_missing_images and not labels_df.empty and not manifest.empty:
        keep = set(manifest["sample_id"].astype(str))
        labels_df = labels_df[labels_df["sample_id"].astype(str).isin(keep)].reset_index(drop=True)
        events_df = contiguous_positive_events(labels_df)
    return manifest, labels_df, events_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ChokePoint manifest and frame-level visitor labels.")
    parser.add_argument("--data-dir", required=True, help="Path to data/chokepoint")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-labels", required=True)
    parser.add_argument("--output-events", default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-frac", type=float, default=0.0)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--no-extract-archives", action="store_true", help="Do not extract .tar.xz archives before scanning")
    parser.add_argument("--overwrite-extract", action="store_true", help="Re-extract archives even if their contents already seem present")
    parser.add_argument("--skip-existing-archives", action="store_true", help="Old behavior: skip an archive if its target folder exists, even if extraction may be partial")
    parser.add_argument("--drop-missing-images", action="store_true", help="Omit XML sequences that do not have a matching frame directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not args.no_extract_archives:
        maybe_extract_archives(
            data_dir,
            overwrite=args.overwrite_extract,
            skip_existing_complete=not args.skip_existing_archives,
        )

    manifest, labels, events = build_manifest(
        data_dir,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        drop_missing_images=args.drop_missing_images,
    )
    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_manifest, index=False)
    Path(args.output_labels).parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output_labels, index=False)
    if args.output_events:
        Path(args.output_events).parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(args.output_events, index=False)

    print(f"Wrote manifest: {args.output_manifest} ({len(manifest)} sequences)")
    print(f"Wrote labels:   {args.output_labels} ({len(labels)} labeled frames)")
    if args.output_events:
        print(f"Wrote events:   {args.output_events} ({len(events)} positive intervals)")
    if len(manifest):
        print(manifest.groupby("split")["sample_id"].count().to_string())
        if "has_images" in manifest.columns:
            print("image-backed sequences by split:")
            print(manifest.groupby("split")["has_images"].sum().astype(int).to_string())
        missing = manifest[manifest["has_images"].astype(int) == 0]
        if len(missing):
            examples = missing["sample_id"].astype(str).head(10).tolist()
            print(
                f"WARNING: {len(missing)} XML files had no matching image directory. "
                f"Examples: {examples}. Run without --no-extract-archives or use --overwrite-extract if extraction was partial."
            )


if __name__ == "__main__":
    main()
