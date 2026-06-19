#!/usr/bin/env python3
"""Evaluate utility of generated privacy-preprocessing pipelines.

This script is designed to run from the project root after
``evaluation.generate_pipelines_for_all_contexts`` has produced a directory such
as ``runs/context_pipeline_generation``.  It discovers selected pipeline specs
for baselines and full-mediator ablations, optionally materializes transformed
manifests/data for each task, invokes the task-specific downstream inference
scripts, and writes a flat utility summary.

The evaluator deliberately uses the existing downstream inference CLIs rather
than importing task datasets directly.  This keeps the privacy-pipeline utility
check aligned with the same programs used for the raw downstream tasks.

Typical smoke test, visitor/chokepoint only:

    python -m evaluation.evaluate_utility \
      --pipeline-root runs/context_pipeline_generation \
      --out-dir runs/utility_eval \
      --tasks visitor_presence_detection \
      --scenario-ids S001 \
      --methods full_mediator,manual,raw \
      --chokepoint-manifest data/chokepoint/manifest.csv \
      --chokepoint-data-root data/chokepoint \
      --chokepoint-infer-module baselines.task.chokepoint.infer_chokepoint \
      --device cpu \
      --max-samples 2

Fall detection, allowing raw/video pipelines to run the downstream app's pose
extractor and pose-output pipelines to bypass it:

    python -m evaluation.evaluate_utility \
      --pipeline-root runs/context_pipeline_generation \
      --out-dir runs/utility_eval \
      --tasks fall_detection \
      --fall-manifest data/le2i/le2i_manifest.csv \
      --fall-data-root data/le2i \
      --fall-extract-pose-module baselines.task.le2i_fall.extract_pose \
      --fall-infer-module baselines.task.le2i_fall.infer \
      --fall-checkpoint outputs/le2i/checkpoint.pt \
      --fall-label-map outputs/le2i/label_map.json

Notes:
  * The generated pipeline spec is symbolic/runtime metadata.  A pipeline can be
    utility-evaluated only if its operators are implemented by smartpriv_runtime.
  * ChokePoint inference currently emits prediction CSVs; provide
    --chokepoint-eval-cmd-template if you have a separate evaluator that turns
    predictions into metrics.
  * For LE2I fall detection, pose output is treated as a valid downstream
    interface.  If a pipeline outputs image/video, this evaluator runs the
    downstream pose extractor first, then the fall classifier.
  * By default, transformed media/keypoint artifacts are temporary.  They are
    deleted after each method is evaluated.  Use --keep-intermediate-data only
    when you intentionally want to inspect blurred images, filtered audio, or
    generated pose files.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".wmv"}

TASK_TO_SHORT = {
    "visitor_presence_detection": "chokepoint",
    "fall_detection": "fall",
    "adl_recognition": "youhome",
    "domestic_sound_monitoring": "home_audio",
}

DEFAULT_METHODS = ["raw", "manual", "direct_llm", "full_mediator"]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


def write_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def parse_csv_list(value: Optional[str], default: Optional[Sequence[str]] = None) -> List[str]:
    if value is None:
        return list(default or [])
    text = str(value).strip()
    if not text:
        return list(default or [])
    return [x.strip() for x in text.split(",") if x.strip()]


def now_ms() -> int:
    return int(time.time() * 1000)


def cap_type(cap: Optional[Dict[str, Any]]) -> str:
    if not cap:
        return ""
    return str(cap.get("semantic_type") or cap.get("media_type") or "")


def cap_schema(cap: Optional[Dict[str, Any]]) -> str:
    if not cap:
        return ""
    return str(cap.get("schema") or "")


def is_pose_cap(cap: Optional[Dict[str, Any]]) -> bool:
    t = cap_type(cap)
    s = cap_schema(cap)
    return t == "application/x-pose-keypoints" or "pose" in s


def is_image_cap(cap: Optional[Dict[str, Any]]) -> bool:
    return cap_type(cap).startswith("image/") or "image" in cap_schema(cap) or "frame" in cap_schema(cap)


def is_video_cap(cap: Optional[Dict[str, Any]]) -> bool:
    return cap_type(cap).startswith("video/") or "video" in cap_schema(cap)


def is_audio_cap(cap: Optional[Dict[str, Any]]) -> bool:
    return cap_type(cap).startswith("audio/") or "audio" in cap_schema(cap)


def has_pipeline_stage(spec: Optional[Dict[str, Any]], operator_id: str) -> bool:
    if not spec:
        return False
    return any(str(s.get("operator_id") or s.get("operator")) == operator_id for s in spec.get("stages", []) or [])


def split_spec_before_stage(spec: Dict[str, Any], operator_id: str) -> Dict[str, Any]:
    """Return a copy of spec with stages before the first operator_id only."""
    out = dict(spec)
    stages = []
    for s in spec.get("stages", []) or []:
        if str(s.get("operator_id") or s.get("operator")) == operator_id:
            break
        stages.append(s)
    out["stages"] = stages
    out["pipeline_id"] = str(spec.get("pipeline_id", "pipe")) + f"_before_{operator_id.replace('.', '_')}"
    return out


def rel_or_abs(path: str | Path, root: Optional[str | Path]) -> Path:
    p = Path(str(path))
    if p.is_absolute():
        return p
    if p.exists():
        return p
    if root:
        return Path(root) / p
    return p


def sorted_images(path: Path) -> List[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def read_csv_rows(path: str | Path, max_rows: Optional[int] = None) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(dict(row))
        return rows, list(reader.fieldnames or [])


def write_csv_rows(rows: Sequence[Dict[str, Any]], path: str | Path, fieldnames: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in fields:
                    fields.append(k)
    else:
        fields = list(fieldnames)
        for r in rows:
            for k in r.keys():
                if k not in fields:
                    fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run_cmd(cmd: Sequence[str], cwd: Optional[str | Path], log_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = now_ms()
    cmd_list = [str(x) for x in cmd]
    if dry_run:
        write_text("DRY RUN\n" + shlex.join(cmd_list) + "\n", log_path)
        return {"cmd": cmd_list, "returncode": 0, "elapsed_ms": 0, "dry_run": True}

    proc = subprocess.run(
        cmd_list,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = now_ms() - started
    write_text("$ " + shlex.join(cmd_list) + "\n\n" + (proc.stdout or ""), log_path)
    return {"cmd": cmd_list, "returncode": proc.returncode, "elapsed_ms": elapsed, "dry_run": False}


def command_base(module: Optional[str], script: Optional[str | Path], python_exe: str) -> List[str]:
    if module:
        return [python_exe, "-m", module]
    if script:
        return [python_exe, str(script)]
    raise ValueError("Need either an inference module or script path.")


def append_if(cmd: List[str], flag: str, value: Optional[Any]) -> None:
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    cmd.extend([flag, str(value)])


def append_bool(cmd: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


# ---------------------------------------------------------------------------
# Pipeline discovery
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MethodRow:
    scenario_id: str
    task: str
    method_id: str
    method_kind: str
    baseline: str
    ablation_mode: Optional[str]
    method_output_dir: Path
    result_json: Optional[Path]
    selected_pipeline_json: Optional[Path]
    pipeline_spec_json: Optional[Path]
    final_output_type: str
    final_output_schema: str
    decision: str
    raw: Dict[str, Any]

    @property
    def label(self) -> str:
        return self.method_id or self.baseline


def _path_or_none(value: Any) -> Optional[Path]:
    if not value:
        return None
    p = Path(str(value))
    return p


def discover_method_rows(pipeline_root: Path) -> List[MethodRow]:
    summary_path = pipeline_root / "summary.json"
    if summary_path.exists():
        rows = load_json(summary_path)
    else:
        # Conservative fallback: scan selected_pipeline.json locations.
        rows = []
        for p in pipeline_root.glob("S*/baselines/*/result.json"):
            method_dir = p.parent
            scenario_id = p.parents[2].name
            method_id = method_dir.name
            result = load_json(p)
            selected = result.get("selected_candidate") or {}
            final_cap = selected.get("final_output_cap") or {}
            rows.append({
                "scenario_id": scenario_id,
                "task": "",
                "method_id": method_id,
                "method_kind": "baseline",
                "baseline": method_id,
                "method_output_dir": str(method_dir),
                "result_json": str(p),
                "selected_pipeline_json": str(method_dir / "selected_pipeline.json") if (method_dir / "selected_pipeline.json").exists() else None,
                "pipeline_spec_json": str(method_dir / "pipeline_spec.json") if (method_dir / "pipeline_spec.json").exists() else None,
                "final_output_type": cap_type(final_cap),
                "final_output_schema": cap_schema(final_cap),
                "decision": result.get("decision", {}).get("decision") if isinstance(result.get("decision"), dict) else result.get("decision"),
            })
        for p in pipeline_root.glob("S*/ablations/*/result.json"):
            method_dir = p.parent
            scenario_id = p.parents[2].name
            ablation_mode = method_dir.name
            result = load_json(p)
            selected = result.get("selected_candidate") or {}
            final_cap = selected.get("final_output_cap") or {}
            rows.append({
                "scenario_id": scenario_id,
                "task": "",
                "method_id": f"ablation:{ablation_mode}",
                "method_kind": "ablation",
                "baseline": f"ablation:{ablation_mode}",
                "ablation_mode": ablation_mode,
                "method_output_dir": str(method_dir),
                "result_json": str(p),
                "selected_pipeline_json": str(method_dir / "selected_pipeline.json") if (method_dir / "selected_pipeline.json").exists() else None,
                "pipeline_spec_json": str(method_dir / "pipeline_spec.json") if (method_dir / "pipeline_spec.json").exists() else None,
                "final_output_type": cap_type(final_cap),
                "final_output_schema": cap_schema(final_cap),
                "decision": result.get("decision", {}).get("decision") if isinstance(result.get("decision"), dict) else result.get("decision"),
            })

    out: List[MethodRow] = []
    for r in rows:
        method_output_dir = Path(str(r.get("method_output_dir") or r.get("baseline_output_dir") or ""))
        if method_output_dir and not method_output_dir.is_absolute():
            # summary rows are typically relative to project root.  Keep as-is if
            # it exists; otherwise interpret relative to pipeline_root parent.
            if not method_output_dir.exists():
                candidate = pipeline_root.parent.parent / method_output_dir
                if candidate.exists():
                    method_output_dir = candidate
        out.append(MethodRow(
            scenario_id=str(r.get("scenario_id") or ""),
            task=str(r.get("task") or ""),
            method_id=str(r.get("method_id") or r.get("baseline") or ""),
            method_kind=str(r.get("method_kind") or ("ablation" if str(r.get("baseline", "")).startswith("ablation:") else "baseline")),
            baseline=str(r.get("baseline") or r.get("baseline_id") or r.get("method_id") or ""),
            ablation_mode=str(r.get("ablation_mode")) if r.get("ablation_mode") not in (None, "") else None,
            method_output_dir=method_output_dir,
            result_json=_path_or_none(r.get("result_json")),
            selected_pipeline_json=_path_or_none(r.get("selected_pipeline_json")),
            pipeline_spec_json=_path_or_none(r.get("pipeline_spec_json")),
            final_output_type=str(r.get("final_output_type") or ""),
            final_output_schema=str(r.get("final_output_schema") or ""),
            decision=str(r.get("decision") or ""),
            raw=dict(r),
        ))
    return out


# ---------------------------------------------------------------------------
# Runtime preprocessing helpers
# ---------------------------------------------------------------------------


class RuntimeAdapter:
    """Small wrapper around smartpriv_runtime, imported lazily."""

    def __init__(self) -> None:
        self._loaded = False
        self.ExecutablePipeline = None
        self.item_from_media = None
        self.save_image = None
        self.save_audio = None

    def load(self) -> None:
        if self._loaded:
            return
        from smartpriv_runtime.pipeline import ExecutablePipeline  # type: ignore
        from smartpriv_runtime.media_io import item_from_media, save_image, save_audio  # type: ignore
        self.ExecutablePipeline = ExecutablePipeline
        self.item_from_media = item_from_media
        self.save_image = save_image
        self.save_audio = save_audio
        self._loaded = True

    def pipeline_from_spec(self, spec_path: Path):
        self.load()
        return self.ExecutablePipeline.from_spec_file(str(spec_path))  # type: ignore[union-attr]

    def item_from_path(self, path: Path, media_type: Optional[str] = None):
        self.load()
        return self.item_from_media(str(path), media_type=media_type)  # type: ignore[union-attr]

    def save_output_item(self, item: Any, output_base: Path, preferred_kind: str = "auto") -> Dict[str, Any]:
        """Save a DataItem-like object and return path metadata.

        The runtime may return images/audio/semantic JSON.  This function does
        not assume a specific DataItem class; it inspects caps/data duck-typed.
        """
        self.load()
        caps = getattr(item, "caps", {}) or {}
        data = getattr(item, "data", None)
        t = cap_type(caps)
        output_base.parent.mkdir(parents=True, exist_ok=True)

        # Import numpy only if needed.  The operator runtime already depends on it.
        try:
            import numpy as np  # type: ignore
        except Exception:  # pragma: no cover
            np = None  # type: ignore

        if np is not None and isinstance(data, np.ndarray) and (t.startswith("image/") or preferred_kind == "image"):
            out_path = output_base.with_suffix(".jpg")
            self.save_image(data, out_path)  # type: ignore[misc]
            return {"kind": "image", "path": str(out_path), "caps": caps}

        if np is not None and isinstance(data, np.ndarray) and (t.startswith("audio/") or preferred_kind == "audio"):
            out_path = output_base.with_suffix(".wav")
            sr = int((getattr(item, "metadata", {}) or {}).get("sample_rate", 16000))
            self.save_audio(out_path, data, sr)  # type: ignore[misc]
            return {"kind": "audio", "path": str(out_path), "caps": caps}

        if isinstance(data, dict) and "keypoints" in data:
            out_path = output_base.with_suffix(".npz")
            if np is not None:
                np.savez_compressed(out_path, keypoints=np.asarray(data.get("keypoints"), dtype=np.float32))
            else:
                write_json(data, output_base.with_suffix(".json"))
                out_path = output_base.with_suffix(".json")
            return {"kind": "pose", "path": str(out_path), "caps": caps}

        # Frame-list output.
        if np is not None and isinstance(data, list) and all(isinstance(x, np.ndarray) for x in data):
            frame_dir = output_base.parent / output_base.stem
            frame_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(data):
                self.save_image(frame, frame_dir / f"{i:06d}.jpg")  # type: ignore[misc]
            return {"kind": "frame_dir", "path": str(frame_dir), "caps": caps}

        # Generic JSON fallback.
        out_path = output_base.with_suffix(".json")
        if hasattr(item, "to_jsonable"):
            obj = item.to_jsonable(include_payload=False)
        else:
            obj = {"caps": caps, "data_repr": repr(data)[:1000]}
        write_json(obj, out_path)
        return {"kind": "json", "path": str(out_path), "caps": caps}


def pipeline_has_no_transform(spec: Optional[Dict[str, Any]]) -> bool:
    if not spec:
        return True
    return len(spec.get("stages", []) or []) == 0


def load_spec(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path and path.exists():
        return load_json(path)
    return None


def materialize_spec(spec: Dict[str, Any], path: Path) -> Path:
    write_json(spec, path)
    return path


def resolve_manifest_media_path(row: Dict[str, str], data_root: Optional[str | Path]) -> Optional[Path]:
    root = Path(data_root) if data_root else None
    for col in ["frame_dir", "video_path", "image_path", "audio_path", "wav_path", "path"]:
        val = row.get(col)
        if not val:
            continue
        candidates = [Path(val)]
        if root:
            candidates.append(root / val)
        for c in candidates:
            if c.exists():
                return c
    return None


def media_type_for_path(path: Path, task: str) -> str:
    if path.is_dir():
        # Most frame directories in these tasks are image/video frames.
        return "image/x-raw" if task == "visitor_presence_detection" else "video/x-raw"
    if path.suffix.lower() in IMAGE_EXTS:
        return "image/x-raw"
    if path.suffix.lower() in VIDEO_EXTS:
        return "video/x-raw"
    if path.suffix.lower() in AUDIO_EXTS:
        return "audio/x-raw"
    return "application/octet-stream"


def transform_image_dir(runtime: RuntimeAdapter, pipe: Any, in_dir: Path, out_dir: Path, max_frames: Optional[int], task: str) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted_images(in_dir)
    if max_frames is not None:
        frames = frames[:max_frames]
    saved = 0
    errors: List[str] = []
    for img in frames:
        try:
            item = runtime.item_from_path(img, media_type="image/x-raw")
            out = pipe.process(item)
            if out is None:
                continue
            meta = runtime.save_output_item(out, out_dir / img.stem, preferred_kind="image")
            saved += 1
        except Exception as exc:
            errors.append(f"{img}: {exc!r}")
    return {"kind": "frame_dir", "path": str(out_dir), "num_inputs": len(frames), "num_outputs": saved, "errors": errors[:20]}


def transform_single_media(runtime: RuntimeAdapter, pipe: Any, in_path: Path, out_base: Path, task: str) -> Dict[str, Any]:
    item = runtime.item_from_path(in_path, media_type=media_type_for_path(in_path, task))
    out = pipe.process(item)
    if out is None:
        return {"kind": "none", "path": None, "num_outputs": 0}
    preferred = "audio" if in_path.suffix.lower() in AUDIO_EXTS else "auto"
    return runtime.save_output_item(out, out_base, preferred_kind=preferred)


def transform_manifest_with_pipeline(
    manifest_path: Path,
    data_root: Optional[str | Path],
    spec_path: Path,
    out_manifest_path: Path,
    out_data_dir: Path,
    task: str,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    max_frames_per_sample: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    rows, fieldnames = read_csv_rows(manifest_path, max_rows=None)
    if split:
        rows = [r for r in rows if str(r.get("split", "")) == str(split)]
    if max_samples is not None:
        rows = rows[:max_samples]

    if dry_run:
        write_csv_rows(rows, out_manifest_path, fieldnames)
        return {"status": "dry_run", "manifest": str(out_manifest_path), "num_rows": len(rows), "num_transformed": 0}

    runtime = RuntimeAdapter()
    pipe = runtime.pipeline_from_spec(spec_path)
    new_rows: List[Dict[str, Any]] = []
    transform_records: List[Dict[str, Any]] = []
    iterator = tqdm(rows, desc=f"preprocess:{task}", unit="sample") if tqdm else rows

    for i, row in enumerate(iterator):
        sample_id = str(row.get("sample_id") or row.get("chunk_id") or i)
        in_path = resolve_manifest_media_path(row, data_root)
        new_row: Dict[str, Any] = dict(row)
        if in_path is None:
            new_row["preprocess_error"] = "missing_input"
            new_rows.append(new_row)
            continue

        sample_base = out_data_dir / sample_id
        try:
            if in_path.is_dir():
                out_dir = sample_base / "frames"
                rec = transform_image_dir(runtime, pipe, in_path, out_dir, max_frames_per_sample, task)
                new_row["frame_dir"] = str(out_dir)
                # Avoid stale source columns that can be preferred by downstream scripts.
                if "video_path" in new_row:
                    new_row["video_path"] = ""
                if "image_path" in new_row:
                    new_row["image_path"] = ""
            else:
                rec = transform_single_media(runtime, pipe, in_path, sample_base / "output", task)
                kind = rec.get("kind")
                out_path = str(rec.get("path") or "")
                if kind == "image":
                    new_row["image_path"] = out_path
                    if "frame_dir" in new_row:
                        new_row["frame_dir"] = ""
                    if "video_path" in new_row:
                        new_row["video_path"] = ""
                elif kind == "audio":
                    new_row["audio_path"] = out_path
                    if "wav_path" in new_row:
                        new_row["wav_path"] = out_path
                elif kind == "pose":
                    new_row["keypoints_path"] = out_path
                elif kind == "frame_dir":
                    new_row["frame_dir"] = out_path
                    if "video_path" in new_row:
                        new_row["video_path"] = ""
                elif kind == "json":
                    new_row["preprocessed_json_path"] = out_path
                elif kind == "none":
                    new_row["preprocess_error"] = "pipeline_returned_none"
            rec["sample_id"] = sample_id
            rec["input_path"] = str(in_path)
            transform_records.append(rec)
        except Exception as exc:
            new_row["preprocess_error"] = repr(exc)
            transform_records.append({
                "sample_id": sample_id,
                "input_path": str(in_path),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
        new_rows.append(new_row)

    write_csv_rows(new_rows, out_manifest_path, fieldnames)
    write_json(transform_records, out_manifest_path.with_suffix(".transforms.json"))
    return {
        "status": "ok",
        "manifest": str(out_manifest_path),
        "num_rows": len(new_rows),
        "num_transformed": sum(1 for r in transform_records if r.get("path")),
        "transform_records_json": str(out_manifest_path.with_suffix(".transforms.json")),
    }


# ---------------------------------------------------------------------------
# Task-specific runners
# ---------------------------------------------------------------------------


def infer_visitor(args: argparse.Namespace, manifest: Path, data_root: Optional[str | Path], out_dir: Path, dry_run: bool) -> Dict[str, Any]:
    if not args.chokepoint_infer_module and not args.chokepoint_infer_script:
        raise ValueError("visitor_presence_detection requires --chokepoint-infer-module or --chokepoint-infer-script")
    output_csv = out_dir / "predictions.csv"
    cmd = command_base(args.chokepoint_infer_module, args.chokepoint_infer_script, args.python)
    cmd.extend(["--manifest", str(manifest), "--output-csv", str(output_csv)])
    append_if(cmd, "--data-root", data_root)
    append_if(cmd, "--split", args.split)
    append_if(cmd, "--model", args.chokepoint_model)
    append_if(cmd, "--device", args.device)
    append_if(cmd, "--stride", args.chokepoint_stride)
    append_if(cmd, "--max-frames", args.max_frames_per_sample)
    append_bool(cmd, "--safe-cuda", args.safe_cuda)
    append_bool(cmd, "--retry-cpu-on-error", args.retry_cpu_on_error)
    append_bool(cmd, "--allow-empty-output", True)
    run = run_cmd(cmd, cwd=args.project_root, log_path=out_dir / "infer.log", dry_run=dry_run)

    metrics: Dict[str, Any] = {}
    if args.chokepoint_eval_cmd_template and not dry_run:
        metrics_json = out_dir / "metrics.json"
        template = args.chokepoint_eval_cmd_template.format(
            predictions=str(output_csv),
            manifest=str(manifest),
            data_root=str(data_root or ""),
            out_json=str(metrics_json),
        )
        eval_cmd = shlex.split(template)
        eval_run = run_cmd(eval_cmd, cwd=args.project_root, log_path=out_dir / "evaluate.log", dry_run=False)
        metrics["eval_command"] = eval_run
        if metrics_json.exists():
            metrics.update(load_json(metrics_json))
    return {"status": "ok" if run["returncode"] == 0 else "error", "inference": run, "output_csv": str(output_csv), "metrics": metrics}


def run_fall_extract_pose(args: argparse.Namespace, manifest: Path, data_root: Optional[str | Path], out_dir: Path, dry_run: bool) -> Tuple[Path, Dict[str, Any]]:
    if not args.fall_extract_pose_module and not args.fall_extract_pose_script:
        raise ValueError("fall image/video path requires --fall-extract-pose-module or --fall-extract-pose-script")
    pose_dir = out_dir / "pose"
    pose_manifest = out_dir / "manifest_with_keypoints.csv"
    cmd = command_base(args.fall_extract_pose_module, args.fall_extract_pose_script, args.python)
    cmd.extend(["--manifest", str(manifest), "--output-dir", str(pose_dir), "--updated-manifest", str(pose_manifest)])
    append_if(cmd, "--data-root", data_root)
    append_if(cmd, "--model", args.fall_pose_model)
    append_if(cmd, "--device", args.device)
    append_if(cmd, "--stride", args.fall_pose_stride)
    append_if(cmd, "--max-frames", args.max_frames_per_sample)
    append_bool(cmd, "--no-sanitize-videos", args.no_sanitize_videos)
    run = run_cmd(cmd, cwd=args.project_root, log_path=out_dir / "extract_pose.log", dry_run=dry_run)
    return pose_manifest, {"extract_pose": run, "pose_manifest": str(pose_manifest), "pose_dir": str(pose_dir)}


def infer_fall(args: argparse.Namespace, manifest: Path, data_root: Optional[str | Path], out_dir: Path, dry_run: bool) -> Dict[str, Any]:
    if not args.fall_infer_module and not args.fall_infer_script:
        raise ValueError("fall_detection requires --fall-infer-module or --fall-infer-script")
    if not args.fall_checkpoint or not args.fall_label_map:
        raise ValueError("fall_detection requires --fall-checkpoint and --fall-label-map")
    output_csv = out_dir / "predictions.csv"
    video_output_csv = out_dir / "predictions_video.csv"
    metrics_json = out_dir / "metrics.json"
    cmd = command_base(args.fall_infer_module, args.fall_infer_script, args.python)
    cmd.extend([
        "--manifest", str(manifest),
        "--checkpoint", str(args.fall_checkpoint),
        "--label-map", str(args.fall_label_map),
        "--output-csv", str(output_csv),
        "--video-output-csv", str(video_output_csv),
        "--metrics-json", str(metrics_json),
    ])
    append_if(cmd, "--data-root", data_root)
    append_if(cmd, "--split", args.split or "test")
    append_if(cmd, "--sample-mode", args.fall_sample_mode)
    append_if(cmd, "--device", args.device)
    append_if(cmd, "--batch-size", args.batch_size)
    append_if(cmd, "--num-workers", args.num_workers)
    run = run_cmd(cmd, cwd=args.project_root, log_path=out_dir / "infer.log", dry_run=dry_run)
    metrics = load_json(metrics_json) if metrics_json.exists() else {}
    return {"status": "ok" if run["returncode"] == 0 else "error", "inference": run, "output_csv": str(output_csv), "video_output_csv": str(video_output_csv), "metrics_json": str(metrics_json), "metrics": metrics}


def infer_home_audio(args: argparse.Namespace, manifest: Path, out_dir: Path, dry_run: bool) -> Dict[str, Any]:
    if not args.home_audio_infer_module and not args.home_audio_infer_script:
        raise ValueError("domestic_sound_monitoring requires --home-audio-infer-module or --home-audio-infer-script")
    if not args.home_audio_checkpoint:
        raise ValueError("domestic_sound_monitoring requires --home-audio-checkpoint")
    output_csv = out_dir / "predictions.csv"
    metrics_json = out_dir / "metrics.json"
    cmd = command_base(args.home_audio_infer_module, args.home_audio_infer_script, args.python)
    cmd.extend([
        "--manifest", str(manifest),
        "--checkpoint", str(args.home_audio_checkpoint),
        "--output-csv", str(output_csv),
        "--metrics-json", str(metrics_json),
    ])
    append_if(cmd, "--split", args.split or "test")
    append_if(cmd, "--device", args.device)
    append_if(cmd, "--batch-size", args.batch_size)
    append_if(cmd, "--num-workers", args.num_workers)
    append_if(cmd, "--threshold", args.home_audio_threshold)
    append_if(cmd, "--backbone", args.home_audio_backbone)
    run = run_cmd(cmd, cwd=args.project_root, log_path=out_dir / "infer.log", dry_run=dry_run)
    metrics = load_json(metrics_json) if metrics_json.exists() else {}
    return {"status": "ok" if run["returncode"] == 0 else "error", "inference": run, "output_csv": str(output_csv), "metrics_json": str(metrics_json), "metrics": metrics}


def infer_youhome(args: argparse.Namespace, manifest: Path, data_root: Optional[str | Path], out_dir: Path, dry_run: bool) -> Dict[str, Any]:
    if not args.youhome_infer_module and not args.youhome_infer_script:
        raise ValueError("adl_recognition requires --youhome-infer-module or --youhome-infer-script")
    if not args.youhome_checkpoint or not args.youhome_label_map:
        raise ValueError("adl_recognition requires --youhome-checkpoint and --youhome-label-map")
    output_csv = out_dir / "predictions.csv"
    metrics_json = out_dir / "metrics.json"
    cmd = command_base(args.youhome_infer_module, args.youhome_infer_script, args.python)
    cmd.extend([
        "--manifest", str(manifest),
        "--checkpoint", str(args.youhome_checkpoint),
        "--label-map", str(args.youhome_label_map),
        "--output-csv", str(output_csv),
        "--metrics-json", str(metrics_json),
    ])
    append_if(cmd, "--data-root", data_root)
    append_if(cmd, "--split", args.split or "test")
    append_if(cmd, "--device", args.device)
    append_if(cmd, "--batch-size", args.batch_size)
    append_if(cmd, "--num-workers", args.num_workers)
    append_if(cmd, "--tta-runs", args.youhome_tta_runs)
    run = run_cmd(cmd, cwd=args.project_root, log_path=out_dir / "infer.log", dry_run=dry_run)
    metrics = load_json(metrics_json) if metrics_json.exists() else {}
    return {"status": "ok" if run["returncode"] == 0 else "error", "inference": run, "output_csv": str(output_csv), "metrics_json": str(metrics_json), "metrics": metrics}


# ---------------------------------------------------------------------------
# Per-method evaluation orchestration
# ---------------------------------------------------------------------------


def task_manifest_and_root(args: argparse.Namespace, task: str) -> Tuple[Path, Optional[str | Path]]:
    if task == "visitor_presence_detection":
        if not args.chokepoint_manifest:
            raise ValueError("visitor_presence_detection requires --chokepoint-manifest")
        return Path(args.chokepoint_manifest), args.chokepoint_data_root
    if task == "fall_detection":
        if not args.fall_manifest:
            raise ValueError("fall_detection requires --fall-manifest")
        return Path(args.fall_manifest), args.fall_data_root
    if task == "adl_recognition":
        if not args.youhome_manifest:
            raise ValueError("adl_recognition requires --youhome-manifest")
        return Path(args.youhome_manifest), args.youhome_data_root
    if task == "domestic_sound_monitoring":
        if not args.home_audio_manifest:
            raise ValueError("domestic_sound_monitoring requires --home-audio-manifest")
        return Path(args.home_audio_manifest), None
    raise ValueError(f"Unknown task {task!r}")


def prepare_manifest_for_method(args: argparse.Namespace, row: MethodRow, work_dir: Path) -> Tuple[Path, Optional[str | Path], Dict[str, Any]]:
    """Return a manifest/data-root to feed into downstream inference."""
    manifest, data_root = task_manifest_and_root(args, row.task)
    spec = load_spec(row.pipeline_spec_json)
    final_cap = (spec or {}).get("final_output_cap") or {
        "semantic_type": row.final_output_type if row.final_output_type.startswith("application/") else None,
        "media_type": row.final_output_type if not row.final_output_type.startswith("application/") else None,
        "schema": row.final_output_schema,
    }

    prep: Dict[str, Any] = {
        "source_manifest": str(manifest),
        "source_data_root": str(data_root) if data_root else None,
        "pipeline_spec": str(row.pipeline_spec_json) if row.pipeline_spec_json else None,
        "final_output_cap": final_cap,
        "preprocessing_status": "not_started",
    }

    # Raw/no-transform: use original manifest.  This includes raw baseline and
    # any selected candidate with no executable preprocessing stages.
    if not spec or pipeline_has_no_transform(spec) or row.baseline == "raw":
        prep["preprocessing_status"] = "passthrough_raw_or_no_transform"
        return manifest, data_root, prep

    # Fall special case: downstream app accepts either image/video or pose.
    # If selected pipeline emits pose, we need a manifest with keypoints_path.
    if row.task == "fall_detection" and is_pose_cap(final_cap):
        if has_pipeline_stage(spec, "op.pose_extractor"):
            # Use the same downstream YOLO pose extractor for utility comparability.
            # If the pipeline had pre-pose transforms, materialize those first.
            pre_pose_spec = split_spec_before_stage(spec, "op.pose_extractor")
            if pre_pose_spec.get("stages"):
                pre_manifest = work_dir / "pre_pose_media_manifest.csv"
                pre_dir = work_dir / "pre_pose_media"
                pre_spec_path = materialize_spec(pre_pose_spec, work_dir / "pre_pose_pipeline_spec.json")
                pre_result = transform_manifest_with_pipeline(
                    manifest,
                    data_root,
                    pre_spec_path,
                    pre_manifest,
                    pre_dir,
                    task=row.task,
                    split=args.split,
                    max_samples=args.max_samples,
                    max_frames_per_sample=args.max_frames_per_sample,
                    dry_run=args.dry_run,
                )
                prep["pre_pose_preprocessing"] = pre_result
                pose_manifest, pose_info = run_fall_extract_pose(args, pre_manifest, None, work_dir, args.dry_run)
            else:
                pose_manifest, pose_info = run_fall_extract_pose(args, manifest, data_root, work_dir, args.dry_run)
            prep.update(pose_info)
            prep["preprocessing_status"] = "pose_extracted_with_downstream_yolo_pose"
            return pose_manifest, None, prep

        # Pipeline claims pose but does not contain a recognized pose extractor.
        # Try runtime pipeline directly; it may still produce keypoints_path rows.
        out_manifest = work_dir / "preprocessed_manifest.csv"
        transformed = transform_manifest_with_pipeline(
            manifest,
            data_root,
            Path(row.pipeline_spec_json),
            out_manifest,
            work_dir / "preprocessed_data",
            task=row.task,
            split=args.split,
            max_samples=args.max_samples,
            max_frames_per_sample=args.max_frames_per_sample,
            dry_run=args.dry_run,
        )
        prep.update(transformed)
        prep["preprocessing_status"] = "runtime_pose_manifest"
        return out_manifest, None, prep

    # Fall image/video output: app runs its internal pose detector before fall classifier.
    if row.task == "fall_detection" and (is_image_cap(final_cap) or is_video_cap(final_cap)):
        media_manifest = work_dir / "preprocessed_media_manifest.csv"
        transformed = transform_manifest_with_pipeline(
            manifest,
            data_root,
            Path(row.pipeline_spec_json),
            media_manifest,
            work_dir / "preprocessed_media",
            task=row.task,
            split=args.split,
            max_samples=args.max_samples,
            max_frames_per_sample=args.max_frames_per_sample,
            dry_run=args.dry_run,
        )
        prep.update(transformed)
        pose_manifest, pose_info = run_fall_extract_pose(args, media_manifest, None, work_dir, args.dry_run)
        prep.update(pose_info)
        prep["preprocessing_status"] = "media_preprocessed_then_downstream_pose"
        return pose_manifest, None, prep

    # Audio task with audio/x-filtered or waveform-like output.
    if row.task == "domestic_sound_monitoring" and is_audio_cap(final_cap):
        out_manifest = work_dir / "preprocessed_manifest.csv"
        transformed = transform_manifest_with_pipeline(
            manifest,
            data_root,
            Path(row.pipeline_spec_json),
            out_manifest,
            work_dir / "preprocessed_audio",
            task=row.task,
            split=args.split,
            max_samples=args.max_samples,
            max_frames_per_sample=args.max_frames_per_sample,
            dry_run=args.dry_run,
        )
        prep.update(transformed)
        prep["preprocessing_status"] = "audio_preprocessed"
        return out_manifest, None, prep

    # Visitor / ADL media outputs: materialize transformed manifest and use that.
    if is_image_cap(final_cap) or is_video_cap(final_cap) or is_audio_cap(final_cap):
        out_manifest = work_dir / "preprocessed_manifest.csv"
        transformed = transform_manifest_with_pipeline(
            manifest,
            data_root,
            Path(row.pipeline_spec_json),
            out_manifest,
            work_dir / "preprocessed_data",
            task=row.task,
            split=args.split,
            max_samples=args.max_samples,
            max_frames_per_sample=args.max_frames_per_sample,
            dry_run=args.dry_run,
        )
        prep.update(transformed)
        prep["preprocessing_status"] = "media_preprocessed"
        return out_manifest, None, prep

    # Semantic outputs are not directly consumable by the attached downstream
    # models unless a task-specific adapter exists.
    prep["preprocessing_status"] = "incompatible_semantic_output_for_downstream_cli"
    raise ValueError(f"Final output {cap_type(final_cap)} / {cap_schema(final_cap)} is not supported by the downstream utility evaluator for task {row.task}.")


def run_downstream_for_method(args: argparse.Namespace, row: MethodRow, manifest: Path, data_root: Optional[str | Path], work_dir: Path) -> Dict[str, Any]:
    if row.task == "visitor_presence_detection":
        return infer_visitor(args, manifest, data_root, work_dir, args.dry_run)
    if row.task == "fall_detection":
        return infer_fall(args, manifest, data_root, work_dir, args.dry_run)
    if row.task == "domestic_sound_monitoring":
        return infer_home_audio(args, manifest, work_dir, args.dry_run)
    if row.task == "adl_recognition":
        return infer_youhome(args, manifest, data_root, work_dir, args.dry_run)
    raise ValueError(f"Unsupported task {row.task!r}")


def flatten_metrics(metrics: Dict[str, Any], prefix: str = "metric") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    def rec(obj: Any, path: List[str]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                rec(v, path + [str(k)])
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            key = prefix + "_" + "_".join(path)
            if len(str(obj)) < 200:
                flat[key] = obj
    rec(metrics, [])
    return flat


def _evaluate_one_with_intermediate_workspace(
    args: argparse.Namespace,
    row: MethodRow,
    work_dir: Path,
    intermediate_dir: Path,
    base: Dict[str, Any],
    started: int,
) -> Dict[str, Any]:
    """Evaluate one generated method using an intermediate-artifact workspace.

    ``work_dir`` is the retained output directory for predictions, metrics, logs,
    and ``utility_result.json``. ``intermediate_dir`` is where transformed media,
    generated manifests, pose .npz files, filtered audio, etc. are written.  By
    default ``intermediate_dir`` is a TemporaryDirectory and is deleted after the
    downstream inference command finishes.
    """
    try:
        manifest, data_root, prep = prepare_manifest_for_method(args, row, intermediate_dir)
        prep["intermediate_artifact_dir"] = str(intermediate_dir)
        prep["intermediate_artifacts_retained"] = bool(args.keep_intermediate_data)
        prep["intermediate_artifact_policy"] = (
            "retained_under_utility_work_dir"
            if args.keep_intermediate_data
            else "temporary_deleted_after_downstream_inference"
        )

        downstream = run_downstream_for_method(args, row, manifest, data_root, work_dir)
        status = "ok" if downstream.get("status") == "ok" else "error"
        result = {
            **base,
            "status": status,
            "elapsed_ms": now_ms() - started,
            "prepared_manifest": str(manifest),
            "prepared_manifest_retained": bool(args.keep_intermediate_data) or prep.get("preprocessing_status") == "passthrough_raw_or_no_transform",
            "prepared_data_root": str(data_root) if data_root else None,
            "intermediate_artifacts_retained": bool(args.keep_intermediate_data),
            "intermediate_artifact_policy": prep["intermediate_artifact_policy"],
            "preprocessing": prep,
            "downstream": downstream,
            **flatten_metrics(downstream.get("metrics", {})),
        }
    except Exception as exc:
        result = {
            **base,
            "status": "error",
            "elapsed_ms": now_ms() - started,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "intermediate_artifacts_retained": bool(args.keep_intermediate_data),
            "intermediate_artifact_policy": (
                "retained_under_utility_work_dir"
                if args.keep_intermediate_data
                else "temporary_deleted_after_downstream_inference"
            ),
        }
        write_text(result["traceback"], work_dir / "traceback.txt")
    return result


def evaluate_one(args: argparse.Namespace, row: MethodRow) -> Dict[str, Any]:
    method_slug = row.method_id.replace(":", "__").replace("/", "_")
    work_dir = Path(args.out_dir) / row.scenario_id / method_slug
    work_dir.mkdir(parents=True, exist_ok=True)
    started = now_ms()
    base: Dict[str, Any] = {
        "scenario_id": row.scenario_id,
        "task": row.task,
        "method_id": row.method_id,
        "method_kind": row.method_kind,
        "baseline": row.baseline,
        "ablation_mode": row.ablation_mode,
        "decision": row.decision,
        "pipeline_spec_json": str(row.pipeline_spec_json) if row.pipeline_spec_json else None,
        "selected_pipeline_json": str(row.selected_pipeline_json) if row.selected_pipeline_json else None,
        "result_json": str(row.result_json) if row.result_json else None,
        "final_output_type": row.final_output_type,
        "final_output_schema": row.final_output_schema,
        "utility_work_dir": str(work_dir),
    }

    if args.keep_intermediate_data:
        intermediate_dir = work_dir / "intermediate_artifacts"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        result = _evaluate_one_with_intermediate_workspace(args, row, work_dir, intermediate_dir, base, started)
        result["intermediate_artifact_dir"] = str(intermediate_dir)
        write_json(result, work_dir / "utility_result.json")
        return result

    temp_parent = Path(args.intermediate_root) if args.intermediate_root else None
    if temp_parent:
        temp_parent.mkdir(parents=True, exist_ok=True)
    prefix = f"utility_{row.scenario_id}_{method_slug}_"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=str(temp_parent) if temp_parent else None) as tmp:
        intermediate_dir = Path(tmp)
        result = _evaluate_one_with_intermediate_workspace(args, row, work_dir, intermediate_dir, base, started)
        # The directory will be removed immediately after this block.  Keep this
        # explicit in the result so stale paths are not mistaken for retained data.
        result["intermediate_artifact_dir"] = str(intermediate_dir)
        result["intermediate_artifact_dir_exists_after_run"] = False
        write_json(result, work_dir / "utility_result.json")
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate downstream utility of generated privacy preprocessing pipelines.")
    p.add_argument("--pipeline-root", default="runs/context_pipeline_generation")
    p.add_argument("--out-dir", default="runs/utility_eval")
    p.add_argument("--project-root", default=".")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--tasks", default="", help="Comma list of task ids; default = all tasks present.")
    p.add_argument("--scenario-ids", default="", help="Comma list such as S001,S002.")
    p.add_argument("--methods", default="", help="Comma list of method ids/baselines, e.g. raw,manual,full_mediator,ablation:utility_only.")
    p.add_argument("--include-ablations", action="store_true", help="Include rows with method_kind=ablation when --methods is not specified.")
    p.add_argument("--split", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-frames-per-sample", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--keep-intermediate-data", action="store_true", help="Retain transformed media/keypoint/audio artifacts. Default is to use a temporary directory and delete them after each downstream run.")
    p.add_argument("--intermediate-root", default=None, help="Optional parent directory for temporary intermediate artifacts. Useful if /tmp is small; artifacts are still deleted unless --keep-intermediate-data is set.")

    # General inference resource knobs.
    p.add_argument("--device", default=None, help="Device passed to downstream inference scripts, e.g. cpu, 0, cuda:0.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--safe-cuda", action="store_true")
    p.add_argument("--retry-cpu-on-error", action="store_true")

    # ChokePoint / visitor.
    p.add_argument("--chokepoint-manifest", default=None)
    p.add_argument("--chokepoint-data-root", default=None)
    p.add_argument("--chokepoint-infer-module", default=None)
    p.add_argument("--chokepoint-infer-script", default=None)
    p.add_argument("--chokepoint-model", default="yolo11n.pt")
    p.add_argument("--chokepoint-stride", type=int, default=None)
    p.add_argument("--chokepoint-eval-cmd-template", default=None, help="Optional command with {predictions}, {manifest}, {data_root}, {out_json} placeholders.")

    # LE2I fall.
    p.add_argument("--fall-manifest", default=None)
    p.add_argument("--fall-data-root", default=None)
    p.add_argument("--fall-extract-pose-module", default=None)
    p.add_argument("--fall-extract-pose-script", default=None)
    p.add_argument("--fall-infer-module", default=None)
    p.add_argument("--fall-infer-script", default=None)
    p.add_argument("--fall-checkpoint", default=None)
    p.add_argument("--fall-label-map", default=None)
    p.add_argument("--fall-pose-model", default="yolo11n-pose.pt")
    p.add_argument("--fall-pose-stride", type=int, default=3)
    p.add_argument("--fall-sample-mode", choices=["window", "video"], default=None)
    p.add_argument("--no-sanitize-videos", action="store_true")

    # CHiME-Home audio.
    p.add_argument("--home-audio-manifest", default=None)
    p.add_argument("--home-audio-infer-module", default=None)
    p.add_argument("--home-audio-infer-script", default=None)
    p.add_argument("--home-audio-checkpoint", default=None)
    p.add_argument("--home-audio-threshold", type=float, default=None)
    p.add_argument("--home-audio-backbone", choices=["logmel_cnn", "ast"], default=None)

    # YouHome ADL.
    p.add_argument("--youhome-manifest", default=None)
    p.add_argument("--youhome-data-root", default=None)
    p.add_argument("--youhome-infer-module", default=None)
    p.add_argument("--youhome-infer-script", default=None)
    p.add_argument("--youhome-checkpoint", default=None)
    p.add_argument("--youhome-label-map", default=None)
    p.add_argument("--youhome-tta-runs", type=int, default=None)
    return p


def filter_rows(args: argparse.Namespace, rows: List[MethodRow]) -> List[MethodRow]:
    tasks = set(parse_csv_list(args.tasks))
    sids = set(parse_csv_list(args.scenario_ids))
    methods = set(parse_csv_list(args.methods))

    out: List[MethodRow] = []
    for r in rows:
        if r.decision and r.decision not in {"select_pipeline", "baseline_raw_release", "raw_release", "select"}:
            # Keep raw rows even if the exact decision string varies; skip clear failures.
            if r.decision in {"error", "no_candidates", "no_compromise", "invalid_or_no_pipeline", "llm_error"}:
                continue
        if tasks and r.task not in tasks:
            continue
        if sids and r.scenario_id not in sids:
            continue
        if methods:
            if r.method_id not in methods and r.baseline not in methods and (r.ablation_mode and f"ablation:{r.ablation_mode}" not in methods):
                continue
        else:
            if r.method_kind == "ablation" and not args.include_ablations:
                continue
        if r.task not in TASK_TO_SHORT:
            continue
        out.append(r)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    pipeline_root = Path(args.pipeline_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = discover_method_rows(pipeline_root)
    selected = filter_rows(args, rows)

    write_json({
        "pipeline_root": str(pipeline_root),
        "out_dir": str(out_dir),
        "total_discovered_rows": len(rows),
        "selected_rows": len(selected),
        "tasks": sorted({r.task for r in selected}),
        "methods": sorted({r.method_id for r in selected}),
        "dry_run": bool(args.dry_run),
        "keep_intermediate_data": bool(args.keep_intermediate_data),
        "intermediate_root": args.intermediate_root,
        "intermediate_artifact_policy": (
            "retained_under_each_method_dir/intermediate_artifacts"
            if args.keep_intermediate_data
            else "temporary_deleted_after_each_downstream_run"
        ),
    }, out_dir / "utility_eval_plan.json")

    results: List[Dict[str, Any]] = []
    iterator: Iterable[MethodRow]
    if tqdm and not args.no_progress:
        iterator = tqdm(selected, desc="utility eval", unit="method")
    else:
        iterator = selected

    for r in iterator:
        if tqdm and not args.no_progress:
            iterator.set_postfix_str(f"{r.scenario_id} {r.method_id}")  # type: ignore[attr-defined]
        result = evaluate_one(args, r)
        results.append(result)
        if args.fail_fast and result.get("status") == "error":
            break

    write_json(results, out_dir / "utility_results.json")
    flat_rows = []
    for res in results:
        row = {k: v for k, v in res.items() if not isinstance(v, (dict, list))}
        # Keep some useful nested file paths/statuses.
        prep = res.get("preprocessing") or {}
        down = res.get("downstream") or {}
        row["preprocessing_status"] = prep.get("preprocessing_status") or prep.get("status")
        row["source_manifest"] = prep.get("source_manifest")
        row["prepared_manifest"] = res.get("prepared_manifest")
        row["prepared_manifest_retained"] = res.get("prepared_manifest_retained")
        row["intermediate_artifacts_retained"] = res.get("intermediate_artifacts_retained")
        row["intermediate_artifact_policy"] = res.get("intermediate_artifact_policy")
        row["downstream_status"] = down.get("status")
        row["downstream_output_csv"] = down.get("output_csv")
        row["downstream_metrics_json"] = down.get("metrics_json")
        if isinstance(down.get("inference"), dict):
            row["downstream_returncode"] = down["inference"].get("returncode")
            row["downstream_elapsed_ms"] = down["inference"].get("elapsed_ms")
        # Include flattened metrics already in top-level res.
        for k, v in res.items():
            if k.startswith("metric_"):
                row[k] = v
        flat_rows.append(row)
    write_csv_rows(flat_rows, out_dir / "utility_summary.csv")

    print(json.dumps({
        "discovered_rows": len(rows),
        "evaluated_rows": len(results),
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "errors": sum(1 for r in results if r.get("status") == "error"),
        "out_dir": str(out_dir),
        "summary_csv": str(out_dir / "utility_summary.csv"),
        "results_json": str(out_dir / "utility_results.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
