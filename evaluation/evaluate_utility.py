#!/usr/bin/env python3
"""Evaluate utility of generated privacy-preprocessing pipelines.

This script is designed to run from the project root after
``evaluation.generate_pipelines_for_all_contexts`` has produced a directory such
as ``runs/context_pipeline_generation`` or ``runs/flexible_context_pipeline_generation``.  It discovers selected pipeline specs
for baselines and full-mediator ablations, optionally materializes transformed
manifests/data for each task, invokes the task-specific downstream inference
scripts, and writes a flat utility summary.

The evaluator deliberately uses the existing downstream inference CLIs rather
than importing task datasets directly.  This keeps the privacy-pipeline utility
check aligned with the same programs used for the raw downstream tasks.

Default: if --tasks is omitted, the evaluator runs every configured task present in the generated pipeline summary and evaluates only --split test.

Typical smoke test, visitor/chokepoint only:

    python -m evaluation.evaluate_utility \
      --pipeline-root runs/context_pipeline_generation \
      --out-dir runs/utility_eval \
      --tasks visitor_presence_detection \
      --scenario-ids S001 \
      --methods full_mediator,manual,raw \
      --chokepoint-manifest data/chokepoint/chokepoint_manifest.csv \
      --chokepoint-data-root data/chokepoint \
      --chokepoint-infer-module baselines.task.chokepoint_presence.infer_yolo \
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
import copy
from collections import Counter
import csv
import dataclasses
import hashlib
import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

EVALUATE_UTILITY_PATCH_VERSION = "2026-06-flexible-cross-run-reuse-v5"

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# Global progress switch.  It is set from --no-progress in main(), and used by
# helper functions that are called deep inside preprocessing/downstream stages.
_PROGRESS_ENABLED = True


def set_progress_enabled(enabled: bool) -> None:
    global _PROGRESS_ENABLED
    _PROGRESS_ENABLED = bool(enabled)


def progress_enabled() -> bool:
    return bool(tqdm is not None and _PROGRESS_ENABLED)


def progress_write(msg: str) -> None:
    if progress_enabled():
        try:
            tqdm.write(str(msg))  # type: ignore[union-attr]
        except Exception:
            print(str(msg), file=sys.stderr, flush=True)
    else:
        # Keep important debug breadcrumbs visible even when tqdm is disabled.
        print(str(msg), file=sys.stderr, flush=True)


def json_safe(obj: Any) -> Any:
    """Best-effort conversion for debug records written before native calls."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(x) for x in obj]
    return repr(obj)


def append_jsonl(record: Dict[str, Any], path: str | Path) -> None:
    """Append one fsynced JSONL record so native crashes still leave breadcrumbs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(json_safe(record), sort_keys=False, default=str)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def write_json_fsynced(data: Any, path: str | Path) -> None:
    """Write JSON and fsync it; used for last-attempt breadcrumbs before C/C++ calls."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, sort_keys=False, default=str)
        f.write("\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)


def preprocess_debug_enabled(args: Optional[argparse.Namespace] = None) -> bool:
    if args is None:
        return True
    return not bool(getattr(args, "no_preprocess_debug", False))


def emit_preprocess_debug(
    debug_dir: Optional[str | Path],
    record: Dict[str, Any],
    *,
    last: bool = True,
    echo: bool = True,
) -> None:
    """Write durable preprocessing debug breadcrumbs and optionally echo a compact line."""
    if debug_dir is None:
        return
    d = Path(debug_dir)
    rec = {"ts_ms": now_ms(), **record}
    try:
        append_jsonl(rec, d / "preprocess_debug.jsonl")
        if last:
            write_json_fsynced(rec, d / "last_preprocess_attempt.json")
    except Exception as exc:
        print(f"[preprocess-debug] failed to write debug record to {d}: {exc!r}", file=sys.stderr, flush=True)
    if echo:
        sample = rec.get("sample_id")
        path = rec.get("resolved_input_path") or rec.get("input_path")
        phase = rec.get("phase")
        task = rec.get("task")
        scenario = rec.get("scenario_id")
        method = rec.get("method_id")
        print(
            f"[preprocess-debug] {scenario or ''}/{method or ''} task={task or ''} "
            f"phase={phase or ''} sample={sample or ''} input={path or ''}",
            file=sys.stderr,
            flush=True,
        )


def compact_manifest_row(row: Dict[str, Any], max_value_len: int = 500) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        text = str(v)
        out[str(k)] = text if len(text) <= max_value_len else text[:max_value_len] + "…"
    return out


def debug_record_for_manifest_row(
    *,
    phase: str,
    task: str,
    row_index: int,
    row: Dict[str, Any],
    data_root: Optional[str | Path],
    manifest_path: Optional[str | Path] = None,
    filtered_manifest_path: Optional[str | Path] = None,
    spec_path: Optional[str | Path] = None,
    out_base: Optional[str | Path] = None,
    scenario_id: Optional[str] = None,
    method_id: Optional[str] = None,
    final_output_type: Optional[str] = None,
    final_output_schema: Optional[str] = None,
) -> Dict[str, Any]:
    sample_id = str(row.get("sample_id") or row.get("chunk_id") or row.get("video_id") or row.get("id") or row_index)
    in_path = resolve_manifest_media_path(row, data_root)
    rec: Dict[str, Any] = {
        "phase": phase,
        "scenario_id": scenario_id,
        "method_id": method_id,
        "task": task,
        "row_index": row_index,
        "sample_id": sample_id,
        "source_manifest": str(manifest_path) if manifest_path else None,
        "filtered_manifest": str(filtered_manifest_path) if filtered_manifest_path else None,
        "source_data_root": str(data_root) if data_root else None,
        "pipeline_spec": str(spec_path) if spec_path else None,
        "out_base": str(out_base) if out_base else None,
        "final_output_type": final_output_type,
        "final_output_schema": final_output_schema,
        "resolved_input_path": str(in_path) if in_path else None,
        "resolved_input_exists": bool(in_path.exists()) if in_path else False,
        "resolved_input_is_dir": bool(in_path.is_dir()) if in_path and in_path.exists() else False,
        "resolved_input_suffix": str(in_path.suffix).lower() if in_path and not in_path.is_dir() else None,
        "resolved_media_type": media_type_for_path(in_path, task) if in_path else None,
        "row": compact_manifest_row(row),
    }
    return rec


def write_manifest_input_debug(
    *,
    debug_dir: Optional[str | Path],
    rows: Sequence[Dict[str, Any]],
    data_root: Optional[str | Path],
    task: str,
    manifest_path: str | Path,
    filtered_manifest_path: str | Path,
    scenario_id: Optional[str],
    method_id: Optional[str],
    final_output_type: Optional[str],
    final_output_schema: Optional[str],
) -> None:
    if debug_dir is None:
        return
    d = Path(debug_dir)
    d.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": "source_manifest_inputs_summary",
        "scenario_id": scenario_id,
        "method_id": method_id,
        "task": task,
        "source_manifest": str(manifest_path),
        "filtered_manifest": str(filtered_manifest_path),
        "source_data_root": str(data_root) if data_root else None,
        "num_rows": len(rows),
        "final_output_type": final_output_type,
        "final_output_schema": final_output_schema,
        "debug_files": {
            "inputs_jsonl": str(d / "preprocess_inputs.jsonl"),
            "events_jsonl": str(d / "preprocess_debug.jsonl"),
            "last_attempt_json": str(d / "last_preprocess_attempt.json"),
        },
    }
    emit_preprocess_debug(d, summary, last=False, echo=False)
    inputs_path = d / "preprocess_inputs.jsonl"
    # Overwrite this per-method input listing for the current run.
    try:
        inputs_path.write_text("", encoding="utf-8")
    except Exception:
        pass
    for i, row in enumerate(rows):
        rec = debug_record_for_manifest_row(
            phase="source_manifest_input",
            task=task,
            row_index=i,
            row=row,
            data_root=data_root,
            manifest_path=manifest_path,
            filtered_manifest_path=filtered_manifest_path,
            scenario_id=scenario_id,
            method_id=method_id,
            final_output_type=final_output_type,
            final_output_schema=final_output_schema,
        )
        append_jsonl(rec, inputs_path)
    print(f"[preprocess-debug] wrote input path listing: {inputs_path}", file=sys.stderr, flush=True)


class PreprocessingStageError(RuntimeError):
    """Raised when preprocessing cannot produce the manifest needed downstream.

    The attached ``prep`` dictionary is copied into utility_result.json so the
    retained logs and failed stage command remain visible even though temporary
    intermediate artifacts may be deleted.
    """

    def __init__(self, message: str, prep: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.prep = dict(prep or {})


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".wmv"}


# Runtime package configuration.  Your repository may expose the runtime either as
# top-level smartpriv_runtime (when mediator/ is on PYTHONPATH) or as
# mediator.smartpriv_runtime (when the project root is on PYTHONPATH).
_RUNTIME_PROJECT_ROOT = Path(".").resolve()
_RUNTIME_PACKAGE = "auto"
_SUBPROCESS_CUDA_VISIBLE_DEVICES: Optional[str] = None
_SUBPROCESS_CUDA_DEVICE_ORDER: str = "PCI_BUS_ID"


def _query_nvidia_smi_gpus() -> List[Dict[str, str]]:
    """Best-effort nvidia-smi query returning index/uuid/name rows."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return []
    rows: List[Dict[str, str]] = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",", 2)]
        if len(parts) == 3:
            rows.append({"index": parts[0], "uuid": parts[1], "name": parts[2]})
    return rows


def _select_cuda_visible_devices(prefer_gpu_name: Optional[str]) -> Optional[str]:
    """Return the UUID for a preferred GPU name, falling back to None.

    Numeric CUDA_VISIBLE_DEVICES values can be ambiguous on mixed systems because
    CUDA ordering may differ from nvidia-smi ordering.  UUIDs are stable, and if
    a subprocess receives a single GPU UUID, that GPU becomes visible cuda:0.
    """
    if not prefer_gpu_name:
        return None
    needle = str(prefer_gpu_name).strip().lower()
    if not needle:
        return None
    for gpu in _query_nvidia_smi_gpus():
        if needle in gpu.get("name", "").lower():
            return gpu.get("uuid") or None
    return None


def configure_gpu_visibility_for_run(args: argparse.Namespace) -> Dict[str, Any]:
    """Force evaluator/subprocess CUDA visibility to a preferred GPU UUID when possible."""
    global _SUBPROCESS_CUDA_VISIBLE_DEVICES, _SUBPROCESS_CUDA_DEVICE_ORDER
    device = str(getattr(args, "device", "auto") or "auto").strip().lower()
    report: Dict[str, Any] = {"enabled": False, "reason": ""}
    if device == "cpu":
        report["reason"] = "device=cpu"
        return report
    if getattr(args, "no_prefer_gpu_env", False):
        report["reason"] = "--no-prefer-gpu-env"
        return report

    explicit = str(getattr(args, "cuda_visible_devices", "") or "").strip()
    selected = explicit or _select_cuda_visible_devices(getattr(args, "prefer_gpu_name", None))
    if not selected:
        report["reason"] = "no preferred GPU UUID found"
        return report

    _SUBPROCESS_CUDA_DEVICE_ORDER = "PCI_BUS_ID"
    _SUBPROCESS_CUDA_VISIBLE_DEVICES = selected
    os.environ["CUDA_DEVICE_ORDER"] = _SUBPROCESS_CUDA_DEVICE_ORDER
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    report.update({
        "enabled": True,
        "CUDA_DEVICE_ORDER": _SUBPROCESS_CUDA_DEVICE_ORDER,
        "CUDA_VISIBLE_DEVICES": selected,
        "prefer_gpu_name": getattr(args, "prefer_gpu_name", None),
        "explicit_cuda_visible_devices": explicit or None,
    })
    return report


def configure_import_paths(project_root: str | Path) -> None:
    """Make project modules importable for both this process and subprocesses."""
    root = Path(project_root).resolve()
    candidates = [root, root / "mediator"]
    for c in candidates:
        if c.exists():
            cs = str(c)
            if cs not in sys.path:
                sys.path.insert(0, cs)


def set_runtime_config(project_root: str | Path, runtime_package: str = "auto") -> None:
    global _RUNTIME_PROJECT_ROOT, _RUNTIME_PACKAGE
    _RUNTIME_PROJECT_ROOT = Path(project_root).resolve()
    _RUNTIME_PACKAGE = str(runtime_package or "auto")
    configure_import_paths(_RUNTIME_PROJECT_ROOT)


def runtime_package_candidates() -> List[str]:
    if _RUNTIME_PACKAGE and _RUNTIME_PACKAGE != "auto":
        return [_RUNTIME_PACKAGE]
    return ["smartpriv_runtime", "mediator.smartpriv_runtime"]


def make_subprocess_env(cwd: Optional[str | Path]) -> Dict[str, str]:
    """Return env with project root and mediator/ added to PYTHONPATH."""
    env = os.environ.copy()
    roots: List[str] = []
    for x in [cwd, _RUNTIME_PROJECT_ROOT, _RUNTIME_PROJECT_ROOT / "mediator"]:
        if x is None:
            continue
        px = Path(x).resolve()
        if px.exists():
            roots.append(str(px))
    existing = env.get("PYTHONPATH", "")
    parts = roots + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([p for p in parts if p]))
    # Encourage downstream Python scripts to flush logs/progress promptly.
    env.setdefault("PYTHONUNBUFFERED", "1")
    if _SUBPROCESS_CUDA_VISIBLE_DEVICES:
        env["CUDA_DEVICE_ORDER"] = _SUBPROCESS_CUDA_DEVICE_ORDER
        env["CUDA_VISIBLE_DEVICES"] = _SUBPROCESS_CUDA_VISIBLE_DEVICES
    return env


def _cuda_available() -> bool:
    """Best-effort CUDA availability check without making torch a hard import."""
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        # If torch is unavailable at evaluator import time, still prefer the
        # conventional GPU device string.  The downstream script will produce the
        # real error or fall back if it supports retry-cpu-on-error.
        return True


def normalized_device(device: Optional[Any], *, ultralytics: bool = False, torch_script: bool = False) -> Optional[str]:
    """Normalize user shorthand for downstream scripts.

    Device policy:
      * None: do not pass --device; downstream script default decides.
      * auto: prefer GPU when CUDA is available, otherwise CPU.
      * gpu/cuda: force the first GPU.

    Ultralytics CLIs in this repo expect '0' or 'cpu'.  Torch classifiers are
    more likely to expect 'cuda:0' or 'cpu'.  Explicit values such as 'cpu',
    '0', '1', 'cuda:1', or 'mps' are preserved except for aliases below.
    """
    if device is None:
        return None
    text = str(device).strip()
    low = text.lower()
    if not text:
        return None
    if low == "auto":
        use_cuda = _cuda_available()
        if ultralytics:
            return "0" if use_cuda else "cpu"
        if torch_script:
            return "cuda:0" if use_cuda else "cpu"
        return "cuda:0" if use_cuda else "cpu"
    if low in {"gpu", "cuda"}:
        if ultralytics:
            return "0"
        if torch_script:
            return "cuda:0"
        return "cuda:0"
    return text


def cuda_device_report() -> Dict[str, Any]:
    """Return a small, best-effort CUDA/GPU report for progress/debug logs."""
    report: Dict[str, Any] = {"torch_imported": False, "cuda_available": None, "device_count": None, "devices": []}
    try:
        import torch  # type: ignore

        report["torch_imported"] = True
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            count = int(torch.cuda.device_count())
            report["device_count"] = count
            devices = []
            for i in range(count):
                info: Dict[str, Any] = {"index": i}
                try:
                    info["name"] = torch.cuda.get_device_name(i)
                except Exception:
                    info["name"] = None
                try:
                    major, minor = torch.cuda.get_device_capability(i)
                    info["capability"] = f"{major}.{minor}"
                except Exception:
                    info["capability"] = None
                devices.append(info)
            report["devices"] = devices
        else:
            report["device_count"] = 0
    except Exception as exc:
        report["torch_error"] = repr(exc)
    report["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return report


def compact_device_report(report: Dict[str, Any]) -> str:
    if report.get("cuda_available") is True:
        names = []
        for d in report.get("devices") or []:
            name = d.get("name") or "unknown GPU"
            idx = d.get("index")
            cap = d.get("capability")
            names.append(f"{idx}:{name}" + (f" cc{cap}" if cap else ""))
        visible = report.get("CUDA_VISIBLE_DEVICES")
        return "CUDA available; visible devices=" + ", ".join(names) + (f"; CUDA_VISIBLE_DEVICES={visible}" if visible else "")
    if report.get("cuda_available") is False:
        return "CUDA not available; using CPU unless downstream script overrides"
    return "CUDA status unknown" + (f"; torch_error={report.get('torch_error')}" if report.get("torch_error") else "")


def resolved_device_summary(device: Optional[Any]) -> Dict[str, Any]:
    report = cuda_device_report()
    return {
        "requested_device": str(device) if device is not None else None,
        "ultralytics_device": normalized_device(device, ultralytics=True),
        "torch_device": normalized_device(device, torch_script=True),
        "cuda_report": report,
        "cuda_summary": compact_device_report(report),
    }


def cli_option_value(cmd: Sequence[str], flag: str) -> Optional[str]:
    vals = [str(x) for x in cmd]
    for i, x in enumerate(vals):
        if x == flag and i + 1 < len(vals):
            return vals[i + 1]
        if x.startswith(flag + "="):
            return x.split("=", 1)[1]
    return None


def describe_command_device(cmd: Sequence[str]) -> str:
    dev = cli_option_value(cmd, "--device")
    cmd_s = " ".join(str(x) for x in cmd)
    if dev is None:
        return "device=downstream-default"
    low = str(dev).lower()
    if low in {"cpu"}:
        kind = "CPU"
    elif low == "mps":
        kind = "Apple MPS"
    elif low.startswith("cuda") or low.isdigit() or low in {"gpu", "0", "1", "2", "3"}:
        kind = "GPU/CUDA"
    else:
        kind = "custom"
    family = "YOLO/Ultralytics" if ("chokepoint" in cmd_s or "extract_pose" in cmd_s or "yolo" in cmd_s.lower()) else "Torch/classifier"
    return f"{family} device={dev} ({kind})"


TASK_TO_SHORT = {
    "visitor_presence_detection": "chokepoint",
    "fall_detection": "fall",
    "adl_recognition": "youhome",
    "domestic_sound_monitoring": "home_audio",
}

DEFAULT_METHODS = ["raw", "manual", "direct_llm", "full_mediator"]

# Utility evaluation defaults should focus on ablations that actually changed
# decisions or selected operator chains in the context-pipeline generation study.
# These are the ablations worth running by default because they probe the main
# utility/privacy tradeoffs: removing CI gates, optimizing utility alone, and
# disabling/weakening least-revealing selection.
DEFAULT_MEANINGFUL_ABLATION_MODES = [
    "utility_only",
    "no_ci_filter",
    "no_least_revealing",
    "latency_first",
]

# These ablations were intentionally left available for explicit experiments,
# but are skipped by the default utility-evaluation policy because they did not
# change selected outputs in the current pipeline-generation summary.
DEFAULT_SKIPPED_ABLATION_MODES = [
    "no_residual_bounds",
    "uniform_risk_weights",
    "metadata_only",
    "no_staged_flows",
    "first_feasible",
]

# Default module candidates for the task-specific downstream CLIs.  These are
# resolved against the user's repository at startup.  The first importable module
# is used.  The canonical project layout currently uses:
#   baselines/task/chokepoint_presence
#   baselines/task/chime_home_audio or chim_home_audio
#   baselines/task/le2i_fall
#   baselines/task/youhome_adl
DEFAULT_INFER_MODULE_CANDIDATES = {
    "chokepoint": [
        # Current project layout/name.
        "baselines.task.chokepoint_presence.infer_yolo",
        # Other possible names in older/local copies.
        "baselines.task.chokepoint_presence.infer_chokepoint",
        "baselines.task.chokepoint_presence.infer",
        # Backward-compatible older/default names.
        "baselines.task.chokepoint.infer_yolo",
        "baselines.task.chokepoint.infer_chokepoint",
        "baselines.task.chokepoint.infer",
    ],
    "fall_extract_pose": [
        "baselines.task.le2i_fall.extract_pose",
    ],
    "fall_infer": [
        "baselines.task.le2i_fall.infer",
        "baselines.task.le2i_fall.infer_fall",
    ],
    "home_audio": [
        "baselines.task.chime_home_audio.infer_home_audio",
        "baselines.task.chime_home_audio.infer",
        # Accept the spelling the repo may use in older local copies.
        "baselines.task.chim_home_audio.infer_home_audio",
        "baselines.task.chim_home_audio.infer",
        # Backward-compatible older/default names.
        "baselines.task.chime_home.infer_home_audio",
        "baselines.task.chime_home.infer",
    ],
    "youhome": [
        "baselines.task.youhome_adl.infer_youhome",
        "baselines.task.youhome_adl.infer",
    ],
}


def _module_is_importable(module_name: str) -> bool:
    """Return True if `python -m module_name` should be resolvable.

    We use find_spec instead of importing the module so heavy ML dependencies are
    not loaded during evaluator setup.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def resolve_module_candidate(name: str, candidates: Sequence[str]) -> str:
    """Pick the first importable module candidate, else keep the first candidate.

    Keeping the first candidate when none are importable makes the downstream
    error message explicit while preserving deterministic behavior.
    """
    for module_name in candidates:
        if _module_is_importable(module_name):
            progress_write(f"[config] {name} module={module_name}")
            return module_name
    fallback = candidates[0]
    progress_write(
        f"[config] WARNING: could not find an importable module for {name}; "
        f"using {fallback}. Override with --{name.replace('_', '-')}-module or --*-script."
    )
    return fallback

# Conventional locations used only when the corresponding CLI argument was not
# supplied.  Missing paths are not fatal unless the user explicitly requested
# that task or passed --strict-task-config.
AUTO_PATH_CANDIDATES = {
    # Prefer the original task-created manifest names used in this project.
    # Split filtering is applied later, so these should be full train/val/test
    # manifests rather than manually pre-filtered files.
    "chokepoint_manifest": [
        "data/chokepoint/chokepoint_manifest.csv",
        "data/chokepoint/chokepoint_test_manifest.csv",
        "data/chokepoint/manifest.csv",
    ],
    "fall_manifest": [
        # Prefer the raw LE2I task manifest.  If a method/pipeline needs pose
        # keypoints and no keypoints_path exists, the evaluator runs the same
        # downstream YOLO-pose extractor used by the fall app.
        "data/le2i/le2i_manifest.csv",
        "data/le2i/manifest.csv",
    ],
    "fall_precomputed_pose_manifest": [
        # Optional speed-up for raw/no-transform fall utility: this manifest
        # already has keypoints_path rows generated from the original videos.
        "outputs/le2i_manifest_with_keypoints.csv",
        "outputs/le2i_pose/le2i_manifest_with_keypoints.csv",
        "data/le2i/le2i_manifest_with_keypoints.csv",
    ],
    "fall_checkpoint": [
        "outputs/le2i_fall_model/best.pt",
        "outputs/le2i_fall_model/last.pt",
        "outputs/le2i_fall_model/checkpoint.pt",
        "outputs/le2i/checkpoint.pt",
        "outputs/le2i/best.pt",
        "outputs/le2i/model.pt",
    ],
    "fall_label_map": [
        "outputs/le2i_fall_model/label_map.json",
        "outputs/le2i/label_map.json",
        "data/le2i/label_map.json",
    ],
    "home_audio_manifest": [
        "data/chime_home/chime_home_manifest.csv",
        "data/chime_home/manifest.csv",
    ],
    "home_audio_checkpoint": [
        "outputs/chime_home_ast/best.pt",
        "outputs/chime_home_ast/last.pt",
        "outputs/chime_home_ast/checkpoint.pt",
        "outputs/chime_home/checkpoint.pt",
        "outputs/chime_home/best.pt",
        "outputs/chime_home/model.pt",
    ],
    "youhome_manifest": [
        "data/youhome/youhome_manifest.csv",
        "data/youhome/manifest.csv",
    ],
    "youhome_checkpoint": [
        "outputs/youhome_adl_av_logmel_audiofixed_v2/best.pt",
        "outputs/youhome_adl_av_logmel_audiofixed_v2/last.pt",
        "outputs/youhome_adl_av_logmel_audiofixed_v2/checkpoint.pt",
        "outputs/youhome/checkpoint.pt",
        "outputs/youhome/best.pt",
        "outputs/youhome/model.pt",
    ],
    "youhome_label_map": [
        "outputs/youhome_adl_av_logmel_audiofixed_v2/label_map.json",
        "outputs/youhome/label_map.json",
        "data/youhome/label_map.json",
    ],
}


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


def is_youhome_av_cap(cap: Optional[Dict[str, Any]]) -> bool:
    """Return True for the YouHome ADL AV interface used by generated pipelines.

    The generator represents ADL-compatible outputs as an application-level AV
    sample rather than as a plain image/video/audio media cap.  Downstream
    inference still consumes a YouHome manifest containing visual and audio
    paths, so the utility evaluator must treat this cap as materializable.
    """
    if not cap:
        return False
    t = cap_type(cap)
    s = cap_schema(cap)
    content_type = str(cap.get("content_type") or "")
    props = cap.get("properties") if isinstance(cap.get("properties"), dict) else {}
    if t == "application/x-youhome-av-sample":
        return True
    if s in {"youhome_av_manifest_or_sample", "youhome_av_manifest", "youhome_av_sample"}:
        return True
    if "youhome_av" in s:
        return True
    if content_type == "av_content" and bool(props.get("youhome_av_compatible")):
        return True
    return False




FLEXIBLE_SEMANTIC_OUTPUT_TYPES = {
    "application/x-detections",
    "application/x-occupancy",
    "application/x-occupancy-count",
    "application/x-binary-occupancy",
    "application/x-sound-event",
    "application/x-sound-event-label",
    "application/x-decibel-level",
    "application/x-activity-event",
    "application/x-activity-label",
    "application/x-safety-event",
    "application/x-security-event",
    "application/x-aggregate",
    "application/x-motion-features",
    "application/x-multimodal-primitives",
}

FLEXIBLE_SEMANTIC_SCHEMAS = {
    "object_detections",
    "person_detections",
    "occupancy_count",
    "room_occupied",
    "sound_event_label",
    "decibel_level_duration",
    "activity_label",
    "fall_or_safety_event",
    "person_at_door_or_intrusion_event",
    "aggregate_summary",
    "motion_features",
    "multimodal_primitives",
}

SEMANTIC_CAP_ALIASES = {
    "application/x-sound-event-label": "application/x-sound-event",
    "application/x-occupancy-count": "application/x-occupancy",
    "application/x-binary-occupancy": "application/x-occupancy",
    "application/x-activity-label": "application/x-activity-event",
    "application/x-safety-event": "application/x-activity-event",
    "application/x-security-event": "application/x-activity-event",
}


def canonical_semantic_type(t: str) -> str:
    return SEMANTIC_CAP_ALIASES.get(str(t or ""), str(t or ""))


def is_flexible_semantic_cap(cap: Optional[Dict[str, Any]]) -> bool:
    if not cap:
        return False
    t = cap_type(cap)
    s = cap_schema(cap)
    if t in FLEXIBLE_SEMANTIC_OUTPUT_TYPES or canonical_semantic_type(t) in FLEXIBLE_SEMANTIC_OUTPUT_TYPES:
        return True
    if s in FLEXIBLE_SEMANTIC_SCHEMAS:
        return True
    props = cap.get("properties") if isinstance(cap.get("properties"), dict) else {}
    inner = props.get("source_semantic_type") or props.get("semantic_type")
    return bool(inner and str(inner) in FLEXIBLE_SEMANTIC_OUTPUT_TYPES)


def task_supports_semantic_adapter(task: str, cap: Optional[Dict[str, Any]]) -> bool:
    if not is_flexible_semantic_cap(cap):
        return False
    t = canonical_semantic_type(cap_type(cap))
    s = cap_schema(cap)
    if task == "visitor_presence_detection":
        return t in {"application/x-detections", "application/x-occupancy", "application/x-aggregate"} or s in {"object_detections", "person_detections", "occupancy_count", "room_occupied", "aggregate_summary"}
    if task == "fall_detection":
        return t in {"application/x-activity-event", "application/x-aggregate", "application/x-motion-features"} or s in {"fall_or_safety_event", "activity_label", "aggregate_summary", "motion_features"}
    if task == "domestic_sound_monitoring":
        return t in {"application/x-sound-event", "application/x-decibel-level", "application/x-aggregate"} or s in {"sound_event_label", "decibel_level_duration", "aggregate_summary"}
    if task == "adl_recognition":
        return t in {"application/x-activity-event", "application/x-detections", "application/x-sound-event", "application/x-aggregate", "application/x-multimodal-primitives"} or s in {"activity_label", "object_detections", "sound_event_label", "aggregate_summary", "multimodal_primitives"}
    return False

def metric_int_value(result: Dict[str, Any], key: str) -> Optional[int]:
    """Best-effort integer extraction from flattened or nested metric fields."""
    if key in result and result.get(key) not in (None, ""):
        try:
            return int(float(str(result.get(key))))
        except Exception:
            pass
    downstream = result.get("downstream") if isinstance(result.get("downstream"), dict) else {}
    metrics = downstream.get("metrics") if isinstance(downstream.get("metrics"), dict) else {}
    nested_key = key[len("metric_"):] if key.startswith("metric_") else key
    if isinstance(metrics, dict) and nested_key in metrics and metrics.get(nested_key) not in (None, ""):
        try:
            return int(float(str(metrics.get(nested_key))))
        except Exception:
            pass
    metrics_json = downstream.get("metrics_json") if isinstance(downstream, dict) else None
    if metrics_json and Path(str(metrics_json)).exists():
        try:
            loaded = load_json(metrics_json)
            if isinstance(loaded, dict) and nested_key in loaded and loaded.get(nested_key) not in (None, ""):
                return int(float(str(loaded.get(nested_key))))
        except Exception:
            pass
    return None


def existing_result_has_known_adl_missing_audio_bug(result: Dict[str, Any], row: "MethodRow") -> bool:
    """Detect old resumed ADL runs where AV manifests lost their audio root.

    Earlier versions of the evaluator accepted application/x-youhome-av-sample
    by routing it through generic media materialization, but returned
    data_root=None.  The preprocessed manifest therefore kept relative
    audio_path values that the YouHome dataset could not resolve, producing
    missing_audio_samples == n.  Such rows are operationally 'ok' but should be
    invalidated and recomputed with the fixed AV adapter.
    """
    if row.task != "adl_recognition":
        return False
    cap = {
        "semantic_type": row.final_output_type if str(row.final_output_type).startswith("application/") else None,
        "media_type": row.final_output_type if not str(row.final_output_type).startswith("application/") else None,
        "schema": row.final_output_schema,
    }
    if not is_youhome_av_cap(cap):
        return False

    prep = result.get("preprocessing") if isinstance(result.get("preprocessing"), dict) else {}
    prep_status = str(prep.get("preprocessing_status") or result.get("preprocessing_status") or "")
    # Raw/no-transform ADL outputs are expected to use the original manifest and
    # should not be invalidated.
    if "passthrough_raw_or_no_transform" in prep_status:
        return False

    missing_audio = metric_int_value(result, "metric_missing_audio_samples")
    n = metric_int_value(result, "metric_n")
    if missing_audio is None or n in (None, 0):
        return False
    return missing_audio >= n


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


def manifest_has_column(path: str | Path, column: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return column in (reader.fieldnames or [])
    except Exception:
        return False


def manifest_has_nonempty_column(path: str | Path, column: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if column not in (reader.fieldnames or []):
                return False
            for row in reader:
                if str(row.get(column) or "").strip():
                    return True
        return False
    except Exception:
        return False


def filter_manifest_rows(
    manifest_path: str | Path,
    out_path: str | Path,
    split: Optional[str],
    max_samples: Optional[int],
) -> Dict[str, Any]:
    """Write a small filtered manifest and return diagnostic metadata.

    This is intentionally used even for raw/no-transform paths so that utility
    evaluation never accidentally evaluates train/validation rows.  It only
    writes a CSV manifest, not copied media.
    """
    rows, fields = read_csv_rows(manifest_path, max_rows=None)
    original_count = len(rows)
    if split:
        rows = [r for r in rows if str(r.get("split", "")) == str(split)]
    split_count = len(rows)
    if max_samples is not None:
        rows = rows[:max_samples]
    write_csv_rows(rows, out_path, fields)
    return {
        "source_manifest": str(manifest_path),
        "filtered_manifest": str(out_path),
        "split": split,
        "max_samples": max_samples,
        "original_row_count": original_count,
        "after_split_row_count": split_count,
        "written_row_count": len(rows),
    }


def choose_existing_path(project_root: str | Path, candidates: Sequence[str]) -> Optional[str]:
    root = Path(project_root)
    for candidate in candidates:
        p = Path(candidate)
        checks = [p] if p.is_absolute() else [p, root / p]
        for check in checks:
            if check.exists():
                return str(check)
    return None


def default_data_root_from_manifest(manifest_path: Optional[str | Path]) -> Optional[str]:
    """Infer a dataset root from a manifest path.

    Your task manifests usually live directly under the dataset root, e.g.
    data/chokepoint/chokepoint_manifest.csv with frame_dir values like
    P1E_S1/P1E_S1_C1.  In that layout, the manifest parent is the correct
    --data-root for both preprocessing and downstream inference.
    """
    if not manifest_path:
        return None
    p = Path(str(manifest_path))
    if not p.is_absolute():
        project_relative = Path(_RUNTIME_PROJECT_ROOT) / p
        if project_relative.exists():
            p = project_relative
    if p.exists():
        return str(p.parent)
    # Even if the manifest does not exist yet, the parent is still the least
    # surprising default for relative media paths.
    return str(p.parent) if str(p.parent) not in {"", "."} else None


def apply_auto_defaults(args: argparse.Namespace) -> None:
    """Fill conventional modules/paths so a single command can cover all tasks.

    The script still requires task data/checkpoints to exist somewhere.  When
    --tasks is omitted or set to auto/all, tasks without enough configuration
    are skipped with diagnostics instead of crashing the whole run.
    """
    if not args.no_auto_task_config:
        args.chokepoint_infer_module = args.chokepoint_infer_module or resolve_module_candidate("chokepoint", DEFAULT_INFER_MODULE_CANDIDATES["chokepoint"])
        args.fall_extract_pose_module = args.fall_extract_pose_module or resolve_module_candidate("fall_extract_pose", DEFAULT_INFER_MODULE_CANDIDATES["fall_extract_pose"])
        args.fall_infer_module = args.fall_infer_module or resolve_module_candidate("fall_infer", DEFAULT_INFER_MODULE_CANDIDATES["fall_infer"])
        args.home_audio_infer_module = args.home_audio_infer_module or resolve_module_candidate("home_audio", DEFAULT_INFER_MODULE_CANDIDATES["home_audio"])
        args.youhome_infer_module = args.youhome_infer_module or resolve_module_candidate("youhome", DEFAULT_INFER_MODULE_CANDIDATES["youhome"])

        for attr, candidates in AUTO_PATH_CANDIDATES.items():
            if getattr(args, attr, None):
                continue
            found = choose_existing_path(args.project_root, candidates)
            if found:
                setattr(args, attr, found)

        # Infer dataset roots from manifest locations when the user does not
        # provide them explicitly.  This is important for manifests whose media
        # paths are relative to the dataset directory rather than the repo root,
        # e.g. ChokePoint frame_dir=P1E_S1/P1E_S1_C1 under data/chokepoint.
        if getattr(args, "chokepoint_manifest", None) and not getattr(args, "chokepoint_data_root", None):
            args.chokepoint_data_root = default_data_root_from_manifest(args.chokepoint_manifest)
        if getattr(args, "fall_manifest", None) and not getattr(args, "fall_data_root", None):
            args.fall_data_root = default_data_root_from_manifest(args.fall_manifest)
        if getattr(args, "youhome_manifest", None) and not getattr(args, "youhome_data_root", None):
            args.youhome_data_root = default_data_root_from_manifest(args.youhome_manifest)


def requested_task_set(args: argparse.Namespace) -> Tuple[set[str], bool]:
    text = str(args.tasks or "").strip().lower()
    if text in {"", "auto", "all", "*"}:
        return set(), True
    return set(parse_csv_list(args.tasks)), False


def task_config_errors(args: argparse.Namespace, task: str) -> List[str]:
    errors: List[str] = []
    if task == "visitor_presence_detection":
        if not args.chokepoint_manifest:
            errors.append("missing --chokepoint-manifest")
        if not (args.chokepoint_infer_module or args.chokepoint_infer_script):
            errors.append("missing --chokepoint-infer-module/--chokepoint-infer-script")
    elif task == "fall_detection":
        if not args.fall_manifest:
            errors.append("missing --fall-manifest")
        if not (args.fall_infer_module or args.fall_infer_script):
            errors.append("missing --fall-infer-module/--fall-infer-script")
        if not args.fall_checkpoint:
            errors.append("missing --fall-checkpoint")
        # If --fall-label-map is missing, infer_fall() will generate a temporary
        # label map from the manifest. Prefer an explicit training-time label_map
        # for publication-quality numbers because class-index order must match
        # the checkpoint.
        # Pose extraction is required when the manifest does not already contain
        # keypoints_path, or when a pipeline outputs image/video for fall.
        if args.fall_manifest and not manifest_has_nonempty_column(args.fall_manifest, "keypoints_path"):
            if not (args.fall_extract_pose_module or args.fall_extract_pose_script):
                errors.append("missing --fall-extract-pose-module/--fall-extract-pose-script for non-pose manifest")
    elif task == "domestic_sound_monitoring":
        if not args.home_audio_manifest:
            errors.append("missing --home-audio-manifest")
        if not (args.home_audio_infer_module or args.home_audio_infer_script):
            errors.append("missing --home-audio-infer-module/--home-audio-infer-script")
        if not args.home_audio_checkpoint:
            errors.append("missing --home-audio-checkpoint")
    elif task == "adl_recognition":
        if not args.youhome_manifest:
            errors.append("missing --youhome-manifest")
        if not (args.youhome_infer_module or args.youhome_infer_script):
            errors.append("missing --youhome-infer-module/--youhome-infer-script")
        if not args.youhome_checkpoint:
            errors.append("missing --youhome-checkpoint")
        # If --youhome-label-map is missing, infer_youhome() will generate a
        # temporary label map from the manifest label/activity column. Prefer an
        # explicit training-time label_map for publication-quality numbers.
    else:
        errors.append(f"unsupported task {task!r}")
    return errors


def configured_tasks(args: argparse.Namespace, tasks_present: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for task in sorted(set(tasks_present)):
        errors = task_config_errors(args, task)
        out[task] = {"configured": not errors, "errors": errors}
    return out


def run_cmd(cmd: Sequence[str], cwd: Optional[str | Path], log_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = now_ms()
    cmd_list = [str(x) for x in cmd]
    if dry_run:
        write_text("DRY RUN\n" + shlex.join(cmd_list) + "\n", log_path)
        progress_write(f"[downstream] dry run: {shlex.join(cmd_list)}")
        return {"cmd": cmd_list, "returncode": 0, "elapsed_ms": 0, "dry_run": True}

    progress_write(f"[downstream] start: {shlex.join(cmd_list)}")
    progress_write(f"[downstream] device: {describe_command_device(cmd_list)}; {compact_device_report(cuda_device_report())}")
    proc = subprocess.run(
        cmd_list,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=make_subprocess_env(cwd),
    )
    elapsed = now_ms() - started
    stdout = proc.stdout or ""
    write_text("$ " + shlex.join(cmd_list) + "\n\n" + stdout, log_path)
    progress_write(f"[downstream] done rc={proc.returncode} elapsed={elapsed/1000:.1f}s log={log_path}")
    result = {
        "cmd": cmd_list,
        "returncode": proc.returncode,
        "elapsed_ms": elapsed,
        "dry_run": False,
        "log_path": str(log_path),
    }
    if proc.returncode != 0:
        # Put the actionable part of the downstream failure directly in
        # utility_results.json so users do not have to manually open infer.log.
        tail = stdout[-6000:] if len(stdout) > 6000 else stdout
        result["stdout_tail"] = tail
        # A compact one-line hint is useful in CSV/summary views.
        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        if lines:
            result["error_summary"] = lines[-1][:1000]
    return result




def ffmpeg_extract_video_frames(
    video_path: str | Path,
    frame_dir: str | Path,
    *,
    cwd: Optional[str | Path] = None,
    log_path: Optional[str | Path] = None,
    dry_run: bool = False,
    max_frames: Optional[int] = None,
    debug_dir: Optional[str | Path] = None,
    debug_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract a video-only frame directory with ffmpeg, bypassing fragile in-process AVI decode.

    Some LE2I AVI files contain malformed/nonstandard audio streams.  OpenCV can
    report that the video opens and then abort the whole Python interpreter with
    native heap corruption when it touches container/audio metadata.  Running
    ffmpeg as a subprocess with ``-map 0:v:0 -an`` isolates decoder failures and
    forces video-only extraction before any runtime/operator code reads frames.
    """
    video_path = Path(video_path)
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        log_path = frame_dir.parent / (frame_dir.name + ".ffmpeg.log")
    log_path = Path(log_path)

    # Start from a clean directory so stale frames do not make a failed extract
    # look successful.
    if not dry_run:
        for old in frame_dir.glob("*.jpg"):
            try:
                old.unlink()
            except Exception:
                pass

    output_pattern = str(frame_dir / "%06d.jpg")
    cmd: List[str] = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-err_detect",
        "ignore_err",
        "-fflags",
        "+genpts",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-vsync",
        "0",
    ]
    if max_frames is not None and max_frames > 0:
        cmd.extend(["-frames:v", str(max_frames)])
    cmd.append(output_pattern)

    emit_preprocess_debug(debug_dir, {
        **(debug_context or {}),
        "phase": "before_ffmpeg_video_frame_extract",
        "resolved_input_path": str(video_path),
        "resolved_input_exists": video_path.exists(),
        "resolved_media_type": "video/x-raw",
        "ffmpeg_frame_dir": str(frame_dir),
        "ffmpeg_log": str(log_path),
        "ffmpeg_cmd": cmd,
        "max_frames": max_frames,
    }, last=True, echo=True)

    run = run_cmd(cmd, cwd=cwd, log_path=log_path, dry_run=dry_run)
    frames = [] if dry_run else sorted(frame_dir.glob("*.jpg"))
    status = "ok" if (dry_run or (run.get("returncode") == 0 and len(frames) > 0)) else "error"
    info = {
        "status": status,
        "video_path": str(video_path),
        "frame_dir": str(frame_dir),
        "num_frames": len(frames),
        "ffmpeg": run,
        "ffmpeg_log": str(log_path),
        "used_video_only_decode": True,
        "max_frames": max_frames,
    }
    emit_preprocess_debug(debug_dir, {
        **(debug_context or {}),
        "phase": "after_ffmpeg_video_frame_extract",
        "resolved_input_path": str(video_path),
        "ffmpeg_frame_dir": str(frame_dir),
        "ffmpeg_log": str(log_path),
        "ffmpeg_status": status,
        "num_extracted_frames": len(frames),
        "returncode": run.get("returncode"),
    }, last=(status != "ok"), echo=True)
    return info


def sanitize_fall_video_manifest_with_ffmpeg(
    args: argparse.Namespace,
    manifest: str | Path,
    data_root: Optional[str | Path],
    out_manifest: str | Path,
    frame_root: str | Path,
    *,
    log_dir: Optional[str | Path] = None,
    dry_run: bool = False,
    debug_dir: Optional[str | Path] = None,
    debug_context: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Rewrite a LE2I manifest so video rows point to ffmpeg-extracted frame dirs.

    The downstream pose extractor is then given frame_dir rows and no video_path,
    avoiding direct OpenCV/FFmpeg AVI decode inside Python.  Rows that already
    have keypoints_path or frame_dir are preserved.
    """
    manifest = Path(manifest)
    out_manifest = Path(out_manifest)
    frame_root = Path(frame_root)
    log_base = Path(log_dir) if log_dir is not None else out_manifest.parent
    rows, fieldnames = read_csv_rows(manifest, max_rows=None)
    new_rows: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    extracted = 0
    unchanged = 0

    emit_preprocess_debug(debug_dir, {
        **(debug_context or {}),
        "phase": "sanitize_fall_video_manifest_start",
        "manifest": str(manifest),
        "out_manifest": str(out_manifest),
        "frame_root": str(frame_root),
        "data_root": str(data_root) if data_root else None,
        "num_rows": len(rows),
        "max_frames_per_sample": getattr(args, "max_frames_per_sample", None),
    }, last=False, echo=True)

    for i, row in enumerate(rows):
        new_row: Dict[str, Any] = dict(row)
        sample_id = str(row.get("sample_id") or row.get("video_id") or row.get("id") or i)
        existing_frame_dir = str(row.get("frame_dir") or "").strip()
        existing_keypoints = str(row.get("keypoints_path") or "").strip()
        in_path = resolve_manifest_media_path(row, data_root)
        row_ctx = {**(debug_context or {}), "sample_id": sample_id, "row_index": i}

        if existing_keypoints or existing_frame_dir or in_path is None or in_path.is_dir() or in_path.suffix.lower() not in VIDEO_EXTS:
            unchanged += 1
            emit_preprocess_debug(debug_dir, {
                **row_ctx,
                "phase": "sanitize_fall_video_manifest_row_unchanged",
                "resolved_input_path": str(in_path) if in_path else None,
                "reason": "existing_keypoints_or_frame_dir_or_not_video",
                "has_keypoints_path": bool(existing_keypoints),
                "has_frame_dir": bool(existing_frame_dir),
            }, last=False, echo=False)
            new_rows.append(new_row)
            continue

        safe_sample = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("_") or f"row_{i}"
        frames_dir = frame_root / safe_sample
        log_path = log_base / "ffmpeg_frame_extract" / f"{safe_sample}.log"
        rec = ffmpeg_extract_video_frames(
            in_path,
            frames_dir,
            cwd=args.project_root,
            log_path=log_path,
            dry_run=dry_run,
            max_frames=getattr(args, "max_frames_per_sample", None),
            debug_dir=debug_dir,
            debug_context={**row_ctx, "phase_context": "sanitize_fall_video_manifest"},
        )
        rec["sample_id"] = sample_id
        rec["row_index"] = i
        records.append(rec)
        if rec.get("status") == "ok":
            extracted += 1
            new_row["frame_dir"] = str(frames_dir)
            # Keep the source path for debugging but prevent downstream scripts
            # from opening the fragile AVI when frame_dir is available.
            new_row["source_video_path"] = str(in_path)
            if "video_path" in new_row:
                new_row["video_path"] = ""
            if "path" in new_row and str(new_row.get("path") or "") == str(row.get("video_path") or ""):
                new_row["path"] = ""
            new_row["ffmpeg_frame_extract_status"] = "ok"
            new_row["ffmpeg_frame_count"] = str(rec.get("num_frames", 0))
        else:
            new_row["ffmpeg_frame_extract_status"] = "error"
            new_row["preprocess_error"] = "ffmpeg_video_frame_extract_failed"
            failures.append(rec)
        new_rows.append(new_row)

    fields = list(fieldnames)
    for extra in ["frame_dir", "source_video_path", "ffmpeg_frame_extract_status", "ffmpeg_frame_count", "preprocess_error"]:
        if extra not in fields:
            fields.append(extra)
    write_csv_rows(new_rows, out_manifest, fields)
    records_path = out_manifest.with_suffix(".ffmpeg_frames.json")
    write_json(records, records_path)
    info = {
        "status": "ok" if not failures else "error",
        "source_manifest": str(manifest),
        "sanitized_manifest": str(out_manifest),
        "frame_root": str(frame_root),
        "num_rows": len(rows),
        "num_videos_extracted": extracted,
        "num_rows_unchanged": unchanged,
        "num_failures": len(failures),
        "records_json": str(records_path),
        "data_root_after_sanitize": None,
    }
    emit_preprocess_debug(debug_dir, {
        **(debug_context or {}),
        "phase": "sanitize_fall_video_manifest_done",
        **info,
    }, last=bool(failures), echo=True)
    return out_manifest, info

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


_CLI_FLAG_CACHE: Dict[Tuple[str, ...], Optional[set[str]]] = {}


def supported_cli_flags(base_cmd: Sequence[str], cwd: Optional[str | Path]) -> Optional[set[str]]:
    """Best-effort argparse --help probing for optional downstream flags.

    Several of the task scripts evolved over time.  This lets the evaluator run
    against older versions by not passing optional flags that the installed
    script does not support.  If help probing itself fails, return None and keep
    the old permissive behavior.
    """
    key = tuple(str(x) for x in base_cmd)
    if key in _CLI_FLAG_CACHE:
        return _CLI_FLAG_CACHE[key]
    try:
        proc = subprocess.run(
            list(key) + ["--help"],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            env=make_subprocess_env(cwd),
        )
        text = proc.stdout or ""
        flags = set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", text))
        _CLI_FLAG_CACHE[key] = flags if flags else None
        return _CLI_FLAG_CACHE[key]
    except Exception:
        _CLI_FLAG_CACHE[key] = None
        return None


def append_if_supported(cmd: List[str], supported: Optional[set[str]], flag: str, value: Optional[Any]) -> None:
    if value is None or value == "":
        return
    if supported is None or flag in supported:
        cmd.extend([flag, str(value)])


def append_bool_if_supported(cmd: List[str], supported: Optional[set[str]], flag: str, enabled: bool) -> None:
    if enabled and (supported is None or flag in supported):
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


def _infer_task_from_summary_row(row: Dict[str, Any]) -> str:
    """Best-effort task inference for nested summary_by_context rows."""
    explicit = str(row.get("task") or "").strip()
    if explicit:
        return explicit
    parts: List[str] = []
    for key in ["request_id", "result_json", "selected_pipeline_json", "pipeline_spec_json", "matched_output_cap", "matched_output_schema", "final_output_type", "final_output_schema", "output_dir"]:
        value = row.get(key)
        if value:
            parts.append(str(value))
    # Compact result.json often contains request_id/task even if summary_by_context does not.
    result_json = row.get("result_json")
    if result_json and Path(str(result_json)).exists():
        try:
            result = load_json(str(result_json))
            parts.extend(str(result.get(k) or "") for k in ["request_id", "task", "scenario_id"])
            if isinstance(result.get("request_identity"), dict):
                parts.extend(str(v) for v in result["request_identity"].values())
        except Exception:
            pass
    text = " ".join(parts).lower()
    if any(x in text for x in ["fall", "le2i", "pose_keypoints", "fall_or_safety_event", "raw_video_stream"]):
        return "fall_detection"
    if any(x in text for x in ["youhome", "adl", "av_manifest", "youhome_av", "activity_label"]):
        return "adl_recognition"
    if any(x in text for x in ["chime", "chimehome", "domestic_audio", "home_audio", "sound_event", "audio_waveform", "speech_removed", "decibel"]):
        return "domestic_sound_monitoring"
    if any(x in text for x in ["chokepoint", "visitor", "occupancy", "object_detections", "room_occupied", "raw_image_frame", "redacted_image_frame"]):
        return "visitor_presence_detection"
    return ""


def _rows_from_summary_by_context(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sid, methods in sorted((data or {}).items()):
        if not isinstance(methods, dict):
            continue
        for method_key, row in sorted(methods.items()):
            if not isinstance(row, dict):
                continue
            r = dict(row)
            r.setdefault("scenario_id", sid)
            if method_key.startswith("ablation:"):
                mode = method_key.split(":", 1)[1]
                r.setdefault("method_id", method_key)
                r.setdefault("method_kind", "ablation")
                r.setdefault("baseline", method_key)
                r.setdefault("ablation_mode", mode)
            else:
                r.setdefault("method_id", method_key)
                r.setdefault("method_kind", "baseline")
                r.setdefault("baseline", method_key)
            if not r.get("method_output_dir") and r.get("output_dir"):
                r["method_output_dir"] = r.get("output_dir")
            if not r.get("task"):
                r["task"] = _infer_task_from_summary_row(r)
            rows.append(r)
    return rows


def _row_truthy(value: Any) -> bool:
    return value not in (None, "", [], {})


def _row_score_for_discovery(row: Dict[str, Any]) -> int:
    """Score how complete a discovered pipeline row is for merge-mode discovery."""
    score = 0
    for key in [
        "scenario_id",
        "task",
        "method_id",
        "method_kind",
        "baseline",
        "decision",
        "method_output_dir",
        "result_json",
        "selected_pipeline_json",
        "pipeline_spec_json",
        "final_output_type",
        "final_output_schema",
        "matched_output_cap",
        "matched_output_schema",
        "operators",
    ]:
        if _row_truthy(row.get(key)):
            score += 1
    # Prefer rows whose referenced files actually exist; summary rows can be stale.
    for key in ["result_json", "selected_pipeline_json", "pipeline_spec_json", "method_output_dir"]:
        value = row.get(key)
        try:
            if value and Path(str(value)).exists():
                score += 3
        except Exception:
            pass
    return score


def _normalize_discovered_method_id(row: Dict[str, Any]) -> str:
    method_id = str(row.get("method_id") or row.get("baseline") or "").strip()
    ablation_mode = row.get("ablation_mode")
    if ablation_mode not in (None, "") and not method_id.startswith("ablation:"):
        method_id = f"ablation:{ablation_mode}"
    return method_id


def _merge_discovered_row_dicts(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two discovery rows for the same scenario/method, preserving the better row."""
    merged = dict(primary)
    for key, value in secondary.items():
        if not _row_truthy(merged.get(key)) and _row_truthy(value):
            merged[key] = value
    # For path fields, prefer an existing path from either row.
    for key in ["method_output_dir", "result_json", "selected_pipeline_json", "pipeline_spec_json"]:
        vals = [primary.get(key), secondary.get(key)]
        existing = None
        for v in vals:
            try:
                if v and Path(str(v)).exists():
                    existing = v
                    break
            except Exception:
                pass
        if existing:
            merged[key] = existing
    # Keep a small provenance note so utility_eval_plan.json can explain why raw/manual are found.
    sources: List[str] = []
    for r in [primary, secondary]:
        src = r.get("_discovery_source")
        if isinstance(src, list):
            sources.extend(str(x) for x in src)
        elif src:
            sources.append(str(src))
    if sources:
        merged["_discovery_source"] = sorted(set(sources))
    return merged


def _rows_from_pipeline_run_scan(pipeline_root: Path) -> List[Dict[str, Any]]:
    """
    Scan the concrete run directories directly.  This is intentionally used even
    when summary.json/summary_by_context.json exists, because summary files can
    be stale or can omit core baselines such as raw/manual after partial reruns.
    """
    rows: List[Dict[str, Any]] = []

    def _selected_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
        selected = result.get("selected_candidate") or result.get("selected_pipeline") or {}
        if not selected and isinstance(result.get("decision"), dict):
            selected = result.get("decision", {}).get("selected_candidate") or {}
        return selected if isinstance(selected, dict) else {}

    def _decision_from_result(result: Dict[str, Any]) -> str:
        decision = result.get("decision")
        if isinstance(decision, dict):
            return str(decision.get("decision") or decision.get("status") or "")
        return str(decision or result.get("status") or "")

    for p in sorted(pipeline_root.glob("S*/baselines/*/result.json")):
        method_dir = p.parent
        scenario_id = p.parents[2].name
        method_id = method_dir.name
        try:
            result = load_json(p)
        except Exception:
            result = {}
        selected = _selected_from_result(result)
        final_cap = selected.get("final_output_cap") or result.get("final_output_cap") or {}
        rows.append({
            "scenario_id": scenario_id,
            "task": str(result.get("task") or ""),
            "method_id": method_id,
            "method_kind": "baseline",
            "baseline": method_id,
            "ablation_mode": None,
            "method_output_dir": str(method_dir),
            "output_dir": str(method_dir),
            "result_json": str(p),
            "selected_pipeline_json": str(method_dir / "selected_pipeline.json") if (method_dir / "selected_pipeline.json").exists() else None,
            "pipeline_spec_json": str(method_dir / "pipeline_spec.json") if (method_dir / "pipeline_spec.json").exists() else None,
            "final_output_type": cap_type(final_cap),
            "final_output_schema": cap_schema(final_cap),
            "matched_output_cap": selected.get("matched_output_cap") or result.get("matched_output_cap"),
            "matched_output_schema": selected.get("matched_output_schema") or result.get("matched_output_schema"),
            "operators": " -> ".join(str(op.get("operator") or op.get("operator_id") or "") for op in (selected.get("operators") or []) if isinstance(op, dict)),
            "decision": _decision_from_result(result),
            "_discovery_source": "filesystem_scan",
        })

    for p in sorted(pipeline_root.glob("S*/ablations/*/result.json")):
        method_dir = p.parent
        scenario_id = p.parents[2].name
        ablation_mode = method_dir.name
        method_id = f"ablation:{ablation_mode}"
        try:
            result = load_json(p)
        except Exception:
            result = {}
        selected = _selected_from_result(result)
        final_cap = selected.get("final_output_cap") or result.get("final_output_cap") or {}
        rows.append({
            "scenario_id": scenario_id,
            "task": str(result.get("task") or ""),
            "method_id": method_id,
            "method_kind": "ablation",
            "baseline": method_id,
            "ablation_mode": ablation_mode,
            "method_output_dir": str(method_dir),
            "output_dir": str(method_dir),
            "result_json": str(p),
            "selected_pipeline_json": str(method_dir / "selected_pipeline.json") if (method_dir / "selected_pipeline.json").exists() else None,
            "pipeline_spec_json": str(method_dir / "pipeline_spec.json") if (method_dir / "pipeline_spec.json").exists() else None,
            "final_output_type": cap_type(final_cap),
            "final_output_schema": cap_schema(final_cap),
            "matched_output_cap": selected.get("matched_output_cap") or result.get("matched_output_cap"),
            "matched_output_schema": selected.get("matched_output_schema") or result.get("matched_output_schema"),
            "operators": " -> ".join(str(op.get("operator") or op.get("operator_id") or "") for op in (selected.get("operators") or []) if isinstance(op, dict)),
            "decision": _decision_from_result(result),
            "_discovery_source": "filesystem_scan",
        })
    return rows


def discover_method_rows(pipeline_root: Path, discovery_mode: str = "merge") -> List[MethodRow]:
    """
    Discover selected pipeline rows for utility evaluation.

    v16 behavior: default merge mode combines summary.json, summary_by_context.json,
    and a filesystem scan.  This prevents partial/stale summaries from silently
    omitting core baselines such as raw/manual while still preserving compact
    summary metadata when it is available.
    """
    discovery_mode = str(discovery_mode or "merge").strip().lower()
    if discovery_mode not in {"merge", "summary", "scan"}:
        raise ValueError(f"Unknown --pipeline-discovery mode: {discovery_mode}")

    summary_path = pipeline_root / "summary.json"
    summary_by_context_path = pipeline_root / "summary_by_context.json"
    raw_rows: List[Dict[str, Any]] = []

    if discovery_mode in {"merge", "summary"}:
        if summary_path.exists():
            loaded = load_json(summary_path)
            if isinstance(loaded, dict) and all(isinstance(v, dict) for v in loaded.values()):
                summary_rows = _rows_from_summary_by_context(loaded)
            else:
                summary_rows = list(loaded or [])
            for r in summary_rows:
                if isinstance(r, dict):
                    rr = dict(r)
                    rr.setdefault("_discovery_source", "summary.json")
                    raw_rows.append(rr)
        if summary_by_context_path.exists():
            for r in _rows_from_summary_by_context(load_json(summary_by_context_path)):
                rr = dict(r)
                rr.setdefault("_discovery_source", "summary_by_context.json")
                raw_rows.append(rr)

    if discovery_mode in {"merge", "scan"}:
        raw_rows.extend(_rows_from_pipeline_run_scan(pipeline_root))

    # If summary mode found nothing, keep the old conservative fallback behavior.
    if not raw_rows and discovery_mode == "summary":
        raw_rows = _rows_from_pipeline_run_scan(pipeline_root)

    # Dedupe by scenario+method. Prefer the more complete row, then backfill
    # missing fields from the other sources.
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in raw_rows:
        if not isinstance(r, dict):
            continue
        if not r.get("scenario_id"):
            # Try to recover scenario id from output/result paths.
            for key in ["output_dir", "method_output_dir", "result_json"]:
                val = str(r.get(key) or "")
                m = re.search(r"/(S\d{3,})/", "/" + val.strip("/"))
                if m:
                    r["scenario_id"] = m.group(1)
                    break
        method_id = _normalize_discovered_method_id(r)
        if method_id:
            r["method_id"] = method_id
        if not r.get("method_kind"):
            r["method_kind"] = "ablation" if str(method_id).startswith("ablation:") else "baseline"
        if not r.get("baseline"):
            r["baseline"] = method_id
        if str(method_id).startswith("ablation:") and not r.get("ablation_mode"):
            r["ablation_mode"] = str(method_id).split(":", 1)[1]
        if not r.get("method_output_dir") and r.get("output_dir"):
            r["method_output_dir"] = r.get("output_dir")
        if not r.get("task"):
            r["task"] = _infer_task_from_summary_row(r)
        key = (str(r.get("scenario_id") or ""), str(method_id or ""))
        if not key[0] or not key[1]:
            continue
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = dict(r)
        else:
            if _row_score_for_discovery(r) >= _row_score_for_discovery(prev):
                by_key[key] = _merge_discovered_row_dicts(dict(r), prev)
            else:
                by_key[key] = _merge_discovered_row_dicts(prev, dict(r))

    rows = [by_key[k] for k in sorted(by_key.keys())]

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
        if not r.get("task"):
            r["task"] = _infer_task_from_summary_row(r)
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
        configure_import_paths(_RUNTIME_PROJECT_ROOT)
        errors: List[str] = []
        for package in runtime_package_candidates():
            try:
                # Import operators to populate the runtime registry before loading specs.
                try:
                    importlib.import_module(f"{package}.operators")
                except Exception as op_exc:
                    # Some deployments import operators elsewhere; keep trying pipeline/media_io.
                    errors.append(f"{package}.operators: {op_exc!r}")
                pipeline_mod = importlib.import_module(f"{package}.pipeline")
                media_mod = importlib.import_module(f"{package}.media_io")
                self.ExecutablePipeline = getattr(pipeline_mod, "ExecutablePipeline")
                self.item_from_media = getattr(media_mod, "item_from_media")
                self.save_image = getattr(media_mod, "save_image")
                self.save_audio = getattr(media_mod, "save_audio")
                self._loaded = True
                return
            except Exception as exc:
                errors.append(f"{package}: {exc!r}")
        raise ModuleNotFoundError(
            "Could not import smartpriv_runtime. Tried packages "
            f"{runtime_package_candidates()} with project_root={_RUNTIME_PROJECT_ROOT}. "
            "For your repo layout, either run from the project root or pass "
            "--runtime-package mediator.smartpriv_runtime. Errors: " + "; ".join(errors)
        )

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

        # Generic JSON fallback. Flexible app requests may intentionally
        # publish semantic primitives (detections, occupancy, sound events,
        # activity labels, aggregates). Preserve the payload so task-specific
        # semantic adapters can evaluate utility without rerunning a legacy
        # media-consuming downstream model.
        out_path = output_base.with_suffix(".json")
        if hasattr(item, "to_jsonable"):
            try:
                obj = item.to_jsonable(include_payload=True)
            except TypeError:
                obj = item.to_jsonable()
        else:
            obj = {"caps": caps, "data": data, "data_repr": repr(data)[:1000]}
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


def transform_image_dir(
    runtime: RuntimeAdapter,
    pipe: Any,
    in_dir: Path,
    out_dir: Path,
    max_frames: Optional[int],
    task: str,
    debug_dir: Optional[str | Path] = None,
    debug_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted_images(in_dir)
    original_frame_count = len(frames)
    if max_frames is not None:
        frames = frames[:max_frames]
    saved = 0
    errors: List[str] = []
    output_kinds: List[str] = []
    json_paths: List[str] = []
    progress_write(
        f"[frames] sample_dir={in_dir.name}: transforming {len(frames)}/{original_frame_count} image frames "
        f"(the frame progress-bar denominator is frames in this one sample; "
        f"max_frames_per_sample={max_frames if max_frames is not None else 'none'})"
    )
    if progress_enabled() and len(frames) > 1:
        frame_iter = tqdm(
            frames,
            desc=f"frames/sample:{in_dir.name[:20]} n={len(frames)}",
            unit="frame",
            leave=False,
            position=2,
            dynamic_ncols=True,
        )
    else:
        frame_iter = frames
    for img in frame_iter:
        if hasattr(frame_iter, "set_postfix_str"):
            try:
                frame_iter.set_postfix_str(img.name[:60])  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            emit_preprocess_debug(debug_dir, {
                **(debug_context or {}),
                "phase": "before_frame_runtime_process",
                "task": task,
                "frame_path": str(img),
                "resolved_input_path": str(img),
                "resolved_input_exists": img.exists(),
                "resolved_input_suffix": img.suffix.lower(),
                "resolved_media_type": "image/x-raw",
                "output_base": str(out_dir / img.stem),
            }, last=True, echo=False)
            item = runtime.item_from_path(img, media_type="image/x-raw")
            out = pipe.process(item)
            if out is None:
                continue
            meta = runtime.save_output_item(out, out_dir / img.stem, preferred_kind="image")
            output_kinds.append(str(meta.get("kind") or ""))
            if meta.get("kind") == "json" and meta.get("path"):
                json_paths.append(str(meta.get("path")))
            saved += 1
        except Exception as exc:
            emit_preprocess_debug(debug_dir, {
                **(debug_context or {}),
                "phase": "frame_runtime_process_error",
                "task": task,
                "frame_path": str(img),
                "resolved_input_path": str(img),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }, last=True, echo=True)
            errors.append(f"{img}: {exc!r}")
    kind = "semantic_dir" if json_paths and len(json_paths) >= max(1, saved) else "frame_dir"
    return {
        "kind": kind,
        "path": str(out_dir),
        "num_inputs": len(frames),
        "num_outputs": saved,
        "errors": errors[:20],
        "output_kinds": sorted(set(output_kinds)),
        "json_paths": json_paths[:20],
        "num_json_outputs": len(json_paths),
    }


def transform_single_media(
    runtime: RuntimeAdapter,
    pipe: Any,
    in_path: Path,
    out_base: Path,
    task: str,
    debug_dir: Optional[str | Path] = None,
    debug_context: Optional[Dict[str, Any]] = None,
    max_frames_per_sample: Optional[int] = None,
) -> Dict[str, Any]:
    media_type = media_type_for_path(in_path, task)

    # Avoid direct in-process decode of LE2I AVI files.  Some files expose a
    # malformed audio stream that OpenCV/FFmpeg can touch even for video reads,
    # causing native heap corruption.  Extract video-only JPEG frames with
    # ffmpeg in a subprocess, then run the runtime pipeline over images.
    if task == "fall_detection" and in_path.suffix.lower() in VIDEO_EXTS:
        frames_dir = out_base.parent / (out_base.stem + "_ffmpeg_frames")
        ffmpeg_log = out_base.parent / (out_base.stem + ".ffmpeg_frames.log")
        extract_info = ffmpeg_extract_video_frames(
            in_path,
            frames_dir,
            cwd=_RUNTIME_PROJECT_ROOT,
            log_path=ffmpeg_log,
            dry_run=False,
            max_frames=max_frames_per_sample,
            debug_dir=debug_dir,
            debug_context={**(debug_context or {}), "task": task, "phase_context": "runtime_single_media_video_sanitize"},
        )
        if extract_info.get("status") != "ok":
            return {"kind": "none", "path": None, "num_outputs": 0, "ffmpeg_frame_extract": extract_info, "error": "ffmpeg_video_frame_extract_failed"}
        rec = transform_image_dir(
            runtime,
            pipe,
            frames_dir,
            out_base.parent / (out_base.stem + "_runtime_frames"),
            max_frames=None,
            task=task,
            debug_dir=debug_dir,
            debug_context={**(debug_context or {}), "task": task, "source_video_path": str(in_path), "ffmpeg_frame_dir": str(frames_dir)},
        )
        rec["ffmpeg_frame_extract"] = extract_info
        rec["source_video_path"] = str(in_path)
        return rec

    emit_preprocess_debug(debug_dir, {
        **(debug_context or {}),
        "phase": "before_single_media_runtime_process",
        "task": task,
        "resolved_input_path": str(in_path),
        "resolved_input_exists": in_path.exists(),
        "resolved_input_is_dir": in_path.is_dir(),
        "resolved_input_suffix": in_path.suffix.lower(),
        "resolved_media_type": media_type,
        "output_base": str(out_base),
    }, last=True, echo=True)
    item = runtime.item_from_path(in_path, media_type=media_type)
    out = pipe.process(item)
    emit_preprocess_debug(debug_dir, {
        **(debug_context or {}),
        "phase": "after_single_media_runtime_process",
        "task": task,
        "resolved_input_path": str(in_path),
        "output_base": str(out_base),
        "returned_none": out is None,
    }, last=True, echo=False)
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
    debug_dir: Optional[str | Path] = None,
    debug_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rows, fieldnames = read_csv_rows(manifest_path, max_rows=None)
    if split:
        rows = [r for r in rows if str(r.get("split", "")) == str(split)]
    split_filtered_count = len(rows)
    if max_samples is not None:
        rows = rows[:max_samples]

    emit_preprocess_debug(debug_dir, {
        **(debug_context or {}),
        "phase": "transform_manifest_start",
        "task": task,
        "manifest_path": str(manifest_path),
        "source_data_root": str(data_root) if data_root else None,
        "pipeline_spec": str(spec_path),
        "out_manifest_path": str(out_manifest_path),
        "out_data_dir": str(out_data_dir),
        "split": split,
        "max_samples": max_samples,
        "rows_to_transform": len(rows),
        "rows_after_split": split_filtered_count,
    }, last=False, echo=True)

    progress_write(
        f"[preprocess] task={task}: transforming {len(rows)} manifest rows/samples "
        f"(preprocess progress-bar denominator; split={split or 'already-filtered/all'}; "
        f"rows_after_split={split_filtered_count}; max_samples={max_samples if max_samples is not None else 'none'})"
    )

    if dry_run:
        write_csv_rows(rows, out_manifest_path, fieldnames)
        return {"status": "dry_run", "manifest": str(out_manifest_path), "num_rows": len(rows), "num_transformed": 0}

    runtime = RuntimeAdapter()
    emit_preprocess_debug(debug_dir, {
        **(debug_context or {}),
        "phase": "before_load_runtime_pipeline",
        "task": task,
        "pipeline_spec": str(spec_path),
        "out_manifest_path": str(out_manifest_path),
    }, last=True, echo=True)
    progress_write(f"[preprocess] loading runtime pipeline spec={spec_path}")
    pipe = runtime.pipeline_from_spec(spec_path)
    emit_preprocess_debug(debug_dir, {
        **(debug_context or {}),
        "phase": "after_load_runtime_pipeline",
        "task": task,
        "pipeline_spec": str(spec_path),
    }, last=False, echo=True)
    progress_write(f"[preprocess] loaded runtime pipeline spec={spec_path}")
    new_rows: List[Dict[str, Any]] = []
    transform_records: List[Dict[str, Any]] = []
    if progress_enabled():
        iterator = tqdm(
            rows,
            desc=f"preprocess samples:{task} n={len(rows)}",
            unit="sample",
            leave=False,
            position=1,
            dynamic_ncols=True,
        )
    else:
        iterator = rows

    for i, row in enumerate(iterator):
        sample_id = str(row.get("sample_id") or row.get("chunk_id") or i)
        if hasattr(iterator, "set_postfix_str"):
            try:
                iterator.set_postfix_str(f"sample={sample_id}")  # type: ignore[attr-defined]
            except Exception:
                pass
        in_path = resolve_manifest_media_path(row, data_root)
        new_row: Dict[str, Any] = dict(row)
        row_debug_context = {
            **(debug_context or {}),
            "pipeline_spec": str(spec_path),
            "out_manifest_path": str(out_manifest_path),
            "out_data_dir": str(out_data_dir),
        }
        row_debug = debug_record_for_manifest_row(
            phase="before_sample_preprocess",
            task=task,
            row_index=i,
            row=row,
            data_root=data_root,
            manifest_path=manifest_path,
            filtered_manifest_path=manifest_path,
            spec_path=spec_path,
            scenario_id=row_debug_context.get("scenario_id"),
            method_id=row_debug_context.get("method_id"),
            final_output_type=row_debug_context.get("final_output_type"),
            final_output_schema=row_debug_context.get("final_output_schema"),
        )
        row_debug.update(row_debug_context)
        emit_preprocess_debug(debug_dir, row_debug, last=True, echo=True)
        if in_path is None:
            new_row["preprocess_error"] = "missing_input"
            if hasattr(iterator, "set_postfix_str"):
                try:
                    iterator.set_postfix_str(f"sample={sample_id} missing_input")  # type: ignore[attr-defined]
                except Exception:
                    pass
            new_rows.append(new_row)
            continue

        if hasattr(iterator, "set_postfix_str"):
            try:
                iterator.set_postfix_str(f"sample={sample_id} input={str(in_path)[-55:]}")  # type: ignore[attr-defined]
            except Exception:
                pass
        sample_base = out_data_dir / sample_id
        try:
            if in_path.is_dir():
                out_dir = sample_base / "frames"
                rec = transform_image_dir(
                    runtime,
                    pipe,
                    in_path,
                    out_dir,
                    max_frames_per_sample,
                    task,
                    debug_dir=debug_dir,
                    debug_context={**row_debug_context, "sample_id": sample_id, "row_index": i, "source_input_path": str(in_path)},
                )
                if rec.get("kind") == "semantic_dir":
                    new_row["semantic_dir"] = str(out_dir)
                    new_row["preprocessed_json_dir"] = str(out_dir)
                else:
                    new_row["frame_dir"] = str(out_dir)
                # Avoid stale source columns that can be preferred by downstream scripts.
                if "video_path" in new_row:
                    new_row["video_path"] = ""
                if "image_path" in new_row:
                    new_row["image_path"] = ""
            else:
                rec = transform_single_media(
                    runtime,
                    pipe,
                    in_path,
                    sample_base / "output",
                    task,
                    debug_dir=debug_dir,
                    debug_context={**row_debug_context, "sample_id": sample_id, "row_index": i, "source_input_path": str(in_path)},
                    max_frames_per_sample=max_frames_per_sample,
                )
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
            emit_preprocess_debug(debug_dir, {
                **(row_debug_context if 'row_debug_context' in locals() else (debug_context or {})),
                "phase": "sample_preprocess_python_error",
                "task": task,
                "sample_id": sample_id,
                "row_index": i,
                "resolved_input_path": str(in_path) if in_path is not None else None,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }, last=True, echo=True)
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
    unresolved = sum(1 for r in new_rows if r.get("preprocess_error") == "missing_input")
    transformed = sum(1 for r in transform_records if r.get("path"))
    status = "ok"
    if len(new_rows) > 0 and transformed == 0 and unresolved == len(new_rows):
        status = "no_inputs_resolved"
    progress_write(
        f"[preprocess] done task={task} rows={len(new_rows)} transformed={transformed} "
        f"missing_inputs={unresolved} status={status} manifest={out_manifest_path}"
    )
    return {
        "status": status,
        "manifest": str(out_manifest_path),
        "num_rows": len(new_rows),
        "num_transformed": transformed,
        "num_missing_inputs": unresolved,
        "transform_records_json": str(out_manifest_path.with_suffix(".transforms.json")),
        "preprocess_debug_jsonl": str(Path(debug_dir) / "preprocess_debug.jsonl") if debug_dir is not None else None,
        "last_preprocess_attempt_json": str(Path(debug_dir) / "last_preprocess_attempt.json") if debug_dir is not None else None,
    }


# ---------------------------------------------------------------------------
# Task-specific runners
# ---------------------------------------------------------------------------


def infer_visitor(args: argparse.Namespace, manifest: Path, data_root: Optional[str | Path], out_dir: Path, dry_run: bool) -> Dict[str, Any]:
    if not args.chokepoint_infer_module and not args.chokepoint_infer_script:
        raise ValueError("visitor_presence_detection requires --chokepoint-infer-module or --chokepoint-infer-script")
    output_csv = out_dir / "predictions.csv"
    error_csv = out_dir / "prediction_errors.csv"
    cmd = command_base(args.chokepoint_infer_module, args.chokepoint_infer_script, args.python)
    supported = supported_cli_flags(cmd, cwd=args.project_root)
    cmd.extend(["--manifest", str(manifest), "--output-csv", str(output_csv)])
    append_if_supported(cmd, supported, "--data-root", data_root)
    append_if_supported(cmd, supported, "--split", args.split)
    append_if_supported(cmd, supported, "--model", args.chokepoint_model)
    append_if_supported(cmd, supported, "--device", normalized_device(args.device, ultralytics=True))
    append_if_supported(cmd, supported, "--prefer-gpu-name", args.prefer_gpu_name)
    append_if_supported(cmd, supported, "--stride", args.chokepoint_stride)
    append_if_supported(cmd, supported, "--max-frames", args.max_frames_per_sample)
    append_if_supported(cmd, supported, "--error-log", error_csv)
    append_bool_if_supported(cmd, supported, "--safe-cuda", args.safe_cuda)
    append_bool_if_supported(cmd, supported, "--retry-cpu-on-error", args.retry_cpu_on_error)
    # Utility evaluation should not abort an entire method because one frame is
    # unreadable or triggers a backend error.  The error CSV records skipped
    # frames for inspection.
    append_bool_if_supported(cmd, supported, "--skip-error-frames", True)
    append_bool_if_supported(cmd, supported, "--allow-empty-output", True)
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
    return {
        "status": "ok" if run["returncode"] == 0 else "error",
        "inference": run,
        "output_csv": str(output_csv),
        "error_csv": str(error_csv) if 'error_csv' in locals() else None,
        "metrics": metrics,
    }


def run_fall_extract_pose(
    args: argparse.Namespace,
    manifest: Path,
    data_root: Optional[str | Path],
    out_dir: Path,
    dry_run: bool,
    log_dir: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    if not args.fall_extract_pose_module and not args.fall_extract_pose_script:
        raise ValueError("fall image/video path requires --fall-extract-pose-module or --fall-extract-pose-script")
    pose_dir = out_dir / "pose"
    pose_manifest = out_dir / "manifest_with_keypoints.csv"
    retained_log_dir = Path(log_dir) if log_dir is not None else out_dir
    retained_log_dir.mkdir(parents=True, exist_ok=True)
    extract_log = retained_log_dir / "extract_pose.log"

    debug_dir = retained_log_dir if preprocess_debug_enabled(args) else None
    sanitize_info: Optional[Dict[str, Any]] = None
    manifest_for_pose = Path(manifest)
    data_root_for_pose: Optional[str | Path] = data_root

    # Default to evaluator-side video sanitization for LE2I.  It rewrites video
    # rows to ffmpeg-extracted frame_dir rows before the downstream pose script
    # can open fragile AVI files through OpenCV.  The existing
    # --no-sanitize-videos flag disables this if you specifically want the old
    # behavior for comparison.
    if not getattr(args, "no_sanitize_videos", False):
        sanitized_manifest = out_dir / "manifest_video_sanitized_for_pose.csv"
        frame_root = out_dir / "ffmpeg_video_frames_for_pose"
        manifest_for_pose, sanitize_info = sanitize_fall_video_manifest_with_ffmpeg(
            args,
            manifest,
            data_root,
            sanitized_manifest,
            frame_root,
            log_dir=retained_log_dir,
            dry_run=dry_run,
            debug_dir=debug_dir,
            debug_context={
                "task": "fall_detection",
                "phase_context": "before_downstream_pose_extract",
                "manifest_for_eval": str(manifest),
                "data_root": str(data_root) if data_root else None,
            },
        )
        data_root_for_pose = None
        if sanitize_info.get("status") != "ok":
            # Still run pose extraction only in dry-run mode.  In real mode a
            # failed frame extraction means the pose manifest cannot be safely
            # produced.
            if not dry_run:
                return pose_manifest, {
                    "extract_pose_status": "error",
                    "extract_pose": {"returncode": None, "error_summary": "ffmpeg video sanitization failed before pose extraction"},
                    "extract_pose_log": str(extract_log),
                    "pose_manifest": str(pose_manifest),
                    "pose_manifest_exists": False,
                    "pose_dir": str(pose_dir),
                    "video_sanitize": sanitize_info,
                }

    cmd = command_base(args.fall_extract_pose_module, args.fall_extract_pose_script, args.python)
    supported = supported_cli_flags(cmd, cwd=args.project_root)
    cmd.extend(["--manifest", str(manifest_for_pose), "--output-dir", str(pose_dir), "--updated-manifest", str(pose_manifest)])
    append_if_supported(cmd, supported, "--data-root", data_root_for_pose)
    append_if_supported(cmd, supported, "--model", args.fall_pose_model)
    append_if_supported(cmd, supported, "--device", normalized_device(args.device, ultralytics=True))
    append_if_supported(cmd, supported, "--stride", args.fall_pose_stride)
    append_if_supported(cmd, supported, "--max-frames", args.max_frames_per_sample)
    # Do not pass --no-sanitize-videos when the evaluator already sanitized the
    # manifest.  If the downstream script has its own sanitizer, allowing it is
    # harmless for frame_dir rows and gives another safety net.
    if getattr(args, "no_sanitize_videos", False):
        append_bool_if_supported(cmd, supported, "--no-sanitize-videos", True)
    run = run_cmd(cmd, cwd=args.project_root, log_path=extract_log, dry_run=dry_run)

    output_exists = bool(pose_manifest.exists()) if not dry_run else True
    status = "ok" if run.get("returncode") == 0 and output_exists else "error"
    if run.get("returncode") == 0 and not output_exists and not dry_run:
        run = dict(run)
        run["error_summary"] = f"Pose extraction returned 0 but did not create expected manifest: {pose_manifest}"
    return pose_manifest, {
        "extract_pose_status": status,
        "extract_pose": run,
        "extract_pose_log": str(extract_log),
        "pose_manifest": str(pose_manifest),
        "pose_manifest_exists": output_exists,
        "pose_dir": str(pose_dir),
        "pose_input_manifest": str(manifest_for_pose),
        "pose_input_data_root": str(data_root_for_pose) if data_root_for_pose else None,
        "video_sanitize": sanitize_info,
    }


def _ensure_fall_pose_manifest_ready(args: argparse.Namespace, pose_manifest: Path, pose_info: Dict[str, Any], prep: Dict[str, Any], status_on_error: str) -> None:
    """Stop before fall inference if pose extraction failed or produced no manifest."""
    if getattr(args, "dry_run", False):
        return
    ok = (pose_info.get("extract_pose_status") == "ok") and pose_manifest.exists()
    if ok:
        return
    prep.update(pose_info)
    prep["preprocessing_status"] = status_on_error
    log_path = pose_info.get("extract_pose_log") or (pose_info.get("extract_pose") or {}).get("log_path")
    msg = (
        "Fall pose extraction did not produce the keypoint manifest needed by fall inference. "
        f"Expected manifest: {pose_manifest}."
        + (f" See retained log: {log_path}." if log_path else "")
    )
    raise PreprocessingStageError(msg, prep)


def infer_fall(args: argparse.Namespace, manifest: Path, data_root: Optional[str | Path], out_dir: Path, dry_run: bool) -> Dict[str, Any]:
    if not args.fall_infer_module and not args.fall_infer_script:
        raise ValueError("fall_detection requires --fall-infer-module or --fall-infer-script")
    if not args.fall_checkpoint:
        raise ValueError("fall_detection requires --fall-checkpoint")
    label_map_path = ensure_task_label_map(args, "fall_detection", manifest, out_dir)
    output_csv = out_dir / "predictions.csv"
    video_output_csv = out_dir / "predictions_video.csv"
    metrics_json = out_dir / "metrics.json"
    cmd = command_base(args.fall_infer_module, args.fall_infer_script, args.python)
    cmd.extend([
        "--manifest", str(manifest),
        "--checkpoint", str(args.fall_checkpoint),
        "--label-map", str(label_map_path),
        "--output-csv", str(output_csv),
        "--video-output-csv", str(video_output_csv),
        "--metrics-json", str(metrics_json),
    ])
    append_if(cmd, "--data-root", data_root)
    append_if(cmd, "--split", args.split or "test")
    append_if(cmd, "--sample-mode", args.fall_sample_mode)
    append_if(cmd, "--device", normalized_device(args.device, torch_script=True))
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
    append_if(cmd, "--device", normalized_device(args.device, torch_script=True))
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
    if not args.youhome_checkpoint:
        raise ValueError("adl_recognition requires --youhome-checkpoint")
    label_map_path = ensure_task_label_map(args, "adl_recognition", manifest, out_dir)
    output_csv = out_dir / "predictions.csv"
    metrics_json = out_dir / "metrics.json"
    cmd = command_base(args.youhome_infer_module, args.youhome_infer_script, args.python)
    cmd.extend([
        "--manifest", str(manifest),
        "--checkpoint", str(args.youhome_checkpoint),
        "--label-map", str(label_map_path),
        "--output-csv", str(output_csv),
        "--metrics-json", str(metrics_json),
    ])
    append_if(cmd, "--data-root", data_root)
    append_if(cmd, "--split", args.split or "test")
    append_if(cmd, "--device", normalized_device(args.device, torch_script=True))
    append_if(cmd, "--batch-size", args.batch_size)
    append_if(cmd, "--num-workers", args.num_workers)
    append_if(cmd, "--tta-runs", args.youhome_tta_runs)
    run = run_cmd(cmd, cwd=args.project_root, log_path=out_dir / "infer.log", dry_run=dry_run)
    metrics = load_json(metrics_json) if metrics_json.exists() else {}
    return {"status": "ok" if run["returncode"] == 0 else "error", "inference": run, "output_csv": str(output_csv), "metrics_json": str(metrics_json), "metrics": metrics}



# ---------------------------------------------------------------------------
# Utility metric computation
# ---------------------------------------------------------------------------

TRUE_COL_CANDIDATES = ["y_true", "true", "target", "label", "gt", "ground_truth", "person_present_gt"]
PRED_COL_CANDIDATES = ["y_pred", "pred", "prediction", "pred_label", "person_present", "person_present_pred"]
PERSON_LABELS = {"person", "pedestrian", "human", "visitor"}


def metric_is_blank(x: Any) -> bool:
    if x is None:
        return True
    try:
        import pandas as pd  # type: ignore
        if pd.isna(x):
            return True
    except Exception:
        pass
    return str(x).strip() == ""


def metric_as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    if metric_is_blank(x):
        return default
    text = str(x).strip()
    stem = Path(text).stem
    for candidate in (text, stem):
        try:
            return int(float(candidate))
        except Exception:
            continue
    m = re.search(r"(\d+)", stem)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return default
    return default


def metric_candidate_paths(value: Any, roots: Iterable[Optional[Path]]) -> List[Path]:
    if metric_is_blank(value):
        return []
    raw = Path(str(value))
    out: List[Path] = []
    if raw.is_absolute():
        out.append(raw)
    else:
        out.append(raw)
        for root in roots:
            if root is not None:
                out.append(root / raw)
    dedup: List[Path] = []
    seen: set[str] = set()
    for pth in out:
        key = str(pth)
        if key not in seen:
            dedup.append(pth)
            seen.add(key)
    return dedup


def metric_infer_data_root(source_manifest: Optional[Path], explicit: Optional[str | Path] = None) -> Optional[Path]:
    if explicit:
        return Path(explicit)
    if source_manifest is None:
        return None
    return source_manifest.parent


def metric_resolve_path(value: Any, source_manifest: Optional[Path] = None, data_root: Optional[Path] = None) -> Optional[Path]:
    roots: List[Optional[Path]] = [data_root, source_manifest.parent if source_manifest is not None else None, Path.cwd(), _RUNTIME_PROJECT_ROOT]
    for pth in metric_candidate_paths(value, roots):
        if pth.exists():
            return pth
    return None


def parse_chokepoint_xml(xml_path: Path) -> Dict[int, int]:
    """Return frame_index -> person_present for common frame/object XML formats."""
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()
    gt: Dict[int, int] = {}

    # CVAT interpolation format: <track label="person"><box frame="123" outside="0" .../></track>
    for track in root.iter():
        if track.tag.lower().endswith("track"):
            label = str(track.attrib.get("label", "")).strip().lower()
            label_is_person = (not label) or label in PERSON_LABELS
            for box in track.iter():
                if not box.tag.lower().endswith("box"):
                    continue
                frame = metric_as_int(box.attrib.get("frame"))
                if frame is None:
                    continue
                outside = str(box.attrib.get("outside", "0")).strip().lower() in {"1", "true", "yes"}
                if not outside and label_is_person:
                    gt[frame] = 1
                else:
                    gt.setdefault(frame, 0)

    # CVAT image format: <image id="123" name="00000123.jpg"><box label="person" .../></image>
    for image in root.iter():
        if not image.tag.lower().endswith("image"):
            continue
        frame = metric_as_int(image.attrib.get("id"), None)
        if frame is None:
            frame = metric_as_int(image.attrib.get("name"), None)
        if frame is None:
            continue
        present = 0
        for child in image.iter():
            tag = child.tag.lower().split("}")[-1]
            if tag in {"box", "polygon", "points", "object"}:
                label = str(child.attrib.get("label", child.attrib.get("name", ""))).strip().lower()
                if (not label) or label in PERSON_LABELS:
                    outside = str(child.attrib.get("outside", "0")).strip().lower() in {"1", "true", "yes"}
                    if not outside:
                        present = 1
                        break
        gt[frame] = max(gt.get(frame, 0), present)

    # Generic frame format: <frame number="..."> ... objects ... </frame>
    for frame_el in root.iter():
        tag = frame_el.tag.lower().split("}")[-1]
        if tag not in {"frame", "img", "image"}:
            continue
        frame = None
        for key in ("number", "num", "frame", "id", "index", "name"):
            frame = metric_as_int(frame_el.attrib.get(key), None)
            if frame is not None:
                break
        if frame is None:
            continue
        present = 0
        for child in frame_el.iter():
            ctag = child.tag.lower().split("}")[-1]
            if ctag in {"object", "person", "box", "bndbox", "bbox"}:
                label = str(child.attrib.get("label", child.attrib.get("name", child.tag))).strip().lower().split("}")[-1]
                if label in PERSON_LABELS or ctag in {"person", "object", "box", "bndbox", "bbox"}:
                    present = 1
                    break
        gt[frame] = max(gt.get(frame, 0), present)

    return gt


def binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    beta2 = 2.0
    f2 = (1 + beta2**2) * precision * recall / (beta2**2 * precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "n": len(y_true),
        "support_pos": sum(y_true),
        "support_neg": len(y_true) - sum(y_true),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "accuracy": accuracy,
    }


def evaluate_chokepoint_metrics(
    prediction_csv: str | Path,
    source_manifest: str | Path,
    data_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Frame-level binary person/visitor-presence metrics for ChokePoint."""
    import pandas as pd  # type: ignore

    pred_path = Path(prediction_csv)
    pred = pd.read_csv(pred_path)
    if pred.empty:
        return {"status": "error", "error": "prediction CSV is empty"}
    required = {"sample_id", "frame_index", "person_present"}
    missing = sorted(required - set(pred.columns))
    if missing:
        return {"status": "error", "error": f"prediction CSV lacks required columns: {missing}"}

    source_manifest_path = Path(source_manifest)
    if not source_manifest_path.is_absolute():
        candidate = _RUNTIME_PROJECT_ROOT / source_manifest_path
        if candidate.exists():
            source_manifest_path = candidate
    if not source_manifest_path.exists():
        return {"status": "error", "error": f"source manifest not found: {source_manifest_path}"}

    root = metric_infer_data_root(source_manifest_path, data_root)
    manifest = pd.read_csv(source_manifest_path)
    if "split" in manifest.columns and "split" in pred.columns:
        split_values = set(str(x) for x in pred["split"].dropna().unique())
        if len(split_values) == 1:
            manifest = manifest[manifest["split"].astype(str) == next(iter(split_values))]
    if "sample_id" not in manifest.columns or "xml_path" not in manifest.columns:
        return {"status": "error", "error": "source manifest lacks sample_id or xml_path columns"}

    pred_samples = set(str(x) for x in pred["sample_id"].astype(str).unique())
    manifest = manifest[manifest["sample_id"].astype(str).isin(pred_samples)]

    missing_xml: List[str] = []
    empty_xml: List[str] = []
    gt_by_sample: Dict[str, Dict[int, int]] = {}
    for _, mrow in manifest.iterrows():
        sample_id = str(mrow["sample_id"])
        xml_path = metric_resolve_path(mrow.get("xml_path"), source_manifest=source_manifest_path, data_root=root)
        if xml_path is None:
            missing_xml.append(sample_id)
            continue
        try:
            gt_map = parse_chokepoint_xml(xml_path)
        except Exception as e:
            return {"status": "error", "error": f"failed parsing XML for sample {sample_id}: {e}"}
        if not gt_map:
            empty_xml.append(sample_id)
        gt_by_sample[sample_id] = gt_map

    y_true: List[int] = []
    y_pred: List[int] = []
    for _, prow in pred.iterrows():
        sample_id = str(prow["sample_id"])
        frame = metric_as_int(prow["frame_index"])
        if frame is None:
            continue
        gt_map = gt_by_sample.get(sample_id)
        if not gt_map:
            continue
        y_true.append(int(gt_map.get(frame, 0)))
        y_pred.append(1 if int(prow["person_present"]) > 0 else 0)

    if not y_true:
        return {
            "status": "error",
            "error": "no prediction rows could be aligned to XML ground truth",
            "missing_xml_samples": missing_xml[:10],
            "empty_xml_samples": empty_xml[:10],
            "n_missing_xml_samples": len(missing_xml),
            "n_empty_xml_samples": len(empty_xml),
        }

    out = binary_metrics(y_true, y_pred)
    out.update({
        "status": "ok",
        "task": "visitor_presence_detection",
        "level": "frame_binary_presence",
        "n_prediction_rows": int(len(pred)),
        "n_aligned_rows": int(len(y_true)),
        "n_missing_xml_samples": len(missing_xml),
        "n_empty_xml_samples": len(empty_xml),
    })
    return out


def evaluate_generic_binary_metrics(prediction_csv: str | Path, task: str) -> Dict[str, Any]:
    """Fallback metrics when prediction CSV already contains true/pred columns."""
    import pandas as pd  # type: ignore

    pred_path = Path(prediction_csv)
    df = pd.read_csv(pred_path)
    true_col = next((c for c in TRUE_COL_CANDIDATES if c in df.columns), None)
    pred_col = next((c for c in PRED_COL_CANDIDATES if c in df.columns and c != true_col), None)
    if true_col is None or pred_col is None:
        return {"status": "skipped", "error": "no generic true/pred columns found"}
    y_true = [1 if int(x) > 0 else 0 for x in df[true_col].fillna(0)]
    y_pred = [1 if int(x) > 0 else 0 for x in df[pred_col].fillna(0)]
    out = binary_metrics(y_true, y_pred)
    out.update({"status": "ok", "task": task, "level": "generic_binary", "true_col": true_col, "pred_col": pred_col})
    return out


def compute_downstream_metrics(args: argparse.Namespace, row: MethodRow, prep: Dict[str, Any], downstream: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    """Compute task metrics after downstream inference and write metrics.json.

    Downstream scripts for fall/audio/ADL may already produce metrics.json; this
    function fills the gap for ChokePoint and also provides a generic binary
    fallback for any prediction CSV that includes both truth and prediction cols.
    """
    if args.dry_run:
        return {"status": "skipped", "error": "dry_run"}
    if downstream.get("status") != "ok":
        return {"status": "skipped", "error": "downstream status is not ok"}
    output_csv = downstream.get("output_csv")
    if not output_csv or not Path(str(output_csv)).exists():
        return {"status": "skipped", "error": "prediction CSV does not exist"}

    # Preserve task metrics produced by downstream scripts if present, but let
    # ChokePoint compute its own frame-level metric file because infer_yolo only
    # produces predictions.
    if row.task == "visitor_presence_detection":
        metrics = evaluate_chokepoint_metrics(
            prediction_csv=output_csv,
            source_manifest=prep.get("source_manifest"),
            data_root=prep.get("source_data_root") or getattr(args, "chokepoint_data_root", None),
        )
    else:
        existing = downstream.get("metrics") if isinstance(downstream.get("metrics"), dict) else {}
        # If this is a resumed result, the nested metrics dictionary may be
        # absent even though the downstream script already wrote metrics.json.
        if not existing and downstream.get("metrics_json") and Path(str(downstream.get("metrics_json"))).exists():
            try:
                existing = load_json(str(downstream.get("metrics_json")))
            except Exception:
                existing = {}
        if existing:
            metrics = dict(existing)
            metrics.setdefault("status", "ok")
            metrics.setdefault("task", row.task)
            metrics.setdefault("level", "downstream_native")
        else:
            metrics = evaluate_generic_binary_metrics(output_csv, row.task)

    metrics_json = work_dir / "metrics.json"
    write_json(metrics, metrics_json)
    downstream["metrics_json"] = str(metrics_json)
    downstream["metrics"] = metrics
    return metrics


def format_metric_line(metrics: Dict[str, Any]) -> str:
    if not metrics:
        return "no metrics"
    status = metrics.get("status") or metrics.get("metric_status") or "unknown"
    if status != "ok":
        return f"status={status} error={metrics.get('error') or metrics.get('metric_error')}"
    parts = [f"status={status}"]
    for k in ("precision", "recall", "f1", "f2", "accuracy"):
        if k in metrics and isinstance(metrics[k], (int, float)):
            parts.append(f"{k}={metrics[k]:.4f}")
    for k in ("tp", "fp", "fn", "tn", "n"):
        if k in metrics:
            parts.append(f"{k}={metrics[k]}")
    return " ".join(parts)

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


def prepare_manifest_for_method(args: argparse.Namespace, row: MethodRow, work_dir: Path, log_dir: Optional[Path] = None) -> Tuple[Path, Optional[str | Path], Dict[str, Any]]:
    """Return a manifest/data-root to feed into downstream inference.

    This function always evaluates the requested split only.  It first writes a
    tiny filtered manifest into the intermediate workspace, then performs any
    required preprocessing against that filtered manifest.  Large transformed
    artifacts still live only in the temporary intermediate workspace unless
    --keep-intermediate-data is set.
    """
    manifest, data_root = task_manifest_and_root(args, row.task)
    spec = load_spec(row.pipeline_spec_json)
    final_cap = (spec or {}).get("final_output_cap") or {
        "semantic_type": row.final_output_type if row.final_output_type.startswith("application/") else None,
        "media_type": row.final_output_type if not row.final_output_type.startswith("application/") else None,
        "schema": row.final_output_schema,
    }

    debug_dir = Path(log_dir) if (log_dir is not None and preprocess_debug_enabled(args)) else None

    filtered_manifest = work_dir / "source_manifest_eval_split.csv"
    filter_info = filter_manifest_rows(
        manifest,
        filtered_manifest,
        split=args.split,
        max_samples=args.max_samples,
    )
    manifest_for_eval = filtered_manifest

    prep: Dict[str, Any] = {
        "source_manifest": str(manifest),
        "source_data_root": str(data_root) if data_root else None,
        "pipeline_spec": str(row.pipeline_spec_json) if row.pipeline_spec_json else None,
        "final_output_cap": final_cap,
        "preprocessing_status": "not_started",
        "evaluation_split": args.split,
        "source_manifest_filter": filter_info,
        "preprocess_debug_dir": str(debug_dir) if debug_dir is not None else None,
        "preprocess_debug_jsonl": str(debug_dir / "preprocess_debug.jsonl") if debug_dir is not None else None,
        "preprocess_inputs_jsonl": str(debug_dir / "preprocess_inputs.jsonl") if debug_dir is not None else None,
        "last_preprocess_attempt_json": str(debug_dir / "last_preprocess_attempt.json") if debug_dir is not None else None,
    }

    # Durable breadcrumb listing all input paths in the filtered split.  This is
    # written before any runtime/operator call so native crashes still leave the
    # exact video/audio/frame path that was about to be processed.
    try:
        debug_rows, _debug_fields = read_csv_rows(filtered_manifest, max_rows=None)
        write_manifest_input_debug(
            debug_dir=debug_dir,
            rows=debug_rows,
            data_root=data_root,
            task=row.task,
            manifest_path=manifest,
            filtered_manifest_path=filtered_manifest,
            scenario_id=row.scenario_id,
            method_id=row.method_id,
            final_output_type=cap_type(final_cap),
            final_output_schema=cap_schema(final_cap),
        )
    except Exception as exc:
        progress_write(f"[preprocess-debug] failed to write source input path listing for {row.scenario_id}/{row.method_id}: {exc!r}")

    debug_context = {
        "scenario_id": row.scenario_id,
        "method_id": row.method_id,
        "task": row.task,
        "final_output_type": cap_type(final_cap),
        "final_output_schema": cap_schema(final_cap),
        "source_manifest": str(manifest),
        "filtered_manifest": str(filtered_manifest),
        "source_data_root": str(data_root) if data_root else None,
    }

    # Fall raw/no-transform needs special handling: the downstream app contains
    # pose extraction, so raw media must be converted to YOLO pose keypoints for
    # the attached fall classifier unless the manifest already has keypoints.
    if row.task == "fall_detection" and (not spec or pipeline_has_no_transform(spec) or row.baseline == "raw"):
        if manifest_has_nonempty_column(manifest_for_eval, "keypoints_path"):
            prep["preprocessing_status"] = "passthrough_existing_pose_manifest_test_split"
            return manifest_for_eval, data_root, prep

        # Optional speed-up: for the raw/no-transform fall app, a precomputed
        # keypoints manifest is equivalent to running the downstream pose
        # extractor over the original videos, but avoids recomputing YOLO-pose.
        precomputed_pose_manifest = getattr(args, "fall_precomputed_pose_manifest", None)
        if precomputed_pose_manifest and Path(str(precomputed_pose_manifest)).exists():
            pose_filtered_manifest = work_dir / "precomputed_pose_manifest_eval_split.csv"
            pose_filter_info = filter_manifest_rows(
                precomputed_pose_manifest,
                pose_filtered_manifest,
                split=args.split,
                max_samples=args.max_samples,
            )
            if manifest_has_nonempty_column(pose_filtered_manifest, "keypoints_path"):
                prep["preprocessing_status"] = "passthrough_precomputed_pose_manifest_test_split"
                prep["source_pose_manifest"] = str(precomputed_pose_manifest)
                prep["source_pose_manifest_filter"] = pose_filter_info
                return pose_filtered_manifest, None, prep

        emit_preprocess_debug(debug_dir, {**debug_context, "phase": "before_run_fall_extract_pose", "manifest_for_eval": str(manifest_for_eval), "data_root": str(data_root) if data_root else None}, last=True, echo=True)
        pose_manifest, pose_info = run_fall_extract_pose(args, manifest_for_eval, data_root, work_dir, args.dry_run, log_dir=log_dir)
        prep.update(pose_info)
        _ensure_fall_pose_manifest_ready(args, pose_manifest, pose_info, prep, "raw_media_downstream_yolo_pose_failed")
        prep["preprocessing_status"] = "raw_media_then_downstream_yolo_pose_test_split"
        return pose_manifest, None, prep

    # Raw/no-transform for other tasks: use the split-filtered manifest only;
    # this does not copy underlying media/audio.
    if not spec or pipeline_has_no_transform(spec) or row.baseline == "raw":
        prep["preprocessing_status"] = "passthrough_raw_or_no_transform_test_split"
        return manifest_for_eval, data_root, prep

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
                    manifest_for_eval,
                    data_root,
                    pre_spec_path,
                    pre_manifest,
                    pre_dir,
                    task=row.task,
                    split=None,
                    max_samples=None,
                    max_frames_per_sample=args.max_frames_per_sample,
                    dry_run=args.dry_run,
                    debug_dir=debug_dir,
                    debug_context={**debug_context, "preprocess_branch": "pre_pose_media"},
                )
                prep["pre_pose_preprocessing"] = pre_result
                emit_preprocess_debug(debug_dir, {**debug_context, "phase": "before_run_fall_extract_pose", "manifest_for_eval": str(pre_manifest), "data_root": None, "after_pre_pose_preprocessing": True}, last=True, echo=True)
                pose_manifest, pose_info = run_fall_extract_pose(args, pre_manifest, None, work_dir, args.dry_run, log_dir=log_dir)
            else:
                emit_preprocess_debug(debug_dir, {**debug_context, "phase": "before_run_fall_extract_pose", "manifest_for_eval": str(manifest_for_eval), "data_root": str(data_root) if data_root else None, "recognized_pose_extractor_stage": True}, last=True, echo=True)
                pose_manifest, pose_info = run_fall_extract_pose(args, manifest_for_eval, data_root, work_dir, args.dry_run, log_dir=log_dir)
            prep.update(pose_info)
            _ensure_fall_pose_manifest_ready(args, pose_manifest, pose_info, prep, "pose_extraction_failed_before_fall_inference")
            prep["preprocessing_status"] = "pose_extracted_with_downstream_yolo_pose_test_split"
            return pose_manifest, None, prep

        # Pipeline claims pose but does not contain a recognized pose extractor.
        # Try runtime pipeline directly; it may still produce keypoints_path rows.
        out_manifest = work_dir / "preprocessed_manifest.csv"
        transformed = transform_manifest_with_pipeline(
            manifest_for_eval,
            data_root,
            Path(row.pipeline_spec_json),
            out_manifest,
            work_dir / "preprocessed_data",
            task=row.task,
            split=None,
            max_samples=None,
            max_frames_per_sample=args.max_frames_per_sample,
            dry_run=args.dry_run,
            debug_dir=debug_dir,
            debug_context={**debug_context, "preprocess_branch": "runtime_pose_manifest"},
        )
        prep.update(transformed)
        prep["preprocessing_status"] = "runtime_pose_manifest_test_split"
        return out_manifest, None, prep

    # Fall image/video output: app runs its internal pose detector before fall classifier.
    if row.task == "fall_detection" and (is_image_cap(final_cap) or is_video_cap(final_cap)):
        media_manifest = work_dir / "preprocessed_media_manifest.csv"
        transformed = transform_manifest_with_pipeline(
            manifest_for_eval,
            data_root,
            Path(row.pipeline_spec_json),
            media_manifest,
            work_dir / "preprocessed_media",
            task=row.task,
            split=None,
            max_samples=None,
            max_frames_per_sample=args.max_frames_per_sample,
            dry_run=args.dry_run,
            debug_dir=debug_dir,
            debug_context={**debug_context, "preprocess_branch": "fall_media_preprocess"},
        )
        prep.update(transformed)
        emit_preprocess_debug(debug_dir, {**debug_context, "phase": "before_run_fall_extract_pose", "manifest_for_eval": str(media_manifest), "data_root": None, "after_media_preprocessing": True}, last=True, echo=True)
        pose_manifest, pose_info = run_fall_extract_pose(args, media_manifest, None, work_dir, args.dry_run, log_dir=log_dir)
        prep.update(pose_info)
        _ensure_fall_pose_manifest_ready(args, pose_manifest, pose_info, prep, "media_preprocessed_downstream_pose_failed")
        prep["preprocessing_status"] = "media_preprocessed_then_downstream_pose_test_split"
        return pose_manifest, None, prep

    # Audio task with audio/x-filtered or waveform-like output.
    if row.task == "domestic_sound_monitoring" and is_audio_cap(final_cap):
        out_manifest = work_dir / "preprocessed_manifest.csv"
        transformed = transform_manifest_with_pipeline(
            manifest_for_eval,
            data_root,
            Path(row.pipeline_spec_json),
            out_manifest,
            work_dir / "preprocessed_audio",
            task=row.task,
            split=None,
            max_samples=None,
            max_frames_per_sample=args.max_frames_per_sample,
            dry_run=args.dry_run,
            debug_dir=debug_dir,
            debug_context={**debug_context, "preprocess_branch": "audio_preprocessed"},
        )
        prep.update(transformed)
        prep["preprocessing_status"] = "audio_preprocessed_test_split"
        return out_manifest, None, prep

    # YouHome ADL AV outputs are application-level caps, but the downstream
    # classifier consumes the usual YouHome manifest with both visual and audio
    # paths.  Generic media materialization previously returned data_root=None,
    # which made the preserved relative audio_path values unresolvable and caused
    # every preprocessed ADL row to report missing_audio_samples == n.
    #
    # We cannot assume the generated pipelines can be regenerated, so adapt the
    # existing spec at evaluation time: materialize whatever visual/audio paths
    # the runtime can produce, preserve companion modality columns from the
    # source manifest, and keep the original data_root so untouched relative
    # paths remain resolvable by YouHomeAVDataset.
    if row.task == "adl_recognition" and is_youhome_av_cap(final_cap):
        out_manifest = work_dir / "preprocessed_manifest.csv"
        transformed = transform_manifest_with_pipeline(
            manifest_for_eval,
            data_root,
            Path(row.pipeline_spec_json),
            out_manifest,
            work_dir / "preprocessed_youhome_av",
            task=row.task,
            split=None,
            max_samples=None,
            max_frames_per_sample=args.max_frames_per_sample,
            dry_run=args.dry_run,
            debug_dir=debug_dir,
            debug_context={**debug_context, "preprocess_branch": "youhome_av_preprocessed"},
        )
        prep.update(transformed)
        prep["preprocessing_status"] = "youhome_av_preprocessed_test_split"
        prep["youhome_av_adapter"] = {
            "policy": "materialize selected pipeline while preserving source companion modality paths",
            "returned_data_root": str(data_root) if data_root else None,
            "reason": "preserved relative audio_path/frame_dir/video_path entries need the original YouHome data root",
        }
        return out_manifest, data_root, prep

    # Visitor / ADL media outputs: materialize transformed manifest and use that.
    if is_image_cap(final_cap) or is_video_cap(final_cap) or is_audio_cap(final_cap):
        out_manifest = work_dir / "preprocessed_manifest.csv"
        transformed = transform_manifest_with_pipeline(
            manifest_for_eval,
            data_root,
            Path(row.pipeline_spec_json),
            out_manifest,
            work_dir / "preprocessed_data",
            task=row.task,
            split=None,
            max_samples=None,
            max_frames_per_sample=args.max_frames_per_sample,
            dry_run=args.dry_run,
            debug_dir=debug_dir,
            debug_context={**debug_context, "preprocess_branch": "media_preprocessed"},
        )
        prep.update(transformed)
        prep["preprocessing_status"] = "media_preprocessed_test_split"
        # For single-modality transformed outputs we use absolute materialized
        # paths and do not need the original data root.  ADL AV is handled above.
        return out_manifest, None, prep

    # Flexible semantic outputs are valid app-facing interfaces for flexible
    # requests. Materialize JSON primitives and let run_downstream_for_method()
    # evaluate them with task-specific semantic adapters instead of invoking a
    # legacy media-consuming downstream CLI.
    if task_supports_semantic_adapter(row.task, final_cap):
        out_manifest = work_dir / "preprocessed_semantic_manifest.csv"
        transformed = transform_manifest_with_pipeline(
            manifest_for_eval,
            data_root,
            Path(row.pipeline_spec_json),
            out_manifest,
            work_dir / "preprocessed_semantic",
            task=row.task,
            split=None,
            max_samples=None,
            max_frames_per_sample=args.max_frames_per_sample,
            dry_run=args.dry_run,
            debug_dir=debug_dir,
            debug_context={**debug_context, "preprocess_branch": "semantic_preprocessed"},
        )
        prep.update(transformed)
        prep["preprocessing_status"] = "semantic_preprocessed_for_flexible_adapter_test_split"
        prep["semantic_adapter"] = {
            "enabled": True,
            "final_output_type": cap_type(final_cap),
            "final_output_schema": cap_schema(final_cap),
            "note": "Evaluated through a task-specific semantic adapter rather than a legacy media-input downstream CLI.",
        }
        return out_manifest, None, prep

    # Semantic outputs are not directly consumable by the attached downstream
    # models unless a task-specific adapter exists.
    prep["preprocessing_status"] = "incompatible_semantic_output_for_downstream_cli"
    raise ValueError(f"Final output {cap_type(final_cap)} / {cap_schema(final_cap)} is not supported by the downstream utility evaluator for task {row.task}.")



def _json_paths_for_semantic_row(row: Dict[str, Any]) -> List[Path]:
    paths: List[Path] = []
    for key in ["preprocessed_json_path", "semantic_json_path"]:
        val = row.get(key)
        if val and Path(str(val)).exists():
            paths.append(Path(str(val)))
    for key in ["preprocessed_json_dir", "semantic_dir", "frame_dir"]:
        val = row.get(key)
        if val and Path(str(val)).exists() and Path(str(val)).is_dir():
            paths.extend(sorted(Path(str(val)).glob("*.json")))
    return paths


def _load_semantic_json(path: Path) -> Dict[str, Any]:
    try:
        obj = load_json(path)
    except Exception:
        return {"_error": f"could_not_read:{path}"}
    if isinstance(obj, dict):
        return obj
    return {"data": obj}


def _semantic_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    data = obj.get("data") if isinstance(obj.get("data"), dict) else None
    if data is not None:
        return data
    for key in ["payload", "value", "result"]:
        if isinstance(obj.get(key), dict):
            return obj[key]
    return obj


def _semantic_labels_and_count(obj: Dict[str, Any]) -> Tuple[List[str], int, bool, float]:
    data = _semantic_payload(obj)
    labels: List[str] = []
    confidence = 0.0
    count = 0
    occupied = False
    if isinstance(data.get("labels"), list):
        for l in data.get("labels") or []:
            if isinstance(l, dict):
                label = str(l.get("label") or l.get("class") or l.get("event_type") or l.get("activity") or "").strip()
                if label:
                    labels.append(label)
                try:
                    confidence = max(confidence, float(l.get("confidence", 0.0)))
                except Exception:
                    pass
            elif l:
                labels.append(str(l))
    if isinstance(data.get("detections"), list):
        detections = data.get("detections") or []
        for d in detections:
            if isinstance(d, dict):
                label = str(d.get("label") or d.get("class") or "").strip()
                if label:
                    labels.append(label)
                try:
                    confidence = max(confidence, float(d.get("confidence", 0.0)))
                except Exception:
                    pass
        count = sum(1 for d in detections if isinstance(d, dict) and str(d.get("label") or d.get("class") or "").lower() in {"person", "face", "human", "visitor"})
    cbl = data.get("count_by_label")
    if isinstance(cbl, dict):
        for k, v in cbl.items():
            if k:
                try:
                    n = int(float(v or 0))
                except Exception:
                    n = 1
                labels.extend([str(k)] * max(1, n))
        for k in ["person", "face", "human", "visitor"]:
            if k in cbl:
                try:
                    count += int(float(cbl.get(k) or 0))
                except Exception:
                    pass
    for key in ["top_label", "label", "event_type", "activity", "class", "pred_label"]:
        if data.get(key):
            labels.append(str(data.get(key)))
    for key in ["count", "occupancy_count", "person_count"]:
        if data.get(key) not in (None, ""):
            try:
                count = max(count, int(float(data.get(key))))
            except Exception:
                pass
    if data.get("occupied") not in (None, ""):
        occupied = bool(data.get("occupied"))
        count = max(count, int(occupied))
    if not occupied:
        occupied = count > 0 or any(str(l).lower() in {"person", "face", "human", "visitor", "room_occupied", "occupied", "presence"} for l in labels)
    return labels, count, occupied, confidence


def _manifest_truth_label(row: Dict[str, Any], task: str) -> str:
    candidates = [
        "label", "labels", "class", "target", "activity", "activity_label", "adl_label",
        "event", "event_type", "sound_label", "y_true", "gt", "ground_truth", "fall_label",
    ]
    for c in candidates:
        val = row.get(c)
        if val not in (None, ""):
            return str(val)
    if task == "fall_detection":
        text = " ".join(str(v) for v in row.values()).lower()
        if "fall" in text:
            return "fall"
        if "non" in text or "normal" in text:
            return "nonfall"
    return ""


def _label_to_binary(label: str, positive_terms: Sequence[str]) -> int:
    text = str(label or "").strip().lower()
    if text in {"1", "true", "yes", "positive", "pos"}:
        return 1
    if text in {"0", "false", "no", "negative", "neg", "none", "normal", "nonfall", "non_fall"}:
        return 0
    return 1 if any(term in text for term in positive_terms) else 0


def multiclass_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    accuracy = sum(1 for t, p_ in zip(y_true, y_pred) if t == p_) / n if n else 0.0
    f1s: List[float] = []
    for lab in labels:
        tp = sum(1 for t, p_ in zip(y_true, y_pred) if t == lab and p_ == lab)
        fp = sum(1 for t, p_ in zip(y_true, y_pred) if t != lab and p_ == lab)
        fn = sum(1 for t, p_ in zip(y_true, y_pred) if t == lab and p_ != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return {
        "status": "ok",
        "n": n,
        "accuracy": accuracy,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "labels": labels,
    }




CHIME_LABELS = ["c", "m", "f", "v", "p", "b", "o"]
CHIME_LABEL_SYNONYMS = {
    "c": {"c", "child", "child_speech", "child speech", "speech_child"},
    "m": {"m", "male", "adult_male", "adult male", "adult_male_speech", "male_speech", "speech_male"},
    "f": {"f", "female", "adult_female", "adult female", "adult_female_speech", "female_speech", "speech_female"},
    "v": {"v", "tv", "television", "video", "video_game", "video game", "game", "games"},
    "p": {"p", "percussive", "knock", "knocks", "bang", "bangs", "crash", "crashes", "footstep", "footsteps"},
    "b": {"b", "broadband", "noise", "appliance", "appliances", "vacuum", "fan"},
    "o": {"o", "other", "other_sound", "identifiable_sound", "unknown_sound"},
}


def normalize_label_text(label: Any) -> str:
    text = str(label or "").strip().lower()
    text = re.sub(r"[^a-z0-9_.]+", "_", text)
    return text.strip("_")


def parse_chime_label_string(label_string: Any) -> List[int]:
    if label_string is None:
        s = ""
    else:
        s = str(label_string).strip().lower()
        if s in {"nan", "none", "null", "-", ""}:
            s = ""
    active = set(s)
    return [1 if lab in active else 0 for lab in CHIME_LABELS]


def chime_truth_vector(row: Dict[str, Any]) -> List[int]:
    vals: List[int] = []
    have_cols = True
    for lab in CHIME_LABELS:
        key = f"label_{lab}"
        if key not in row or row.get(key) in (None, ""):
            have_cols = False
            break
        try:
            vals.append(1 if int(float(str(row.get(key)))) > 0 else 0)
        except Exception:
            vals.append(0)
    if have_cols:
        return vals
    for key in ["majorityvote", "annotation_a1", "annotation_a2", "annotation_a3", "label", "sound_label"]:
        if row.get(key) not in (None, ""):
            return parse_chime_label_string(row.get(key))
    return [0] * len(CHIME_LABELS)


def chime_pred_vector_from_labels(labels: Sequence[str]) -> List[int]:
    active = set()
    normalized = {normalize_label_text(x) for x in labels if str(x or "").strip()}
    compact = set("".join(x for x in normalized if len(x) <= 7))
    for lab in CHIME_LABELS:
        syns = {normalize_label_text(x) for x in CHIME_LABEL_SYNONYMS[lab]}
        if lab in compact or normalized & syns:
            active.add(lab)
    # If the semantic output includes generic speech labels, map cautiously to c/m/f only
    # when no specific speech subtype is present; this is a coarse proxy, not a claim
    # that the hub recovered speaker demographics.
    if not (active & {"c", "m", "f"}) and any("speech" in x or "voice" in x for x in normalized):
        active.update(["c", "m", "f"])
    return [1 if lab in active else 0 for lab in CHIME_LABELS]


def multilabel_binary_metrics(y_true: List[List[int]], y_pred: List[List[int]], labels: Sequence[str]) -> Dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {"status": "skipped", "error": "no multilabel rows", "n": 0}
    L = len(labels)
    tp = fp = tn = fn = 0
    per_label: Dict[str, Any] = {}
    f1s: List[float] = []
    for j, lab in enumerate(labels):
        lt = [int(row[j]) for row in y_true]
        lp = [int(row[j]) for row in y_pred]
        ltp = sum(1 for t, p_ in zip(lt, lp) if t == 1 and p_ == 1)
        ltn = sum(1 for t, p_ in zip(lt, lp) if t == 0 and p_ == 0)
        lfp = sum(1 for t, p_ in zip(lt, lp) if t == 0 and p_ == 1)
        lfn = sum(1 for t, p_ in zip(lt, lp) if t == 1 and p_ == 0)
        tp += ltp; tn += ltn; fp += lfp; fn += lfn
        prec = ltp / (ltp + lfp) if (ltp + lfp) else 0.0
        rec = ltp / (ltp + lfn) if (ltp + lfn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        per_label[lab] = {"tp": ltp, "fp": lfp, "tn": ltn, "fn": lfn, "precision": prec, "recall": rec, "f1": f1}
    micro_precision = tp / (tp + fp) if (tp + fp) else 0.0
    micro_recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0
    exact_match = sum(1 for t, p_ in zip(y_true, y_pred) if list(map(int, t)) == list(map(int, p_))) / n
    hamming_accuracy = (tp + tn) / (n * L) if L else 0.0
    return {
        "status": "ok",
        "n": n,
        "num_labels": L,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "hamming_accuracy": hamming_accuracy,
        "exact_match": exact_match,
        "per_label": per_label,
    }


def label_map_from_manifest_rows(rows: Sequence[Dict[str, Any]], task: str) -> Dict[str, int]:
    labels: List[str] = []
    if task == "fall_detection":
        # The LE2I manifest normally has a binary label column. Prefer the common
        # training convention when both labels are present; otherwise include only
        # labels observed in the filtered manifest.
        observed = {str(r.get("label") or "").strip() for r in rows if str(r.get("label") or "").strip()}
        if observed <= {"fall", "nonfall"}:
            labels = [x for x in ["fall", "nonfall"] if x in observed or len(observed) == 2]
        else:
            labels = sorted(observed)
    elif task == "adl_recognition":
        for r in rows:
            lab = str(r.get("label") or r.get("activity") or "").strip()
            if lab and lab not in labels:
                labels.append(lab)
        labels = sorted(labels)
    else:
        labels = sorted({str(r.get("label") or "").strip() for r in rows if str(r.get("label") or "").strip()})
    if not labels:
        labels = ["unknown"]
    return {lab: i for i, lab in enumerate(labels)}


def ensure_task_label_map(args: argparse.Namespace, task: str, manifest: Path, out_dir: Path) -> Path:
    if task == "fall_detection":
        existing = getattr(args, "fall_label_map", None)
        if existing and Path(str(existing)).exists():
            return Path(str(existing))
    elif task == "adl_recognition":
        existing = getattr(args, "youhome_label_map", None)
        if existing and Path(str(existing)).exists():
            return Path(str(existing))
    else:
        raise ValueError(f"No task label-map handling for {task}")
    rows, _ = read_csv_rows(manifest)
    label_map = label_map_from_manifest_rows(rows, task)
    out = out_dir / ("auto_label_map.json" if task != "fall_detection" else "auto_fall_label_map.json")
    write_json(label_map, out)
    progress_write(f"[config] generated temporary {task} label map at {out}: {label_map}")
    return out

def infer_semantic_for_method(args: argparse.Namespace, row: MethodRow, manifest: Path, data_root: Optional[str | Path], work_dir: Path) -> Dict[str, Any]:
    """Run a task-specific flexible semantic adapter.

    Semantic adapters live outside this evaluator so the old downstream inference
    CLIs remain fixed-interface black boxes.  The adapter consumes semantic JSON
    artifacts already emitted by the SmartPriv runtime and either computes
    metrics directly or emits a task prediction CSV for metric-only evaluation.
    It does not call infer_yolo, infer_chime, infer_fall, or infer_youhome.
    """
    try:
        from evaluation.semantic_adapters import run_semantic_adapter
    except Exception:
        # Allows running this file directly as a script from evaluation/.
        from semantic_adapters import run_semantic_adapter  # type: ignore
    return run_semantic_adapter(args=args, row=row, manifest=manifest, data_root=data_root, work_dir=work_dir)


def run_downstream_for_method(args: argparse.Namespace, row: MethodRow, manifest: Path, data_root: Optional[str | Path], work_dir: Path) -> Dict[str, Any]:
    final_cap = {
        "semantic_type": row.final_output_type if str(row.final_output_type).startswith("application/") else None,
        "media_type": row.final_output_type if not str(row.final_output_type).startswith("application/") else None,
        "schema": row.final_output_schema,
    }
    if task_supports_semantic_adapter(row.task, final_cap):
        return infer_semantic_for_method(args, row, manifest, data_root, work_dir)
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
        progress_write(f"[method] {row.scenario_id}/{row.method_id}: prepare/preprocess start task={row.task}; evaluator device request={args.device}; {compact_device_report(cuda_device_report())}")
        manifest, data_root, prep = prepare_manifest_for_method(args, row, intermediate_dir, log_dir=work_dir)
        if not getattr(args, "dry_run", False) and not Path(manifest).exists():
            raise PreprocessingStageError(f"Prepared manifest does not exist before downstream inference: {manifest}", prep)
        progress_write(f"[method] {row.scenario_id}/{row.method_id}: downstream start manifest={manifest}")
        prep["intermediate_artifact_dir"] = str(intermediate_dir)
        prep["intermediate_artifacts_retained"] = bool(args.keep_intermediate_data)
        prep["intermediate_artifact_policy"] = (
            "retained_under_utility_work_dir"
            if args.keep_intermediate_data
            else "temporary_deleted_after_downstream_inference"
        )

        downstream = run_downstream_for_method(args, row, manifest, data_root, work_dir)
        status = "ok" if downstream.get("status") == "ok" else "error"
        if status == "ok":
            metrics = compute_downstream_metrics(args, row, prep, downstream, work_dir)
            progress_write(f"[metrics] {row.scenario_id}/{row.method_id}: {format_metric_line(metrics)}")
        progress_write(f"[method] {row.scenario_id}/{row.method_id}: downstream done status={status}")
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
    except PreprocessingStageError as exc:
        prep = dict(getattr(exc, "prep", {}) or {})
        prep.setdefault("intermediate_artifact_dir", str(intermediate_dir))
        prep.setdefault("intermediate_artifacts_retained", bool(args.keep_intermediate_data))
        prep.setdefault(
            "intermediate_artifact_policy",
            "retained_under_utility_work_dir" if args.keep_intermediate_data else "temporary_deleted_after_downstream_inference",
        )
        result = {
            **base,
            "status": "error",
            "elapsed_ms": now_ms() - started,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "intermediate_artifacts_retained": bool(args.keep_intermediate_data),
            "intermediate_artifact_policy": prep["intermediate_artifact_policy"],
            "preprocessing": prep,
            "downstream": {"status": "skipped", "reason": "preprocessing_failed_before_downstream"},
        }
        write_text(result["traceback"], work_dir / "traceback.txt")
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


def method_work_dir(args: argparse.Namespace, row: MethodRow) -> Path:
    method_slug = row.method_id.replace(":", "__").replace("/", "_")
    return Path(args.out_dir) / row.scenario_id / method_slug


def downstream_output_exists(result: Dict[str, Any]) -> bool:
    downstream = result.get("downstream") or {}
    output_csv = downstream.get("output_csv")
    if output_csv:
        return Path(str(output_csv)).exists()
    # Some error/no-output cases have no CSV.  Treat them as incomplete for
    # resume purposes so a later run can retry after configs are fixed.
    return False


def result_has_ok_metrics(result: Dict[str, Any]) -> bool:
    downstream = result.get("downstream") or {}
    metrics = downstream.get("metrics") if isinstance(downstream.get("metrics"), dict) else {}
    if metrics.get("status") == "ok":
        return True
    if result.get("metric_status") == "ok":
        return True
    metrics_json = downstream.get("metrics_json")
    if metrics_json and Path(str(metrics_json)).exists():
        try:
            loaded = load_json(metrics_json)
            return isinstance(loaded, dict) and loaded.get("status", "ok") == "ok"
        except Exception:
            return False
    return False


def load_existing_completed_result(args: argparse.Namespace, row: MethodRow, cache_key: str, cache_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load a previous ok utility_result.json instead of rerunning it.

    The default evaluator behavior is resume-friendly: if the per-method result
    exists, downstream completed successfully, and the prediction CSV is still on
    disk, we reuse it.  Metrics are backfilled or recomputed when missing, so
    older visitor-only runs can be upgraded without rerunning YOLO.
    """
    if getattr(args, "rerun_existing", False):
        return None
    work_dir = method_work_dir(args, row)
    result_path = work_dir / "utility_result.json"
    if not result_path.exists():
        return None
    try:
        result = load_json(result_path)
    except Exception as exc:
        progress_write(f"[resume] ignoring unreadable existing result {result_path}: {exc!r}")
        return None

    if result.get("status") != "ok" or (result.get("downstream") or {}).get("status") != "ok":
        progress_write(f"[resume] existing result is not ok; rerunning {row.scenario_id}/{row.method_id}")
        return None
    if not downstream_output_exists(result):
        progress_write(f"[resume] existing result has no prediction CSV on disk; rerunning {row.scenario_id}/{row.method_id}")
        return None

    # Make sure the resumed row identity matches the requested row.  This avoids
    # accidentally reusing a stale file after renaming methods or changing out_dir.
    if str(result.get("scenario_id")) != str(row.scenario_id) or str(result.get("method_id")) != str(row.method_id):
        progress_write(f"[resume] existing result identity mismatch in {result_path}; rerunning")
        return None

    if existing_result_has_known_adl_missing_audio_bug(result, row):
        progress_write(
            f"[resume] existing ADL AV result has missing_audio_samples == n; "
            f"rerunning {row.scenario_id}/{row.method_id} with fixed YouHome AV manifest adapter"
        )
        return None

    downstream = result.get("downstream") or {}
    metrics_missing = not result_has_ok_metrics(result)
    if metrics_missing or getattr(args, "recompute_existing_metrics", False):
        progress_write(f"[resume] backfilling metrics for existing result {row.scenario_id}/{row.method_id}")
        prep = result.get("preprocessing") if isinstance(result.get("preprocessing"), dict) else {}
        try:
            compute_downstream_metrics(args, row, prep, downstream, work_dir)
            # Remove stale flattened metric fields before re-flattening.
            for k in list(result.keys()):
                if str(k).startswith("metric_"):
                    del result[k]
            result["downstream"] = downstream
            result.update(flatten_metrics(downstream.get("metrics", {})))
        except Exception as exc:
            progress_write(f"[resume] metric backfill failed for {row.scenario_id}/{row.method_id}: {exc!r}")

    result["resume_status"] = "reused_existing_result"
    result["cache_status"] = "existing_result"
    result["cache_key"] = cache_key
    result["pipeline_fingerprint"] = cache_info.get("pipeline_fingerprint")
    result.setdefault("utility_work_dir", str(work_dir))
    write_json(result, result_path)
    return result


def evaluate_one(args: argparse.Namespace, row: MethodRow) -> Dict[str, Any]:
    work_dir = method_work_dir(args, row)
    method_slug = row.method_id.replace(":", "__").replace("/", "_")
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
    p.add_argument("--request-mode", choices=["legacy", "flexible"], default="legacy", help="Default path mode. flexible uses runs/flexible_context_pipeline_generation and runs/utility_eval_flexible unless explicit paths are supplied.")
    p.add_argument("--pipeline-root", default=None)
    p.add_argument(
        "--pipeline-discovery",
        choices=["merge", "summary", "scan"],
        default="merge",
        help=(
            "How to discover selected pipeline rows. Default merge combines summary.json, "
            "summary_by_context.json, and filesystem scanning so raw/manual baselines are not "
            "lost when a summary file is stale or partial. Use summary for old behavior or scan "
            "to ignore summaries and inspect S*/baselines and S*/ablations directories directly."
        ),
    )
    p.add_argument("--out-dir", default=None)
    p.add_argument("--project-root", default=".")
    p.add_argument("--runtime-package", default="auto", help="Runtime package for executable pipelines: auto, smartpriv_runtime, or mediator.smartpriv_runtime. Default auto tries both and also adds project_root/mediator to PYTHONPATH.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--tasks", default="auto", help="Comma list of task ids, or auto/all. Default auto evaluates all configured tasks present in the pipeline summary.")
    p.add_argument("--scenario-ids", default="", help="Comma list such as S001,S002.")
    p.add_argument("--methods", default="", help="Comma list of method ids/baselines, e.g. raw,manual,full_mediator,ablation:utility_only. If provided, this exact method filter overrides the default ablation policy.")
    p.add_argument(
        "--ablation-policy",
        choices=["meaningful", "all", "none"],
        default="meaningful",
        help=(
            "Which ablations to include when --methods is not specified. "
            "Default meaningful runs only the ablations that changed decisions/operator chains in the pipeline-generation summary: "
            + ",".join(DEFAULT_MEANINGFUL_ABLATION_MODES) + ". "
            "Use none for baselines only, or all for every discovered ablation."
        ),
    )
    p.add_argument(
        "--ablation-modes",
        default=",".join(DEFAULT_MEANINGFUL_ABLATION_MODES),
        help=(
            "Comma list of ablation modes used by --ablation-policy meaningful. "
            "Default: " + ",".join(DEFAULT_MEANINGFUL_ABLATION_MODES) + ". "
            "This is ignored when --methods is provided."
        ),
    )
    p.add_argument("--include-ablations", action="store_true", help="Backward-compatible alias for --ablation-policy all when --methods is not specified.")
    p.add_argument("--no-ablations", action="store_true", help="Evaluate baselines only. Alias for --ablation-policy none when --methods is not specified.")
    p.add_argument(
        "--keep-ablation-matches-full",
        action="store_true",
        help=(
            "Do not skip ablation rows whose selected decision/operator/output signature matches full_mediator "
            "for the same scenario. By default these duplicate ablation rows are skipped unless --methods is explicit."
        ),
    )
    p.add_argument("--split", default="test", help="Dataset split to evaluate. Default: test. This is applied before preprocessing so train/val rows are not processed.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-frames-per-sample", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--no-preprocess-debug",
        action="store_true",
        help=(
            "Disable durable preprocessing breadcrumbs. By default each method writes "
            "preprocess_inputs.jsonl, preprocess_debug.jsonl, and last_preprocess_attempt.json "
            "under its utility output directory before runtime/operator calls. This helps "
            "identify the exact video/audio/frame path involved if native decoders abort."
        ),
    )
    p.add_argument(
        "--no-task-pipeline-cache",
        action="store_true",
        help=(
            "Disable the in-process cache that reuses results for duplicate task+pipeline pairs. "
            "By default, if two selected rows have the same task and executable pipeline fingerprint, "
            "only the first row is preprocessed/inferred and later rows are recorded as cache hits."
        ),
    )
    p.add_argument("--rerun-existing", action="store_true", help="Do not resume/reuse existing per-method utility_result.json files; rerun preprocessing and downstream inference.")
    p.add_argument(
        "--reuse-from-dir",
        action="append",
        default=None,
        help=(
            "Optional previous utility-evaluation directory to reuse from when the same task+pipeline fingerprint already has an ok result. "
            "May be repeated or comma-separated. In flexible mode the default is runs/utility_eval when it exists, plus runs/flexible_utility_eval when it exists."
        ),
    )
    p.add_argument("--no-cross-run-reuse", action="store_true", help="Disable reuse of matching utility_result.json files from previous utility-evaluation directories.")
    p.add_argument("--analysis-only", action="store_true", help="Write/print the preflight plan and exit without evaluating any rows.")
    p.add_argument(
        "--summarize-existing-only",
        action="store_true",
        help=(
            "Do not run preprocessing or downstream inference. Scan existing utility_result.json files "
            "and rebuild utility_results.json, utility_summary.csv, and utility_metrics_summary.csv under --out-dir."
        ),
    )
    p.add_argument(
        "--summary-scan-dir",
        action="append",
        default=None,
        help=(
            "Directory to scan for existing utility_result.json files when --summarize-existing-only is used. "
            "May be repeated or comma-separated. Default: --out-dir."
        ),
    )
    p.add_argument(
        "--wide-metrics-summary",
        action="store_true",
        help=(
            "When rebuilding summaries, also write utility_metrics_summary_wide.csv containing every metric_* column, "
            "including per-class metrics. By default utility_metrics_summary.csv stays compact and publication-friendly."
        ),
    )
    p.add_argument("--recompute-existing-metrics", action="store_true", help="When resuming existing results, recompute metrics from prediction CSVs even if metrics already exist.")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--keep-intermediate-data", action="store_true", help="Retain transformed media/keypoint/audio artifacts. Default is to use a temporary directory and delete them after each downstream run.")
    p.add_argument("--intermediate-root", default=None, help="Optional parent directory for temporary intermediate artifacts. Useful if /tmp is small; artifacts are still deleted unless --keep-intermediate-data is set.")
    p.add_argument("--no-auto-task-config", action="store_true", help="Disable conventional default module/path discovery.")
    p.add_argument("--strict-task-config", action="store_true", help="Error if any requested or discovered task is missing manifest/checkpoint/inference configuration. By default auto mode skips unconfigured tasks.")
    p.add_argument("--yes", "-y", action="store_true", help="Skip the interactive preflight confirmation prompt. Useful for scripts/cluster jobs.")
    p.add_argument("--no-preflight-confirm", action="store_true", help="Print the task-configuration preflight report but do not wait for Enter.")
    p.add_argument("--no-preflight-report", action="store_true", help="Do not print the task-configuration preflight report.")

    # General inference resource knobs.
    p.add_argument(
        "--device",
        default="auto",
        help=(
            "Device passed to downstream inference scripts. Default: auto, which prefers GPU when CUDA is available. "
            "Use cpu to force CPU, 0 for Ultralytics GPU 0, cuda:0 for Torch GPU 0, or gpu/cuda as aliases."
        ),
    )
    p.add_argument("--prefer-gpu-name", default="RTX 2070", help="Preferred physical GPU name substring. By default, the evaluator sets CUDA_VISIBLE_DEVICES to this GPU's UUID so mixed-GPU machines expose it as cuda:0.")
    p.add_argument("--cuda-visible-devices", default="", help="Explicit CUDA_VISIBLE_DEVICES value to pass to evaluator subprocesses, preferably a GPU UUID. Overrides --prefer-gpu-name.")
    p.add_argument("--no-prefer-gpu-env", action="store_true", help="Do not override CUDA_VISIBLE_DEVICES based on --prefer-gpu-name.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--safe-cuda", action="store_true")
    p.add_argument("--retry-cpu-on-error", action="store_true")

    # ChokePoint / visitor.
    p.add_argument("--chokepoint-manifest", default=None)
    p.add_argument("--chokepoint-data-root", default=None)
    p.add_argument("--chokepoint-infer-module", default=None, help="Default auto: first importable of baselines.task.chokepoint_presence.infer_yolo, infer_chokepoint, infer, or older aliases")
    p.add_argument("--chokepoint-infer-script", default=None)
    p.add_argument("--chokepoint-model", default="yolo11n.pt")
    p.add_argument("--chokepoint-stride", type=int, default=None)
    p.add_argument("--chokepoint-eval-cmd-template", default=None, help="Optional command with {predictions}, {manifest}, {data_root}, {out_json} placeholders.")

    # LE2I fall.
    p.add_argument("--fall-manifest", default=None)
    p.add_argument("--fall-precomputed-pose-manifest", default=None, help="Optional LE2I manifest with keypoints_path; used to avoid rerunning YOLO-pose for raw/no-transform fall baselines.")
    p.add_argument("--fall-data-root", default=None)
    p.add_argument("--fall-extract-pose-module", default=None, help="Default auto: baselines.task.le2i_fall.extract_pose")
    p.add_argument("--fall-extract-pose-script", default=None)
    p.add_argument("--fall-infer-module", default=None, help="Default auto: baselines.task.le2i_fall.infer")
    p.add_argument("--fall-infer-script", default=None)
    p.add_argument("--fall-checkpoint", default=None)
    p.add_argument("--fall-label-map", default=None)
    p.add_argument("--fall-pose-model", default="yolo11n-pose.pt")
    p.add_argument("--fall-pose-stride", type=int, default=3)
    p.add_argument("--fall-sample-mode", choices=["window", "video"], default=None)
    p.add_argument(
        "--no-sanitize-videos",
        action="store_true",
        help=(
            "Disable evaluator-side LE2I video sanitization. By default fall pose extraction "
            "rewrites video rows to ffmpeg-extracted video-only frame_dir rows before invoking "
            "the pose extractor, avoiding OpenCV/native crashes on malformed AVI audio streams."
        ),
    )

    # CHiME-Home audio.
    p.add_argument("--home-audio-manifest", default=None)
    p.add_argument("--home-audio-infer-module", default=None, help="Default auto: first importable of baselines.task.chime_home_audio.infer_home_audio, baselines.task.chim_home_audio.infer_home_audio, or older aliases")
    p.add_argument("--home-audio-infer-script", default=None)
    p.add_argument("--home-audio-checkpoint", default=None)
    p.add_argument("--home-audio-threshold", type=float, default=None)
    p.add_argument("--home-audio-backbone", choices=["logmel_cnn", "ast"], default=None)

    # YouHome ADL.
    p.add_argument("--youhome-manifest", default=None)
    p.add_argument("--youhome-data-root", default=None)
    p.add_argument("--youhome-infer-module", default=None, help="Default auto: baselines.task.youhome_adl.infer_youhome")
    p.add_argument("--youhome-infer-script", default=None)
    p.add_argument("--youhome-checkpoint", default=None)
    p.add_argument("--youhome-label-map", default=None)
    p.add_argument("--youhome-tta-runs", type=int, default=None)
    return p


def available_ablation_modes(rows: Iterable[MethodRow]) -> set[str]:
    return {str(r.ablation_mode) for r in rows if r.method_kind == "ablation" and r.ablation_mode}


def selected_default_ablation_modes(args: argparse.Namespace, rows: Iterable[MethodRow]) -> set[str]:
    """Return ablation modes selected by the default utility-evaluation policy.

    Explicit --methods is handled outside this helper and always wins.  The
    default policy intentionally evaluates only ablations that changed decisions
    or operator chains in the current context-pipeline generation results.
    """
    all_modes = available_ablation_modes(rows)
    if getattr(args, "no_ablations", False):
        return set()
    if getattr(args, "include_ablations", False):
        return set(all_modes)

    policy = str(getattr(args, "ablation_policy", "meaningful") or "meaningful").strip().lower()
    if policy == "none":
        return set()
    if policy == "all":
        return set(all_modes)

    requested = parse_csv_list(getattr(args, "ablation_modes", ""), DEFAULT_MEANINGFUL_ABLATION_MODES)
    if any(str(x).strip().lower() in {"all", "*"} for x in requested):
        return set(all_modes)
    return {m for m in requested if m in all_modes}


def row_decision_operator_signature(row: MethodRow) -> Tuple[str, str, str, str]:
    """Signature used to skip ablation rows that duplicate full_mediator.

    Utility evaluation is expensive, so for default ablation runs we only keep
    rows that change the selected decision/operator/output relative to the full
    mediator in the same scenario.  Pipeline ids alone are not enough because
    equivalent operator chains can be emitted under different ids.
    """
    return (
        str(row.decision or ""),
        str(row.raw.get("operators") or ""),
        str(row.final_output_type or ""),
        str(row.final_output_schema or ""),
    )


def ablation_differs_from_full(row: MethodRow, full_row: Optional[MethodRow]) -> bool:
    if full_row is None:
        return True
    return row_decision_operator_signature(row) != row_decision_operator_signature(full_row)


# ---------------------------------------------------------------------------
# Task/pipeline de-duplication helpers
# ---------------------------------------------------------------------------


_PIPELINE_CACHE_DROP_KEYS = {
    # Generated identifiers/provenance.  These can differ across contexts even
    # when the executable operator chain is identical.
    "pipeline_id",
    "candidate_id",
    "stage_id",
    "id",
    "uuid",
    "name",
    "display_name",
    "description",
    "created_at",
    "updated_at",
    "source",
    "source_path",
    "result_json",
    "selected_pipeline_json",
    "pipeline_spec_json",
    "score",
    "rank",
    "rationale",
    "reason",
}


def _canonicalize_pipeline_for_cache(obj: Any) -> Any:
    """Return a stable, mostly-functional view of a generated pipeline spec.

    The cache should treat two specs as the same when they execute the same
    operators with the same parameters, even if they were emitted for different
    contexts and therefore have different generated ids or prose metadata.
    """
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k in sorted(obj.keys(), key=str):
            ks = str(k)
            if ks in _PIPELINE_CACHE_DROP_KEYS or ks.startswith("_"):
                continue
            out[ks] = _canonicalize_pipeline_for_cache(obj[k])
        return out
    if isinstance(obj, list):
        return [_canonicalize_pipeline_for_cache(x) for x in obj]
    return obj


def pipeline_execution_fingerprint(row: MethodRow) -> Tuple[str, Dict[str, Any]]:
    """Return a short fingerprint for the executable part of one method row."""
    spec = load_spec(row.pipeline_spec_json)
    if spec:
        # Most runtime-relevant information lives in stages.  Keeping the final
        # output cap prevents unsafe reuse across pipelines with the same early
        # stages but different downstream interface.
        functional = {
            "stages": _canonicalize_pipeline_for_cache(spec.get("stages", [])),
            "final_output_cap": _canonicalize_pipeline_for_cache(
                spec.get("final_output_cap")
                or {
                    "semantic_type": row.final_output_type if row.final_output_type.startswith("application/") else None,
                    "media_type": row.final_output_type if not row.final_output_type.startswith("application/") else None,
                    "schema": row.final_output_schema,
                }
            ),
            "input_cap": _canonicalize_pipeline_for_cache(spec.get("input_cap") or spec.get("input_caps")),
        }
        fp = stable_hash(functional)
        return fp, {"source": "pipeline_spec", "pipeline_fingerprint": fp, "functional_signature": functional}

    fallback = {
        "source": "row_signature_fallback",
        "decision_operator_signature": row_decision_operator_signature(row),
        "baseline": row.baseline,
        "method_id": row.method_id,
        "final_output_type": row.final_output_type,
        "final_output_schema": row.final_output_schema,
    }
    fp = stable_hash(fallback)
    fallback["pipeline_fingerprint"] = fp
    return fp, fallback


def task_pipeline_cache_key(row: MethodRow) -> Tuple[str, Dict[str, Any]]:
    """Cache key used inside a single utility-evaluation invocation.

    Same task + same executable pipeline means the same task manifest, split,
    preprocessing result, downstream predictions, and utility metrics for this
    invocation.  Different tasks are not shared even if their operator chains
    look identical because the downstream interface/checkpoint is task-specific.
    """
    fp, info = pipeline_execution_fingerprint(row)
    key = f"task={row.task}|pipeline={fp}"
    info = dict(info)
    info.update({"task": row.task, "cache_key": key})
    return key, info


def task_pipeline_cache_plan(rows: Sequence[MethodRow]) -> Dict[str, Any]:
    buckets: Dict[str, List[MethodRow]] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in rows:
        key, info = task_pipeline_cache_key(row)
        if key not in buckets:
            buckets[key] = []
            meta[key] = info
            order.append(key)
        buckets[key].append(row)

    groups: List[Dict[str, Any]] = []
    for i, key in enumerate(order, 1):
        bucket = buckets[key]
        first = bucket[0]
        groups.append({
            "unique_run_index": i,
            "cache_key": key,
            "task": first.task,
            "pipeline_fingerprint": meta[key].get("pipeline_fingerprint"),
            "row_count": len(bucket),
            "first_row": {"scenario_id": first.scenario_id, "method_id": first.method_id},
            "rows": [{"scenario_id": r.scenario_id, "method_id": r.method_id} for r in bucket],
        })

    return {
        "selected_rows": len(rows),
        "unique_task_pipeline_runs": len(order),
        "duplicate_rows_reused_from_cache": max(0, len(rows) - len(order)),
        "groups": groups,
        "key_order": order,
    }


def _result_downstream_output_exists(result: Dict[str, Any]) -> bool:
    """Return True when a stored utility result still points at reusable outputs."""
    downstream = result.get("downstream") if isinstance(result.get("downstream"), dict) else {}
    for key in ["output_csv", "metrics_json"]:
        value = downstream.get(key)
        if value and Path(str(value)).exists():
            return True
    metrics = downstream.get("metrics") if isinstance(downstream.get("metrics"), dict) else {}
    if metrics and str(metrics.get("status", "ok")) == "ok":
        return True
    # Semantic adapters and older runs sometimes flatten metrics at top-level.
    if result.get("metric_status") == "ok":
        return True
    return False


def reusable_utility_result(result: Dict[str, Any]) -> bool:
    """Return True when a utility_result.json is safe to reuse across directories."""
    if result.get("status") != "ok":
        return False
    downstream = result.get("downstream") if isinstance(result.get("downstream"), dict) else {}
    if downstream and downstream.get("status") not in {None, "ok"}:
        return False
    return _result_downstream_output_exists(result)


def _fingerprint_from_result(result: Dict[str, Any]) -> Optional[str]:
    fp = result.get("pipeline_fingerprint")
    if fp:
        return str(fp)
    cache_key = str(result.get("cache_key") or "")
    m = re.search(r"pipeline=([^|]+)", cache_key)
    if m:
        return m.group(1)
    spec_path = result.get("pipeline_spec_json")
    if spec_path and Path(str(spec_path)).exists():
        try:
            spec = load_json(str(spec_path))
            final_output_type = str(result.get("final_output_type") or "")
            functional = {
                "stages": _canonicalize_pipeline_for_cache(spec.get("stages", [])),
                "final_output_cap": _canonicalize_pipeline_for_cache(
                    spec.get("final_output_cap")
                    or {
                        "semantic_type": final_output_type if final_output_type.startswith("application/") else None,
                        "media_type": final_output_type if not final_output_type.startswith("application/") else None,
                        "schema": result.get("final_output_schema"),
                    }
                ),
                "input_cap": _canonicalize_pipeline_for_cache(spec.get("input_cap") or spec.get("input_caps")),
            }
            return stable_hash(functional)
        except Exception:
            return None
    return None


def _cache_key_from_result(result: Dict[str, Any]) -> Optional[str]:
    key = result.get("cache_key")
    if key:
        return str(key)
    task = result.get("task")
    fp = _fingerprint_from_result(result)
    if task and fp:
        return f"task={task}|pipeline={fp}"
    return None


def _parse_reuse_from_dirs(args: argparse.Namespace) -> List[Path]:
    raw_items: List[str] = []
    for item in getattr(args, "reuse_from_dir", None) or []:
        raw_items.extend(parse_csv_list(str(item)))
    if not raw_items and str(getattr(args, "request_mode", "")) == "flexible":
        raw_items = ["runs/utility_eval", "runs/flexible_utility_eval"]
    out_dir = Path(str(args.out_dir)).resolve() if getattr(args, "out_dir", None) else None
    dirs: List[Path] = []
    seen: set[str] = set()
    for item in raw_items:
        p = Path(item)
        if not p.is_absolute():
            p = Path(args.project_root) / p
        try:
            resolved = p.resolve()
        except Exception:
            resolved = p
        if out_dir and resolved == out_dir:
            continue
        key = str(resolved)
        if key in seen or not resolved.exists():
            continue
        seen.add(key)
        dirs.append(resolved)
    return dirs


def build_cross_run_reuse_index(args: argparse.Namespace, selected: Sequence[MethodRow]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Index previous utility_eval directories by task+pipeline fingerprint."""
    selected_keys = {task_pipeline_cache_key(r)[0] for r in selected}
    if getattr(args, "rerun_existing", False) or getattr(args, "no_cross_run_reuse", False):
        return {}, {
            "enabled": False,
            "reason": "disabled by --rerun-existing or --no-cross-run-reuse",
            "reuse_from_dirs": [],
            "selected_unique_keys": len(selected_keys),
            "matched_unique_keys": 0,
            "matched_rows": 0,
            "index_size": 0,
        }

    reuse_dirs = _parse_reuse_from_dirs(args)
    index: Dict[str, Dict[str, Any]] = {}
    scanned = 0
    usable = 0
    for reuse_dir in reuse_dirs:
        for result_path in reuse_dir.glob("S*/**/utility_result.json"):
            scanned += 1
            try:
                result = load_json(result_path)
            except Exception:
                continue
            if not isinstance(result, dict) or not reusable_utility_result(result):
                continue
            key = _cache_key_from_result(result)
            if not key or key not in selected_keys:
                continue
            # Keep first match to make reuse deterministic by reuse_dir order.
            index.setdefault(key, result)
            usable += 1

    matched_rows = sum(1 for r in selected if task_pipeline_cache_key(r)[0] in index)
    return index, {
        "enabled": True,
        "reuse_from_dirs": [str(p) for p in reuse_dirs],
        "scanned_result_files": scanned,
        "usable_matching_result_files": usable,
        "selected_unique_keys": len(selected_keys),
        "matched_unique_keys": len(index),
        "matched_rows": matched_rows,
        "index_size": len(index),
        "policy": "Reuse ok utility_result.json files from previous directories when task+pipeline fingerprint matches. New per-row result files are still written under --out-dir and nested prediction/metric paths may point to the reused directory.",
    }


def current_out_dir_reuse_plan(args: argparse.Namespace, selected: Sequence[MethodRow]) -> Dict[str, Any]:
    """Count current --out-dir results that appear reusable before the run starts."""
    keys: set[str] = set()
    rows = 0
    for r in selected:
        cache_key, _ = task_pipeline_cache_key(r)
        result_path = method_work_dir(args, r) / "utility_result.json"
        if not result_path.exists():
            continue
        try:
            result = load_json(result_path)
        except Exception:
            continue
        if str(result.get("scenario_id")) != str(r.scenario_id) or str(result.get("method_id")) != str(r.method_id):
            continue
        if reusable_utility_result(result) and not existing_result_has_known_adl_missing_audio_bug(result, r):
            keys.add(cache_key)
            rows += 1
    return {
        "matched_unique_keys": len(keys),
        "matched_rows": rows,
        "keys": sorted(keys),
    }


def pending_execution_plan(
    selected: Sequence[MethodRow],
    current_reuse_plan: Dict[str, Any],
    cross_reuse_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize how many unique task+pipeline executions still require work."""
    selected_keys: Dict[str, Dict[str, Any]] = {}
    schemas: set[Tuple[str, str]] = set()
    types: set[str] = set()
    for r in selected:
        key, info = task_pipeline_cache_key(r)
        selected_keys.setdefault(key, info)
        schemas.add((str(r.final_output_type or ""), str(r.final_output_schema or "")))
        if r.final_output_type:
            types.add(str(r.final_output_type))

    current_keys = set(current_reuse_plan.get("keys") or [])
    cross_keys = set(cross_reuse_index.keys())
    reusable_keys = current_keys | cross_keys
    pending_keys = [k for k in selected_keys if k not in reusable_keys]
    pending_rows = sum(1 for r in selected if task_pipeline_cache_key(r)[0] in set(pending_keys))
    pending_schemas: set[Tuple[str, str]] = set()
    for r in selected:
        if task_pipeline_cache_key(r)[0] in set(pending_keys):
            pending_schemas.add((str(r.final_output_type or ""), str(r.final_output_schema or "")))

    return {
        "selected_rows": len(selected),
        "selected_unique_task_pipeline_runs": len(selected_keys),
        "selected_unique_output_schemas": len(schemas),
        "selected_unique_output_types": len(types),
        "already_ok_in_out_dir_unique_runs": len(current_keys),
        "cross_run_reusable_unique_runs": len(cross_keys - current_keys),
        "pending_unique_task_pipeline_runs": len(pending_keys),
        "pending_rows": pending_rows,
        "pending_unique_output_schemas": len(pending_schemas),
        "pending_cache_keys": pending_keys,
        "selected_output_schemas": sorted([{"final_output_type": t, "final_output_schema": s} for t, s in schemas], key=lambda x: (x["final_output_type"], x["final_output_schema"])),
        "pending_output_schemas": sorted([{"final_output_type": t, "final_output_schema": s} for t, s in pending_schemas], key=lambda x: (x["final_output_type"], x["final_output_schema"])),
    }


def clone_cross_run_result_for_row(
    args: argparse.Namespace,
    row: MethodRow,
    cached_result: Dict[str, Any],
    cache_key: str,
    cache_info: Dict[str, Any],
) -> Dict[str, Any]:
    result = clone_cached_result_for_row(args, row, cached_result, cache_key, cache_info)
    result["cache_status"] = "cross_run_hit"
    result["cross_run_reuse"] = True
    result["reused_from_external"] = {
        "scenario_id": cached_result.get("scenario_id"),
        "method_id": cached_result.get("method_id"),
        "utility_work_dir": cached_result.get("utility_work_dir"),
        "utility_result_json": str(Path(str(cached_result.get("utility_work_dir", ""))) / "utility_result.json")
        if cached_result.get("utility_work_dir")
        else None,
    }
    result["reused_outputs_note"] = (
        "This row was not rerun. It reuses a previous utility result with the same task+pipeline fingerprint; "
        "nested downstream/preprocessing paths may point to the previous utility-evaluation directory."
    )
    if result.get("utility_work_dir"):
        write_json(result, Path(str(result["utility_work_dir"])) / "utility_result.json")
    return result


def clone_cached_result_for_row(
    args: argparse.Namespace,
    row: MethodRow,
    cached_result: Dict[str, Any],
    cache_key: str,
    cache_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a per-row result record that points at an already-run equivalent."""
    method_slug = row.method_id.replace(":", "__").replace("/", "_")
    work_dir = Path(args.out_dir) / row.scenario_id / method_slug
    work_dir.mkdir(parents=True, exist_ok=True)

    original = {
        "scenario_id": cached_result.get("scenario_id"),
        "method_id": cached_result.get("method_id"),
        "utility_work_dir": cached_result.get("utility_work_dir"),
        "utility_result_json": str(Path(str(cached_result.get("utility_work_dir", ""))) / "utility_result.json")
        if cached_result.get("utility_work_dir")
        else None,
    }
    result = copy.deepcopy(cached_result)
    # Make the row identity match the requested scenario/method while keeping
    # nested paths to the reused predictions/logs/metrics intact.
    result.update({
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
        "elapsed_ms": 0,
        "cache_status": "hit",
        "cache_key": cache_key,
        "pipeline_fingerprint": cache_info.get("pipeline_fingerprint"),
        "reused_from": original,
        "reused_outputs_note": "Nested downstream/preprocessing paths point to the first equivalent task+pipeline run.",
    })
    write_json(result, work_dir / "utility_result.json")
    return result


def filter_rows(args: argparse.Namespace, rows: List[MethodRow]) -> List[MethodRow]:
    explicit_tasks, auto_tasks = requested_task_set(args)
    sids = set(parse_csv_list(args.scenario_ids))
    methods = set(parse_csv_list(args.methods))
    default_ablation_modes = selected_default_ablation_modes(args, rows) if not methods else set()
    full_by_scenario_task = {
        (r.scenario_id, r.task): r
        for r in rows
        if r.method_id == "full_mediator" or r.baseline == "full_mediator"
    }
    tasks_present = {r.task for r in rows if r.task in TASK_TO_SHORT}
    config = configured_tasks(args, tasks_present)
    configured = {t for t, info in config.items() if info.get("configured")}

    if args.strict_task_config:
        check_tasks = tasks_present if auto_tasks else explicit_tasks
        missing = {t: config.get(t, {"errors": ["task not present"]}).get("errors", []) for t in check_tasks if config.get(t, {}).get("configured") is not True}
        if missing:
            raise ValueError("Missing utility-evaluation configuration for task(s): " + json.dumps(missing, indent=2))

    out: List[MethodRow] = []
    for r in rows:
        if r.decision and r.decision not in {"select_pipeline", "baseline_raw_release", "raw_release", "select"}:
            # Keep raw rows even if the exact decision string varies; skip clear failures.
            if r.decision in {"error", "no_candidates", "no_compromise", "invalid_or_no_pipeline", "llm_error"}:
                continue
        if auto_tasks:
            if r.task not in configured:
                continue
        elif explicit_tasks and r.task not in explicit_tasks:
            continue
        if sids and r.scenario_id not in sids:
            continue
        if methods:
            method_matches = (
                r.method_id in methods
                or r.baseline in methods
                or bool(r.ablation_mode and f"ablation:{r.ablation_mode}" in methods)
            )
            if not method_matches:
                continue
        else:
            if r.method_kind == "ablation" and r.ablation_mode not in default_ablation_modes:
                continue
            skip_matching_ablation = (
                str(getattr(args, "ablation_policy", "meaningful") or "meaningful") == "meaningful"
                and not getattr(args, "include_ablations", False)
                and not getattr(args, "keep_ablation_matches_full", False)
            )
            if (
                r.method_kind == "ablation"
                and skip_matching_ablation
                and not ablation_differs_from_full(r, full_by_scenario_task.get((r.scenario_id, r.task)))
            ):
                continue
        if r.task not in TASK_TO_SHORT:
            continue
        out.append(r)
    return out


def _preflight_path_exists(args: argparse.Namespace, value: Optional[Any]) -> bool:
    if value is None or str(value).strip() == "":
        return False
    p = Path(str(value))
    if p.is_absolute():
        return p.exists()
    return p.exists() or (Path(args.project_root) / p).exists()


def _preflight_module_status(module: Optional[str], script: Optional[str | Path]) -> str:
    if script:
        return "script:ok" if Path(str(script)).exists() else "script:MISSING"
    if module:
        return "module:ok" if _module_is_importable(str(module)) else "module:NOT_IMPORTABLE"
    return "MISSING"


def _preflight_asset(args: argparse.Namespace, label: str, attr: str, required: bool = True) -> Dict[str, Any]:
    value = getattr(args, attr, None)
    exists = _preflight_path_exists(args, value)
    return {
        "label": label,
        "attr": attr,
        "value": str(value) if value not in (None, "") else "",
        "exists": bool(exists),
        "required": bool(required),
        "status": "ok" if exists else ("missing" if required else "optional_missing"),
    }


def _preflight_python_import_status(args: argparse.Namespace, module_name: str) -> str:
    """Check an optional runtime dependency in the same Python env used for downstream commands."""
    try:
        proc = subprocess.run(
            [str(args.python), "-c", f"import {module_name}"],
            cwd=str(args.project_root) if getattr(args, "project_root", None) else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            env=make_subprocess_env(getattr(args, "project_root", None)),
        )
        if proc.returncode == 0:
            return "python-import:ok"
        tail = (proc.stdout or "").strip().splitlines()[-1:]
        return "python-import:MISSING" + (f" ({tail[0][:160]})" if tail else "")
    except Exception as exc:
        return f"python-import:UNKNOWN ({exc!r})"


def build_preflight_report(args: argparse.Namespace, rows: Sequence[MethodRow], selected: Sequence[MethodRow], task_config: Dict[str, Dict[str, Any]], cache_plan: Dict[str, Any], cross_run_reuse: Optional[Dict[str, Any]] = None, pending_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    tasks_present = sorted({r.task for r in rows if r.task in TASK_TO_SHORT})
    selected_tasks = sorted({r.task for r in selected if r.task in TASK_TO_SHORT})
    selected_rows_by_task = {t: sum(1 for r in selected if r.task == t) for t in tasks_present}
    discovered_rows_by_task = {t: sum(1 for r in rows if r.task == t) for t in tasks_present}

    task_assets: Dict[str, List[Dict[str, Any]]] = {
        "visitor_presence_detection": [
            _preflight_asset(args, "ChokePoint manifest", "chokepoint_manifest", True),
            _preflight_asset(args, "ChokePoint data root", "chokepoint_data_root", False),
            _preflight_asset(args, "YOLO detector model", "chokepoint_model", False),
        ],
        "fall_detection": [
            _preflight_asset(args, "LE2I raw manifest", "fall_manifest", True),
            _preflight_asset(args, "LE2I precomputed pose manifest", "fall_precomputed_pose_manifest", False),
            _preflight_asset(args, "LE2I data root", "fall_data_root", False),
            _preflight_asset(args, "LE2I fall checkpoint", "fall_checkpoint", True),
            _preflight_asset(args, "LE2I label map", "fall_label_map", True),
            _preflight_asset(args, "YOLO pose model", "fall_pose_model", False),
        ],
        "domestic_sound_monitoring": [
            _preflight_asset(args, "CHiME-Home manifest", "home_audio_manifest", True),
            _preflight_asset(args, "CHiME-Home checkpoint", "home_audio_checkpoint", True),
        ],
        "adl_recognition": [
            _preflight_asset(args, "YouHome manifest", "youhome_manifest", True),
            _preflight_asset(args, "YouHome data root", "youhome_data_root", False),
            _preflight_asset(args, "YouHome checkpoint", "youhome_checkpoint", True),
            _preflight_asset(args, "YouHome label map", "youhome_label_map", True),
        ],
    }

    task_modules = {
        "visitor_presence_detection": {
            "infer": _preflight_module_status(getattr(args, "chokepoint_infer_module", None), getattr(args, "chokepoint_infer_script", None)),
        },
        "fall_detection": {
            "extract_pose": _preflight_module_status(getattr(args, "fall_extract_pose_module", None), getattr(args, "fall_extract_pose_script", None)),
            "infer": _preflight_module_status(getattr(args, "fall_infer_module", None), getattr(args, "fall_infer_script", None)),
        },
        "domestic_sound_monitoring": {
            "infer": _preflight_module_status(getattr(args, "home_audio_infer_module", None), getattr(args, "home_audio_infer_script", None)),
        },
        "adl_recognition": {
            "infer": _preflight_module_status(getattr(args, "youhome_infer_module", None), getattr(args, "youhome_infer_script", None)),
        },
    }

    task_dependencies = {
        "visitor_presence_detection": {"ultralytics": _preflight_python_import_status(args, "ultralytics")},
        "fall_detection": {"ultralytics": _preflight_python_import_status(args, "ultralytics"), "torch": _preflight_python_import_status(args, "torch")},
        "domestic_sound_monitoring": {"torchaudio": _preflight_python_import_status(args, "torchaudio"), "torch": _preflight_python_import_status(args, "torch")},
        "adl_recognition": {"torchaudio": _preflight_python_import_status(args, "torchaudio"), "torch": _preflight_python_import_status(args, "torch")},
    }

    report_tasks: Dict[str, Any] = {}
    for task in tasks_present:
        config_info = task_config.get(task, {"configured": False, "errors": ["not present in task_config"]})
        required_missing_assets = [a for a in task_assets.get(task, []) if a.get("required") and not a.get("exists")]
        module_bad = [name for name, status in task_modules.get(task, {}).items() if str(status).endswith("NOT_IMPORTABLE") or status == "MISSING"]
        deps = task_dependencies.get(task, {})
        dependency_bad = [name for name, status in deps.items() if "MISSING" in str(status)]
        fully_ready = bool(config_info.get("configured")) and not required_missing_assets and not module_bad and not dependency_bad
        report_tasks[task] = {
            "generated_rows": discovered_rows_by_task.get(task, 0),
            "selected_rows": selected_rows_by_task.get(task, 0),
            "selected_for_this_run": task in selected_tasks,
            "configured_by_evaluator": bool(config_info.get("configured")),
            "fully_ready": fully_ready,
            "config_errors": list(config_info.get("errors") or []),
            "modules": task_modules.get(task, {}),
            "dependencies": deps,
            "assets": task_assets.get(task, []),
        }

    return {
        "requested_tasks": args.tasks,
        "tasks_present_in_pipeline_summary": tasks_present,
        "selected_tasks": selected_tasks,
        "total_discovered_rows": len(rows),
        "discovered_rows_by_method": dict(sorted(Counter(r.method_id for r in rows).items())),
        "selected_rows": len(selected),
        "selected_rows_by_method": dict(sorted(Counter(r.method_id for r in selected).items())),
        "pipeline_discovery": getattr(args, "pipeline_discovery", "merge"),
        "unique_task_pipeline_runs": cache_plan.get("unique_task_pipeline_runs"),
        "duplicate_rows_reused_from_cache": cache_plan.get("duplicate_rows_reused_from_cache"),
        "cross_run_reuse": cross_run_reuse or {},
        "pending_execution_plan": pending_plan or {},
        "tasks": report_tasks,
        "all_present_tasks_fully_ready": all(report_tasks[t].get("fully_ready") for t in tasks_present),
    }


def print_preflight_report(report: Dict[str, Any]) -> None:
    print("\n=== Utility evaluation preflight ===", flush=True)
    print(f"Requested tasks: {report.get('requested_tasks')}", flush=True)
    print("Tasks present in pipeline summary: " + ", ".join(report.get("tasks_present_in_pipeline_summary") or []), flush=True)
    print("Selected tasks for this run: " + (", ".join(report.get("selected_tasks") or []) or "<none>"), flush=True)
    print(
        f"Rows: discovered={report.get('total_discovered_rows')} selected={report.get('selected_rows')} "
        f"unique_task_pipeline_runs={report.get('unique_task_pipeline_runs')} "
        f"cache_duplicates={report.get('duplicate_rows_reused_from_cache')}",
        flush=True,
    )
    pending = report.get("pending_execution_plan") or {}
    if pending:
        print(
            "Execution reuse analysis: "
            f"selected_unique_runs={pending.get('selected_unique_task_pipeline_runs')} "
            f"selected_output_schemas={pending.get('selected_unique_output_schemas')} "
            f"already_ok_in_out_dir={pending.get('already_ok_in_out_dir_unique_runs')} "
            f"cross_run_reusable={pending.get('cross_run_reusable_unique_runs')} "
            f"STILL_TO_RUN_unique={pending.get('pending_unique_task_pipeline_runs')} "
            f"STILL_TO_RUN_rows={pending.get('pending_rows')} "
            f"STILL_TO_RUN_output_schemas={pending.get('pending_unique_output_schemas')}",
            flush=True,
        )
    cross = report.get("cross_run_reuse") or {}
    if cross:
        print(
            "Cross-run reuse: "
            f"enabled={cross.get('enabled')} dirs={cross.get('reuse_from_dirs')} "
            f"scanned={cross.get('scanned_result_files')} "
            f"matched_unique_runs={cross.get('matched_unique_keys')} "
            f"matched_rows={cross.get('matched_rows')}",
            flush=True,
        )
    for task, info in (report.get("tasks") or {}).items():
        state = "READY" if info.get("fully_ready") else "NOT READY"
        selected = "selected" if info.get("selected_for_this_run") else "not selected/skipped"
        print(f"\n[{state}] {task} ({selected}) generated_rows={info.get('generated_rows')} selected_rows={info.get('selected_rows')}", flush=True)
        if info.get("config_errors"):
            for err in info.get("config_errors") or []:
                print(f"  config error: {err}", flush=True)
        modules = info.get("modules") or {}
        if modules:
            for name, status in modules.items():
                print(f"  module {name}: {status}", flush=True)
        dependencies = info.get("dependencies") or {}
        if dependencies:
            for name, status in dependencies.items():
                print(f"  dependency {name}: {status}", flush=True)
        for asset in info.get("assets") or []:
            req = "required" if asset.get("required") else "optional"
            exists = "ok" if asset.get("exists") else "missing"
            value = asset.get("value") or "<unset>"
            print(f"  {req:8s} {asset.get('label')}: {exists} :: {value}", flush=True)
    if not report.get("all_present_tasks_fully_ready"):
        print(
            "\nWARNING: At least one task present in the pipeline summary is not fully ready. "
            "In auto mode, unconfigured tasks are skipped unless --strict-task-config is set.",
            flush=True,
        )
    print("====================================\n", flush=True)


def maybe_confirm_preflight(args: argparse.Namespace, report: Dict[str, Any]) -> None:
    if getattr(args, "no_preflight_report", False):
        return
    print_preflight_report(report)
    if getattr(args, "yes", False) or getattr(args, "no_preflight_confirm", False):
        return
    try:
        input("Press Enter to continue with this utility-evaluation plan, or Ctrl-C to abort... ")
    except EOFError:
        raise SystemExit(
            "Preflight confirmation requires interactive stdin. Re-run with --yes or --no-preflight-confirm for noninteractive execution."
        )


# ---------------------------------------------------------------------------
# Existing-result summary rebuild
# ---------------------------------------------------------------------------


def _parse_repeated_csv_dirs(values: Optional[Sequence[str]], default: Sequence[str]) -> List[Path]:
    """Parse repeated/comma-separated directory arguments while preserving order."""
    raw: List[str] = []
    if values:
        for value in values:
            raw.extend(parse_csv_list(value))
    if not raw:
        raw = [str(x) for x in default]
    out: List[Path] = []
    seen: set[str] = set()
    for item in raw:
        p = Path(str(item)).expanduser()
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _infer_ids_from_utility_result_path(path: Path) -> Dict[str, str]:
    """Infer scenario/method identifiers from .../<scenario>/<method>/utility_result.json."""
    method_slug = path.parent.name
    scenario_id = path.parent.parent.name if path.parent.parent != path.parent else ""
    method_id = method_slug.replace("__", ":")
    return {"scenario_id": scenario_id, "method_id": method_id}


def flatten_utility_result_for_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    """Create one CSV-friendly row from a utility_result.json dictionary."""
    row: Dict[str, Any] = {}
    for k, v in res.items():
        if not isinstance(v, (dict, list)):
            row[k] = v

    prep = res.get("preprocessing") if isinstance(res.get("preprocessing"), dict) else {}
    down = res.get("downstream") if isinstance(res.get("downstream"), dict) else {}
    inf = down.get("inference") if isinstance(down.get("inference"), dict) else {}

    row.setdefault("preprocessing_status", prep.get("preprocessing_status") or prep.get("status"))
    row.setdefault("source_manifest", prep.get("source_manifest"))
    row.setdefault("prepared_manifest", res.get("prepared_manifest"))
    row.setdefault("prepared_manifest_retained", res.get("prepared_manifest_retained"))
    row.setdefault("intermediate_artifacts_retained", res.get("intermediate_artifacts_retained"))
    row.setdefault("intermediate_artifact_policy", res.get("intermediate_artifact_policy"))
    row.setdefault("downstream_status", down.get("status"))
    row.setdefault("downstream_output_csv", down.get("output_csv"))
    row.setdefault("downstream_metrics_json", down.get("metrics_json"))
    row.setdefault("downstream_returncode", inf.get("returncode"))
    row.setdefault("downstream_elapsed_ms", inf.get("elapsed_ms"))

    # Preserve any top-level metric_* values, and backfill from downstream.metrics
    # for older utility_result.json files that only stored nested metrics.
    metrics = down.get("metrics") if isinstance(down.get("metrics"), dict) else {}
    if metrics:
        for k, v in flatten_metrics(metrics).items():
            row.setdefault(k, v)
    metrics_json = down.get("metrics_json")
    if metrics_json and Path(str(metrics_json)).exists():
        try:
            loaded_metrics = load_json(metrics_json)
            if isinstance(loaded_metrics, dict):
                for k, v in flatten_metrics(loaded_metrics).items():
                    row.setdefault(k, v)
        except Exception:
            pass
    for k, v in res.items():
        if str(k).startswith("metric_"):
            row[k] = v
    return row



COMPACT_METRIC_SUMMARY_COLS = [
    "scenario_id", "task", "method_id", "status", "downstream_status",
    "preprocessing_status", "metric_status", "metric_level", "metric_n",
    "metric_precision", "metric_recall", "metric_f1", "metric_f2", "metric_accuracy",
    "metric_tp", "metric_fp", "metric_fn", "metric_tn", "metric_error",
    "downstream_output_csv", "downstream_metrics_json",
]

# A small, publication-useful extension set.  Avoid per-class columns by default
# because ADL/YouHome can add hundreds of metric_per_class_* columns and make
# utility_metrics_summary.csv unexpectedly huge.  The full wide table is still
# available as utility_metrics_summary_wide.csv or with --wide-metrics-summary.
COMPACT_EXTRA_METRIC_COLS = [
    "metric_macro_f1",
    "metric_micro_f1",
    "metric_weighted_f1",
    "metric_balanced_accuracy",
    "metric_avg_precision",
    "metric_ap",
    "metric_mae",
    "metric_exact_match",
    "metric_video_level_accuracy",
    "metric_video_level_macro_f1",
    "metric_video_level_weighted_f1",
    "metric_video_level_n",
    "metric_n_prediction_rows",
    "metric_n_aligned_rows",
    "metric_missing_audio_samples",
    "metric_missing_image_samples",
    "metric_n_missing_xml_samples",
    "metric_n_empty_xml_samples",
]

def _metric_summary_columns(flat_rows: Sequence[Dict[str, Any]], *, wide: bool = False) -> Tuple[List[str], List[str]]:
    """Return compact and wide metric-summary column lists.

    Compact mode intentionally mirrors the normal evaluator output plus a small
    whitelist of useful aggregate metrics.  Wide mode includes all metric_*
    fields, including per-class metrics, and can therefore be much larger.
    """
    discovered_metric_cols = sorted(
        k for row in flat_rows for k in row.keys()
        if str(k).startswith("metric_")
    )
    compact_cols = list(COMPACT_METRIC_SUMMARY_COLS)
    for c in COMPACT_EXTRA_METRIC_COLS:
        if c in discovered_metric_cols and c not in compact_cols:
            compact_cols.append(c)
    if wide:
        wide_cols = list(compact_cols)
        for c in discovered_metric_cols:
            if c not in wide_cols:
                wide_cols.append(c)
    else:
        wide_cols = []
    return compact_cols, wide_cols

def write_utility_summary_files_from_results(
    results: Sequence[Dict[str, Any]],
    out_dir: Path,
    *,
    wide_metrics_summary: bool = False,
) -> Dict[str, Any]:
    """Write aggregate JSON/CSV summaries from existing result dictionaries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        [dict(r) for r in results],
        key=lambda r: (
            str(r.get("scenario_id") or ""),
            str(r.get("task") or ""),
            str(r.get("method_id") or ""),
            str(r.get("utility_work_dir") or ""),
        ),
    )
    write_json(ordered, out_dir / "utility_results.json")

    flat_rows = [flatten_utility_result_for_summary(r) for r in ordered]
    write_csv_rows(flat_rows, out_dir / "utility_summary.csv")

    metric_cols, wide_metric_cols = _metric_summary_columns(flat_rows, wide=wide_metrics_summary)
    metric_rows = [{c: row.get(c) for c in metric_cols} for row in flat_rows]
    write_csv_rows(metric_rows, out_dir / "utility_metrics_summary.csv", fieldnames=metric_cols)

    wide_metrics_summary_csv = None
    if wide_metric_cols:
        wide_rows = [{c: row.get(c) for c in wide_metric_cols} for row in flat_rows]
        wide_path = out_dir / "utility_metrics_summary_wide.csv"
        write_csv_rows(wide_rows, wide_path, fieldnames=wide_metric_cols)
        wide_metrics_summary_csv = str(wide_path)

    report = {
        "rebuilt_at_ms": now_ms(),
        "out_dir": str(out_dir),
        "results_json": str(out_dir / "utility_results.json"),
        "summary_csv": str(out_dir / "utility_summary.csv"),
        "metrics_summary_csv": str(out_dir / "utility_metrics_summary.csv"),
        "metrics_summary_wide_csv": wide_metrics_summary_csv,
        "metrics_summary_columns": len(metric_cols),
        "wide_metrics_summary_columns": len(wide_metric_cols) if wide_metric_cols else None,
        "results": len(ordered),
        "ok": sum(1 for r in ordered if r.get("status") == "ok"),
        "errors": sum(1 for r in ordered if r.get("status") == "error"),
        "statuses": dict(sorted({str(r.get("status") or ""): sum(1 for x in ordered if str(x.get("status") or "") == str(r.get("status") or "")) for r in ordered}.items())),
        "tasks": sorted({str(r.get("task") or "") for r in ordered if r.get("task")}),
        "methods": sorted({str(r.get("method_id") or "") for r in ordered if r.get("method_id")}),
        "unique_task_pipeline_fingerprints": len({
            (str(r.get("task") or ""), str(r.get("pipeline_fingerprint") or ""))
            for r in ordered
            if r.get("pipeline_fingerprint")
        }),
    }
    write_json(report, out_dir / "utility_summary_rebuild_report.json")
    return report


def rebuild_existing_utility_summaries(args: argparse.Namespace) -> int:
    """Scan existing utility_result.json files and rebuild aggregate summaries.

    This mode deliberately does not call pipeline preprocessing, downstream
    inference, task configuration, or model-loading code. It is safe to run after
    partial experiments or after combining raw/manual/direct/full/ablation runs.
    """
    out_dir = Path(args.out_dir)
    scan_dirs = _parse_repeated_csv_dirs(getattr(args, "summary_scan_dir", None), default=[out_dir])
    all_results: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    seen_paths: set[str] = set()
    scanned_files = 0
    skipped: List[Dict[str, Any]] = []

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            skipped.append({"path": str(scan_dir), "reason": "scan_dir_missing"})
            continue
        for result_path in sorted(scan_dir.glob("**/utility_result.json")):
            # Avoid accidentally reading a file we just wrote in some unusual layout.
            key_path = str(result_path.resolve())
            if key_path in seen_paths:
                continue
            seen_paths.add(key_path)
            scanned_files += 1
            try:
                result = load_json(result_path)
            except Exception as exc:
                skipped.append({"path": str(result_path), "reason": f"load_error:{exc!r}"})
                continue
            if not isinstance(result, dict):
                skipped.append({"path": str(result_path), "reason": "not_a_json_object"})
                continue

            inferred = _infer_ids_from_utility_result_path(result_path)
            result.setdefault("scenario_id", inferred.get("scenario_id"))
            result.setdefault("method_id", inferred.get("method_id"))
            result.setdefault("utility_work_dir", str(result_path.parent))
            result.setdefault("utility_result_json", str(result_path))

            # Prefer results already under --out-dir over external scan dirs;
            # otherwise keep the newest mtime for the same scenario/method/task.
            dedupe_key = (
                str(result.get("scenario_id") or inferred.get("scenario_id") or ""),
                str(result.get("method_id") or inferred.get("method_id") or ""),
                str(result.get("task") or ""),
            )
            existing = all_results.get(dedupe_key)
            if existing is None:
                all_results[dedupe_key] = result
            else:
                old_path = Path(str(existing.get("utility_result_json") or ""))
                new_in_out = out_dir.resolve() in result_path.resolve().parents or result_path.parent == out_dir.resolve()
                old_in_out = old_path.exists() and (out_dir.resolve() in old_path.resolve().parents or old_path.parent == out_dir.resolve())
                choose_new = False
                if new_in_out and not old_in_out:
                    choose_new = True
                elif new_in_out == old_in_out:
                    try:
                        choose_new = result_path.stat().st_mtime >= old_path.stat().st_mtime
                    except Exception:
                        choose_new = True
                if choose_new:
                    all_results[dedupe_key] = result

    results = list(all_results.values())
    report = write_utility_summary_files_from_results(
        results,
        out_dir,
        wide_metrics_summary=bool(getattr(args, "wide_metrics_summary", False)),
    )
    report.update({
        "mode": "summarize_existing_only",
        "scan_dirs": [str(p) for p in scan_dirs],
        "scanned_utility_result_files": scanned_files,
        "deduped_results": len(results),
        "skipped": skipped[:200],
        "num_skipped": len(skipped),
    })
    write_json(report, out_dir / "utility_summary_rebuild_report.json")
    print(json.dumps(report, indent=2), flush=True)
    return 0

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    set_progress_enabled(not args.no_progress)
    args.project_root = str(Path(args.project_root).resolve())
    set_runtime_config(args.project_root, args.runtime_package)
    gpu_visibility = configure_gpu_visibility_for_run(args)
    if gpu_visibility.get("enabled"):
        progress_write(
            "[device] forcing CUDA visibility: "
            f"CUDA_DEVICE_ORDER={gpu_visibility.get('CUDA_DEVICE_ORDER')} "
            f"CUDA_VISIBLE_DEVICES={gpu_visibility.get('CUDA_VISIBLE_DEVICES')} "
            f"prefer_gpu_name={gpu_visibility.get('prefer_gpu_name')}"
        )
    elif gpu_visibility.get("reason"):
        progress_write(f"[device] not overriding CUDA visibility: {gpu_visibility.get('reason')}")
    if args.pipeline_root is None:
        args.pipeline_root = "runs/flexible_context_pipeline_generation" if args.request_mode == "flexible" else "runs/context_pipeline_generation"
    if args.out_dir is None:
        args.out_dir = "runs/utility_eval_flexible" if args.request_mode == "flexible" else "runs/utility_eval"
    pipeline_root = Path(args.pipeline_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "summarize_existing_only", False):
        return rebuild_existing_utility_summaries(args)

    apply_auto_defaults(args)

    rows = discover_method_rows(pipeline_root, getattr(args, "pipeline_discovery", "merge"))
    requested_methods = set(parse_csv_list(args.methods))
    default_ablation_modes = selected_default_ablation_modes(args, rows) if not requested_methods else set()
    discovered_ablation_modes = available_ablation_modes(rows)
    selected = filter_rows(args, rows)
    cache_enabled = not bool(getattr(args, "no_task_pipeline_cache", False))
    cache_plan = task_pipeline_cache_plan(selected)
    cache_key_order = {key: i + 1 for i, key in enumerate(cache_plan.get("key_order", []))}
    cache_group_sizes = {g["cache_key"]: g["row_count"] for g in cache_plan.get("groups", [])}
    cross_run_reuse_index, cross_run_reuse_plan = build_cross_run_reuse_index(args, selected)
    current_reuse_plan = current_out_dir_reuse_plan(args, selected)
    pending_plan = pending_execution_plan(selected, current_reuse_plan, cross_run_reuse_index)

    task_config = configured_tasks(args, {r.task for r in rows if r.task in TASK_TO_SHORT})
    device_summary = resolved_device_summary(args.device)
    progress_write(
        f"[device] requested={device_summary['requested_device']} "
        f"ultralytics={device_summary['ultralytics_device']} torch={device_summary['torch_device']} "
        f"| {device_summary['cuda_summary']}"
    )
    preflight_report = build_preflight_report(args, rows, selected, task_config, cache_plan, cross_run_reuse_plan, pending_plan)
    maybe_confirm_preflight(args, preflight_report)
    write_json({
        "request_mode": args.request_mode,
        "pipeline_root": str(pipeline_root),
        "out_dir": str(out_dir),
        "total_discovered_rows": len(rows),
        "discovered_rows_by_method": dict(sorted(Counter(r.method_id for r in rows).items())),
        "selected_rows": len(selected),
        "selected_rows_by_method": dict(sorted(Counter(r.method_id for r in selected).items())),
        "pipeline_discovery": getattr(args, "pipeline_discovery", "merge"),
        "requested_tasks": args.tasks,
        "selected_tasks": sorted({r.task for r in selected}),
        "task_config": task_config,
        "preflight_report": preflight_report,
        "pending_execution_plan": pending_plan,
        "cross_run_reuse": cross_run_reuse_plan,
        "current_out_dir_reuse": current_reuse_plan,
        "methods": sorted({r.method_id for r in selected}),
        "ablation_policy": ("explicit_methods" if requested_methods else args.ablation_policy),
        "default_meaningful_ablation_modes": DEFAULT_MEANINGFUL_ABLATION_MODES,
        "default_skipped_ablation_modes": DEFAULT_SKIPPED_ABLATION_MODES,
        "discovered_ablation_modes": sorted(discovered_ablation_modes),
        "selected_default_ablation_modes": sorted(default_ablation_modes),
        "selected_ablation_modes": sorted({r.ablation_mode for r in selected if r.ablation_mode}),
        "skip_ablation_matches_full": (
            not bool(args.keep_ablation_matches_full)
            and not bool(requested_methods)
            and str(args.ablation_policy) == "meaningful"
            and not bool(args.include_ablations)
        ),
        "split": args.split,
        "runtime_package": args.runtime_package,
        "project_root": args.project_root,
        "device_summary": device_summary,
        "gpu_visibility": gpu_visibility,
        "dry_run": bool(args.dry_run),
        "keep_intermediate_data": bool(args.keep_intermediate_data),
        "intermediate_root": args.intermediate_root,
        "intermediate_artifact_policy": (
            "retained_under_each_method_dir/intermediate_artifacts"
            if args.keep_intermediate_data
            else "temporary_deleted_after_each_downstream_run"
        ),
        "resume_existing_results": {
            "enabled": not bool(getattr(args, "rerun_existing", False)),
            "policy": "Reuse per-method utility_result.json when status/downstream are ok and prediction CSV/metrics still exists; backfill metrics if missing.",
            "rerun_existing": bool(getattr(args, "rerun_existing", False)),
            "recompute_existing_metrics": bool(getattr(args, "recompute_existing_metrics", False)),
        },
        "task_pipeline_cache": {
            "enabled": cache_enabled,
            "policy": (
                "Within this evaluator invocation, rows with the same task and executable pipeline fingerprint "
                "reuse the first row's preprocessing/downstream/metric outputs."
            ),
            **cache_plan,
        },
    }, out_dir / "utility_eval_plan.json")

    progress_write(
        f"[plan] selected utility rows={len(selected)}. "
        f"The outer progress denominator is selected scenario/method rows after task/method/split filters. "
        f"Unique task+pipeline runs={cache_plan['unique_task_pipeline_runs']}; "
        f"cache-reused duplicate rows={cache_plan['duplicate_rows_reused_from_cache']} "
        f"(cache {'enabled' if cache_enabled else 'disabled'}). "
        f"Already ok in out_dir={pending_plan['already_ok_in_out_dir_unique_runs']}; "
        f"cross-run reusable={pending_plan['cross_run_reusable_unique_runs']}; "
        f"still-to-run unique={pending_plan['pending_unique_task_pipeline_runs']}; "
        f"still-to-run output schemas={pending_plan['pending_unique_output_schemas']}."
    )

    if args.analysis_only:
        print(json.dumps({
            "analysis_only": True,
            "out_dir": str(out_dir),
            "pipeline_root": str(pipeline_root),
            "pending_execution_plan": pending_plan,
            "cross_run_reuse": cross_run_reuse_plan,
            "current_out_dir_reuse": current_reuse_plan,
            "plan_json": str(out_dir / "utility_eval_plan.json"),
        }, indent=2), flush=True)
        return 0

    results: List[Dict[str, Any]] = []
    task_pipeline_cache: Dict[str, Dict[str, Any]] = {}
    task_pipeline_cache_info: Dict[str, Dict[str, Any]] = {}
    iterator: Iterable[MethodRow]
    if progress_enabled():
        iterator = tqdm(
            selected,
            desc=f"utility rows n={len(selected)} unique_runs={cache_plan['unique_task_pipeline_runs']}",
            unit="row",
            position=0,
            dynamic_ncols=True,
        )
    else:
        iterator = selected

    for row_index, r in enumerate(iterator, 1):
        cache_key, cache_info = task_pipeline_cache_key(r)
        pipeline_fp = cache_info.get("pipeline_fingerprint")
        unique_index = cache_key_order.get(cache_key, len(task_pipeline_cache) + 1)
        duplicate_count = cache_group_sizes.get(cache_key, 1)
        cache_status = "hit" if cache_enabled and cache_key in task_pipeline_cache else ("miss" if cache_enabled else "disabled")
        if tqdm and not args.no_progress:
            iterator.set_postfix_str(
                f"row={row_index}/{len(selected)} run={unique_index}/{cache_plan['unique_task_pipeline_runs']} "
                f"cache={cache_status} task={r.task} scenario={r.scenario_id} method={r.method_id} pipe={pipeline_fp}",
            )  # type: ignore[attr-defined]

        progress_write(
            f"[utility] row {row_index}/{len(selected)}; unique task+pipeline run "
            f"{unique_index}/{cache_plan['unique_task_pipeline_runs']}; cache={cache_status}; "
            f"task={r.task}; scenario={r.scenario_id}; method={r.method_id}; "
            f"pipeline_fingerprint={pipeline_fp}; rows_sharing_this_key={duplicate_count}"
        )

        if cache_enabled and cache_key in task_pipeline_cache:
            original = task_pipeline_cache[cache_key]
            progress_write(
                f"[cache] reuse task={r.task} pipeline_fingerprint={pipeline_fp}: "
                f"{r.scenario_id}/{r.method_id} uses outputs from "
                f"{original.get('scenario_id')}/{original.get('method_id')}"
            )
            result = clone_cached_result_for_row(args, r, original, cache_key, cache_info)
        else:
            existing = load_existing_completed_result(args, r, cache_key, cache_info)
            if existing is not None:
                progress_write(
                    f"[resume] reused existing ok result for {r.scenario_id}/{r.method_id}; "
                    f"use --rerun-existing to force recomputation"
                )
                result = existing
            else:
                cross_reused = None if getattr(args, "rerun_existing", False) else cross_run_reuse_index.get(cache_key)
                if cross_reused is not None:
                    progress_write(
                        f"[cross-run-reuse] task={r.task} pipeline_fingerprint={pipeline_fp}: "
                        f"{r.scenario_id}/{r.method_id} reuses previous utility outputs from "
                        f"{cross_reused.get('scenario_id')}/{cross_reused.get('method_id')} "
                        f"at {cross_reused.get('utility_work_dir')}"
                    )
                    result = clone_cross_run_result_for_row(args, r, cross_reused, cache_key, cache_info)
                else:
                    result = evaluate_one(args, r)
                    result["cache_status"] = cache_status
                    result["cache_key"] = cache_key if cache_enabled else None
                    result["pipeline_fingerprint"] = pipeline_fp
            if cache_enabled:
                task_pipeline_cache[cache_key] = result
                task_pipeline_cache_info[cache_key] = cache_info
            # evaluate_one writes before cache fields are attached; update the
            # per-method result file so cache/resume diagnostics are visible there too.
            if result.get("utility_work_dir"):
                write_json(result, Path(str(result["utility_work_dir"])) / "utility_result.json")
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

    metric_cols = [
        "scenario_id", "task", "method_id", "status", "downstream_status",
        "preprocessing_status", "metric_status", "metric_level", "metric_n",
        "metric_precision", "metric_recall", "metric_f1", "metric_f2", "metric_accuracy",
        "metric_tp", "metric_fp", "metric_fn", "metric_tn", "metric_error",
        "downstream_output_csv", "downstream_metrics_json",
    ]
    metric_rows = [{c: row.get(c) for c in metric_cols} for row in flat_rows]
    write_csv_rows(metric_rows, out_dir / "utility_metrics_summary.csv", fieldnames=metric_cols)

    print(json.dumps({
        "discovered_rows": len(rows),
        "evaluated_rows": len(results),
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "errors": sum(1 for r in results if r.get("status") == "error"),
        "selected_tasks": sorted({r.get("task") for r in results if r.get("task")}),
        "split": args.split,
        "out_dir": str(out_dir),
        "summary_csv": str(out_dir / "utility_summary.csv"),
        "metrics_summary_csv": str(out_dir / "utility_metrics_summary.csv"),
        "results_json": str(out_dir / "utility_results.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
