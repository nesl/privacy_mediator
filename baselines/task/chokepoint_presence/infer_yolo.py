"""YOLO person-presence baseline for ChokePoint portal/visitor monitoring.

Input manifest columns produced by ``make_manifest``:
    sample_id,split,frame_dir,xml_path,...

Output frame-level CSV with person counts and boxes. For image directories with
numeric frame names (e.g., 00000233.jpg), the emitted ``frame_index`` is parsed
from the filename so predictions align with the XML ground truth.

This script is deliberately defensive for CUDA/cuDNN issues. Some PyTorch/CUDA
combinations can fail on individual inputs with errors like
``GET was unable to find an engine to execute this computation``. Use
``--safe-cuda`` or ``--retry-cpu-on-error`` if that happens.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path
from typing import Any, Iterator

import cv2
import pandas as pd
from tqdm import tqdm

from baselines.task.common.manifest import filter_split, load_manifest
from baselines.task.common.video import iter_video_frames, sorted_images

PRED_COLUMNS = ["sample_id", "split", "frame_index", "person_count", "person_present", "boxes_json", "frame_path"]
ERROR_COLUMNS = ["sample_id", "split", "frame_index", "frame_path", "error", "traceback"]


def _device_wants_cuda(device: object) -> bool:
    """Return True when an Ultralytics/PyTorch device string implies CUDA."""
    if device is None:
        return False
    text = str(device).strip().lower()
    if not text or text == "cpu":
        return False
    return True


def _query_nvidia_gpus() -> list[dict[str, str]]:
    """Return GPUs from nvidia-smi as index/uuid/name dictionaries.

    We use UUIDs for CUDA_VISIBLE_DEVICES because CUDA/PyTorch device ordering can
    differ from the order shown by nvidia-smi. UUID selection is the least
    ambiguous way to hide the unsupported TITAN V and expose only the RTX 2070.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name",
        "--format=csv,noheader",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception:
        return []

    gpus: list[dict[str, str]] = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",", 2)]
        if len(parts) != 3:
            continue
        index, uuid, name = parts
        gpus.append({"index": index, "uuid": uuid, "name": name})
    return gpus


def maybe_auto_select_cuda_gpu(prefer_name: str | None, device: object, enabled: bool = True) -> None:
    """Hide all GPUs except the preferred one before importing torch/ultralytics.

    This must run before Ultralytics imports PyTorch. Passing --device 0 to
    Ultralytics only chooses among *visible* CUDA devices; it does not change
    which physical GPU is CUDA device 0. On mixed systems, PyTorch can still see
    the unsupported TITAN V as GPU0 unless CUDA_VISIBLE_DEVICES is set first.
    """
    if not enabled or not _device_wants_cuda(device):
        return

    # Do not override an explicit user choice.
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        print(f"Using existing CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        return

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    if not prefer_name:
        return

    gpus = _query_nvidia_gpus()
    needle = prefer_name.lower()
    for gpu in gpus:
        if needle in gpu["name"].lower():
            # Use UUID instead of index to avoid nvidia-smi/CUDA ordering mismatch.
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
            print(
                "Auto-selected CUDA GPU: "
                f"nvidia-smi index={gpu['index']} name={gpu['name']} uuid={gpu['uuid']}. "
                "Inside PyTorch/Ultralytics this will be cuda:0."
            )
            return

    if gpus:
        names = "; ".join(f"{g['index']}:{g['name']}" for g in gpus)
        print(f"Warning: preferred GPU name {prefer_name!r} not found. Available GPUs: {names}")


def report_torch_cuda_device(device: object, require_not_titan: bool = True) -> None:
    """Print the visible torch CUDA device and optionally block TITAN V."""
    if not _device_wants_cuda(device):
        return
    try:
        import torch
    except Exception as e:
        print(f"Warning: could not import torch for CUDA device check: {e}")
        return

    if not torch.cuda.is_available():
        print("Warning: CUDA was requested, but torch.cuda.is_available() is False.")
        return

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"Torch visible cuda:0 = {name}, compute capability={capability}")
    if require_not_titan and "titan v" in name.lower():
        raise SystemExit(
            "Refusing to run on TITAN V because this PyTorch/cuDNN build does not support SM 7.0. "
            "Set CUDA_VISIBLE_DEVICES to the RTX 2070 UUID/index, or use --prefer-gpu-name 'RTX 2070'."
        )


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip() == ""


def _candidate_paths(value: object, data_root: Path | None) -> list[Path]:
    if _is_blank(value):
        return []
    raw = Path(str(value))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        # Try the path as written first. This avoids double-prefixing values like
        # data/chokepoint/P1E_S1/P1E_S1_C1 when --data-root data/chokepoint is supplied.
        candidates.append(raw)
        if data_root is not None:
            candidates.append(data_root / raw)
            # If raw already starts with the root path text, data_root / raw is wrong;
            # Path(raw) above covers that case.
    # Preserve order while dropping duplicates.
    out: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out


def _has_images(path: Path) -> bool:
    return path.is_dir() and len(sorted_images(path)) > 0


def build_frame_dir_index(data_root: str | Path | None) -> dict[str, Path]:
    """Map sample_id-like directory names to image directories under data_root."""
    if not data_root:
        return {}
    root = Path(data_root)
    if not root.exists():
        return {}
    index: dict[str, Path] = {}
    counts: dict[str, int] = {}
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        if p.name == "groundtruth" or "groundtruth" in p.parts:
            continue
        imgs = sorted_images(p)
        if imgs and len(imgs) > counts.get(p.name, -1):
            index[p.name] = p
            counts[p.name] = len(imgs)
    return index


def resolve_input_path(row: pd.Series, data_root: str | Path | None, frame_dir_index: dict[str, Path]) -> Path | None:
    root = Path(data_root) if data_root else None
    for col in ("frame_dir", "video_path", "image_path"):
        for p in _candidate_paths(row.get(col), root):
            if p.exists():
                return p

    sample_id = str(row.get("sample_id", "")).strip()
    group_id = str(row.get("group_id", "")).strip()
    if sample_id:
        if sample_id in frame_dir_index:
            return frame_dir_index[sample_id]
        if root is not None:
            fallback_candidates = [root / sample_id]
            if group_id:
                fallback_candidates.insert(0, root / group_id / sample_id)
            for p in fallback_candidates:
                if p.exists():
                    return p
    return None


def boxes_from_result(result, conf_threshold: float):
    boxes = []
    if result.boxes is None:
        return boxes
    for b in result.boxes:
        cls = int(b.cls.item()) if b.cls is not None else -1
        conf = float(b.conf.item()) if b.conf is not None else 0.0
        # COCO class 0 = person.
        if cls == 0 and conf >= conf_threshold:
            xyxy = b.xyxy[0].detach().cpu().numpy().tolist()
            boxes.append({"xyxy": [float(x) for x in xyxy], "conf": conf, "class_id": cls})
    return boxes


def frame_index_from_path(path: Path, fallback: int) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return fallback


def iter_chokepoint_frames(path: Path, stride: int = 1, max_frames: int | None = None) -> Iterator[tuple[int, Any, str]]:
    """Yield frames, preserving numeric frame indexes for frame directories.

    For image directories we yield the image path rather than an already-loaded
    ndarray. That lets Ultralytics handle decoding/preprocessing, which is often
    more robust across unusual images. For videos, we still yield decoded frames.
    """
    if path.is_dir():
        kept = 0
        imgs = sorted_images(path)
        for ordinal, img_path in enumerate(imgs):
            if ordinal % stride != 0:
                continue
            # Validate that OpenCV can read the file, but pass the path to YOLO.
            # Passing paths avoids some CUDA/preprocess edge cases.
            if cv2.imread(str(img_path)) is None:
                continue
            yield frame_index_from_path(img_path, ordinal), str(img_path), str(img_path)
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break
        return
    for frame_idx, frame in iter_video_frames(path, stride=stride, max_frames=max_frames):
        yield frame_idx, frame, str(path)


def configure_torch(safe_cuda: bool = False, disable_cudnn: bool = False) -> None:
    """Apply conservative CUDA settings after torch is imported.

    ``safe_cuda`` keeps CUDA enabled but avoids cuDNN autotuning choices that can
    trigger backend engine failures on some machines. ``disable_cudnn`` is slower
    but is a stronger fallback.
    """
    try:
        import torch

        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
            if disable_cudnn:
                torch.backends.cudnn.enabled = False
        if safe_cuda and hasattr(torch, "set_float32_matmul_precision"):
            # Avoid aggressive precision/autotune paths.
            torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def predict_one(model, source: Any, kwargs: dict, conf: float):
    result = model.predict(source, **kwargs)[0]
    return boxes_from_result(result, conf)


def _write_csvs(output_csv: str, rows: list[dict], error_log: str | None, error_rows: list[dict]) -> None:
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PRED_COLUMNS).to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(rows)} prediction rows)")

    if error_log or error_rows:
        error_path = Path(error_log) if error_log else out_path.with_suffix(".errors.csv")
        error_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(error_rows, columns=ERROR_COLUMNS).to_csv(error_path, index=False)
        print(f"Wrote {error_path} ({len(error_rows)} skipped/error rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default=None, help="Optional split filter, e.g. train/val/test")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics detection model, e.g. yolo11n.pt/yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.05, help="Detector confidence floor. Use a low value and tune post-hoc in evaluate/tune_postprocess.")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Ultralytics device. Defaults to CUDA device 0; use 'cpu' to force CPU.")
    parser.add_argument("--prefer-gpu-name", default="RTX 2070", help="When using CUDA and CUDA_VISIBLE_DEVICES is unset, expose only the GPU whose name contains this text. Use empty string to disable name matching.")
    parser.add_argument("--no-auto-select-gpu", action="store_true", help="Do not set CUDA_VISIBLE_DEVICES automatically before importing Torch/Ultralytics.")
    parser.add_argument("--allow-titan-v", action="store_true", help="Allow running even if torch sees TITAN V as cuda:0. Not recommended with current cuDNN builds.")
    parser.add_argument("--augment", action="store_true", help="Use YOLO test-time augmentation; slower but sometimes improves detections")
    parser.add_argument("--all-classes", action="store_true", help="Do not restrict YOLO NMS to COCO person class")
    parser.add_argument("--safe-cuda", action="store_true", help="Use conservative CUDA/cuDNN settings to avoid backend engine errors")
    parser.add_argument("--disable-cudnn", action="store_true", help="Disable cuDNN convolutions. Slower, but can avoid 'unable to find an engine' errors")
    parser.add_argument("--retry-cpu-on-error", action="store_true", help="If CUDA inference fails on a frame, retry that frame on CPU")
    parser.add_argument("--skip-error-frames", action="store_true", help="Skip frames that still fail after retrying; otherwise raise the error")
    parser.add_argument("--error-log", default=None, help="Optional CSV path of rows/frames that failed during inference")
    parser.add_argument("--flush-cache-every", type=int, default=500, help="Clear CUDA cache every N frames; 0 disables")
    parser.add_argument("--respect-has-images", action="store_true", help="Honor has_images=0 in the manifest. Default is to try path fallbacks first.")
    parser.add_argument("--fail-on-missing-input", action="store_true", help="Raise immediately when a manifest row has no usable frame_dir/video_path/image_path")
    parser.add_argument("--allow-empty-output", action="store_true", help="Do not exit nonzero when no prediction rows were produced")
    args = parser.parse_args()

    prefer_gpu_name = args.prefer_gpu_name.strip() if args.prefer_gpu_name is not None else ""
    maybe_auto_select_cuda_gpu(
        prefer_name=prefer_gpu_name or None,
        device=args.device,
        enabled=not args.no_auto_select_gpu,
    )

    # Must be set before importing ultralytics/torch to affect cuDNN v8 API use.
    if args.safe_cuda or args.disable_cudnn:
        os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Install ultralytics first: pip install ultralytics") from e

    configure_torch(safe_cuda=args.safe_cuda, disable_cudnn=args.disable_cudnn)
    report_torch_cuda_device(args.device, require_not_titan=not args.allow_titan_v)

    model = YOLO(args.model)
    cpu_model = None

    df = filter_split(load_manifest(args.manifest), args.split)
    if args.respect_has_images and "has_images" in df.columns:
        df = df[df["has_images"].fillna(1).astype(int) == 1].reset_index(drop=True)

    frame_dir_index = build_frame_dir_index(args.data_root)
    if args.data_root and not frame_dir_index:
        print(f"Warning: no image directories found under --data-root {args.data_root}")

    rows: list[dict] = []
    error_rows: list[dict] = []
    n_frames = 0
    n_inputs = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="chokepoint_yolo"):
        sample_id = row.get("sample_id", i)
        path = resolve_input_path(row, args.data_root, frame_dir_index)
        if path is None or not path.exists():
            msg = "No usable frame_dir, video_path, or image_path found; tried manifest fields and data_root/group_id/sample_id fallbacks"
            error_rows.append(
                {
                    "sample_id": sample_id,
                    "split": row.get("split", ""),
                    "frame_index": "",
                    "frame_path": "",
                    "error": msg,
                    "traceback": "",
                }
            )
            if args.fail_on_missing_input:
                _write_csvs(args.output_csv, rows, args.error_log, error_rows)
                raise ValueError(f"Row {i} sample={sample_id}: {msg}")
            continue

        n_inputs += 1
        frame_count_for_input = 0
        for frame_idx, frame_or_path, frame_path in iter_chokepoint_frames(path, stride=args.stride, max_frames=args.max_frames):
            frame_count_for_input += 1
            kwargs = {"imgsz": args.imgsz, "verbose": False, "conf": args.conf}
            if not args.all_classes:
                kwargs["classes"] = [0]
            if args.augment:
                kwargs["augment"] = True
            if args.device is not None:
                kwargs["device"] = args.device

            try:
                boxes = predict_one(model, frame_or_path, kwargs, args.conf)
            except RuntimeError as e:
                msg = str(e)
                # Conservative recovery path for CUDA/cuDNN runtime failures.
                recovered = False
                boxes = []
                if args.retry_cpu_on_error and ("CUDA" in msg or "cudnn" in msg.lower() or "engine" in msg.lower()):
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    try:
                        if cpu_model is None:
                            cpu_model = YOLO(args.model)
                        cpu_kwargs = dict(kwargs)
                        cpu_kwargs["device"] = "cpu"
                        boxes = predict_one(cpu_model, frame_or_path, cpu_kwargs, args.conf)
                        recovered = True
                    except Exception as cpu_e:
                        msg = msg + " | CPU retry failed: " + repr(cpu_e)
                if not recovered:
                    error_rows.append(
                        {
                            "sample_id": sample_id,
                            "split": row.get("split", ""),
                            "frame_index": int(frame_idx),
                            "frame_path": frame_path,
                            "error": msg,
                            "traceback": traceback.format_exc(limit=3),
                        }
                    )
                    if args.skip_error_frames:
                        continue
                    _write_csvs(args.output_csv, rows, args.error_log, error_rows)
                    raise RuntimeError(f"YOLO failed on sample={sample_id} frame={frame_idx} path={frame_path}: {msg}") from e
            except Exception as e:
                error_rows.append(
                    {
                        "sample_id": sample_id,
                        "split": row.get("split", ""),
                        "frame_index": int(frame_idx),
                        "frame_path": frame_path,
                        "error": repr(e),
                        "traceback": traceback.format_exc(limit=3),
                    }
                )
                if args.skip_error_frames:
                    continue
                _write_csvs(args.output_csv, rows, args.error_log, error_rows)
                raise

            rows.append(
                {
                    "sample_id": sample_id,
                    "split": row.get("split", ""),
                    "frame_index": int(frame_idx),
                    "person_count": len(boxes),
                    "person_present": int(len(boxes) > 0),
                    "boxes_json": json.dumps(boxes),
                    "frame_path": frame_path,
                }
            )
            n_frames += 1
            if args.flush_cache_every and n_frames % args.flush_cache_every == 0:
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        if frame_count_for_input == 0:
            error_rows.append(
                {
                    "sample_id": sample_id,
                    "split": row.get("split", ""),
                    "frame_index": "",
                    "frame_path": str(path),
                    "error": "Input path exists but yielded no readable frames/images",
                    "traceback": "",
                }
            )

    _write_csvs(args.output_csv, rows, args.error_log, error_rows)
    print(f"Processed {n_inputs} usable input sequences and {n_frames} sampled frames")

    if not rows and not args.allow_empty_output:
        hint = "No prediction rows were produced. Check the error log, manifest frame_dir values, --data-root, and whether archives were fully extracted."
        raise SystemExit(hint)


if __name__ == "__main__":
    main()
