"""Extract YOLO-pose keypoints for Le2i/ImViA videos.

Typical use with the user's layout::

    python -m baselines.task.le2i_fall.make_manifest \
      --data-dir data/le2i --output-csv data/le2i/le2i_manifest.csv

    python -m baselines.task.le2i_fall.extract_pose \
      --manifest data/le2i/le2i_manifest.csv \
      --output-dir outputs/le2i_pose \
      --updated-manifest outputs/le2i_manifest_with_keypoints.csv

The script also supports ``--data-dir`` without a pre-existing manifest; it will
create one automatically.

Updates in this version:
  * CUDA is the default inference mode.
  * Before importing Ultralytics/Torch, the script tries to prefer the RTX 2070
    by setting CUDA_VISIBLE_DEVICES to the matching physical GPU index. This
    prevents PyTorch from selecting an unsupported TITAN V as GPU 0.
  * Video files are sanitized by default with ffmpeg before decoding. This strips
    malformed audio streams that can trigger messages such as
    ``[mp3float] Header missing`` and can crash native video-decoding libraries.
  * Use ``--device cpu`` to force CPU mode.
  * Use ``--no-sanitize-videos`` to disable sanitization.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm

from baselines.task.common.manifest import load_manifest, resolve_path
from baselines.task.common.video import iter_video_frames
from baselines.task.le2i_fall.le2i_layout import make_manifest


VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
    ".mpg",
    ".mpeg",
    ".m4v",
    ".wmv",
}


def is_video_file(path: Path) -> bool:
    """Return True if ``path`` looks like a video file rather than a frame dir."""
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def cuda_requested(device: str) -> bool:
    """Return True if the requested YOLO device is a CUDA device."""
    d = str(device).strip().lower()
    return d not in {"cpu", "mps"}


def configure_preferred_cuda_visible_device(prefer_gpu_name: str | None) -> None:
    """
    Prefer a physical GPU before torch is imported.

    This is important on machines with both a TITAN V and an RTX 2070. Newer
    PyTorch CUDA wheels may not support the TITAN V's compute capability 7.0,
    while the RTX 2070's compute capability 7.5 is supported. Setting
    CUDA_VISIBLE_DEVICES here makes the RTX 2070 become CUDA device 0 inside
    PyTorch/Ultralytics.
    """
    if not prefer_gpu_name:
        return

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(
            "CUDA_VISIBLE_DEVICES is already set to "
            f"{os.environ['CUDA_VISIBLE_DEVICES']!r}; not overriding it."
        )
        return

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        print("nvidia-smi not found; leaving CUDA_VISIBLE_DEVICES unchanged.")
        return

    try:
        proc = subprocess.run(
            [nvidia_smi, "-L"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        print(
            "Could not query nvidia-smi; leaving CUDA_VISIBLE_DEVICES unchanged."
            + (f" nvidia-smi stderr: {stderr}" if stderr else "")
        )
        return

    needle = prefer_gpu_name.lower()
    for line in proc.stdout.splitlines():
        # Example:
        # GPU 1: NVIDIA GeForce RTX 2070 (UUID: GPU-...)
        m = re.match(r"\s*GPU\s+(\d+)\s*:\s*(.*?)\s*(?:\(|$)", line)
        if not m:
            continue
        physical_index, gpu_name = m.group(1), m.group(2).strip()
        if needle in gpu_name.lower():
            os.environ["CUDA_VISIBLE_DEVICES"] = physical_index
            print(
                "Selected preferred CUDA GPU before importing torch: "
                f"physical GPU {physical_index} ({gpu_name}). "
                "It will appear as device 0 to PyTorch."
            )
            return

    print(
        f"Preferred GPU name {prefer_gpu_name!r} was not found in nvidia-smi -L; "
        "leaving CUDA_VISIBLE_DEVICES unchanged."
    )


def parse_cuda_device_index(device: str) -> int:
    """Parse common CUDA device strings into a visible CUDA device index."""
    d = str(device).strip().lower()

    if d in {"cuda", "gpu"}:
        return 0

    if d.startswith("cuda:"):
        return int(d.split(":", 1)[1])

    if d.isdigit():
        return int(d)

    raise ValueError(
        f"Unsupported CUDA device string {device!r}. Use '0', '1', 'cuda:0', or 'cpu'."
    )


def validate_torch_device(device: str, min_cuda_cc: float) -> None:
    """
    Validate CUDA availability and print the selected device.

    This runs after torch/ultralytics can be imported. It gives a clear error if
    CUDA was requested but the selected visible device is unavailable or below
    the minimum supported compute capability.
    """
    if not cuda_requested(device):
        print("Using CPU for YOLO inference.")
        return

    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Use --device cpu for CPU mode, or check your PyTorch CUDA install."
        )

    device_index = parse_cuda_device_index(device)
    device_count = torch.cuda.device_count()
    if device_index < 0 or device_index >= device_count:
        raise SystemExit(
            f"CUDA device {device_index} was requested, but PyTorch only sees "
            f"{device_count} CUDA device(s)."
        )

    name = torch.cuda.get_device_name(device_index)
    major, minor = torch.cuda.get_device_capability(device_index)
    cc = float(f"{major}.{minor}")

    print(f"Using CUDA device {device_index}: {name} (compute capability {major}.{minor}).")

    if cc < min_cuda_cc:
        raise SystemExit(
            f"Selected CUDA device {device_index} ({name}) has compute capability {major}.{minor}, "
            f"which is below the configured minimum {min_cuda_cc}. This is likely your TITAN V.\n"
            "Use the RTX 2070 by running with CUDA_VISIBLE_DEVICES set to its physical index, e.g.\n"
            "  CUDA_VISIBLE_DEVICES=1 python -m baselines.task.le2i_fall.extract_pose ... --device 0\n"
            "or keep the default --prefer-gpu-name 'RTX 2070' if nvidia-smi can see it.\n"
            "To bypass this check, pass --min-cuda-cc 0.0."
        )


def sanitized_video_path(src: Path, cache_dir: Path, overwrite: bool = False) -> Path:
    """
    Create an audio-free, re-encoded MP4 copy of a video.

    This avoids FFmpeg/OpenCV/native decoder crashes caused by malformed audio
    streams like:
        [mp3float] Header missing
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit(
            "ffmpeg is required for video sanitization, but it was not found. "
            "Install it with: sudo apt install ffmpeg\n"
            "Or rerun with --no-sanitize-videos to skip this step."
        )

    src = Path(src)
    cache_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha1(str(src.resolve()).encode("utf-8")).hexdigest()[:12]
    dst = cache_dir / f"{src.stem}_{digest}_video_only.mp4"
    tmp = cache_dir / f".{src.stem}_{digest}_video_only.tmp.mp4"

    if dst.exists() and not overwrite:
        return dst

    if tmp.exists():
        tmp.unlink()

    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-err_detect",
        "ignore_err",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(tmp),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        raise RuntimeError(
            f"ffmpeg failed while sanitizing video:\n  {src}\n\n"
            f"Command stderr:\n{stderr if stderr else '[no stderr output]'}"
        ) from e

    tmp.replace(dst)
    return dst


def extract_for_path(
    model,
    path: Path,
    stride: int,
    max_frames: int | None,
    device: str,
) -> np.ndarray:
    keypoints = []

    for _, frame in iter_video_frames(path, stride=stride, max_frames=max_frames):
        results = model.predict(frame, verbose=False, device=device)

        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            keypoints.append(np.zeros((17, 3), dtype=np.float32))
            continue

        k = results[0].keypoints.data.cpu().numpy()  # [N, 17, 3]

        if k.ndim != 3 or k.shape[1] == 0:
            keypoints.append(np.zeros((17, 3), dtype=np.float32))
        else:
            best = int(np.nanargmax(k[:, :, 2].mean(axis=1)))
            keypoints.append(k[best].astype(np.float32))

    if not keypoints:
        keypoints = [np.zeros((17, 3), dtype=np.float32)]

    return np.stack(keypoints, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pose keypoints for Le2i videos.")
    parser.add_argument("--manifest", default=None, help="CSV manifest. If omitted, --data-dir is scanned.")
    parser.add_argument("--data-dir", default=None, help="Le2i root such as data/le2i; used to auto-create a manifest if needed.")
    parser.add_argument("--data-root", default=None, help="Optional root for relative paths in an existing manifest.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--updated-manifest", default=None)
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Ultralytics pose model, e.g. yolo11n-pose.pt or yolov8n-pose.pt")
    parser.add_argument("--stride", type=int, default=3, help="Read every Nth video frame.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=13, help="Used only when auto-creating a manifest.")
    parser.add_argument(
        "--device",
        default="0",
        help="Device for YOLO inference. Default: 0, meaning the first visible CUDA GPU. Use 'cpu' for CPU mode.",
    )
    parser.add_argument(
        "--prefer-gpu-name",
        default="RTX 2070",
        help=(
            "Preferred physical GPU name to expose before torch import when CUDA is requested. "
            "Default: 'RTX 2070'. Use an empty string to disable this behavior."
        ),
    )
    parser.add_argument(
        "--min-cuda-cc",
        type=float,
        default=7.5,
        help="Minimum CUDA compute capability allowed when CUDA is requested. Default: 7.5.",
    )
    parser.add_argument(
        "--sanitize-videos",
        dest="sanitize_videos",
        action="store_true",
        default=True,
        help="Strip audio and re-encode video files before decoding. Enabled by default.",
    )
    parser.add_argument(
        "--no-sanitize-videos",
        dest="sanitize_videos",
        action="store_false",
        help="Disable video sanitization and decode original files directly.",
    )
    args = parser.parse_args()

    # If CUDA is requested, choose the RTX 2070 before importing Ultralytics/Torch.
    if cuda_requested(args.device):
        prefer_gpu_name = args.prefer_gpu_name.strip() if args.prefer_gpu_name else ""
        configure_preferred_cuda_visible_device(prefer_gpu_name or None)

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Install ultralytics first: pip install ultralytics") from e

    validate_torch_device(args.device, min_cuda_cc=args.min_cuda_cc)

    if args.manifest is None:
        if args.data_dir is None:
            raise SystemExit("Provide either --manifest or --data-dir")
        auto_manifest = Path(args.output_dir) / "le2i_manifest.csv"
        make_manifest(args.data_dir, auto_manifest, seed=args.seed)
        manifest_path = auto_manifest
    else:
        manifest_path = Path(args.manifest)

    df = load_manifest(manifest_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    keypoint_paths = []
    decoded_paths = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="extract_pose"):
        sample_id = str(row.get("sample_id", i))
        path = resolve_path(row.get("video_path", None), args.data_root) or resolve_path(
            row.get("frame_dir", None),
            args.data_root,
        )

        if path is None:
            raise ValueError(f"Row {i} must contain video_path or frame_dir")

        out_path = out_dir / f"{sample_id}.npz"

        if out_path.exists() and not args.overwrite:
            keypoint_paths.append(str(out_path))
            decoded_paths.append(str(path))
            continue

        decode_path = path
        if args.sanitize_videos and is_video_file(path):
            decode_path = sanitized_video_path(
                src=path,
                cache_dir=out_dir / "_sanitized_videos",
                overwrite=args.overwrite,
            )

        kpts = extract_for_path(
            model=model,
            path=decode_path,
            stride=args.stride,
            max_frames=args.max_frames,
            device=args.device,
        )

        np.savez_compressed(
            out_path,
            keypoints=kpts,
            stride=np.array(args.stride, dtype=np.int32),
            pose_model=np.array(args.model),
            video_path=np.array(str(path)),
            decoded_video_path=np.array(str(decode_path)),
            device=np.array(str(args.device)),
        )

        keypoint_paths.append(str(out_path))
        decoded_paths.append(str(decode_path))

    df["keypoints_path"] = keypoint_paths
    df["extraction_stride"] = args.stride
    df["pose_model"] = args.model
    df["decoded_video_path"] = decoded_paths
    df["pose_device"] = args.device

    updated = args.updated_manifest or str(
        Path(manifest_path).with_name(Path(manifest_path).stem + "_with_keypoints.csv")
    )

    Path(updated).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(updated, index=False)
    print(f"Wrote {updated}")

    if "split" in df.columns:
        print(df.groupby(["split", "label"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
