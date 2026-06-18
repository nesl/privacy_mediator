from __future__ import annotations
import csv, json, wave
from pathlib import Path
from typing import Any, Dict, List, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

def try_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None

def list_media_files(path: str | Path, modality: str) -> List[Path]:
    p = Path(path)
    if not p.exists(): return []
    if p.is_file(): return [p]
    if modality == "image":
        exts = IMAGE_EXTS
    elif modality == "video":
        exts = IMAGE_EXTS | VIDEO_EXTS
    elif modality == "audio":
        exts = AUDIO_EXTS
    else:
        exts = set()
    return sorted([x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in exts])

def sample_video_frames(video_path: str | Path, max_frames: int = 24) -> List[Any]:
    cv2 = try_import("cv2")
    if cv2 is None: return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return []
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = list(range(max_frames)) if n <= 0 else sorted(set(int(i * max(1, n - 1) / max(1, max_frames - 1)) for i in range(max_frames)))
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None: frames.append(frame)
    cap.release()
    return frames

def load_image_frames(path: str | Path, max_frames: int = 32) -> List[Any]:
    cv2 = try_import("cv2")
    if cv2 is None: return []
    p = Path(path)
    files = list_media_files(p, "video" if (p.is_dir() or p.suffix.lower() in VIDEO_EXTS) else "image")
    frames = []
    for f in files[:max_frames]:
        if f.suffix.lower() in VIDEO_EXTS:
            frames.extend(sample_video_frames(f, max_frames=max_frames))
            if len(frames) >= max_frames: break
        else:
            img = cv2.imread(str(f))
            if img is not None: frames.append(img)
    return frames[:max_frames]

def image_area(frame: Any) -> int:
    try:
        h, w = frame.shape[:2]
        return int(h * w)
    except Exception:
        return 0

def read_timeseries(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists(): return []
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, list): return [dict(x) for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for key in ("records", "rows", "events", "data", "observations"):
                if isinstance(data.get(key), list):
                    return [dict(x) for x in data[key] if isinstance(x, dict)]
            return [data]
    if p.suffix.lower() in {".csv", ".tsv"}:
        delim = "\t" if p.suffix.lower() == ".tsv" else ","
        with open(p, newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f, delimiter=delim)]
    return []

def numeric(v: Any) -> Optional[float]:
    if v is None: return None
    try:
        if isinstance(v, str) and not v.strip(): return None
        return float(v)
    except Exception:
        return None

def wav_basic_stats(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    out: Dict[str, Any] = {"path": str(p)}
    try:
        with wave.open(str(p), "rb") as wf:
            out["channels"] = wf.getnchannels()
            out["sample_width"] = wf.getsampwidth()
            out["sample_rate"] = wf.getframerate()
            out["frames"] = wf.getnframes()
            out["duration_sec"] = wf.getnframes() / float(wf.getframerate() or 1)
    except Exception as exc:
        out["error"] = str(exc)
    return out
