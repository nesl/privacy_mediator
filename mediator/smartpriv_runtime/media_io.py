from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .data_model import DataItem


def load_image(path: Union[str, Path], rgb: bool = True) -> np.ndarray:
    img = Image.open(path)
    img = img.convert("RGB" if rgb else "L")
    return np.asarray(img)


def save_image(arr: np.ndarray, path: Union[str, Path]) -> None:
    arr2 = np.asarray(arr)
    if arr2.dtype != np.uint8:
        arr2 = np.clip(arr2, 0, 255).astype(np.uint8)
    Image.fromarray(arr2).save(path)


def load_audio(path: Union[str, Path], mono: bool = True) -> Tuple[np.ndarray, int]:
    """Load audio using soundfile when available, falling back to stdlib wave."""
    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(str(path), always_2d=False)
        if mono and getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float32), int(sr)
    except Exception:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        dtype = np.int16 if sampwidth == 2 else np.uint8
        data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if dtype == np.int16:
            data /= 32768.0
        else:
            data = (data - 128.0) / 128.0
        if nch > 1:
            data = data.reshape(-1, nch)
            if mono:
                data = data.mean(axis=1)
        return data, sr


def save_audio(path: Union[str, Path], audio: np.ndarray, sample_rate: int) -> None:
    try:
        import soundfile as sf  # type: ignore

        sf.write(str(path), np.asarray(audio), int(sample_rate))
        return
    except Exception:
        x = np.asarray(audio)
        x = np.clip(x, -1, 1)
        pcm = (x * 32767).astype("<i2")
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1 if pcm.ndim == 1 else pcm.shape[1])
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm.tobytes())


def item_from_media(path: Union[str, Path], media_type: Optional[str] = None) -> DataItem:
    p = Path(path)
    suffix = p.suffix.lower()
    if media_type is None:
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            media_type = "image/x-raw"
        elif suffix in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}:
            media_type = "audio/x-raw"
        elif suffix in {".json"}:
            raise ValueError("Use DataItem.from_jsonable/load_json_item for JSON input.")
        else:
            media_type = "video/x-raw"
    if media_type.startswith("image/"):
        return DataItem(
            caps={"media_type": "image/x-raw", "schema": "raw_image_frame", "properties": {"sensorPrimitive": "image_frame"}},
            data=load_image(p),
            metadata={"source_path": str(p)},
        )
    if media_type.startswith("audio/"):
        audio, sr = load_audio(p)
        return DataItem(
            caps={"media_type": "audio/x-raw", "schema": "raw_audio_waveform", "properties": {"sensorPrimitive": "audio_waveform"}},
            data=audio,
            metadata={"source_path": str(p), "sample_rate": sr},
        )
    # For videos we avoid pulling in ffmpeg by default. Users can pass decoded frames
    # as a list of image arrays, or install imageio/opencv and use their own Source.
    return DataItem(
        caps={"media_type": "video/x-raw", "schema": "raw_video_stream", "properties": {"sensorPrimitive": "video_stream"}},
        data={"video_path": str(p)},
        metadata={"source_path": str(p)},
    )
