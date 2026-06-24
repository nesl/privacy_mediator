from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .base import Operator, semantic_item, merge_caps
from .data_model import DataItem, cap_type, ensure_list
from .media_io import save_audio, save_image
from .registry import register

try:  # OpenCV is the core vision dependency.
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_image(data: Any) -> Optional[np.ndarray]:
    if isinstance(data, np.ndarray):
        if data.ndim in {2, 3}:
            return data
    if isinstance(data, dict) and "frame" in data and isinstance(data["frame"], np.ndarray):
        return data["frame"]
    return None


def _iter_frames(item: DataItem) -> List[np.ndarray]:
    data = item.data
    if isinstance(data, list) and all(isinstance(x, np.ndarray) for x in data):
        return data
    img = _as_image(data)
    return [] if img is None else [img]


def _replace_image_data(original: Any, image: np.ndarray) -> Any:
    if isinstance(original, dict) and "frame" in original:
        out = dict(original)
        out["frame"] = image
        return out
    return image


def _redacted_media_caps(caps: Dict[str, Any]) -> Dict[str, Any]:
    """Mark media output as redacted rather than raw while preserving properties."""
    update: Dict[str, Any] = {"properties": {"redacted": True}}
    t = cap_type(caps)
    if t.startswith("image/"):
        update["media_type"] = "image/x-redacted"
        update["schema"] = "redacted_image_frame"
    elif t.startswith("video/"):
        update["media_type"] = "video/x-redacted"
        update["schema"] = "redacted_video_stream"
    return merge_caps(caps, update)


def _box_xyxy(box: Dict[str, Any], width: int, height: int) -> Tuple[int, int, int, int]:
    if "bbox" in box:
        b = box["bbox"]
    elif "box" in box:
        b = box["box"]
    else:
        b = box
    if isinstance(b, dict):
        x1 = b.get("x1", b.get("left", b.get("x", 0)))
        y1 = b.get("y1", b.get("top", b.get("y", 0)))
        x2 = b.get("x2", x1 + b.get("w", b.get("width", 0)))
        y2 = b.get("y2", y1 + b.get("h", b.get("height", 0)))
    else:
        x1, y1, x2, y2 = list(b)[:4]
        # Treat [x,y,w,h] as width/height if x2/y2 look small relative to x1/y1.
        if x2 <= x1 or y2 <= y1:
            x2, y2 = x1 + x2, y1 + y2
    vals = [float(x1), float(y1), float(x2), float(y2)]
    if max(vals) <= 1.5:  # normalized coordinates
        vals = [vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height]
    x1, y1, x2, y2 = [int(round(v)) for v in vals]
    x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def _annotation_label(ann: Dict[str, Any]) -> str:
    return str(ann.get("label") or ann.get("class") or ann.get("content_type") or ann.get("type") or "")


def _region_annotations(item: DataItem, targets: Sequence[str]) -> List[Dict[str, Any]]:
    targets_l = {t.lower() for t in targets}
    return [a for a in item.annotations if _annotation_label(a).lower() in targets_l and ("bbox" in a or "box" in a or "x1" in a)]


def _cv_gray(image: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return image if image.ndim == 2 else image.mean(axis=2).astype(np.uint8)
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _audio_array(item: DataItem) -> Tuple[Optional[np.ndarray], int]:
    data = item.data
    if isinstance(data, np.ndarray):
        return np.asarray(data, dtype=np.float32), int(item.metadata.get("sample_rate") or item.metadata.get("sr") or 16000)
    if isinstance(data, dict) and "audio" in data:
        return np.asarray(data["audio"], dtype=np.float32), int(data.get("sample_rate") or item.metadata.get("sample_rate") or 16000)
    return None, int(item.metadata.get("sample_rate") or 16000)


def _rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32))) + 1e-12))
    return 20.0 * math.log10(max(rms, 1e-12))


def _zero_crossing_rate(audio: np.ndarray) -> float:
    if audio.size < 2:
        return 0.0
    x = audio.astype(np.float32)
    return float(np.mean(np.abs(np.diff(np.signbit(x)))))


def _simple_tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_']+", s.lower())


# ---------------------------------------------------------------------------
# Dataflow utility operators
# ---------------------------------------------------------------------------


@register
class SourceOperator(Operator):
    operator_id = "op.source"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        # In generated symbolic pipelines the source is a placeholder. Runtime callers
        # pass the actual source item to ExecutablePipeline.process().
        return item


@register
class SampleOperator(Operator):
    operator_id = "op.sample"

    def __init__(self, sample_period_ms: Optional[int] = None, max_rate_hz: Optional[float] = None, **params: Any) -> None:
        super().__init__(sample_period_ms=sample_period_ms, max_rate_hz=max_rate_hz, **params)
        self.last_emit_ms: Optional[float] = None
        if sample_period_ms is None and max_rate_hz:
            sample_period_ms = int(1000.0 / max(float(max_rate_hz), 1e-9))
        self.sample_period_ms = sample_period_ms

    def apply(self, item: DataItem) -> Optional[DataItem]:
        t = _safe_float(item.metadata.get("timestamp_ms"), default=float("nan"))
        if math.isnan(t):
            t = _safe_float(item.metadata.get("at_ms"), default=0.0)
        if self.sample_period_ms and self.last_emit_ms is not None and t > 0:
            if t - self.last_emit_ms < self.sample_period_ms:
                return None
        if t > 0:
            self.last_emit_ms = t
        caps = merge_caps(item.caps, {"properties": {"sampled": True}})
        return item.clone(caps=caps, metadata={"sample_period_ms": self.sample_period_ms})


@register
class WindowOperator(Operator):
    operator_id = "op.window"

    def __init__(self, window_ms: int = 1000, stride_ms: Optional[int] = None, **params: Any) -> None:
        super().__init__(window_ms=window_ms, stride_ms=stride_ms, **params)
        self.window_ms = int(window_ms)
        self.stride_ms = int(stride_ms or window_ms)
        self.buffer: List[DataItem] = []

    def apply(self, item: DataItem) -> Optional[DataItem]:
        t = _safe_float(item.metadata.get("timestamp_ms"), default=0.0)
        if t <= 0:
            caps = merge_caps(item.caps, {"properties": {"windowed": True}})
            return item.clone(caps=caps, metadata={"window_ms": self.window_ms, "stride_ms": self.stride_ms})
        self.buffer.append(item)
        cutoff = t - self.window_ms
        self.buffer = [x for x in self.buffer if _safe_float(x.metadata.get("timestamp_ms"), t) >= cutoff]
        caps = {"semantic_type": "application/x-windowed-events", "schema": "window"}
        # Preserve media caps for raw media because downstream media operators often expect it.
        if cap_type(item.caps).startswith(("video/", "audio/")):
            caps = merge_caps(item.caps, {"properties": {"windowed": True}})
            data = item.data
        else:
            data = [x.to_jsonable(include_payload=True) for x in self.buffer]
        return DataItem(caps=caps, data=data, annotations=list(item.annotations), metadata={"window_ms": self.window_ms, "timestamp_ms": t})


@register
class TriggerGateOperator(Operator):
    operator_id = "op.trigger_gate"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        condition = self.params.get("condition") or "always"
        ok = self._check_condition(item, condition)
        if not ok:
            return None
        caps = merge_caps(item.caps, {"properties": {"event_triggered": True, "gated": True}})
        return item.clone(caps=caps, metadata={"gate_condition": condition})

    def _check_condition(self, item: DataItem, condition: Any) -> bool:
        if condition is None or condition is True:
            return True
        if isinstance(condition, str) and condition.strip().lower() in {"", "always", "true"}:
            return True
        if isinstance(condition, dict):
            # Supported safe forms:
            # {"annotation_label": "person", "min_confidence": 0.5}
            # {"data_key": "value", "op": ">=", "value": 5}
            if "annotation_label" in condition:
                label = str(condition["annotation_label"]).lower()
                min_conf = _safe_float(condition.get("min_confidence"), 0.0)
                return any(_annotation_label(a).lower() == label and _safe_float(a.get("confidence", 1.0), 1.0) >= min_conf for a in item.annotations)
            if "data_key" in condition:
                data = item.data if isinstance(item.data, dict) else {}
                lhs = _safe_float(data.get(condition["data_key"]), 0.0)
                rhs = _safe_float(condition.get("value"), 0.0)
                op = condition.get("op", ">=")
                return {">=": lhs >= rhs, ">": lhs > rhs, "<=": lhs <= rhs, "<": lhs < rhs, "==": lhs == rhs}.get(op, False)
        if isinstance(condition, str):
            # Convenience conditions without unsafe eval.
            if condition.startswith("annotation:"):
                label = condition.split(":", 1)[1].strip().lower()
                return any(_annotation_label(a).lower() == label for a in item.annotations)
            if condition.startswith("data_exists:") and isinstance(item.data, dict):
                return condition.split(":", 1)[1].strip() in item.data
        return False


@register
class JoinFuseOperator(Operator):
    operator_id = "op.join_fuse"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        # Single-input fallback: pack current observations into a fused event record.
        join_type = self.params.get("join_type", "OR")
        if isinstance(item.data, list):
            inputs = item.data
        else:
            inputs = [item.to_jsonable(include_payload=False)]
        labels = sorted({_annotation_label(a) for a in item.annotations if _annotation_label(a)})
        data = {
            "join_type": join_type,
            "inputs": inputs,
            "labels": labels,
            "timestamp_ms": item.metadata.get("timestamp_ms"),
        }
        anns = list(item.annotations) + [{"label": "fused_event", "confidence": 1.0, "source": "join_fuse"}]
        return semantic_item("fused_event_record", "application/x-fused-event", data, item, anns)


@register
class RoutePublishOperator(Operator):
    operator_id = "op.route_publish"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        protocol = self.params.get("protocol", "return")
        destination = self.params.get("destination") or self.params.get("path")
        if protocol == "file" or destination:
            out_path = Path(str(destination or "pipeline_output.json"))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if cap_type(item.caps).startswith("image/") and isinstance(item.data, np.ndarray):
                save_image(item.data, out_path)
            elif cap_type(item.caps).startswith("audio/") and isinstance(item.data, np.ndarray):
                save_audio(out_path, item.data, int(item.metadata.get("sample_rate", 16000)))
            else:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(item.to_jsonable(include_payload=True), f, indent=2)
            return item.clone(metadata={"published_to": str(out_path)})
        if protocol in {"REST", "http", "https"} and self.params.get("url"):
            try:
                import requests  # type: ignore

                resp = requests.post(self.params["url"], json=item.to_jsonable(include_payload=True), timeout=float(self.params.get("timeout", 5)))
                return item.clone(metadata={"publish_status_code": resp.status_code})
            except Exception as e:
                return item.clone(metadata={"publish_error": repr(e)})
        return item.clone(caps={"semantic_type": "external/application-output", "schema": "output_to_application"})


@register
class SchemaAdapterOperator(Operator):
    operator_id = "op.schema_adapter"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        schema = self.params.get("target_schema") or item.caps.get("schema") or "app_accepted_schema"
        semantic_type = self.params.get("target_semantic_type") or item.caps.get("semantic_type") or item.caps.get("media_type") or "application-defined"
        data = self._adapt_data(schema, item)
        return DataItem(caps={"semantic_type": semantic_type, "schema": schema}, data=data, annotations=list(item.annotations), metadata=dict(item.metadata))

    def _adapt_data(self, schema: str, item: DataItem) -> Any:
        if schema in {"occupancy_count", "room_occupied"}:
            count = None
            if isinstance(item.data, dict):
                count = item.data.get("count") or item.data.get("occupancy_count")
                if count is None and "occupied" in item.data:
                    return {"occupied": bool(item.data["occupied"]), "count": int(bool(item.data["occupied"]))}
            count = int(count or sum(1 for a in item.annotations if _annotation_label(a).lower() == "person"))
            return {"occupied": count > 0, "count": count}
        if schema == "keyword_or_intent":
            if isinstance(item.data, dict):
                return {"intent": item.data.get("intent"), "keyword": item.data.get("keyword"), "confidence": item.data.get("confidence", 1.0)}
            return {"intent": str(item.data), "confidence": 1.0}
        if schema in {"sound_event_label", "activity_label", "fall_or_safety_event", "person_at_door_or_intrusion_event"}:
            if isinstance(item.data, dict):
                return item.data
            labels = [_annotation_label(a) for a in item.annotations if _annotation_label(a)]
            return {"labels": labels, "confidence": max([_safe_float(a.get("confidence"), 0.0) for a in item.annotations] or [0.0])}
        return item.data


# ---------------------------------------------------------------------------
# Vision inference / transformations
# ---------------------------------------------------------------------------


@register
class PersonObjectDetectorOperator(Operator):
    operator_id = "op.person_object_detector"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        frames = _iter_frames(item)
        if not frames:
            return semantic_item("object_detections", "application/x-detections", {"detections": []}, item, [])
        image = frames[0]
        detections: List[Dict[str, Any]] = []
        classes = [str(c).lower() for c in ensure_list(self.params.get("classes") or ["person", "face"])]
        if "face" in classes or "person" in classes:
            detections.extend(self._detect_faces(image))
        if "person" in classes:
            detections.extend(self._detect_people(image))
        # Remove boxes if requested, while preserving count.
        if self.params.get("include_boxes", True) is False:
            detections = [{k: v for k, v in d.items() if k != "bbox"} for d in detections]
        data = {"detections": detections, "count_by_label": _count_by_label(detections)}
        return semantic_item("object_detections", "application/x-detections", data, item, detections)

    def _detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        backend = str(self.params.get("backend") or self.params.get("face_backend") or "auto").lower()
        if backend in {"auto", "dlib"}:
            faces = self._detect_faces_dlib(image)
            if faces or backend == "dlib":
                return faces
        if cv2 is None:
            return []
        gray = _cv_gray(image)
        cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            return []
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
        return [{"label": "face", "bbox": [int(x), int(y), int(x + w), int(y + h)], "confidence": 0.7, "source": "opencv_haar"} for (x, y, w, h) in faces]

    def _detect_faces_dlib(self, image: np.ndarray) -> List[Dict[str, Any]]:
        try:
            import dlib  # type: ignore

            detector = dlib.get_frontal_face_detector()
            gray = _cv_gray(image)
            rects = detector(gray, int(self.params.get("dlib_upsample", 0)))
            out = []
            for r in rects:
                out.append({
                    "label": "face",
                    "bbox": [int(r.left()), int(r.top()), int(r.right()), int(r.bottom())],
                    "confidence": 0.75,
                    "source": "dlib_hog",
                })
            return out
        except Exception:
            return []

    def _detect_people(self, image: np.ndarray) -> List[Dict[str, Any]]:
        backend = str(self.params.get("backend") or self.params.get("person_backend") or "auto").lower()
        if backend in {"auto", "yolo", "ultralytics"}:
            people = self._detect_people_yolo(image, force=(backend in {"yolo", "ultralytics"}))
            if people or backend in {"yolo", "ultralytics"}:
                return people
        if cv2 is None:
            return []
        # OpenCV's default HOG person detector expects at least a 64x128-ish image;
        # on some OpenCV builds tiny inputs can segfault rather than raising.
        if image.shape[0] < 128 or image.shape[1] < 64:
            return []
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            # HOG expects BGR/gray but works well enough with RGB arrays.
            rects, weights = hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)
            out = []
            for (x, y, w, h), conf in zip(rects, weights):
                out.append({"label": "person", "bbox": [int(x), int(y), int(x + w), int(y + h)], "confidence": float(conf), "source": "opencv_hog"})
            return out
        except Exception:
            return []

    def _detect_people_yolo(self, image: np.ndarray, force: bool = False) -> List[Dict[str, Any]]:
        # Optional higher-quality backend. In auto mode, only use a local model path
        # to avoid surprise downloads during offline deployments/tests. Set
        # backend="yolo" or yolo_auto_download=True to allow Ultralytics defaults.
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception:
            return []
        model_ref = self.params.get("yolo_model") or self.params.get("model")
        if not model_ref:
            if force or self.params.get("yolo_auto_download"):
                model_ref = "yolov8n.pt"
            else:
                return []
        try:
            model = YOLO(str(model_ref))
            # Ultralytics accepts RGB numpy arrays. Use verbose=False to keep tests quiet.
            results = model.predict(image, verbose=False, conf=float(self.params.get("confidence", 0.25)))
            out: List[Dict[str, Any]] = []
            for res in results:
                names = getattr(res, "names", {}) or getattr(model, "names", {}) or {}
                boxes = getattr(res, "boxes", None)
                if boxes is None:
                    continue
                for b in boxes:
                    cls_id = int(b.cls[0]) if getattr(b, "cls", None) is not None else -1
                    label = str(names.get(cls_id, cls_id))
                    if label != "person":
                        continue
                    xyxy = b.xyxy[0].detach().cpu().numpy().tolist()
                    conf = float(b.conf[0]) if getattr(b, "conf", None) is not None else 0.0
                    out.append({
                        "label": "person",
                        "bbox": [int(round(v)) for v in xyxy[:4]],
                        "confidence": conf,
                        "source": "ultralytics_yolo",
                    })
            return out
        except Exception:
            return []


@register
class PoseExtractorOperator(Operator):
    operator_id = "op.pose_extractor"

    # Cache YOLO models across operator instances so a streaming pipeline does not
    # reload weights for every item.
    _YOLO_MODEL_CACHE: Dict[str, Any] = {}

    def apply(self, item: DataItem) -> Optional[DataItem]:
        """Extract pose keypoints.

        Default backend: Ultralytics YOLO pose, producing the same COCO-17 style
        temporal keypoint representation used by the LE2I/ImViA preprocessing
        script (`keypoints` with shape T x 17 x 3).  MediaPipe remains available
        with backend="mediapipe" for compatibility/debugging, but is no longer
        the default because the fall downstream model is trained on YOLO/COCO17
        pose sequences.
        """
        backend = str(self.params.get("backend") or self.params.get("pose_backend") or "yolo").lower()
        frames = self._frames_for_pose(item)

        if backend in {"mediapipe", "mp"}:
            return self._apply_mediapipe(item, frames)

        # Default path: YOLO pose.  If Ultralytics/model loading fails, optionally
        # fall back to MediaPipe only when explicitly requested.  Otherwise return
        # a typed empty/failed YOLO pose object so failures are visible downstream.
        yolo_result = self._apply_yolo_pose(item, frames)
        if yolo_result is not None:
            return yolo_result
        if bool(self.params.get("fallback_to_mediapipe", False)):
            return self._apply_mediapipe(item, frames)
        return self._empty_yolo_pose_item(item, warning="YOLO pose backend unavailable or failed")

    def _frames_for_pose(self, item: DataItem) -> List[np.ndarray]:
        frames = _iter_frames(item)
        if frames:
            return frames
        data = item.data
        if isinstance(data, dict):
            # Common manifest-like runtime forms.
            if isinstance(data.get("frames"), list) and all(isinstance(x, np.ndarray) for x in data["frames"]):
                return list(data["frames"])
            for key in ["image_path", "frame_path"]:
                path = data.get(key)
                if path:
                    img = self._read_image_path(Path(str(path)))
                    return [] if img is None else [img]
            if data.get("frame_dir"):
                return self._read_frame_dir(Path(str(data["frame_dir"])))
            if data.get("video_path"):
                return self._read_video_path(Path(str(data["video_path"])))
        return []

    def _stride_and_limit_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        stride = max(1, int(self.params.get("stride") or self.params.get("frame_stride") or 1))
        max_frames_raw = self.params.get("max_frames")
        max_frames = None if max_frames_raw in {None, "", 0, "0"} else int(max_frames_raw)
        selected = frames[::stride]
        if max_frames is not None:
            selected = selected[:max_frames]
        return selected

    def _read_image_path(self, path: Path) -> Optional[np.ndarray]:
        if cv2 is None or not path.exists():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_frame_dir(self, path: Path) -> List[np.ndarray]:
        if cv2 is None or not path.exists() or not path.is_dir():
            return []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = sorted([x for x in path.iterdir() if x.suffix.lower() in exts])
        max_frames_raw = self.params.get("max_frames")
        max_frames = None if max_frames_raw in {None, "", 0, "0"} else int(max_frames_raw)
        stride = max(1, int(self.params.get("stride") or self.params.get("frame_stride") or 1))
        out: List[np.ndarray] = []
        for fp in files[::stride]:
            img = self._read_image_path(fp)
            if img is not None:
                out.append(img)
                if max_frames is not None and len(out) >= max_frames:
                    break
        return out

    def _read_video_path(self, path: Path) -> List[np.ndarray]:
        if cv2 is None or not path.exists():
            return []
        stride = max(1, int(self.params.get("stride") or self.params.get("frame_stride") or 1))
        max_frames_raw = self.params.get("max_frames")
        max_frames = None if max_frames_raw in {None, "", 0, "0"} else int(max_frames_raw)
        cap = cv2.VideoCapture(str(path))
        out: List[np.ndarray] = []
        idx = 0
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if max_frames is not None and len(out) >= max_frames:
                        break
                idx += 1
        finally:
            cap.release()
        return out

    def _default_yolo_device(self) -> str:
        requested = self.params.get("device") or self.params.get("pose_device")
        if requested is not None:
            return str(requested)
        try:
            import torch  # type: ignore
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _get_yolo_model(self):
        from ultralytics import YOLO  # type: ignore

        model_ref = (
            self.params.get("pose_model")
            or self.params.get("yolo_pose_model")
            or self.params.get("model")
            or "yolo11n-pose.pt"
        )
        key = str(model_ref)
        model = self._YOLO_MODEL_CACHE.get(key)
        if model is None:
            model = YOLO(key)
            self._YOLO_MODEL_CACHE[key] = model
        return model, key

    def _apply_yolo_pose(self, item: DataItem, frames: List[np.ndarray]) -> Optional[DataItem]:
        try:
            model, model_ref = self._get_yolo_model()
        except Exception as e:
            if bool(self.params.get("emit_warning_on_failure", True)):
                return self._empty_yolo_pose_item(item, warning=f"Could not load YOLO pose model: {e!r}")
            return None

        frames = self._stride_and_limit_frames(frames)
        device = self._default_yolo_device()
        keypoints: List[np.ndarray] = []
        pose_records: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []
        conf_threshold = float(self.params.get("confidence") or self.params.get("conf") or 0.25)

        for frame_idx, frame in enumerate(frames):
            try:
                results = model.predict(frame, verbose=False, device=device, conf=conf_threshold)
                k = None
                if results and getattr(results[0], "keypoints", None) is not None and len(results[0].keypoints) > 0:
                    k = results[0].keypoints.data.cpu().numpy()  # [N, 17, 3]
                if k is None or k.ndim != 3 or k.shape[1] == 0:
                    best = np.zeros((17, 3), dtype=np.float32)
                    confidence = 0.0
                else:
                    best_idx = int(np.nanargmax(k[:, :, 2].mean(axis=1)))
                    best = k[best_idx].astype(np.float32)
                    confidence = float(np.nanmean(best[:, 2]))
            except Exception as e:
                best = np.zeros((17, 3), dtype=np.float32)
                confidence = 0.0
                pose_records.append({
                    "frame_index": frame_idx,
                    "keypoints": best.tolist(),
                    "confidence": confidence,
                    "skeleton": "coco17",
                    "source": "ultralytics_yolo_pose",
                    "warning": repr(e),
                })
                keypoints.append(best)
                annotations.append({"label": "pose", "keypoints": best.tolist(), "confidence": confidence, "source": "ultralytics_yolo_pose"})
                continue

            keypoints.append(best)
            pose_records.append({
                "frame_index": frame_idx,
                "keypoints": best.tolist(),
                "confidence": confidence,
                "skeleton": "coco17",
                "source": "ultralytics_yolo_pose",
            })
            annotations.append({"label": "pose", "keypoints": best.tolist(), "confidence": confidence, "source": "ultralytics_yolo_pose"})

        if not keypoints:
            empty = np.zeros((1, 17, 3), dtype=np.float32)
            keypoints_array = empty
            pose_records = [{
                "frame_index": 0,
                "keypoints": empty[0].tolist(),
                "confidence": 0.0,
                "skeleton": "coco17",
                "source": "ultralytics_yolo_pose",
                "warning": "no frames available",
            }]
        else:
            keypoints_array = np.stack(keypoints, axis=0).astype(np.float32)  # [T, 17, 3]

        caps = {
            "semantic_type": "application/x-pose-keypoints",
            "schema": self.params.get("output_schema") or "yolo_coco17_pose_sequence",
            "properties": {
                "skeleton": "coco17",
                "backend": "yolo",
                "shape": list(keypoints_array.shape),
            },
        }
        data = {
            "keypoints": keypoints_array.tolist(),
            "keypoints_shape": list(keypoints_array.shape),
            "poses": pose_records,
            "skeleton": "coco17",
            "backend": "yolo",
            "pose_model": model_ref,
            "device": device,
            "stride": int(self.params.get("stride") or self.params.get("frame_stride") or 1),
        }
        return DataItem(caps=caps, data=data, annotations=annotations, metadata={**dict(item.metadata), "pose_backend": "yolo", "pose_model": model_ref})

    def _empty_yolo_pose_item(self, item: DataItem, warning: str = "") -> DataItem:
        keypoints_array = np.zeros((1, 17, 3), dtype=np.float32)
        caps = {
            "semantic_type": "application/x-pose-keypoints",
            "schema": self.params.get("output_schema") or "yolo_coco17_pose_sequence",
            "properties": {"skeleton": "coco17", "backend": "yolo", "shape": list(keypoints_array.shape)},
        }
        data = {
            "keypoints": keypoints_array.tolist(),
            "keypoints_shape": list(keypoints_array.shape),
            "poses": [{"frame_index": 0, "keypoints": keypoints_array[0].tolist(), "confidence": 0.0, "skeleton": "coco17", "source": "unavailable", "warning": warning}],
            "skeleton": "coco17",
            "backend": "yolo",
            "warning": warning,
        }
        return DataItem(caps=caps, data=data, annotations=[], metadata={**dict(item.metadata), "pose_backend": "yolo", "pose_warning": warning})

    def _apply_mediapipe(self, item: DataItem, frames: List[np.ndarray]) -> Optional[DataItem]:
        if not frames:
            return semantic_item("pose_keypoints", "application/x-pose-keypoints", {"poses": []}, item, [])
        image = frames[0]
        poses = self._mediapipe_pose(image)
        anns = [{"label": "pose", "keypoints": p.get("keypoints"), "confidence": p.get("confidence", 0.0), "source": p.get("source", "pose_extractor")} for p in poses]
        out = semantic_item("pose_keypoints", "application/x-pose-keypoints", {"poses": poses, "skeleton": "mediapipe33", "backend": "mediapipe"}, item, anns)
        out.caps = merge_caps(out.caps, {"schema": "mediapipe33_pose", "properties": {"skeleton": "mediapipe33", "backend": "mediapipe"}})
        return out

    def _mediapipe_pose(self, image: np.ndarray) -> List[Dict[str, Any]]:
        try:
            import mediapipe as mp  # type: ignore

            mp_pose = mp.solutions.pose
            with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
                res = pose.process(image if image.ndim == 3 else np.stack([image] * 3, axis=-1))
            if not res.pose_landmarks:
                return []
            kps = []
            vis = []
            for lm in res.pose_landmarks.landmark:
                kps.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z), "visibility": float(lm.visibility)})
                vis.append(float(lm.visibility))
            return [{"keypoints": kps, "confidence": float(np.mean(vis)), "skeleton": "mediapipe33", "source": "mediapipe"}]
        except Exception as e:
            return [{"keypoints": [], "confidence": 0.0, "skeleton": self.params.get("skeleton", "mediapipe33"), "source": "unavailable", "warning": repr(e)}]

@register
class OcrScreenDetectorOperator(Operator):
    operator_id = "op.ocr_screen_detector"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        image = _as_image(item.data)
        regions: List[Dict[str, Any]] = []
        if image is not None:
            regions = self._ocr_regions(image)
            if not regions:
                regions = self._bright_rect_regions(image)
        if self.params.get("emit_text") is False:
            for r in regions:
                r.pop("text", None)
        data = {"regions": regions}
        return semantic_item("ocr_or_screen_regions", "application/x-text-regions", data, item, regions)

    def _ocr_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        try:
            import pytesseract  # type: ignore

            d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            out = []
            n = len(d.get("text", []))
            for i in range(n):
                text = str(d["text"][i]).strip()
                conf = _safe_float(d.get("conf", [0] * n)[i], -1.0)
                if text and conf > 30:
                    x, y, w, h = [int(d[k][i]) for k in ["left", "top", "width", "height"]]
                    out.append({"label": "screen", "bbox": [x, y, x + w, y + h], "text": text, "confidence": conf / 100.0, "source": "pytesseract"})
            return out
        except Exception:
            return []

    def _bright_rect_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if cv2 is None:
            return []
        gray = _cv_gray(image)
        # Crude fallback for screens/documents: large high-contrast rectangles.
        edges = cv2.Canny(gray, 60, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]
        out = []
        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            area = ww * hh
            if area > 0.02 * w * h and ww > 40 and hh > 20:
                out.append({"label": "screen", "bbox": [x, y, x + ww, y + hh], "confidence": 0.4, "source": "contour_fallback"})
        return out[:20]


@register
class RegionSelectCropOperator(Operator):
    operator_id = "op.region_select_crop"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        image = _as_image(item.data)
        if image is None:
            return item.clone(caps=merge_caps(item.caps, {"properties": {"field_of_view_minimized": True}}))
        h, w = image.shape[:2]
        roi = self._select_roi(item, w, h)
        x1, y1, x2, y2 = roi
        crop = image[y1:y2, x1:x2].copy()
        caps = merge_caps(item.caps, {"properties": {"field_of_view_minimized": True}})
        return item.clone(caps=caps, data=_replace_image_data(item.data, crop), metadata={"crop_bbox": [x1, y1, x2, y2]})

    def _select_roi(self, item: DataItem, w: int, h: int) -> Tuple[int, int, int, int]:
        if self.params.get("static_roi"):
            return _box_xyxy({"bbox": self.params["static_roi"]}, w, h)
        label = self.params.get("region_label") or self.params.get("target") or "person"
        anns = _region_annotations(item, [str(label), "person", "face", "screen"])
        if anns:
            boxes = [_box_xyxy(a, w, h) for a in anns]
            x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes); x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
            pad = int(self.params.get("padding_px", 10))
            return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
        return 0, 0, w, h


@register
class RegionMaskBlurOperator(Operator):
    operator_id = "op.region_mask_blur"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        image = _as_image(item.data)
        if image is None:
            return item.clone(caps=_redacted_media_caps(item.caps))
        out = image.copy()
        h, w = out.shape[:2]
        targets = ensure_list(self.params.get("targets") or self.params.get("target") or "face")
        targets = [str(t).lower() for t in targets]
        anns = _region_annotations(item, targets)
        if not anns and any(t in {"face", "body", "person"} for t in targets):
            # Detect regions internally if the planner did not include a detector stage.
            det = PersonObjectDetectorOperator(classes=["person", "face"])(item)
            if det:
                anns = _region_annotations(det, targets + ["person"])
        if "background" in targets:
            anns = self._inverse_foreground_regions(item, w, h, anns)
        for ann in anns:
            x1, y1, x2, y2 = _box_xyxy(ann, w, h)
            out[y1:y2, x1:x2] = self._redact_patch(out[y1:y2, x1:x2])
        caps = _redacted_media_caps(item.caps)
        return item.clone(caps=caps, data=_replace_image_data(item.data, out), metadata={"redacted_targets": targets, "redacted_regions": len(anns)})

    def _redact_patch(self, patch: np.ndarray) -> np.ndarray:
        if patch.size == 0:
            return patch
        method = self.params.get("method", "blur")
        if method == "mask":
            return np.zeros_like(patch)
        if method == "pixelate" and cv2 is not None:
            h, w = patch.shape[:2]
            block = max(2, int(self.params.get("strength", 12)))
            small = cv2.resize(patch, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        if cv2 is not None:
            k = max(3, int(self.params.get("strength", 25)) | 1)
            return cv2.GaussianBlur(patch, (k, k), 0)
        return np.zeros_like(patch)

    def _inverse_foreground_regions(self, item: DataItem, w: int, h: int, foreground: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Simple fallback: redact the whole image except foreground boxes by returning
        # one whole-image region. A production implementation would create a mask.
        return [{"label": "background", "bbox": [0, 0, w, h], "confidence": 1.0}]


# ---------------------------------------------------------------------------
# Audio inference / transformations
# ---------------------------------------------------------------------------


@register
class AudioLevelExtractorOperator(Operator):
    operator_id = "op.audio_level_extractor"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        audio, sr = _audio_array(item)
        if audio is None:
            return semantic_item("decibel_level_duration", "application/x-decibel-level", {"dbfs": None, "duration_s": None}, item, [])
        win_ms = int(self.params.get("window_ms") or 1000)
        data = {
            "dbfs": _rms_db(audio),
            "duration_s": float(len(audio)) / float(sr),
            "window_ms": win_ms,
        }
        ann = {"label": "decibel_level", "value": data["dbfs"], "confidence": 1.0}
        return semantic_item("decibel_level_duration", "application/x-decibel-level", data, item, [ann])


@register
class SpeechSoundClassifierOperator(Operator):
    operator_id = "op.speech_sound_classifier"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        audio, sr = _audio_array(item)
        labels: List[Dict[str, Any]] = []
        if audio is not None:
            labels = self._heuristic_labels(audio, sr)
        if self.params.get("speech_flag_only"):
            has_speech = any(l["label"] == "speech" for l in labels)
            labels = [{"label": "speech" if has_speech else "non_speech", "confidence": max([l.get("confidence", 0.0) for l in labels] or [0.5])}]
        allowed = {str(x).lower() for x in ensure_list(self.params.get("labels")) if x}
        if allowed:
            labels = [l for l in labels if str(l.get("label", "")).lower() in allowed]
        data = {"labels": labels, "top_label": labels[0]["label"] if labels else None}
        anns = [{"label": l["label"], "confidence": l.get("confidence", 0.0), "source": "heuristic_audio"} for l in labels]
        return semantic_item("sound_event_label", "application/x-sound-event-label", data, item, anns)

    def _heuristic_labels(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        db = _rms_db(audio)
        zcr = _zero_crossing_rate(audio)
        labels: List[Dict[str, Any]] = []
        if db < -45:
            labels.append({"label": "silence", "confidence": 0.8})
        else:
            # Very rough stand-ins for real classifiers such as YAMNet.
            if 0.02 <= zcr <= 0.18 and -45 < db < -5:
                labels.append({"label": "speech", "confidence": 0.55})
            if zcr > 0.22 and db > -25:
                labels.append({"label": "glass_break_or_alarm", "confidence": 0.45})
            if db > -12:
                labels.append({"label": "loud_noise", "confidence": 0.6})
            if not labels:
                labels.append({"label": "ambient_sound", "confidence": 0.5})
        return sorted(labels, key=lambda x: x.get("confidence", 0.0), reverse=True)


@register
class KeywordIntentExtractorOperator(Operator):
    operator_id = "op.keyword_intent_extractor"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        transcript = ""
        if isinstance(item.data, dict):
            transcript = str(item.data.get("transcript") or item.data.get("text") or "")
        elif isinstance(item.data, str):
            transcript = item.data
        allowed = ensure_list(self.params.get("allowed_intents") or self.params.get("keywords") or [])
        result = self._match_intent(transcript, allowed)
        return semantic_item("keyword_or_intent", "application/x-command-intent", result, item, [{"label": "command_intent", **result}])

    def _match_intent(self, transcript: str, allowed: Sequence[Any]) -> Dict[str, Any]:
        toks = set(_simple_tokenize(transcript))
        if not allowed:
            return {"intent": None, "keyword": None, "confidence": 0.0, "transcript_available": bool(transcript)}
        for intent in allowed:
            if isinstance(intent, dict):
                name = str(intent.get("intent") or intent.get("name"))
                kws = [str(x).lower() for x in ensure_list(intent.get("keywords") or name)]
            else:
                name = str(intent)
                kws = [name.lower()]
            for kw in kws:
                kw_toks = set(_simple_tokenize(kw))
                if kw_toks and kw_toks.issubset(toks):
                    return {"intent": name, "keyword": kw, "confidence": 0.9, "transcript_available": bool(transcript)}
        return {"intent": None, "keyword": None, "confidence": 0.0, "transcript_available": bool(transcript)}


@register
class SpeechContentRemovalOperator(Operator):
    operator_id = "op.speech_content_removal"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        mode = self.params.get("mode", "remove_speech_segments")
        if isinstance(item.data, str) or (isinstance(item.data, dict) and ("transcript" in item.data or "text" in item.data)):
            text = item.data if isinstance(item.data, str) else str(item.data.get("transcript") or item.data.get("text") or "")
            if mode == "intent_only":
                return KeywordIntentExtractorOperator(allowed_intents=self.params.get("allowed_intents") or [])(DataItem(caps={"semantic_type": "application/x-transcript"}, data=text, metadata=dict(item.metadata)))
            redacted = self._redact_text(text)
            return semantic_item("redacted_transcript", "application/x-redacted-transcript", {"redacted_text": redacted}, item, [{"label": "redacted_transcript"}])
        audio, sr = _audio_array(item)
        if audio is None:
            return item.clone(caps={"media_type": "audio/x-filtered", "properties": {"speech_content_removed": True}})
        if mode == "voice_anonymize":
            filtered = self._voice_anonymize(audio, sr)
        else:
            filtered = self._mute_speech_like_segments(audio, sr)
        return item.clone(caps={"media_type": "audio/x-filtered", "properties": {"speech_content_removed": True}}, data=filtered, metadata={"sample_rate": sr})

    def _redact_text(self, text: str) -> str:
        policy = self.params.get("redaction_policy", "numbers_names")
        out = text
        out = re.sub(r"\b\d{2,}\b", "[NUMBER]", out)
        out = re.sub(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", "[NAME]", out)
        return out

    def _mute_speech_like_segments(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Lightweight VAD: mute frames with speech-like energy and zcr.
        x = audio.copy()
        frame = max(1, int(sr * 0.03))
        hop = frame
        for start in range(0, len(x), hop):
            seg = x[start:start + frame]
            if seg.size == 0:
                continue
            db = _rms_db(seg); zcr = _zero_crossing_rate(seg)
            if -45 < db < -5 and 0.02 <= zcr <= 0.18:
                x[start:start + frame] = 0
        return x

    def _voice_anonymize(self, audio: np.ndarray, sr: int) -> np.ndarray:
        try:
            import librosa  # type: ignore

            return librosa.effects.pitch_shift(audio.astype(np.float32), sr=sr, n_steps=float(self.params.get("pitch_steps", 2.0)))
        except Exception:
            # Fallback: reverse short chunks. This is not strong anonymization, but it
            # avoids returning the untouched waveform when optional libraries are absent.
            x = audio.copy()
            chunk = max(1, int(sr * 0.05))
            for i in range(0, len(x), chunk):
                x[i:i + chunk] = x[i:i + chunk][::-1]
            return x


# ---------------------------------------------------------------------------
# Multimodal AV packaging and component-wise minimization
# ---------------------------------------------------------------------------


def _av_base_caps(existing: Dict[str, Any], **flags: Any) -> Dict[str, Any]:
    """Return caps for a YouHome-compatible audio/video sample."""
    props = dict(existing.get("properties", {}) or {})
    props.update({k: v for k, v in flags.items() if v is not None})
    props.setdefault("components", [
        {"media_type": "image/x-raw", "schema": "youhome_video_frames", "role": "visual"},
        {"media_type": "audio/x-raw", "schema": "youhome_audio_waveform", "role": "audio"},
    ])
    props.setdefault("sensorPrimitive", ["image_frame", "audio_waveform"])
    return merge_caps(existing, {
        "semantic_type": "application/x-youhome-av-sample",
        "schema": "youhome_av_manifest_or_sample",
        "properties": props,
    })


def _copy_av_data(data: Any) -> Any:
    if isinstance(data, dict):
        out: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                out[k] = v.copy()
            elif isinstance(v, list):
                out[k] = [x.copy() if isinstance(x, np.ndarray) else x for x in v]
            else:
                out[k] = v
        return out
    return data


def _av_frame_key(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    for key in ["frames", "video_frames", "images", "image_frames"]:
        if isinstance(data.get(key), list) and all(isinstance(x, np.ndarray) for x in data.get(key, [])):
            return key
    for key in ["frame", "image"]:
        if isinstance(data.get(key), np.ndarray):
            return key
    return None


def _av_get_frames(data: Any) -> List[np.ndarray]:
    key = _av_frame_key(data)
    if key is None or not isinstance(data, dict):
        return []
    value = data.get(key)
    if isinstance(value, list):
        return [x for x in value if isinstance(x, np.ndarray)]
    if isinstance(value, np.ndarray):
        return [value]
    return []


def _av_set_frames(data: Any, frames: List[np.ndarray]) -> Any:
    out = _copy_av_data(data)
    if not isinstance(out, dict):
        return out
    key = _av_frame_key(out) or "frames"
    if key in {"frame", "image"}:
        out[key] = frames[0] if frames else out.get(key)
    else:
        out[key] = frames
    return out


def _av_audio_key(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    for key in ["audio", "waveform", "audio_waveform"]:
        if isinstance(data.get(key), np.ndarray):
            return key
    return None


def _av_get_audio(data: Any, metadata: Dict[str, Any]) -> Tuple[Optional[np.ndarray], int]:
    key = _av_audio_key(data)
    if key is None or not isinstance(data, dict):
        return None, int(metadata.get("sample_rate") or metadata.get("sr") or 16000)
    return np.asarray(data[key], dtype=np.float32), int(data.get("sample_rate") or metadata.get("sample_rate") or metadata.get("sr") or 16000)


def _av_set_audio(data: Any, audio: np.ndarray, sr: int) -> Any:
    out = _copy_av_data(data)
    if not isinstance(out, dict):
        return out
    key = _av_audio_key(out) or "audio"
    out[key] = audio
    out["sample_rate"] = sr
    return out


@register
class YouHomeAVAdapterOperator(Operator):
    operator_id = "op.youhome_av_adapter"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        """Normalize a manifest/sample-like item into the YouHome AV sample cap.

        This operator is an interface adapter: it does not infer activities or
        alter content. It is useful when the runtime item already represents a
        synchronized image/audio sample but its caps have not yet been normalized.
        """
        caps = _av_base_caps(item.caps, av_synchronized=True, youhome_av_compatible=True)
        return item.clone(caps=caps, metadata={"av_adapter": "youhome_av_manifest_or_sample"})


@register
class AVVisualRedactOperator(Operator):
    operator_id = "op.av_visual_redact"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        frames = _av_get_frames(item.data)
        caps = _av_base_caps(
            item.caps,
            visual_redacted=True,
            redacted=True,
            youhome_av_compatible=True,
        )
        if not frames:
            return item.clone(caps=caps, metadata={"av_visual_redact": "metadata_only_no_frame_payload"})
        targets = self.params.get("targets") or self.params.get("target") or ["face", "screen"]
        redact = RegionMaskBlurOperator(targets=targets, method=self.params.get("method", "blur"), strength=self.params.get("strength", 25))
        redacted_frames: List[np.ndarray] = []
        for frame in frames:
            frame_item = DataItem(
                caps={"media_type": "image/x-raw", "schema": "raw_image_frame"},
                data=frame,
                annotations=list(item.annotations),
                metadata=dict(item.metadata),
            )
            out = redact.apply(frame_item)
            redacted_frames.append(out.data if out is not None and isinstance(out.data, np.ndarray) else frame)
        data = _av_set_frames(item.data, redacted_frames)
        return item.clone(caps=caps, data=data, metadata={"av_visual_redacted": True, "redacted_frames": len(redacted_frames)})


@register
class AVAudioSpeechFilterOperator(Operator):
    operator_id = "op.av_audio_speech_filter"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        audio, sr = _av_get_audio(item.data, item.metadata)
        caps = _av_base_caps(
            item.caps,
            speech_content_removed=True,
            audio_filtered=True,
            youhome_av_compatible=True,
        )
        if audio is None:
            return item.clone(caps=caps, metadata={"av_audio_speech_filter": "metadata_only_no_audio_payload"})
        filt = SpeechContentRemovalOperator(mode=self.params.get("mode", "remove_speech_segments"))
        audio_item = DataItem(caps={"media_type": "audio/x-raw", "schema": "raw_audio_waveform"}, data=audio, metadata={**dict(item.metadata), "sample_rate": sr})
        out = filt.apply(audio_item)
        filtered = out.data if out is not None and isinstance(out.data, np.ndarray) else audio
        data = _av_set_audio(item.data, filtered, sr)
        return item.clone(caps=caps, data=data, metadata={"av_audio_speech_filtered": True, "sample_rate": sr})


@register
class AVAudioSilenceOperator(Operator):
    operator_id = "op.av_audio_silence"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        audio, sr = _av_get_audio(item.data, item.metadata)
        caps = _av_base_caps(
            item.caps,
            audio_silenced=True,
            speech_content_removed=True,
            speaker_identity_removed=True,
            youhome_av_compatible=True,
        )
        if audio is None:
            return item.clone(caps=caps, metadata={"av_audio_silence": "metadata_only_no_audio_payload"})
        data = _av_set_audio(item.data, np.zeros_like(audio), sr)
        return item.clone(caps=caps, data=data, metadata={"av_audio_silenced": True, "sample_rate": sr})


@register
class AVVideoBlackoutOperator(Operator):
    operator_id = "op.av_video_blackout"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        frames = _av_get_frames(item.data)
        caps = _av_base_caps(
            item.caps,
            visual_suppressed=True,
            video_blackout=True,
            youhome_av_compatible=True,
        )
        if not frames:
            return item.clone(caps=caps, metadata={"av_video_blackout": "metadata_only_no_frame_payload"})
        blanked = [np.zeros_like(frame) for frame in frames]
        data = _av_set_frames(item.data, blanked)
        return item.clone(caps=caps, data=data, metadata={"av_video_blackout": True, "blanked_frames": len(blanked)})


# ---------------------------------------------------------------------------
# Derived representations and minimization
# ---------------------------------------------------------------------------


@register
class OccupancyDeriverOperator(Operator):
    operator_id = "op.occupancy_deriver"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        count = self._derive_count(item)
        binary = bool(self.params.get("binary", False))
        if binary:
            data = {"occupied": count > 0, "count": count, "spatial_scope": self.params.get("spatial_scope", "room")}
            return semantic_item("room_occupied", "application/x-binary-occupancy", data, item, [{"label": "room_occupied", "value": count > 0, "confidence": 1.0}])
        data = {"count": count, "spatial_scope": self.params.get("spatial_scope", "room")}
        return semantic_item("occupancy_count", "application/x-occupancy-count", data, item, [{"label": "occupancy_count", "value": count, "confidence": 1.0}])

    def _derive_count(self, item: DataItem) -> int:
        if isinstance(item.data, dict):
            if "count" in item.data:
                return int(item.data["count"])
            if "detections" in item.data:
                return sum(1 for d in item.data["detections"] if str(d.get("label", "")).lower() in {"person", "face"})
            val = item.data.get("motion") or item.data.get("occupied") or item.data.get("value")
            if isinstance(val, bool):
                return int(val)
            if isinstance(val, (int, float)):
                threshold = _safe_float(self.params.get("threshold"), 0.5)
                return int(float(val) >= threshold)
        return sum(1 for a in item.annotations if _annotation_label(a).lower() in {"person", "face"})


@register
class ActivityEventClassifierOperator(Operator):
    operator_id = "op.activity_event_classifier"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        labels = [str(x).lower() for x in ensure_list(self.params.get("labels") or [])]
        result = self._classify(item, labels)
        schema = "fall_or_safety_event" if result.get("event_type") in {"fall", "safety_event", "glass_break_or_alarm", "alarm"} else "activity_label"
        semantic_type = "application/x-safety-event" if schema == "fall_or_safety_event" else "application/x-activity-label"
        return semantic_item(schema, semantic_type, result, item, [{"label": result.get("event_type") or result.get("activity") or "activity", "confidence": result.get("confidence", 0.0)}])

    def _classify(self, item: DataItem, labels: Sequence[str]) -> Dict[str, Any]:
        # From sound labels.
        if isinstance(item.data, dict):
            sound_labels = [str(l.get("label", "")) for l in item.data.get("labels", []) if isinstance(l, dict)]
            if any("glass" in l or "alarm" in l for l in sound_labels):
                return {"event_type": "glass_break_or_alarm", "confidence": 0.6, "source": "sound_event_label"}
            if any("speech" in l for l in sound_labels):
                return {"activity": "conversation_or_speech_present", "confidence": 0.5, "source": "sound_event_label"}
            # From pose keypoints.  Supports both the new YOLO/COCO17 sequence
            # output (keypoints: T x 17 x 3) and the legacy MediaPipe pose list.
            poses = item.data.get("poses")
            keypoints = item.data.get("keypoints")
            pose_obj: Any = None
            if keypoints is not None:
                pose_obj = {"keypoints": keypoints, "skeleton": item.data.get("skeleton", "coco17")}
            elif isinstance(poses, list) and poses:
                pose_obj = poses[0]
            if pose_obj is not None:
                fall = self._pose_looks_fallen(pose_obj)
                return {"event_type": "fall" if fall else "posture_activity", "confidence": 0.55 if fall else 0.4, "source": "pose"}
            if "detections" in item.data:
                count = sum(1 for d in item.data["detections"] if str(d.get("label", "")).lower() in {"person", "face"})
                return {"activity": "person_present" if count else "no_person", "count": count, "confidence": 0.6, "source": "detections"}
        # From raw audio fallback.
        if cap_type(item.caps).startswith("audio/"):
            sound = SpeechSoundClassifierOperator()(item)
            return self._classify(sound or item, labels)
        return {"activity": "unknown", "confidence": 0.0}

    def _pose_looks_fallen(self, pose: Dict[str, Any]) -> bool:
        kps = pose.get("keypoints") if isinstance(pose, dict) else pose

        # New YOLO/COCO17 representation: either [17,3] or [T,17,3].  Use the
        # first frame for this lightweight heuristic classifier.  The real LE2I
        # downstream model should consume the whole sequence.
        try:
            arr = np.asarray(kps, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] > 0:
                arr = arr[0]
            if arr.ndim == 2 and arr.shape[0] >= 5:
                conf = arr[:, 2] if arr.shape[1] >= 3 else np.ones((arr.shape[0],), dtype=np.float32)
                visible = arr[conf > 0.20]
                if visible.size == 0:
                    visible = arr
                xs = visible[:, 0]
                ys = visible[:, 1]
                width = float(np.nanmax(xs) - np.nanmin(xs))
                height = float(np.nanmax(ys) - np.nanmin(ys))
                return width > height * 1.3
        except Exception:
            pass

        # Legacy MediaPipe representation: list of dicts with normalized x/y and
        # visibility.  Retain the previous 33-landmark heuristic.
        kps_list = kps or []
        if len(kps_list) < 29:
            return False
        visible = [p for p in kps_list if isinstance(p, dict) and p.get("visibility", 1.0) > 0.3]
        if not visible:
            return False
        xs = [p["x"] for p in visible]
        ys = [p["y"] for p in visible]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return width > height * 1.3

@register
class AggregateGeneralizeOperator(Operator):
    operator_id = "op.aggregate_generalize"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        temporal = self.params.get("temporal_granularity_ms")
        spatial = self.params.get("spatial_granularity", "room")
        data = self._aggregate(item)
        data.update({"temporal_granularity_ms": temporal, "spatial_granularity": spatial, "category_granularity": self.params.get("category_granularity", "coarse")})
        return semantic_item("aggregate_summary", "application/x-aggregate", data, item, [{"label": "aggregate_summary", "confidence": 1.0}])

    def _aggregate(self, item: DataItem) -> Dict[str, Any]:
        if isinstance(item.data, dict):
            if "detections" in item.data:
                return {"count_by_label": _count_by_label(item.data["detections"])}
            if "labels" in item.data:
                return {"count_by_label": _count_by_label(item.data["labels"])}
            if "count" in item.data or "occupied" in item.data:
                return {"count": item.data.get("count"), "occupied": item.data.get("occupied")}
            if "dbfs" in item.data:
                thresholds = self.params.get("thresholds") or {"quiet": -45, "loud": -15}
                db = _safe_float(item.data["dbfs"], -120)
                if db >= _safe_float(thresholds.get("loud"), -15):
                    band = "loud"
                elif db <= _safe_float(thresholds.get("quiet"), -45):
                    band = "quiet"
                else:
                    band = "normal"
                return {"sound_level_band": band}
        return {"count_by_label": _count_by_label(item.annotations)}


@register
class DropDiscardOperator(Operator):
    operator_id = "op.drop_discard"

    def apply(self, item: DataItem) -> Optional[DataItem]:
        return None


def _count_by_label(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        label = _annotation_label(r) or "unknown"
        counts[label] = counts.get(label, 0) + 1
    return counts
