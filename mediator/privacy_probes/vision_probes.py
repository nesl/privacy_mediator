from __future__ import annotations
from typing import Any, Dict, List
from .schema import ProbeResult, init_residual_vector
from .utils import image_area, load_image_frames, try_import

def _risk_from_count(count: int, medium: int = 3, high: int = 8) -> str:
    if count <= 0: return "none"
    if count < medium: return "low"
    if count < high: return "medium"
    return "high"

def probe_face_detection(path: str, max_frames: int = 32) -> ProbeResult:
    probe_id = "vision.face_detection"
    cv2 = try_import("cv2")
    if cv2 is None:
        return ProbeResult.unavailable(probe_id, "vision", "opencv-python is not installed")
    frames = load_image_frames(path, max_frames=max_frames)
    if not frames:
        return ProbeResult(probe_id, "vision", "no_artifacts", init_residual_vector("none"), {"frames": 0}, {"name": "opencv_haar"})
    cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        return ProbeResult.unavailable(probe_id, "vision", "OpenCV Haar face cascade unavailable")
    detections = []
    total_faces = 0
    frames_with_face = 0
    max_area_ratio = 0.0
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
        total_faces += len(faces)
        if len(faces) > 0: frames_with_face += 1
        area = max(1, image_area(frame))
        for (x, y, w, h) in faces:
            ratio = (w * h) / float(area)
            max_area_ratio = max(max_area_ratio, ratio)
            detections.append({"frame": i, "x": int(x), "y": int(y), "w": int(w), "h": int(h), "area_ratio": ratio})
    residual = init_residual_vector("none")
    residual["face"] = _risk_from_count(total_faces, medium=2, high=6)
    if total_faces == 0:
        residual["identity"] = "none"
    elif max_area_ratio > 0.03 and frames_with_face >= 2:
        residual["identity"] = "medium"
    else:
        residual["identity"] = "low"
    return ProbeResult(
        probe_id, "vision", "ok", residual,
        {"frames": len(frames), "frames_with_face": frames_with_face, "total_faces": total_faces,
         "max_face_area_ratio": max_area_ratio, "detections_sample": detections[:10]},
        {"name": "opencv_haar", "cascade": "haarcascade_frontalface_default"},
    )

def probe_person_body_presence(path: str, max_frames: int = 32) -> ProbeResult:
    probe_id = "vision.person_body_presence"
    cv2 = try_import("cv2")
    if cv2 is None:
        return ProbeResult.unavailable(probe_id, "vision", "opencv-python is not installed")
    frames = load_image_frames(path, max_frames=max_frames)
    if not frames:
        return ProbeResult(probe_id, "vision", "no_artifacts", init_residual_vector("none"), {"frames": 0}, {"name": "opencv_hog"})
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    total_people = 0
    frames_with_person = 0
    max_area_ratio = 0.0
    boxes_sample = []
    for i, frame in enumerate(frames):
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        if len(boxes) > 0: frames_with_person += 1
        total_people += len(boxes)
        area = max(1, image_area(frame))
        for j, box in enumerate(boxes):
            x, y, w, h = [int(v) for v in box]
            max_area_ratio = max(max_area_ratio, (w * h) / float(area))
            if len(boxes_sample) < 10:
                boxes_sample.append({"frame": i, "x": x, "y": y, "w": w, "h": h, "weight": float(weights[j]) if j < len(weights) else None})
    residual = init_residual_vector("none")
    if total_people > 0:
        residual["body_shape"] = "medium" if max_area_ratio > 0.05 else "low"
        residual["clothing"] = "medium" if max_area_ratio > 0.05 else "low"
        residual["activity"] = "low"
        residual["aggregate_presence"] = "high"
        residual["co_presence"] = "medium" if total_people >= 2 else "low"
        residual["trajectory"] = "low" if frames_with_person >= 2 else "none"
    return ProbeResult(
        probe_id, "vision", "ok", residual,
        {"frames": len(frames), "frames_with_person": frames_with_person, "total_people_boxes": total_people,
         "max_person_area_ratio": max_area_ratio, "boxes_sample": boxes_sample},
        {"name": "opencv_hog_default_people_detector"},
    )

def probe_ocr_visible_text(path: str, max_frames: int = 16) -> ProbeResult:
    probe_id = "vision.ocr_visible_text"
    cv2 = try_import("cv2")
    frames = load_image_frames(path, max_frames=max_frames)
    if not frames:
        return ProbeResult(probe_id, "vision", "no_artifacts", init_residual_vector("none"), {"frames": 0}, {})
    easyocr = try_import("easyocr")
    pytesseract = try_import("pytesseract")
    texts: List[Dict[str, Any]] = []
    backend = {}
    errors: List[str] = []
    if easyocr is not None:
        backend = {"name": "easyocr"}
        try:
            reader = easyocr.Reader(["en"], gpu=False)
            for i, frame in enumerate(frames):
                for bbox, text, conf in reader.readtext(frame):
                    if text and float(conf) >= 0.2:
                        texts.append({"frame": i, "text": str(text), "confidence": float(conf)})
        except Exception as exc:
            errors.append(str(exc))
    elif pytesseract is not None and cv2 is not None:
        backend = {"name": "pytesseract"}
        try:
            for i, frame in enumerate(frames):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray)
                for w in [w.strip() for w in text.split() if w.strip()][:25]:
                    texts.append({"frame": i, "text": w, "confidence": None})
        except Exception as exc:
            errors.append(str(exc))
    else:
        return ProbeResult.unavailable(probe_id, "vision", "Neither easyocr nor pytesseract is installed", {"recommended": ["easyocr", "pytesseract"]})
    residual = init_residual_vector("none")
    n = len(texts)
    residual["visible_text"] = "none" if n == 0 else ("low" if n <= 3 else ("medium" if n <= 15 else "high"))
    sensitive_hits = [t for t in texts if any(ch.isdigit() for ch in t.get("text", "")) or len(t.get("text", "")) >= 8]
    if sensitive_hits:
        residual["identity"] = "low"
    return ProbeResult(
        probe_id, "vision", "ok" if not errors else "partial", residual,
        {"frames": len(frames), "num_text_items": n, "text_sample": texts[:20], "sensitive_like_text_items": sensitive_hits[:10]},
        backend, errors,
    )

def probe_scene_context(path: str, max_frames: int = 16) -> ProbeResult:
    probe_id = "vision.scene_object_context"
    ultralytics = try_import("ultralytics")
    if ultralytics is None:
        return ProbeResult.unavailable(probe_id, "vision", "ultralytics is not installed; install ultralytics for YOLO object/context probing")
    try:
        YOLO = getattr(ultralytics, "YOLO")
        model = YOLO("yolov8n.pt")
    except Exception as exc:
        return ProbeResult.unavailable(probe_id, "vision", f"Could not load YOLO backend: {exc}")
    frames = load_image_frames(path, max_frames=max_frames)
    if not frames:
        return ProbeResult(probe_id, "vision", "no_artifacts", init_residual_vector("none"), {"frames": 0}, {"name": "ultralytics_yolov8n"})
    object_counts: Dict[str, int] = {}
    for frame in frames:
        try:
            for r in model.predict(frame, verbose=False):
                names = getattr(r, "names", {})
                boxes = getattr(r, "boxes", None)
                if boxes is None: continue
                for cls in boxes.cls.tolist():
                    name = str(names.get(int(cls), int(cls)))
                    object_counts[name] = object_counts.get(name, 0) + 1
        except Exception:
            continue
    residual = init_residual_vector("none")
    if object_counts:
        residual["location"] = "low"
        residual["activity"] = "low"
    if any(k in object_counts for k in ["bed", "toilet", "laptop", "cell phone", "book", "tv"]):
        residual["location"] = "medium"
        if "laptop" in object_counts or "book" in object_counts:
            residual["visible_text"] = "low"
    return ProbeResult(probe_id, "vision", "ok", residual, {"frames": len(frames), "object_counts": object_counts}, {"name": "ultralytics_yolov8n"})
