from __future__ import annotations

from pathlib import Path

import numpy as np

from smartpriv_runtime import DataItem, ExecutablePipeline, registered_operator_ids
from smartpriv_runtime.registry import make_operator


CONTRACT_OPERATOR_IDS = {
    "op.source",
    "op.sample",
    "op.window",
    "op.trigger_gate",
    "op.join_fuse",
    "op.route_publish",
    "op.schema_adapter",
    "op.person_object_detector",
    "op.pose_extractor",
    "op.ocr_screen_detector",
    "op.audio_level_extractor",
    "op.speech_sound_classifier",
    "op.keyword_intent_extractor",
    "op.region_select_crop",
    "op.region_mask_blur",
    "op.speech_content_removal",
    "op.occupancy_deriver",
    "op.activity_event_classifier",
    "op.aggregate_generalize",
    "op.drop_discard",
}


def _image(width: int = 96, height: int = 72) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw a bright rectangle so OCR/screen fallback has structured image content.
    img[15:50, 20:80] = 255
    return img


def _audio(sr: int = 16_000) -> np.ndarray:
    t = np.linspace(0, 1, sr, endpoint=False)
    return (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_every_contract_operator_is_registered():
    assert CONTRACT_OPERATOR_IDS.issubset(set(registered_operator_ids()))


def test_source_operator_passthrough():
    item = DataItem(caps={"media_type": "image/x-raw"}, data=_image())
    out = make_operator("op.source")(item)
    assert out is not None
    assert out.data is item.data
    assert out.caps["media_type"] == "image/x-raw"


def test_sample_operator_rate_limits_by_timestamp():
    op = make_operator("op.sample", sample_period_ms=500)
    first = op(DataItem(caps={"semantic_type": "application/x-sensor-reading"}, data={"value": 1}, metadata={"timestamp_ms": 1000}))
    second = op(DataItem(caps={"semantic_type": "application/x-sensor-reading"}, data={"value": 2}, metadata={"timestamp_ms": 1200}))
    third = op(DataItem(caps={"semantic_type": "application/x-sensor-reading"}, data={"value": 3}, metadata={"timestamp_ms": 1600}))
    assert first is not None
    assert second is None
    assert third is not None
    assert third.metadata["sample_period_ms"] == 500


def test_window_operator_marks_raw_media_as_windowed():
    item = DataItem(caps={"media_type": "audio/x-raw"}, data=_audio(), metadata={"sample_rate": 16000})
    out = make_operator("op.window", window_ms=1000)(item)
    assert out is not None
    assert out.caps["media_type"] == "audio/x-raw"
    assert out.caps["properties"]["windowed"] is True


def test_trigger_gate_allows_and_blocks_items():
    item = DataItem(
        caps={"semantic_type": "application/x-detections"},
        data={"detections": []},
        annotations=[{"label": "person", "confidence": 0.9}],
    )
    assert make_operator("op.trigger_gate", condition={"annotation_label": "person", "min_confidence": 0.5})(item) is not None
    assert make_operator("op.trigger_gate", condition={"annotation_label": "dog", "min_confidence": 0.5})(item) is None


def test_join_fuse_outputs_fused_event():
    item = DataItem(caps={"semantic_type": "application/x-event"}, data={"event": "a"}, annotations=[{"label": "alarm"}])
    out = make_operator("op.join_fuse", join_type="OR")(item)
    assert out is not None
    assert out.caps["semantic_type"] == "application/x-fused-event"
    assert out.data["join_type"] == "OR"
    assert "alarm" in out.data["labels"]


def test_route_publish_can_write_json(tmp_path: Path):
    out_path = tmp_path / "published.json"
    item = DataItem(caps={"semantic_type": "application/x-event"}, data={"event": "test"})
    out = make_operator("op.route_publish", protocol="file", destination=str(out_path))(item)
    assert out is not None
    assert out_path.exists()
    assert out.metadata["published_to"] == str(out_path)


def test_schema_adapter_converts_occupancy_shape():
    item = DataItem(caps={"semantic_type": "application/x-occupancy-count"}, data={"count": 2})
    out = make_operator("op.schema_adapter", target_schema="room_occupied", target_semantic_type="application/x-binary-occupancy")(item)
    assert out is not None
    assert out.caps["schema"] == "room_occupied"
    assert out.data == {"occupied": True, "count": 2}


def test_person_object_detector_returns_detection_record_even_on_blank_image():
    # Use OpenCV backend explicitly so this test never attempts optional YOLO downloads.
    item = DataItem(caps={"media_type": "image/x-raw"}, data=np.zeros((96, 96, 3), dtype=np.uint8))
    out = make_operator("op.person_object_detector", backend="opencv", classes=["person", "face"])(item)
    assert out is not None
    assert out.caps["semantic_type"] == "application/x-detections"
    assert "detections" in out.data
    assert "count_by_label" in out.data


def test_pose_extractor_returns_downstream_compatible_yolo_pose_schema():
    item = DataItem(caps={"media_type": "image/x-raw"}, data=_image())
    out = make_operator("op.pose_extractor")(item)
    assert out is not None
    assert out.caps["semantic_type"] == "application/x-pose-keypoints"
    assert out.caps["schema"] == "yolo_coco17_pose_sequence"
    assert out.caps["properties"]["skeleton"] == "coco17"
    assert "poses" in out.data


def test_ocr_screen_detector_returns_region_schema():
    item = DataItem(caps={"media_type": "image/x-raw"}, data=_image())
    out = make_operator("op.ocr_screen_detector", emit_text=False)(item)
    assert out is not None
    assert out.caps["semantic_type"] == "application/x-text-regions"
    assert "regions" in out.data
    for region in out.data["regions"]:
        assert "text" not in region


def test_audio_level_extractor_computes_dbfs_and_duration():
    item = DataItem(caps={"media_type": "audio/x-raw"}, data=_audio(), metadata={"sample_rate": 16000})
    out = make_operator("op.audio_level_extractor", window_ms=1000)(item)
    assert out is not None
    assert out.caps["schema"] == "decibel_level_duration"
    assert isinstance(out.data["dbfs"], float)
    assert 0.99 <= out.data["duration_s"] <= 1.01


def test_speech_sound_classifier_labels_audio():
    item = DataItem(caps={"media_type": "audio/x-raw"}, data=_audio(), metadata={"sample_rate": 16000})
    out = make_operator("op.speech_sound_classifier")(item)
    assert out is not None
    assert out.caps["schema"] == "sound_event_label"
    assert "labels" in out.data
    assert out.data["top_label"] is not None


def test_keyword_intent_extractor_matches_allowed_intent():
    item = DataItem(caps={"semantic_type": "application/x-transcript"}, data="please turn on the kitchen lights")
    out = make_operator("op.keyword_intent_extractor", allowed_intents=[{"intent": "lights_on", "keywords": ["turn on"]}])(item)
    assert out is not None
    assert out.caps["schema"] == "keyword_or_intent"
    assert out.data["intent"] == "lights_on"
    assert out.data["confidence"] > 0


def test_region_select_crop_uses_static_roi():
    item = DataItem(caps={"media_type": "image/x-raw"}, data=_image(100, 80))
    out = make_operator("op.region_select_crop", static_roi=[10, 10, 40, 30])(item)
    assert out is not None
    assert out.data.shape[:2] == (20, 30)
    assert out.caps["properties"]["field_of_view_minimized"] is True


def test_region_mask_blur_masks_annotated_region():
    img = np.full((40, 40, 3), 255, dtype=np.uint8)
    item = DataItem(
        caps={"media_type": "image/x-raw"},
        data=img,
        annotations=[{"label": "face", "bbox": [5, 5, 20, 20], "confidence": 1.0}],
    )
    out = make_operator("op.region_mask_blur", target="face", method="mask")(item)
    assert out is not None
    assert out.caps["properties"]["redacted"] is True
    assert np.all(out.data[5:20, 5:20] == 0)
    assert np.all(out.data[25:35, 25:35] == 255)


def test_speech_content_removal_redacts_transcript():
    item = DataItem(caps={"semantic_type": "application/x-transcript"}, data="Alice Smith said code 12345")
    out = make_operator("op.speech_content_removal", mode="redact_transcript")(item)
    assert out is not None
    assert out.caps["semantic_type"] == "application/x-redacted-transcript"
    assert "[NAME]" in out.data["redacted_text"]
    assert "[NUMBER]" in out.data["redacted_text"]


def test_occupancy_deriver_counts_people():
    item = DataItem(
        caps={"semantic_type": "application/x-detections", "schema": "object_detections"},
        data={"detections": [{"label": "person"}, {"label": "face"}, {"label": "dog"}]},
    )
    out = make_operator("op.occupancy_deriver", binary=False)(item)
    assert out is not None
    assert out.caps["schema"] == "occupancy_count"
    assert out.data["count"] == 2


def test_activity_event_classifier_detects_audio_safety_event():
    item = DataItem(
        caps={"semantic_type": "application/x-sound-event-label", "schema": "sound_event_label"},
        data={"labels": [{"label": "glass_break_or_alarm", "confidence": 0.8}]},
    )
    out = make_operator("op.activity_event_classifier")(item)
    assert out is not None
    # Flexible contracts canonicalize activity and safety decisions to the
    # application/x-activity-event family while retaining the safety schema.
    assert out.caps["semantic_type"] == "application/x-activity-event"
    assert out.caps["schema"] == "fall_or_safety_event"
    assert out.data["event_type"] == "glass_break_or_alarm"


def test_aggregate_generalize_counts_labels_and_adds_granularity():
    item = DataItem(
        caps={"semantic_type": "application/x-sound-event-label"},
        data={"labels": [{"label": "speech"}, {"label": "speech"}, {"label": "alarm"}]},
    )
    out = make_operator("op.aggregate_generalize", temporal_granularity_ms=60000, spatial_granularity="room")(item)
    assert out is not None
    assert out.caps["schema"] == "aggregate_summary"
    assert out.data["count_by_label"] == {"speech": 2, "alarm": 1}
    assert out.data["temporal_granularity_ms"] == 60000


def test_drop_discard_returns_none():
    item = DataItem(caps={"semantic_type": "application/x-event"}, data={"event": "discard_me"})
    assert make_operator("op.drop_discard")(item) is None


def test_executable_pipeline_can_chain_multiple_operators():
    item = DataItem(caps={"semantic_type": "application/x-detections"}, data={"detections": [{"label": "person"}]})
    pipe = ExecutablePipeline([
        {"operator_id": "op.occupancy_deriver", "parameters": {"binary": True}},
        {"operator_id": "op.schema_adapter", "parameters": {"target_schema": "room_occupied", "target_semantic_type": "application/x-binary-occupancy"}},
    ])
    out = pipe.process(item)
    assert out is not None
    assert out.data["occupied"] is True
    assert out.data["count"] == 1
