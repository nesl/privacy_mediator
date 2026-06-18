from __future__ import annotations

import numpy as np

from smartpriv_runtime import DataItem, ExecutablePipeline, registered_operator_ids


def test_registry_has_contract_operator_ids():
    expected = {
        "op.source", "op.sample", "op.window", "op.trigger_gate", "op.join_fuse",
        "op.route_publish", "op.schema_adapter", "op.person_object_detector", "op.pose_extractor",
        "op.ocr_screen_detector", "op.audio_level_extractor", "op.speech_sound_classifier",
        "op.keyword_intent_extractor", "op.region_select_crop", "op.region_mask_blur",
        "op.speech_content_removal", "op.occupancy_deriver", "op.activity_event_classifier",
        "op.aggregate_generalize", "op.drop_discard",
    }
    assert expected.issubset(set(registered_operator_ids()))


def test_audio_level_pipeline():
    audio = np.zeros(16000, dtype=np.float32)
    audio[1000:3000] = 0.1
    item = DataItem(caps={"media_type": "audio/x-raw"}, data=audio, metadata={"sample_rate": 16000})
    pipe = ExecutablePipeline([
        {"operator_id": "op.audio_level_extractor", "parameters": {}},
        {"operator_id": "op.schema_adapter", "parameters": {"target_schema": "decibel_level_duration", "target_semantic_type": "application/x-decibel-level"}},
    ])
    out = pipe.process(item)
    assert out is not None
    assert out.caps["schema"] == "decibel_level_duration"
    assert "dbfs" in out.data


def test_occupancy_from_detections():
    item = DataItem(
        caps={"semantic_type": "application/x-detections", "schema": "object_detections"},
        data={"detections": [{"label": "person", "bbox": [0, 0, 10, 10]}]},
    )
    pipe = ExecutablePipeline([
        {"operator_id": "op.occupancy_deriver", "parameters": {"binary": True}},
    ])
    out = pipe.process(item)
    assert out is not None
    assert out.data["occupied"] is True
