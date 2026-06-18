#!/usr/bin/env python3
"""Executable SmartPriv pipeline generated from symbolic candidate pipe_16179c121268.

Usage:
  python candidate_01_pipe_16179c121268.py --input frame.jpg --output out.json
  python candidate_01_pipe_16179c121268.py --input audio.wav --output out.json --media-type audio/x-raw
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from smartpriv_runtime.data_model import dump_json_item
from smartpriv_runtime.media_io import item_from_media
from smartpriv_runtime.pipeline import ExecutablePipeline

PIPELINE_SPEC = {
  "schema_version": "smartpriv_executable_pipeline_spec_v1",
  "pipeline_id": "pipe_16179c121268",
  "matched_output_cap": "binary_room_occupied",
  "residual_score": 5,
  "stages": [
    {
      "operator_id": "op.source",
      "variant": "Source",
      "parameters": {},
      "symbolic_output_cap": {
        "media_type": "video/x-raw",
        "content_type": "video_content",
        "schema": "raw_video_stream",
        "properties": {
          "sensorPrimitive": "video_stream"
        }
      }
    },
    {
      "operator_id": "op.person_object_detector",
      "variant": "Person / Object Detector",
      "parameters": {},
      "symbolic_output_cap": {
        "semantic_type": "application/x-detections",
        "schema": "object_detections",
        "properties": {
          "interpretedObservation": [
            "person_detected",
            "object_detected",
            "face_detected"
          ]
        }
      }
    },
    {
      "operator_id": "op.occupancy_deriver",
      "variant": "Occupancy Deriver",
      "parameters": {},
      "symbolic_output_cap": {
        "semantic_type": "application/x-occupancy-count",
        "schema": "occupancy_count",
        "properties": {
          "interpretedObservation": "occupancy_count"
        }
      }
    },
    {
      "operator_id": "op.aggregate_generalize",
      "variant": "Aggregate / Generalize",
      "parameters": {},
      "symbolic_output_cap": {
        "semantic_type": "application/x-aggregate",
        "schema": "aggregate_summary"
      }
    },
    {
      "operator_id": "op.schema_adapter",
      "variant": "Schema Adapter(room_occupied)",
      "parameters": {
        "target_schema": "room_occupied",
        "target_semantic_type": "application/x-binary-occupancy"
      },
      "symbolic_output_cap": {
        "semantic_type": "application/x-binary-occupancy",
        "schema": "room_occupied",
        "affordance": null,
        "properties": {
          "required_informationType": {},
          "adapter_target": "binary_room_occupied"
        }
      }
    },
    {
      "operator_id": "op.route_publish",
      "variant": "Route / Publish(output_to_application)",
      "parameters": {
        "recipient": [
          "local_hvac_controller"
        ],
        "purpose": [
          "energy_management"
        ]
      },
      "symbolic_output_cap": {
        "semantic_type": "external/application-output",
        "schema": "output_to_application"
      }
    }
  ],
  "unsupported_operator_ids": [],
  "notes": [
    "The first op.source stage is symbolic. Pass a DataItem to ExecutablePipeline.process().",
    "Generated specs are executable best-effort implementations of the operator contracts; optional ML libraries improve operator quality."
  ]
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run generated SmartPriv pipeline")
    parser.add_argument("--input", required=True, help="Input image/audio/video path")
    parser.add_argument("--media-type", default=None, help="Override media type, e.g. image/x-raw or audio/x-raw")
    parser.add_argument("--output", default="pipeline_output.json", help="Path for JSON summary output")
    parser.add_argument("--include-payload", action="store_true", help="Include JSON-serializable payloads in output when possible")
    args = parser.parse_args()

    item = item_from_media(args.input, media_type=args.media_type)
    pipe = ExecutablePipeline(PIPELINE_SPEC["stages"])
    result = pipe.process(item)
    dump_json_item(result, args.output, include_payload=args.include_payload)
    print(json.dumps({"pipeline_id": PIPELINE_SPEC.get("pipeline_id"), "output": args.output, "dropped": result is None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
