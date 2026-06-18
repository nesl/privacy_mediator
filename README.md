# SmartPriv executable operator runtime

This bundle turns the symbolic operator contracts into executable Python operators and adds executable-pipeline emission to `generate_pipeline_candidates.py`.

## Contents

- `smartpriv_runtime/` — runtime package.
  - `operators.py` implements all 20 contract operator ids:
    `op.source`, `op.sample`, `op.window`, `op.trigger_gate`, `op.join_fuse`, `op.route_publish`, `op.schema_adapter`, `op.person_object_detector`, `op.pose_extractor`, `op.ocr_screen_detector`, `op.audio_level_extractor`, `op.speech_sound_classifier`, `op.keyword_intent_extractor`, `op.region_select_crop`, `op.region_mask_blur`, `op.speech_content_removal`, `op.occupancy_deriver`, `op.activity_event_classifier`, `op.aggregate_generalize`, and `op.drop_discard`.
  - `pipeline.py` assembles `ExecutablePipeline` objects from a candidate JSON, a full planner output, or an executable spec.
  - `codegen.py` attaches runtime specs to symbolic candidates and emits runnable Python programs.
- `generate_pipeline_candidates.py` — modified planner. New flags:
  - `--emit-executable-specs`: add `executable_pipeline_spec` to each candidate.
  - `--emit-program-dir DIR`: write runnable Python files for top candidates.
  - `--emit-program-top-k N`: number of candidate programs to emit.
- `run_pipeline.py` — generic runner for candidate/spec JSON.
- `operator_contracts.json` — the uploaded contract catalog copied into this bundle.
- `examples/` — example occupancy request, planner output, generated candidate programs, and smoke-test input/output.
- `requirements.txt` and `requirements-optional.txt`.

## Installation

```bash
cd smartpriv_operator_runtime
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For optional higher-quality operators:

```bash
pip install -r requirements-optional.txt
```

`pytesseract` also needs the system binary:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

`dlib` can require build tooling:

```bash
sudo apt-get install -y build-essential cmake
pip install dlib
```

The default implementation intentionally avoids requiring heavyweight model files. It uses OpenCV Haar/HOG detectors and lightweight audio heuristics by default, with optional `mediapipe`, `pytesseract`, `librosa`, or future YOLO/dlib extensions improving quality.

## Generate symbolic candidates with executable specs

```bash
PYTHONPATH=. python generate_pipeline_candidates.py \
  --operators operator_contracts.json \
  --request examples/request_occupancy_image.json \
  --out examples/occupancy_candidates.json \
  --emit-executable-specs \
  --emit-program-dir examples/generated_programs \
  --emit-program-top-k 2
```

The JSON candidates will include an `executable_pipeline_spec` field. The emitted Python programs are thin wrappers around the runtime.

## Run a selected pipeline

Using the generic runner:

```bash
PYTHONPATH=. python run_pipeline.py \
  --pipeline examples/occupancy_candidates.json \
  --input examples/test_frame.png \
  --output examples/runtime_output.json
```

Using a generated candidate program:

```bash
PYTHONPATH=. python examples/generated_programs/candidate_01_*.py \
  --input examples/test_frame.png \
  --output examples/program_output.json
```

## Data model

Operators pass `DataItem` objects:

```python
DataItem(
    caps={"media_type": "image/x-raw", "schema": "raw_image_frame"},
    data=image_array,
    annotations=[],
    metadata={"timestamp_ms": 12345}
)
```

Semantic operators replace raw payloads with contract-aligned semantic records, for example `application/x-detections`, `application/x-occupancy-count`, `application/x-sound-event-label`, or `application/x-safety-event`.

## Notes and limitations

- This is an executable reference implementation, not a benchmark-quality perception stack.
- Operators are contract-faithful in terms of input/output shape and minimization behavior, but detectors/classifiers are lightweight unless optional ML backends are installed.
- `op.source` is symbolic in generated pipelines; runtime callers pass a `DataItem` or media file to the pipeline runner.
- `op.join_fuse` supports a single-stream fallback. Multi-input fusion can be extended by feeding multiple `DataItem`s into a custom scheduler.
- `op.route_publish` returns by default, writes files when a destination/path is supplied, and can POST JSON with `protocol=REST` and `url=...`.
