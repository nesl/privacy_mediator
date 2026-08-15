# Privacy Mediator

This repository implements a privacy-aware preprocessing mediator for smart-space sensor data. It generates type-compatible preprocessing pipelines, checks contextual-integrity constraints and residual disclosure, selects a feasible least-revealing pipeline, runs comparison baselines and ablations, measures downstream task utility, and generates survey questions from pipeline outputs.

Run commands below from the repository root.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The task models have additional dependencies in `baselines/requirements.txt` and `baselines/task/requirements.txt`. Some evaluation commands also require separately downloaded datasets and trained checkpoints; see the README in each directory under `baselines/task/`.

LLM-assisted experiments read the API key from the environment:

```bash
export OPENAI_API_KEY="[add key locally]"
```

Do not commit keys or pass them directly on the command line, where they may be retained in shell history or visible in process listings.

## Datasets and local storage

Datasets are not distributed with this repository. Download each dataset from its official provider and comply with its license and consent restrictions. Store raw datasets under the repository-local `data/` directory and generated manifests alongside their corresponding dataset. Store extracted features, checkpoints, predictions, and metrics under `outputs/`.

Both top-level directories are ignored by Git. Do not force-add raw media, annotations, participant identifiers, trained weights, or generated predictions. The survey scenarios under `survey/data/` are an exception: they are versioned configuration inputs. Survey responses and exports remain under the ignored `survey/outputs/` directory.

The expected high-level layout is:

```text
privacy_mediator/
  data/                         # downloaded datasets; never commit
    le2i/
    chokepoint/
    chime_home/
    youhome/
  outputs/                      # models, features, predictions; never commit
  survey/data/                  # versioned scenario/glossary JSON inputs
  survey/outputs/               # response DBs and exports; never commit
```

### Le2i / ImViA fall detection

Place the extracted Le2i folders under `data/le2i`. The loader supports the dataset's repeated nested directory names:

```text
data/le2i/
  Home_01/Home_01/
    Videos/
      video (1).avi
    Annotation_files/
      video (1).txt
  Coffee_room_01/Coffee_room_01/
    Videos/...
    Annotation_files/...
  Office/Office/
    Videos/...                  # may have no annotations
  Lecture_room/Lecture_room/
    Videos/...                  # may have no annotations
```

Build the manifest, extract pose sequences, and train the classifier:

```bash
python -m baselines.task.le2i_fall.make_manifest \
  --data-dir data/le2i \
  --output-csv data/le2i/le2i_manifest.csv \
  --seed 13

python -m baselines.task.le2i_fall.extract_pose \
  --manifest data/le2i/le2i_manifest.csv \
  --output-dir outputs/le2i_pose \
  --updated-manifest outputs/le2i_manifest_with_keypoints.csv \
  --model yolo11n-pose.pt \
  --stride 3

python -m baselines.task.le2i_fall.train \
  --manifest outputs/le2i_manifest_with_keypoints.csv \
  --output-dir outputs/le2i_fall_model \
  --sample-mode window
```

The manifest builder labels annotated videos and assigns train/validation/test splits. Videos without annotation files are retained as `split=unlabeled`. Pose arrays are written as `.npz` files under `outputs/le2i_pose`; checkpoints and `label_map.json` are written under `outputs/le2i_fall_model`. See `baselines/task/le2i_fall/README.md` for annotation semantics and evaluation commands.

### ChokePoint visitor presence

Place the archives or extracted sequences under `data/chokepoint`:

```text
data/chokepoint/
  groundtruth/
    P1E_S1_C1.xml
  P1E_S1/
    P1E_S1_C1/
      00000000.jpg
      00000001.jpg
  P1E_S1.tar.xz               # optional if already extracted
  groundtruth.tar.xz          # optional if already extracted
```

Create the sequence manifest and frame/event labels with:

```bash
python -m baselines.task.chokepoint_presence.make_manifest \
  --data-dir data/chokepoint \
  --output-manifest data/chokepoint/chokepoint_manifest.csv \
  --output-labels data/chokepoint/chokepoint_labels.csv \
  --output-events data/chokepoint/chokepoint_events.csv \
  --seed 13
```

The baseline uses a pretrained YOLO person detector; it does not train YOLO weights. Predictions, tuned post-processing parameters, and evaluation metrics should go under `outputs/chokepoint/`. See `baselines/task/chokepoint_presence/README.md` for inference, threshold tuning, and event evaluation.

### CHiME-Home domestic audio

Place CHiME-Home/DCASE 2016 metadata and chunks under `data/chime_home`:

```text
data/chime_home/
  development_chunks_refined.csv
  evaluation_chunks_refined.csv
  chunks/
    <chunkname>.csv
    <chunkname>.16kHz.wav
    <chunkname>.48kHz.wav
```

Build the manifest with:

```bash
python -m baselines.task.chime_home_audio.make_manifest \
  --root data/chime_home \
  --output-csv data/chime_home/chime_home_manifest.csv \
  --summary-json data/chime_home/chime_home_manifest_summary.json \
  --sample-rate-khz 16
```

Store trained log-mel CNN or AST checkpoints and predictions under `outputs/chime_home_logmel/` or `outputs/chime_home_ast/`. This is a multi-label task over domestic sound sources; see `baselines/task/chime_home_audio/README.md` for labels, training, threshold tuning, and metrics.

### YouHome activity recognition

Preserve the participant/activity/session nesting under `data/youhome`:

```text
data/youhome/
  data/<participant>/<activity>/<session>/
    *.mp4
    *_1.jpg
    *_2.jpg
    audio.wav
  label/<participant>/<activity>/<session>/
    *_1.txt
    *_2.txt
```

The activity folder name supplies the activity class. Optional label text files contain YOLO-format person boxes used for crop-based privacy variants, not activity labels.

```bash
python -m baselines.task.youhome_adl.make_manifest \
  --root data/youhome \
  --output-csv data/youhome/youhome_manifest.csv \
  --summary-json data/youhome/youhome_manifest_summary.json \
  --split-by participant \
  --seed 13
```

Participant-level splitting is recommended to prevent subject leakage. Store audio-visual checkpoints, label maps, predictions, and reports under `outputs/youhome_adl_av/`. See `baselines/task/youhome_adl/README.md` for training and for full, cropped, blurred, low-resolution, image-only, and audio-only variants.

## Run one privacy-mediator request

The complete mediator generates candidates, evaluates contextual-integrity constraints, optionally applies empirical privacy probes, and selects a pipeline.

Fixed/downstream-compatible request:

```bash
python -m mediator.full_mediator \
  --request app_requests/request_home_entry_intrusion_security.json \
  --out-dir runs/mediator_entry_demo
```

Flexible-output request:

```bash
python -m mediator.full_mediator \
  --request-mode flexible \
  --request app_requests/templates/request_app_visitor_chokepoint_flexible.json \
  --out-dir runs/mediator_visitor_flexible
```

Add LLM contextual-norm judgment with `--use-llm`. Add an empirical privacy-probe artifact manifest with `--probe-artifacts PATH` and, if needed, `--probe-config PATH`. List supported ablations with:

```bash
python -m mediator.full_mediator --list-ablation-modes
```

Run an ablation by repeating `--ablation-mode`, for example:

```bash
python -m mediator.full_mediator \
  --request app_requests/request_home_entry_intrusion_security.json \
  --ablation-mode utility_only \
  --ablation-mode no_ci_filter \
  --out-dir runs/mediator_entry_ablated
```

## Run experiments across contexts

`evaluation.generate_pipelines_for_all_contexts` runs raw, manual, direct-LLM, and full-mediator methods across survey contexts. By default it uses flexible application contracts and writes to `runs/flexible_context_pipeline_generation`.

Full flexible experiment:

```bash
python -m evaluation.generate_pipelines_for_all_contexts \
  --request-mode flexible \
  --contexts survey/data/ci_focused_user_study_context.json \
  --baselines raw,manual,direct_llm,full_mediator \
  --out-dir runs/flexible_context_pipeline_generation
```

The default also runs the configured mediator ablations. Disable them with `--ablations none`, or supply a comma-separated subset. Use `--scenario-ids S001,S002` for a small experiment and `--artifact-detail full` when full candidate/debug artifacts are required.

Older fixed-interface/downstream-compatible experiment:

```bash
python -m evaluation.generate_pipelines_for_all_contexts \
  --request-mode downstream_compatible \
  --contexts survey/data/ci_focused_user_study_context.json \
  --out-dir runs/context_pipeline_generation
```

Offline run without an OpenAI key:

```bash
python -m evaluation.generate_pipelines_for_all_contexts \
  --request-mode flexible \
  --baselines raw,manual,full_mediator \
  --ablations none \
  --out-dir runs/flexible_context_pipeline_generation_offline
```

For a single request, the standalone baseline runner is documented in `preprocessing_baselines/README.md`.

## Measure downstream utility

Pipeline compatibility and declared capabilities are symbolic utility proxies. `evaluation.evaluate_utility` performs the stronger check: it materializes supported preprocessing pipelines and runs the same downstream task evaluators used for raw data.

Preview configuration without running inference:

```bash
python -m evaluation.evaluate_utility \
  --request-mode flexible \
  --analysis-only
```

Smoke test a task/scenario on one sample:

```bash
python -m evaluation.evaluate_utility \
  --request-mode flexible \
  --tasks visitor_presence_detection \
  --scenario-ids S001 \
  --methods raw,manual,full_mediator \
  --max-samples 1 \
  --out-dir runs/utility_eval_smoke
```

Full configured flexible evaluation:

```bash
python -m evaluation.evaluate_utility \
  --request-mode flexible \
  --pipeline-root runs/flexible_context_pipeline_generation \
  --out-dir runs/utility_eval_flexible
```

The evaluator auto-discovers conventional manifests, checkpoints, and inference modules. Use `--strict-task-config` to fail instead of skipping missing task configuration. Detailed dataset/checkpoint overrides and output descriptions are in `evaluation/README_evaluate_utility.md`.

## Run the survey

The survey server uses only the Python standard library. It reads context scenarios from `survey/data/ci_focused_user_study_context.json`, incorporates generated pipeline outputs, and stores responses in SQLite.

Preview questions without creating a participant session:

```bash
python -m survey.server \
  --k 25 \
  --pipeline-output-dir runs/flexible_context_pipeline_generation \
  --survey-output-scope all \
  --assignment-mode sequential \
  --write-question-preview-json survey/question_preview.json
```

Start a local survey using placeholder contact details:

```bash
python -m survey.server \
  --host 127.0.0.1 \
  --port 5000 \
  --k 25 \
  --pipeline-output-dir runs/flexible_context_pipeline_generation \
  --survey-output-scope all
```

Open `http://127.0.0.1:5000`. The default database is `survey/outputs/responses.db`.

For a deployed study, provide approved contact information at runtime rather than committing it:

```bash
python -m survey.server \
  --host 0.0.0.0 \
  --port 5000 \
  --k 25 \
  --pipeline-output-dir runs/flexible_context_pipeline_generation \
  --survey-output-scope all \
  --study-contact-name "[add name]" \
  --study-contact-email "[add email]" \
  --rights-contact-name "[add office or contact]" \
  --rights-contact-email "[add email]" \
  --rights-contact-phone "[add phone]"
```

Useful survey modes:

- `--survey-output-scope all` includes all generated output representations.
- The default, `new_flexible_only`, creates a supplemental survey containing flexible outputs not covered by the previous fixed run. Set both `--pipeline-output-dir` and `--previous-pipeline-output-dir` for this mode.
- `--no-pipeline-outputs` creates context-only questions.
- `--db PATH` selects a different response database.
- `--assignment-mode sequential` makes previews deterministic; the default balances assignments using existing database counts.

Administrative/export endpoints include `/admin/summary`, `/admin/export.csv`, `/admin/export.json`, `/admin/flows.json`, and `/admin/survey_items.json`. Do not expose these endpoints publicly without an authenticated reverse proxy; the built-in server does not provide production authentication or TLS.

More survey behavior and preview details are documented in `survey/README.md`.

## Tests and additional documentation

Run the mediator tests with:

```bash
PYTHONPATH=mediator python -m pytest mediator/tests
```

Additional guides:

- `baselines/task/README.md`: downstream utility baselines.
- `preprocessing_baselines/README.md`: raw, manual, and direct-LLM baselines.
- `evaluation/README_evaluate_utility.md`: utility-evaluation configuration and outputs.
- `survey/README.md`: survey question generation and preview behavior.

Generated data, model outputs, run directories, survey responses, logs, and virtual environments are ignored by Git. Review staged changes before every commit to avoid accidentally adding participant data or secrets.
