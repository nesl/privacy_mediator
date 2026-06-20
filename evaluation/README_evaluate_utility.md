# To generate all pipelines for the given set of contexts:

python -m evaluation.generate_pipelines_for_all_contexts \
  --operators norms/operator_contracts.json \
  --contexts survey/data/ci_focused_user_study_context_only_dedup_32_no_output_readable.json \
  --app-request-dir app_requests/templates \
  --candidate-generator mediator/generate_pipeline_candidates.py \
  --constraints norms/ci_constraints.json \
  --evaluator mediator/contextual_integrity_evaluator.py \
  --selector mediator/pipeline_selection.py \
  --full-mediator-module mediator/full_mediator.py \
  --baselines raw,manual,direct_llm,full_mediator \
  --ablations utility_only,no_ci_filter,no_residual_bounds,no_least_revealing,uniform_risk_weights,metadata_only,no_staged_flows,first_feasible,latency_first \
  --out-dir runs/context_pipeline_generation


# Utility evaluation for generated privacy pipelines

This README describes how to run `evaluation/evaluate_utility.py` after pipeline generation has written a directory such as `runs/context_pipeline_generation/`.

The evaluator does four things:

1. discovers selected pipelines from `runs/context_pipeline_generation`;
2. filters each task manifest to the evaluation split, defaulting to `test`;
3. runs the selected preprocessing pipeline, unless the method is raw/no-transform;
4. runs the downstream task inference script and writes utility metrics such as precision, recall, F1, F2, and accuracy when ground truth is available.

## Install/copy the patched evaluator

From the project root:

```bash
cp /mnt/data/evaluate_utility_with_metrics.py evaluation/evaluate_utility.py
```

The evaluator assumes your runtime package is available as `mediator.smartpriv_runtime` or `smartpriv_runtime`. With your current repository layout, use:

```bash
--runtime-package mediator.smartpriv_runtime
```

## Quick smoke test: one scenario, two samples

This is the fastest command to verify module paths, GPU selection, preprocessing, downstream inference, and metric computation.

```bash
python -m evaluation.evaluate_utility \
  --pipeline-root runs/context_pipeline_generation \
  --out-dir runs/utility_eval_smoke \
  --project-root . \
  --runtime-package mediator.smartpriv_runtime \
  --scenario-ids S001 \
  --max-samples 2
```

`--max-samples 2` means: filter the task manifest to the test split first, then evaluate only the first 2 test rows for each method. It is for debugging only; omit it for final numbers.

## Smoke test only visitor presence / ChokePoint

```bash
python -m evaluation.evaluate_utility \
  --pipeline-root runs/context_pipeline_generation \
  --out-dir runs/utility_eval_chokepoint_smoke \
  --project-root . \
  --runtime-package mediator.smartpriv_runtime \
  --tasks visitor_presence_detection \
  --scenario-ids S001 \
  --methods raw,manual,direct_llm,full_mediator \
  --max-samples 2
```

The evaluator should automatically resolve your ChokePoint module as:

```text
baselines.task.chokepoint_presence.infer_yolo
```

If needed, force it explicitly:

```bash
--chokepoint-infer-module baselines.task.chokepoint_presence.infer_yolo
```

## Full visitor-presence evaluation on the full test split

```bash
python -m evaluation.evaluate_utility \
  --pipeline-root runs/context_pipeline_generation \
  --out-dir runs/utility_eval_chokepoint_full \
  --project-root . \
  --runtime-package mediator.smartpriv_runtime \
  --tasks visitor_presence_detection
```

This evaluates all discovered visitor-presence contexts/methods on the full ChokePoint test split.

## Full evaluation across all configured tasks

```bash
python -m evaluation.evaluate_utility \
  --pipeline-root runs/context_pipeline_generation \
  --out-dir runs/utility_eval \
  --project-root . \
  --runtime-package mediator.smartpriv_runtime
```

By default, `--tasks auto` discovers tasks present in the generated pipeline summary and skips tasks that are missing manifests, checkpoints, label maps, or inference modules. To make missing task configuration an error, add:

```bash
--strict-task-config
```

If your checkpoints are not in the default locations, pass them explicitly:

```bash
python -m evaluation.evaluate_utility \
  --pipeline-root runs/context_pipeline_generation \
  --out-dir runs/utility_eval \
  --project-root . \
  --runtime-package mediator.smartpriv_runtime \
  --fall-checkpoint outputs/le2i/checkpoint.pt \
  --fall-label-map outputs/le2i/label_map.json \
  --home-audio-checkpoint outputs/chime_home/checkpoint.pt \
  --youhome-checkpoint outputs/youhome/checkpoint.pt \
  --youhome-label-map outputs/youhome/label_map.json
```

## GPU behavior

The default is:

```bash
--device auto
```

On your mixed-GPU machine, the evaluator prefers the RTX 2070 by setting `CUDA_VISIBLE_DEVICES` to the RTX 2070 UUID. This avoids accidentally exposing the TITAN V as `cuda:0`.

Useful device options:

```bash
--device auto      # default; prefer GPU if CUDA is available
--device cpu       # force CPU
--device 0         # Ultralytics GPU 0 among visible devices
--device cuda:0    # Torch classifier GPU 0 among visible devices
```

Useful mixed-GPU options:

```bash
--prefer-gpu-name "RTX 2070"
--cuda-visible-devices GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
--no-prefer-gpu-env
```

You should see output like:

```text
[device] forcing CUDA visibility: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-...
[device] requested=auto ultralytics=0 torch=cuda:0 | CUDA available; visible devices=0:NVIDIA GeForce RTX 2070 cc7.5
```

## Main outputs

The evaluator writes these files under `--out-dir`:

```text
runs/utility_eval/
  utility_eval_plan.json          # resolved tasks, methods, module paths, device settings
  utility_results.json            # full nested result records
  utility_summary.csv             # flat execution + metric summary
  utility_metrics_summary.csv     # compact metric-focused table
  S001/raw/predictions.csv        # downstream predictions for one method
  S001/raw/metrics.json           # per-method metrics
  S001/raw/infer.log              # downstream stdout/stderr
  S001/raw/prediction_errors.csv  # skipped frame/path errors, if supported by the task script
```

For a compact metrics table:

```bash
python - <<'PY'
import pandas as pd
p = 'runs/utility_eval/utility_metrics_summary.csv'
df = pd.read_csv(p)
cols = [
    'scenario_id', 'task', 'method_id', 'status', 'downstream_status',
    'metric_status', 'metric_precision', 'metric_recall', 'metric_f1',
    'metric_f2', 'metric_accuracy', 'metric_tp', 'metric_fp', 'metric_fn', 'metric_tn'
]
print(df[[c for c in cols if c in df.columns]].to_string(index=False))
PY
```

## What metrics mean for ChokePoint

For `visitor_presence_detection`, metrics are frame-level binary person/visitor presence:

```text
prediction: person_present from YOLO output
truth: person present in the ChokePoint XML for that frame
```

The evaluator reports:

```text
precision, recall, F1, F2, accuracy, TP, FP, FN, TN
```

`F2` weights recall more heavily than precision, which is often useful when missing a visitor/person is worse than producing an extra positive frame.

## Intermediate artifacts

By default, transformed images/audio/keypoints are written to a temporary directory and deleted after downstream inference. This avoids filling the disk.

To keep the intermediate preprocessed data for debugging:

```bash
--keep-intermediate-data
```

To put temporary artifacts somewhere other than `/tmp`:

```bash
--intermediate-root /path/with/more/space
```

## Troubleshooting

Check the method log first:

```bash
sed -n '1,220p' runs/utility_eval/S001/raw/infer.log
```

Check skipped or missing inputs:

```bash
cat runs/utility_eval/S001/raw/prediction_errors.csv
```

Check module resolution and task configuration:

```bash
cat runs/utility_eval/utility_eval_plan.json
```

Common fixes:

```bash
# Force the ChokePoint module name.
--chokepoint-infer-module baselines.task.chokepoint_presence.infer_yolo

# Force the runtime package for /mediator/smartpriv_runtime.
--runtime-package mediator.smartpriv_runtime

# Force CPU to distinguish CUDA problems from path/import problems.
--device cpu
```
