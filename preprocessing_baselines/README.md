# Preprocessing Baselines

This folder contains three baselines for the SmartPriv-style privacy mediator:

1. **Raw data** (`raw_baseline.py`): forwards an allowed source representation without preprocessing. This is the no-mediation utility upper-bound / privacy lower-bound comparison.
2. **Manual generic** (`manual_baseline.py`): uses a fixed app-level rule table, such as occupancy-count for automation, pose/activity for fall safety, detections for security, and audio labels for audio apps. It does not adapt to CI context, guests, bystanders, space, residual bounds, or transmission principles.
3. **Direct LLM** (`direct_llm_baseline.py`): asks an LLM to directly choose a pipeline from the operator catalog. The symbolic generator is used only after the fact to validate whether the returned operator sequence corresponds to a type-compatible generated candidate. Invalid chains are counted as failures and are not repaired.

## Run all baselines

```bash
python -m preprocessing_baselines.run_baselines \
  --operators norms/operator_contracts.json \
  --request app_requests/request.json \
  --candidate-generator mediator/generate_pipeline_candidates.py \
  --out-dir runs/baselines/request1 \
  --skip-direct-llm
```

To include the direct LLM baseline:

```bash
export OPENAI_API_KEY=...
python -m preprocessing_baselines.run_baselines \
  --operators norms/operator_contracts.json \
  --request app_requests/request.json \
  --environment environments/env.json \
  --candidate-generator mediator/generate_pipeline_candidates.py \
  --out-dir runs/baselines/request1 \
  --llm-model gpt-4o-mini
```

## Individual scripts

```bash
python -m preprocessing_baselines.raw_baseline \
  --operators norms/operator_contracts.json \
  --request app_requests/request.json \
  --candidate-generator mediator/generate_pipeline_candidates.py \
  --out runs/baselines/raw.json

python -m preprocessing_baselines.manual_baseline \
  --operators norms/operator_contracts.json \
  --request app_requests/request.json \
  --candidate-generator mediator/generate_pipeline_candidates.py \
  --out runs/baselines/manual.json

python -m preprocessing_baselines.direct_llm_baseline \
  --operators norms/operator_contracts.json \
  --request app_requests/request.json \
  --environment environments/env.json \
  --candidate-generator mediator/generate_pipeline_candidates.py \
  --out runs/baselines/direct_llm.json
```

## Notes

- The manual and direct-LLM baselines use a **relaxed copy** of the request for compilation/validation: utility requirements and source requirements are preserved, but contextual residual/CI filters are removed. This prevents the baselines from accidentally receiving the full system's privacy reasoning.
- The manual baseline can emit a non-catalog symbolic fallback when no generated candidate matches the fixed rule. Use `--no-symbolic-fallback` or `--no-manual-symbolic-fallback` to count such cases as failures instead.
- Output JSON follows `smartpriv_preprocessing_baseline_output_v1` and includes `decision`, `candidates`, and `diagnostics` fields.
