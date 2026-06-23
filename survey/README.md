# Survey question preview JSON

This patch adds an optional, disabled-by-default flag to `survey/server.py`:

```bash
python survey/server.py \
  --k 25 \
  --pipeline-output-dir runs/context_pipeline_generation \
  --write-question-preview-json survey/question_preview.json
```

The server writes `question_preview.json` at startup before serving the web UI. The preview uses the same assignment and materialization functions as the live survey, but it does **not** create a participant session or save responses.

Optional deterministic seeds for the preview assignment:

```bash
--preview-participant-id PREVIEW_PARTICIPANT \
--preview-session-id PREVIEW_SESSION
```

The JSON contains:

- global survey item-pool summary;
- pipeline-load metadata;
- the exact assigned `k` questions;
- participant-visible fields: task title, vignette, display fields, rating prompt, scale, confidence prompt, and free-text prompt;
- output summary: plain-language label, description, privacy class;
- hidden analysis metadata: method IDs, baseline/ablation linkage, final output caps, information types, matched output caps, and context-family metadata.

Because the default `least_rated_balanced` assignment mode uses existing session assignment counts from the responses DB, a preview may change if the DB already contains prior sessions. For a completely fixed preview independent of prior counts, either use a clean DB path or run with `--assignment-mode sequential`.
