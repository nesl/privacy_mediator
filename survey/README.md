# Survey server

The survey turns contextual-integrity scenarios and generated mediator outputs into participant-facing questions. It stores sessions and responses in SQLite and provides JSON/CSV administrative exports.

Run commands from the repository root.

## Start locally

Generate flexible pipelines first, or use an existing run directory:

```bash
python -m evaluation.generate_pipelines_for_all_contexts \
  --request-mode flexible \
  --contexts survey/data/ci_focused_user_study_context.json \
  --out-dir runs/flexible_context_pipeline_generation
```

Then start the survey:

```bash
python -m survey.server \
  --host 127.0.0.1 \
  --port 5000 \
  --k 25 \
  --pipeline-output-dir runs/flexible_context_pipeline_generation \
  --survey-output-scope all
```

Open `http://127.0.0.1:5000`. Responses are stored in `survey/outputs/responses.db` unless `--db` specifies another path.

The HTML template contains no personal contact information. Its default labels are `[add name]`, `[add email]`, and `[add phone]`. For an approved deployment, supply contacts at runtime:

```bash
python -m survey.server \
  --host 0.0.0.0 \
  --port 5000 \
  --study-contact-name "[add name]" \
  --study-contact-email "[add email]" \
  --rights-contact-name "[add office or contact]" \
  --rights-contact-email "[add email]" \
  --rights-contact-phone "[add phone]"
```

Values are HTML-escaped before insertion. Command-line values can be visible in process listings and shell history, so do not use these options for secrets. Contact details are public survey content, not credentials.

## Output scopes

- `--survey-output-scope all` includes all output representations found in the selected pipeline run.
- `--survey-output-scope new_flexible_only` is the default supplemental mode. It compares `--pipeline-output-dir` with `--previous-pipeline-output-dir` and includes only genuinely new flexible output schemas.
- `--no-pipeline-outputs` ignores pipeline runs and creates context-only items.
- `--include-no-output-variants` also creates cases for methods that deny sharing or select no output.

The default pipeline directories are:

- flexible: `runs/flexible_context_pipeline_generation`;
- previous fixed/downstream-compatible: `runs/context_pipeline_generation`.

## Preview questions

Use `--write-question-preview-json` to inspect the exact participant-visible questions without creating a session or saving responses:

```bash
python -m survey.server \
  --k 25 \
  --pipeline-output-dir runs/flexible_context_pipeline_generation \
  --survey-output-scope all \
  --assignment-mode sequential \
  --write-question-preview-json survey/question_preview.json
```

Optional deterministic assignment seeds are available through `--preview-participant-id` and `--preview-session-id`. With the default `least_rated_balanced` assignment, existing assignment counts in the response database can change a preview. Use `--assignment-mode sequential` for a fixed preview.

The preview contains only question content visible on the webpage. It does not contain method IDs, pipeline IDs, source paths, technical output caps, hidden assignment metadata, or attention-check answer keys.

## Administration and exports

The server exposes:

- `/admin/summary`
- `/admin/export.csv`
- `/admin/export.json`
- `/admin/flows.json`
- `/admin/survey_items.json`

The built-in HTTP server has no production authentication or TLS. Bind to `127.0.0.1` for local work. For public deployment, place it behind an authenticated TLS reverse proxy and restrict the `/admin/` routes.

Participant responses and identifiers are sensitive. The default `survey/outputs/` directory is ignored by Git; keep custom database/export paths out of version control as well.

## Other useful options

```bash
python -m survey.server --help
```

- `--flow-file PATH`: context-scenario JSON.
- `--db PATH`: SQLite response database.
- `--seed N`: assignment seed.
- `--assignment-mode least_rated_balanced|sequential`: assignment strategy.
- `--max-per-scenario-group N`: cap variants from one scenario family per participant; `0` disables the cap.
- `--supplemental-output-schemas a,b`: override schemas included by supplemental mode.
