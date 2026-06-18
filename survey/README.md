# Context-only CI acceptability survey

This version uses 32 deduplicated context-only scenarios.

It intentionally does **not** show or store participant-facing shared data/output values. The data/output slot is null and should be filled later when constructing full information flows.

Run:

```bash
python server.py --host 0.0.0.0 --port 5000 --k 25
```

Admin endpoints:

- `/admin/summary`
- `/admin/export.csv`
- `/admin/export.json`
- `/admin/flows.json`
