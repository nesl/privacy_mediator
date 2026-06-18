#!/usr/bin/env python3
import json
from pathlib import Path
from privacy_probes import run_privacy_probes

tmp = Path("tmp_timeseries_probe.json")
tmp.write_text(json.dumps([
    {"timestamp": "2026-01-01T00:00:00", "occupancy_count": 0},
    {"timestamp": "2026-01-01T00:05:00", "occupancy_count": 2},
    {"timestamp": "2026-01-01T00:10:00", "occupancy_count": 3},
]))

report = run_privacy_probes(
    artifacts=[{"path": str(tmp), "modality": "timeseries", "pipeline_id": "pipe_test"}],
    pipeline_id="pipe_test",
    metadata_residual={"aggregate_presence": "medium"},
)
print(json.dumps(report.to_dict(), indent=2))
assert report.probe_residual["aggregate_presence"] == "high"
assert report.combined_residual["aggregate_presence"] == "high"
tmp.unlink()
