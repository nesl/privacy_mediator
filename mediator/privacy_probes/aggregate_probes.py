from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, List
from .schema import ProbeResult, init_residual_vector
from .utils import numeric, read_timeseries

def _get_any(row: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in row:
            return row[k]
    return None

def probe_aggregate_presence(path: str) -> ProbeResult:
    probe_id = "aggregate.presence_copresence"
    rows = read_timeseries(path)
    if not rows:
        return ProbeResult(probe_id, "timeseries", "no_artifacts", init_residual_vector("none"), {"rows": 0}, {"name": "builtin_timeseries"})
    counts, occupied = [], []
    for row in rows:
        c = numeric(_get_any(row, ["count", "occupancy_count", "num_people", "people_count", "n"]))
        if c is not None: counts.append(c)
        occ = _get_any(row, ["occupied", "room_occupied", "presence", "is_occupied"])
        if isinstance(occ, str): occ = occ.strip().lower() in {"true", "1", "yes", "occupied"}
        if occ is not None: occupied.append(bool(occ))
    residual = init_residual_vector("none")
    max_count = max(counts) if counts else (1 if any(occupied) else 0)
    nonzero = sum(1 for c in counts if c > 0) + sum(1 for x in occupied if x)
    if counts:
        residual["aggregate_presence"] = "high"
        residual["co_presence"] = "high" if max_count >= 2 else "low"
    elif occupied:
        residual["aggregate_presence"] = "medium"
    if nonzero > 0:
        residual["activity"] = "low"
        residual["location"] = "low"
    return ProbeResult(
        probe_id, "timeseries", "ok", residual,
        {"rows": len(rows), "count_rows": len(counts), "occupied_rows": len(occupied), "max_count": max_count, "nonzero_presence_observations": nonzero},
        {"name": "builtin_timeseries"},
    )

def probe_trajectory_from_tracks(path: str) -> ProbeResult:
    probe_id = "aggregate.trajectory_tracks"
    rows = read_timeseries(path)
    if not rows:
        return ProbeResult(probe_id, "timeseries", "no_artifacts", init_residual_vector("none"), {"rows": 0}, {"name": "builtin_timeseries"})
    track_points: Dict[str, List[Any]] = defaultdict(list)
    coordinate_rows = 0
    for row in rows:
        tid = _get_any(row, ["track_id", "person_id", "id", "object_id"])
        x = numeric(_get_any(row, ["x", "cx", "lon", "longitude"]))
        y = numeric(_get_any(row, ["y", "cy", "lat", "latitude"]))
        if tid is not None and x is not None and y is not None:
            track_points[str(tid)].append((x, y))
            coordinate_rows += 1
    residual = init_residual_vector("none")
    num_tracks = len(track_points)
    long_tracks = sum(1 for pts in track_points.values() if len(pts) >= 3)
    if coordinate_rows > 0:
        residual["location"] = "medium"
        residual["trajectory"] = "high" if long_tracks > 0 else ("medium" if num_tracks > 0 else "low")
        if long_tracks > 0:
            residual["identity"] = "low"
        if num_tracks >= 2:
            residual["co_presence"] = "medium"
    return ProbeResult(
        probe_id, "timeseries", "ok", residual,
        {"rows": len(rows), "coordinate_rows": coordinate_rows, "num_tracks": num_tracks, "long_tracks_len_ge_3": long_tracks,
         "track_lengths_sample": {k: len(v) for k, v in list(track_points.items())[:10]}},
        {"name": "builtin_timeseries"},
    )

def probe_routine_inference(path: str) -> ProbeResult:
    probe_id = "aggregate.routine_inference"
    rows = read_timeseries(path)
    if not rows:
        return ProbeResult(probe_id, "timeseries", "no_artifacts", init_residual_vector("none"), {"rows": 0}, {"name": "builtin_timeseries"})
    timestamps = []
    for row in rows:
        t = _get_any(row, ["timestamp", "time", "datetime", "ts"])
        if t is not None: timestamps.append(str(t))
    residual = init_residual_vector("none")
    if len(rows) >= 24 and timestamps:
        residual["activity"] = "medium"
        residual["location"] = "medium"
        residual["trajectory"] = "low"
    if len(rows) >= 7 * 24 and timestamps:
        residual["activity"] = "high"
        residual["location"] = "high"
        residual["trajectory"] = "medium"
    return ProbeResult(
        probe_id, "timeseries", "ok", residual,
        {"rows": len(rows), "timestamped_rows": len(timestamps), "reason": "long timestamped low-dimensional outputs can reveal routines/home-away patterns"},
        {"name": "builtin_timeseries"},
    )
