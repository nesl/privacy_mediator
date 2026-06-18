from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .data_model import DataItem, dump_json_item, load_json_item
from .registry import make_operator


class ExecutablePipeline:
    def __init__(self, stages: Sequence[Dict[str, Any]]) -> None:
        self.stage_specs = list(stages)
        self.stages = [make_operator(spec["operator_id"], **(spec.get("parameters") or {})) for spec in self.stage_specs]

    @classmethod
    def from_candidate(cls, candidate: Dict[str, Any]) -> "ExecutablePipeline":
        if "executable_pipeline_spec" in candidate:
            stages = candidate["executable_pipeline_spec"]["stages"]
        else:
            # Backwards compatible with symbolic records that only have operators.
            stages = [
                {"operator_id": op["operator"], "parameters": op.get("parameters") or {}}
                for op in candidate.get("operators", [])
                if op.get("operator") not in {"op.source"}
            ]
        # The source stage is symbolic in most generated pipelines. Runtime users pass a DataItem.
        stages = [s for s in stages if s.get("operator_id") != "op.source"]
        return cls(stages)

    @classmethod
    def from_spec_file(cls, path: str | Path) -> "ExecutablePipeline":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if "candidates" in obj:
            # Use selected candidate if this is a full generator output.
            selected = obj.get("decision", {}).get("selected_pipeline_id")
            candidates = obj.get("candidates", [])
            cand = next((c for c in candidates if c.get("pipeline_id") == selected), candidates[0] if candidates else None)
            if cand is None:
                raise ValueError("No candidates in spec file")
            return cls.from_candidate(cand)
        if "operators" in obj or "executable_pipeline_spec" in obj:
            return cls.from_candidate(obj)
        if "stages" in obj:
            return cls(obj["stages"])
        raise ValueError(f"Unrecognized pipeline spec shape: {path}")

    def process(self, item: DataItem) -> Optional[DataItem]:
        current: Optional[DataItem] = item
        for stage in self.stages:
            current = stage(current)
            if current is None:
                return None
        return current

    def to_jsonable(self) -> Dict[str, Any]:
        return {"stages": self.stage_specs}
