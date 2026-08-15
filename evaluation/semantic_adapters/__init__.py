"""Task-specific semantic output adapters for flexible utility evaluation.

Adapters do not call legacy downstream inference modules.  They consume semantic
JSON outputs materialized from SmartPriv pipelines and either:
  * convert them into a task prediction CSV and optionally call a metric-only
    evaluator, or
  * compute metrics directly against the task manifest labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def run_semantic_adapter(args: Any, row: Any, manifest: Path, data_root: Optional[str | Path], work_dir: Path) -> Dict[str, Any]:
    task = str(getattr(row, "task", "") or "")
    if task == "visitor_presence_detection":
        from . import chokepoint_presence
        return chokepoint_presence.run(args, row, manifest, data_root, work_dir)
    if task == "domestic_sound_monitoring":
        from . import chime_home_audio
        return chime_home_audio.run(args, row, manifest, data_root, work_dir)
    if task == "fall_detection":
        from . import le2i_fall
        return le2i_fall.run(args, row, manifest, data_root, work_dir)
    if task == "adl_recognition":
        from . import youhome_adl
        return youhome_adl.run(args, row, manifest, data_root, work_dir)
    raise ValueError(f"No semantic adapter registered for task {task!r}")
