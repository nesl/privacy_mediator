"""Small PyTorch helpers."""
from __future__ import annotations

from pathlib import Path
import random
import os
import json
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 13) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: str | Path, model: torch.nn.Module, **metadata: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    return ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
