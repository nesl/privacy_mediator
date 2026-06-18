from __future__ import annotations
from typing import Any, Dict
from .schema import RESIDUAL_ATTRIBUTES, init_residual_vector, max_risk, normalize_risk

def combine_metadata_and_probe_residual(
    metadata_residual: Dict[str, Any] | None,
    probe_residual: Dict[str, Any] | None,
    policy: str = "conservative_max",
) -> Dict[str, str]:
    """Combine symbolic metadata residuals with empirical probe residuals.

    Default: attribute-wise max(metadata, probe).
    """
    meta = metadata_residual or {}
    probe = probe_residual or {}
    out = init_residual_vector("none")
    for attr in RESIDUAL_ATTRIBUTES:
        m = normalize_risk(meta.get(attr, "none"))
        p = normalize_risk(probe.get(attr, "none"))
        if policy == "probe_only":
            out[attr] = p
        elif policy == "metadata_only":
            out[attr] = m
        elif policy == "conservative_max":
            out[attr] = max_risk(m, p)
        else:
            raise ValueError(f"Unknown combine policy: {policy}")
    return out
