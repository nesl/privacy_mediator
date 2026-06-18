from __future__ import annotations
import dataclasses
from typing import Any, Dict, List, Optional

RISK_ORDER: Dict[str, int] = {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": 4}
RISK_LEVELS = {v: k for k, v in RISK_ORDER.items()}

RESIDUAL_ATTRIBUTES: List[str] = [
    "identity", "face", "body_shape", "clothing", "gait", "speech_content",
    "speaker_identity", "activity", "location", "trajectory", "co_presence",
    "visible_text", "aggregate_presence",
]

def normalize_risk(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (int, float)):
        if value <= 0: return "none"
        if value <= 1: return "low"
        if value <= 2: return "medium"
        if value <= 3: return "high"
        return "unknown"
    s = str(value).strip().lower()
    if s in RISK_ORDER: return s
    if "unknown" in s: return "unknown"
    if "high" in s: return "high"
    if "medium" in s: return "medium"
    if "low" in s: return "low"
    if "none" in s or "removed" in s or "absent" in s: return "none"
    return "unknown"

def max_risk(a: Any, b: Any) -> str:
    a_s, b_s = normalize_risk(a), normalize_risk(b)
    return RISK_LEVELS[max(RISK_ORDER[a_s], RISK_ORDER[b_s])]

def init_residual_vector(default: str = "none") -> Dict[str, str]:
    return {a: normalize_risk(default) for a in RESIDUAL_ATTRIBUTES}

@dataclasses.dataclass
class ArtifactSpec:
    """A transformed output artifact to probe.

    modality: image, video, audio, timeseries/aggregate/semantic.
    path may be a file or a directory.
    """
    path: str
    modality: str
    kind: str = "transformed_output"
    pipeline_id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ArtifactSpec":
        return ArtifactSpec(
            path=str(d["path"]),
            modality=str(d["modality"]),
            kind=str(d.get("kind", "transformed_output")),
            pipeline_id=d.get("pipeline_id"),
            metadata=dict(d.get("metadata", {}) or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

@dataclasses.dataclass
class ProbeResult:
    probe_id: str
    modality: str
    status: str
    residual: Dict[str, str]
    evidence: Dict[str, Any]
    backend: Dict[str, Any] = dataclasses.field(default_factory=dict)
    errors: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def unavailable(probe_id: str, modality: str, reason: str, backend: Optional[Dict[str, Any]] = None) -> "ProbeResult":
        return ProbeResult(
            probe_id=probe_id, modality=modality, status="unavailable",
            residual=init_residual_vector("none"),
            evidence={"reason": reason}, backend=backend or {}, errors=[reason],
        )

@dataclasses.dataclass
class ProbeReport:
    schema_version: str
    pipeline_id: Optional[str]
    artifacts: List[Dict[str, Any]]
    probe_results: List[Dict[str, Any]]
    probe_residual: Dict[str, str]
    metadata_residual: Dict[str, str]
    combined_residual: Dict[str, str]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
