"""
privacy_probes: empirical residual-disclosure probes for SmartPriv/mediator pipelines.
"""
from .schema import RESIDUAL_ATTRIBUTES, RISK_ORDER, ArtifactSpec, ProbeResult, ProbeReport, init_residual_vector, max_risk
from .combine import combine_metadata_and_probe_residual
from .probe_runner import run_privacy_probes
from .mediator_integration import run_probes_for_mediator_result, attach_probe_report_to_mediator_result

__all__ = [
    "RESIDUAL_ATTRIBUTES", "RISK_ORDER", "ArtifactSpec", "ProbeResult", "ProbeReport",
    "init_residual_vector", "max_risk", "combine_metadata_and_probe_residual",
    "run_privacy_probes", "run_probes_for_mediator_result", "attach_probe_report_to_mediator_result",
]
