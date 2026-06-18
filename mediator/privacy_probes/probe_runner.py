from __future__ import annotations
from typing import Any, Dict, List, Optional
from .aggregate_probes import probe_aggregate_presence, probe_routine_inference, probe_trajectory_from_tracks
from .audio_probes import probe_asr_speech_content, probe_audio_presence_basic, probe_speaker_count_diarization
from .combine import combine_metadata_and_probe_residual
from .schema import ArtifactSpec, ProbeReport, ProbeResult, init_residual_vector, max_risk

def _merge_probe_residual(results: List[ProbeResult]) -> Dict[str, str]:
    merged = init_residual_vector("none")
    for r in results:
        if r.status in {"unavailable", "no_artifacts"}:
            continue
        for attr, risk in r.residual.items():
            if attr in merged:
                merged[attr] = max_risk(merged[attr], risk)
    return merged

def _run_for_artifact(artifact: ArtifactSpec, probe_config: Optional[Dict[str, Any]] = None) -> List[ProbeResult]:
    cfg = probe_config or {}
    modality = artifact.modality.lower()
    results: List[ProbeResult] = []
    if modality in {"image", "video", "vision"}:
        max_frames = int(cfg.get("max_frames", 32))
        from .vision_probes import probe_face_detection, probe_ocr_visible_text, probe_person_body_presence, probe_scene_context
        results.append(probe_face_detection(artifact.path, max_frames=max_frames))
        results.append(probe_person_body_presence(artifact.path, max_frames=max_frames))
        if cfg.get("enable_ocr", True):
            results.append(probe_ocr_visible_text(artifact.path, max_frames=min(max_frames, 16)))
        if cfg.get("enable_yolo_scene_probe", False):
            results.append(probe_scene_context(artifact.path, max_frames=min(max_frames, 16)))
    elif modality == "audio":
        results.append(probe_audio_presence_basic(artifact.path))
        if cfg.get("enable_asr", True):
            results.append(probe_asr_speech_content(artifact.path, model_name=str(cfg.get("whisper_model", "base"))))
        if cfg.get("enable_diarization", False):
            results.append(probe_speaker_count_diarization(artifact.path))
    elif modality in {"timeseries", "aggregate", "semantic"}:
        results.append(probe_aggregate_presence(artifact.path))
        results.append(probe_trajectory_from_tracks(artifact.path))
        results.append(probe_routine_inference(artifact.path))
    else:
        results.append(ProbeResult.unavailable("unknown.modality", artifact.modality, f"Unsupported artifact modality: {artifact.modality}"))
    return results

def run_privacy_probes(
    artifacts: List[ArtifactSpec] | List[Dict[str, Any]],
    pipeline_id: Optional[str] = None,
    metadata_residual: Optional[Dict[str, Any]] = None,
    probe_config: Optional[Dict[str, Any]] = None,
    combine_policy: str = "conservative_max",
) -> ProbeReport:
    specs = [a if isinstance(a, ArtifactSpec) else ArtifactSpec.from_dict(a) for a in artifacts]
    if pipeline_id is not None:
        specs = [s for s in specs if s.pipeline_id in {None, pipeline_id}]
    all_results: List[ProbeResult] = []
    for artifact in specs:
        all_results.extend(_run_for_artifact(artifact, probe_config=probe_config))
    probe_residual = _merge_probe_residual(all_results)
    metadata_residual_norm = metadata_residual or init_residual_vector("none")
    combined = combine_metadata_and_probe_residual(metadata_residual_norm, probe_residual, policy=combine_policy)
    return ProbeReport(
        schema_version="smartpriv_privacy_probe_report_v1",
        pipeline_id=pipeline_id,
        artifacts=[s.to_dict() for s in specs],
        probe_results=[r.to_dict() for r in all_results],
        probe_residual=probe_residual,
        metadata_residual=metadata_residual_norm,
        combined_residual=combined,
        notes=[
            "Probe residuals measure inferability from transformed artifacts.",
            "Policy decisions should be made by the CI evaluator, not by probes directly.",
            "Default combination policy is attribute-wise conservative max(metadata, probe).",
        ],
    )
