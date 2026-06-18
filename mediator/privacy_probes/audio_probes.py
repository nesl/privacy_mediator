from __future__ import annotations
import re
from typing import Any, Dict, List
from .schema import ProbeResult, init_residual_vector
from .utils import list_media_files, try_import, wav_basic_stats

def _words_from_text(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text or "")

def probe_asr_speech_content(path: str, model_name: str = "base") -> ProbeResult:
    probe_id = "audio.asr_speech_content"
    whisper = try_import("whisper")
    if whisper is None:
        return ProbeResult.unavailable(probe_id, "audio", "openai-whisper is not installed; install openai-whisper for ASR probing")
    files = list_media_files(path, "audio")
    if not files:
        return ProbeResult(probe_id, "audio", "no_artifacts", init_residual_vector("none"), {"files": 0}, {"name": "whisper"})
    try:
        model = whisper.load_model(model_name)
    except Exception as exc:
        return ProbeResult.unavailable(probe_id, "audio", f"Could not load Whisper model {model_name}: {exc}", {"name": "whisper", "model": model_name})
    transcripts, errors = [], []
    for f in files:
        try:
            r = model.transcribe(str(f), fp16=False)
            text = str(r.get("text", "")).strip()
            transcripts.append({"path": str(f), "text": text, "num_words": len(_words_from_text(text))})
        except Exception as exc:
            errors.append(f"{f}: {exc}")
    total_words = sum(t["num_words"] for t in transcripts)
    residual = init_residual_vector("none")
    residual["speech_content"] = "none" if total_words == 0 else ("low" if total_words <= 3 else ("medium" if total_words <= 20 else "high"))
    if total_words > 0:
        residual["activity"] = "low"
    return ProbeResult(
        probe_id, "audio", "ok" if not errors else "partial", residual,
        {"files": len(files), "total_words": total_words, "transcripts": transcripts},
        {"name": "whisper", "model": model_name}, errors,
    )

def probe_speaker_count_diarization(path: str) -> ProbeResult:
    probe_id = "audio.speaker_diarization"
    pyannote = try_import("pyannote.audio")
    if pyannote is None:
        return ProbeResult.unavailable(probe_id, "audio", "pyannote.audio is not installed; install pyannote.audio for diarization probing")
    files = list_media_files(path, "audio")
    if not files:
        return ProbeResult(probe_id, "audio", "no_artifacts", init_residual_vector("none"), {"files": 0}, {"name": "pyannote.audio"})
    import os
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        return ProbeResult.unavailable(probe_id, "audio", "HUGGINGFACE_TOKEN/HF_TOKEN not set; pyannote diarization requires authentication", {"name": "pyannote.audio"})
    try:
        from pyannote.audio import Pipeline  # type: ignore
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    except Exception as exc:
        return ProbeResult.unavailable(probe_id, "audio", f"Could not load pyannote diarization pipeline: {exc}")
    speakers, errors, sample = set(), [], []
    for f in files:
        try:
            diarization = pipeline(str(f))
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.add(str(speaker))
                if len(sample) < 20:
                    sample.append({"path": str(f), "start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
        except Exception as exc:
            errors.append(f"{f}: {exc}")
    residual = init_residual_vector("none")
    if len(speakers) == 1:
        residual["speaker_identity"] = "low"
    elif len(speakers) > 1:
        residual["speaker_identity"] = "medium"
        residual["co_presence"] = "high"
    return ProbeResult(
        probe_id, "audio", "ok" if not errors else "partial", residual,
        {"files": len(files), "num_speakers": len(speakers), "speakers": sorted(speakers), "segments_sample": sample},
        {"name": "pyannote.audio", "model": "pyannote/speaker-diarization-3.1"}, errors,
    )

def probe_audio_presence_basic(path: str) -> ProbeResult:
    probe_id = "audio.basic_presence"
    files = list_media_files(path, "audio")
    if not files:
        return ProbeResult(probe_id, "audio", "no_artifacts", init_residual_vector("none"), {"files": 0}, {"name": "wave_builtin"})
    stats = [wav_basic_stats(f) if f.suffix.lower() == ".wav" else {"path": str(f), "note": "non-wav metadata only"} for f in files]
    total_duration = sum(float(s.get("duration_sec", 0) or 0) for s in stats)
    residual = init_residual_vector("none")
    if total_duration > 0:
        residual["activity"] = "low"
        residual["co_presence"] = "low"
    if total_duration >= 60:
        residual["activity"] = "medium"
    return ProbeResult(
        probe_id, "audio", "ok", residual,
        {"files": len(files), "total_duration_sec": total_duration, "stats": stats},
        {"name": "python_wave_builtin"},
    )
