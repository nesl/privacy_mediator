"""Semantic utility adapter for CHiME-Home domestic sound monitoring."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .common import (
    json_paths_for_semantic_row,
    load_semantic_json,
    multilabel_binary_metrics,
    normalize_label_text,
    read_csv_rows,
    semantic_labels_and_count,
    semantic_payload,
    write_csv_rows,
    write_json,
)

CHIME_LABELS = ["c", "m", "f", "v", "p", "b", "o"]
CHIME_LABEL_SYNONYMS = {
    "c": {"c", "child", "child_speech", "child speech", "speech_child"},
    "m": {"m", "male", "adult_male", "adult male", "adult_male_speech", "male_speech", "speech_male"},
    "f": {"f", "female", "adult_female", "adult female", "adult_female_speech", "female_speech", "speech_female"},
    "v": {"v", "tv", "television", "video", "video_game", "video game", "game", "games"},
    "p": {"p", "percussive", "knock", "knocks", "bang", "bangs", "crash", "crashes", "footstep", "footsteps"},
    "b": {"b", "broadband", "noise", "appliance", "appliances", "vacuum", "fan"},
    "o": {"o", "other", "other_sound", "identifiable_sound", "unknown_sound"},
}


def parse_chime_label_string(label_string: Any) -> List[int]:
    if label_string is None:
        s = ""
    else:
        s = str(label_string).strip().lower()
        if s in {"nan", "none", "null", "-", ""}:
            s = ""
    active = set(s)
    return [1 if lab in active else 0 for lab in CHIME_LABELS]


def chime_truth_vector(row: Dict[str, Any]) -> List[int]:
    vals: List[int] = []
    have_cols = True
    for lab in CHIME_LABELS:
        key = f"label_{lab}"
        if key not in row or row.get(key) in (None, ""):
            have_cols = False
            break
        try:
            vals.append(1 if int(float(str(row.get(key)))) > 0 else 0)
        except Exception:
            vals.append(0)
    if have_cols:
        return vals
    for key in ["majorityvote", "annotation_a1", "annotation_a2", "annotation_a3", "label", "sound_label"]:
        if row.get(key) not in (None, ""):
            return parse_chime_label_string(row.get(key))
    return [0] * len(CHIME_LABELS)


def chime_pred_vector_from_labels(labels: Sequence[str]) -> List[int]:
    active = set()
    normalized = {normalize_label_text(x) for x in labels if str(x or "").strip()}
    compact = set("".join(x for x in normalized if len(x) <= 7))
    for lab in CHIME_LABELS:
        syns = {normalize_label_text(x) for x in CHIME_LABEL_SYNONYMS[lab]}
        if lab in compact or normalized & syns:
            active.add(lab)
    # Generic speech is weaker than speaker-type labels.  Mark c/m/f so the
    # metric reveals that this representation supports speech presence but not
    # demographic subtyping with high precision.
    if not (active & {"c", "m", "f"}) and any("speech" in x or "voice" in x for x in normalized):
        active.update(["c", "m", "f"])
    return [1 if lab in active else 0 for lab in CHIME_LABELS]


def run(args: Any, row: Any, manifest: Path, data_root: Optional[str | Path], work_dir: Path) -> Dict[str, Any]:
    dry_run = bool(getattr(args, "dry_run", False))
    output_csv = work_dir / "predictions.csv"
    metrics_json = work_dir / "metrics.json"
    rows, _ = read_csv_rows(manifest)
    pred_rows: List[Dict[str, Any]] = []
    y_true: List[List[int]] = []
    y_pred: List[List[int]] = []

    if not dry_run:
        for i, r in enumerate(rows):
            chunk_id = str(r.get("chunk_id") or r.get("sample_id") or i)
            json_paths = json_paths_for_semantic_row(r)
            labels: List[str] = []
            confidence = 0.0
            decibel_values: List[float] = []
            for jp in json_paths[:64]:
                obj = load_semantic_json(jp)
                labs, _count, _occ, conf = semantic_labels_and_count(obj)
                labels.extend(labs)
                confidence = max(confidence, conf)
                data = semantic_payload(obj)
                for key in ["db", "dB", "decibel", "decibel_level", "rms_db", "sound_level_db"]:
                    if data.get(key) not in (None, ""):
                        try:
                            decibel_values.append(float(data.get(key)))
                        except Exception:
                            pass
            pred_vec = chime_pred_vector_from_labels(labels)
            if not any(pred_vec) and decibel_values:
                # Decibel-only outputs are coarse noise-monitoring utilities, not
                # full CHiME event classifiers.  Mark broadband/noise only.
                pred_vec[CHIME_LABELS.index("b")] = 1 if max(decibel_values) > -35.0 else 0
            true_vec = chime_truth_vector(r)
            y_true.append(true_vec)
            y_pred.append(pred_vec)
            rec = {
                "chunk_id": chunk_id,
                "audio_path": r.get("audio_path") or r.get("audio_16k_path") or "",
                "majorityvote": r.get("majorityvote", ""),
                "true_label_string": "".join(lab for lab, val in zip(CHIME_LABELS, true_vec) if val),
                "pred_label_string": "".join(lab for lab, val in zip(CHIME_LABELS, pred_vec) if val),
                "confidence": confidence,
                "semantic_labels": ";".join(labels[:16]),
                "semantic_json_count": len(json_paths),
                "split": r.get("split", ""),
            }
            for lab, tv, pv in zip(CHIME_LABELS, true_vec, pred_vec):
                rec[f"true_{lab}"] = int(tv)
                rec[f"pred_{lab}"] = int(pv)
                rec[f"score_{lab}"] = float(pv)
            pred_rows.append(rec)

    write_csv_rows(pred_rows, output_csv)
    metrics = multilabel_binary_metrics(y_true, y_pred, CHIME_LABELS)
    metrics.update({
        "task": getattr(row, "task", "domestic_sound_monitoring"),
        "level": "semantic_chime_multilabel_adapter",
        "semantic_adapter": "chime_home_audio",
        "calls_legacy_inference": False,
        "adapter_note": "Sound-event semantic outputs are compared directly to CHiME multi-label columns. Decibel-only outputs are treated as coarse noise-monitoring proxies, not full event-tagging equivalence.",
    })
    write_json(metrics, metrics_json)
    return {
        "status": "ok",
        "inference": {"semantic_adapter": "chime_home_audio", "returncode": 0, "dry_run": dry_run, "calls_legacy_inference": False},
        "output_csv": str(output_csv),
        "metrics_json": str(metrics_json),
        "metrics": metrics,
    }
