"""Label definitions for CHiME-Home / DCASE 2016 domestic audio tagging."""

from __future__ import annotations

LABELS = ["c", "m", "f", "v", "p", "b", "o"]

LABEL_MEANINGS = {
    "c": "child speech",
    "m": "adult male speech",
    "f": "adult female speech",
    "v": "video game / TV",
    "p": "percussive sounds such as knocks, bangs, crashes, footsteps",
    "b": "broadband noise such as appliances",
    "o": "other identifiable sounds",
}


def parse_label_string(label_string: str | float | None) -> list[int]:
    """Convert a CHiME-Home label string such as 'cv' into a multi-hot vector.

    The refined files use compact strings where each character is one active
    label. Empty, NaN, or unknown values become the all-zero vector.
    """
    if label_string is None:
        s = ""
    else:
        s = str(label_string).strip().lower()
        if s in {"nan", "none", "null", "-", ""}:
            s = ""
    active = set(s)
    return [1 if lab in active else 0 for lab in LABELS]


def encode_label_string(label_string: str | float | None) -> dict[str, int]:
    vec = parse_label_string(label_string)
    return {f"label_{lab}": int(v) for lab, v in zip(LABELS, vec)}
