from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio

from .labels import LABELS


class ChimeHomeDataset(Dataset):
    """Dataset for CHiME-Home 4-second multi-label audio chunks."""

    def __init__(
        self,
        manifest: str | Path,
        split: str | None = None,
        sample_rate: int = 16000,
        duration: float = 4.0,
        audio_column: str = "audio_path",
        feature_mode: Literal["waveform", "ast"] = "waveform",
        ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    ) -> None:
        self.manifest = Path(manifest)
        self.df = pd.read_csv(self.manifest)
        if split is not None:
            self.df = self.df[self.df["split"].astype(str) == split].copy()
        self.df = self.df.reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split={split!r} in {manifest}")
        self.sample_rate = int(sample_rate)
        self.num_samples = int(round(duration * sample_rate))
        self.audio_column = audio_column
        self.feature_mode = feature_mode
        self.ast_model_name = ast_model_name
        self._ast_extractor = None
        for lab in LABELS:
            if f"label_{lab}" not in self.df.columns:
                raise ValueError(f"Manifest missing label column label_{lab}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(path)
        wav, sr = torchaudio.load(str(path))
        # Downmix to mono.
        wav = wav.mean(dim=0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.numel() < self.num_samples:
            wav = torch.nn.functional.pad(wav, (0, self.num_samples - wav.numel()))
        elif wav.numel() > self.num_samples:
            wav = wav[: self.num_samples]
        return wav.float()

    def _to_ast_features(self, wav: torch.Tensor) -> torch.Tensor:
        # Lazy import so the log-mel baseline does not require transformers.
        if self._ast_extractor is None:
            from transformers import AutoFeatureExtractor
            self._ast_extractor = AutoFeatureExtractor.from_pretrained(self.ast_model_name)
        arr = wav.detach().cpu().numpy().astype(np.float32)
        feat = self._ast_extractor(arr, sampling_rate=self.sample_rate, return_tensors="pt")
        return feat["input_values"].squeeze(0).float()

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = Path(str(row[self.audio_column]))
        wav = self._load_audio(path)
        y = torch.tensor([float(row[f"label_{lab}"]) for lab in LABELS], dtype=torch.float32)
        x = self._to_ast_features(wav) if self.feature_mode == "ast" else wav
        return {
            "x": x,
            "y": y,
            "chunk_id": str(row["chunk_id"]),
            "audio_path": str(path),
            "majorityvote": str(row.get("majorityvote", "")),
        }


def collate_batch(batch):
    xs = torch.stack([b["x"] for b in batch], dim=0)
    ys = torch.stack([b["y"] for b in batch], dim=0)
    return {
        "x": xs,
        "y": ys,
        "chunk_id": [b["chunk_id"] for b in batch],
        "audio_path": [b["audio_path"] for b in batch],
        "majorityvote": [b["majorityvote"] for b in batch],
    }
