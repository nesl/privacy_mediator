"""Audio loading and feature helpers."""
from __future__ import annotations

from pathlib import Path
import torch
import torchaudio


def load_audio(path: str | Path, sample_rate: int = 16000, duration: float | None = None) -> torch.Tensor:
    """Return a mono waveform tensor with shape [num_samples]."""
    waveform, sr = torchaudio.load(str(path))
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.reshape(-1)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if duration is not None:
        target = int(sample_rate * duration)
        if waveform.numel() < target:
            waveform = torch.nn.functional.pad(waveform, (0, target - waveform.numel()))
        elif waveform.numel() > target:
            waveform = waveform[:target]
    return waveform


class LogMelExtractor(torch.nn.Module):
    def __init__(self, sample_rate: int = 16000, n_mels: int = 64, n_fft: int = 1024, hop_length: int = 320):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        spec = self.to_db(self.melspec(waveform))
        return spec
