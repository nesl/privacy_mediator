from __future__ import annotations

import torch
from torch import nn
import torchaudio


class LogMelCNN(nn.Module):
    """Small modern log-mel CNN baseline for domestic audio tagging."""

    def __init__(self, num_labels: int = 7, sample_rate: int = 16000, n_mels: int = 64):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=320,
            n_mels=n_mels,
            f_min=50,
            f_max=sample_rate // 2,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_labels)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [B, T]
        x = self.mel(waveform)
        x = self.amplitude_to_db(x)
        # Per-example normalization.
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        x = (x - mean) / std
        x = x.unsqueeze(1)  # [B, 1, Mel, Frames]
        emb = self.net(x).flatten(1)
        return self.classifier(emb)


class ASTTagger(nn.Module):
    """Audio Spectrogram Transformer feature encoder + multilabel head."""

    def __init__(
        self,
        num_labels: int = 7,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        freeze_encoder: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        try:
            from transformers import ASTModel
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "AST backbone requires transformers. Install with: pip install transformers"
            ) from e
        self.encoder = ASTModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_labels))

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_values=input_values)
        # Use CLS token representation.
        emb = out.last_hidden_state[:, 0, :]
        return self.classifier(emb)


def build_model(
    backbone: str,
    num_labels: int = 7,
    sample_rate: int = 16000,
    ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    freeze_encoder: bool = True,
) -> nn.Module:
    """Factory used by train/infer scripts.

    Keeping this function in model.py avoids import errors when infer.py expects
    a single entry point for both the log-mel CNN and AST backbones.
    """
    if backbone == "logmel_cnn":
        return LogMelCNN(num_labels=num_labels, sample_rate=sample_rate)
    if backbone == "ast":
        return ASTTagger(num_labels=num_labels, model_name=ast_model_name, freeze_encoder=freeze_encoder)
    raise ValueError(f"Unknown backbone: {backbone}")
