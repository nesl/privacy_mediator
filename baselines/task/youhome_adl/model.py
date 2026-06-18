from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as tv_models

from baselines.task.common.audio import LogMelExtractor

TemporalPool = Literal["mean", "gru"]
AudioBackbone = Literal["logmel_cnn", "wav2vec2"]
Modality = Literal["image", "audio", "av"]


class ImageResNetEncoder(nn.Module):
    """ResNet frame encoder with optional temporal modeling over sampled frames."""

    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = True,
        freeze: bool = False,
        temporal_pool: TemporalPool = "gru",
        temporal_hidden_dim: int = 256,
        temporal_layers: int = 1,
        temporal_dropout: float = 0.1,
    ):
        super().__init__()
        if arch == "resnet50":
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            net = tv_models.resnet50(weights=weights)
            dim = net.fc.in_features
        elif arch == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            net = tv_models.resnet18(weights=weights)
            dim = net.fc.in_features
        else:
            raise ValueError(f"Unsupported image arch: {arch}")

        net.fc = nn.Identity()
        self.net = net
        self.temporal_pool = temporal_pool
        self.frame_dim = dim

        if temporal_pool == "mean":
            self.temporal = None
            self.out_dim = dim
        elif temporal_pool == "gru":
            # Bidirectional GRU gives the classifier a chance to use frame order
            # rather than collapsing frames with an unconditional mean.
            self.temporal = nn.GRU(
                input_size=dim,
                hidden_size=temporal_hidden_dim,
                num_layers=temporal_layers,
                batch_first=True,
                bidirectional=True,
                dropout=temporal_dropout if temporal_layers > 1 else 0.0,
            )
            self.out_dim = temporal_hidden_dim * 2
        else:
            raise ValueError(f"Unsupported temporal_pool={temporal_pool}")

        if freeze:
            for p in self.net.parameters():
                p.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Accept [B,C,H,W] or [B,T,C,H,W]. For frame sequences, encode each
        # sampled frame and aggregate over time.
        if image.ndim == 5:
            b, t, c, h, w = image.shape
            feats = self.net(image.reshape(b * t, c, h, w)).reshape(b, t, -1)
            if self.temporal_pool == "mean":
                return feats.mean(dim=1)
            if self.temporal_pool == "gru":
                out, _ = self.temporal(feats)
                return out.mean(dim=1)
            raise RuntimeError(f"Unexpected temporal_pool={self.temporal_pool}")
        return self.net(image)


class LogMelCNNEncoder(nn.Module):
    """Trainable audio encoder over log-mel spectrograms.

    This is often a better starting point than frozen speech Wav2Vec2 for
    non-speech household sounds. SpecAugment-style masking is applied only
    during training to reduce overfitting.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        out_dim: int = 256,
        dropout: float = 0.2,
        spec_augment: bool = True,
        freq_mask_width: int = 8,
        time_mask_width: int = 24,
    ):
        super().__init__()
        self.features = LogMelExtractor(sample_rate=sample_rate, n_mels=n_mels)
        self.spec_augment = spec_augment
        self.freq_mask_width = max(0, int(freq_mask_width))
        self.time_mask_width = max(0, int(time_mask_width))
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out_dim = out_dim

    def _apply_spec_augment(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: [B, n_mels, time]
        if not self.training or not self.spec_augment:
            return spec
        b, f, t = spec.shape
        if f <= 1 or t <= 1:
            return spec
        spec = spec.clone()
        for i in range(b):
            if self.freq_mask_width > 0:
                width = int(torch.randint(0, min(self.freq_mask_width, f) + 1, (1,), device=spec.device).item())
                if width > 0:
                    start = int(torch.randint(0, max(1, f - width + 1), (1,), device=spec.device).item())
                    spec[i, start:start + width, :] = 0
            if self.time_mask_width > 0:
                width = int(torch.randint(0, min(self.time_mask_width, t) + 1, (1,), device=spec.device).item())
                if width > 0:
                    start = int(torch.randint(0, max(1, t - width + 1), (1,), device=spec.device).item())
                    spec[i, :, start:start + width] = 0
        return spec

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform [B,T]
        spec = self.features(waveform)  # [B,n_mels,time]
        spec = self._apply_spec_augment(spec)
        if spec.ndim == 3:
            spec = spec.unsqueeze(1)
        return self.cnn(spec).flatten(1)


class Wav2Vec2Encoder(nn.Module):
    """Torchaudio Wav2Vec2 encoder.

    It is frozen by default because full fine-tuning is memory intensive, but
    the training CLI can now opt into fine-tuning with --finetune-wav2vec2.
    """

    def __init__(self, freeze: bool = True):
        super().__init__()
        import torchaudio
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.model = bundle.get_model()
        self.out_dim = 768
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform [B,T]
        features, _ = self.model.extract_features(waveform)
        last = features[-1]  # [B,frames,768]
        return last.mean(dim=1)


class AVADLClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        modality: Modality = "av",
        image_arch: str = "resnet18",
        image_pretrained: bool = True,
        audio_backbone: AudioBackbone = "logmel_cnn",
        freeze_backbones: bool = False,
        freeze_wav2vec2: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.4,
        temporal_pool: TemporalPool = "gru",
        temporal_hidden_dim: int = 256,
        temporal_layers: int = 1,
        logmel_dim: int = 256,
        logmel_spec_augment: bool = True,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.modality = modality
        self.image_encoder = None
        self.audio_encoder = None
        in_dim = 0

        if modality in {"image", "av"}:
            self.image_encoder = ImageResNetEncoder(
                image_arch,
                pretrained=image_pretrained,
                freeze=freeze_backbones,
                temporal_pool=temporal_pool,
                temporal_hidden_dim=temporal_hidden_dim,
                temporal_layers=temporal_layers,
            )
            in_dim += self.image_encoder.out_dim

        if modality in {"audio", "av"}:
            if audio_backbone == "wav2vec2":
                self.audio_encoder = Wav2Vec2Encoder(freeze=freeze_wav2vec2)
            elif audio_backbone == "logmel_cnn":
                self.audio_encoder = LogMelCNNEncoder(sample_rate=sample_rate, out_dim=logmel_dim, spec_augment=logmel_spec_augment)
            else:
                raise ValueError(f"Unsupported audio_backbone={audio_backbone}")
            if freeze_backbones:
                for p in self.audio_encoder.parameters():
                    p.requires_grad = False
            in_dim += self.audio_encoder.out_dim

        if in_dim <= 0:
            raise ValueError(f"Unsupported modality={modality}")

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image: torch.Tensor | None = None, audio: torch.Tensor | None = None) -> torch.Tensor:
        feats = []
        if self.modality in {"image", "av"}:
            if image is None:
                raise ValueError("image input required for image/av modality")
            feats.append(self.image_encoder(image))
        if self.modality in {"audio", "av"}:
            if audio is None:
                raise ValueError("audio input required for audio/av modality")
            feats.append(self.audio_encoder(audio))
        return self.classifier(torch.cat(feats, dim=1))
