from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class ConditionalMemoryRouter(nn.Module):
    """Shared key/value memory, conditionally read by both modalities."""

    def __init__(self, dim: int, memory_size: int = 16):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(memory_size, dim) / math.sqrt(dim))
        self.values = nn.Parameter(torch.randn(memory_size, dim) / math.sqrt(dim))
        self.q_text = nn.Linear(2 * dim, dim)
        self.q_image = nn.Linear(2 * dim, dim)
        self.norm_text = nn.LayerNorm(dim)
        self.norm_image = nn.LayerNorm(dim)

    def _read(self, query: torch.Tensor):
        weights = (query @ self.keys.t() / math.sqrt(query.size(-1))).softmax(-1)
        return weights @ self.values, weights

    def forward(self, text: torch.Tensor, image: torch.Tensor):
        dt, wt = self._read(self.q_text(torch.cat([text, image], dim=-1)))
        di, wi = self._read(self.q_image(torch.cat([image, text], dim=-1)))
        return self.norm_text(text + dt), self.norm_image(image + di), wt, wi


class ModalityGatedFusion(nn.Module):
    """Channel-wise modality gates followed by a compact projection."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.gates = nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.Sigmoid())
        self.out = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, text: torch.Tensor, image: torch.Tensor):
        joined = torch.cat([text, image], dim=-1)
        gt, gi = self.gates(joined).chunk(2, dim=-1)
        return self.out(torch.cat([gt * text, gi * image], dim=-1)), gt, gi


@dataclass
class CoMemConfig:
    text_dim: int = 384
    image_dim: int = 768
    hidden_dim: int = 128
    memory_size: int = 16
    num_classes: int = 3
    dropout: float = 0.1


class TextImageCoMem(nn.Module):
    """CoMem fusion head operating on frozen MiniLM and DINOv2 features."""

    def __init__(self, cfg: CoMemConfig = CoMemConfig()):
        super().__init__()
        self.cfg = cfg
        self.text_proj = nn.Sequential(
            nn.Linear(cfg.text_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()
        )
        self.image_proj = nn.Sequential(
            nn.Linear(cfg.image_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()
        )
        self.router = ConditionalMemoryRouter(cfg.hidden_dim, cfg.memory_size)
        self.fusion = ModalityGatedFusion(cfg.hidden_dim, cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.text_head = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.image_head = nn.Linear(cfg.hidden_dim, cfg.num_classes)

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor):
        text_pre = self.text_proj(text_features)
        image_pre = self.image_proj(image_features)
        text, image, memory_text, memory_image = self.router(text_pre, image_pre)
        fused, gate_text, gate_image = self.fusion(text, image)
        return {
            "logits": self.classifier(fused),
            "text_logits": self.text_head(text_pre),
            "image_logits": self.image_head(image_pre),
            "memory_text": memory_text,
            "memory_image": memory_image,
            "gate_text": gate_text,
            "gate_image": gate_image,
        }
