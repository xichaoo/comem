from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from mvsa_comem.model import TextImageCoMem

model = TextImageCoMem()
text = torch.randn(8, 384)
image = torch.randn(8, 768)
labels = torch.randint(0, 3, (8,))
output = model(text, image)
loss = (
    torch.nn.functional.cross_entropy(output["logits"], labels)
    + 0.15 * torch.nn.functional.cross_entropy(output["text_logits"], labels)
    + 0.15 * torch.nn.functional.cross_entropy(output["image_logits"], labels)
)
loss.backward()
assert output["logits"].shape == (8, 3)
assert output["memory_text"].shape == (8, 16)
assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
print({"loss": loss.item(), "parameters": sum(p.numel() for p in model.parameters())})
