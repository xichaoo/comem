from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    """Reads a single, auditable feature cache produced by extract_features.py."""

    def __init__(self, cache_path: str | Path, split: str):
        payload = torch.load(cache_path, map_location="cpu")
        mask = [value == split for value in payload["split"]]
        indices = torch.tensor([i for i, keep in enumerate(mask) if keep], dtype=torch.long)
        self.ids = [payload["ids"][i] for i in indices.tolist()]
        self.text = payload["text"].index_select(0, indices).float()
        self.image = payload["image"].index_select(0, indices).float()
        self.labels = payload["labels"].index_select(0, indices).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.text[index], self.image[index], self.labels[index]
