from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / "data" / "processed" / "manifest.jsonl"
OUTPUT = PROJECT / "data" / "features" / "minilm_dinov2_vitb14.pt"
TEXT_DIR = PROJECT / "pretrained" / "all-MiniLM-L6-v2"
DINO_REPO = PROJECT / "third_party" / "dinov2"
DINO_WEIGHTS = PROJECT / "pretrained" / "dinov2_vitb14_pretrain.pth"


class Samples(Dataset):
    def __init__(self):
        self.rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = self.transform(Image.open(PROJECT / row["image"]).convert("RGB"))
        return row["id"], row["text"], image, row["label"], row["split"]


def main():
    if OUTPUT.exists():
        print(f"Feature cache already exists: {OUTPUT}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_DIR)
    text_model = AutoModel.from_pretrained(TEXT_DIR).to(device).eval().requires_grad_(False)
    dino = torch.hub.load(str(DINO_REPO), "dinov2_vitb14", source="local", pretrained=False)
    state = torch.load(DINO_WEIGHTS, map_location="cpu")
    dino.load_state_dict(state)
    dino = dino.to(device).eval().requires_grad_(False)
    loader = DataLoader(Samples(), batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    ids, splits, labels, text_features, image_features = [], [], [], [], []
    with torch.inference_mode():
        for batch_ids, texts, images, batch_labels, batch_splits in tqdm(loader, desc="features"):
            tokens = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            hidden = text_model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled_text = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled_image = dino(images.to(device, non_blocking=True))
            ids.extend(batch_ids)
            splits.extend(batch_splits)
            labels.append(batch_labels)
            text_features.append(pooled_text.cpu().half())
            image_features.append(pooled_image.cpu().half())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ids": ids,
            "split": splits,
            "labels": torch.cat(labels),
            "text": torch.cat(text_features),
            "image": torch.cat(image_features),
            "text_encoder": "sentence-transformers/all-MiniLM-L6-v2 masked mean of last hidden state",
            "image_encoder": "DINOv2 ViT-B/14 CLS feature",
        },
        OUTPUT,
    )
    print(f"Saved {len(ids)} samples to {OUTPUT}")


if __name__ == "__main__":
    main()
