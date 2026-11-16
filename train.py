from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from mvsa_comem.data import CachedFeatureDataset
from mvsa_comem.model import CoMemConfig, TextImageCoMem


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def classification_metrics(targets, predictions, num_classes=3):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    recalls, f1s = [], []
    for cls in range(num_classes):
        tp = int(((targets == cls) & (predictions == cls)).sum())
        fp = int(((targets != cls) & (predictions == cls)).sum())
        fn = int(((targets == cls) & (predictions != cls)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
    return {
        "accuracy": float((targets == predictions).mean()),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    targets, joint, text, image = [], [], [], []
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    for text_x, image_x, labels in loader:
        text_x, image_x, labels = text_x.to(device), image_x.to(device), labels.to(device)
        output = model(text_x, image_x)
        loss_total += criterion(output["logits"], labels).item() * labels.size(0)
        targets.extend(labels.cpu().tolist())
        joint.extend(output["logits"].argmax(1).cpu().tolist())
        text.extend(output["text_logits"].argmax(1).cpu().tolist())
        image.extend(output["image_logits"].argmax(1).cpu().tolist())
    metrics = classification_metrics(targets, joint)
    metrics["loss"] = loss_total / len(targets)
    metrics["text_accuracy"] = float((np.asarray(targets) == np.asarray(text)).mean())
    metrics["image_accuracy"] = float((np.asarray(targets) == np.asarray(image)).mean())
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/comem_seed0.json")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    config_path = (PROJECT / args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_path = PROJECT / cfg["feature_cache"]
    loaders = {
        split: DataLoader(
            CachedFeatureDataset(cache_path, split),
            batch_size=cfg["batch_size"],
            shuffle=split == "train",
            num_workers=cfg["num_workers"],
            pin_memory=device.type == "cuda",
        )
        for split in ("train", "val", "test")
    }
    model_cfg = CoMemConfig(
        hidden_dim=cfg["hidden_dim"], memory_size=cfg["memory_size"], dropout=cfg["dropout"]
    )
    model = TextImageCoMem(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss()
    run_dir = PROJECT / cfg["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, history, best_f1 = 1, [], -1.0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        history = state.get("history", [])
        best_f1 = state.get("best_f1", -1.0)

    print(json.dumps({"device": str(device), "parameters": sum(p.numel() for p in model.parameters())}))
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        running, count = 0.0, 0
        aux_scale = cfg["aux_weight"] * min(1.0, epoch / max(cfg["aux_warmup_epochs"], 1))
        for text_x, image_x, labels in loaders["train"]:
            text_x, image_x, labels = text_x.to(device), image_x.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(text_x, image_x)
            joint_loss = criterion(output["logits"], labels)
            aux_loss = 0.5 * (
                criterion(output["text_logits"], labels) + criterion(output["image_logits"], labels)
            )
            loss = joint_loss + aux_scale * aux_loss
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            count += labels.size(0)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": running / count, "lr": scheduler.get_last_lr()[0]}
        for split in ("val", "test"):
            for key, value in evaluate(model, loaders[split], device).items():
                row[f"{split}_{key}"] = value
        history.append(row)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_f1": max(best_f1, row["val_macro_f1"]),
            "config": cfg,
            "history": history,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if row["val_macro_f1"] > best_f1:
            best_f1 = row["val_macro_f1"]
            torch.save(checkpoint, run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
