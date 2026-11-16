from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from mvsa_comem.data import CachedFeatureDataset
from mvsa_comem.model import CoMemConfig, TextImageCoMem


LABEL_NAMES = ["Positive", "Negative", "Neutral"]
STAGE_NAMES = [
    "Encoder concat.",
    "Projected concat.",
    "After CMR",
    "After MGF",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize CoMem stages on MVSA-Single.")
    parser.add_argument("--checkpoint", default="runs/comem_seed0/best.pt")
    parser.add_argument("--feature-cache", default="data/features/minilm_dinov2_vitb14.pt")
    parser.add_argument("--output", default="runs/comem_seed0/tsne_stages")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=30.0)
    return parser.parse_args()


def proportional_stratified_indices(labels: np.ndarray, count: int, seed: int):
    if count > len(labels):
        raise ValueError(f"Requested {count} samples from a test set of {len(labels)}.")
    classes, class_counts = np.unique(labels, return_counts=True)
    exact = count * class_counts / class_counts.sum()
    allocation = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - allocation))[: count - allocation.sum()]:
        allocation[index] += 1
    rng = np.random.default_rng(seed)
    selected = []
    for cls, take in zip(classes, allocation):
        candidates = np.flatnonzero(labels == cls)
        selected.extend(rng.choice(candidates, size=int(take), replace=False).tolist())
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


@torch.inference_mode()
def extract_stages(model: TextImageCoMem, dataset: CachedFeatureDataset, indices: np.ndarray):
    model.eval()
    text_raw = dataset.text.index_select(0, torch.from_numpy(indices))
    image_raw = dataset.image.index_select(0, torch.from_numpy(indices))
    labels = dataset.labels.index_select(0, torch.from_numpy(indices)).numpy()
    sample_ids = [dataset.ids[i] for i in indices.tolist()]

    text_pre = model.text_proj(text_raw)
    image_pre = model.image_proj(image_raw)
    routed_text, routed_image, _, _ = model.router(text_pre, image_pre)
    joined = torch.cat([routed_text, routed_image], dim=-1)
    gate_text, gate_image = model.fusion.gates(joined).chunk(2, dim=-1)

    stages = {
        STAGE_NAMES[0]: torch.cat([text_raw, image_raw], dim=-1),
        STAGE_NAMES[1]: torch.cat([text_pre, image_pre], dim=-1),
        STAGE_NAMES[2]: joined,
        STAGE_NAMES[3]: torch.cat([gate_text * routed_text, gate_image * routed_image], dim=-1),
    }
    return {name: tensor.cpu().numpy() for name, tensor in stages.items()}, labels, sample_ids


def representation_metrics(features: np.ndarray, labels: np.ndarray):
    x = StandardScaler().fit_transform(features)
    classes = np.unique(labels)
    centroids = np.stack([x[labels == cls].mean(axis=0) for cls in classes])
    within = np.mean([
        np.mean(np.sum((x[labels == cls] - centroids[index]) ** 2, axis=1))
        for index, cls in enumerate(classes)
    ])
    distances = pairwise_distances(centroids)
    between = distances[np.triu_indices_from(distances, k=1)].mean()
    return {
        "silhouette_cosine": float(silhouette_score(x, labels, metric="cosine")),
        "within_class_scatter": float(within),
        "between_centroid_distance": float(between),
        "separation_ratio": float((between ** 2) / max(within, 1e-12)),
    }


def embed_tsne(features: np.ndarray, seed: int, perplexity: float):
    x = StandardScaler().fit_transform(features)
    x = PCA(n_components=min(50, x.shape[0] - 1, x.shape[1]), random_state=seed).fit_transform(x)
    kwargs = dict(n_components=2, perplexity=perplexity, init="pca",
                  learning_rate=200.0, random_state=seed)
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = 2000
    else:
        kwargs["n_iter"] = 2000
    return TSNE(**kwargs).fit_transform(x)


def plot_embeddings(embeddings, labels, output_dir: Path):
    colors = ["#59D96B", "#4C78A8", "#E45756"]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.6), constrained_layout=True)
    handles = []
    for panel, (axis, stage) in enumerate(zip(axes.flat, STAGE_NAMES)):
        embedding = embeddings[stage]
        for cls, name in enumerate(LABEL_NAMES):
            mask = labels == cls
            handle = axis.scatter(
                embedding[mask, 0], embedding[mask, 1], s=28, c=colors[cls],
                label=name, alpha=0.84, edgecolors="white", linewidths=0.4,
            )
            if panel == 0:
                handles.append(handle)
        axis.set_title(f"({chr(97 + panel)}) {stage}", fontsize=14, pad=8)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#B8B8B8")
            spine.set_linewidth(0.8)
    fig.legend(handles, LABEL_NAMES, loc="upper center", ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 1.025))
    fig.savefig(output_dir / "mvsa_single_stage_tsne.png", dpi=350, bbox_inches="tight")
    fig.savefig(output_dir / "mvsa_single_stage_tsne.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = PROJECT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(PROJECT / args.checkpoint, map_location="cpu")
    train_config = checkpoint["config"]
    model = TextImageCoMem(CoMemConfig(
        hidden_dim=train_config["hidden_dim"], memory_size=train_config["memory_size"],
        dropout=train_config["dropout"],
    ))
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = CachedFeatureDataset(PROJECT / args.feature_cache, "test")
    indices = proportional_stratified_indices(dataset.labels.numpy(), args.samples, args.seed)
    features, labels, sample_ids = extract_stages(model, dataset, indices)

    metrics = {stage: representation_metrics(features[stage], labels) for stage in STAGE_NAMES}
    embeddings = {stage: embed_tsne(features[stage], args.seed, args.perplexity)
                  for stage in STAGE_NAMES}
    plot_embeddings(embeddings, labels, output_dir)

    np.savez_compressed(
        output_dir / "stage_representations.npz", labels=labels,
        sample_ids=np.asarray(sample_ids),
        **{f"features_{i}": features[stage] for i, stage in enumerate(STAGE_NAMES)},
        **{f"tsne_{i}": embeddings[stage] for i, stage in enumerate(STAGE_NAMES)},
    )
    with (output_dir / "separation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", *next(iter(metrics.values())).keys()])
        writer.writeheader()
        for stage in STAGE_NAMES:
            writer.writerow({"stage": stage, **metrics[stage]})
    metadata = {
        "checkpoint": str((PROJECT / args.checkpoint).resolve()),
        "selected_epoch": checkpoint["epoch"],
        "samples": len(labels),
        "sampling": "proportional stratified random sample from the official test split",
        "class_counts": {LABEL_NAMES[c]: int((labels == c).sum()) for c in range(3)},
        "seed": args.seed,
        "perplexity": args.perplexity,
        "stage_definition": {
            STAGE_NAMES[0]: "concatenated cached MiniLM and DINOv2 encoder features",
            STAGE_NAMES[1]: "concatenated modality-specific projected features",
            STAGE_NAMES[2]: "concatenated CMR-enhanced modality features",
            STAGE_NAMES[3]: "concatenated channel-gated modality features before compact projection",
        },
        "metrics": metrics,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
