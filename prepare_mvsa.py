from __future__ import annotations

import csv
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
MEDIA = PROJECT / "data" / "processed" / "media"
MANIFEST = PROJECT / "data" / "processed" / "manifest.jsonl"
PROTOCOL = PROJECT / "data" / "processed" / "protocol.json"

LABEL_TO_ID = {"positive": 0, "negative": 1, "neutral": 2}
SPLIT_COUNTS = {
    "positive": {"train": 2146, "val": 268, "test": 269},
    "neutral": {"train": 376, "val": 47, "test": 47},
    "negative": {"train": 1086, "val": 135, "test": 137},
}


def read_annotations():
    with (RAW / "labelResultAll.txt").open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return {row["ID"]: tuple(row["text,image"].split(",")) for row in rows}


def main():
    annotations = read_annotations()
    MEDIA.mkdir(parents=True, exist_ok=True)
    samples, removed = [], []
    with zipfile.ZipFile(RAW / "data.zip") as archive:
        image_ids = sorted(
            (Path(name).stem for name in archive.namelist() if name.lower().endswith(".jpg")),
            key=int,
        )
        for sample_id in image_ids:
            text_label, image_label = annotations[sample_id]
            if {text_label, image_label} == {"positive", "negative"}:
                removed.append(sample_id)
                continue
            label = text_label if text_label != "neutral" else image_label
            image_member, text_member = f"data/{sample_id}.jpg", f"data/{sample_id}.txt"
            image_path, text_path = MEDIA / f"{sample_id}.jpg", MEDIA / f"{sample_id}.txt"
            if not image_path.exists():
                image_path.write_bytes(archive.read(image_member))
            if not text_path.exists():
                text_path.write_bytes(archive.read(text_member))
            text = text_path.read_text(encoding="utf-8", errors="replace").strip()
            samples.append(
                {
                    "id": sample_id,
                    "text": text,
                    "image": str(image_path.relative_to(PROJECT)).replace("\\", "/"),
                    "text_label": text_label,
                    "image_label": image_label,
                    "label_name": label,
                    "label": LABEL_TO_ID[label],
                }
            )

    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["label_name"]].append(sample)
    rng = random.Random(0)
    for label, items in grouped.items():
        rng.shuffle(items)
        cursor = 0
        for split in ("train", "val", "test"):
            count = SPLIT_COUNTS[label][split]
            for sample in items[cursor : cursor + count]:
                sample["split"] = split
            cursor += count
        assert cursor == len(items), (label, cursor, len(items))
    samples.sort(key=lambda item: int(item["id"]))
    with MANIFEST.open("w", encoding="utf-8") as handle:
        for sample in samples:
            # ASCII escaping also protects JSONL from Unicode line separators in tweets.
            handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
    stats = Counter((sample["split"], sample["label_name"]) for sample in samples)
    protocol = {
        "dataset": "MVSA-Single",
        "raw_annotations": len(annotations),
        "available_pairs": len(image_ids),
        "excluded_direct_positive_negative_conflicts": len(removed),
        "included_pairs": len(samples),
        "split_seed": 0,
        "split_rule": "stratified 8:1:1 with fixed per-class counts",
        "label_rule": "agreement, or the non-neutral label when exactly one view is neutral",
        "label_to_id": LABEL_TO_ID,
        "counts": {f"{split}/{label}": value for (split, label), value in sorted(stats.items())},
        "excluded_ids": removed,
    }
    PROTOCOL.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in protocol.items() if key != "excluded_ids"}, indent=2))


if __name__ == "__main__":
    main()
