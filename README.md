# MVSA-Single CoMem

Independent text-image multimodal fusion experiment on MVSA-Single.

## Protocol

- frozen text encoder: `sentence-transformers/all-MiniLM-L6-v2` (384-D)
- frozen image encoder: DINOv2 ViT-B/14 (768-D)
- trainable fusion: shared conditional memory routing and channel-wise modality gating
- task: three-way sentiment classification
- model selection: validation macro-F1; the test set is evaluated once for each saved epoch summary

All raw data, processed manifests, cached features, logs, checkpoints, and results stay inside this project.

## Run

```powershell
$env:PYTHONPATH = "$PWD/src"
python scripts/prepare_mvsa.py
python scripts/extract_features.py
python scripts/train.py --config configs/comem_seed0.json
```

The preparation script records the exact filtering and split protocol in `data/processed/protocol.json`.
