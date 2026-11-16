"""Validation-only regularization search; evaluate selected checkpoint on test once."""
import sys
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'scripts'))
from mvsa_comem.model import TextImageCoMem, CoMemConfig
from train import evaluate, seed_everything


def main():
    torch.set_num_threads(2)
    out = PROJECT / 'runs/regularization_search_seed0'
    out.mkdir(parents=True, exist_ok=True)
    payload = torch.load(PROJECT / 'data/features/minilm_dinov2_vitb14.pt', map_location='cpu')
    loaders = {}
    for split in ['train', 'val', 'test']:
        ix = torch.tensor([i for i, s in enumerate(payload['split']) if s == split])
        ds = TensorDataset(payload['text'][ix].float(), payload['image'][ix].float(), payload['labels'][ix].long())
        loaders[split] = DataLoader(ds, batch_size=64, shuffle=split == 'train')
    counts = torch.bincount(loaders['train'].dataset.tensors[2], minlength=3).float()
    candidates = [
        dict(dim=d, dropout=.4, wd=.02, smoothing=.05, balance=b, mixup=m)
        for d, b, m in [(64,0.,0.), (128,0.,0.), (64,.5,0.), (128,.5,0.),
                         (64,0.,.3), (128,0.,.3), (64,.5,.3), (128,.5,.3)]
    ]
    results = []
    global_best = -1.
    for number, cfg in enumerate(candidates):
        seed_everything(0)
        model = TextImageCoMem(CoMemConfig(hidden_dim=cfg['dim'], dropout=cfg['dropout']))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=cfg['wd'])
        weights = (counts.sum()/counts).pow(cfg['balance'])
        weights /= weights.mean()
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg['smoothing'])
        best, stale, history = -1., 0, []
        for epoch in range(1, 41):
            model.train()
            for text, image, labels in loaders['train']:
                optimizer.zero_grad(set_to_none=True)
                # Feature dropout regularizes both modality projections as well.
                text = nn.functional.dropout(text, p=.15, training=True)
                image = nn.functional.dropout(image, p=.15, training=True)
                lam = np.random.beta(cfg['mixup'], cfg['mixup']) if cfg['mixup'] else 1.
                perm = torch.randperm(len(labels))
                pred = model(lam*text+(1-lam)*text[perm], lam*image+(1-lam)*image[perm])
                def loss(logits):
                    return lam*criterion(logits, labels)+(1-lam)*criterion(logits, labels[perm])
                objective = loss(pred['logits']) + .1*(loss(pred['text_logits'])+loss(pred['image_logits']))/2
                objective.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            val = evaluate(model, loaders['val'], torch.device('cpu'))
            history.append(dict(epoch=epoch, **val))
            if val['macro_f1'] > best:
                best, stale = val['macro_f1'], 0
                if best > global_best:
                    global_best = best
                    torch.save(dict(model=model.state_dict(), config=cfg, epoch=epoch,
                                    validation=val, selection='validation macro_f1', candidate=number), out/'best.pt')
            else:
                stale += 1
            if stale >= 8:
                break
        row = dict(candidate=number, config=cfg, best_val_macro_f1=best, history=history)
        results.append(row)
        (out/'validation_search.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(json.dumps(dict(candidate=number, config=cfg, best_val_macro_f1=best)), flush=True)
    selected = torch.load(out/'best.pt', map_location='cpu')
    model = TextImageCoMem(CoMemConfig(hidden_dim=selected['config']['dim'], dropout=selected['config']['dropout']))
    model.load_state_dict(selected['model'])
    test = evaluate(model, loaders['test'], torch.device('cpu'))
    summary = dict(selected_candidate=selected['candidate'], config=selected['config'],
                   epoch=selected['epoch'], validation=selected['validation'], test=test,
                   protocol='8 fixed candidates, seed 0; validation macro-F1 selection; one final test evaluation',
                   parameters=sum(p.numel() for p in model.parameters()))
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
