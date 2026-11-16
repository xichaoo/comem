from pathlib import Path

from transformers import AutoModel, AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
target = PROJECT / "pretrained" / "bert-base-uncased"
target.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained(target)
model.save_pretrained(target)
print(f"Saved BERT to {target}")
