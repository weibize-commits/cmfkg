from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer


os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = ROOT / "paper_revision" / "ner_benchmark"
TEST_BIO = BENCH_ROOT / "test_set" / "test.bio"
LABELS_FILE = ROOT / "labels.txt"
OUTPUT_DIR = BENCH_ROOT / "outputs"

MAX_LENGTH = 256
BATCH_SIZE = 16

MODELS = {
    "SikuRoBERTa-CRF_archived": {
        "checkpoint_dir": ROOT / "models" / "sikuroberta_crf",
        "base_model": "SIKU-BERT/sikuroberta",
    },
    "Chinese-RoBERTa-CRF_archived": {
        "checkpoint_dir": ROOT / "models" / "chinese_roberta_crf",
        "base_model": "hfl/chinese-roberta-wwm-ext",
    },
}


def load_labels() -> tuple[list[str], dict[str, int], dict[int, str]]:
    labels = [line.strip() for line in LABELS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return labels, {x: i for i, x in enumerate(labels)}, {i: x for i, x in enumerate(labels)}


def load_bio(path: Path) -> list[tuple[list[str], list[str]]]:
    sentences: list[tuple[list[str], list[str]]] = []
    chars: list[str] = []
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            if chars:
                sentences.append((chars, labels))
                chars, labels = [], []
            continue
        parts = line.split("\t")
        if len(parts) == 2:
            chars.append(parts[0])
            labels.append(parts[1])
    if chars:
        sentences.append((chars, labels))
    return sentences


class NERDataset(Dataset):
    def __init__(self, sentences, tokenizer, label2id, max_length=MAX_LENGTH):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        chars, labels = self.sentences[idx]
        if len(chars) > self.max_length - 2:
            chars = chars[: self.max_length - 2]
            labels = labels[: self.max_length - 2]
        tokens = ["[CLS]"] + chars + ["[SEP]"]
        token_labels = ["O"] + labels + ["O"]
        unk_id = self.tokenizer.unk_token_id
        input_ids = [
            token_id if token_id is not None else unk_id
            for token_id in self.tokenizer.convert_tokens_to_ids(tokens)
        ]
        label_ids = [self.label2id.get(label, 0) for label in token_labels]
        attention_mask = [1] * len(input_ids)
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            label_ids += [0] * pad_len
            attention_mask += [0] * pad_len
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


class BertCRF(nn.Module):
    def __init__(self, model_name_or_path: str, num_labels: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.classifier(self.dropout(outputs.last_hidden_state))
        return self.crf.decode(emissions, mask=attention_mask.bool())


def entities_from_tags(tags: list[str], sent_idx: int) -> set[tuple[int, int, int, str]]:
    ents = set()
    start = None
    label = None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-"):
            if start is not None:
                ents.add((sent_idx, start, i, label))
            start = i
            label = tag[2:]
        elif tag.startswith("I-") and start is not None and tag[2:] == label:
            continue
        else:
            if start is not None:
                ents.add((sent_idx, start, i, label))
            start = None
            label = None
    return ents


def prf(pred: set[tuple[Any, ...]], gold: set[tuple[Any, ...]]) -> dict[str, Any]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_model(name: str, checkpoint_dir: Path, base_model: str) -> dict[str, Any]:
    labels, label2id, id2label = load_labels()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertCRF(base_model, len(label2id)).to(device)
    checkpoint = torch.load(checkpoint_dir / "best_model.pt", weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_sents = load_bio(TEST_BIO)
    dataset = NERDataset(test_sents, tokenizer, label2id)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    pred_entities = set()
    gold_entities = set()
    sent_offset = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_tensor = batch["labels"].to(device)
            preds = model(input_ids, attention_mask)
            for i, pred in enumerate(preds):
                mask = attention_mask[i].bool()
                true_ids = labels_tensor[i][mask].cpu().tolist()
                pred_ids = pred[1:-1] if len(pred) > 1 else pred
                true_ids = true_ids[1:-1] if len(true_ids) > 1 else true_ids
                n = min(len(pred_ids), len(true_ids))
                pred_tags = [id2label[x] for x in pred_ids[:n]]
                gold_tags = [id2label[x] for x in true_ids[:n]]
                pred_entities |= entities_from_tags(pred_tags, sent_offset)
                gold_entities |= entities_from_tags(gold_tags, sent_offset)
                sent_offset += 1

    overall = prf(pred_entities, gold_entities)
    per_label = {}
    for label in sorted({x[3] for x in gold_entities} | {x[3] for x in pred_entities}):
        per_label[label] = prf(
            {x for x in pred_entities if x[3] == label},
            {x for x in gold_entities if x[3] == label},
        )
    return {
        "method": name,
        "base_model": base_model,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_best_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_val_f1": float(checkpoint.get("val_f1", float("nan"))),
        "max_length": MAX_LENGTH,
        "test_passages": len(test_sents),
        **overall,
        "per_label": per_label,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    for name, cfg in MODELS.items():
        try:
            results.append(evaluate_model(name, cfg["checkpoint_dir"], cfg["base_model"]))
        except Exception as exc:
            failures.append(
                {
                    "method": name,
                    "base_model": cfg["base_model"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    payload = {"results": results, "failures": failures}
    (OUTPUT_DIR / "archived_bert_crf_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (OUTPUT_DIR / "archived_bert_crf_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "base_model",
                "test_passages",
                "max_length",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "checkpoint_best_epoch",
                "checkpoint_best_val_f1",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
