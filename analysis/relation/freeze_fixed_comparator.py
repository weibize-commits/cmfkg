from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "paper_revision" / "relation_improvement"
DATA_DIR = EXPERIMENT_DIR / "data"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
MODEL_DIR = EXPERIMENT_DIR / "models"
CONFIG_DIR = EXPERIMENT_DIR / "configs"
PREPARE_PATH = EXPERIMENT_DIR / "analysis" / "prepare_and_tune_local.py"
BUILDER_PATH = ROOT / "paper_revision" / "tools" / "build_relation_baseline_package.py"
SEED = 20260731


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    builder = load_module("relation_builder_comparator", BUILDER_PATH)
    prepare = load_module("relation_prepare_comparator", PREPARE_PATH)
    split_records = {
        split: {
            "passages": read_jsonl(DATA_DIR / f"{split}_passages.jsonl"),
            "candidates": read_jsonl(DATA_DIR / f"{split}_candidates.jsonl"),
        }
        for split in ["train", "dev", "test"]
    }
    train_gold = read_jsonl(DATA_DIR / "train_gold.jsonl")
    dev_gold = read_jsonl(DATA_DIR / "dev_gold.jsonl")
    train_gold_keys = {builder.relation_key(row) for row in train_gold}
    dev_gold_keys = {builder.relation_key(row) for row in dev_gold}
    texts: dict[str, dict[str, Any]] = {}
    for bundle in split_records.values():
        for row in bundle["passages"]:
            texts[str(row["sample_id"])] = {"frozen_text": str(row["frozen_text"])}

    train_candidates = split_records["train"]["candidates"]
    dev_candidates = split_records["dev"]["candidates"]
    test_candidates = split_records["test"]["candidates"]
    train_features = [builder.feature_text(row, texts) for row in train_candidates]
    dev_features = [builder.feature_text(row, texts) for row in dev_candidates]
    test_features = [builder.feature_text(row, texts) for row in test_candidates]
    train_y = np.array(
        [builder.relation_key(row) in train_gold_keys for row in train_candidates],
        dtype=int,
    )
    dev_y = np.array(
        [builder.relation_key(row) in dev_gold_keys for row in dev_candidates],
        dtype=int,
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 3),
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=SEED,
                ),
            ),
        ]
    )
    model.fit(train_features, train_y)
    dev_scores = model.predict_proba(dev_features)[:, 1]
    threshold, precision, recall, f1 = prepare.best_global_threshold(dev_y, dev_scores)
    test_scores = model.predict_proba(test_features)[:, 1]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "fixed_charngram_comparator.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    score_records = [
        builder.prediction_record(
            candidate,
            "fixed_charngram_comparator_score",
            float(score),
            "score_only_test_labels_not_evaluated",
        )
        for candidate, score in zip(test_candidates, test_scores)
    ]
    score_path = OUTPUT_DIR / "fixed_charngram_test_scores_blinded.jsonl"
    write_jsonl(score_path, score_records)

    freeze = {
        "status": "fixed_comparator_frozen_before_confirmatory_test_evaluation",
        "method": "character_ngram_logistic_regression",
        "features": "legacy structured relation text used in the 2026-07-30 baseline",
        "tfidf_analyzer": "char",
        "ngram_range": [1, 3],
        "min_df": 2,
        "classifier": "LogisticRegression",
        "class_weight": "balanced",
        "solver": "liblinear",
        "threshold_selected_on_dev": threshold,
        "development_precision": precision,
        "development_recall": recall,
        "development_f1": f1,
        "train_passages": len(split_records["train"]["passages"]),
        "development_passages": len(split_records["dev"]["passages"]),
        "confirmatory_passages": len(split_records["test"]["passages"]),
        "model_sha256": sha256(model_path),
        "confirmatory_scores_sha256": sha256(score_path),
        "confirmatory_test_evaluated": False,
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "fixed_comparator_freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(freeze, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
