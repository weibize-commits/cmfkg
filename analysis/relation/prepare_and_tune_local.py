from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PACKAGE = ROOT / "paper_revision" / "relation_baselines"
EXPERIMENT_DIR = ROOT / "paper_revision" / "relation_improvement"
DATA_DIR = EXPERIMENT_DIR / "data"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
MODEL_DIR = EXPERIMENT_DIR / "models"
CONFIG_DIR = EXPERIMENT_DIR / "configs"
BUILDER_PATH = ROOT / "paper_revision" / "tools" / "build_relation_baseline_package.py"

SEED = 20260731
DEV_PASSAGES = 50
NEW_TEST_VERSION = "relation_confirmatory_65_human_adjudicated_20260731"


def load_builder():
    spec = importlib.util.spec_from_file_location("relation_package_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
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


def relation_counts(
    ids: list[str],
    records_by_sample: dict[str, list[dict[str, Any]]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sid in ids:
        counts.update(str(row["relation"]) for row in records_by_sample.get(sid, []))
    return counts


def choose_distribution_balanced_dev(
    remaining_ids: list[str],
    records_by_sample: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], list[str], int, float]:
    """Use labels only for a deterministic stratified split, never for model tuning."""
    total = relation_counts(remaining_ids, records_by_sample)
    target_fraction = DEV_PASSAGES / len(remaining_ids)
    relation_types = sorted(total)
    best: tuple[float, int, list[str]] | None = None
    for offset in range(5000):
        seed = SEED + offset
        rng = random.Random(seed)
        candidate = remaining_ids[:]
        rng.shuffle(candidate)
        dev_ids = sorted(candidate[:DEV_PASSAGES])
        dev_counts = relation_counts(dev_ids, records_by_sample)
        distribution_error = sum(
            abs(dev_counts[relation] - total[relation] * target_fraction)
            / max(1.0, total[relation] * target_fraction)
            for relation in relation_types
        )
        zero_penalty = sum(
            5.0
            for relation in relation_types
            if total[relation] >= 5 and dev_counts[relation] == 0
        )
        score = distribution_error + zero_penalty
        if best is None or score < best[0]:
            best = (score, seed, dev_ids)
    assert best is not None
    test_ids = sorted(set(remaining_ids) - set(best[2]))
    return best[2], test_ids, best[1], best[0]


def serialise_gold(builder, records: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    return [
        builder.serialise_relation(record, f"{split_name.upper()}-GOLD-{index:05d}")
        for index, record in enumerate(records, start=1)
    ]


def prediction_key(builder, record: dict[str, Any]) -> tuple[Any, ...]:
    return builder.relation_key(record)


def rich_feature_text(builder, candidate: dict[str, Any], texts: dict[str, dict[str, Any]]) -> str:
    sid = str(candidate["sample_id"])
    text = str(texts[sid]["frozen_text"])
    head = candidate["head"]
    tail = candidate["tail"]
    relation = str(candidate["relation"])
    distance = builder.relation_distance(head, tail)
    keep, heuristic_score, heuristic_reason = builder.heuristic_score(candidate, texts)
    if distance <= 5:
        distance_bin = "D00_05"
    elif distance <= 15:
        distance_bin = "D06_15"
    elif distance <= 30:
        distance_bin = "D16_30"
    elif distance <= 60:
        distance_bin = "D31_60"
    elif distance <= 100:
        distance_bin = "D61_100"
    elif distance <= 150:
        distance_bin = "D101_150"
    else:
        distance_bin = "D151_PLUS"
    order = "HEAD_BEFORE_TAIL" if int(head["start_char"]) <= int(tail["start_char"]) else "TAIL_BEFORE_HEAD"
    between = builder.relation_between(text, head, tail)[:180]
    window = builder.relation_window(text, head, tail, width=100)[:500]
    structured = " ".join(
        [
            f"REL_{relation}",
            f"REL_{relation}",
            f"HEADTYPE_{head['type']}",
            f"TAILTYPE_{tail['type']}",
            distance_bin,
            order,
            f"HEURISTIC_{heuristic_reason}",
            f"HEURISTIC_KEEP_{int(bool(keep))}",
            f"HEURISTIC_SCORE_{int(round(float(heuristic_score) * 10))}",
            f"HEAD_TEXT_{head['text']}",
            f"TAIL_TEXT_{tail['text']}",
        ]
    )
    return f"{structured} [BETWEEN] {between} [WINDOW] {window}"


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return float(precision), float(recall), float(f1)


def best_global_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float, float, float]:
    best = (-1.0, -1.0, -1.0, 0.5)
    for threshold in np.arange(0.05, 0.951, 0.025):
        precision, recall, f1 = prf(y_true, scores >= threshold)
        candidate = (f1, recall, precision, float(round(threshold, 3)))
        if candidate > best:
            best = candidate
    return best[3], best[2], best[1], best[0]


def relation_specific_thresholds(
    candidates: list[dict[str, Any]],
    y_true: np.ndarray,
    scores: np.ndarray,
    global_threshold: float,
) -> tuple[dict[str, float], tuple[float, float, float]]:
    relation_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        relation_to_indices[str(candidate["relation"])].append(index)
    thresholds: dict[str, float] = {}
    for relation, indices in relation_to_indices.items():
        relation_y = y_true[indices]
        relation_scores = scores[indices]
        positives = int(relation_y.sum())
        negatives = int(len(relation_y) - positives)
        if positives < 5 or negatives < 5:
            thresholds[relation] = global_threshold
            continue
        threshold, _, _, _ = best_global_threshold(relation_y, relation_scores)
        thresholds[relation] = threshold
    predictions = np.array(
        [
            score >= thresholds.get(str(candidate["relation"]), global_threshold)
            for candidate, score in zip(candidates, scores)
        ],
        dtype=bool,
    )
    return thresholds, prf(y_true, predictions)


def prediction_records(
    builder,
    candidates: list[dict[str, Any]],
    scores: np.ndarray,
    thresholds: dict[str, float],
    method: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate, score in zip(candidates, scores):
        threshold = thresholds[str(candidate["relation"])]
        if score < threshold:
            continue
        records.append(
            builder.prediction_record(
                candidate,
                method,
                float(score),
                f"relation_threshold={threshold:.3f}",
            )
        )
    return records


def main() -> None:
    for path in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, CONFIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    builder = load_builder()
    texts, entities_by_sample, gold_records, audit_bundle = builder.build_gold_package()
    formal_usable_ids = sorted(
        sid
        for sid in texts
        if sid.startswith("HR-") and audit_bundle["status"].get(sid) == "usable"
    )
    pilot_usable_ids = sorted(
        sid
        for sid in texts
        if sid.startswith("PILOT-") and audit_bundle["status"].get(sid) == "usable"
    )
    old_test_ids = sorted(
        json.loads(line)["sample_id"]
        for line in (SOURCE_PACKAGE / "test_set" / "frozen_test_300.jsonl").open(
            encoding="utf-8"
        )
        if line.strip()
    )
    remaining_ids = sorted(set(formal_usable_ids) - set(old_test_ids))
    if (
        len(old_test_ids) != 300
        or len(remaining_ids) != 115
        or len(pilot_usable_ids) != 26
    ):
        raise RuntimeError(
            "Expected 300 diagnostic formal passages, 115 remaining formal passages, "
            f"and 26 usable pilot passages; found {len(old_test_ids)}, "
            f"{len(remaining_ids)}, and {len(pilot_usable_ids)}"
        )

    records_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in gold_records:
        records_by_sample[str(record["sample_id"])].append(record)

    dev_ids, confirmatory_ids, split_seed, split_score = choose_distribution_balanced_dev(
        remaining_ids,
        records_by_sample,
    )
    split_ids = {
        "train": sorted(old_test_ids + pilot_usable_ids),
        "dev": dev_ids,
        "test": confirmatory_ids,
    }

    all_gold_keys = {builder.relation_key(record) for record in gold_records}
    split_candidates: dict[str, list[dict[str, Any]]] = {}
    split_gold: dict[str, list[dict[str, Any]]] = {}
    for split_name, ids in split_ids.items():
        id_set = set(ids)
        candidates = builder.generate_candidates(ids, texts, entities_by_sample)
        gold = [record for record in gold_records if str(record["sample_id"]) in id_set]
        samples = [
            {
                "sample_id": sid,
                "split": split_name,
                "split_seed": split_seed,
                "version": NEW_TEST_VERSION,
                "frozen_text": texts[sid]["frozen_text"],
                "raw_text_sha256": texts[sid]["raw_text_sha256"],
            }
            for sid in ids
        ]
        split_candidates[split_name] = candidates
        split_gold[split_name] = gold
        write_jsonl(DATA_DIR / f"{split_name}_passages.jsonl", samples)
        write_jsonl(DATA_DIR / f"{split_name}_candidates.jsonl", candidates)
        write_jsonl(
            DATA_DIR / f"{split_name}_gold.jsonl",
            serialise_gold(builder, gold, split_name),
        )

    train_candidates = split_candidates["train"]
    dev_candidates = split_candidates["dev"]
    train_features = [rich_feature_text(builder, row, texts) for row in train_candidates]
    dev_features = [rich_feature_text(builder, row, texts) for row in dev_candidates]
    train_y = np.array(
        [builder.relation_key(row) in all_gold_keys for row in train_candidates],
        dtype=int,
    )
    dev_y = np.array(
        [builder.relation_key(row) in all_gold_keys for row in dev_candidates],
        dtype=int,
    )

    search_rows: list[dict[str, Any]] = []
    best_bundle: dict[str, Any] | None = None
    for ngram_range in [(1, 3), (2, 4), (2, 5)]:
        for min_df in [1, 2]:
            for c_value in [0.5, 1.0, 2.0, 4.0]:
                for class_weight in [None, "balanced"]:
                    model = Pipeline(
                        [
                            (
                                "tfidf",
                                TfidfVectorizer(
                                    analyzer="char",
                                    ngram_range=ngram_range,
                                    min_df=min_df,
                                    sublinear_tf=True,
                                    max_features=250_000,
                                ),
                            ),
                            (
                                "clf",
                                LogisticRegression(
                                    C=c_value,
                                    class_weight=class_weight,
                                    max_iter=2000,
                                    solver="liblinear",
                                    random_state=SEED,
                                ),
                            ),
                        ]
                    )
                    model.fit(train_features, train_y)
                    dev_scores = model.predict_proba(dev_features)[:, 1]
                    threshold, precision, recall, f1 = best_global_threshold(dev_y, dev_scores)
                    relation_thresholds, relation_metrics = relation_specific_thresholds(
                        dev_candidates,
                        dev_y,
                        dev_scores,
                        threshold,
                    )
                    rel_precision, rel_recall, rel_f1 = relation_metrics
                    use_relation_thresholds = rel_f1 > f1
                    selected_f1 = rel_f1 if use_relation_thresholds else f1
                    selected_recall = rel_recall if use_relation_thresholds else recall
                    selected_precision = rel_precision if use_relation_thresholds else precision
                    row = {
                        "ngram_min": ngram_range[0],
                        "ngram_max": ngram_range[1],
                        "min_df": min_df,
                        "C": c_value,
                        "class_weight": class_weight or "none",
                        "global_threshold": threshold,
                        "global_precision": precision,
                        "global_recall": recall,
                        "global_f1": f1,
                        "relation_thresholds_used": use_relation_thresholds,
                        "selected_precision": selected_precision,
                        "selected_recall": selected_recall,
                        "selected_f1": selected_f1,
                    }
                    search_rows.append(row)
                    rank = (selected_f1, selected_recall, selected_precision)
                    if best_bundle is None or rank > best_bundle["rank"]:
                        thresholds = (
                            relation_thresholds
                            if use_relation_thresholds
                            else {
                                str(candidate["relation"]): threshold
                                for candidate in dev_candidates
                            }
                        )
                        best_bundle = {
                            "rank": rank,
                            "row": row,
                            "model": model,
                            "dev_scores": dev_scores,
                            "thresholds": thresholds,
                        }

    assert best_bundle is not None
    pd.DataFrame(search_rows).sort_values(
        ["selected_f1", "selected_recall", "selected_precision"],
        ascending=False,
    ).to_csv(OUTPUT_DIR / "local_model_dev_search.csv", index=False, encoding="utf-8-sig")

    with (MODEL_DIR / "local_symbolic_text_model.pkl").open("wb") as handle:
        pickle.dump(best_bundle["model"], handle)
    dev_predictions = prediction_records(
        builder,
        dev_candidates,
        best_bundle["dev_scores"],
        best_bundle["thresholds"],
        "cmfkg_local_symbolic_text_dev",
    )
    write_jsonl(OUTPUT_DIR / "cmfkg_local_dev_predictions.jsonl", dev_predictions)

    test_candidates = split_candidates["test"]
    test_features = [rich_feature_text(builder, row, texts) for row in test_candidates]
    test_scores = best_bundle["model"].predict_proba(test_features)[:, 1]
    score_records: list[dict[str, Any]] = []
    for candidate, score in zip(test_candidates, test_scores):
        record = builder.prediction_record(
            candidate,
            "cmfkg_local_symbolic_text_frozen_score",
            float(score),
            "score_only_test_labels_not_evaluated",
        )
        score_records.append(record)
    write_jsonl(OUTPUT_DIR / "confirmatory_test_scores_blinded.jsonl", score_records)

    split_rows: list[dict[str, Any]] = []
    for split_name, ids in split_ids.items():
        counts = relation_counts(ids, records_by_sample)
        split_rows.append(
            {
                "split": split_name,
                "passages": len(ids),
                "candidate_pairs": len(split_candidates[split_name]),
                "gold_relations": len(split_gold[split_name]),
                "relation_distribution": json.dumps(
                    dict(sorted(counts.items())),
                    ensure_ascii=False,
                ),
            }
        )
    pd.DataFrame(split_rows).to_csv(
        OUTPUT_DIR / "split_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_gold_path = DATA_DIR / "test_gold.jsonl"
    freeze = {
        "status": "local_model_frozen_before_confirmatory_test_evaluation",
        "experiment_version": NEW_TEST_VERSION,
        "independent_unit": "passage",
        "split_strategy": (
            "The previously inspected 300-passage formal benchmark and 26 pilot "
            "passages are used only for training. The remaining 115 previously "
            "uninspected usable formal passages were split into 50 development and "
            "65 confirmatory passages using deterministic relation-distribution "
            "balancing. Gold labels were used only to stratify."
        ),
        "split_seed": split_seed,
        "split_balance_score": split_score,
        "train_passages": len(split_ids["train"]),
        "dev_passages": len(split_ids["dev"]),
        "confirmatory_test_passages": len(split_ids["test"]),
        "selected_local_model": best_bundle["row"],
        "relation_thresholds": best_bundle["thresholds"],
        "model_path": str((MODEL_DIR / "local_symbolic_text_model.pkl").relative_to(ROOT)),
        "model_sha256": sha256(MODEL_DIR / "local_symbolic_text_model.pkl"),
        "confirmatory_test_passages_sha256": sha256(DATA_DIR / "test_passages.jsonl"),
        "confirmatory_test_candidates_sha256": sha256(DATA_DIR / "test_candidates.jsonl"),
        "confirmatory_test_gold_sha256": sha256(test_gold_path),
        "confirmatory_test_scores_sha256": sha256(
            OUTPUT_DIR / "confirmatory_test_scores_blinded.jsonl"
        ),
        "test_evaluated": False,
        "next_step": (
            "Tune the LLM triage policy on the development split, freeze it, then run "
            "one confirmatory evaluation on the 65-passage test."
        ),
    }
    (CONFIG_DIR / "local_model_freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(freeze, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
