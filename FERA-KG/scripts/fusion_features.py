from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fera_kg.evaluation import full_relation_verification_metrics  # noqa: E402
from fera_kg.schema_guard import (  # noqa: E402
    LABELS,
    apply_schema_guard,
    inspect_relation_signature,
)


DATA_ROOT = PROJECT_ROOT.parent / "confidential_review_data" / "model_development"
OOF_ROOT = PROJECT_ROOT.parent / "confidential_review_data" / "oof"
TASKS = ["graph_anchor_wide", "fera_ecrp", "fera_ecrp_sourcebal", "text_entity"]
LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}
FROZEN_ONTOLOGY_GROUP_ROUTING = {
    "auxiliary_long_tail": "oof_logistic_stacker",
    "ingredient_condition": "fera_ecrp_sourcebal",
    "ingredient_effect": "oof_logistic_stacker",
    "ingredient_taboo": "fera_ecrp_sourcebal",
    "nature_flavor_meridian": "oof_logistic_stacker",
    "recipe_condition": "oof_logistic_stacker",
    "recipe_cooking_method": "oof_soft_expert_router",
    "recipe_ingredient": "fixed_dual_evidence",
}
FROZEN_PRESENT_REJECT_THRESHOLD = 0.70
FROZEN_OTHER_REJECT_THRESHOLD = 0.80


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "items": metrics["four_class"]["items"],
        "four_class_macro_f1": metrics["four_class"]["macro_f1"],
        "four_class_accuracy": metrics["four_class"]["accuracy"],
        "three_decision_macro_f1": metrics["three_decision"]["macro_f1"],
        "three_decision_accuracy": metrics["three_decision"]["accuracy"],
        "abstain_recall": metrics["abstain_recall"],
        "gold_abstain_false_present_rate": metrics["gold_abstain_false_present_rate"],
        "four_class_per_class": metrics["four_class"]["per_class"],
        "four_class_confusion_matrix": metrics["four_class"]["confusion_matrix"],
        "three_decision_per_class": metrics["three_decision"]["per_class"],
    }


def evaluate(rows: list[dict[str, Any]], predicted: list[str]) -> dict[str, Any]:
    return compact(
        full_relation_verification_metrics(
            [str(row["label"]) for row in rows], predicted
        )
    )


def guarded_predictions(
    rows: list[dict[str, Any]], probabilities: np.ndarray
) -> list[str]:
    predicted = []
    for row, values in zip(rows, probabilities.tolist()):
        guarded, _ = apply_schema_guard(
            values,
            str(row["relation"]),
            str(row["head_type"]),
            str(row["tail_type"]),
        )
        predicted.append(LABELS[int(np.argmax(guarded))])
    return predicted


def load_rows(split: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(DATA_ROOT / f"{split}.jsonl")
    paths = read_jsonl(DATA_ROOT / "graph_paths_v1" / f"{split}.jsonl")
    path_by_id = {str(row["item_id"]): row for row in paths}
    if {str(row["item_id"]) for row in rows} != set(path_by_id):
        raise ValueError(f"item/path mismatch for {split}")
    return rows, path_by_id


def oof_probabilities(task: str) -> dict[str, list[float]]:
    path = OOF_ROOT / f"{task}_oof_predictions.jsonl"
    rows = read_jsonl(path)
    return {
        str(row["item_id"]): [float(row["probabilities"][label]) for label in LABELS]
        for row in rows
    }


def entropy(values: Iterable[float]) -> float:
    return -sum(value * math.log(max(value, 1.0e-12)) for value in values)


def feature_frame(
    rows: list[dict[str, Any]],
    path_by_id: dict[str, dict[str, Any]],
    probabilities: dict[str, dict[str, list[float]]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        item_id = str(row["item_id"])
        graph = path_by_id[item_id]
        local = graph.get("local_inventory", {})
        signature = inspect_relation_signature(
            str(row["relation"]), str(row["head_type"]), str(row["tail_type"])
        )
        text = str(row.get("passage_text", ""))
        head = str(row.get("head_text", ""))
        tail = str(row.get("tail_text", ""))
        record: dict[str, Any] = {
            "relation": str(row["relation"]),
            "relation_group": str(row["relation_group"]),
            "head_type": str(row["head_type"]),
            "tail_type": str(row["tail_type"]),
            "dataset_origin": str(row.get("dataset_origin", "")),
            "provenance_resolution": str(row.get("provenance_resolution", "")),
            "signature_known": float(signature.signature_known),
            "signature_valid": float(signature.signature_valid),
            "text_length": float(len(text)),
            "head_length": float(len(head)),
            "tail_length": float(len(tail)),
            "head_occurrences": float(text.count(head)) if head else 0.0,
            "tail_occurrences": float(text.count(tail)) if tail else 0.0,
            "span_distance": float(abs(int(row.get("head_start", 0)) - int(row.get("tail_start", 0)))),
            "head_position": float(int(row.get("head_start", 0)) / max(1, len(text))),
            "tail_position": float(int(row.get("tail_start", 0)) / max(1, len(text))),
            "replacement_character_rate": float(text.count("�") / max(1, len(text))),
            "path_count": float(graph.get("path_count", 0)),
            "passage_conditioned_path_count": float(
                graph.get("passage_conditioned_path_count", 0)
            ),
            "semantic_only_path_count": float(graph.get("semantic_only_path_count", 0)),
            "actual_typed_mentions": float(local.get("actual_typed_mentions", 0)),
            "head_in_actual_inventory": float(local.get("head_in_actual_inventory", False)),
            "tail_in_actual_inventory": float(local.get("tail_in_actual_inventory", False)),
            "enumeration_truncated": float(graph.get("enumeration_truncated", False)),
        }
        task_values = []
        for task in TASKS:
            values = probabilities[task][item_id]
            task_values.append(values)
            for label, value in zip(LABELS, values):
                record[f"{task}__p_{label}"] = float(value)
            ordered = sorted(values, reverse=True)
            record[f"{task}__entropy"] = entropy(values)
            record[f"{task}__margin"] = float(ordered[0] - ordered[1])
            record[f"{task}__judgeability"] = float(values[0] + values[1])
            record[f"{task}__support_given_judgeable"] = float(
                values[0] / max(values[0] + values[1], 1.0e-12)
            )
        for left_index, left in enumerate(TASKS):
            for right_index in range(left_index + 1, len(TASKS)):
                right = TASKS[right_index]
                record[f"l1__{left}__{right}"] = float(
                    sum(
                        abs(task_values[left_index][index] - task_values[right_index][index])
                        for index in range(len(LABELS))
                    )
                )
        record["expert_vote_count"] = float(
            len({int(np.argmax(values)) for values in task_values})
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def make_pipeline(frame: pd.DataFrame, *, c_value: float) -> Pipeline:
    numeric = [column for column in frame.columns if is_numeric_dtype(frame[column])]
    categorical = [column for column in frame.columns if column not in numeric]
    processor = ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )
    return Pipeline(
        [
            ("features", processor),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=20261001,
                ),
            ),
        ]
    )


def align_probabilities(
    raw: np.ndarray, classes: Iterable[Any], desired: list[Any]
) -> np.ndarray:
    classes = list(classes)
    aligned = np.zeros((raw.shape[0], len(desired)), dtype=float)
    for destination, label in enumerate(desired):
        if label in classes:
            aligned[:, destination] = raw[:, classes.index(label)]
    totals = aligned.sum(axis=1, keepdims=True)
    return aligned / np.maximum(totals, 1.0e-12)


def expert_matrix(
    rows: list[dict[str, Any]], probabilities: dict[str, dict[str, list[float]]]
) -> np.ndarray:
    return np.asarray(
        [
            [probabilities[task][str(row["item_id"])] for task in TASKS]
            for row in rows
        ],
        dtype=float,
    )


def soft_router_probabilities(
    manager: Pipeline,
    frame: pd.DataFrame,
    expert_values: np.ndarray,
) -> np.ndarray:
    raw = manager.predict_proba(frame)
    weights = align_probabilities(
        raw, manager.named_steps["classifier"].classes_, list(range(len(TASKS)))
    )
    return (weights[:, :, None] * expert_values).sum(axis=1)


def stack_probabilities(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(frame)
    return align_probabilities(
        raw, model.named_steps["classifier"].classes_, LABELS
    )


def select_predictor_c(
    mode: str,
    frame: pd.DataFrame,
    rows: list[dict[str, Any]],
    probability_maps: dict[str, dict[str, list[float]]],
) -> tuple[float, list[dict[str, Any]]]:
    labels = np.asarray([str(row["label"]) for row in rows])
    groups = np.asarray([str(row["passage_id"]) for row in rows])
    values = expert_matrix(rows, probability_maps)
    if mode == "router":
        targets = np.asarray(
            [
                int(
                    np.argmax(
                        [
                            probability_maps[task][str(row["item_id"])][
                                LABEL_INDEX[str(row["label"])]
                            ]
                            for task in TASKS
                        ]
                    )
                )
                for row in rows
            ]
        )
    else:
        targets = labels
    candidates = [0.03, 0.1, 0.3, 1.0, 3.0]
    summaries = []
    splitter = GroupKFold(n_splits=5)
    for c_value in candidates:
        predicted = [""] * len(rows)
        for train_index, heldout_index in splitter.split(frame, targets, groups):
            model = make_pipeline(frame.iloc[train_index], c_value=c_value)
            model.fit(frame.iloc[train_index], targets[train_index])
            if mode == "router":
                probabilities = soft_router_probabilities(
                    model, frame.iloc[heldout_index], values[heldout_index]
                )
            else:
                probabilities = stack_probabilities(model, frame.iloc[heldout_index])
            heldout_rows = [rows[index] for index in heldout_index]
            heldout_predictions = guarded_predictions(heldout_rows, probabilities)
            for index, label in zip(heldout_index, heldout_predictions):
                predicted[int(index)] = label
        metrics = evaluate(rows, predicted)
        score = 0.65 * metrics["three_decision_macro_f1"] + 0.35 * metrics[
            "four_class_macro_f1"
        ]
        summaries.append({"c": c_value, "selection_score": score, "metrics": metrics})
    best = max(
        summaries,
        key=lambda row: (
            row["selection_score"],
            row["metrics"]["three_decision_macro_f1"],
            -row["c"],
        ),
    )
    return float(best["c"]), summaries


def select_rejector_c(
    frame: pd.DataFrame, rows: list[dict[str, Any]]
) -> tuple[float, list[dict[str, Any]]]:
    targets = np.asarray(
        [
            int(str(row["label"]) in {"INSUFFICIENT_CONTEXT", "ENTITY_OR_TYPE_ERROR"})
            for row in rows
        ]
    )
    groups = np.asarray([str(row["passage_id"]) for row in rows])
    candidates = [0.03, 0.1, 0.3, 1.0, 3.0]
    summaries = []
    splitter = GroupKFold(n_splits=5)
    for c_value in candidates:
        scores = np.zeros(len(rows), dtype=float)
        for train_index, heldout_index in splitter.split(frame, targets, groups):
            model = make_pipeline(frame.iloc[train_index], c_value=c_value)
            model.fit(frame.iloc[train_index], targets[train_index])
            raw = model.predict_proba(frame.iloc[heldout_index])
            scores[heldout_index] = align_probabilities(
                raw, model.named_steps["classifier"].classes_, [0, 1]
            )[:, 1]
        summaries.append(
            {
                "c": c_value,
                "average_precision": float(average_precision_score(targets, scores)),
            }
        )
    best = max(summaries, key=lambda row: (row["average_precision"], -row["c"]))
    return float(best["c"]), summaries


def fixed_dual_probabilities(
    rows: list[dict[str, Any]], maps: dict[str, dict[str, list[float]]]
) -> np.ndarray:
    output = []
    for row in rows:
        item_id = str(row["item_id"])
        graph = maps["graph_anchor_wide"][item_id]
        if str(row["relation_group"]).startswith("recipe_"):
            path = maps["fera_ecrp"][item_id]
            text = maps["text_entity"][item_id]
            output.append(
                [
                    0.75 * graph[index]
                    + 0.10 * path[index]
                    + 0.15 * text[index]
                    for index in range(len(LABELS))
                ]
            )
        else:
            output.append(graph)
    return np.asarray(output, dtype=float)


def candidate_probability_sets(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    maps: dict[str, dict[str, list[float]]],
    stacker: Pipeline,
    manager: Pipeline,
) -> dict[str, np.ndarray]:
    experts = expert_matrix(rows, maps)
    candidates = {
        task: experts[:, index, :] for index, task in enumerate(TASKS)
    }
    candidates["uniform_expert_mean"] = experts.mean(axis=1)
    candidates["fixed_dual_evidence"] = fixed_dual_probabilities(rows, maps)
    candidates["oof_logistic_stacker"] = stack_probabilities(stacker, frame)
    candidates["oof_soft_expert_router"] = soft_router_probabilities(
        manager, frame, experts
    )
    candidates["ontology_group_router"] = np.stack(
        [
            candidates[FROZEN_ONTOLOGY_GROUP_ROUTING[str(row["relation_group"])]]
            [index]
            for index, row in enumerate(rows)
        ]
    )
    return candidates


def apply_rejector(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    reject_scores: np.ndarray,
    present_threshold: float,
    other_threshold: float,
) -> list[str]:
    output = []
    for row, values, reject_score in zip(rows, probabilities.tolist(), reject_scores):
        guarded, decision = apply_schema_guard(
            values,
            str(row["relation"]),
            str(row["head_type"]),
            str(row["tail_type"]),
        )
        label = LABELS[int(np.argmax(guarded))]
        if decision.routed_label is not None:
            output.append(decision.routed_label)
            continue
        threshold = present_threshold if label == "SUPPORTED" else other_threshold
        if label in {"SUPPORTED", "NOT_SUPPORTED"} and reject_score >= threshold:
            label = "INSUFFICIENT_CONTEXT"
        output.append(label)
    return output


def threshold_grid(scores: np.ndarray) -> list[float]:
    values = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
    for quantile in np.linspace(0.1, 0.9, 9):
        values.add(float(np.quantile(scores, quantile)))
    return sorted(values)


def tune_on_calibration(
    rows: list[dict[str, Any]],
    candidates: dict[str, np.ndarray],
    reject_scores: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    reference_predictions = guarded_predictions(
        rows, candidates["fera_ecrp_sourcebal"]
    )
    reference = evaluate(rows, reference_predictions)
    grid = threshold_grid(reject_scores)
    trials = []
    for name in [
        "fixed_dual_evidence",
        "oof_logistic_stacker",
        "oof_soft_expert_router",
    ]:
        for present_threshold in grid:
            for other_threshold in grid:
                predictions = apply_rejector(
                    rows,
                    candidates[name],
                    reject_scores,
                    present_threshold,
                    other_threshold,
                )
                metrics = evaluate(rows, predictions)
                abstain_rate = sum(
                    value in {"INSUFFICIENT_CONTEXT", "ENTITY_OR_TYPE_ERROR"}
                    for value in predictions
                ) / len(predictions)
                feasible = (
                    metrics["abstain_recall"]
                    >= reference["abstain_recall"] + 0.08
                    and metrics["gold_abstain_false_present_rate"]
                    <= reference["gold_abstain_false_present_rate"]
                    and abstain_rate <= 0.40
                )
                trials.append(
                    {
                        "predictor": name,
                        "present_threshold": present_threshold,
                        "other_threshold": other_threshold,
                        "predicted_abstain_rate": abstain_rate,
                        "feasible": feasible,
                        "metrics": metrics,
                    }
                )
    feasible = [row for row in trials if row["feasible"]]
    pool = feasible if feasible else trials
    selected = max(
        pool,
        key=lambda row: (
            row["metrics"]["three_decision_macro_f1"],
            row["metrics"]["four_class_macro_f1"],
            row["metrics"]["abstain_recall"],
            -row["metrics"]["gold_abstain_false_present_rate"],
            -row["predicted_abstain_rate"],
        ),
    )
    selected = dict(selected)
    selected["calibration_constraints_satisfied"] = bool(feasible)
    return selected, trials, reference


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def clustered_bootstrap(
    rows: list[dict[str, Any]],
    baseline: list[str],
    candidate: list[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    by_passage: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_passage[str(row["passage_id"])].append(index)
    passages = sorted(by_passage)
    generator = random.Random(seed)
    four, three = [], []
    gold_all = [str(row["label"]) for row in rows]
    for _ in range(replicates):
        sampled = generator.choices(passages, k=len(passages))
        indices = [index for passage in sampled for index in by_passage[passage]]
        gold = [gold_all[index] for index in indices]
        left = [baseline[index] for index in indices]
        right = [candidate[index] for index in indices]
        left_metrics = full_relation_verification_metrics(gold, left)
        right_metrics = full_relation_verification_metrics(gold, right)
        four.append(
            right_metrics["four_class"]["macro_f1"]
            - left_metrics["four_class"]["macro_f1"]
        )
        three.append(
            right_metrics["three_decision"]["macro_f1"]
            - left_metrics["three_decision"]["macro_f1"]
        )
    def summary(values: list[float]) -> dict[str, Any]:
        return {
            "mean": statistics.mean(values),
            "ci95_percentile": [percentile(values, 0.025), percentile(values, 0.975)],
            "probability_above_zero": sum(value > 0.0 for value in values) / len(values),
        }
    return {
        "resampling_unit": "passage_id",
        "clusters": len(passages),
        "replicates": replicates,
        "seed": seed,
        "four_class_macro_f1_difference": summary(four),
        "three_decision_macro_f1_difference": summary(three),
    }
