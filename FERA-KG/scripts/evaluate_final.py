from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))



DATA_ROOT = Path(os.environ.get("FERA_KG_DATA_ROOT", ROOT / "data"))
RESULT_ROOT = Path(os.environ.get("FERA_KG_PREDICTION_ROOT", DATA_ROOT / "predictions"))
NATURAL_GOLD = DATA_ROOT / "evaluation" / "natural_reference.jsonl"
NATURAL_MEMBERSHIP = DATA_ROOT / "evaluation" / "natural_reference.jsonl"
NATURAL_DIAGNOSTIC = DATA_ROOT / "comparators" / "natural_construction.json"
CONTROLLED_COMPARATOR = DATA_ROOT / "comparators" / "controlled_primary.json"
CONTROLLED_DIAGNOSTIC = DATA_ROOT / "comparators" / "controlled_construction.json"
STRESS_GOLD = DATA_ROOT / "evaluation" / "stress_reference.jsonl"
STRESS_COMPARATOR = DATA_ROOT / "comparators" / "additional_stress_primary.json"
WTR_EVALUATION = ROOT / "results/external/wtr_prove_v1/evaluation_v1.json"
WTR_DATASET_METADATA = ROOT / "data/external/wtr_prove_v1/metadata.json"
WTR_VERIFIER_METADATA = ROOT / "results/external/wtr_prove_v1/semantic_verifier_metadata_v1.json"
OUTPUT = Path(os.environ.get("FERA_KG_EVALUATION_OUTPUT", ROOT / "outputs" / "final_evaluation.json"))
REPORT = Path(os.environ.get("FERA_KG_EVALUATION_REPORT", ROOT / "outputs" / "final_evaluation.md"))
LABELS = ['SUPPORTED', 'NOT_SUPPORTED', 'INSUFFICIENT_CONTEXT', 'ENTITY_OR_TYPE_ERROR']
ABSTAIN_LABELS = {"INSUFFICIENT_CONTEXT", "ENTITY_OR_TYPE_ERROR"}
WTR_LABELS = ["SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_CONTEXT"]
WTR_T2_LABEL_COUNTS = {
    "SUPPORTED": 301,
    "NOT_SUPPORTED": 24,
    "INSUFFICIENT_CONTEXT": 84,
}
EXPECTED_DEEP_SEEDS = [20260921, 20260922, 20260923, 20260924, 20260925]
EXPECTED_FUSION_SEEDS = [20261101, 20261102, 20261103, 20261104, 20261105]
EXPECTED_EXPERT_FAMILIES = ['graph_anchor_wide', 'fera_ecrp', 'fera_ecrp_sourcebal', 'text_entity', 'same_backbone_no_ecrp']
NATURAL_COHORTS = {
    "primary_natural_600": {
        "membership": "natural_primary",
        "items": 600,
        "source_works": 34,
        "labels": {
            "SUPPORTED": 428,
            "NOT_SUPPORTED": 104,
            "INSUFFICIENT_CONTEXT": 14,
            "ENTITY_OR_TYPE_ERROR": 54,
        },
    },
    "difficulty_track_200": {
        "membership": "difficulty_enriched",
        "items": 200,
        "source_works": 27,
        "labels": {
            "SUPPORTED": 10,
            "NOT_SUPPORTED": 95,
            "INSUFFICIENT_CONTEXT": 70,
            "ENTITY_OR_TYPE_ERROR": 25,
        },
    },
}
STRESS_RESOURCES = {
    "ic_stress": {
        "title": "TCFT insufficient-context stress resource",
        "items": 150,
        "groups": 150,
        "cluster_size": 1,
    },
    "tiangongkaiwu": {
        "title": "Tiangong Kaiwu zero-shot stress resource",
        "items": 356,
        "groups": 89,
        "cluster_size": 4,
    },
}


# The same grouped bootstrap draws are reused for every comparator evaluated on
# a resource.  Caching the item multiplicities preserves the registered random
# resampling scheme while avoiding repeated Python/sklearn work for each model.
_BOOTSTRAP_WEIGHT_CACHE: dict[tuple[Any, ...], np.ndarray] = {}
_BOOTSTRAP_F1_CACHE: dict[tuple[Any, ...], np.ndarray] = {}


def _bootstrap_weights(
    groups: list[str],
    *,
    replicates: int,
    seed: int,
    resample_within_groups: bool,
) -> tuple[tuple[Any, ...], np.ndarray]:
    key = (tuple(groups), replicates, seed, resample_within_groups)
    cached = _BOOTSTRAP_WEIGHT_CACHE.get(key)
    if cached is not None:
        return key, cached

    group_to_indices: dict[str, np.ndarray] = {}
    for index, group in enumerate(groups):
        group_to_indices.setdefault(str(group), []).append(index)
    group_to_indices = {
        group: np.asarray(indices, dtype=np.int32)
        for group, indices in group_to_indices.items()
    }
    unique = sorted(group_to_indices)
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(groups)), dtype=np.uint16)
    for replicate in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices: list[int] = []
        for group in sampled:
            within_group = group_to_indices[str(group)]
            if resample_within_groups:
                indices.extend(
                    int(index)
                    for index in rng.choice(
                        within_group, size=len(within_group), replace=True
                    )
                )
            else:
                indices.extend(int(index) for index in within_group)
        weights[replicate] = np.bincount(indices, minlength=len(groups)).astype(
            np.uint16, copy=False
        )
    _BOOTSTRAP_WEIGHT_CACHE[key] = weights
    return key, weights


def _bootstrap_macro_f1(
    gold: list[str],
    predicted: list[str],
    *,
    weight_key: tuple[Any, ...],
    weights: np.ndarray,
) -> np.ndarray:
    cache_key = (weight_key, tuple(gold), tuple(predicted))
    cached = _BOOTSTRAP_F1_CACHE.get(cache_key)
    if cached is not None:
        return cached

    label_to_index = {label: index for index, label in enumerate(LABELS)}
    gold_index = np.asarray([label_to_index[label] for label in gold], dtype=np.int8)
    predicted_index = np.asarray(
        [label_to_index[label] for label in predicted], dtype=np.int8
    )
    confusion = np.zeros((len(weights), len(LABELS), len(LABELS)), dtype=np.float64)
    for true_index in range(len(LABELS)):
        for predicted_label_index in range(len(LABELS)):
            mask = (gold_index == true_index) & (
                predicted_index == predicted_label_index
            )
            if np.any(mask):
                confusion[:, true_index, predicted_label_index] = weights[
                    :, mask
                ].sum(axis=1)
    true_positive = np.diagonal(confusion, axis1=1, axis2=2)
    denominator = confusion.sum(axis=1) + confusion.sum(axis=2)
    class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    result = class_f1.mean(axis=1)
    _BOOTSTRAP_F1_CACHE[cache_key] = result
    return result


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_registered_input(recorded_path: Any, *, context: str) -> Path:
    if not isinstance(recorded_path, str) or not recorded_path.strip():
        raise ValueError(f"{context}: missing registered path")
    path = Path(recorded_path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"{context}: registered path leaves the project root: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{context}: registered artifact is missing: {path}")
    return path


def compact_metrics(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    if not gold or len(gold) != len(predicted):
        raise ValueError(
            f"Metric inputs must be non-empty and aligned: "
            f"gold={len(gold)}, predicted={len(predicted)}"
        )
    unknown_gold = sorted(set(gold) - set(LABELS))
    unknown_predicted = sorted(set(predicted) - set(LABELS))
    if unknown_gold or unknown_predicted:
        raise ValueError(
            f"Metric inputs contain labels outside the fixed label set: "
            f"gold={unknown_gold}, predicted={unknown_predicted}"
        )
    matrix = confusion_matrix(gold, predicted, labels=LABELS)
    per_class = {}
    for index, label in enumerate(LABELS):
        support = int(sum(value == label for value in gold))
        tp = int(matrix[index, index])
        predicted_count = int(matrix[:, index].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        score = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": score,
        }
    gold_abstain = [index for index, value in enumerate(gold) if value in ABSTAIN_LABELS]
    predicted_abstain = [value in ABSTAIN_LABELS for value in predicted]
    covered = [index for index, value in enumerate(predicted_abstain) if not value]
    predicted_present = [
        index for index, value in enumerate(predicted) if value == "SUPPORTED"
    ]
    three_gold = [
        "ABSTAIN" if value in ABSTAIN_LABELS else ("PRESENT" if value == "SUPPORTED" else "ABSENT")
        for value in gold
    ]
    three_predicted = [
        "ABSTAIN" if value in ABSTAIN_LABELS else ("PRESENT" if value == "SUPPORTED" else "ABSENT")
        for value in predicted
    ]
    reason_metrics = {}
    for reason in sorted(ABSTAIN_LABELS):
        indices = [index for index, value in enumerate(gold) if value == reason]
        reason_metrics[reason] = {
            "support": len(indices),
            "correct_reason_recall": (
                sum(predicted[index] == reason for index in indices) / len(indices)
                if indices
                else None
            ),
            "any_abstention_recall": (
                sum(predicted[index] in ABSTAIN_LABELS for index in indices) / len(indices)
                if indices
                else None
            ),
            "false_present_rate": (
                sum(predicted[index] == "SUPPORTED" for index in indices) / len(indices)
                if indices
                else None
            ),
        }
    return {
        "items": len(gold),
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, labels=LABELS, average="macro", zero_division=0)),
        "three_decision_macro_f1": float(
            f1_score(
                three_gold,
                three_predicted,
                labels=["PRESENT", "ABSENT", "ABSTAIN"],
                average="macro",
                zero_division=0,
            )
        ),
        "three_decision_accuracy": float(accuracy_score(three_gold, three_predicted)),
        "label_distribution": dict(Counter(gold)),
        "prediction_distribution": dict(Counter(predicted)),
        "per_class": per_class,
        "confusion_matrix": {
            gold_label: {
                predicted_label: int(matrix[row, column])
                for column, predicted_label in enumerate(LABELS)
            }
            for row, gold_label in enumerate(LABELS)
        },
        "abstention": {
            "gold_abstention_items": len(gold_abstain),
            "any_abstention_recall": (
                sum(predicted[index] in ABSTAIN_LABELS for index in gold_abstain)
                / len(gold_abstain)
                if gold_abstain
                else None
            ),
            "false_present_rate_on_gold_abstention": (
                sum(predicted[index] == "SUPPORTED" for index in gold_abstain)
                / len(gold_abstain)
                if gold_abstain
                else None
            ),
            "coverage": len(covered) / len(gold) if gold else None,
            "covered_accuracy": (
                sum(predicted[index] == gold[index] for index in covered) / len(covered)
                if covered
                else None
            ),
            "knowledge_admission_error": (
                sum(gold[index] != "SUPPORTED" for index in predicted_present)
                / len(predicted_present)
                if predicted_present
                else None
            ),
            "present_predictions": len(predicted_present),
            "supported_recall": per_class["SUPPORTED"]["recall"],
            "by_reason": reason_metrics,
        },
    }


def validated_probability_array(
    probabilities: list[list[float]] | np.ndarray,
    *,
    expected_items: int,
    context: str,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (expected_items, len(LABELS)):
        raise ValueError(
            f"{context}: probability shape {values.shape} does not match "
            f"{expected_items} items and {len(LABELS)} labels"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{context}: probability vectors contain non-finite values")
    if np.any(values < -1e-8) or np.any(values > 1.0 + 1e-8):
        raise ValueError(f"{context}: probability vectors contain values outside [0, 1]")
    row_sums = values.sum(axis=1)
    if np.any(row_sums <= 0.0) or not np.allclose(
        row_sums, 1.0, rtol=0.0, atol=2e-5
    ):
        raise ValueError(f"{context}: probability vectors do not sum to one within tolerance")
    values = np.clip(values, 1e-12, 1.0)
    return values / values.sum(axis=1, keepdims=True)


def probability_metrics(
    gold: list[str], probabilities: list[list[float]], *, bins: int = 15
) -> dict[str, Any]:
    """Return raw-probability calibration and confidence-coverage diagnostics."""
    if not gold:
        raise ValueError("Probability metrics require at least one labelled item")
    unknown_gold = sorted(set(gold) - set(LABELS))
    if unknown_gold:
        raise ValueError(f"Probability metrics contain unknown gold labels: {unknown_gold}")
    if bins <= 0:
        raise ValueError("Probability metrics require a positive number of ECE bins")
    values = validated_probability_array(
        probabilities,
        expected_items=len(gold),
        context="probability metrics",
    )
    label_index = {label: index for index, label in enumerate(LABELS)}
    gold_indices = np.asarray([label_index[label] for label in gold], dtype=int)
    one_hot = np.eye(len(LABELS), dtype=np.float64)[gold_indices]
    confidence = values.max(axis=1)
    predicted_indices = values.argmax(axis=1)
    correct = (predicted_indices == gold_indices).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    calibration_bins: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        if count:
            mean_confidence = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            ece += count / len(gold) * abs(accuracy - mean_confidence)
        else:
            mean_confidence = None
            accuracy = None
        calibration_bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    order = np.argsort(-confidence)
    retained_correct = np.cumsum(correct[order])
    coverage_grid = np.linspace(0.1, 1.0, 10)
    risk_coverage = []
    for target in coverage_grid:
        retained = max(1, int(np.ceil(target * len(gold))))
        coverage = retained / len(gold)
        accuracy = float(retained_correct[retained - 1] / retained)
        risk_coverage.append(
            {
                "coverage": coverage,
                "accuracy": accuracy,
                "risk": 1.0 - accuracy,
                "minimum_confidence": float(confidence[order][retained - 1]),
            }
        )
    risk_values = np.asarray([point["risk"] for point in risk_coverage], dtype=float)
    coverage_values = np.asarray(
        [point["coverage"] for point in risk_coverage], dtype=float
    )
    aurc_grid = float(np.trapezoid(risk_values, coverage_values))
    return {
        "label_order": LABELS,
        "top_label_definition": "argmax of the raw four-class probability vector",
        "ece_bins": bins,
        "raw_argmax_accuracy": float(correct.mean()),
        "negative_log_likelihood": float(
            -np.log(values[np.arange(len(gold)), gold_indices]).mean()
        ),
        "multiclass_brier_sum": float(np.square(values - one_hot).sum(axis=1).mean()),
        "top_label_ece_15_bins": float(ece),
        "mean_confidence": float(confidence.mean()),
        "risk_coverage_grid": risk_coverage,
        "risk_coverage_grid_area": aurc_grid,
        "calibration_bins": calibration_bins,
    }


def grouped_bootstrap_difference(
    gold: list[str],
    candidate: list[str],
    comparator: list[str],
    groups: list[str],
    *,
    replicates: int = 20000,
    seed: int = 20260908,
    unit: str = "source_work",
    resample_within_groups: bool = True,
) -> dict[str, Any]:
    if not gold or not (
        len(gold) == len(candidate) == len(comparator) == len(groups)
    ):
        raise ValueError("Grouped paired-bootstrap inputs must be non-empty and aligned")
    if any(not str(group).strip() for group in groups):
        raise ValueError("Grouped paired-bootstrap inputs contain a blank group identifier")
    unknown = sorted(
        (set(gold) | set(candidate) | set(comparator)) - set(LABELS)
    )
    if unknown:
        raise ValueError(f"Grouped paired-bootstrap inputs contain unknown labels: {unknown}")
    unique = sorted(set(groups))
    weight_key, weights = _bootstrap_weights(
        groups,
        replicates=replicates,
        seed=seed,
        resample_within_groups=resample_within_groups,
    )
    differences = _bootstrap_macro_f1(
        gold, candidate, weight_key=weight_key, weights=weights
    ) - _bootstrap_macro_f1(
        gold, comparator, weight_key=weight_key, weights=weights
    )
    point = f1_score(gold, candidate, labels=LABELS, average="macro", zero_division=0) - f1_score(
        gold, comparator, labels=LABELS, average="macro", zero_division=0
    )
    return {
        "resampling_unit": unit,
        "resampling_scheme": (
            "two-stage group-then-within-group bootstrap"
            if resample_within_groups
            else "whole-cluster bootstrap; all rows in each sampled cluster retained"
        ),
        "paired": True,
        "groups": len(unique),
        "replicates": replicates,
        "seed": seed,
        "point_difference": float(point),
        "bootstrap_mean": float(differences.mean()),
        "ci95_percentile": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "probability_above_zero": float(np.mean(differences > 0.0)),
    }


def grouped_bootstrap_mean_difference(
    correct: list[float],
    intervention: list[float],
    groups: list[str],
    *,
    replicates: int = 5000,
    seed: int = 20260909,
) -> dict[str, Any]:
    if not correct or not (len(correct) == len(intervention) == len(groups)):
        raise ValueError("Grouped mean-difference bootstrap inputs must be non-empty and aligned")
    if any(not str(group).strip() for group in groups):
        raise ValueError("Grouped mean-difference bootstrap contains a blank group identifier")
    unique = sorted(set(groups))
    _, weights = _bootstrap_weights(
        groups,
        replicates=replicates,
        seed=seed,
        resample_within_groups=True,
    )
    item_differences = np.asarray(correct, dtype=float) - np.asarray(
        intervention, dtype=float
    )
    denominators = weights.sum(axis=1, dtype=np.float64)
    differences = weights.astype(np.float64) @ item_differences
    differences = np.divide(
        differences,
        denominators,
        out=np.zeros_like(differences),
        where=denominators > 0,
    )
    return {
        "estimand": "mean correct-minus-intervention probability",
        "resampling_unit": "source work, then item within source work",
        "resampling_scheme": "two-stage group-then-within-group paired bootstrap",
        "paired": True,
        "groups": len(unique),
        "replicates": replicates,
        "seed": seed,
        "point_difference": float(item_differences.mean()),
        "ci95_percentile": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "probability_above_zero": float(np.mean(differences > 0.0)),
    }


def indexed_predictions(payload: dict[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    if list(payload.get("labels", [])) != LABELS:
        raise ValueError(
            f"Unified prediction label order {payload.get('labels')!r} does not match "
            f"the fixed order {LABELS!r}"
        )
    rows = payload["predictions"]
    if int(payload.get("items", len(rows))) != len(rows):
        raise ValueError("Unified prediction item count does not match its row count")
    item_ids = [str(row["item_id"]) for row in rows]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Unified prediction payload contains duplicate item identifiers")
    unknown = sorted(
        {str(row["predicted_label"]) for row in rows} - set(LABELS)
    )
    if unknown:
        raise ValueError(f"Unified prediction payload contains unknown labels: {unknown}")
    systems: dict[str, list[str]] = {
        "FERA-KG unified fusion": [str(row["predicted_label"]) for row in rows]
    }
    if rows and "component_predictions" in rows[0]:
        component_names = set(rows[0]["component_predictions"])
        if any(set(row.get("component_predictions", {})) != component_names for row in rows):
            raise ValueError("Component-prediction keys are inconsistent across items")
        for name in sorted(component_names):
            systems[f"component::{name}"] = [
                str(row["component_predictions"][name]) for row in rows
            ]
    if rows and rows[0].get("fusion_retraining_ablations"):
        variant_names = set(rows[0]["fusion_retraining_ablations"])
        if any(
            set(row.get("fusion_retraining_ablations", {})) != variant_names
            for row in rows
        ):
            raise ValueError("Fusion-ablation keys are inconsistent across items")
        for name in sorted(variant_names):
            systems[f"fusion_retrained::{name}"] = [
                str(row["fusion_retraining_ablations"][name]["predicted_label"])
                for row in rows
            ]
    for name, predictions in systems.items():
        unknown = sorted(set(predictions) - set(LABELS))
        if unknown:
            raise ValueError(f"{name} contains unknown predicted labels: {unknown}")
    return item_ids, systems


def indexed_probability_vectors(
    payload: dict[str, Any],
) -> dict[str, list[list[float]]]:
    rows = payload["predictions"]
    for row in rows:
        if set(row.get("probabilities", {})) != set(LABELS):
            raise ValueError(
                f"{row.get('item_id')}: fused probability keys do not match the fixed label set"
            )
    systems: dict[str, list[list[float]]] = {
        "FERA-KG unified fusion": [
            [float(row["probabilities"][label]) for label in LABELS]
            for row in rows
        ]
    }
    if rows and "component_probabilities" in rows[0]:
        component_names = set(rows[0]["component_probabilities"])
        if any(set(row.get("component_probabilities", {})) != component_names for row in rows):
            raise ValueError("Component-probability keys are inconsistent across items")
        for name in sorted(component_names):
            systems[f"component::{name}"] = [
                [float(value) for value in row["component_probabilities"][name]]
                for row in rows
            ]
    if rows and rows[0].get("fusion_retraining_ablations"):
        variant_names = set(rows[0]["fusion_retraining_ablations"])
        if any(
            set(row.get("fusion_retraining_ablations", {})) != variant_names
            for row in rows
        ):
            raise ValueError("Fusion-ablation probability keys are inconsistent across items")
        for name in sorted(variant_names):
            systems[f"fusion_retrained::{name}"] = [
                [
                    float(row["fusion_retraining_ablations"][name]["probabilities"][label])
                    for label in LABELS
                ]
                for row in rows
            ]
    return systems


def validate_five_seed_aggregation(
    unified_payload: dict[str, Any], *, context: str
) -> dict[str, Any]:
    """Verify that each stored expert vector is the mean of the five fixed seeds."""
    configuration = unified_payload.get("configuration", {})
    if [int(seed) for seed in configuration.get("deep_seeds", [])] != EXPECTED_DEEP_SEEDS:
        raise ValueError(f"{context}: unified payload does not register the fixed deep seeds")
    if [int(seed) for seed in configuration.get("fusion_seeds", [])] != EXPECTED_FUSION_SEEDS:
        raise ValueError(f"{context}: unified payload does not register the fixed fusion seeds")
    if int(configuration.get("deep_seed_count", -1)) != len(EXPECTED_DEEP_SEEDS):
        raise ValueError(f"{context}: unified payload deep-seed count is not five")
    if int(configuration.get("fusion_seed_count", -1)) != len(EXPECTED_FUSION_SEEDS):
        raise ValueError(f"{context}: unified payload fusion-seed count is not five")
    raw_path = unified_payload.get("expert_payload")
    if not raw_path:
        raise ValueError(f"{context}: no expert-payload path is registered")
    expert_path = Path(str(raw_path))
    if not expert_path.is_absolute():
        expert_path = ROOT / expert_path
    if not expert_path.is_file():
        expert_path = DATA_ROOT / "predictions" / "components" / Path(str(raw_path)).name
    if not expert_path.is_file():
        raise FileNotFoundError(f"{context}: registered expert payload is missing: {expert_path}")
    expected_hash = str(unified_payload.get("expert_payload_sha256", ""))
    observed_hash = sha256(expert_path)
    if not expected_hash or observed_hash != expected_hash:
        raise ValueError(f"{context}: registered expert-payload hash does not match")

    expert = read_json(expert_path)
    if list(expert.get("labels", [])) != LABELS:
        raise ValueError(f"{context}: expert-payload label order is not fixed to {LABELS}")
    seeds = [int(seed) for seed in expert.get("seeds", [])]
    if seeds != EXPECTED_DEEP_SEEDS:
        raise ValueError(
            f"{context}: expected the five fixed deep seeds {EXPECTED_DEEP_SEEDS}, got {seeds}"
        )
    families = list(expert.get("families", {}))
    if set(families) != set(EXPECTED_EXPERT_FAMILIES):
        raise ValueError(
            f"{context}: expected expert families {EXPECTED_EXPERT_FAMILIES}, got {families}"
        )
    expert_rows = expert.get("predictions", [])
    unified_rows = unified_payload.get("predictions", [])
    expert_ids = [str(row["item_id"]) for row in expert_rows]
    unified_ids = [str(row["item_id"]) for row in unified_rows]
    if expert_ids != unified_ids:
        raise ValueError(f"{context}: expert and unified item orders differ")

    seed_keys = {str(seed) for seed in EXPECTED_DEEP_SEEDS}
    for expert_row, unified_row in zip(expert_rows, unified_rows):
        item_id = str(expert_row["item_id"])
        for family in EXPECTED_EXPERT_FAMILIES:
            by_seed = expert_row.get("probabilities_by_seed", {}).get(family, {})
            if set(by_seed) != seed_keys:
                raise ValueError(
                    f"{context}/{item_id}/{family}: per-seed probability keys are incomplete"
                )
            seed_matrix = validated_probability_array(
                [by_seed[str(seed)] for seed in EXPECTED_DEEP_SEEDS],
                expected_items=len(EXPECTED_DEEP_SEEDS),
                context=f"{context}/{item_id}/{family}/per-seed",
            )
            stored_mean = validated_probability_array(
                [expert_row["probabilities"][family]],
                expected_items=1,
                context=f"{context}/{item_id}/{family}/stored-mean",
            )[0]
            recomputed_mean = seed_matrix.mean(axis=0)
            recomputed_mean /= recomputed_mean.sum()
            if not np.allclose(stored_mean, recomputed_mean, rtol=0.0, atol=1e-10):
                raise ValueError(
                    f"{context}/{item_id}/{family}: stored expert vector is not the "
                    "arithmetic mean of the five fixed seeds"
                )
            unified_components = unified_row.get("component_probabilities", {})
            if family in unified_components:
                unified_mean = validated_probability_array(
                    [unified_components[family]],
                    expected_items=1,
                    context=f"{context}/{item_id}/{family}/unified-mean",
                )[0]
                if not np.allclose(unified_mean, stored_mean, rtol=0.0, atol=1e-10):
                    raise ValueError(
                        f"{context}/{item_id}/{family}: unified component vector differs "
                        "from the verified five-seed mean"
                    )
    return {
        "verified": True,
        "seeds": seeds,
        "seed_count": len(seeds),
        "fusion_seeds": EXPECTED_FUSION_SEEDS,
        "fusion_seed_count": len(EXPECTED_FUSION_SEEDS),
        "expert_families": EXPECTED_EXPERT_FAMILIES,
        "aggregation": "arithmetic mean of probability vectors before fusion",
        "expert_payload_sha256": observed_hash,
    }


def natural_evaluation() -> dict[str, Any]:
    gold_rows = read_jsonl(NATURAL_GOLD)
    gold_by_id = {str(row["item_id"]): str(row["final_label"]) for row in gold_rows}
    source_by_id = {str(row["item_id"]): str(row["source_work"]) for row in gold_rows}
    if len(gold_by_id) != len(gold_rows):
        raise ValueError("Natural human reference contains duplicate item identifiers")
    if len(gold_rows) != 800:
        raise ValueError(f"Natural human reference has {len(gold_rows)} items; expected 800")
    if any(not value.strip() or value in {"None", "nan"} for value in source_by_id.values()):
        raise ValueError("Natural human reference contains a missing source-work identifier")

    membership_rows = read_jsonl(NATURAL_MEMBERSHIP)
    membership_by_id = {str(row["item_id"]): row for row in membership_rows}
    if len(membership_by_id) != len(membership_rows):
        raise ValueError("Natural cohort manifest contains duplicate item identifiers")
    if set(membership_by_id) != set(gold_by_id):
        raise ValueError(
            "Natural cohort manifest does not match the 800-item human reference"
        )
    for item_id, gold_row in zip(
        (str(row["item_id"]) for row in gold_rows), gold_rows
    ):
        membership_row = membership_by_id[item_id]
        if str(membership_row.get("candidate_hash")) != str(
            gold_row.get("candidate_hash")
        ):
            raise ValueError(
                f"Natural cohort manifest candidate hash mismatch for {item_id}"
            )
        if membership_row.get("cohort_internal") not in {
            definition["membership"] for definition in NATURAL_COHORTS.values()
        }:
            raise ValueError(
                f"Unrecognised explicit natural cohort for {item_id}: "
                f"{membership_row.get('cohort_internal')!r}"
            )
    correct_payload = read_json(RESULT_ROOT / "natural_registered.json")
    ensemble_validation = {
        "correct": validate_five_seed_aggregation(
            correct_payload, context="natural/correct"
        )
    }
    item_ids, systems = indexed_predictions(correct_payload)
    probability_systems = indexed_probability_vectors(correct_payload)
    if set(item_ids) != set(gold_by_id):
        raise ValueError("Natural unified predictions do not match the 800-item human reference")
    diagnostic = read_json(NATURAL_DIAGNOSTIC)
    diagnostic_map = {
        str(row["item_id"]): str(row["predicted_label"])
        for row in diagnostic["predictions"]
    }
    if set(diagnostic_map) != set(item_ids):
        raise ValueError(
            "Natural construction-only diagnostic does not match the 800-item reference"
        )
    systems["construction-only diagnostic"] = [diagnostic_map[item_id] for item_id in item_ids]
    systems["constant SUPPORTED prevalence reference"] = ["SUPPORTED"] * len(item_ids)

    cohorts = {
        cohort: [
            index
            for index, item_id in enumerate(item_ids)
            if membership_by_id[item_id]["cohort_internal"]
            == definition["membership"]
        ]
        for cohort, definition in NATURAL_COHORTS.items()
    }
    for cohort, indices in cohorts.items():
        definition = NATURAL_COHORTS[cohort]
        if len(indices) != definition["items"]:
            raise ValueError(
                f"Explicit {cohort} membership has {len(indices)} items; "
                f"expected {definition['items']}"
            )
        distribution = Counter(gold_by_id[item_ids[index]] for index in indices)
        if dict(distribution) != definition["labels"]:
            raise ValueError(
                f"Explicit {cohort} reference distribution is {dict(distribution)}; "
                f"expected {definition['labels']}"
            )
    if set(cohorts["primary_natural_600"]) & set(cohorts["difficulty_track_200"]):
        raise ValueError("Natural cohort memberships overlap")
    if set(cohorts["primary_natural_600"]) | set(
        cohorts["difficulty_track_200"]
    ) != set(range(len(item_ids))):
        raise ValueError("Natural cohort memberships do not exhaust all 800 items")
    output: dict[str, Any] = {}
    for cohort, indices in cohorts.items():
        definition = NATURAL_COHORTS[cohort]
        gold = [gold_by_id[item_ids[index]] for index in indices]
        groups = [source_by_id[item_ids[index]] for index in indices]
        if len(set(groups)) != int(definition["source_works"]):
            raise ValueError(
                f"Explicit {cohort} contains {len(set(groups))} source works; "
                f"expected {definition['source_works']}"
            )
        cohort_systems = {
            name: [values[index] for index in indices] for name, values in systems.items()
        }
        majority_label = max(LABELS, key=lambda label: gold.count(label))
        cohort_systems["cohort-majority baseline"] = [majority_label] * len(gold)
        metrics = {
            name: compact_metrics(gold, predictions)
            for name, predictions in cohort_systems.items()
        }
        calibration = {
            name: probability_metrics(
                gold, [vectors[index] for index in indices]
            )
            for name, vectors in probability_systems.items()
        }
        comparisons = {}
        full = cohort_systems["FERA-KG unified fusion"]
        for comparator in (
            "construction-only diagnostic",
            "cohort-majority baseline",
            "component::text_entity",
            "component::graph_anchor_wide",
            "component::same_backbone_no_ecrp",
            "fusion_retrained::without_ecrp_experts",
            "fusion_retrained::without_passage_local_evidence",
            "fusion_retrained::without_raw_construction_and_surface_features",
        ):
            if comparator in cohort_systems:
                comparisons[f"full_minus::{comparator}"] = grouped_bootstrap_difference(
                    gold, full, cohort_systems[comparator], groups
                )
        output[cohort] = {
            "membership_field": "cohort_internal",
            "membership_value": NATURAL_COHORTS[cohort]["membership"],
            "items": len(indices),
            "source_works": len(set(groups)),
            "metrics": metrics,
            "probability_metrics": calibration,
            "comparisons": comparisons,
        }

    intervention_payloads = {
        condition: read_json(RESULT_ROOT / {"correct": "natural_registered.json", "shuffled": "natural_shuffled.json", "empty": "natural_empty.json"}[condition])
        for condition in ("correct", "shuffled", "empty")
    }
    for condition in ("shuffled", "empty"):
        ensemble_validation[condition] = validate_five_seed_aggregation(
            intervention_payloads[condition], context=f"natural/{condition}"
        )
    intervention_systems = {}
    for condition, payload in intervention_payloads.items():
        condition_ids, condition_systems = indexed_predictions(payload)
        if condition_ids != item_ids:
            raise ValueError(
                f"Natural {condition} intervention order does not match the correct-passage order"
            )
        intervention_systems[condition] = condition_systems
    intervention_probabilities = {
        condition: indexed_probability_vectors(payload)
        for condition, payload in intervention_payloads.items()
    }
    interventions: dict[str, Any] = {}
    for cohort, indices in cohorts.items():
        gold = [gold_by_id[item_ids[index]] for index in indices]
        groups = [source_by_id[item_ids[index]] for index in indices]
        interventions[cohort] = {}
        for system in (
            "FERA-KG unified fusion",
            "component::fera_ecrp",
            "component::fera_ecrp_sourcebal",
            "component::text_entity",
            "component::same_backbone_no_ecrp",
        ):
            values = {
                condition: [condition_systems[system][index] for index in indices]
                for condition, condition_systems in intervention_systems.items()
                if system in condition_systems
            }
            if set(values) != {"correct", "shuffled", "empty"}:
                continue
            probabilities = {
                condition: [
                    intervention_probabilities[condition][system][index]
                    for index in indices
                ]
                for condition in ("correct", "shuffled", "empty")
            }
            support_index = LABELS.index("SUPPORTED")
            support_probabilities = {
                condition: [row[support_index] for row in rows]
                for condition, rows in probabilities.items()
            }
            interventions[cohort][system] = {
                "interpretation": (
                    "Information-dependence diagnostic evaluated against the original labels; "
                    "the shuffled and empty passages were not independently adjudicated and "
                    "their label-based scores are not counterfactual accuracy estimates."
                ),
                "metrics": {
                    condition: compact_metrics(gold, predictions)
                    for condition, predictions in values.items()
                },
                "correct_minus_shuffled": grouped_bootstrap_difference(
                    gold, values["correct"], values["shuffled"], groups
                ),
                "correct_minus_empty": grouped_bootstrap_difference(
                    gold, values["correct"], values["empty"], groups
                ),
                "changed_decisions": {
                    "shuffled": sum(a != b for a, b in zip(values["correct"], values["shuffled"])),
                    "empty": sum(a != b for a, b in zip(values["correct"], values["empty"])),
                },
                "mean_absolute_probability_change": {
                    "shuffled": float(
                        np.mean(
                            np.abs(
                                np.asarray(probabilities["correct"])
                                - np.asarray(probabilities["shuffled"])
                            )
                        )
                    ),
                    "empty": float(
                        np.mean(
                            np.abs(
                                np.asarray(probabilities["correct"])
                                - np.asarray(probabilities["empty"])
                            )
                        )
                    ),
                },
                "support_probability_change": {
                    "correct_minus_shuffled": grouped_bootstrap_mean_difference(
                        support_probabilities["correct"],
                        support_probabilities["shuffled"],
                        groups,
                    ),
                    "correct_minus_empty": grouped_bootstrap_mean_difference(
                        support_probabilities["correct"],
                        support_probabilities["empty"],
                        groups,
                        seed=20260910,
                    ),
                },
            }
    return {
        "gold_sha256": sha256(NATURAL_GOLD),
        "cohort_membership_sha256": sha256(NATURAL_MEMBERSHIP),
        "cohort_membership_rule": (
            "Explicit cohort_internal values from the frozen natural-candidate "
            "manifest; item-number ranges are not cohort definitions."
        ),
        "diagnostic_sha256": sha256(NATURAL_DIAGNOSTIC),
        "five_seed_aggregation_validation": ensemble_validation,
        "cohorts": output,
        "passage_interventions": interventions,
    }


def load_controlled_reference() -> tuple[list[dict[str, Any]], list[str], dict[str, str], dict[str, Any]]:
    rows = read_jsonl(DATA_ROOT / "evaluation" / "controlled_reference.jsonl")
    ordered_ids = [str(row["item_id"]) for row in rows]
    gold = {str(row["item_id"]): str(row["final_label"]) for row in rows}
    statuses = Counter(str(row.get("reference_status", "")) for row in rows)
    return rows, ordered_ids, gold, {
        "disagreements_adjudicated": int(
            statuses.get("third_annotator_adjudication", 0)
        ),
        "individual_annotator_identifiers_included": False,
    }


def controlled_evaluation() -> dict[str, Any]:
    frozen_rows, ordered_ids, gold_map, human = load_controlled_reference()
    payload = read_json(RESULT_ROOT / "controlled.json")
    ensemble_validation = validate_five_seed_aggregation(
        payload, context="controlled"
    )
    item_ids, systems = indexed_predictions(payload)
    probability_systems = indexed_probability_vectors(payload)
    if item_ids != ordered_ids:
        raise ValueError("Controlled unified predictions do not match the fixed human-reference order")
    comparator_payload = read_json(CONTROLLED_COMPARATOR)
    comparator = {
        str(row["item_id"]): str(row["predicted_label"])
        for row in comparator_payload["predictions"]
    }
    if set(comparator) != set(item_ids):
        raise ValueError(
            "Controlled fixed comparator does not match the fixed human-reference items"
        )
    systems["fixed pre-existing comparator"] = [comparator[item_id] for item_id in item_ids]
    diagnostic_payload = read_json(CONTROLLED_DIAGNOSTIC)
    diagnostic = {
        str(row["item_id"]): str(row["predicted_label"])
        for row in diagnostic_payload["predictions"]
    }
    if set(diagnostic) != set(item_ids):
        raise ValueError(
            "Controlled construction-only diagnostic does not match the fixed "
            "human-reference items"
        )
    systems["construction-only diagnostic"] = [
        diagnostic[item_id] for item_id in item_ids
    ]
    systems["constant SUPPORTED prevalence reference"] = ["SUPPORTED"] * len(item_ids)
    gold = [gold_map[item_id] for item_id in item_ids]
    row_by_id = {
        str(row["item_id"]): row for row in frozen_rows
    }
    groups = [
        str(
            row_by_id[item_id].get("source_work")
            or row_by_id[item_id].get("book")
            or str(row_by_id[item_id].get("source_key", item_id)).split("::", 1)[0]
        )
        for item_id in item_ids
    ]
    if len(set(groups)) != 33:
        raise ValueError(
            f"Controlled challenge contains {len(set(groups))} source works; expected 33"
        )
    metrics = {name: compact_metrics(gold, values) for name, values in systems.items()}
    comparisons = {}
    full = systems["FERA-KG unified fusion"]
    for name, values in systems.items():
        if name == "FERA-KG unified fusion":
            continue
        comparisons[f"full_minus::{name}"] = grouped_bootstrap_difference(
            gold, full, values, groups, unit="source_work_then_passage"
        )
    return {
        "items": len(item_ids),
        "source_works": len(set(groups)),
        "human_annotation": human,
        "five_seed_aggregation_validation": ensemble_validation,
        "metrics": metrics,
        "probability_metrics": {
            name: probability_metrics(gold, values)
            for name, values in probability_systems.items()
        },
        "comparisons": comparisons,
    }


def stress_evaluation() -> dict[str, Any]:
    reference_rows = read_jsonl(STRESS_GOLD)
    gold_map = {
        str(row["confirmatory_id"]): str(row["final_decision"])
        for row in reference_rows
    }
    if len(gold_map) != len(reference_rows):
        raise ValueError("Stress human reference contains duplicate item identifiers")
    dataset_map = {
        str(row["confirmatory_id"]): str(row["dataset"])
        for row in reference_rows
    }
    observed_dataset_counts = Counter(dataset_map.values())
    expected_dataset_counts = {
        name: int(definition["items"])
        for name, definition in STRESS_RESOURCES.items()
    }
    if dict(observed_dataset_counts) != expected_dataset_counts:
        raise ValueError(
            f"Stress-resource membership is {dict(observed_dataset_counts)}; "
            f"expected {expected_dataset_counts}"
        )
    group_map = {
        str(row["confirmatory_id"]): str(
            row.get("source_cluster_id") or row.get("source_key") or row["confirmatory_id"]
        )
        for row in reference_rows
    }
    payload = read_json(RESULT_ROOT / "additional_stress.json")
    ensemble_validation = validate_five_seed_aggregation(
        payload, context="stress"
    )
    item_ids, systems = indexed_predictions(payload)
    probability_systems = indexed_probability_vectors(payload)
    if set(item_ids) != set(gold_map):
        raise ValueError("Unified stress predictions do not match the human reference")
    comparator_payload = read_json(STRESS_COMPARATOR)
    comparator_map = {
        str(row["item_id"]): str(row["predicted_label"])
        for row in comparator_payload["predictions"]
    }
    if set(comparator_map) != set(item_ids):
        raise ValueError("Stress fixed comparator does not match the human-reference items")
    systems["fixed pre-existing comparator"] = [comparator_map[item_id] for item_id in item_ids]
    output = {}
    for dataset, definition in STRESS_RESOURCES.items():
        title = str(definition["title"])
        indices = [
            index for index, item_id in enumerate(item_ids)
            if dataset_map[item_id] == dataset
        ]
        gold = [gold_map[item_ids[index]] for index in indices]
        groups = [group_map[item_ids[index]] for index in indices]
        group_sizes = Counter(groups)
        if len(indices) != int(definition["items"]):
            raise ValueError(
                f"{dataset} has {len(indices)} items; expected {definition['items']}"
            )
        if len(group_sizes) != int(definition["groups"]):
            raise ValueError(
                f"{dataset} has {len(group_sizes)} source clusters; "
                f"expected {definition['groups']}"
            )
        if set(group_sizes.values()) != {int(definition["cluster_size"])}:
            raise ValueError(
                f"{dataset} source-cluster sizes are {sorted(set(group_sizes.values()))}; "
                f"expected {definition['cluster_size']}"
            )
        selected_systems = {
            name: [values[index] for index in indices]
            for name, values in systems.items()
        }
        full = selected_systems["FERA-KG unified fusion"]
        output[dataset] = {
            "title": title,
            "items": len(indices),
            "groups": len(set(groups)),
            "metrics": {
                name: compact_metrics(gold, values)
                for name, values in selected_systems.items()
            },
            "probability_metrics": {
                name: probability_metrics(
                    gold, [values[index] for index in indices]
                )
                for name, values in probability_systems.items()
            },
            "comparisons": {
                f"full_minus::{name}": grouped_bootstrap_difference(
                    gold,
                    full,
                    values,
                    groups,
                    unit="registered source-passage cluster",
                    resample_within_groups=False,
                )
                for name, values in selected_systems.items()
                if name != "FERA-KG unified fusion"
            },
        }
    return {
        "gold_sha256": sha256(STRESS_GOLD),
        "five_seed_aggregation_validation": ensemble_validation,
        "results": output,
        "interpretation": (
            "The Tiangong Kaiwu evaluation is a zero-shot domain-shift stress test, "
            "not an in-domain benchmark or evidence of cross-domain generality."
        ),
    }


def external_protocol_audit() -> dict[str, Any]:
    """Verify and register the existing ProVe WTR protocol audit read-only.

    The WTR experiment used a separate fixed-prompt semantic verifier and a
    source-grouped artifact-only diagnostic.  It did not run the Chinese
    FERA-KG encoder and therefore must not be presented as external validation
    of FERA-KG.
    """

    evaluation = read_json(WTR_EVALUATION)
    if evaluation.get("status") != "WTR_EXTERNAL_EVALUATION_COMPLETE":
        raise ValueError("WTR evaluation has an unexpected status")
    if evaluation.get("primary_reference") != "gold_t2":
        raise ValueError("WTR evaluation must use gold_t2 as its primary reference")
    if list(evaluation.get("labels", [])) != WTR_LABELS:
        raise ValueError("WTR evaluation label order differs from the fixed three-label order")

    dataset_metadata = read_json(WTR_DATASET_METADATA)
    if dataset_metadata.get("status") != "PUBLIC_EXTERNAL_BENCHMARK_PREPARED":
        raise ValueError("WTR prepared-data metadata has an unexpected status")
    if int(dataset_metadata.get("items", -1)) != 409:
        raise ValueError("WTR prepared-data metadata does not record 409 items")
    if int(dataset_metadata.get("source_domains", -1)) != 32:
        raise ValueError("WTR prepared-data metadata does not record 32 source domains")
    if int(dataset_metadata.get("properties", -1)) != 76:
        raise ValueError("WTR prepared-data metadata does not record 76 properties")
    if dataset_metadata.get("t2_label_counts") != WTR_T2_LABEL_COUNTS:
        raise ValueError("WTR prepared-data metadata has an unexpected gold_t2 distribution")

    verifier_metadata = read_json(WTR_VERIFIER_METADATA)
    if verifier_metadata.get("status") != (
        "WTR_PROGRAMMATIC_VERIFICATION_COMPLETE_LABELS_NOT_SENT_TO_MODEL"
    ):
        raise ValueError("WTR semantic-verifier metadata has an unexpected status")
    if int(verifier_metadata.get("items", -1)) != 409:
        raise ValueError("WTR semantic-verifier metadata does not record 409 items")
    if verifier_metadata.get("credential_material_saved") is not False:
        raise ValueError("WTR semantic-verifier metadata does not exclude saved credentials")

    registered = evaluation.get("inputs")
    if not isinstance(registered, dict):
        raise ValueError("WTR evaluation does not contain a registered input map")
    input_paths: dict[str, Path] = {}
    input_hashes: dict[str, str] = {}
    for role in ("artifact", "data", "semantic"):
        path = resolve_registered_input(
            registered.get(role), context=f"WTR {role} input"
        )
        observed_hash = sha256(path)
        expected_hash = str(registered.get(f"{role}_sha256", ""))
        if observed_hash != expected_hash:
            raise ValueError(
                f"WTR {role} artifact hash mismatch: expected {expected_hash}, "
                f"observed {observed_hash}"
            )
        input_paths[role] = path
        input_hashes[role] = observed_hash

    prepared_rows = read_jsonl(input_paths["data"])
    item_ids = [str(row["item_id"]) for row in prepared_rows]
    if len(prepared_rows) != 409 or len(set(item_ids)) != 409:
        raise ValueError("WTR prepared data must contain 409 uniquely identified items")
    if len({str(row["source_domain"]) for row in prepared_rows}) != 32:
        raise ValueError("WTR prepared data do not contain exactly 32 source domains")
    if len({str(row["property_id"]) for row in prepared_rows}) != 76:
        raise ValueError("WTR prepared data do not contain exactly 76 properties")
    observed_t2_counts = Counter(str(row["gold_t2"]) for row in prepared_rows)
    if dict(observed_t2_counts) != WTR_T2_LABEL_COUNTS:
        raise ValueError(
            f"WTR gold_t2 distribution is {dict(observed_t2_counts)}; "
            f"expected {WTR_T2_LABEL_COUNTS}"
        )
    if dataset_metadata.get("output_sha256") != input_hashes["data"]:
        raise ValueError("WTR prepared-data metadata does not identify the evaluated data")
    if verifier_metadata.get("input_sha256") != input_hashes["data"]:
        raise ValueError("WTR verifier metadata input hash differs from the evaluated data")
    if verifier_metadata.get("output_sha256") != input_hashes["semantic"]:
        raise ValueError("WTR verifier metadata output hash differs from the evaluated output")

    artifact_payload = read_json(input_paths["artifact"])
    if artifact_payload.get("status") != "WTR_SOURCE_GROUPED_ARTIFACT_ONLY_OOF_COMPLETE":
        raise ValueError("WTR artifact-only diagnostic has an unexpected status")
    artifact_folds = artifact_payload.get("folds")
    if (
        not isinstance(artifact_folds, list)
        or len(artifact_folds) != 5
        or {int(fold.get("fold", -1)) for fold in artifact_folds}
        != {1, 2, 3, 4, 5}
        or "source_domain" not in str(artifact_payload.get("split", ""))
    ):
        raise ValueError("WTR artifact-only diagnostic is not registered as source-domain grouped")
    if artifact_payload.get("input_sha256") != input_hashes["data"]:
        raise ValueError("WTR artifact-only diagnostic input hash differs from the evaluated data")

    semantic_rows = read_jsonl(input_paths["semantic"])
    artifact_predictions = {
        str(row["item_id"]): str(row["predicted_label"])
        for row in artifact_payload.get("predictions", [])
    }
    semantic_predictions = {
        str(row["item_id"]): str(row["predicted_label"])
        for row in semantic_rows
    }
    expected_ids = set(item_ids)
    if len(artifact_predictions) != 409 or set(artifact_predictions) != expected_ids:
        raise ValueError("WTR artifact-only predictions do not match the 409 prepared items")
    if len(semantic_predictions) != 409 or set(semantic_predictions) != expected_ids:
        raise ValueError("WTR semantic-verifier predictions do not match the 409 prepared items")

    gold = [str(row["gold_t2"]) for row in prepared_rows]
    stored_primary = evaluation.get("evaluations", {}).get("gold_t2", {})
    audited_outputs: dict[str, dict[str, Any]] = {}
    prediction_maps = {
        "artifact_only": artifact_predictions,
        "semantic_verifier": semantic_predictions,
    }
    for system, predictions_by_id in prediction_maps.items():
        predicted = [predictions_by_id[item_id] for item_id in item_ids]
        unknown = sorted(set(predicted) - set(WTR_LABELS))
        if unknown:
            raise ValueError(f"WTR {system} contains unknown labels: {unknown}")
        observed_matrix = confusion_matrix(
            gold, predicted, labels=WTR_LABELS
        ).astype(int).tolist()
        observed_accuracy = float(accuracy_score(gold, predicted))
        observed_macro_f1 = float(
            f1_score(
                gold,
                predicted,
                labels=WTR_LABELS,
                average="macro",
                zero_division=0,
            )
        )
        stored = stored_primary.get(system)
        if not isinstance(stored, dict):
            raise ValueError(f"WTR evaluation lacks stored gold_t2 metrics for {system}")
        if int(stored.get("items", -1)) != 409:
            raise ValueError(f"WTR {system} stored metric count is not 409")
        if list(stored.get("label_order", [])) != WTR_LABELS:
            raise ValueError(f"WTR {system} stored label order changed")
        if stored.get("confusion_matrix") != observed_matrix:
            raise ValueError(f"WTR {system} stored confusion matrix cannot be reproduced")
        if not np.isclose(
            float(stored.get("accuracy")), observed_accuracy, rtol=0.0, atol=1e-12
        ) or not np.isclose(
            float(stored.get("macro_f1")), observed_macro_f1, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"WTR {system} stored summary metrics cannot be reproduced")
        audited_outputs[system] = {
            "items": 409,
            "accuracy": observed_accuracy,
            "macro_f1": observed_macro_f1,
            "confusion_matrix": observed_matrix,
        }

    bootstrap = stored_primary.get("source_bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError("WTR evaluation lacks its source-domain bootstrap record")
    if int(bootstrap.get("replicates", -1)) != 5000 or int(
        bootstrap.get("source_domains", -1)
    ) != 32:
        raise ValueError("WTR bootstrap must record 5,000 replicates over 32 domains")
    if bootstrap.get("method") != "two-stage source-domain bootstrap":
        raise ValueError("WTR bootstrap method changed")
    for key, value in bootstrap.items():
        if key.endswith("_ci95"):
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"WTR bootstrap interval {key} is malformed")
            bounds = np.asarray(value, dtype=np.float64)
            if not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
                raise ValueError(f"WTR bootstrap interval {key} is invalid")

    artifacts = {
        "evaluation": {
            "path": Path(os.path.relpath(WTR_EVALUATION, ROOT)).as_posix().replace("\\", "/"),
            "sha256": sha256(WTR_EVALUATION),
        },
        "dataset_metadata": {
            "path": Path(os.path.relpath(WTR_DATASET_METADATA, ROOT)).as_posix().replace("\\", "/"),
            "sha256": sha256(WTR_DATASET_METADATA),
        },
        "verifier_metadata": {
            "path": Path(os.path.relpath(WTR_VERIFIER_METADATA, ROOT)).as_posix().replace("\\", "/"),
            "sha256": sha256(WTR_VERIFIER_METADATA),
        },
    }
    for role, path in input_paths.items():
        artifacts[role] = {
            "path": Path(os.path.relpath(path, ROOT)).as_posix().replace("\\", "/"),
            "sha256": input_hashes[role],
        }

    return {
        "status": "VERIFIED_READ_ONLY_EXTERNAL_PROTOCOL_AUDIT",
        "resource": "ProVe Web-Text Relation (WTR)",
        "scope": (
            "Approximate transfer audit of a three-label evidence-verification protocol "
            "on public English web evidence using a separate fixed-prompt semantic "
            "verifier and a source-domain-grouped artifact-only diagnostic."
        ),
        "not_fera_kg_external_validation": True,
        "items": 409,
        "source_domains": 32,
        "properties": 76,
        "primary_reference": "gold_t2",
        "label_order": WTR_LABELS,
        "primary_label_distribution": WTR_T2_LABEL_COUNTS,
        "audited_outputs": audited_outputs,
        "source_domain_bootstrap": bootstrap,
        "verifier": {
            "model_identifier": verifier_metadata.get("model"),
            "temperature": verifier_metadata.get("temperature"),
            "labels_sent_to_model": False,
            "credential_material_saved": False,
            "prompt_sha256": verifier_metadata.get("prompt_sha256"),
        },
        "dataset_record": dataset_metadata.get("dataset_record"),
        "license": dataset_metadata.get("license"),
        "artifacts": artifacts,
        "interpretation": (
            "This block registers a separate protocol audit. It is not an external "
            "evaluation or validation of the Chinese FERA-KG graph encoder."
        ),
    }


def main() -> None:
    payload = {
        "status": "UNIFIED_FINAL_EVALUATION_COMPLETE",
        "controlled": controlled_evaluation(),
        "natural": natural_evaluation(),
        "stress_tests": stress_evaluation(),
        "external_protocol_audit": {
            "status": "WITHHELD_FROM_ANONYMOUS_REVIEW_ARCHIVE",
            "scope": "Separate protocol audit; not a FERA-KG external validation.",
            "reason": "Third-party source material and hosted-verifier traces are not included."
        },
        "inferential_scope": {
            "primary_natural_cohort": 600,
            "difficulty_track": 200,
            "controlled_challenge_set": 450,
            "bootstrap_unit": (
                "two-stage source-work then within-work passage resampling for natural and "
                "controlled resources; whole source-passage clusters for stress resources"
            ),
            "zero_division": 0,
            "fixed_label_set": LABELS,
            "deep_ensemble_seeds": EXPECTED_DEEP_SEEDS,
            "fusion_ensemble_seeds": EXPECTED_FUSION_SEEDS,
            "warning": (
                "These are within-task results on the registered evaluation resources, "
                "not a claim of field-wide state of the art."
            ),
        },
        "inputs": {
            path.name: sha256(path)
            for path in (
                RESULT_ROOT / "controlled.json",
                RESULT_ROOT / "natural_registered.json",
                RESULT_ROOT / "natural_shuffled.json",
                RESULT_ROOT / "natural_empty.json",
                RESULT_ROOT / "additional_stress.json",
            )
        },
        "registered_comparator_inputs": {
            CONTROLLED_COMPARATOR.name: sha256(CONTROLLED_COMPARATOR),
            CONTROLLED_DIAGNOSTIC.name: sha256(CONTROLLED_DIAGNOSTIC),
            NATURAL_DIAGNOSTIC.name: sha256(NATURAL_DIAGNOSTIC),
            STRESS_COMPARATOR.name: sha256(STRESS_COMPARATOR),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    primary = payload["natural"]["cohorts"]["primary_natural_600"]
    difficulty = payload["natural"]["cohorts"]["difficulty_track_200"]
    controlled = payload["controlled"]
    lines = [
        "# Unified final evaluation",
        "",
        "All results below use the same five-seed full-training expert configuration and the source-work-grouped OOF fusion artifact.",
        "",
        "| Evaluation resource | Items | Macro-F1 | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name, block in (
        ("Primary natural cohort", primary),
        ("Difficulty track", difficulty),
        ("Controlled challenge", controlled),
    ):
        metric = block["metrics"]["FERA-KG unified fusion"]
        lines.append(f"| {name} | {metric['items']} | {metric['macro_f1']:.4f} | {metric['accuracy']:.4f} |")
    lines.extend([
        "",
        "The JSON result contains complete confusion matrices, reason-specific abstention results, component baselines, fusion retraining ablations, passage interventions, and grouped bootstrap intervals.",
    ])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
