from __future__ import annotations

from collections import defaultdict

import numpy as np

LABELS = ['SUPPORTED', 'NOT_SUPPORTED', 'INSUFFICIENT_CONTEXT', 'ENTITY_OR_TYPE_ERROR']


def percentile(values: list[float]) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def fast_metrics(gold_codes: np.ndarray, predicted_codes: np.ndarray) -> tuple[float, float, float]:
    matrix = np.bincount(gold_codes * 4 + predicted_codes, minlength=16).reshape(4, 4)
    true_positive = np.diag(matrix).astype(float)
    predicted_support = matrix.sum(axis=0).astype(float)
    gold_support = matrix.sum(axis=1).astype(float)
    precision = np.divide(
        true_positive, predicted_support, out=np.zeros_like(true_positive), where=predicted_support > 0
    )
    recall = np.divide(true_positive, gold_support, out=np.zeros_like(true_positive), where=gold_support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    macro_f1 = float(f1.mean())
    weighted_f1 = float(np.average(f1, weights=gold_support))
    accuracy = float(true_positive.sum() / matrix.sum())
    return macro_f1, weighted_f1, accuracy


def stratified_bootstrap(
    gold: list[str], candidate: list[str], comparator: list[str], replicates: int = 20_000, seed: int = 20260902
) -> dict:
    rng = np.random.default_rng(seed)
    groups = {label: np.array([index for index, value in enumerate(gold) if value == label]) for label in LABELS}
    label_to_code = {label: index for index, label in enumerate(LABELS)}
    gold_codes = np.array([label_to_code[value] for value in gold], dtype=np.int8)
    candidate_codes = np.array([label_to_code[value] for value in candidate], dtype=np.int8)
    comparator_codes = np.array([label_to_code[value] for value in comparator], dtype=np.int8)
    candidate_macro = []
    difference_macro = []
    candidate_weighted = []
    difference_weighted = []
    candidate_accuracy = []
    difference_accuracy = []
    for _ in range(replicates):
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in groups.values()])
        candidate_macro_value, candidate_weighted_value, candidate_accuracy_value = fast_metrics(
            gold_codes[sampled], candidate_codes[sampled]
        )
        comparator_macro_value, comparator_weighted_value, comparator_accuracy_value = fast_metrics(
            gold_codes[sampled], comparator_codes[sampled]
        )
        candidate_macro.append(float(candidate_macro_value))
        difference_macro.append(float(candidate_macro_value - comparator_macro_value))
        candidate_weighted.append(float(candidate_weighted_value))
        difference_weighted.append(float(candidate_weighted_value - comparator_weighted_value))
        candidate_accuracy.append(float(candidate_accuracy_value))
        difference_accuracy.append(float(candidate_accuracy_value - comparator_accuracy_value))
    return {
        "method": "gold-label-stratified paired bootstrap; original class counts retained in every replicate",
        "replicates": replicates,
        "seed": seed,
        "candidate_four_class_macro_f1_ci95": percentile(candidate_macro),
        "candidate_minus_primary_macro_f1_ci95": percentile(difference_macro),
        "candidate_weighted_f1_ci95": percentile(candidate_weighted),
        "candidate_minus_primary_weighted_f1_ci95": percentile(difference_weighted),
        "candidate_accuracy_ci95": percentile(candidate_accuracy),
        "candidate_minus_primary_accuracy_ci95": percentile(difference_accuracy),
    }


def source_two_stage_bootstrap(
    rows: list[dict], gold: list[str], candidate: list[str], comparator: list[str], replicates: int = 20_000, seed: int = 20260903
) -> dict:
    rng = np.random.default_rng(seed)
    label_to_code = {label: index for index, label in enumerate(LABELS)}
    gold_codes = np.array([label_to_code[value] for value in gold], dtype=np.int8)
    candidate_codes = np.array([label_to_code[value] for value in candidate], dtype=np.int8)
    comparator_codes = np.array([label_to_code[value] for value in comparator], dtype=np.int8)
    source_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        source_to_indices[row["book"]].append(index)
    sources = sorted(source_to_indices)
    differences = []
    for _ in range(replicates):
        sampled_sources = rng.choice(sources, size=len(sources), replace=True)
        sampled_indices = []
        for source in sampled_sources:
            indices = source_to_indices[source]
            sampled_indices.extend(rng.choice(indices, size=len(indices), replace=True).tolist())
        sampled = np.asarray(sampled_indices, dtype=int)
        candidate_value, _, _ = fast_metrics(gold_codes[sampled], candidate_codes[sampled])
        comparator_value, _, _ = fast_metrics(gold_codes[sampled], comparator_codes[sampled])
        differences.append(float(candidate_value - comparator_value))
    return {
        "method": "two-stage source-document bootstrap; books/files resampled first and passages resampled within selected books/files",
        "source_clusters": len(sources),
        "replicates": replicates,
        "seed": seed,
        "candidate_minus_primary_macro_f1_ci95": percentile(differences),
    }
