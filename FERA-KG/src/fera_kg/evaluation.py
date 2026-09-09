from __future__ import annotations

from collections import Counter
from typing import Any


FOUR_CLASS_LABELS = [
    "SUPPORTED",
    "NOT_SUPPORTED",
    "INSUFFICIENT_CONTEXT",
    "ENTITY_OR_TYPE_ERROR",
]
THREE_DECISION_LABELS = ["PRESENT", "ABSENT", "ABSTAIN"]


def to_three_decision(label: str) -> str:
    if label == "SUPPORTED":
        return "PRESENT"
    if label == "NOT_SUPPORTED":
        return "ABSENT"
    if label in {"INSUFFICIENT_CONTEXT", "ENTITY_OR_TYPE_ERROR"}:
        return "ABSTAIN"
    raise ValueError(f"unknown four-class label: {label}")


def classification_metrics(
    gold: list[str],
    predicted: list[str],
    labels: list[str],
) -> dict[str, Any]:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("gold and predicted must have the same positive length")
    unknown = (set(gold) | set(predicted)) - set(labels)
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")
    confusion = {
        left: {right: 0 for right in labels}
        for left in labels
    }
    for left, right in zip(gold, predicted):
        confusion[left][right] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in labels if other != label)
        false_negative = sum(confusion[label][other] for other in labels if other != label)
        support = sum(confusion[label].values())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    accuracy = sum(left == right for left, right in zip(gold, predicted)) / len(gold)
    macro_f1 = sum(float(per_class[label]["f1"]) for label in labels) / len(labels)
    weighted_f1 = sum(float(per_class[label]["f1"]) * int(per_class[label]["support"]) for label in labels) / len(gold)
    return {
        "items": len(gold),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "gold_counts": dict(Counter(gold)),
        "prediction_counts": dict(Counter(predicted)),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def full_relation_verification_metrics(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    four_class = classification_metrics(gold, predicted, FOUR_CLASS_LABELS)
    gold_three = [to_three_decision(label) for label in gold]
    predicted_three = [to_three_decision(label) for label in predicted]
    three_decision = classification_metrics(gold_three, predicted_three, THREE_DECISION_LABELS)
    gold_abstain = [label == "ABSTAIN" for label in gold_three]
    predicted_abstain = [label == "ABSTAIN" for label in predicted_three]
    true_abstain = sum(left and right for left, right in zip(gold_abstain, predicted_abstain))
    abstain_total = sum(gold_abstain)
    false_present_on_abstain = sum(
        gold_label == "ABSTAIN" and predicted_label == "PRESENT"
        for gold_label, predicted_label in zip(gold_three, predicted_three)
    )
    return {
        "four_class": four_class,
        "three_decision": three_decision,
        "abstain_recall": true_abstain / abstain_total if abstain_total else 0.0,
        "gold_abstain_false_present_rate": false_present_on_abstain / abstain_total if abstain_total else 0.0,
    }
