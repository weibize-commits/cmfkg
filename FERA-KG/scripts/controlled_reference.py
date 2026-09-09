from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "confidential_review_data"
CONTROLLED_REFERENCE = DATA_ROOT / "evaluation" / "controlled" / "items.jsonl"
COMPARATOR = DATA_ROOT / "comparators" / "controlled_primary.json"
CANDIDATE = DATA_ROOT / "predictions" / "controlled.json"
LABELS = ["SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_CONTEXT", "ENTITY_OR_TYPE_ERROR"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_gold() -> tuple[list[dict[str, Any]], list[str], dict[str, str], dict[str, Any]]:
    rows = read_jsonl(CONTROLLED_REFERENCE)
    ordered = [str(row["item_id"]) for row in rows]
    gold = {str(row["item_id"]): str(row["label"]) for row in rows}
    statuses = Counter(str(row.get("reference_status", "")) for row in rows)
    evaluator_rows = [dict(row, confirmatory_id=row["item_id"]) for row in rows]
    return evaluator_rows, ordered, gold, {
        "disagreements_adjudicated": statuses.get("third_annotator_adjudication", 0),
        "individual_annotator_identifiers_included": False,
    }


def validate_probabilities(values: Iterable[float], item_id: str, system: str) -> list[float]:
    row = [float(value) for value in values]
    if len(row) != 4 or not math.isclose(sum(row), 1.0, abs_tol=2e-5):
        raise ValueError(f"{system}/{item_id}: invalid four-label probabilities")
    return row


def load_sealed_system(path: Path, ordered_ids: list[str]) -> tuple[list[str], list[list[float]]]:
    rows = read_json(path)["predictions"]
    if [str(row["item_id"]) for row in rows] != ordered_ids:
        raise ValueError(f"prediction order differs: {path}")
    predictions = [str(row["predicted_label"]) for row in rows]
    probabilities = [
        validate_probabilities(
            [row["probabilities"][label] for label in LABELS],
            str(row["item_id"]),
            path.stem,
        )
        for row in rows
    ]
    return predictions, probabilities
