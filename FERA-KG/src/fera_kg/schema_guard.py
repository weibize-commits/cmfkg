from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


LABELS = [
    "SUPPORTED",
    "NOT_SUPPORTED",
    "INSUFFICIENT_CONTEXT",
    "ENTITY_OR_TYPE_ERROR",
]
ENTITY_OR_TYPE_ERROR_INDEX = LABELS.index("ENTITY_OR_TYPE_ERROR")

# These signatures are fixed by the TCFT ontology. They are not estimated from
# development or confirmatory labels.
RELATION_TYPE_SIGNATURES: dict[str, tuple[str, str]] = {
    "treats": ("食材", "病症"),
    "hasEffect": ("食材", "功效"),
    "containsIngredient": ("食疗方", "食材"),
    "treatedByRecipe": ("病症", "食疗方"),
    "usesCookingMethod": ("食疗方", "烹饪方法"),
    "contraindicates": ("食材", "禁忌"),
    "hasNature": ("食材", "食性"),
    "hasFlavor": ("食材", "食味"),
    "entersmeridian": ("食材", "归经"),
}


@dataclass(frozen=True)
class SchemaGuardDecision:
    relation: str
    observed_head_type: str
    observed_tail_type: str
    expected_head_type: str | None
    expected_tail_type: str | None
    signature_known: bool
    signature_valid: bool
    routed_label: str | None

    def to_dict(self) -> dict[str, str | bool | None]:
        return asdict(self)


def inspect_relation_signature(
    relation: str,
    head_type: str,
    tail_type: str,
) -> SchemaGuardDecision:
    expected = RELATION_TYPE_SIGNATURES.get(relation)
    if expected is None:
        return SchemaGuardDecision(
            relation=relation,
            observed_head_type=head_type,
            observed_tail_type=tail_type,
            expected_head_type=None,
            expected_tail_type=None,
            signature_known=False,
            signature_valid=True,
            routed_label=None,
        )
    valid = (head_type, tail_type) == expected
    return SchemaGuardDecision(
        relation=relation,
        observed_head_type=head_type,
        observed_tail_type=tail_type,
        expected_head_type=expected[0],
        expected_tail_type=expected[1],
        signature_known=True,
        signature_valid=valid,
        routed_label=None if valid else "ENTITY_OR_TYPE_ERROR",
    )


def apply_schema_guard(
    probabilities: Sequence[float],
    relation: str,
    head_type: str,
    tail_type: str,
) -> tuple[list[float], SchemaGuardDecision]:
    if len(probabilities) != len(LABELS):
        raise ValueError(
            f"expected {len(LABELS)} probabilities, found {len(probabilities)}"
        )
    decision = inspect_relation_signature(relation, head_type, tail_type)
    if decision.routed_label is None:
        return [float(value) for value in probabilities], decision
    guarded = [0.0] * len(LABELS)
    guarded[ENTITY_OR_TYPE_ERROR_INDEX] = 1.0
    return guarded, decision
