from fera_kg.schema_guard import apply_schema_guard, inspect_relation_signature


def test_valid_signature_retains_probabilities() -> None:
    probabilities = [0.1, 0.2, 0.3, 0.4]
    guarded, decision = apply_schema_guard(
        probabilities, "treats", "食材", "病症"
    )
    assert guarded == probabilities
    assert decision.signature_valid
    assert decision.routed_label is None


def test_invalid_signature_routes_to_entity_type_error() -> None:
    guarded, decision = apply_schema_guard(
        [0.7, 0.1, 0.1, 0.1], "treats", "食疗方", "病症"
    )
    assert guarded == [0.0, 0.0, 0.0, 1.0]
    assert not decision.signature_valid
    assert decision.routed_label == "ENTITY_OR_TYPE_ERROR"


def test_unknown_relation_passes_through_without_rejection() -> None:
    decision = inspect_relation_signature("unknownRelation", "食材", "病症")
    assert not decision.signature_known
    assert decision.signature_valid
    assert decision.routed_label is None
