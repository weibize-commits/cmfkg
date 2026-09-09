from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


_SPACE_OR_PUNCTUATION = re.compile(r"[\s\W_]+", flags=re.UNICODE)
_SOURCE_EXTENSION = re.compile(r"\.(?:txt|pdf|docx?|xlsx?)$", flags=re.IGNORECASE)
_SOURCE_PREFIX = re.compile(r"^\d+[\s._-]*")


def normalize_surface(value: Any) -> str:
    """Return a conservative comparison form without changing Chinese characters."""

    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    return _SPACE_OR_PUNCTUATION.sub("", text).lower()


def normalize_source_title(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = _SOURCE_EXTENSION.sub("", text)
    text = _SOURCE_PREFIX.sub("", text)
    text = re.sub(r"[（(][^）)]*(?:饮食|二种|部分)[^）)]*[）)]", "", text)
    return normalize_surface(text)


def _visible_character_ratio(value: str) -> float:
    if not value:
        return 0.0
    useful = sum(
        1
        for character in value
        if character.isalnum() or "\u3400" <= character <= "\u9fff"
    )
    return useful / len(value)


@dataclass
class EntityIdentityLexicon:
    surface_type_counts: dict[str, Counter[str]]
    surface_role_counts: dict[str, Counter[str]]
    source_titles: set[str]


def fit_identity_lexicon(rows: Iterable[dict[str, Any]]) -> EntityIdentityLexicon:
    """Fit training-only entity/type statistics from labels not marked as identity errors."""

    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_titles: set[str] = set()
    for row in rows:
        for field in ("source_name", "book"):
            title = normalize_source_title(row.get(field, ""))
            if title:
                source_titles.add(title)
        if str(row.get("label", "")) == "ENTITY_OR_TYPE_ERROR":
            continue
        for role in ("head", "tail"):
            surface = normalize_surface(row.get(f"{role}_text", ""))
            entity_type = str(row.get(f"{role}_type", ""))
            if not surface or not entity_type:
                continue
            type_counts[surface][entity_type] += 1
            role_counts[surface][role] += 1
    return EntityIdentityLexicon(
        surface_type_counts=dict(type_counts),
        surface_role_counts=dict(role_counts),
        source_titles=source_titles,
    )


def _span_features(
    row: dict[str, Any], role: str, lexicon: EntityIdentityLexicon
) -> dict[str, float]:
    raw = str(row.get(f"{role}_text", ""))
    surface = normalize_surface(raw)
    entity_type = str(row.get(f"{role}_type", ""))
    passage = str(row.get("passage_text", ""))
    counts = lexicon.surface_type_counts.get(surface, Counter())
    total = sum(counts.values())
    expected = counts.get(entity_type, 0)
    alternative = total - expected
    dominant = max(counts.values(), default=0)
    normalized_passage = normalize_surface(passage)
    source_title = surface in lexicon.source_titles if surface else False
    locally_quoted = bool(
        raw
        and any(
            token in passage
            for token in (f"《{raw}》", f"<{raw}》", f"〈{raw}〉")
        )
    )
    return {
        f"{role}__identity_surface_seen": float(total > 0),
        f"{role}__identity_surface_count": float(total),
        f"{role}__identity_expected_type_count": float(expected),
        f"{role}__identity_alternative_type_count": float(alternative),
        f"{role}__identity_expected_type_ratio": float(expected / max(total, 1)),
        f"{role}__identity_dominant_type_ratio": float(dominant / max(total, 1)),
        f"{role}__identity_seen_only_as_other_type": float(total > 0 and expected == 0),
        f"{role}__identity_single_character": float(len(surface) == 1),
        f"{role}__identity_two_or_fewer_characters": float(len(surface) <= 2),
        f"{role}__identity_visible_character_ratio": _visible_character_ratio(raw),
        f"{role}__identity_normalized_occurrences": float(
            normalized_passage.count(surface) if surface else 0
        ),
        f"{role}__identity_known_source_title": float(source_title),
        f"{role}__identity_quoted_title_context": float(locally_quoted),
        f"{role}__identity_recipe_suffix": float(
            surface.endswith(("方", "汤", "丸", "散", "羹", "粥", "酒", "饮"))
        ),
        f"{role}__identity_meridian_suffix": float(surface.endswith("经")),
    }


def identity_feature_records(
    rows: Iterable[dict[str, Any]], lexicon: EntityIdentityLexicon
) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for row in rows:
        head = _span_features(row, "head", lexicon)
        tail = _span_features(row, "tail", lexicon)
        records.append(
            {
                **head,
                **tail,
                "identity__either_seen_only_as_other_type": max(
                    head["head__identity_seen_only_as_other_type"],
                    tail["tail__identity_seen_only_as_other_type"],
                ),
                "identity__either_known_source_title": max(
                    head["head__identity_known_source_title"],
                    tail["tail__identity_known_source_title"],
                ),
                "identity__either_single_character": max(
                    head["head__identity_single_character"],
                    tail["tail__identity_single_character"],
                ),
            }
        )
    return records


def add_identity_features(frame: Any, rows: list[dict[str, Any]], lexicon: EntityIdentityLexicon) -> Any:
    records = identity_feature_records(rows, lexicon)
    return add_identity_feature_records(frame, records)


def add_identity_feature_records(frame: Any, records: list[dict[str, float]]) -> Any:
    result = frame.copy()
    if len(records) != len(result):
        raise ValueError("identity feature rows do not align with the input frame")
    if not records:
        return result
    for column in records[0]:
        result[column] = [record[column] for record in records]
    return result
