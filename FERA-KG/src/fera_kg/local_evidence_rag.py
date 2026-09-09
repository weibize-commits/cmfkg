from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np


SEGMENT_BOUNDARY = re.compile(r"(?<=[。！？!?；;])|\n+")


def query_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(field, ""))
        for field in (
            "head_type",
            "head_text",
            "relation",
            "relation_display",
            "relation_definition",
            "tail_type",
            "tail_text",
        )
    )


def structured_full_text(item: dict[str, Any]) -> str:
    return (
        f"[QUERY] {query_text(item)} "
        f"[HEAD] {item.get('head_text', '')} "
        f"[RELATION] {item.get('relation', '')} "
        f"[TAIL] {item.get('tail_text', '')} "
        f"[PASSAGE] {item.get('passage_text', '')}"
    )


def split_segments(text: str) -> list[tuple[int, int, str]]:
    output: list[tuple[int, int, str]] = []
    cursor = 0
    for part in SEGMENT_BOUNDARY.split(text):
        if not part:
            continue
        start = text.find(part, cursor)
        if start < 0:
            start = cursor
        end = start + len(part)
        cleaned = part.strip()
        if cleaned:
            left_trim = len(part) - len(part.lstrip())
            right_trim = len(part) - len(part.rstrip())
            output.append((start + left_trim, end - right_trim, cleaned))
        cursor = end
    if not output and text.strip():
        start = len(text) - len(text.lstrip())
        output.append((start, len(text.rstrip()), text.strip()))
    return output


def _contains_span(start: int, end: int, span_start: int, span_end: int) -> bool:
    return start <= span_start and span_end <= end


def serialize_paths(path_item: dict[str, Any], limit: int = 6) -> str:
    parts: list[str] = []
    for path in (path_item.get("paths") or [])[:limit]:
        nodes = [str(node.get("text", "")) for node in path.get("nodes") or []]
        edges = [str(edge) for edge in path.get("edges") or []]
        sequence: list[str] = []
        for index, node in enumerate(nodes):
            sequence.append(node)
            if index < len(edges):
                sequence.append(edges[index])
        kind = "PASSAGE_PATH" if path.get("passage_conditioned") else "SEMANTIC_PATH"
        parts.append(f"[{kind}] " + " -> ".join(sequence))
    return " ".join(parts)


@dataclass
class RetrievedEvidence:
    text: str
    feature_record: dict[str, float]
    selected_indices: list[int]
    scores: list[float]


class PassageLocalRetriever:
    """Retrieve evidence units only from the frozen passage for each item."""

    def __init__(self, *, top_k: int = 3, max_features: int = 120_000) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 4),
            min_df=2,
            max_features=max_features,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )

    def fit(self, items: list[dict[str, Any]]) -> "PassageLocalRetriever":
        documents: list[str] = []
        for item in items:
            documents.append(query_text(item))
            documents.extend(segment[2] for segment in split_segments(str(item["passage_text"])))
        self.vectorizer.fit(documents)
        return self

    def retrieve(
        self,
        item: dict[str, Any],
        path_item: dict[str, Any] | None = None,
    ) -> RetrievedEvidence:
        passage = str(item.get("passage_text", ""))
        segments = split_segments(passage)
        if not segments:
            segments = [(0, 0, "")]
        query = query_text(item)
        query_matrix = self.vectorizer.transform([query])
        segment_matrix = self.vectorizer.transform([segment[2] for segment in segments])
        similarities = np.asarray((segment_matrix @ query_matrix.T).toarray()).reshape(-1)

        head_start = int(item.get("head_start", 0) or 0)
        head_end = int(item.get("head_end", head_start) or head_start)
        tail_start = int(item.get("tail_start", 0) or 0)
        tail_end = int(item.get("tail_end", tail_start) or tail_start)
        mandatory: set[int] = set()
        for index, (start, end, _) in enumerate(segments):
            if _contains_span(start, end, head_start, head_end) or _contains_span(
                start, end, tail_start, tail_end
            ):
                mandatory.add(index)
                if index > 0:
                    mandatory.add(index - 1)
                if index + 1 < len(segments):
                    mandatory.add(index + 1)

        ranked = list(np.argsort(-similarities))
        selected = set(ranked[: self.top_k]) | mandatory
        selected_indices = sorted(selected)
        selected_scores = [float(similarities[index]) for index in selected_indices]
        selected_text = " ".join(segments[index][2] for index in selected_indices)
        graph_text = serialize_paths(path_item or {})
        evidence_text = (
            f"[QUERY] {query} [RETRIEVED_LOCAL_EVIDENCE] {selected_text} "
            f"[TARGET_EDGE_DELETED_GRAPH] {graph_text}"
        )

        head = str(item.get("head_text", ""))
        tail = str(item.get("tail_text", ""))
        score_order = sorted((float(score) for score in similarities), reverse=True)
        max_score = score_order[0] if score_order else 0.0
        second_score = score_order[1] if len(score_order) > 1 else 0.0
        selected_characters = sum(len(segments[index][2]) for index in selected_indices)
        replacement = sum(passage.count(mark) for mark in "□■�")
        punctuation = sum(passage.count(mark) for mark in "。！？!?；;，,：:")
        exact_head_span = passage[head_start:head_end] == head if head else False
        exact_tail_span = passage[tail_start:tail_end] == tail if tail else False
        record = {
            "bias": 1.0,
            "passage_length_log": math.log1p(len(passage)),
            "segment_count_log": math.log1p(len(segments)),
            "selected_segment_count": float(len(selected_indices)),
            "selected_character_ratio": selected_characters / max(len(passage), 1),
            "retrieval_max_score": max_score,
            "retrieval_second_score": second_score,
            "retrieval_score_margin": max_score - second_score,
            "retrieval_mean_selected_score": float(np.mean(selected_scores))
            if selected_scores
            else 0.0,
            "full_head_count_log": math.log1p(passage.count(head)) if head else 0.0,
            "full_tail_count_log": math.log1p(passage.count(tail)) if tail else 0.0,
            "selected_has_head": float(bool(head) and head in selected_text),
            "selected_has_tail": float(bool(tail) and tail in selected_text),
            "selected_has_both": float(bool(head) and bool(tail) and head in selected_text and tail in selected_text),
            "exact_head_span": float(exact_head_span),
            "exact_tail_span": float(exact_tail_span),
            "replacement_character_ratio": replacement / max(len(passage), 1),
            "punctuation_density": punctuation / max(len(passage), 1),
            "span_distance_log": math.log1p(abs(tail_start - head_end)),
            "cross_sentence": float(
                any(mark in passage[min(head_end, tail_end) : max(head_start, tail_start)] for mark in "。！？!?\n")
            ),
        }
        return RetrievedEvidence(evidence_text, record, selected_indices, selected_scores)


def omission_variant(item: dict[str, Any]) -> dict[str, Any]:
    """Create a same-length missing-evidence variant while preserving endpoints."""

    output = dict(item)
    passage = str(item.get("passage_text", ""))
    head_start = int(item.get("head_start", 0) or 0)
    head_end = int(item.get("head_end", head_start) or head_start)
    tail_start = int(item.get("tail_start", 0) or 0)
    tail_end = int(item.get("tail_end", tail_start) or tail_start)
    protected = [(max(head_start, 0), max(head_end, 0)), (max(tail_start, 0), max(tail_end, 0))]
    segments = split_segments(passage)
    relevant = [
        (start, end)
        for start, end, _ in segments
        if any(_contains_span(start, end, left, right) for left, right in protected)
    ]
    if relevant:
        mask_start = min(start for start, _ in relevant)
        mask_end = max(end for _, end in relevant)
    else:
        mask_start = max(min(head_start, tail_start) - 60, 0)
        mask_end = min(max(head_end, tail_end) + 60, len(passage))
    characters = list(passage)
    for index in range(mask_start, mask_end):
        if any(left <= index < right for left, right in protected):
            continue
        if not characters[index].isspace():
            characters[index] = "□"
    output["passage_text"] = "".join(characters)
    output["controlled_evidence_omission"] = True
    return output
