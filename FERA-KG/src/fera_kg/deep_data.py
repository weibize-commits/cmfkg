from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


SPECIAL_TOKENS = [
    "[REL]",
    "[/REL]",
    "[QUERY]",
    "[/QUERY]",
    "[TEXT]",
    "[/TEXT]",
    "[H]",
    "[/H]",
    "[T]",
    "[/T]",
    "[BRIDGE]",
    "[GRAPH]",
    "[/GRAPH]",
    "[PATH]",
    "[/PATH]",
]


EDGE_DISPLAY_NAMES = {
    "treats": "治疗或食疗关联",
    "hasEffect": "具有传统功效",
    "containsIngredient": "包含食材",
    "treatedByRecipe": "由食疗方治疗",
    "usesCookingMethod": "采用制备或烹饪方法",
    "contraindicates": "具有食用禁忌",
    "hasNature": "具有食性",
    "hasFlavor": "具有食味",
    "entersmeridian": "归入经络",
    "recordedInText": "记录于古籍来源",
    "affectsOrgan": "与脏腑关联",
    "authoredBy": "由医家撰写",
    "correspondsToOrgan": "与脏腑对应",
    "recommendsIngredient": "推荐食材",
    "mentioned_in": "提及于当前段落",
    "mentions": "当前段落提及",
}


def _valid_span(text: str, start: int, end: int, surface: str) -> bool:
    return 0 <= start < end <= len(text) and text[start:end] == surface


def _locate(text: str, surface: str, start: int, end: int) -> tuple[int, int]:
    if _valid_span(text, start, end, surface):
        return start, end
    found = text.find(surface)
    if found >= 0:
        return found, found + len(surface)
    return -1, -1


def _mark_window(
    passage: str,
    head: tuple[int, int],
    tail: tuple[int, int],
    window_start: int,
    window_end: int,
) -> str:
    spans = []
    if head[0] >= window_start and head[1] <= window_end:
        spans.append((head[0], head[1], "[H]", "[/H]"))
    if tail[0] >= window_start and tail[1] <= window_end:
        spans.append((tail[0], tail[1], "[T]", "[/T]"))
    spans.sort(key=lambda row: (row[0], row[1]))
    if len(spans) == 2 and spans[1][0] < spans[0][1]:
        spans = []
    marked = passage[window_start:window_end]
    for start, end, left, right in reversed(spans):
        local_start, local_end = start - window_start, end - window_start
        marked = marked[:local_start] + left + marked[local_start:local_end] + right + marked[local_end:]
    return marked


def pack_evidence_window(
    item: dict[str, Any],
    *,
    passage_override: Optional[str] = None,
    character_budget: int = 1200,
) -> str:
    """Pack query and provenance text while retaining both candidate endpoints when possible."""
    passage = str(passage_override if passage_override is not None else item["passage_text"])
    head_surface, tail_surface = str(item["head_text"]), str(item["tail_text"])
    if passage_override is None:
        head = _locate(passage, head_surface, int(item["head_start"]), int(item["head_end"]))
        tail = _locate(passage, tail_surface, int(item["tail_start"]), int(item["tail_end"]))
    else:
        head = _locate(passage, head_surface, -1, -1)
        tail = _locate(passage, tail_surface, -1, -1)

    valid_positions = [span for span in (head, tail) if span[0] >= 0]
    if not valid_positions:
        evidence = passage[:character_budget]
    else:
        left = min(span[0] for span in valid_positions)
        right = max(span[1] for span in valid_positions)
        if right - left <= character_budget:
            extra = character_budget - (right - left)
            start = max(0, left - extra // 2)
            end = min(len(passage), right + extra - (left - start))
            start = max(0, end - character_budget)
            evidence = _mark_window(passage, head, tail, start, end)
        else:
            half = max(120, character_budget // 2)
            windows = []
            for span in (head, tail):
                if span[0] < 0:
                    continue
                start = max(0, span[0] - half // 2)
                end = min(len(passage), start + half)
                start = max(0, end - half)
                windows.append(_mark_window(passage, head, tail, start, end))
            evidence = " [BRIDGE] ".join(windows)

    return (
        f"[REL] {item['relation']}。{item['relation_definition']} [/REL] "
        f"[QUERY] [H] {head_surface} [/H]，类型为{item['head_type']}；"
        f"[T] {tail_surface} [/T]，类型为{item['tail_type']}。 [/QUERY] "
        f"[TEXT] {evidence} [/TEXT]"
    )


def _display_edge(edge: str) -> str:
    inverse_prefix = "inverse::"
    if edge.startswith(inverse_prefix):
        base = edge[len(inverse_prefix):]
        return f"逆向{EDGE_DISPLAY_NAMES.get(base, base)}"
    return EDGE_DISPLAY_NAMES.get(edge, edge)


def _path_relevance(path: dict[str, Any], passage: str) -> tuple[float, int, int, str]:
    """Rank paths without labels, favoring source-linked and text-grounded paths."""
    nodes = [
        str(node.get("text", ""))
        for node in path.get("nodes") or []
        if str(node.get("type", "")) != "__PASSAGE__"
    ]
    internal_nodes = nodes[1:-1] if len(nodes) > 2 else []
    lexical_hits = sum(bool(node) and node in passage for node in internal_nodes)
    passage_conditioned = int(bool(path.get("passage_conditioned")))
    semantic_edges = int(path.get("semantic_edges", 0))
    hops = int(path.get("hops", len(path.get("edges") or [])))
    score = 4.0 * passage_conditioned + 1.5 * lexical_hits + 0.25 * semantic_edges - 0.15 * hops
    signature = "|".join(str(edge) for edge in path.get("edges") or [])
    return score, passage_conditioned, -hops, signature


def linearize_paths(
    path_item: dict[str, Any],
    *,
    passage: str,
    max_paths: int = 6,
) -> str:
    """Serialize a small, evidence-ranked local subgraph for token-level alignment."""
    paths = list(path_item.get("paths") or [])
    paths.sort(key=lambda path: _path_relevance(path, passage), reverse=True)
    rendered = []
    for path in paths[:max_paths]:
        nodes = list(path.get("nodes") or [])
        edges = [str(edge) for edge in path.get("edges") or []]
        if len(nodes) != len(edges) + 1:
            continue
        segments = []
        for index, edge in enumerate(edges):
            left = nodes[index]
            right = nodes[index + 1]
            left_text = "当前古籍段落" if str(left.get("type", "")) == "__PASSAGE__" else str(left.get("text", ""))
            right_text = "当前古籍段落" if str(right.get("type", "")) == "__PASSAGE__" else str(right.get("text", ""))
            if index == 0:
                segments.append(f"{left_text}（{left.get('type', '')}）")
            segments.append(f"经由{_display_edge(edge)}到{right_text}（{right.get('type', '')}）")
        provenance = "来源内路径" if path.get("passage_conditioned") else "背景图路径"
        rendered.append(f"[PATH] {provenance}：{'，'.join(segments)}。 [/PATH]")
    if not rendered:
        return "[GRAPH] 未检索到删除目标边后的可用局部路径。 [/GRAPH]"
    return f"[GRAPH] {' '.join(rendered)} [/GRAPH]"


def pack_evidence_and_paths(
    item: dict[str, Any],
    path_item: dict[str, Any],
    *,
    passage_override: Optional[str] = None,
    character_budget: int = 900,
    max_paths: int = 6,
) -> str:
    """Jointly encode the source passage and a compact linearized local subgraph."""
    passage = str(passage_override if passage_override is not None else item["passage_text"])
    evidence = pack_evidence_window(
        item,
        passage_override=passage_override,
        character_budget=character_budget,
    )
    graph = linearize_paths(path_item, passage=passage, max_paths=max_paths)
    return f"{evidence} {graph}"


def _character_ngrams(text: str, n: int = 2) -> set[str]:
    compact = "".join(text.split())
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[index:index + n] for index in range(len(compact) - n + 1)}


def build_counterfactual_map(items: list[dict[str, Any]]) -> dict[str, str]:
    """Choose hard negative passages from training data only."""
    negative_rows = [item for item in items if item["label"] != "SUPPORTED"]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    relation_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in negative_rows:
        groups[(str(item["relation"]), str(item["head_type"]), str(item["tail_type"]))].append(item)
        relation_groups[str(item["relation"])].append(item)
    all_negative = negative_rows
    counterfactuals: dict[str, str] = {}
    for item in items:
        if item["label"] != "SUPPORTED":
            continue
        candidates = groups.get(
            (str(item["relation"]), str(item["head_type"]), str(item["tail_type"])),
            [],
        )
        if not candidates:
            candidates = relation_groups.get(str(item["relation"]), [])
        if not candidates:
            candidates = all_negative
        target_grams = _character_ngrams(str(item["passage_text"]))
        best_score, best_passage = -1.0, ""
        target_length = max(1, len(str(item["passage_text"])))
        for candidate in candidates:
            candidate_text = str(candidate["passage_text"])
            grams = _character_ngrams(candidate_text)
            union = len(target_grams | grams)
            lexical = len(target_grams & grams) / union if union else 0.0
            length_similarity = math.exp(-abs(len(candidate_text) - target_length) / target_length)
            endpoint_bonus = 0.15 * (
                str(item["head_text"]) in candidate_text or str(item["tail_text"]) in candidate_text
            )
            score = lexical + 0.20 * length_similarity + endpoint_bonus
            if score > best_score:
                best_score, best_passage = score, candidate_text
        counterfactuals[str(item["item_id"])] = best_passage
    return counterfactuals


class PathVocabulary:
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, edge_to_id: dict[str, int], type_to_id: dict[str, int]) -> None:
        self.edge_to_id = edge_to_id
        self.type_to_id = type_to_id

    @classmethod
    def fit(
        cls,
        items: Iterable[dict[str, Any]],
        path_items: Iterable[dict[str, Any]],
    ) -> "PathVocabulary":
        edges: Counter[str] = Counter()
        types: Counter[str] = Counter()
        for item in items:
            edges[str(item["relation"])] += 1
            types[str(item["head_type"])] += 1
            types[str(item["tail_type"])] += 1
        for row in path_items:
            for path in row.get("paths") or []:
                edges.update(str(edge) for edge in path.get("edges") or [])
                types.update(str(node.get("type", "")) for node in path.get("nodes") or [])
        edge_to_id = {cls.PAD: 0, cls.UNK: 1}
        type_to_id = {cls.PAD: 0, cls.UNK: 1}
        for value, _ in sorted(edges.items(), key=lambda row: (-row[1], row[0])):
            edge_to_id[value] = len(edge_to_id)
        for value, _ in sorted(types.items(), key=lambda row: (-row[1], row[0])):
            type_to_id[value] = len(type_to_id)
        return cls(edge_to_id, type_to_id)

    def edge_id(self, value: str) -> int:
        return self.edge_to_id.get(value, self.edge_to_id[self.UNK])

    def type_id(self, value: str) -> int:
        return self.type_to_id.get(value, self.type_to_id[self.UNK])

    def to_dict(self) -> dict[str, Any]:
        return {"edge_to_id": self.edge_to_id, "type_to_id": self.type_to_id}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PathVocabulary":
        return cls(
            {str(key): int(value) for key, value in payload["edge_to_id"].items()},
            {str(key): int(value) for key, value in payload["type_to_id"].items()},
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class EncodedGraph:
    edge_ids: list[list[int]]
    node_type_ids: list[list[int]]
    hop_mask: list[list[bool]]
    path_mask: list[bool]
    path_features: list[list[float]]
    graph_features: list[float]
    query_relation_id: int
    head_type_id: int
    tail_type_id: int


def encode_graph(
    item: dict[str, Any],
    path_item: dict[str, Any],
    vocabulary: PathVocabulary,
    *,
    max_paths: int = 16,
    max_hops: int = 3,
) -> EncodedGraph:
    paths = list(path_item.get("paths") or [])
    paths.sort(
        key=lambda path: (
            -int(bool(path.get("passage_conditioned"))),
            -int(path.get("semantic_edges", 0)),
            int(path.get("hops", max_hops)),
            "|".join(str(edge) for edge in path.get("edges") or []),
        )
    )
    paths = paths[:max_paths]
    edge_ids, node_type_ids, hop_mask, path_mask, path_features = [], [], [], [], []
    for path_index in range(max_paths):
        if path_index < len(paths):
            path = paths[path_index]
            edges = [str(edge) for edge in path.get("edges") or []][:max_hops]
            nodes = [str(node.get("type", "")) for node in path.get("nodes") or []][:max_hops + 1]
            hops = min(len(edges), max_hops)
            edge_row = [vocabulary.edge_id(edge) for edge in edges]
            node_row = [vocabulary.type_id(node_type) for node_type in nodes]
            edge_row += [0] * (max_hops - len(edge_row))
            node_row += [0] * (max_hops + 1 - len(node_row))
            edge_ids.append(edge_row)
            node_type_ids.append(node_row)
            hop_mask.append([index < hops for index in range(max_hops)])
            path_mask.append(hops > 0)
            path_features.append([
                float(bool(path.get("passage_conditioned"))),
                float(path.get("semantic_edges", 0)) / max(1, max_hops),
                float(hops) / max(1, max_hops),
            ])
        else:
            edge_ids.append([0] * max_hops)
            node_type_ids.append([0] * (max_hops + 1))
            hop_mask.append([False] * max_hops)
            path_mask.append(False)
            path_features.append([0.0, 0.0, 0.0])
    count = float(path_item.get("path_count", 0))
    passage_count = float(path_item.get("passage_conditioned_path_count", 0))
    semantic_count = float(path_item.get("semantic_only_path_count", 0))
    local = path_item.get("local_inventory") or {}
    graph_features = [
        math.log1p(count) / math.log(33.0),
        passage_count / max(1.0, count),
        semantic_count / max(1.0, count),
        math.log1p(float(local.get("actual_typed_mentions", 0))) / math.log(129.0),
        float(bool(local.get("head_in_actual_inventory"))),
        float(bool(local.get("tail_in_actual_inventory"))),
    ]
    return EncodedGraph(
        edge_ids=edge_ids,
        node_type_ids=node_type_ids,
        hop_mask=hop_mask,
        path_mask=path_mask,
        path_features=path_features,
        graph_features=graph_features,
        query_relation_id=vocabulary.edge_id(str(item["relation"])),
        head_type_id=vocabulary.type_id(str(item["head_type"])),
        tail_type_id=vocabulary.type_id(str(item["tail_type"])),
    )
