from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NamedTuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fera_kg.audit.common import asset_path, iter_jsonl, load_json, write_json, write_markdown  # noqa: E402


Node = tuple[str, str]
PASSAGE_NODE: Node = ("__PASSAGE__", "__QUERY_LOCAL_PASSAGE__")
BENCHMARK_DIR = PROJECT_ROOT.parent / "confidential_review_data" / "model_development"
OUTPUT_DIR = BENCHMARK_DIR / "graph_paths_v1"
SPLIT_FILES = {
    "train": BENCHMARK_DIR / "train.jsonl",
    "development": BENCHMARK_DIR / "development.jsonl",
    "calibration": BENCHMARK_DIR / "calibration.jsonl",
    "retrospective_standard_test": BENCHMARK_DIR / "retrospective_standard_test_features.jsonl",
    "source_held_out_test": BENCHMARK_DIR / "source_held_out_test_features.jsonl",
}


class Edge(NamedTuple):
    neighbour: Node
    relation: str
    direction: str


def _node(value: Any) -> Node:
    if not isinstance(value, dict):
        return "", ""
    return str(value.get("type", "")).strip(), str(value.get("text", "")).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_graph() -> tuple[dict[Node, list[Edge]], Counter[tuple[Node, str, Node]]]:
    adjacency: dict[Node, list[Edge]] = defaultdict(list)
    exact_edges: Counter[tuple[Node, str, Node]] = Counter()
    for row in iter_jsonl(asset_path("asserted_graph")):
        head, tail = _node(row.get("head")), _node(row.get("tail"))
        relation = str(row.get("relation", "")).strip()
        exact_edges[(head, relation, tail)] += 1
        adjacency[head].append(Edge(tail, relation, "forward"))
        adjacency[tail].append(Edge(head, relation, "reverse"))
    for node in adjacency:
        adjacency[node].sort(key=lambda edge: (edge.relation, edge.direction, edge.neighbour))
    return adjacency, exact_edges


def _load_chunk_entities() -> dict[tuple[str, str], set[Node]]:
    result: dict[tuple[str, str], set[Node]] = {}
    for row in iter_jsonl(asset_path("corpus_passages_with_entities")):
        key = (str(row.get("book", "")), str(row.get("chunk_idx", "")))
        result[key] = {_node(entity) for entity in row.get("entities") or []}
    return result


def _selected_chunks(item: dict[str, Any]) -> list[str]:
    value = str(item.get("chunk_idx", "")).strip()
    if not value or value == "NO_MATCH":
        return []
    return [part for part in value.split("|") if part]


def _blocked_query_edge(current: Node, edge: Edge, head: Node, relation: str, tail: Node) -> bool:
    return (
        edge.relation == relation
        and ((current == head and edge.neighbour == tail) or (current == tail and edge.neighbour == head))
    )


def _enumerate_paths(
    adjacency: dict[Node, list[Edge]],
    local_mentions: set[Node],
    head: Node,
    relation: str,
    tail: Node,
    maximum_hops: int,
    raw_cap: int,
) -> tuple[list[dict[str, Any]], bool]:
    paths: dict[tuple[Any, ...], dict[str, Any]] = {}
    explored = 0
    truncated = False
    stack: list[tuple[Node, list[Node], list[str], int]] = [(head, [head], [], 0)]
    sorted_mentions = sorted(local_mentions)
    while stack:
        node, nodes, edge_tokens, semantic_edges = stack.pop()
        if len(edge_tokens) >= maximum_hops:
            continue
        neighbours: list[tuple[Node, str, bool]] = []
        if node == PASSAGE_NODE:
            neighbours.extend((mention, "mentions", False) for mention in sorted_mentions)
        else:
            for edge in adjacency.get(node, []):
                if _blocked_query_edge(node, edge, head, relation, tail):
                    continue
                token = edge.relation if edge.direction == "forward" else f"inverse::{edge.relation}"
                neighbours.append((edge.neighbour, token, True))
            if node in local_mentions:
                neighbours.append((PASSAGE_NODE, "mentioned_in", False))
        neighbours.sort(key=lambda value: (value[1], value[0]), reverse=True)
        for neighbour, edge_token, is_semantic in neighbours:
            explored += 1
            if explored > raw_cap:
                truncated = True
                stack.clear()
                break
            if neighbour in nodes:
                continue
            next_nodes = nodes + [neighbour]
            next_edges = edge_tokens + [edge_token]
            next_semantic = semantic_edges + int(is_semantic)
            if neighbour == tail:
                if next_semantic == 0:
                    continue
                signature = tuple(next_nodes) + ("__EDGES__",) + tuple(next_edges)
                paths[signature] = {
                    "nodes": [
                        {"type": node_type, "text": node_text}
                        for node_type, node_text in next_nodes
                    ],
                    "edges": next_edges,
                    "hops": len(next_edges),
                    "semantic_edges": next_semantic,
                    "passage_conditioned": PASSAGE_NODE in next_nodes,
                }
            elif len(next_edges) < maximum_hops:
                stack.append((neighbour, next_nodes, next_edges, next_semantic))
    ordered = sorted(
        paths.values(),
        key=lambda path: (
            path["hops"],
            -path["semantic_edges"],
            tuple(path["edges"]),
            tuple((node["type"], node["text"]) for node in path["nodes"]),
        ),
    )
    return ordered, truncated


def _select_paths(paths: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    maximum = int(config["maximum_paths_per_item"])
    passage_quota = int(config["passage_conditioned_quota"])
    semantic_quota = int(config["semantic_only_quota"])
    passage = [path for path in paths if path["passage_conditioned"]]
    semantic = [path for path in paths if not path["passage_conditioned"]]
    selected = passage[:passage_quota] + semantic[:semantic_quota]
    selected_signatures = {
        (tuple(path["edges"]), tuple((node["type"], node["text"]) for node in path["nodes"]))
        for path in selected
    }
    for path in paths:
        signature = (
            tuple(path["edges"]),
            tuple((node["type"], node["text"]) for node in path["nodes"]),
        )
        if signature in selected_signatures:
            continue
        selected.append(path)
        selected_signatures.add(signature)
        if len(selected) == maximum:
            break
    return sorted(
        selected[:maximum],
        key=lambda path: (path["hops"], not path["passage_conditioned"], tuple(path["edges"])),
    )


def main() -> None:
    config = load_json(PROJECT_ROOT / "configs" / "graph" / "path_cache.json")
    adjacency, exact_edges = _load_graph()
    chunk_entities = _load_chunk_entities()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_summaries: dict[str, Any] = {}
    output_files: list[Path] = []
    total_items = 0
    for split, input_path in SPLIT_FILES.items():
        rows = list(iter_jsonl(input_path))
        output_path = OUTPUT_DIR / f"{split}.jsonl"
        output_files.append(output_path)
        coverage = Counter()
        origin_coverage: dict[str, Counter[str]] = defaultdict(Counter)
        relation_coverage: dict[str, Counter[str]] = defaultdict(Counter)
        truncated_items = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for item in rows:
                total_items += 1
                head = (str(item["head_type"]), str(item["head_text"]))
                tail = (str(item["tail_type"]), str(item["tail_text"]))
                relation = str(item["relation"])
                actual_mentions: set[Node] = set()
                chunks = _selected_chunks(item)
                for chunk_idx in chunks:
                    actual_mentions.update(chunk_entities.get((str(item["book"]), chunk_idx), set()))
                head_in_inventory = head in actual_mentions
                tail_in_inventory = tail in actual_mentions
                local_mentions = set(actual_mentions)
                local_mentions.update((head, tail))
                raw_paths, truncated = _enumerate_paths(
                    adjacency,
                    local_mentions,
                    head,
                    relation,
                    tail,
                    int(config["maximum_hops"]),
                    int(config["raw_enumeration_cap"]),
                )
                selected = _select_paths(raw_paths, config)
                passage_count = sum(path["passage_conditioned"] for path in selected)
                semantic_count = len(selected) - passage_count
                status = "path_available" if selected else "no_path"
                coverage[status] += 1
                origin_coverage[str(item["dataset_origin"])][status] += 1
                relation_coverage[str(item["relation_group"])][status] += 1
                truncated_items += int(truncated)
                payload = {
                    "item_id": item["item_id"],
                    "split": split,
                    "query": {
                        "head": {"type": head[0], "text": head[1]},
                        "relation": relation,
                        "tail": {"type": tail[0], "text": tail[1]},
                    },
                    "provenance": {
                        "book": item["book"],
                        "chunk_indices": chunks,
                        "resolution": item.get("provenance_resolution", ""),
                    },
                    "local_inventory": {
                        "actual_typed_mentions": len(actual_mentions),
                        "head_in_actual_inventory": head_in_inventory,
                        "tail_in_actual_inventory": tail_in_inventory,
                        "query_endpoints_added_to_passage_node": True,
                    },
                    "target_deletion": {
                        "direct_query_edge_records_before_deletion": exact_edges[(head, relation, tail)],
                        "all_direct_query_predicate_edges_between_endpoints_blocked": True,
                    },
                    "path_count": len(selected),
                    "passage_conditioned_path_count": passage_count,
                    "semantic_only_path_count": semantic_count,
                    "enumeration_truncated": truncated,
                    "paths": selected,
                    "label_fields_present": False,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        split_summaries[split] = {
            "items": len(rows),
            "coverage": dict(coverage),
            "path_coverage_rate": coverage["path_available"] / len(rows) if rows else 0.0,
            "enumeration_truncated_items": truncated_items,
            "origin_coverage": {key: dict(value) for key, value in origin_coverage.items()},
            "relation_group_coverage": {key: dict(value) for key, value in relation_coverage.items()},
        }

    if total_items != 1462:
        raise RuntimeError(f"expected 1462 benchmark items, found {total_items}")
    manifest = {
        "version": config["version"],
        "status": "READY",
        "items": total_items,
        "graph_nodes_with_adjacency": len(adjacency),
        "graph_directed_edge_records": sum(exact_edges.values()),
        "maximum_hops": config["maximum_hops"],
        "maximum_paths_per_item": config["maximum_paths_per_item"],
        "split_summaries": split_summaries,
        "test_labels_accessed": False,
        "output_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in output_files
        ],
    }
    write_json(PROJECT_ROOT / "manifests" / "graph_path_cache_v1.json", manifest)
    summary_lines = "\n".join(
        f"- {split}: {summary['coverage'].get('path_available', 0)}/{summary['items']} = {summary['path_coverage_rate']:.4f}; truncated {summary['enumeration_truncated_items']}"
        for split, summary in split_summaries.items()
    )
    report = f"""# Target-deleted hybrid graph-path cache v1

## Decision

**READY**

- Benchmark items: {total_items}
- Asserted graph edge records: {sum(exact_edges.values())}
- Maximum path length: {config['maximum_hops']}
- Maximum retained paths per item: {config['maximum_paths_per_item']}
- Pure passage co-mention paths accepted: no
- Direct query-predicate edges between endpoints blocked: yes
- Test labels accessed: no

## Split coverage

{summary_lines}

The cache retains both semantic-only and passage-conditioned paths. Query
endpoints are attached to the local passage node because their frozen spans are
part of every benchmark item, while separate flags record whether the archived
chunk entity inventory originally contained each endpoint.
"""
    write_markdown(PROJECT_ROOT / "reports" / "graph_path_cache_v1.md", report)
    print(PROJECT_ROOT / "manifests" / "graph_path_cache_v1.json")


if __name__ == "__main__":
    main()
