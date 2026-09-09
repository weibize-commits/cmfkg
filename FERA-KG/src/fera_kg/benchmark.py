from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = PROJECT_ROOT.parent / "confidential_review_data" / "model_development"
GRAPH_PATH_DIR = BENCHMARK_DIR / "graph_paths_v1"
LABELED_DEVELOPMENT_SPLITS = {"train", "development", "calibration"}
BENCHMARK_SPLITS = LABELED_DEVELOPMENT_SPLITS | {
    "retrospective_standard_test",
    "source_held_out_test",
}


def load_labeled_split(
    split: str,
    *,
    benchmark_dir: Path = BENCHMARK_DIR,
) -> list[dict[str, Any]]:
    if split not in LABELED_DEVELOPMENT_SPLITS:
        raise ValueError(
            f"{split!r} is not a labeled development split; test labels are accessible only to the evaluation entry point"
        )
    path = benchmark_dir / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any("label" not in row for row in rows):
        raise ValueError(f"{path} is empty or lacks labels")
    return rows


def load_graph_path_split(
    split: str,
    *,
    graph_path_dir: Path = GRAPH_PATH_DIR,
) -> list[dict[str, Any]]:
    """Load graph paths, which are feature-only and contain no benchmark labels."""
    if split not in BENCHMARK_SPLITS:
        raise ValueError(f"unknown benchmark split: {split!r}")
    path = graph_path_dir / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"{path} is empty")
    forbidden = {"label", "gold_label", "gold", "decision"}
    for row in rows:
        if forbidden.intersection(row):
            raise ValueError(f"{path} contains a forbidden label field")
        if row.get("label_fields_present") is not False:
            raise ValueError(f"{path} does not explicitly declare label_fields_present=false")
    return rows


def model_text(item: dict[str, Any]) -> str:
    passage = str(item["passage_text"])
    head_start, head_end = int(item["head_start"]), int(item["head_end"])
    tail_start, tail_end = int(item["tail_start"]), int(item["tail_end"])
    local_start = max(min(head_start, tail_start) - 100, 0)
    local_end = min(max(head_end, tail_end) + 100, len(passage))
    if max(head_start, tail_start) < min(head_end, tail_end):
        distance_token = "SPAN_OVERLAP"
    else:
        distance = max(head_start, tail_start) - min(head_end, tail_end)
        if distance == 0:
            distance_token = "GAP_0"
        elif distance <= 10:
            distance_token = "GAP_1_10"
        elif distance <= 50:
            distance_token = "GAP_11_50"
        elif distance <= 200:
            distance_token = "GAP_51_200"
        else:
            distance_token = "GAP_GT_200"
    between_left, between_right = sorted(((head_start, head_end), (tail_start, tail_end)))
    between = passage[between_left[1]:between_right[0]]
    sentence_token = "SAME_SENTENCE" if not any(mark in between for mark in "。！？!?\n") else "CROSS_SENTENCE"
    pair = (
        f"{item['head_type']} {item['head_text']} {item['relation']} "
        f"{item['tail_type']} {item['tail_text']}"
    )
    return (
        f"[PAIR] {pair} [PAIR_REPEAT] {pair} "
        f"[STRUCTURE] {distance_token} {sentence_token} "
        f"[RELATION] {item['relation']} {item['relation_display']} "
        f"[HEAD_TYPE] {item['head_type']} [HEAD] {item['head_text']} "
        f"[TAIL_TYPE] {item['tail_type']} [TAIL] {item['tail_text']} "
        f"[LOCAL_EVIDENCE] {passage[local_start:local_end]} "
        f"[FULL_EVIDENCE] {passage}"
    )
