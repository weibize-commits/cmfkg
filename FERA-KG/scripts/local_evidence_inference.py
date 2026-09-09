from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fera_kg.baselines import graph_feature_dict  # noqa: E402
from fera_kg.local_evidence_rag import structured_full_text  # noqa: E402
import local_evidence_features as rag  # noqa: E402
import local_evidence_pipeline as v3  # noqa: E402
import fusion_models as et  # noqa: E402
import fusion_features as sg  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_expert(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expert must be TASK=PATH")
    task, path = value.split("=", 1)
    if task not in sg.TASKS:
        raise argparse.ArgumentTypeError(f"unknown expert task: {task}")
    return task, Path(path)


def load_expert_map(task: str, path: Path) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, list[float]] = {}
    for row in payload["predictions"]:
        item_id = str(row["item_id"])
        values = row["probabilities"][task]
        if len(values) != len(sg.LABELS):
            raise ValueError(f"{task}: invalid probability length for {item_id}")
        output[item_id] = [float(value) for value in values]
    return output


def rag_probabilities(
    artifact: dict[str, Any],
    rows: list[dict[str, Any]],
    paths: list[dict[str, Any]],
) -> np.ndarray:
    retriever = artifact["retriever"]
    retriever.top_k = 5
    retrieved = [
        retriever.retrieve(item, path) for item, path in zip(rows, paths)
    ]
    full_text = artifact["full_vectorizer"].transform(
        [structured_full_text(row) for row in rows]
    )
    graph = artifact["graph_vectorizer"].transform(
        [graph_feature_dict(item, path) for item, path in zip(rows, paths)]
    )
    numeric = artifact["numeric_vectorizer"].transform(
        [row.feature_record for row in retrieved]
    )
    matrix = hstack((full_text, graph, numeric), format="csr", dtype=np.float32)
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    models = artifact["models"]
    return np.column_stack(
        [
            rag.align_probability(models["verdict"], matrix),
            rag.align_probability(models["sufficiency"], matrix),
            rag.align_probability(models["entity"], matrix),
        ]
    )


def add_retrieval_features(
    artifact: dict[str, Any],
    rows: list[dict[str, Any]],
    paths: list[dict[str, Any]],
    frame: pd.DataFrame,
) -> None:
    retriever = artifact["retriever"]
    for top_k in (2, 3, 5):
        retriever.top_k = top_k
        records = [
            retriever.retrieve(item, path).feature_record
            for item, path in zip(rows, paths)
        ]
        for key in records[0]:
            frame[f"local_top{top_k}__{key}"] = [row[key] for row in records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--graph-paths", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=PROJECT_ROOT / "artifacts" / "rag_augmented_sg_fera_v3.joblib")
    parser.add_argument("--expert", type=parse_expert, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    paths = read_jsonl(args.graph_paths)
    if [str(row["item_id"]) for row in rows] != [str(row["item_id"]) for row in paths]:
        raise ValueError("feature and graph-path order differ")
    if any(str(row.get("label_status", "")) != "FIXED_INFERENCE_PLACEHOLDER_NOT_HUMAN_GOLD" for row in rows):
        raise ValueError("dataset is not a sealed no-human-gold inference input")

    expert_paths = dict(args.expert)
    if set(expert_paths) != set(sg.TASKS):
        raise ValueError(f"exactly these experts are required: {sg.TASKS}")
    expert_maps = {
        task: load_expert_map(task, expert_paths[task]) for task in sg.TASKS
    }
    item_ids = {str(row["item_id"]) for row in rows}
    for task, values in expert_maps.items():
        if set(values) != item_ids:
            raise ValueError(f"{task}: prediction item set does not match input")

    artifact = joblib.load(args.artifact)
    path_map = {str(row["item_id"]): row for row in paths}
    frame = sg.feature_frame(rows, path_map, expert_maps)
    add_retrieval_features(artifact["rag"], rows, paths, frame)
    local_rag = rag_probabilities(artifact["rag"], rows, paths)
    for name, values in v3.rag_probability_columns(local_rag).items():
        frame[name] = values

    expected_features = list(artifact["features"])
    missing = sorted(set(expected_features) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected_features))
    allow_subset = bool(artifact.get("allow_feature_subset", False))
    if missing or (extra and not allow_subset):
        raise ValueError(f"feature schema mismatch missing={missing} extra={extra}")
    frame = frame[expected_features]

    seed_probabilities = [
        et.aligned_probabilities(model, frame) for model in artifact["models"]
    ]
    probabilities = np.mean(np.stack(seed_probabilities), axis=0)
    predicted = rag.decide(v3.stage_view(probabilities), artifact["thresholds"])

    payload = {
        "status": "BLIND_PREDICTIONS_COMPLETE_HUMAN_GOLD_NOT_ACCESSED",
        "method": "RAG-augmented SG-FERA v3",
        "dataset": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "graph_paths": str(args.graph_paths),
        "graph_paths_sha256": sha256(args.graph_paths),
        "artifact": str(args.artifact),
        "artifact_sha256": sha256(args.artifact),
        "expert_inputs": {
            task: {"path": str(path), "sha256": sha256(path)}
            for task, path in expert_paths.items()
        },
        "items": len(rows),
        "labels": list(sg.LABELS),
        "thresholds": artifact["thresholds"],
        "predictions": [
            {
                "item_id": str(row["item_id"]),
                "predicted_label": label,
                "probabilities": {
                    name: float(value)
                    for name, value in zip(sg.LABELS, values)
                },
            }
            for row, label, values in zip(rows, predicted, probabilities)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "items": len(rows),
                "predicted_counts": {
                    label: predicted.count(label) for label in sg.LABELS
                },
                "sha256": sha256(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
