from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import local_evidence_features as rag  # noqa: E402
import local_evidence_pipeline as v3  # noqa: E402
import local_evidence_inference as blind  # noqa: E402
import fusion_models as et  # noqa: E402
import fusion_features as sg  # noqa: E402

EXPECTED_DEEP_SEEDS = [20260921, 20260922, 20260923, 20260924, 20260925]
EXPECTED_FUSION_SEEDS = [20261101, 20261102, 20261103, 20261104, 20261105]
EXPECTED_FAMILIES = [*sg.TASKS, "same_backbone_no_ecrp"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--graph-paths", type=Path, required=True)
    parser.add_argument("--expert-payload", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "artifacts/rag_augmented_source_work_oof_v1.joblib",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.dataset)
    paths = read_jsonl(args.graph_paths)
    item_ids = [str(row["item_id"]) for row in rows]
    if item_ids != [str(row["item_id"]) for row in paths]:
        raise ValueError("Dataset and graph paths are not aligned")

    expert_payload = json.loads(args.expert_payload.read_text(encoding="utf-8"))
    expert_rows = expert_payload["predictions"]
    labels = list(sg.LABELS)
    if list(expert_payload.get("labels", [])) != labels:
        raise ValueError("Expert payload label order does not match the fixed label order")
    if [int(seed) for seed in expert_payload.get("seeds", [])] != EXPECTED_DEEP_SEEDS:
        raise ValueError("Expert payload does not contain the five fixed full-training seeds")
    if set(expert_payload.get("families", {})) != set(EXPECTED_FAMILIES):
        raise ValueError("Expert payload does not contain the five expected model families")
    if int(expert_payload.get("items", len(expert_rows))) != len(expert_rows):
        raise ValueError("Expert payload item count does not match its row count")
    if item_ids != [str(row["item_id"]) for row in expert_rows]:
        raise ValueError("Expert payload and dataset are not aligned")
    expert_maps: dict[str, dict[str, list[float]]] = {task: {} for task in sg.TASKS}
    optional_no_ecrp: dict[str, list[float]] = {}
    for row in expert_rows:
        item_id = str(row["item_id"])
        for task in sg.TASKS:
            values = row["probabilities"][task]
            expert_maps[task][item_id] = [float(value) for value in values]
        if "same_backbone_no_ecrp" in row["probabilities"]:
            optional_no_ecrp[item_id] = [float(value) for value in row["probabilities"]["same_backbone_no_ecrp"]]

    artifact = joblib.load(args.artifact)
    if list(artifact.get("labels", [])) != labels:
        raise ValueError("Fusion artifact label order does not match the fixed label order")
    if artifact.get("source_work_oof") is not True:
        raise ValueError("Fusion artifact is not registered as source-work-grouped OOF")
    if int(artifact.get("oof_deep_seed_count", -1)) != len(EXPECTED_DEEP_SEEDS):
        raise ValueError("Fusion artifact records an unexpected OOF deep-seed count")
    if artifact.get("source_work_crossfit_verified") is not True:
        raise ValueError("Fusion artifact does not record verified source-work fold separation")
    if int(artifact.get("training_source_works", -1)) != 29:
        raise ValueError("Fusion artifact does not record the 29 training source works")
    if len(artifact.get("models", [])) != len(EXPECTED_FUSION_SEEDS):
        raise ValueError("Fusion artifact does not contain five full fusion models")
    recorded_fusion_seeds = [
        int(seed)
        for seed in artifact.get("fixed_hyperparameters", {}).get("seeds", [])
    ]
    if recorded_fusion_seeds != EXPECTED_FUSION_SEEDS:
        raise ValueError("Fusion artifact does not contain the five fixed fusion seeds")
    for variant, models in artifact.get("models_by_variant", {}).items():
        if len(models) != len(EXPECTED_FUSION_SEEDS):
            raise ValueError(f"Fusion variant {variant} does not contain five models")
    path_map = {str(row["item_id"]): row for row in paths}
    frame = sg.feature_frame(rows, path_map, expert_maps)
    blind.add_retrieval_features(artifact["rag"], rows, paths, frame)
    local_probabilities = blind.rag_probabilities(artifact["rag"], rows, paths)
    for name, values in v3.rag_probability_columns(local_probabilities).items():
        frame[name] = values
    expected = list(artifact["features"])
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing fusion features: {missing}")
    frame = frame[expected]

    probabilities_by_seed = [et.aligned_probabilities(model, frame) for model in artifact["models"]]
    probabilities = np.mean(np.stack(probabilities_by_seed), axis=0)
    if not np.all(np.isfinite(probabilities)) or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-5
    ):
        raise ValueError("Fused probability ensemble is not finite and row-normalised")
    predictions = rag.decide(v3.stage_view(probabilities), artifact["thresholds"])

    variant_outputs: dict[str, dict[str, Any]] = {}
    for variant, models in artifact.get("models_by_variant", {}).items():
        selected = list(artifact["features_by_variant"][variant])
        variant_frame = frame[selected]
        by_seed = [et.aligned_probabilities(model, variant_frame) for model in models]
        variant_probabilities = np.mean(np.stack(by_seed), axis=0)
        if not np.all(np.isfinite(variant_probabilities)) or not np.allclose(
            variant_probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-5
        ):
            raise ValueError(f"Fusion variant {variant} produced invalid probabilities")
        variant_outputs[variant] = {
            "probabilities": variant_probabilities,
            "predictions": rag.decide(
                v3.stage_view(variant_probabilities), artifact["thresholds"]
            ),
        }

    component_predictions: dict[str, list[str]] = {}
    for task in sg.TASKS:
        component_predictions[task] = [
            labels[int(np.argmax(expert_maps[task][item_id]))] for item_id in item_ids
        ]
    if optional_no_ecrp:
        component_predictions["same_backbone_no_ecrp"] = [
            labels[int(np.argmax(optional_no_ecrp[item_id]))] for item_id in item_ids
        ]

    payload = {
        "status": "UNIFIED_FULL_TRAINING_ENSEMBLE_AND_SOURCE_WORK_OOF_FUSION_COMPLETE",
        "configuration": {
            "deep_ensemble": "five full-training models per expert with seeds 20260921-20260925",
            "deep_seed_count": len(EXPECTED_DEEP_SEEDS),
            "deep_seeds": EXPECTED_DEEP_SEEDS,
            "deep_probability_aggregation": "arithmetic mean before fusion",
            "fusion_training": "source-work-grouped out-of-fold probabilities on 1,416 training items",
            "fusion_seed_count": len(EXPECTED_FUSION_SEEDS),
            "fusion_seeds": EXPECTED_FUSION_SEEDS,
            "fusion_probability_aggregation": "arithmetic mean before selective decision",
            "thresholds": artifact["thresholds"],
            "passage_diagnostic": str(rows[0].get("passage_diagnostic", "not_applicable")) if rows else None,
        },
        "dataset": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "graph_paths": str(args.graph_paths),
        "graph_paths_sha256": sha256(args.graph_paths),
        "expert_payload": str(args.expert_payload),
        "expert_payload_sha256": sha256(args.expert_payload),
        "artifact": str(args.artifact),
        "artifact_sha256": sha256(args.artifact),
        "items": len(rows),
        "labels": labels,
        "predictions": [
            {
                "item_id": item_id,
                "predicted_label": predicted,
                "probabilities": {label: float(value) for label, value in zip(labels, values)},
                "component_predictions": {
                    name: component_predictions[name][index] for name in component_predictions
                },
                "component_probabilities": {
                    task: expert_maps[task][item_id] for task in sg.TASKS
                }
                | ({"same_backbone_no_ecrp": optional_no_ecrp[item_id]} if optional_no_ecrp else {}),
                "local_evidence_probabilities": {
                    "verdict_support": float(local_probabilities[index, 0]),
                    "evidence_sufficient": float(local_probabilities[index, 1]),
                    "entity_valid": float(local_probabilities[index, 2]),
                },
                "fusion_retraining_ablations": {
                    variant: {
                        "predicted_label": values["predictions"][index],
                        "probabilities": {
                            label: float(probability)
                            for label, probability in zip(
                                labels, values["probabilities"][index]
                            )
                        },
                    }
                    for variant, values in variant_outputs.items()
                },
            }
            for index, (item_id, predicted, values) in enumerate(zip(item_ids, predictions, probabilities))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "items": len(rows), "sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
