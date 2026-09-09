from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fera_kg.local_evidence_rag import PassageLocalRetriever  # noqa: E402
import local_evidence_features as rag  # noqa: E402
import fusion_models as et  # noqa: E402
import fusion_features as sg  # noqa: E402


FOLD_ASSIGNMENTS = PROJECT_ROOT.parent / "confidential_review_data" / "grouped_folds" / "fold_assignments.jsonl"
RESULT = PROJECT_ROOT / "results" / "development" / "rag_augmented_sg_fera_v3.json"
GRID = PROJECT_ROOT / "results" / "development" / "rag_augmented_sg_fera_v3_grid.json"
PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "rag_augmented_sg_fera_v3_predictions.jsonl"
)
ARTIFACT = PROJECT_ROOT / "artifacts" / "rag_augmented_sg_fera_v3.joblib"
REPORT = PROJECT_ROOT / "reports" / "rag_augmented_sg_fera_v3.md"
SEEDS = [20261101, 20261102, 20261103, 20261104, 20261105]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rag_probability_columns(values: np.ndarray) -> dict[str, np.ndarray]:
    supported = values[:, 0]
    insufficient = values[:, 1]
    entity = values[:, 2]
    return {
        "local_rag__p_supported": supported,
        "local_rag__p_insufficient": insufficient,
        "local_rag__p_entity_error": entity,
        "local_rag__judgeability": 1.0 - np.maximum(insufficient, entity),
        "local_rag__abstain_score": np.maximum(insufficient, entity),
        "local_rag__abstain_margin": np.abs(insufficient - entity),
        "local_rag__support_margin": np.abs(supported - 0.5),
    }


def add_retrieval_features(
    train: list[dict[str, Any]],
    development: list[dict[str, Any]],
    train_paths: list[dict[str, Any]],
    development_paths: list[dict[str, Any]],
    train_frame: pd.DataFrame,
    development_frame: pd.DataFrame,
) -> None:
    retriever = PassageLocalRetriever(top_k=5).fit(train)
    for top_k in (2, 3, 5):
        retriever.top_k = top_k
        train_records = [
            retriever.retrieve(item, path).feature_record
            for item, path in zip(train, train_paths)
        ]
        development_records = [
            retriever.retrieve(item, path).feature_record
            for item, path in zip(development, development_paths)
        ]
        for key in train_records[0]:
            train_frame[f"local_top{top_k}__{key}"] = [row[key] for row in train_records]
            development_frame[f"local_top{top_k}__{key}"] = [
                row[key] for row in development_records
            ]


def oof_rag_probabilities(
    train: list[dict[str, Any]], train_paths: list[dict[str, Any]]
) -> np.ndarray:
    assignments = {
        str(row["item_id"]): int(row["fold"])
        for row in rag.read_jsonl(FOLD_ASSIGNMENTS)
    }
    if set(assignments) != {str(row["item_id"]) for row in train}:
        raise ValueError("RAG OOF assignments do not match training items")
    output = np.zeros((len(train), 3), dtype=np.float64)
    covered = np.zeros(len(train), dtype=bool)
    for fold in range(5):
        fit_indices = [
            index
            for index, row in enumerate(train)
            if assignments[str(row["item_id"])] != fold
        ]
        held_indices = [
            index
            for index, row in enumerate(train)
            if assignments[str(row["item_id"])] == fold
        ]
        fit_rows = [train[index] for index in fit_indices]
        fit_paths = [train_paths[index] for index in fit_indices]
        held_rows = [train[index] for index in held_indices]
        held_paths = [train_paths[index] for index in held_indices]
        blocks = rag.build_feature_blocks(
            fit_rows, held_rows, fit_paths, held_paths, top_k=5
        )
        labels = np.asarray(
            [rag.LABELS.index(str(row["label"])) for row in fit_rows], dtype=np.int64
        )
        probabilities, _ = rag.stage_probabilities(
            blocks["full_train"],
            blocks["full_development"],
            labels,
            c=0.25,
            omission_matrix=blocks["full_omission"],
            omission_weight=0.35,
        )
        output[held_indices] = probabilities
        covered[held_indices] = True
        print(f"RAG_OOF_FOLD_COMPLETE fold={fold} items={len(held_indices)}", flush=True)
    if not bool(covered.all()):
        raise RuntimeError("not all training items received OOF RAG probabilities")
    return output


def full_rag_probabilities(
    train: list[dict[str, Any]],
    development: list[dict[str, Any]],
    train_paths: list[dict[str, Any]],
    development_paths: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    blocks = rag.build_feature_blocks(
        train, development, train_paths, development_paths, top_k=5
    )
    labels = np.asarray(
        [rag.LABELS.index(str(row["label"])) for row in train], dtype=np.int64
    )
    probabilities, models = rag.stage_probabilities(
        blocks["full_train"],
        blocks["full_development"],
        labels,
        c=0.25,
        omission_matrix=blocks["full_omission"],
        omission_weight=0.35,
    )
    return probabilities, {
        "models": models,
        "retriever": blocks["retriever"],
        "full_vectorizer": blocks["full_vectorizer"],
        "graph_vectorizer": blocks["graph_vectorizer"],
        "numeric_vectorizer": blocks["numeric_vectorizer"],
    }


def stage_view(probabilities: np.ndarray) -> np.ndarray:
    support_given_judgeable = probabilities[:, 0] / np.maximum(
        probabilities[:, 0] + probabilities[:, 1], 1.0e-12
    )
    return np.column_stack(
        [support_given_judgeable, probabilities[:, 2], probabilities[:, 3]]
    )


def select_thresholds(
    rows: list[dict[str, Any]], probabilities: np.ndarray
) -> dict[str, Any]:
    selected = rag.select_thresholds(
        [str(row["label"]) for row in rows], stage_view(probabilities)
    )
    return {
        "thresholds": selected["thresholds"],
        "predicted": selected["predicted"],
        "metrics": sg.compact(selected["metrics"]),
    }


def main() -> None:
    train, train_path_map = sg.load_rows("train")
    development, development_path_map = sg.load_rows("development")
    train_paths = [train_path_map[str(row["item_id"])] for row in train]
    development_paths = [
        development_path_map[str(row["item_id"])] for row in development
    ]
    train_maps = {task: sg.oof_probabilities(task) for task in sg.TASKS}
    development_maps = {
        task: sg.full_development_probabilities(task) for task in sg.TASKS
    }
    train_frame = sg.feature_frame(train, train_path_map, train_maps)
    development_frame = sg.feature_frame(
        development, development_path_map, development_maps
    )
    add_retrieval_features(
        train,
        development,
        train_paths,
        development_paths,
        train_frame,
        development_frame,
    )
    train_rag = oof_rag_probabilities(train, train_paths)
    development_rag, rag_artifact = full_rag_probabilities(
        train, development, train_paths, development_paths
    )
    for name, values in rag_probability_columns(train_rag).items():
        train_frame[name] = values
    for name, values in rag_probability_columns(development_rag).items():
        development_frame[name] = values

    labels = [str(row["label"]) for row in train]
    oof_predictions = rag.decide(train_rag, {"entity": 0.5, "insufficient": 0.5, "supported": 0.5})
    oof_metrics = sg.evaluate(train, oof_predictions)
    full_rag_selected = rag.select_thresholds(
        [str(row["label"]) for row in development], development_rag
    )

    trials: list[dict[str, Any]] = []
    for max_depth in (8, 12, 16, None):
        for min_samples_leaf in (2, 3, 5, 8):
            for max_features in ("sqrt", 0.5):
                parameters = {
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                }
                model = et.make_model(
                    train_frame,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    seed=SEEDS[0],
                    n_estimators=250,
                )
                model.fit(train_frame, labels)
                probabilities = et.aligned_probabilities(model, development_frame)
                selected = select_thresholds(development, probabilities)
                metrics = selected["metrics"]
                safe = (
                    metrics["abstain_recall"] >= 0.70
                    and metrics["gold_abstain_false_present_rate"] <= 0.08
                )
                objective = 0.55 * metrics["three_decision_macro_f1"] + 0.45 * metrics["four_class_macro_f1"]
                trials.append(
                    {
                        "parameters": parameters,
                        "thresholds": selected["thresholds"],
                        "safe": safe,
                        "objective": objective,
                        "metrics": metrics,
                    }
                )
                print(
                    "GRID",
                    json.dumps(parameters, ensure_ascii=False),
                    f"safe={safe}",
                    f"4F1={metrics['four_class_macro_f1']:.6f}",
                    f"3F1={metrics['three_decision_macro_f1']:.6f}",
                    flush=True,
                )

    trials.sort(
        key=lambda row: (
            row["safe"],
            row["objective"],
            row["metrics"]["four_class_macro_f1"],
        ),
        reverse=True,
    )
    chosen = trials[0]
    seed_models = []
    seed_probabilities = []
    for seed in SEEDS:
        model = et.make_model(
            train_frame,
            **chosen["parameters"],
            seed=seed,
            n_estimators=600,
        )
        model.fit(train_frame, labels)
        seed_models.append(model)
        seed_probabilities.append(et.aligned_probabilities(model, development_frame))
    ensemble_probability = np.mean(np.stack(seed_probabilities), axis=0)
    final = select_thresholds(development, ensemble_probability)

    baseline = json.loads(
        (PROJECT_ROOT / "results" / "development" / "sg_fera_extratrees_v2.json").read_text(
            encoding="utf-8"
        )
    )["development"]["ensemble_metrics"]
    payload = {
        "status": "DEVELOPMENT_ONLY_R9_GOLD_NOT_ACCESSED",
        "method": "RAG-augmented SG-FERA v3",
        "training_protocol": "Four deep experts and the local RAG expert use passage-disjoint OOF predictions for stacker training.",
        "train_items": len(train),
        "development_items": len(development),
        "source_overlap": len(
            {str(row["book"]) for row in train}
            & {str(row["book"]) for row in development}
        ),
        "local_rag_configuration": {
            "top_k": 5,
            "modality": "full passage plus target-edge-deleted graph",
            "C": 0.25,
            "omission_augmentation_weight": 0.35,
        },
        "local_rag_oof_default_metrics": oof_metrics,
        "local_rag_development_metrics": sg.compact(full_rag_selected["metrics"]),
        "selected_grid_row": chosen,
        "ensemble_thresholds": final["thresholds"],
        "ensemble_metrics": final["metrics"],
        "difference_vs_sg_fera_extratrees_v2": et.difference(final["metrics"], baseline),
        "feature_count": len(train_frame.columns),
        "input_hashes": {
            "train": sha256(sg.DATA_ROOT / "train.jsonl"),
            "development": sha256(sg.DATA_ROOT / "development.jsonl"),
            "fold_assignments": sha256(FOLD_ASSIGNMENTS),
        },
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    GRID.write_text(json.dumps({"trials": trials}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with PREDICTIONS.open("w", encoding="utf-8") as handle:
        for row, prediction, values in zip(development, final["predicted"], ensemble_probability):
            handle.write(
                json.dumps(
                    {
                        "item_id": row["item_id"],
                        "passage_id": row["passage_id"],
                        "gold_label": row["label"],
                        "predicted_label": prediction,
                        "probabilities": {
                            label: float(value) for label, value in zip(sg.LABELS, values)
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    joblib.dump(
        {
            "models": seed_models,
            "rag": rag_artifact,
            "thresholds": final["thresholds"],
            "features": list(train_frame.columns),
            "labels": sg.LABELS,
        },
        ARTIFACT,
    )
    metrics = final["metrics"]
    REPORT.write_text(
        "# RAG-augmented SG-FERA v3\n\n"
        "Development only. R9 labels were not accessed.\n\n"
        f"- Four-class Macro-F1: {metrics['four_class_macro_f1']:.6f}\n"
        f"- Three-decision Macro-F1: {metrics['three_decision_macro_f1']:.6f}\n"
        f"- Abstain recall: {metrics['abstain_recall']:.6f}\n"
        f"- False PRESENT on gold abstain: {metrics['gold_abstain_false_present_rate']:.6f}\n"
        f"- Difference from SG-FERA v2: {payload['difference_vs_sg_fera_extratrees_v2']['four_class_macro_f1']:+.6f}\n",
        encoding="utf-8",
    )
    print("COMPLETE", json.dumps(payload["ensemble_metrics"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
