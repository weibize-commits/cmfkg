from __future__ import annotations

import hashlib
import json
import platform
import sys
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fera_kg.baselines import graph_feature_dict  # noqa: E402
from fera_kg.evaluation import FOUR_CLASS_LABELS, full_relation_verification_metrics  # noqa: E402
from fera_kg.local_evidence_rag import (  # noqa: E402
    PassageLocalRetriever,
    omission_variant,
    structured_full_text,
)


SEED = 20260901
LABELS = list(FOUR_CLASS_LABELS)
SUPPORTED, NOT_SUPPORTED, INSUFFICIENT, ENTITY_ERROR = range(4)
DATA_DIR = PROJECT_ROOT.parent / "confidential_review_data" / "model_development"
PATH_DIR = DATA_DIR / "graph_paths_v1"
RESULT = PROJECT_ROOT / "results" / "development" / "provenance_local_rag_v1.json"
PREDICTIONS = PROJECT_ROOT / "results" / "development" / "provenance_local_rag_v1_predictions.jsonl"
ARTIFACT = PROJECT_ROOT / "artifacts" / "provenance_local_rag_v1.joblib"
REPORT = PROJECT_ROOT / "reports" / "provenance_local_rag_v1.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_alignment(items: list[dict[str, Any]], paths: list[dict[str, Any]]) -> None:
    if [str(row["item_id"]) for row in items] != [str(row["item_id"]) for row in paths]:
        raise ValueError("item and graph-path order differ")
    if any(row.get("label_fields_present") is not False for row in paths):
        raise ValueError("graph-path cache lacks the no-label assertion")


def align_probability(model: Any, matrix: Any, positive: int = 1) -> np.ndarray:
    raw = model.predict_proba(matrix)
    classes = list(model.classes_)
    if positive not in classes:
        return np.full(matrix.shape[0], float(classes[0] == positive))
    return raw[:, classes.index(positive)]


def make_classifier(c: float) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=4000,
        random_state=SEED,
        solver="liblinear",
    )


def build_feature_blocks(
    train: list[dict[str, Any]],
    development: list[dict[str, Any]],
    train_paths: list[dict[str, Any]],
    development_paths: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    retriever = PassageLocalRetriever(top_k=top_k).fit(train)
    train_retrieved = [retriever.retrieve(item, path) for item, path in zip(train, train_paths)]
    development_retrieved = [
        retriever.retrieve(item, path) for item, path in zip(development, development_paths)
    ]
    omission_items = [omission_variant(item) for item in train if item["label"] == "SUPPORTED"]
    omission_paths = [path for item, path in zip(train, train_paths) if item["label"] == "SUPPORTED"]
    omission_retrieved = [
        retriever.retrieve(item, path) for item, path in zip(omission_items, omission_paths)
    ]

    selected_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=120_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    selected_train = selected_vectorizer.fit_transform([row.text for row in train_retrieved])
    selected_development = selected_vectorizer.transform([row.text for row in development_retrieved])
    selected_omission = selected_vectorizer.transform([row.text for row in omission_retrieved])

    full_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=120_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    full_train = full_vectorizer.fit_transform([structured_full_text(row) for row in train])
    full_development = full_vectorizer.transform(
        [structured_full_text(row) for row in development]
    )
    full_omission = full_vectorizer.transform(
        [structured_full_text(row) for row in omission_items]
    )

    graph_vectorizer = DictVectorizer(sparse=True)
    graph_train = graph_vectorizer.fit_transform(
        [graph_feature_dict(item, path) for item, path in zip(train, train_paths)]
    )
    graph_development = graph_vectorizer.transform(
        [graph_feature_dict(item, path) for item, path in zip(development, development_paths)]
    )
    graph_omission = graph_vectorizer.transform(
        [graph_feature_dict(item, path) for item, path in zip(omission_items, omission_paths)]
    )

    numeric_vectorizer = DictVectorizer(sparse=True)
    numeric_train = numeric_vectorizer.fit_transform(
        [row.feature_record for row in train_retrieved]
    )
    numeric_development = numeric_vectorizer.transform(
        [row.feature_record for row in development_retrieved]
    )
    numeric_omission = numeric_vectorizer.transform(
        [row.feature_record for row in omission_retrieved]
    )

    def combine(*parts: Any) -> csr_matrix:
        matrix = hstack(parts, format="csr", dtype=np.float32)
        matrix.indices = matrix.indices.astype(np.int32, copy=False)
        matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
        return matrix

    return {
        "retriever": retriever,
        "selected_vectorizer": selected_vectorizer,
        "full_vectorizer": full_vectorizer,
        "graph_vectorizer": graph_vectorizer,
        "numeric_vectorizer": numeric_vectorizer,
        "selected_train": combine(selected_train, graph_train, numeric_train),
        "selected_development": combine(
            selected_development, graph_development, numeric_development
        ),
        "selected_omission": combine(
            selected_omission, graph_omission, numeric_omission
        ),
        "full_train": combine(full_train, graph_train, numeric_train),
        "full_development": combine(full_development, graph_development, numeric_development),
        "full_omission": combine(full_omission, graph_omission, numeric_omission),
        "fusion_train": combine(selected_train, full_train, graph_train, numeric_train),
        "fusion_development": combine(
            selected_development, full_development, graph_development, numeric_development
        ),
        "fusion_omission": combine(
            selected_omission, full_omission, graph_omission, numeric_omission
        ),
        "retrieval_diagnostics": {
            "train_mean_selected_segments": float(
                np.mean([len(row.selected_indices) for row in train_retrieved])
            ),
            "development_mean_selected_segments": float(
                np.mean([len(row.selected_indices) for row in development_retrieved])
            ),
            "development_selected_both_rate": float(
                np.mean([row.feature_record["selected_has_both"] for row in development_retrieved])
            ),
        },
    }


def stage_probabilities(
    train_matrix: csr_matrix,
    development_matrix: csr_matrix,
    labels: np.ndarray,
    *,
    c: float,
    omission_matrix: csr_matrix | None,
    omission_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    entity_model = make_classifier(c)
    entity_model.fit(train_matrix, (labels == ENTITY_ERROR).astype(int))
    entity_probability = align_probability(entity_model, development_matrix)

    non_entity = np.where(labels != ENTITY_ERROR)[0]
    sufficiency_train = train_matrix[non_entity]
    sufficiency_target = (labels[non_entity] == INSUFFICIENT).astype(int)
    sample_weight = np.ones(len(non_entity), dtype=np.float64)
    if omission_matrix is not None:
        sufficiency_train = vstack([sufficiency_train, omission_matrix], format="csr")
        sufficiency_target = np.concatenate(
            [sufficiency_target, np.ones(omission_matrix.shape[0], dtype=np.int64)]
        )
        sample_weight = np.concatenate(
            [sample_weight, np.full(omission_matrix.shape[0], omission_weight)]
        )
    sufficiency_model = make_classifier(c)
    sufficiency_model.fit(sufficiency_train, sufficiency_target, sample_weight=sample_weight)
    insufficient_probability = align_probability(sufficiency_model, development_matrix)

    judgeable = np.where(np.isin(labels, [SUPPORTED, NOT_SUPPORTED]))[0]
    verdict_model = make_classifier(c)
    verdict_model.fit(train_matrix[judgeable], (labels[judgeable] == SUPPORTED).astype(int))
    supported_probability = align_probability(verdict_model, development_matrix)
    probabilities = np.column_stack(
        [supported_probability, insufficient_probability, entity_probability]
    )
    return probabilities, {
        "entity": entity_model,
        "sufficiency": sufficiency_model,
        "verdict": verdict_model,
    }


def threshold_grid() -> list[dict[str, float]]:
    return [
        {"entity": entity, "insufficient": insufficient, "supported": supported}
        for entity, insufficient, supported in product(
            np.arange(0.25, 0.76, 0.05),
            np.arange(0.20, 0.76, 0.05),
            np.arange(0.35, 0.66, 0.05),
        )
    ]


def decide(probabilities: np.ndarray, thresholds: dict[str, float]) -> list[str]:
    output: list[str] = []
    for supported, insufficient, entity in probabilities:
        if entity >= thresholds["entity"]:
            output.append(LABELS[ENTITY_ERROR])
        elif insufficient >= thresholds["insufficient"]:
            output.append(LABELS[INSUFFICIENT])
        elif supported >= thresholds["supported"]:
            output.append(LABELS[SUPPORTED])
        else:
            output.append(LABELS[NOT_SUPPORTED])
    return output


def select_thresholds(gold: list[str], probabilities: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for thresholds in threshold_grid():
        predicted = decide(probabilities, thresholds)
        metrics = full_relation_verification_metrics(gold, predicted)
        safe = (
            metrics["abstain_recall"] >= 0.70
            and metrics["gold_abstain_false_present_rate"] <= 0.08
        )
        key = (
            safe,
            metrics["four_class"]["macro_f1"],
            metrics["three_decision"]["macro_f1"],
            metrics["four_class"]["accuracy"],
            -metrics["gold_abstain_false_present_rate"],
        )
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "thresholds": {name: float(value) for name, value in thresholds.items()},
                "metrics": metrics,
                "predicted": predicted,
            }
    if best is None:
        raise RuntimeError("threshold selection failed")
    return best


def main() -> None:
    train = read_jsonl(DATA_DIR / "train.jsonl")
    development = read_jsonl(DATA_DIR / "development.jsonl")
    train_paths = read_jsonl(PATH_DIR / "train.jsonl")
    development_paths = read_jsonl(PATH_DIR / "development.jsonl")
    validate_alignment(train, train_paths)
    validate_alignment(development, development_paths)
    train_sources = {str(row["book"]) for row in train}
    development_sources = {str(row["book"]) for row in development}
    if train_sources & development_sources:
        raise ValueError("train and development sources overlap")
    labels = np.asarray([LABELS.index(str(row["label"])) for row in train], dtype=np.int64)
    gold = [str(row["label"]) for row in development]

    experiments: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    for top_k in (2, 3, 5):
        blocks = build_feature_blocks(
            train, development, train_paths, development_paths, top_k=top_k
        )
        for modality, c, augmentation, omission_weight in product(
            ("selected", "full", "fusion"),
            (0.25, 1.0, 4.0),
            (False, True),
            (0.35, 0.65),
        ):
            if not augmentation and omission_weight != 0.35:
                continue
            probabilities, models = stage_probabilities(
                blocks[f"{modality}_train"],
                blocks[f"{modality}_development"],
                labels,
                c=c,
                omission_matrix=blocks[f"{modality}_controlled_omission"] if augmentation else None,
                omission_weight=omission_weight,
            )
            selected = select_thresholds(gold, probabilities)
            name = (
                f"local_rag_top{top_k}_{modality}_c{c:g}_"
                f"omit{omission_weight:g}" if augmentation else f"local_rag_top{top_k}_{modality}_c{c:g}_noomit"
            )
            record = {
                "name": name,
                "top_k": top_k,
                "modality": modality,
                "C": c,
                "omission_augmentation": augmentation,
                "omission_weight": omission_weight if augmentation else 0.0,
                "thresholds": selected["thresholds"],
                "metrics": selected["metrics"],
                "retrieval_diagnostics": blocks["retrieval_diagnostics"],
            }
            experiments.append(record)
            artifacts[name] = {
                "models": models,
                "blocks": {
                    key: value
                    for key, value in blocks.items()
                    if key.endswith("vectorizer") or key == "retriever"
                },
                "thresholds": selected["thresholds"],
                "modality": modality,
                "top_k": top_k,
                "predicted": selected["predicted"],
                "probabilities": probabilities,
            }
            print(
                name,
                f"4F1={selected['metrics']['four_class']['macro_f1']:.6f}",
                f"3F1={selected['metrics']['three_decision']['macro_f1']:.6f}",
                f"AR={selected['metrics']['abstain_recall']:.6f}",
                flush=True,
            )

    experiments.sort(
        key=lambda row: (
            row["metrics"]["abstain_recall"] >= 0.70
            and row["metrics"]["gold_abstain_false_present_rate"] <= 0.08,
            row["metrics"]["four_class"]["macro_f1"],
            row["metrics"]["three_decision"]["macro_f1"],
        ),
        reverse=True,
    )
    best = experiments[0]
    best_artifact = artifacts[best["name"]]
    payload = {
        "status": "DEVELOPMENT_ONLY_R9_GOLD_NOT_ACCESSED",
        "method": "Provenance-Constrained Passage-Local Evidence-Sufficiency Graph RAG v1",
        "train_items": len(train),
        "development_items": len(development),
        "train_sources": len(train_sources),
        "development_sources": len(development_sources),
        "source_overlap": 0,
        "selected": best,
        "experiments": experiments,
        "input_hashes": {
            "train": sha256(DATA_DIR / "train.jsonl"),
            "development": sha256(DATA_DIR / "development.jsonl"),
            "train_graph_paths": sha256(PATH_DIR / "train.jsonl"),
            "development_graph_paths": sha256(PATH_DIR / "development.jsonl"),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            key: value
            for key, value in best_artifact.items()
            if key not in {"predicted", "probabilities"}
        },
        ARTIFACT,
    )
    with PREDICTIONS.open("w", encoding="utf-8") as handle:
        for row, predicted, probability in zip(
            development, best_artifact["predicted"], best_artifact["probabilities"]
        ):
            handle.write(
                json.dumps(
                    {
                        "item_id": row["item_id"],
                        "gold_label": row["label"],
                        "predicted_label": predicted,
                        "stage_probabilities": {
                            "supported_given_judgeable": float(probability[0]),
                            "insufficient_context": float(probability[1]),
                            "entity_or_type_error": float(probability[2]),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    table = "\n".join(
        "| {name} | {f4:.4f} | {f3:.4f} | {acc:.4f} | {ar:.4f} | {fp:.4f} |".format(
            name=row["name"],
            f4=row["metrics"]["four_class"]["macro_f1"],
            f3=row["metrics"]["three_decision"]["macro_f1"],
            acc=row["metrics"]["four_class"]["accuracy"],
            ar=row["metrics"]["abstain_recall"],
            fp=row["metrics"]["gold_abstain_false_present_rate"],
        )
        for row in experiments[:20]
    )
    REPORT.write_text(
        "# Provenance-constrained passage-local RAG v1\n\n"
        "Development-only experiment. The R9 gold labels were not accessed. Retrieval is "
        "restricted to evidence units inside each frozen passage and target-edge-deleted graph paths.\n\n"
        "| Configuration | 4-class Macro-F1 | 3-decision Macro-F1 | Accuracy | Abstain recall | False PRESENT |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |\n"
        f"{table}\n\nSelected: **{best['name']}**.\n",
        encoding="utf-8",
    )
    print("SELECTED", best["name"])


if __name__ == "__main__":
    main()
