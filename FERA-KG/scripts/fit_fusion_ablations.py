from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import local_evidence_pipeline as v3  # noqa: E402
import fit_grouped_fusion as source_refit  # noqa: E402
import fusion_models as et  # noqa: E402
import fusion_features as sg  # noqa: E402


ARTIFACT = ROOT / "artifacts" / "final_fusion.joblib"
MANIFEST = ROOT / "artifacts" / "final_fusion_manifest.json"
REPORT = ROOT / "results" / "fusion_ablations.json"
SEEDS = [20261101, 20261102, 20261103, 20261104, 20261105]
THRESHOLDS = {"entity": 0.30, "insufficient": 0.40, "supported": 0.50}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keep_without(columns: list[str], predicates: list[Any]) -> list[str]:
    return [name for name in columns if not any(predicate(name) for predicate in predicates)]


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    ecrp = lambda name: "fera_ecrp" in name
    text = lambda name: "text_entity" in name
    local = lambda name: name.startswith("local_top") or name.startswith("local_rag")
    graph_raw_names = {
        "path_count",
        "passage_conditioned_path_count",
        "semantic_only_path_count",
        "actual_typed_mentions",
        "head_in_actual_inventory",
        "tail_in_actual_inventory",
        "enumeration_truncated",
    }
    graph_expert = lambda name: (
        name.startswith("graph_anchor_wide__")
        or "fera_ecrp" in name
        or (name.startswith("l1__") and (
            "graph_anchor_wide" in name or "fera_ecrp" in name
        ))
        or name in graph_raw_names
    )
    raw = set(columns[:24])
    expert_only = [name for name in columns if 24 <= columns.index(name) <= 62]
    variants = {
        "full": list(columns),
        "without_ecrp_experts": keep_without(columns, [ecrp]),
        "without_text_expert": keep_without(columns, [text]),
        "without_passage_local_evidence": keep_without(columns, [local]),
        "without_raw_construction_and_surface_features": [
            name for name in columns if name not in raw
        ],
        "without_graph_experts_and_explicit_graph_diagnostics": keep_without(
            columns, [graph_expert]
        ),
        "deep_expert_outputs_only": expert_only,
    }
    for name, values in variants.items():
        if not values:
            raise ValueError(f"Feature ablation {name} is empty")
        if len(values) != len(set(values)):
            raise ValueError(f"Feature ablation {name} contains duplicate columns")
    return variants


def fit_ensemble(frame: Any, labels: list[str], columns: list[str]) -> list[Any]:
    selected = frame[columns]
    models = []
    for seed in SEEDS:
        print(
            f"FUSION_FIT_START seed={seed} features={len(columns)}",
            flush=True,
        )
        model = et.make_model(
            selected,
            max_depth=8,
            min_samples_leaf=2,
            max_features=0.5,
            seed=seed,
            n_estimators=600,
        )
        model.fit(selected, labels)
        models.append(model)
        print(f"FUSION_FIT_COMPLETE seed={seed}", flush=True)
    return models


def main() -> None:
    train, train_path_map = sg.load_rows("train")
    development, development_path_map = sg.load_rows("development")
    train_ids = {str(row["item_id"]) for row in train}
    source_names = {str(row.get("source_name", "")).strip() for row in train}
    if "" in source_names:
        raise ValueError("Fusion-training data contain a missing source_name grouping field")
    assignment_rows = source_refit.read_jsonl(
        source_refit.SOURCE_FOLD_ROOT / "fold_assignments.jsonl"
    )
    assignments = {str(row["item_id"]): int(row["fold"]) for row in assignment_rows}
    if len(assignments) != len(assignment_rows) or set(assignments) != train_ids:
        raise ValueError("Canonical source-work fold assignments do not match training items")
    source_folds: dict[str, set[int]] = {}
    for row in train:
        source = str(row["source_name"]).strip()
        source_folds.setdefault(source, set()).add(assignments[str(row["item_id"])])
    if any(len(folds) != 1 for folds in source_folds.values()):
        raise ValueError("A training source work occurs in more than one OOF fold")
    expert_maps = {
        task: source_refit.source_oof_probabilities(task, train_ids)
        for task in sg.TASKS
    }
    train_paths = [train_path_map[str(row["item_id"])] for row in train]
    development_paths = [
        development_path_map[str(row["item_id"])] for row in development
    ]

    train_frame = sg.feature_frame(train, train_path_map, expert_maps)
    # These retrieval features contain no labels. The local-evidence probability
    # block below is separately generated out of fold by source work.
    development_stub = sg.feature_frame(
        development,
        development_path_map,
        {
            task: {
                str(row["item_id"]): [0.25, 0.25, 0.25, 0.25]
                for row in development
            }
            for task in sg.TASKS
        },
    )
    v3.add_retrieval_features(
        train,
        development,
        train_paths,
        development_paths,
        train_frame,
        development_stub,
    )
    local_oof = source_refit.source_work_oof_rag_probabilities(train, train_paths)
    _, rag_artifact = v3.full_rag_probabilities(
        train,
        development,
        train_paths,
        development_paths,
    )
    for name, values in v3.rag_probability_columns(local_oof).items():
        train_frame[name] = values

    columns = list(train_frame.columns)
    if len(columns) != 130:
        raise ValueError(f"Expected 130 fusion features, observed {len(columns)}")
    variants = feature_sets(columns)
    labels = [str(row["label"]) for row in train]
    models_by_variant: dict[str, list[Any]] = {}
    for variant, selected in variants.items():
        print(
            f"ABLATION_START variant={variant} features={len(selected)}",
            flush=True,
        )
        models_by_variant[variant] = fit_ensemble(train_frame, labels, selected)
        print(f"ABLATION_COMPLETE variant={variant}", flush=True)

    artifact = {
        # Backward-compatible keys used by the unified predictor.
        "models": models_by_variant["full"],
        "features": variants["full"],
        "rag": rag_artifact,
        "thresholds": THRESHOLDS,
        "labels": list(sg.LABELS),
        # Explicit retraining-ablation inventory.
        "models_by_variant": models_by_variant,
        "features_by_variant": variants,
        "source_work_oof": True,
        "oof_deep_seed_count": 5,
        "oof_local_evidence_grouping": "source_work",
        "training_items": len(train),
        "training_source_works": len(source_names),
        "source_work_field": "source_name",
        "source_work_crossfit_verified": True,
        "fixed_hyperparameters": {
            "trees_per_seed": 600,
            "max_depth": 8,
            "min_samples_leaf": 2,
            "max_features": 0.5,
            "class_weight": "balanced",
            "seeds": SEEDS,
        },
        "software": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "interpretation": (
            "Each fusion ablation is refitted from scratch under a fixed feature mask. "
            "The deep encoders are held fixed; matched deep-module retraining is reported separately."
        ),
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT)
    manifest = {
        "status": "UNIFIED_SOURCE_WORK_OOF_FUSION_AND_RETRAINING_ABLATIONS_COMPLETE",
        "artifact": str(ARTIFACT),
        "artifact_sha256": sha256(ARTIFACT),
        "feature_count": len(columns),
        "variant_feature_counts": {
            name: len(values) for name, values in variants.items()
        },
        "variant_features": variants,
        "training_items": len(train),
        "training_source_works": len(source_names),
        "source_work_field": "source_name",
        "source_work_crossfit_verified": True,
        "deep_oof_grouping": "source_work",
        "local_evidence_oof_grouping": "source_work",
        "seeds": SEEDS,
        "thresholds": THRESHOLDS,
        "software": artifact["software"],
        "interpretation": artifact["interpretation"],
        "source_oof_sha256": {
            task: sha256(
                source_refit.SOURCE_OOF_ROOT / f"{task}_oof_predictions.jsonl"
            )
            for task in sg.TASKS
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    REPORT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
