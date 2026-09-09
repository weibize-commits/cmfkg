from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import fusion_features as sg  # noqa: E402


OUTPUT = PROJECT_ROOT / "results" / "development" / "sg_fera_extratrees_v2.json"
GRID_OUTPUT = (
    PROJECT_ROOT / "results" / "development" / "sg_fera_extratrees_grid_v2.json"
)
REPORT = PROJECT_ROOT / "reports" / "sg_fera_extratrees_v2.md"
ARTIFACT = PROJECT_ROOT / "artifacts" / "sg_fera_extratrees_v2.joblib"
CONFIG = PROJECT_ROOT / "configs" / "models" / "sg_fera_extratrees_v2.json"
SEEDS = [20261001, 20261002, 20261003, 20261004, 20261005]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model(
    frame: Any,
    *,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: str | float,
    seed: int,
    n_estimators: int,
) -> Pipeline:
    numeric = [column for column in frame.columns if is_numeric_dtype(frame[column])]
    categorical = [column for column in frame.columns if column not in numeric]
    processor = ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )
    classifier = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("features", processor), ("classifier", classifier)])


def aligned_probabilities(model: Pipeline, frame: Any) -> np.ndarray:
    raw = model.predict_proba(frame)
    return sg.align_probabilities(
        raw, model.named_steps["classifier"].classes_, sg.LABELS
    )


def compact_grid_row(
    parameters: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, Any]:
    objective = (
        0.55 * metrics["three_decision_macro_f1"]
        + 0.45 * metrics["four_class_macro_f1"]
    )
    eligible = (
        metrics["abstain_recall"] >= 0.65
        and metrics["gold_abstain_false_present_rate"] <= 0.10
    )
    return {
        "parameters": parameters,
        "selection_objective": objective,
        "passes_development_safety_constraints": eligible,
        "metrics": metrics,
    }


def difference(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "four_class_macro_f1": candidate["four_class_macro_f1"]
        - baseline["four_class_macro_f1"],
        "three_decision_macro_f1": candidate["three_decision_macro_f1"]
        - baseline["three_decision_macro_f1"],
        "abstain_recall": candidate["abstain_recall"] - baseline["abstain_recall"],
        "gold_abstain_false_present_rate": candidate[
            "gold_abstain_false_present_rate"
        ]
        - baseline["gold_abstain_false_present_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the nonlinear SG-FERA OOF stacker and freeze it for prior stage."
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261021)
    parser.add_argument("--grid-estimators", type=int, default=400)
    parser.add_argument("--final-estimators", type=int, default=500)
    args = parser.parse_args()

    train_rows, train_paths = sg.load_rows("train")
    development_rows, development_paths = sg.load_rows("development")
    train_maps = {task: sg.oof_probabilities(task) for task in sg.TASKS}
    development_maps = {
        task: sg.full_development_probabilities(task) for task in sg.TASKS
    }
    train_frame = sg.feature_frame(train_rows, train_paths, train_maps)
    development_frame = sg.feature_frame(
        development_rows, development_paths, development_maps
    )
    train_labels = [str(row["label"]) for row in train_rows]

    grid: list[dict[str, Any]] = []
    for max_depth in [8, 12, 16, None]:
        for min_samples_leaf in [2, 3, 5, 8]:
            for max_features in ["sqrt", 0.5]:
                parameters = {
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                    "n_estimators": args.grid_estimators,
                    "random_state": SEEDS[0],
                }
                model = make_model(
                    train_frame,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    seed=SEEDS[0],
                    n_estimators=args.grid_estimators,
                )
                model.fit(train_frame, train_labels)
                probabilities = aligned_probabilities(model, development_frame)
                predictions = sg.guarded_predictions(development_rows, probabilities)
                grid.append(
                    compact_grid_row(parameters, sg.evaluate(development_rows, predictions))
                )

    eligible = [
        row for row in grid if row["passes_development_safety_constraints"]
    ]
    selection_pool = eligible if eligible else grid
    selected_grid = max(
        selection_pool,
        key=lambda row: (
            row["selection_objective"],
            -row["metrics"]["gold_abstain_false_present_rate"],
            row["metrics"]["abstain_recall"],
            -row["parameters"]["min_samples_leaf"],
        ),
    )
    selected = dict(selected_grid["parameters"])
    selected["n_estimators"] = args.final_estimators

    models: list[Pipeline] = []
    seed_results: list[dict[str, Any]] = []
    seed_probabilities: list[np.ndarray] = []
    for seed in SEEDS:
        model = make_model(
            train_frame,
            max_depth=selected["max_depth"],
            min_samples_leaf=selected["min_samples_leaf"],
            max_features=selected["max_features"],
            seed=seed,
            n_estimators=args.final_estimators,
        )
        model.fit(train_frame, train_labels)
        probabilities = aligned_probabilities(model, development_frame)
        predictions = sg.guarded_predictions(development_rows, probabilities)
        seed_results.append({"seed": seed, "metrics": sg.evaluate(development_rows, predictions)})
        seed_probabilities.append(probabilities)
        models.append(model)

    ensemble_probabilities = np.mean(np.stack(seed_probabilities), axis=0)
    ensemble_predictions = sg.guarded_predictions(
        development_rows, ensemble_probabilities
    )
    ensemble_metrics = sg.evaluate(development_rows, ensemble_predictions)

    expert_values = sg.expert_matrix(development_rows, development_maps)
    fixed_dual_values = sg.fixed_dual_probabilities(development_rows, development_maps)
    baseline_probabilities = {
        task: expert_values[:, index, :] for index, task in enumerate(sg.TASKS)
    }
    baseline_probabilities["uniform_expert_mean"] = expert_values.mean(axis=1)
    baseline_probabilities["fixed_dual_evidence"] = fixed_dual_values
    baseline_predictions = {
        name: sg.guarded_predictions(development_rows, probabilities)
        for name, probabilities in baseline_probabilities.items()
    }
    baseline_metrics = {
        name: sg.evaluate(development_rows, predictions)
        for name, predictions in baseline_predictions.items()
    }

    comparison_names = [
        "graph_anchor_wide",
        "fera_ecrp_sourcebal",
        "fixed_dual_evidence",
    ]
    comparisons = {}
    for index, name in enumerate(comparison_names):
        comparisons[name] = {
            "point_difference": difference(ensemble_metrics, baseline_metrics[name]),
            "passage_clustered_bootstrap": sg.clustered_bootstrap(
                development_rows,
                baseline_predictions[name],
                ensemble_predictions,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + index,
            ),
        }

    input_paths = []
    for task in sg.TASKS:
        input_paths.append(sg.OOF_ROOT / f"{task}_oof_predictions.jsonl")
        input_paths.extend(
            sorted(
                sg.task_root(task).glob(
                    "results/deep_development/*/best_development_predictions.jsonl"
                )
            )
        )
    input_files = [
        {
            "path": Path(os.path.relpath(path, PROJECT_ROOT)).as_posix(),
            "sha256": sha256(path),
        }
        for path in input_paths
    ]

    predictions = []
    for row, prediction, probabilities in zip(
        development_rows, ensemble_predictions, ensemble_probabilities.tolist()
    ):
        predictions.append(
            {
                "item_id": str(row["item_id"]),
                "passage_id": str(row["passage_id"]),
                "gold_label": str(row["label"]),
                "predicted_label": prediction,
                "probabilities": {
                    label: float(value) for label, value in zip(sg.LABELS, probabilities)
                },
            }
        )

    configuration = {
        "method_name": "SG-FERA nonlinear OOF stacker v2",
        "training_predictions": "passage-disjoint out-of-fold predictions only",
        "base_experts": sg.TASKS,
        "features": list(train_frame.columns),
        "classifier": "sklearn.ensemble.ExtraTreesClassifier",
        "selected_hyperparameters": selected,
        "ensemble_seeds": SEEDS,
        "schema_guard": True,
        "selection_split": "development",
        "selection_constraints": {
            "abstain_recall_at_least": 0.65,
            "gold_abstain_false_present_rate_at_most": 0.10,
        },
        "selection_objective": "0.55 * three-decision Macro-F1 + 0.45 * four-class Macro-F1",
        "prior_stage_labels_accessed": False,
    }
    output = {
        "status": "DEVELOPMENT_MODEL_SELECTION_FROZEN_BEFORE_PRIOR_STAGE_LABEL_ACCESS",
        "configuration": configuration,
        "training": {
            "items": len(train_rows),
            "passages": len({str(row["passage_id"]) for row in train_rows}),
        },
        "development": {
            "items": len(development_rows),
            "passages": len(
                {str(row["passage_id"]) for row in development_rows}
            ),
            "selected_grid_row": selected_grid,
            "seed_results": seed_results,
            "ensemble_metrics": ensemble_metrics,
            "baseline_metrics": baseline_metrics,
            "comparisons": comparisons,
        },
        "predictions": predictions,
        "input_files": input_files,
        "prior_stage_labels_accessed": False,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    GRID_OUTPUT.write_text(
        json.dumps(
            {
                "selection_protocol": configuration,
                "trials": grid,
                "selected": selected_grid,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    CONFIG.write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    joblib.dump(
        {
            "configuration": configuration,
            "models": models,
            "labels": sg.LABELS,
            "feature_columns": list(train_frame.columns),
        },
        ARTIFACT,
    )

    report_lines = [
        "# SG-FERA nonlinear OOF stacker v2",
        "",
        "The nonlinear stacker was trained only on passage-disjoint out-of-fold base-model predictions. The development grid, safety-constrained selection rule and all five final random seeds are retained. prior stage labels were not accessed.",
        "",
        f"- Selected hyperparameters: `{json.dumps(selected, ensure_ascii=False, sort_keys=True)}`",
        f"- Development four-class Macro-F1: {ensemble_metrics['four_class_macro_f1']:.6f}",
        f"- Development three-decision Macro-F1: {ensemble_metrics['three_decision_macro_f1']:.6f}",
        f"- Development abstain recall: {ensemble_metrics['abstain_recall']:.6f}",
        f"- Development false-present rate on gold-abstain: {ensemble_metrics['gold_abstain_false_present_rate']:.6f}",
        "",
    ]
    for name in comparison_names:
        comparison = comparisons[name]
        four_ci = comparison["passage_clustered_bootstrap"][
            "four_class_macro_f1_difference"
        ]["ci95_percentile"]
        three_ci = comparison["passage_clustered_bootstrap"][
            "three_decision_macro_f1_difference"
        ]["ci95_percentile"]
        report_lines.extend(
            [
                f"## Versus {name}",
                "",
                f"- Four-class Macro-F1 difference: {comparison['point_difference']['four_class_macro_f1']:+.6f}; 95% clustered-bootstrap CI [{four_ci[0]:+.6f}, {four_ci[1]:+.6f}]",
                f"- Three-decision Macro-F1 difference: {comparison['point_difference']['three_decision_macro_f1']:+.6f}; 95% clustered-bootstrap CI [{three_ci[0]:+.6f}, {three_ci[1]:+.6f}]",
                f"- Abstain-recall difference: {comparison['point_difference']['abstain_recall']:+.6f}",
                f"- False-present-rate difference: {comparison['point_difference']['gold_abstain_false_present_rate']:+.6f}",
                "",
            ]
        )
    report_lines.extend(
        [
            f"- Result SHA-256: `{sha256(OUTPUT)}`",
            f"- Grid SHA-256: `{sha256(GRID_OUTPUT)}`",
            f"- Configuration SHA-256: `{sha256(CONFIG)}`",
            f"- Artifact SHA-256: `{sha256(ARTIFACT)}`",
            "",
            "These are development results. The independent locked prior stage set remains the sole confirmatory test.",
            "",
        ]
    )
    REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    compact = {
        "selected_hyperparameters": selected,
        "ensemble_metrics": ensemble_metrics,
        "comparisons": comparisons,
        "output": str(OUTPUT),
        "artifact": str(ARTIFACT),
    }
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
