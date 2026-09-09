from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("FERA_KG_DATA_ROOT", ROOT / "data"))
SEEDS = [20260921, 20260922, 20260923, 20260924, 20260925]
FAMILIES = {
    "graph_anchor_wide": "final-graph_anchor_wide-seed-",
    "fera_ecrp": "final-fera_ecrp-seed-",
    "fera_ecrp_sourcebal": "final-fera_ecrp_sourcebal-seed-",
    "text_entity": "final-text_entity-seed-",
    "same_backbone_no_ecrp": "final-same_backbone_no_ecrp-seed-",
}
DATASETS = {
    "controlled": "inference/controlled",
    "natural_registered": "inference/natural/registered",
    "natural_shuffled": "inference/natural/shuffled",
    "natural_empty": "inference/natural/empty",
    "additional_stress": "inference/additional_stress",
}


def run(arguments: list[str], *, environment: dict[str, str] | None = None) -> None:
    subprocess.run(arguments, cwd=ROOT, env=environment, check=True)


def evaluate_and_render(prediction_root: Path | None) -> None:
    environment = os.environ.copy()
    if prediction_root is not None:
        environment["FERA_KG_PREDICTION_ROOT"] = str(prediction_root.resolve())
    run([sys.executable, "scripts/evaluate_final.py"], environment=environment)
    evaluation = ROOT / "outputs" / "final_evaluation.json"
    print(f"Frozen evaluation written to {evaluation}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the portable final FERA-KG pipeline.")
    parser.add_argument(
        "--mode",
        choices=("frozen-evaluation", "inference-and-evaluation"),
        default="frozen-evaluation",
    )
    parser.add_argument("--deep-results-root", type=Path, default=ROOT / "results" / "deep_development")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "final_fusion.joblib")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "predictions")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Companion confidential data directory is missing: {DATA_ROOT}")
    if args.mode == "frozen-evaluation":
        evaluate_and_render(None)
        return
    if not args.artifact.is_file():
        raise FileNotFoundError(
            f"Fusion artifact is missing: {args.artifact}. Run scripts/fit_fusion_ablations.py first."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    family_arguments: list[str] = []
    for family, prefix in FAMILIES.items():
        family_arguments.extend(("--family", f"{family}={prefix}"))
    seed_arguments = ["--seeds", *[str(seed) for seed in SEEDS]]
    for name, relative in DATASETS.items():
        dataset_root = DATA_ROOT / relative
        expert = args.output_root / f"{name}_components.json"
        fusion = args.output_root / f"{name}.json"
        run(
            [
                sys.executable,
                "scripts/predict_component_ensembles.py",
                "--dataset",
                str(dataset_root / "items.jsonl"),
                "--graph-paths",
                str(dataset_root / "target_edge_deleted_paths.jsonl"),
                "--deep-results-root",
                str(args.deep_results_root),
                *family_arguments,
                *seed_arguments,
                "--batch-size",
                str(args.batch_size),
                "--output",
                str(expert),
            ]
        )
        run(
            [
                sys.executable,
                "scripts/predict_final_fusion.py",
                "--dataset",
                str(dataset_root / "items.jsonl"),
                "--graph-paths",
                str(dataset_root / "target_edge_deleted_paths.jsonl"),
                "--expert-payload",
                str(expert),
                "--artifact",
                str(args.artifact),
                "--output",
                str(fusion),
            ]
        )
    evaluate_and_render(args.output_root)


if __name__ == "__main__":
    main()
