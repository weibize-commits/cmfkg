from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_CONTEXT", "ENTITY_OR_TYPE_ERROR"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average five seeds within every source-work-held-out cross-fit fold."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/deep_development")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260921, 20260922, 20260923, 20260924, 20260925])
    args = parser.parse_args()

    assignments = {
        str(row["item_id"]): int(row["fold"])
        for row in read_jsonl(args.fold_root / "fold_assignments.jsonl")
    }
    merged: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold in range(args.folds):
        expected = {item_id for item_id, assigned in assignments.items() if assigned == fold}
        by_seed: dict[str, dict[str, dict[str, Any]]] = {}
        for seed in args.seeds:
            run_name = f"{args.run_prefix}-f{fold}-s{seed}"
            run_dir = args.results_root / run_name
            prediction_path = run_dir / "best_development_predictions.jsonl"
            rows = read_jsonl(prediction_path)
            indexed = {str(row["item_id"]): row for row in rows}
            if set(indexed) != expected:
                raise ValueError(
                    f"fold {fold} seed {seed} mismatch: missing={len(expected-set(indexed))} "
                    f"extra={len(set(indexed)-expected)}"
                )
            by_seed[str(seed)] = indexed
            source_files.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "run_name": run_name,
                    "predictions_sha256": sha256(prediction_path),
                    "metrics_sha256": sha256(run_dir / "best_development_metrics.json"),
                    "config_sha256": sha256(run_dir / "run_config.json"),
                }
            )
        for item_id in sorted(expected):
            if item_id in seen:
                raise ValueError(f"duplicate OOF item: {item_id}")
            seen.add(item_id)
            vectors = np.asarray(
                [
                    [float(by_seed[str(seed)][item_id]["probabilities"][label]) for label in LABELS]
                    for seed in args.seeds
                ],
                dtype=float,
            )
            mean = vectors.mean(axis=0)
            template = by_seed[str(args.seeds[0])][item_id]
            merged.append(
                {
                    "task": args.task,
                    "fold": fold,
                    "item_id": item_id,
                    "passage_id": template["passage_id"],
                    "source_name": template["source_name"],
                    "relation_group": template["relation_group"],
                    "gold_label": template["gold_label"],
                    "predicted_label": LABELS[int(np.argmax(mean))],
                    "probabilities": {label: float(mean[index]) for index, label in enumerate(LABELS)},
                    "probabilities_by_seed": {
                        str(seed): {
                            label: float(vectors[seed_index, label_index])
                            for label_index, label in enumerate(LABELS)
                        }
                        for seed_index, seed in enumerate(args.seeds)
                    },
                }
            )
    if seen != set(assignments):
        raise ValueError("source-work OOF coverage is incomplete")
    merged.sort(key=lambda row: row["item_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged), encoding="utf-8"
    )
    manifest = {
        "status": "SOURCE_WORK_DISJOINT_FIVE_SEED_OOF_PREDICTIONS",
        "task": args.task,
        "items": len(merged),
        "folds": args.folds,
        "seeds_per_fold": args.seeds,
        "grouping_unit": "source_work",
        "items_covered_exactly_once": True,
        "fold_assignment_sha256": sha256(args.fold_root / "fold_assignments.jsonl"),
        "output_sha256": sha256(args.output),
        "source_files": source_files,
        "development_and_test_items_used": False,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "items": len(merged)}))


if __name__ == "__main__":
    main()
