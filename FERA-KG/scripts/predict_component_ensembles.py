from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ensemble_calibration import load_run_probabilities  # noqa: E402
from train_model import select_device  # noqa: E402


LABELS = ["SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_CONTEXT", "ENTITY_OR_TYPE_ERROR"]
MAX_FLOAT16_SUM_DEVIATION = 2.0e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed multi-seed expert ensembles on registered evaluation features."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--graph-paths", type=Path, required=True)
    parser.add_argument("--deep-results-root", type=Path, required=True)
    parser.add_argument(
        "--family",
        action="append",
        required=True,
        help="Comparator mapping in family=run-prefix form; repeat for each family.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20260829, 20260830, 20260831, 20260832, 20260833],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--expected-seed-count", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean_vectors(rows: list[list[list[float]]]) -> list[list[float]]:
    if not rows:
        raise ValueError("At least one seed-level probability matrix is required")
    item_count = len(rows[0])
    for seed_index, seed_rows in enumerate(rows):
        if len(seed_rows) != item_count:
            raise ValueError(
                f"Seed probability matrices are not aligned: seed index {seed_index} "
                f"has {len(seed_rows)} rows, expected {item_count}"
            )
        for item_index, vector in enumerate(seed_rows):
            values = [float(value) for value in vector]
            if len(values) != len(LABELS):
                raise ValueError(
                    f"Seed {seed_index}, item {item_index}: expected {len(LABELS)} probabilities"
                )
            if any(not math.isfinite(value) for value in values):
                raise ValueError(
                    f"Seed {seed_index}, item {item_index}: non-finite probability"
                )
            if any(value < -1e-8 or value > 1.0 + 1e-8 for value in values):
                raise ValueError(
                    f"Seed {seed_index}, item {item_index}: probability outside [0, 1]"
                )
            if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=2e-5):
                raise ValueError(
                    f"Seed {seed_index}, item {item_index}: probabilities do not sum to one"
                )
    return [
        [
            sum(seed_rows[item_index][label_index] for seed_rows in rows) / len(rows)
            for label_index in range(len(LABELS))
        ]
        for item_index in range(len(rows[0]))
    ]


def renormalize_probability_matrices(rows: list[list[list[float]]]) -> float:
    """Remove harmless float16 softmax rounding while rejecting malformed output."""
    maximum_deviation = 0.0
    for seed_index, seed_rows in enumerate(rows):
        for item_index, vector in enumerate(seed_rows):
            values = [float(value) for value in vector]
            if len(values) != len(LABELS) or any(not math.isfinite(value) for value in values):
                raise ValueError(
                    f"Seed {seed_index}, item {item_index}: malformed probability vector"
                )
            if any(value < -1e-8 or value > 1.0 + 1e-8 for value in values):
                raise ValueError(
                    f"Seed {seed_index}, item {item_index}: probability outside [0, 1]"
                )
            total = sum(values)
            deviation = abs(total - 1.0)
            maximum_deviation = max(maximum_deviation, deviation)
            if total <= 0.0 or deviation > MAX_FLOAT16_SUM_DEVIATION:
                raise ValueError(
                    f"Seed {seed_index}, item {item_index}: probability sum {total:.8f} "
                    f"exceeds the float16 normalization tolerance"
                )
            seed_rows[item_index] = [value / total for value in values]
    return maximum_deviation


def parse_families(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --family value: {value}")
        family, prefix = value.split("=", 1)
        family, prefix = family.strip(), prefix.strip()
        if not family or not prefix or family in result:
            raise ValueError(f"invalid or duplicate --family value: {value}")
        result[family] = prefix
    return result


def main() -> None:
    args = parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Ensemble seeds must be non-empty and unique")
    if args.expected_seed_count is not None and len(args.seeds) != args.expected_seed_count:
        raise ValueError(
            f"Expected {args.expected_seed_count} ensemble seeds, got {len(args.seeds)}"
        )
    families = parse_families(args.family)
    items = read_jsonl(args.dataset)
    paths = read_jsonl(args.graph_paths)
    if [str(row["item_id"]) for row in items] != [str(row["item_id"]) for row in paths]:
        raise ValueError("confirmatory dataset and graph-path rows are not aligned")
    device = select_device()
    family_probabilities: dict[str, list[list[list[float]]]] = {
        family: [] for family in families
    }
    run_metadata: dict[str, list[dict[str, Any]]] = {family: [] for family in families}
    for family, prefix in families.items():
        for seed in args.seeds:
            run_dir = args.deep_results_root / f"{prefix}{seed}"
            probabilities, metadata = load_run_probabilities(
                run_dir, items, paths, device, batch_size=args.batch_size
            )
            family_probabilities[family].append(probabilities)
            run_metadata[family].append({**metadata, "seed": seed})
            print(json.dumps({"family": family, "seed": seed, "items": len(probabilities)}), flush=True)
    normalization_records = {
        family: renormalize_probability_matrices(probabilities)
        for family, probabilities in family_probabilities.items()
    }
    means = {
        family: mean_vectors(probabilities)
        for family, probabilities in family_probabilities.items()
    }
    payload = {
        "status": "POST_CONFIRMATORY_BASELINE_EXTENSION_INFERENCE_COMPLETE_LABELS_NOT_READ",
        "dataset": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "graph_paths": str(args.graph_paths),
        "graph_paths_sha256": sha256(args.graph_paths),
        "items": len(items),
        "labels": LABELS,
        "seeds": args.seeds,
        "ensemble_aggregation": "arithmetic mean of class-probability vectors",
        "probability_postprocessing": (
            "Each four-class vector was renormalized to unit sum after float16 "
            "inference; raw sums deviating from one by more than 0.002 were rejected."
        ),
        "maximum_raw_probability_sum_deviation": normalization_records,
        "families": families,
        "run_metadata": run_metadata,
        "predictions": [
            {
                "item_id": str(item["item_id"]),
                "probabilities": {
                    family: means[family][item_index] for family in families
                },
                "probabilities_by_seed": {
                    family: {
                        str(seed): family_probabilities[family][seed_index][item_index]
                        for seed_index, seed in enumerate(args.seeds)
                    }
                    for family in families
                },
            }
            for item_index, item in enumerate(items)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "items": len(items), "sha256": sha256(args.output)}), flush=True)


if __name__ == "__main__":
    main()
