from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ensemble_calibration import load_run_probabilities  # noqa: E402
from fera_kg.benchmark import load_graph_path_split, load_labeled_split  # noqa: E402
from fera_kg.deep_models import LABELS  # noqa: E402
from train_model import select_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frozen seed and ensemble probabilities for labeled development splits."
    )
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20260901, 20260902, 20260903, 20260904, 20260905],
    )
    parser.add_argument(
        "--splits", nargs="+", choices=["train", "development", "calibration"], required=True
    )
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    graph_path_dir = benchmark_dir / "graph_paths_v1"
    device = select_device()
    output_rows: list[dict[str, Any]] = []
    runs: dict[str, list[dict[str, Any]]] = {}
    split_hashes: dict[str, dict[str, str]] = {}

    for split in args.splits:
        items = load_labeled_split(split, benchmark_dir=benchmark_dir)
        paths = load_graph_path_split(split, graph_path_dir=graph_path_dir)
        if [str(item["item_id"]) for item in items] != [str(path["item_id"]) for path in paths]:
            raise ValueError(f"item/path order mismatch for {split}")
        seed_probabilities: list[list[list[float]]] = []
        split_runs: list[dict[str, Any]] = []
        for seed in args.seeds:
            run_dir = PROJECT_ROOT / "results" / "deep_development" / f"{args.run_prefix}-s{seed}"
            probabilities, metadata = load_run_probabilities(
                run_dir, items, paths, device, args.batch_size
            )
            seed_probabilities.append(probabilities)
            split_runs.append(metadata)
        runs[split] = split_runs
        for index, item in enumerate(items):
            seed_rows = {
                str(seed): {
                    label: float(seed_probabilities[seed_index][index][label_index])
                    for label_index, label in enumerate(LABELS)
                }
                for seed_index, seed in enumerate(args.seeds)
            }
            ensemble = {
                label: sum(
                    seed_probabilities[seed_index][index][label_index]
                    for seed_index in range(len(args.seeds))
                )
                / len(args.seeds)
                for label_index, label in enumerate(LABELS)
            }
            output_rows.append(
                {
                    "item_id": str(item["item_id"]),
                    "split": split,
                    "seed_probabilities": seed_rows,
                    "ensemble_probabilities": ensemble,
                }
            )
        split_hashes[split] = {
            "items": sha256(benchmark_dir / f"{split}.jsonl"),
            "graph_paths": sha256(graph_path_dir / f"{split}.jsonl"),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": "LABELED_DEVELOPMENT_SPLIT_PREDICTIONS_TEST_LABELS_NOT_ACCESSED",
        "run_prefix": args.run_prefix,
        "seeds": args.seeds,
        "splits": args.splits,
        "rows": len(output_rows),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "runs": runs,
        "input_hashes": split_hashes,
        "test_labels_accessed": False,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "rows": len(output_rows), "sha256": manifest["output_sha256"]}))


if __name__ == "__main__":
    main()
