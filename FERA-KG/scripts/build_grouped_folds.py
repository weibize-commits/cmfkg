from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT.parent / "confidential_review_data" / "model_development"
DEFAULT_OUTPUT = PROJECT_ROOT.parent / "confidential_review_data" / "grouped_folds"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "items": len(rows),
        "passages": len({str(row["passage_id"]) for row in rows}),
        "sources": len({str(row["source_name"]) for row in rows}),
        "label_counts": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        "relation_group_counts": dict(
            sorted(Counter(str(row["relation_group"]) for row in rows).items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create source-work-disjoint cross-fitting folds from the packaged development data."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--group-by",
        choices=["passage", "source_work"],
        default="source_work",
        help="Grouping unit held out from every cross-fitting training fold.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    rows = read_jsonl(source / "train.jsonl")
    path_rows = read_jsonl(source / "graph_paths_v1" / "train.jsonl")
    path_by_id = {str(row["item_id"]): row for row in path_rows}
    if len(path_by_id) != len(path_rows):
        raise ValueError("duplicate graph-path item IDs")
    if {str(row["item_id"]) for row in rows} != set(path_by_id):
        raise ValueError("benchmark and graph-path item sets differ")

    labels = [str(row["label"]) for row in rows]
    group_field = "passage_id" if args.group_by == "passage" else "source_name"
    groups = [str(row[group_field]) for row in rows]
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    assignment: dict[str, int] = {}
    manifest_folds = []
    for fold, (train_indices, heldout_indices) in enumerate(
        splitter.split(rows, labels, groups)
    ):
        train_rows = [rows[index] for index in train_indices]
        heldout_rows = [rows[index] for index in heldout_indices]
        train_groups = {str(row[group_field]) for row in train_rows}
        heldout_groups = {str(row[group_field]) for row in heldout_rows}
        if train_groups & heldout_groups:
            raise AssertionError(f"{args.group_by} leakage in fold {fold}")
        for row in heldout_rows:
            item_id = str(row["item_id"])
            if item_id in assignment:
                raise AssertionError(f"item assigned to multiple folds: {item_id}")
            assignment[item_id] = fold

        fold_root = output / f"fold_{fold}"
        fold_train = [dict(row, split="train") for row in train_rows]
        fold_heldout = [dict(row, split="development") for row in heldout_rows]
        fold_train_paths = [
            dict(path_by_id[str(row["item_id"])], split="train") for row in train_rows
        ]
        fold_heldout_paths = [
            dict(path_by_id[str(row["item_id"])], split="development")
            for row in heldout_rows
        ]
        write_jsonl(fold_root / "train.jsonl", fold_train)
        write_jsonl(fold_root / "development.jsonl", fold_heldout)
        write_jsonl(fold_root / "graph_paths_v1" / "train.jsonl", fold_train_paths)
        write_jsonl(
            fold_root / "graph_paths_v1" / "development.jsonl", fold_heldout_paths
        )
        manifest_folds.append(
            {
                "fold": fold,
                "train": summarize(fold_train),
                "heldout": summarize(fold_heldout),
                "grouping_unit": args.group_by,
                "group_field": group_field,
                "group_overlap": 0,
                "files": {
                    "train": sha256(fold_root / "train.jsonl"),
                    "development": sha256(fold_root / "development.jsonl"),
                    "train_graph_paths": sha256(
                        fold_root / "graph_paths_v1" / "train.jsonl"
                    ),
                    "development_graph_paths": sha256(
                        fold_root / "graph_paths_v1" / "development.jsonl"
                    ),
                },
            }
        )

    if len(assignment) != len(rows):
        raise AssertionError("cross-fit assignment is incomplete")
    assignment_path = output / "fold_assignments.jsonl"
    write_jsonl(
        assignment_path,
        [
            {"item_id": item_id, "fold": fold}
            for item_id, fold in sorted(assignment.items())
        ],
    )
    manifest = {
        "status": f"TRAIN_ONLY_{args.group_by.upper()}_DISJOINT_CROSSFIT_FOLDS",
        "source": Path(os.path.relpath(source, PROJECT_ROOT)).as_posix(),
        "source_train_sha256": sha256(source / "train.jsonl"),
        "source_train_graph_paths_sha256": sha256(
            source / "graph_paths_v1" / "train.jsonl"
        ),
        "folds": args.folds,
        "seed": args.seed,
        "grouping_unit": args.group_by,
        "group_field": group_field,
        "assignment_sha256": sha256(assignment_path),
        "all_items_assigned_once": True,
        "development_and_test_items_used": False,
        "fold_summaries": manifest_folds,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "items": len(rows), "folds": args.folds}))


if __name__ == "__main__":
    main()
