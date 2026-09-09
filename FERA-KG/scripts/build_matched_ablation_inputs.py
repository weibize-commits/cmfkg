from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT.parent / "confidential_review_data" / "matched_retraining" / "model_development"
TEST_ROOT = ROOT.parent / "confidential_review_data" / "inference" / "controlled"
OUTPUT = ROOT / "outputs" / "matched_inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build label-preserving passage and graph interventions for matched ECRP ablations."
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shuffled_passages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) < 2:
        raise ValueError("at least two rows are required for a passage permutation")
    order = sorted(
        range(len(rows)),
        key=lambda index: hashlib.sha256(str(rows[index]["item_id"]).encode("utf-8")).hexdigest(),
    )
    donor_for: dict[int, int] = {}
    for position, recipient in enumerate(order):
        donor_for[recipient] = order[(position + 1) % len(order)]
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        donor = rows[donor_for[index]]
        revised = copy.deepcopy(row)
        revised["passage_text"] = str(donor["passage_text"])
        revised["head_start"] = -1
        revised["head_end"] = -1
        revised["tail_start"] = -1
        revised["tail_end"] = -1
        revised["ablation_intervention"] = "deterministic_deranged_passage"
        revised["ablation_donor_item_id"] = str(donor["item_id"])
        revised["normalized_passage_sha256"] = hashlib.sha256(
            str(donor["passage_text"]).encode("utf-8")
        ).hexdigest()
        output.append(revised)
    if any(row["item_id"] == row["ablation_donor_item_id"] for row in output):
        raise AssertionError("passage permutation contains a fixed point")
    return output


def empty_passages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        revised = copy.deepcopy(row)
        revised["passage_text"] = ""
        revised["head_start"] = -1
        revised["head_end"] = -1
        revised["tail_start"] = -1
        revised["tail_end"] = -1
        revised["ablation_intervention"] = "empty_passage"
        revised["normalized_passage_sha256"] = hashlib.sha256(b"").hexdigest()
        output.append(revised)
    return output


def no_paths(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        revised = copy.deepcopy(row)
        revised["paths"] = []
        revised["path_count"] = 0
        revised["passage_conditioned_path_count"] = 0
        revised["semantic_only_path_count"] = 0
        revised["enumeration_truncated"] = False
        revised["ablation_intervention"] = "all_paths_removed_before_training"
        output.append(revised)
    return output


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    split_sources = {
        "train": TRAIN_ROOT / "train.jsonl",
        "development": TRAIN_ROOT / "development.jsonl",
        "confirmatory": TEST_ROOT / "items.jsonl",
    }
    path_sources = {
        "train": TRAIN_ROOT / "graph_paths_v1/train.jsonl",
        "development": TRAIN_ROOT / "graph_paths_v1/development.jsonl",
        "confirmatory": TEST_ROOT / "target_edge_deleted_paths.jsonl",
    }
    manifest: dict[str, Any] = {
        "version": "tcft_ecrp_matched_ablation_inputs_v1",
        "status": "LABEL_PRESERVING_INTERVENTION_INPUTS_ONLY",
        "variants": {},
    }
    for split, source in split_sources.items():
        rows = read_jsonl(source)
        variants = {
            "correct_passage": rows,
            "shuffled_passage": shuffled_passages(rows),
            "empty_passage": empty_passages(rows),
        }
        for variant, variant_rows in variants.items():
            target_name = "confirmatory_features.jsonl" if split == "confirmatory" else f"{split}.jsonl"
            target = args.output / variant / target_name
            write_jsonl(target, variant_rows)
            manifest["variants"].setdefault(variant, {})[split] = {
                "items": len(variant_rows),
                "path": Path(os.path.relpath(target, ROOT)).as_posix(),
                "sha256": sha256(target),
            }
    for split, source in path_sources.items():
        rows = no_paths(read_jsonl(source))
        target_name = "confirmatory_features.jsonl" if split == "confirmatory" else f"{split}.jsonl"
        target = args.output / "no_paths/graph_paths_v1" / target_name
        write_jsonl(target, rows)
        manifest["variants"].setdefault("no_paths", {})[split] = {
            "items": len(rows),
            "path": Path(os.path.relpath(target, ROOT)).as_posix(),
            "sha256": sha256(target),
        }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
