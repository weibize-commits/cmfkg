from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "confidential_review_data" / "inference" / "natural" / "registered"
OUTPUT = ROOT.parent / "confidential_review_data" / "inference" / "natural"
VARIANTS = ("registered", "shuffled", "empty")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_rank(recipient: str, donor: str) -> str:
    return hashlib.sha256(f"20260908|{recipient}|{donor}".encode("utf-8")).hexdigest()


def donor_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_signature: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_signature[(str(row["relation"]), str(row["head_type"]), str(row["tail_type"]))].append(row)
        by_group[str(row["relation_group"])].append(row)
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row["item_id"])
        source = str(row["source_name"])
        signature = (str(row["relation"]), str(row["head_type"]), str(row["tail_type"]))
        pools = (by_signature[signature], by_group[str(row["relation_group"])], rows)
        candidates: list[dict[str, Any]] = []
        for pool in pools:
            candidates = [
                donor
                for donor in pool
                if str(donor["item_id"]) != item_id
                and str(donor["source_name"]) != source
                and str(donor.get("passage_text", "")) != str(row.get("passage_text", ""))
            ]
            if candidates:
                break
        if not candidates:
            raise RuntimeError(f"No admissible shuffled-passage donor for {item_id}")
        output[item_id] = min(candidates, key=lambda donor: stable_rank(item_id, str(donor["item_id"])))
    return output


def feature_variants(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    donors = donor_map(rows)
    correct, shuffled, empty = [], [], []
    for row in rows:
        base = copy.deepcopy(row)
        base["passage_diagnostic"] = "correct_passage"
        correct.append(base)

        donor = donors[str(row["item_id"])]
        changed = copy.deepcopy(row)
        changed["passage_text"] = str(donor["passage_text"])
        changed["head_start"] = -1
        changed["head_end"] = -1
        changed["tail_start"] = -1
        changed["tail_end"] = -1
        changed["passage_diagnostic"] = "deterministic_cross_work_shuffled_passage"
        changed["passage_donor_item_id"] = str(donor["item_id"])
        changed["passage_donor_source_work"] = str(donor["source_name"])
        shuffled.append(changed)

        blank = copy.deepcopy(row)
        blank["passage_text"] = ""
        blank["head_start"] = -1
        blank["head_end"] = -1
        blank["tail_start"] = -1
        blank["tail_end"] = -1
        blank["passage_diagnostic"] = "empty_passage"
        empty.append(blank)
    return {
        "registered": correct,
        "shuffled": shuffled,
        "empty": empty,
    }


def main() -> None:
    rows = read_jsonl(SOURCE / "items.jsonl")
    paths = read_jsonl(SOURCE / "target_edge_deleted_paths.jsonl")
    ids = [str(row["item_id"]) for row in rows]
    if len(rows) != 800 or ids != [str(row["item_id"]) for row in paths]:
        raise ValueError("Natural-cohort feature and graph-path inputs are not aligned at 800 items")
    variants = feature_variants(rows)
    manifest: dict[str, Any] = {
        "version": "tcft_natural_800_passage_diagnostics_v1",
        "items": 800,
        "labels_accessed": False,
        "design": (
            "Candidate triples, provenance identifiers, and target-edge-deleted graph records are held fixed. "
            "Only the registered passage is kept, deterministically replaced by a different source work, or emptied. "
            "All passage-dependent neural encodings, passage-local evidence selection, surface diagnostics, and fusion "
            "predictions must be recomputed downstream."
        ),
        "variants": {},
    }
    for variant in VARIANTS:
        dataset = OUTPUT / variant / "items.jsonl"
        graph_path = OUTPUT / variant / "target_edge_deleted_paths.jsonl"
        variant_paths = []
        for path_row in paths:
            copied = copy.deepcopy(path_row)
            copied["passage_diagnostic"] = variant
            copied["graph_context_held_fixed"] = True
            variant_paths.append(copied)
        write_jsonl(dataset, variants[variant])
        write_jsonl(graph_path, variant_paths)
        manifest["variants"][variant] = {
            "dataset": Path(os.path.relpath(dataset, ROOT)).as_posix(),
            "dataset_sha256": sha256(dataset),
            "graph_paths": Path(os.path.relpath(graph_path, ROOT)).as_posix(),
            "graph_paths_sha256": sha256(graph_path),
        }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "variants": list(VARIANTS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

