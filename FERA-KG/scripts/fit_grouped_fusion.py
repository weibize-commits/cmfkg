from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import local_evidence_features as rag  # noqa: E402
import local_evidence_pipeline as v3  # noqa: E402
import fusion_features as sg  # noqa: E402


DATA_ROOT = ROOT.parent / "confidential_review_data"
SOURCE_FOLD_ROOT = DATA_ROOT / "grouped_folds"
SOURCE_OOF_ROOT = DATA_ROOT / "oof"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_oof_probabilities(task: str, expected_ids: set[str]) -> dict[str, list[float]]:
    path = SOURCE_OOF_ROOT / f"{task}_oof_predictions.jsonl"
    manifest = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not manifest.is_file():
        raise FileNotFoundError(
            f"Missing source-work OOF output for {task}: {path}. "
            "Run and download the four source-work multiseed jobs first."
        )
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("grouping_unit") != "source_work" or metadata.get("items_covered_exactly_once") is not True:
        raise ValueError(f"{task}: source-work OOF manifest is not valid")
    if int(metadata.get("folds", -1)) != 5:
        raise ValueError(f"{task}: expected five source-work folds")
    if [int(seed) for seed in metadata.get("seeds_per_fold", [])] != EXPECTED_DEEP_SEEDS:
        raise ValueError(f"{task}: OOF manifest does not contain the five fixed seeds")
    expected_runs = {
        (fold, seed) for fold in range(5) for seed in EXPECTED_DEEP_SEEDS
    }
    observed_runs = {
        (int(record["fold"]), int(record["seed"]))
        for record in metadata.get("source_files", [])
    }
    if observed_runs != expected_runs or len(metadata.get("source_files", [])) != 25:
        raise ValueError(f"{task}: OOF manifest does not register all 5 folds x 5 seeds")
    if int(metadata.get("items", -1)) != len(expected_ids):
        raise ValueError(f"{task}: OOF manifest item count does not match training data")
    assignment = SOURCE_FOLD_ROOT / "fold_assignments.jsonl"
    if metadata.get("fold_assignment_sha256") != sha256(assignment):
        raise ValueError(f"{task}: OOF manifest references a different fold assignment")
    rows = read_jsonl(path)
    row_ids = [str(row["item_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f"{task}: OOF prediction rows contain duplicate item identifiers")
    indexed = {
        str(row["item_id"]): [float(row["probabilities"][label]) for label in sg.LABELS]
        for row in rows
    }
    if set(indexed) != expected_ids:
        raise ValueError(
            f"{task}: OOF item mismatch; missing={len(expected_ids-set(indexed))}, "
            f"extra={len(set(indexed)-expected_ids)}"
        )
    for item_id, probabilities in indexed.items():
        values = np.asarray(probabilities, dtype=np.float64)
        if values.shape != (len(sg.LABELS),) or not np.all(np.isfinite(values)):
            raise ValueError(f"{task}/{item_id}: invalid OOF probability vector")
        if np.any(values < -1e-8) or np.any(values > 1.0 + 1e-8):
            raise ValueError(f"{task}/{item_id}: OOF probability outside [0, 1]")
        if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=2e-5):
            raise ValueError(f"{task}/{item_id}: OOF probabilities do not sum to one")
    seed_keys = {str(seed) for seed in EXPECTED_DEEP_SEEDS}
    for row in rows:
        item_id = str(row["item_id"])
        by_seed = row.get("probabilities_by_seed", {})
        if set(by_seed) != seed_keys:
            raise ValueError(f"{task}/{item_id}: incomplete per-seed OOF probabilities")
        seed_vectors = np.asarray(
            [
                [float(by_seed[str(seed)][label]) for label in sg.LABELS]
                for seed in EXPECTED_DEEP_SEEDS
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(seed_vectors)) or not np.allclose(
            seed_vectors.sum(axis=1), 1.0, rtol=0.0, atol=2e-5
        ):
            raise ValueError(f"{task}/{item_id}: invalid per-seed OOF probabilities")
        recomputed = seed_vectors.mean(axis=0)
        stored = np.asarray(indexed[item_id], dtype=np.float64)
        if not np.allclose(recomputed, stored, rtol=0.0, atol=1e-10):
            raise ValueError(
                f"{task}/{item_id}: stored OOF probabilities are not the five-seed mean"
            )
    return indexed


def source_work_oof_rag_probabilities(
    rows: list[dict[str, Any]], paths: list[dict[str, Any]]
) -> np.ndarray:
    assignment_rows = read_jsonl(SOURCE_FOLD_ROOT / "fold_assignments.jsonl")
    assignments = {
        str(row["item_id"]): int(row["fold"])
        for row in assignment_rows
    }
    if len(assignments) != len(assignment_rows):
        raise ValueError("Source-work fold assignments contain duplicate item identifiers")
    expected = {str(row["item_id"]) for row in rows}
    if set(assignments) != expected:
        raise ValueError("Source-work fold assignments do not match fusion-training items")
    if set(assignments.values()) != set(range(5)):
        raise ValueError("Source-work fold assignments must contain exactly folds 0-4")
    source_folds: dict[str, set[int]] = {}
    for row in rows:
        source = str(row.get("source_name", "")).strip()
        if not source:
            raise ValueError("Fusion-training row is missing the source_name grouping field")
        source_folds.setdefault(source, set()).add(assignments[str(row["item_id"])])
    leaked_sources = sorted(source for source, folds in source_folds.items() if len(folds) != 1)
    if leaked_sources:
        raise ValueError(
            f"Source-work cross-fitting leakage detected for {len(leaked_sources)} sources"
        )
    output = np.zeros((len(rows), 3), dtype=np.float64)
    covered = np.zeros(len(rows), dtype=bool)
    for fold in range(5):
        fit_indices = [
            index for index, row in enumerate(rows)
            if assignments[str(row["item_id"])] != fold
        ]
        held_indices = [
            index for index, row in enumerate(rows)
            if assignments[str(row["item_id"])] == fold
        ]
        fit_rows = [rows[index] for index in fit_indices]
        fit_paths = [paths[index] for index in fit_indices]
        held_rows = [rows[index] for index in held_indices]
        held_paths = [paths[index] for index in held_indices]
        blocks = rag.build_feature_blocks(
            fit_rows, held_rows, fit_paths, held_paths, top_k=5
        )
        labels = np.asarray(
            [rag.LABELS.index(str(row["label"])) for row in fit_rows],
            dtype=np.int64,
        )
        probabilities, _ = rag.stage_probabilities(
            blocks["full_train"],
            blocks["full_development"],
            labels,
            c=0.25,
            omission_matrix=blocks["full_omission"],
            omission_weight=0.35,
        )
        output[held_indices] = probabilities
        covered[held_indices] = True
        print(f"SOURCE_WORK_LOCAL_EVIDENCE_FOLD_COMPLETE fold={fold} items={len(held_indices)}", flush=True)
    if not bool(covered.all()):
        raise RuntimeError("Not all fusion-training items received source-work OOF local-evidence probabilities")
    return output

