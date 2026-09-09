from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parent / "confidential_review_data" / "results" / "matched_retraining"
ITEMS = ROOT.parent / "confidential_review_data" / "inference" / "controlled" / "items.jsonl"
REFERENCE = ROOT.parent / "confidential_review_data" / "evaluation" / "controlled" / "items.jsonl"
OUT_JSON = ROOT / "outputs" / "matched_module_analysis.json"
OUT_CSV = ROOT / "outputs" / "matched_module_analysis.csv"
LABELS = ["SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_CONTEXT", "ENTITY_OR_TYPE_ERROR"]
TASKS = [
    "ecrp_correct",
    "unconditioned_path",
    "counterfactual_off",
    "no_paths",
    "shuffled_passage",
    "empty_passage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse matched five-seed ECRP ablations.")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260926)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def macro(gold: list[str], predicted: list[str]) -> float:
    return float(f1_score(gold, predicted, labels=LABELS, average="macro", zero_division=0))


def source_bootstrap_interval(
    gold: list[str],
    candidate: list[str],
    comparator: list[str],
    sources: list[str],
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, source in enumerate(sources):
        grouped[source].append(index)
    names = sorted(grouped)
    rng = np.random.default_rng(seed)
    label_to_code = {label: index for index, label in enumerate(LABELS)}
    gold_codes = np.asarray([label_to_code[value] for value in gold], dtype=np.int8)
    candidate_codes = np.asarray([label_to_code[value] for value in candidate], dtype=np.int8)
    comparator_codes = np.asarray([label_to_code[value] for value in comparator], dtype=np.int8)

    def fast_macro(g: np.ndarray, p: np.ndarray) -> float:
        matrix = np.bincount(g.astype(int) * 4 + p.astype(int), minlength=16).reshape(4, 4)
        tp = np.diag(matrix).astype(float)
        denominator = 2.0 * tp + matrix.sum(axis=0) - tp + matrix.sum(axis=1) - tp
        per_class = np.divide(
            2.0 * tp,
            denominator,
            out=np.zeros(4, dtype=float),
            where=denominator > 0,
        )
        return float(per_class.mean())

    differences = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_sources = rng.choice(names, size=len(names), replace=True)
        sampled: list[int] = []
        for source in sampled_sources:
            indices = grouped[str(source)]
            sampled.extend(rng.choice(indices, size=len(indices), replace=True).tolist())
        sampled_array = np.asarray(sampled, dtype=int)
        sampled_gold = gold_codes[sampled_array]
        differences[replicate] = fast_macro(
            sampled_gold, candidate_codes[sampled_array]
        ) - fast_macro(sampled_gold, comparator_codes[sampled_array])
    return [float(value) for value in np.quantile(differences, [0.025, 0.975])]


def main() -> None:
    args = parse_args()
    items = read_jsonl(ITEMS)
    references = {str(row["item_id"]): str(row["label"]) for row in read_jsonl(REFERENCE)}
    item_ids = [str(row["item_id"]) for row in items]
    gold = [references[item_id] for item_id in item_ids]
    sources = [str(row["source_name"]) for row in items]
    task_payloads: dict[str, Any] = {}
    missing = []
    for task in TASKS:
        path = args.results / f"{task}_predictions.json"
        if not path.exists():
            missing.append(str(path))
            continue
        task_payloads[task] = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        raise FileNotFoundError("Matched ablation outputs are incomplete:\n" + "\n".join(missing))

    summaries: dict[str, Any] = {}
    ensemble_labels: dict[str, list[str]] = {}
    seed_labels: dict[str, dict[str, list[str]]] = {}
    for task, payload in task_payloads.items():
        predictions = payload["predictions"]
        if [str(row["item_id"]) for row in predictions] != item_ids:
            raise ValueError(f"item order mismatch for {task}")
        seeds = [str(seed) for seed in payload["seeds"]]
        seed_labels[task] = {}
        per_seed = {}
        for seed in seeds:
            labels = [
                LABELS[int(np.argmax(row["probabilities_by_seed"][task][seed]))]
                for row in predictions
            ]
            seed_labels[task][seed] = labels
            per_seed[seed] = macro(gold, labels)
        means = np.asarray(list(per_seed.values()), dtype=float)
        ensemble = [
            LABELS[int(np.argmax(row["probabilities"][task]))]
            for row in predictions
        ]
        ensemble_labels[task] = ensemble
        summaries[task] = {
            "per_seed_macro_f1": per_seed,
            "seed_mean": float(means.mean()),
            "seed_sample_standard_deviation": float(means.std(ddof=1)),
            "ensemble_macro_f1": macro(gold, ensemble),
        }

    reference_task = "ecrp_correct"
    contrasts: dict[str, Any] = {}
    for index, task in enumerate(TASKS[1:], start=1):
        common_seeds = sorted(set(seed_labels[reference_task]) & set(seed_labels[task]))
        per_seed_difference = {
            seed: macro(gold, seed_labels[reference_task][seed])
            - macro(gold, seed_labels[task][seed])
            for seed in common_seeds
        }
        differences = np.asarray(list(per_seed_difference.values()), dtype=float)
        candidate_score = summaries[reference_task]["ensemble_macro_f1"]
        comparator_score = summaries[task]["ensemble_macro_f1"]
        contrasts[task] = {
            "reference_minus_variant_ensemble_macro_f1": candidate_score - comparator_score,
            "paired_seed_differences": per_seed_difference,
            "paired_seed_mean_difference": float(differences.mean()),
            "paired_seed_sample_standard_deviation": float(differences.std(ddof=1)),
            "source_work_bootstrap_95_ci": source_bootstrap_interval(
                gold,
                ensemble_labels[reference_task],
                ensemble_labels[task],
                sources,
                replicates=args.bootstrap,
                seed=args.seed + index,
            ),
        }

    report = {
        "status": "COMPLETE_MATCHED_RETRAINING_ANALYSIS",
        "items": len(items),
        "labels_fixed_for_scoring": LABELS,
        "zero_division": 0,
        "tasks": summaries,
        "contrasts_against_ecrp_correct": contrasts,
        "interpretation_rule": (
            "A module-level claim requires the matched-seed direction to be stable and the source-work "
            "bootstrap interval for the ensemble difference to exclude zero."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task",
                "seed_mean_macro_f1",
                "seed_sd",
                "ensemble_macro_f1",
                "ecrp_correct_minus_variant",
                "source_work_ci_low",
                "source_work_ci_high",
            ]
        )
        for task in TASKS:
            contrast = contrasts.get(task)
            writer.writerow(
                [
                    task,
                    summaries[task]["seed_mean"],
                    summaries[task]["seed_sample_standard_deviation"],
                    summaries[task]["ensemble_macro_f1"],
                    "" if contrast is None else contrast["reference_minus_variant_ensemble_macro_f1"],
                    "" if contrast is None else contrast["source_work_bootstrap_95_ci"][0],
                    "" if contrast is None else contrast["source_work_bootstrap_95_ci"][1],
                ]
            )
    print(json.dumps({"output": str(OUT_JSON), "table": str(OUT_CSV)}))


if __name__ == "__main__":
    main()
