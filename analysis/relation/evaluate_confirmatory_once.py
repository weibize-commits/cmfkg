from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "paper_revision" / "relation_improvement"
DATA_DIR = EXPERIMENT_DIR / "data"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
CONFIG_DIR = EXPERIMENT_DIR / "configs"
PREDICTION_DIR = EXPERIMENT_DIR / "predictions"
BUILDER_PATH = ROOT / "paper_revision" / "tools" / "build_relation_baseline_package.py"

SEED = 20260731
BOOTSTRAP_ROUNDS = 5000


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def candidate_key(record: dict[str, Any]) -> tuple[Any, ...]:
    def clean(value: Any) -> str:
        return "".join(str(value or "").split())

    head = record["head"]
    tail = record["tail"]
    return (
        str(record["sample_id"]),
        int(head["start_char"]),
        int(head["end_char"]),
        clean(head["text"]),
        str(head["type"]),
        str(record["relation"]),
        int(tail["start_char"]),
        int(tail["end_char"]),
        clean(tail["text"]),
        str(tail["type"]),
    )


def llm_digest(record: dict[str, Any]) -> str:
    def clean(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    head = record["head"]
    tail = record["tail"]
    key = (
        str(record["sample_id"]),
        int(head["start_char"]),
        int(head["end_char"]),
        clean(head["text"]),
        str(head["type"]),
        str(record["relation"]),
        int(tail["start_char"]),
        int(tail["end_char"]),
        clean(tail["text"]),
        str(tail["type"]),
    )
    payload = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()[:20]


def llm_support_score(decision: dict[str, Any]) -> float:
    label = str(decision.get("label") or "unsupported")
    confidence = float(decision.get("confidence") or 0.0)
    if label == "supported":
        return max(0.75, confidence)
    if label == "plausible":
        return 0.55 + 0.25 * confidence
    return 0.15 * (1.0 - confidence)


def prf_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def count_for_samples(
    gold_by_sample: dict[str, set[tuple[Any, ...]]],
    pred_by_sample: dict[str, set[tuple[Any, ...]]],
    sampled_ids: list[str],
) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for sid in sampled_ids:
        gold = gold_by_sample.get(sid, set())
        pred = pred_by_sample.get(sid, set())
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return tp, fp, fn


def bootstrap_metrics(
    gold_by_sample: dict[str, set[tuple[Any, ...]]],
    pred_by_sample: dict[str, set[tuple[Any, ...]]],
    sample_ids: list[str],
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = random.Random(seed)
    values = {"precision": [], "recall": [], "f1": []}
    for _ in range(BOOTSTRAP_ROUNDS):
        sampled = [rng.choice(sample_ids) for _ in sample_ids]
        metrics = prf_counts(*count_for_samples(gold_by_sample, pred_by_sample, sampled))
        for name, value in zip(["precision", "recall", "f1"], metrics):
            values[name].append(value)
    return {
        name: (
            float(np.percentile(items, 2.5)),
            float(np.percentile(items, 97.5)),
        )
        for name, items in values.items()
    }


def paired_bootstrap_difference(
    gold_by_sample: dict[str, set[tuple[Any, ...]]],
    left: dict[str, set[tuple[Any, ...]]],
    right: dict[str, set[tuple[Any, ...]]],
    sample_ids: list[str],
    seed: int,
) -> tuple[float, float, float, float]:
    observed_left = prf_counts(
        *count_for_samples(gold_by_sample, left, sample_ids)
    )[2]
    observed_right = prf_counts(
        *count_for_samples(gold_by_sample, right, sample_ids)
    )[2]
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(BOOTSTRAP_ROUNDS):
        sampled = [rng.choice(sample_ids) for _ in sample_ids]
        left_f1 = prf_counts(*count_for_samples(gold_by_sample, left, sampled))[2]
        right_f1 = prf_counts(*count_for_samples(gold_by_sample, right, sampled))[2]
        differences.append(left_f1 - right_f1)
    low, high = np.percentile(differences, [2.5, 97.5])
    difference_array = np.array(differences)
    lower_tail = (int(np.sum(difference_array <= 0)) + 1) / (
        BOOTSTRAP_ROUNDS + 1
    )
    upper_tail = (int(np.sum(difference_array >= 0)) + 1) / (
        BOOTSTRAP_ROUNDS + 1
    )
    two_sided_p = 2 * min(lower_tail, upper_tail)
    return observed_left - observed_right, float(low), float(high), float(min(1.0, two_sided_p))


def by_sample(records: Iterable[dict[str, Any]]) -> dict[str, set[tuple[Any, ...]]]:
    result: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for record in records:
        result[str(record["sample_id"])].add(candidate_key(record))
    return result


def verify_freezes(
    local_freeze: dict[str, Any],
    comparator_freeze: dict[str, Any],
    policy_freeze: dict[str, Any],
    llm_manifest: dict[str, Any],
) -> None:
    checks = [
        (
            sha256(DATA_DIR / "test_passages.jsonl"),
            local_freeze["confirmatory_test_passages_sha256"],
            "test passages",
        ),
        (
            sha256(DATA_DIR / "test_candidates.jsonl"),
            local_freeze["confirmatory_test_candidates_sha256"],
            "test candidates",
        ),
        (
            sha256(DATA_DIR / "test_gold.jsonl"),
            local_freeze["confirmatory_test_gold_sha256"],
            "test gold",
        ),
        (
            sha256(OUTPUT_DIR / "confirmatory_test_scores_blinded.jsonl"),
            local_freeze["confirmatory_test_scores_sha256"],
            "local test scores",
        ),
        (
            sha256(OUTPUT_DIR / "fixed_charngram_test_scores_blinded.jsonl"),
            comparator_freeze["confirmatory_scores_sha256"],
            "comparator test scores",
        ),
    ]
    for observed, expected, label in checks:
        if observed != expected:
            raise RuntimeError(f"Freeze hash mismatch for {label}")
    if llm_manifest.get("status") != "complete":
        raise RuntimeError("Confirmatory LLM run is incomplete")
    if llm_manifest.get("prompt_sha256") != policy_freeze.get("llm_prompt_sha256"):
        raise RuntimeError("Confirmatory LLM prompt does not match frozen development prompt")
    if int(llm_manifest.get("decision_count") or 0) != int(
        llm_manifest.get("total_candidates") or -1
    ):
        raise RuntimeError("Confirmatory LLM decisions are incomplete")


def main() -> None:
    if (OUTPUT_DIR / "confirmatory_evaluation_manifest.json").exists():
        raise RuntimeError(
            "Confirmatory evaluation has already been run. Refusing a second look at "
            "the frozen test labels."
        )

    builder = load_module("relation_builder_confirmatory", BUILDER_PATH)
    local_freeze = json.loads(
        (CONFIG_DIR / "local_model_freeze.json").read_text(encoding="utf-8")
    )
    comparator_freeze = json.loads(
        (CONFIG_DIR / "fixed_comparator_freeze.json").read_text(encoding="utf-8")
    )
    policy_freeze = json.loads(
        (CONFIG_DIR / "llm_policy_freeze.json").read_text(encoding="utf-8")
    )
    llm_manifest = json.loads(
        (EXPERIMENT_DIR / "llm_runs" / "test" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    verify_freezes(local_freeze, comparator_freeze, policy_freeze, llm_manifest)

    passages = read_jsonl(DATA_DIR / "test_passages.jsonl")
    candidates = read_jsonl(DATA_DIR / "test_candidates.jsonl")
    gold = read_jsonl(DATA_DIR / "test_gold.jsonl")
    local_scores_raw = read_jsonl(
        OUTPUT_DIR / "confirmatory_test_scores_blinded.jsonl"
    )
    comparator_scores_raw = read_jsonl(
        OUTPUT_DIR / "fixed_charngram_test_scores_blinded.jsonl"
    )
    decisions = read_jsonl(OUTPUT_DIR / "deepseek_triage_test_decisions.jsonl")
    sample_ids = [str(row["sample_id"]) for row in passages]
    texts = {
        str(row["sample_id"]): {"frozen_text": str(row["frozen_text"])}
        for row in passages
    }
    gold_by_sample = by_sample(gold)
    candidate_by_key = {candidate_key(row): row for row in candidates}
    local_score_by_key = {
        candidate_key(row): float(row["score"]) for row in local_scores_raw
    }
    comparator_score_by_key = {
        candidate_key(row): float(row["score"]) for row in comparator_scores_raw
    }
    decision_by_digest = {str(row["candidate_digest"]): row for row in decisions}
    if (
        len(candidate_by_key) != len(candidates)
        or len(local_score_by_key) != len(candidates)
        or len(comparator_score_by_key) != len(candidates)
        or len(decision_by_digest) != len(candidates)
    ):
        raise RuntimeError("Candidate, score, or LLM-decision cardinality mismatch")

    method_records: dict[str, list[dict[str, Any]]] = {
        "typed_cooccurrence": candidates[:],
        "distance_trigger_heuristic": [],
        "fixed_charngram_comparator": [],
        "local_symbolic_text": [],
        "deepseek_supported_only": [],
        "deepseek_supported_or_plausible": [],
        "cmfkg_frozen_hybrid": [],
    }
    local_thresholds = {
        str(key): float(value)
        for key, value in local_freeze["relation_thresholds"].items()
    }
    policy_family = str(policy_freeze["selected_family"])
    policy = policy_freeze["selected_parameters"]

    for candidate in candidates:
        key = candidate_key(candidate)
        local_score = local_score_by_key[key]
        comparator_score = comparator_score_by_key[key]
        _, heuristic_score, _ = builder.heuristic_score(candidate, texts)
        heuristic_keep = bool(builder.heuristic_score(candidate, texts)[0])
        decision = decision_by_digest[llm_digest(candidate)]
        llm_score = llm_support_score(decision)
        label = str(decision["label"])

        if heuristic_keep:
            method_records["distance_trigger_heuristic"].append(candidate)
        if comparator_score >= float(comparator_freeze["threshold_selected_on_dev"]):
            method_records["fixed_charngram_comparator"].append(candidate)
        if local_score >= local_thresholds.get(
            str(candidate["relation"]),
            float(local_freeze["selected_local_model"]["global_threshold"]),
        ):
            method_records["local_symbolic_text"].append(candidate)
        if label == "supported":
            method_records["deepseek_supported_only"].append(candidate)
        if label in {"supported", "plausible"}:
            method_records["deepseek_supported_or_plausible"].append(candidate)

        if policy_family == "local_only":
            hybrid_positive = local_score >= float(policy["threshold"])
        elif policy_family == "local_relation_thresholds":
            hybrid_positive = local_score >= local_thresholds.get(
                str(candidate["relation"]),
                float(local_freeze["selected_local_model"]["global_threshold"]),
            )
        elif policy_family == "relation_threshold_llm_rescue_veto":
            relation_threshold = local_thresholds.get(
                str(candidate["relation"]),
                float(local_freeze["selected_local_model"]["global_threshold"]),
            )
            base_positive = local_score >= relation_threshold
            llm_accept = label == "supported" or (
                bool(policy["accept_plausible"]) and label == "plausible"
            )
            rescued = (
                (not base_positive)
                and llm_accept
                and local_score
                >= max(0.0, relation_threshold - float(policy["rescue_margin"]))
            )
            veto_margin = policy.get("veto_margin")
            vetoed = (
                base_positive
                and veto_margin is not None
                and label == "unsupported"
                and local_score < relation_threshold + float(veto_margin)
            )
            hybrid_positive = (base_positive or rescued) and not vetoed
        elif policy_family == "weighted_score":
            combined = (
                float(policy["local_weight"]) * local_score
                + float(policy["llm_weight"]) * llm_score
                + float(policy["heuristic_weight"]) * float(heuristic_score)
            )
            hybrid_positive = combined >= float(policy["threshold"])
        elif policy_family == "protected_local_gate":
            llm_accept = label == "supported" or (
                bool(policy["accept_plausible"]) and label == "plausible"
            )
            hybrid_positive = local_score >= float(policy["upper"]) or (
                local_score >= float(policy["lower"]) and llm_accept
            )
        else:
            raise RuntimeError(f"Unknown frozen policy family: {policy_family}")
        if hybrid_positive:
            method_records["cmfkg_frozen_hybrid"].append(candidate)

    prediction_files: dict[str, Path] = {}
    for method, records in method_records.items():
        path = PREDICTION_DIR / f"{method}.jsonl"
        write_jsonl(path, records)
        prediction_files[method] = path

    metric_rows: list[dict[str, Any]] = []
    pred_by_method: dict[str, dict[str, set[tuple[Any, ...]]]] = {}
    for index, (method, records) in enumerate(method_records.items()):
        pred_by_sample = by_sample(records)
        pred_by_method[method] = pred_by_sample
        tp, fp, fn = count_for_samples(gold_by_sample, pred_by_sample, sample_ids)
        precision, recall, f1 = prf_counts(tp, fp, fn)
        intervals = bootstrap_metrics(
            gold_by_sample,
            pred_by_sample,
            sample_ids,
            SEED + index,
        )
        metric_rows.append(
            {
                "method": method,
                "n_passages": len(sample_ids),
                "gold_relations": len(gold),
                "candidate_pairs": len(candidates),
                "predicted_relations": len(records),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "precision_ci95_low": intervals["precision"][0],
                "precision_ci95_high": intervals["precision"][1],
                "recall": recall,
                "recall_ci95_low": intervals["recall"][0],
                "recall_ci95_high": intervals["recall"][1],
                "f1": f1,
                "f1_ci95_low": intervals["f1"][0],
                "f1_ci95_high": intervals["f1"][1],
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values("f1", ascending=False)
    metrics.to_csv(
        OUTPUT_DIR / "confirmatory_relation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    relation_types = sorted({str(row["relation"]) for row in candidates})
    by_relation_rows: list[dict[str, Any]] = []
    gold_keys = {candidate_key(row) for row in gold}
    for method, records in method_records.items():
        pred_keys = {candidate_key(row) for row in records}
        for relation in relation_types:
            relation_gold = {key for key in gold_keys if key[5] == relation}
            relation_pred = {key for key in pred_keys if key[5] == relation}
            tp = len(relation_gold & relation_pred)
            fp = len(relation_pred - relation_gold)
            fn = len(relation_gold - relation_pred)
            precision, recall, f1 = prf_counts(tp, fp, fn)
            by_relation_rows.append(
                {
                    "method": method,
                    "relation": relation,
                    "gold": len(relation_gold),
                    "predicted": len(relation_pred),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    pd.DataFrame(by_relation_rows).to_csv(
        OUTPUT_DIR / "confirmatory_by_relation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_rows: list[dict[str, Any]] = []
    for comparison_index, right_name in enumerate(
        [
            "fixed_charngram_comparator",
            "local_symbolic_text",
            "distance_trigger_heuristic",
            "deepseek_supported_or_plausible",
        ]
    ):
        difference, low, high, p_value = paired_bootstrap_difference(
            gold_by_sample,
            pred_by_method["cmfkg_frozen_hybrid"],
            pred_by_method[right_name],
            sample_ids,
            SEED + 100 + comparison_index,
        )
        paired_rows.append(
            {
                "left_method": "cmfkg_frozen_hybrid",
                "right_method": right_name,
                "f1_difference": difference,
                "ci95_low": low,
                "ci95_high": high,
                "paired_bootstrap_two_sided_p": p_value,
                "bootstrap_unit": "passage",
                "bootstrap_rounds": BOOTSTRAP_ROUNDS,
            }
        )
    pd.DataFrame(paired_rows).to_csv(
        OUTPUT_DIR / "confirmatory_paired_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(
        OUTPUT_DIR / "confirmatory_relation_experiment.xlsx",
        engine="openpyxl",
    ) as writer:
        metrics.to_excel(writer, sheet_name="00_metrics", index=False)
        pd.DataFrame(paired_rows).to_excel(
            writer,
            sheet_name="01_paired_bootstrap",
            index=False,
        )
        pd.DataFrame(by_relation_rows).to_excel(
            writer,
            sheet_name="02_by_relation",
            index=False,
        )

    best = metrics.iloc[0].to_dict()
    hybrid = metrics.loc[metrics["method"] == "cmfkg_frozen_hybrid"].iloc[0].to_dict()
    comparator = metrics.loc[
        metrics["method"] == "fixed_charngram_comparator"
    ].iloc[0].to_dict()
    summary = f"""# Confirmatory relation experiment

- Frozen confirmatory set: {len(sample_ids)} formal passages, {len(candidates)} schema-valid candidate pairs, {len(gold)} expert-adjudicated relations.
- Independent bootstrap unit: passage; {BOOTSTRAP_ROUNDS} resamples.
- Best method: `{best['method']}`, F1={best['f1']:.4f} (95% CI {best['f1_ci95_low']:.4f}-{best['f1_ci95_high']:.4f}).
- Frozen CMFKG hybrid: precision={hybrid['precision']:.4f}, recall={hybrid['recall']:.4f}, F1={hybrid['f1']:.4f} (95% CI {hybrid['f1_ci95_low']:.4f}-{hybrid['f1_ci95_high']:.4f}).
- Fixed character n-gram comparator: precision={comparator['precision']:.4f}, recall={comparator['recall']:.4f}, F1={comparator['f1']:.4f} (95% CI {comparator['f1_ci95_low']:.4f}-{comparator['f1_ci95_high']:.4f}).
- The prompt, model, thresholds and combination policy were frozen on the 50-passage development set before this evaluation.
"""
    (OUTPUT_DIR / "confirmatory_summary.md").write_text(summary, encoding="utf-8")

    manifest = {
        "status": "confirmatory_test_evaluated_once",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "independent_unit": "passage",
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "seed": SEED,
        "test_passages": len(sample_ids),
        "test_candidate_pairs": len(candidates),
        "test_gold_relations": len(gold),
        "local_freeze_sha256": sha256(CONFIG_DIR / "local_model_freeze.json"),
        "comparator_freeze_sha256": sha256(
            CONFIG_DIR / "fixed_comparator_freeze.json"
        ),
        "policy_freeze_sha256": sha256(CONFIG_DIR / "llm_policy_freeze.json"),
        "llm_test_manifest_sha256": sha256(
            EXPERIMENT_DIR / "llm_runs" / "test" / "run_manifest.json"
        ),
        "metrics_sha256": sha256(OUTPUT_DIR / "confirmatory_relation_metrics.csv"),
        "paired_bootstrap_sha256": sha256(
            OUTPUT_DIR / "confirmatory_paired_bootstrap.csv"
        ),
        "best_method": str(best["method"]),
        "best_f1": float(best["f1"]),
        "cmfkg_hybrid_f1": float(hybrid["f1"]),
        "fixed_comparator_f1": float(comparator["f1"]),
    }
    (OUTPUT_DIR / "confirmatory_evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
