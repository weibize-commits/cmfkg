from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "paper_revision" / "relation_improvement"
DATA_DIR = EXPERIMENT_DIR / "data"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
MODEL_DIR = EXPERIMENT_DIR / "models"
CONFIG_DIR = EXPERIMENT_DIR / "configs"
PREPARE_PATH = EXPERIMENT_DIR / "analysis" / "prepare_and_tune_local.py"
BUILDER_PATH = ROOT / "paper_revision" / "tools" / "build_relation_baseline_package.py"


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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
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


def candidate_digest(candidate: dict[str, Any]) -> str:
    def clean(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    head = candidate["head"]
    tail = candidate["tail"]
    key = (
        str(candidate["sample_id"]),
        int(head["start_char"]),
        int(head["end_char"]),
        clean(head["text"]),
        str(head["type"]),
        str(candidate["relation"]),
        int(tail["start_char"]),
        int(tail["end_char"]),
        clean(tail["text"]),
        str(tail["type"]),
    )
    payload = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()[:20]


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return float(precision), float(recall), float(f1)


def llm_support_score(decision: dict[str, Any]) -> float:
    label = str(decision.get("label") or "unsupported")
    confidence = float(decision.get("confidence") or 0.0)
    if label == "supported":
        return max(0.75, confidence)
    if label == "plausible":
        return 0.55 + 0.25 * confidence
    return 0.15 * (1.0 - confidence)


def rank(metrics: tuple[float, float, float]) -> tuple[float, float, float]:
    precision, recall, f1 = metrics
    return f1, recall, precision


def main() -> None:
    builder = load_module("relation_builder_policy", BUILDER_PATH)
    prepare = load_module("relation_prepare_policy", PREPARE_PATH)

    candidates = read_jsonl(DATA_DIR / "dev_candidates.jsonl")
    gold = read_jsonl(DATA_DIR / "dev_gold.jsonl")
    passages = read_jsonl(DATA_DIR / "dev_passages.jsonl")
    decisions = read_jsonl(OUTPUT_DIR / "deepseek_triage_dev_decisions.jsonl")
    manifest = json.loads(
        (EXPERIMENT_DIR / "llm_runs" / "dev" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    local_freeze = json.loads(
        (CONFIG_DIR / "local_model_freeze.json").read_text(encoding="utf-8")
    )
    if manifest.get("status") != "complete":
        raise RuntimeError("Development LLM run is incomplete")

    decision_by_digest = {
        str(record["candidate_digest"]): record for record in decisions
    }
    if len(decision_by_digest) != len(candidates):
        raise RuntimeError(
            f"Expected {len(candidates)} LLM decisions, found {len(decision_by_digest)}"
        )

    texts = {
        str(record["sample_id"]): {"frozen_text": str(record["frozen_text"])}
        for record in passages
    }
    gold_keys = {builder.relation_key(record) for record in gold}
    y_true = np.array(
        [builder.relation_key(candidate) in gold_keys for candidate in candidates],
        dtype=bool,
    )
    with (MODEL_DIR / "local_symbolic_text_model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    local_features = [
        prepare.rich_feature_text(builder, candidate, texts) for candidate in candidates
    ]
    local_scores = model.predict_proba(local_features)[:, 1]
    llm_scores = np.array(
        [
            llm_support_score(decision_by_digest[candidate_digest(candidate)])
            for candidate in candidates
        ],
        dtype=float,
    )
    heuristic_scores = np.array(
        [
            float(builder.heuristic_score(candidate, texts)[1])
            for candidate in candidates
        ],
        dtype=float,
    )
    labels = np.array(
        [
            decision_by_digest[candidate_digest(candidate)]["label"]
            for candidate in candidates
        ],
        dtype=object,
    )

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    def consider(
        family: str,
        parameters: dict[str, Any],
        predictions: np.ndarray,
        combined_score: np.ndarray,
    ) -> None:
        nonlocal best
        metrics = prf(y_true, predictions)
        row = {
            "family": family,
            **parameters,
            "precision": metrics[0],
            "recall": metrics[1],
            "f1": metrics[2],
        }
        rows.append(row)
        candidate = {
            "rank": rank(metrics),
            "family": family,
            "parameters": parameters,
            "metrics": metrics,
            "predictions": predictions,
            "combined_score": combined_score,
        }
        if best is None or candidate["rank"] > best["rank"]:
            best = candidate

    for threshold in np.arange(0.05, 0.951, 0.025):
        consider(
            "local_only",
            {"threshold": float(round(threshold, 3))},
            local_scores >= threshold,
            local_scores,
        )

    frozen_relation_thresholds = {
        str(relation): float(threshold)
        for relation, threshold in local_freeze["relation_thresholds"].items()
    }
    per_candidate_thresholds = np.array(
        [
            frozen_relation_thresholds.get(
                str(candidate["relation"]),
                float(local_freeze["selected_local_model"]["global_threshold"]),
            )
            for candidate in candidates
        ],
        dtype=float,
    )
    relation_base = local_scores >= per_candidate_thresholds
    consider(
        "local_relation_thresholds",
        {
            "threshold_source": "local_model_freeze.json",
        },
        relation_base,
        local_scores,
    )

    for rescue_margin in np.arange(0.05, 0.301, 0.05):
        for accept_plausible in [False, True]:
            llm_accept = labels == "supported"
            if accept_plausible:
                llm_accept = np.isin(labels, ["supported", "plausible"])
            for veto_margin in [None, 0.0, 0.05, 0.10]:
                predictions = relation_base.copy()
                rescue = (
                    (~relation_base)
                    & llm_accept
                    & (local_scores >= np.maximum(0.0, per_candidate_thresholds - rescue_margin))
                )
                predictions = predictions | rescue
                if veto_margin is not None:
                    veto = (
                        relation_base
                        & (labels == "unsupported")
                        & (local_scores < per_candidate_thresholds + veto_margin)
                    )
                    predictions = predictions & (~veto)
                consider(
                    "relation_threshold_llm_rescue_veto",
                    {
                        "threshold_source": "local_model_freeze.json",
                        "rescue_margin": float(round(rescue_margin, 2)),
                        "accept_plausible": accept_plausible,
                        "veto_margin": veto_margin,
                    },
                    predictions,
                    local_scores,
                )

    weight_values = np.arange(0.0, 1.01, 0.1)
    for local_weight in weight_values:
        for llm_weight in weight_values:
            heuristic_weight = 1.0 - local_weight - llm_weight
            if heuristic_weight < -1e-9:
                continue
            heuristic_weight = max(0.0, heuristic_weight)
            combined = (
                local_weight * local_scores
                + llm_weight * llm_scores
                + heuristic_weight * heuristic_scores
            )
            for threshold in np.arange(0.1, 0.901, 0.025):
                consider(
                    "weighted_score",
                    {
                        "local_weight": float(round(local_weight, 2)),
                        "llm_weight": float(round(llm_weight, 2)),
                        "heuristic_weight": float(round(heuristic_weight, 2)),
                        "threshold": float(round(threshold, 3)),
                    },
                    combined >= threshold,
                    combined,
                )

    for lower in np.arange(0.1, 0.601, 0.05):
        for upper in np.arange(max(0.5, lower + 0.1), 0.951, 0.05):
            for accept_plausible in [False, True]:
                llm_accept = labels == "supported"
                if accept_plausible:
                    llm_accept = np.isin(labels, ["supported", "plausible"])
                predictions = (local_scores >= upper) | (
                    (local_scores >= lower) & llm_accept
                )
                consider(
                    "protected_local_gate",
                    {
                        "lower": float(round(lower, 2)),
                        "upper": float(round(upper, 2)),
                        "accept_plausible": accept_plausible,
                    },
                    predictions,
                    local_scores,
                )

    assert best is not None
    search = pd.DataFrame(rows).sort_values(
        ["f1", "recall", "precision"],
        ascending=False,
    )
    search.to_csv(
        OUTPUT_DIR / "llm_policy_dev_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction_records: list[dict[str, Any]] = []
    for candidate, predicted, score in zip(
        candidates,
        best["predictions"],
        best["combined_score"],
    ):
        if not predicted:
            continue
        prediction_records.append(
            builder.prediction_record(
                candidate,
                "cmfkg_hybrid_dev",
                float(score),
                f"{best['family']}:{json.dumps(best['parameters'], sort_keys=True)}",
            )
        )
    write_jsonl(OUTPUT_DIR / "cmfkg_hybrid_dev_predictions.jsonl", prediction_records)

    freeze = {
        "status": "llm_policy_frozen_before_confirmatory_test_evaluation",
        "independent_unit": "passage",
        "development_passages": len(passages),
        "development_candidates": len(candidates),
        "development_gold_relations": len(gold),
        "selected_family": best["family"],
        "selected_parameters": best["parameters"],
        "development_precision": best["metrics"][0],
        "development_recall": best["metrics"][1],
        "development_f1": best["metrics"][2],
        "llm_prompt_version": manifest["prompt_version"],
        "llm_prompt_sha256": manifest["prompt_sha256"],
        "llm_dev_decisions_sha256": sha256(
            OUTPUT_DIR / "deepseek_triage_dev_decisions.jsonl"
        ),
        "local_model_sha256": sha256(MODEL_DIR / "local_symbolic_text_model.pkl"),
        "policy_search_sha256": sha256(OUTPUT_DIR / "llm_policy_dev_search.csv"),
        "confirmatory_test_evaluated": False,
        "next_step": (
            "Run the already frozen prompt on the 65-passage confirmatory split, "
            "apply this policy without modification, and evaluate once."
        ),
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "llm_policy_freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(freeze, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
