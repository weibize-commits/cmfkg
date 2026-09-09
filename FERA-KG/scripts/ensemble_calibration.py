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

from fera_kg.benchmark import (  # noqa: E402
    BENCHMARK_DIR,
    GRAPH_PATH_DIR,
    load_graph_path_split,
    load_labeled_split,
)
from fera_kg.deep_data import PathVocabulary, SPECIAL_TOKENS  # noqa: E402
from fera_kg.deep_models import FERAKGModel, LABELS  # noqa: E402
from fera_kg.evaluation import full_relation_verification_metrics  # noqa: E402
from train_model import (  # noqa: E402
    DevelopmentDataset,
    collate,
    move_to_device,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a frozen five-seed deep ensemble.")
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260829, 20260830, 20260831, 20260832, 20260833])
    parser.add_argument("--split", default="calibration", choices=["calibration"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection-metric",
        choices=["four_class_macro_f1", "three_decision_macro_f1"],
        default="four_class_macro_f1",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def apply_temperature(probabilities: list[list[float]], temperature: float) -> list[list[float]]:
    return [
        softmax([math.log(max(value, 1.0e-12)) / temperature for value in row])
        for row in probabilities
    ]


def nll(probabilities: list[list[float]], gold: list[str]) -> float:
    label_index = {label: index for index, label in enumerate(LABELS)}
    return -sum(
        math.log(max(row[label_index[label]], 1.0e-12))
        for row, label in zip(probabilities, gold)
    ) / len(gold)


def threshold_predictions(
    probabilities: list[list[float]],
    judgeability_threshold: float,
    support_threshold: float,
) -> list[str]:
    predictions = []
    for row in probabilities:
        judgeability = row[0] + row[1]
        if judgeability >= judgeability_threshold:
            conditional_support = row[0] / max(judgeability, 1.0e-12)
            predictions.append("SUPPORTED" if conditional_support >= support_threshold else "NOT_SUPPORTED")
        else:
            abstain = row[2] + row[3]
            insufficient = row[2] / max(abstain, 1.0e-12)
            predictions.append("INSUFFICIENT_CONTEXT" if insufficient >= 0.5 else "ENTITY_OR_TYPE_ERROR")
    return predictions


def load_run_probabilities(
    run_dir: Path,
    items: list[dict[str, Any]],
    paths: list[dict[str, Any]],
    device: Any,
    batch_size: int,
) -> tuple[list[list[float]], dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    arguments = config["arguments"]
    tokenizer_dir = run_dir / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    else:
        # Archived confirmatory checkpoints retain the frozen run configuration,
        # path vocabulary and state dictionary. Recreate the training tokenizer
        # deterministically from the recorded base checkpoint and the same ordered
        # special-token list when the redundant tokenizer copy is not archived.
        tokenizer = AutoTokenizer.from_pretrained(str(arguments["checkpoint"]), use_fast=True)
        tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    vocabulary = PathVocabulary.from_dict(
        json.loads((run_dir / "path_vocabulary.json").read_text(encoding="utf-8"))
    )
    dataset = DevelopmentDataset(
        items,
        paths,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        max_length=int(arguments["max_length"]),
        max_paths=int(arguments["max_paths"]),
        max_hops=int(arguments.get("max_hops", 3)),
        family=str(arguments["family"]),
        linearized_paths=int(arguments.get("linearized_paths", 6)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    model = FERAKGModel(
        checkpoint=str(arguments["checkpoint"]),
        family=str(arguments["family"]),
        edge_vocab_size=len(vocabulary.edge_to_id),
        type_vocab_size=len(vocabulary.type_to_id),
        tokenizer_size=len(tokenizer),
        hidden_dim=int(arguments["hidden_dim"]),
        max_hops=int(arguments.get("max_hops", 3)),
        text_pooling=str(arguments["text_pooling"]),
        marker_token_ids={
            token: int(tokenizer.convert_tokens_to_ids(token))
            for token in ("[REL]", "[/REL]", "[H]", "[/H]", "[T]", "[/T]")
        },
        cer_margin=float(arguments.get("cer_margin", 0.2)),
        cer_weight=float(arguments.get("cer_weight", 0.0)),
        evidence_path_attention=bool(arguments.get("evidence_path_attention", False)),
    ).to(device)
    checkpoint = run_dir / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model.eval()
    amp_enabled = bool(arguments.get("amp", False) and device.type == "cuda")
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp_enabled
            ):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    graph=batch["graph"],
                )
            probabilities.extend(output["probabilities"].detach().float().cpu().tolist())
    metadata = {
        "run": run_dir.name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "run_config_sha256": sha256(run_dir / "run_config.json"),
    }
    del model
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probabilities, metadata


def main() -> None:
    args = parse_args()
    device = select_device()
    items = load_labeled_split(args.split)
    paths = load_graph_path_split(args.split)
    gold = [str(item["label"]) for item in items]
    runs, all_probabilities = [], []
    for seed in args.seeds:
        run_dir = PROJECT_ROOT / "results" / "deep_development" / f"{args.run_prefix}-s{seed}"
        probabilities, metadata = load_run_probabilities(
            run_dir, items, paths, device, args.batch_size
        )
        all_probabilities.append(probabilities)
        runs.append(metadata)
    ensemble = [
        [
            sum(seed_probabilities[item_index][label_index] for seed_probabilities in all_probabilities)
            / len(all_probabilities)
            for label_index in range(len(LABELS))
        ]
        for item_index in range(len(items))
    ]
    uncalibrated_predictions = [LABELS[max(range(len(row)), key=row.__getitem__)] for row in ensemble]
    uncalibrated_metrics = full_relation_verification_metrics(gold, uncalibrated_predictions)

    temperature_candidates = [0.50 + 0.05 * index for index in range(51)]
    temperature = min(temperature_candidates, key=lambda value: nll(apply_temperature(ensemble, value), gold))
    calibrated = apply_temperature(ensemble, temperature)
    threshold_candidates = [0.30 + 0.02 * index for index in range(21)]
    grid = []
    for judgeability_threshold in threshold_candidates:
        for support_threshold in threshold_candidates:
            predicted = threshold_predictions(calibrated, judgeability_threshold, support_threshold)
            metrics = full_relation_verification_metrics(gold, predicted)
            grid.append((
                metrics["four_class"]["macro_f1"],
                metrics["three_decision"]["macro_f1"],
                metrics["four_class"]["accuracy"],
                -abs(judgeability_threshold - 0.5) - abs(support_threshold - 0.5),
                judgeability_threshold,
                support_threshold,
                metrics,
            ))
    if args.selection_metric == "four_class_macro_f1":
        selected = max(grid, key=lambda row: row[:4])
    else:
        selected = max(
            grid,
            key=lambda row: (
                row[1],
                row[0],
                row[2],
                row[3],
            ),
        )
    payload = {
        "status": "CALIBRATION_ONLY_TEST_LABELS_NOT_ACCESSED",
        "run_prefix": args.run_prefix,
        "seeds": args.seeds,
        "runs": runs,
        "split": args.split,
        "items": len(items),
        "temperature": temperature,
        "judgeability_threshold": selected[4],
        "support_threshold": selected[5],
        "reason_threshold": 0.5,
        "selection_metric": args.selection_metric,
        "uncalibrated_nll": nll(ensemble, gold),
        "calibrated_nll": nll(calibrated, gold),
        "uncalibrated_metrics": uncalibrated_metrics,
        "selected_threshold_metrics": selected[6],
        "input_hashes": {
            "calibration": sha256(BENCHMARK_DIR / "calibration.jsonl"),
            "calibration_graph_paths": sha256(GRAPH_PATH_DIR / "calibration.jsonl"),
        },
        "test_labels_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "temperature": temperature,
        "judgeability_threshold": selected[4],
        "support_threshold": selected[5],
        "uncalibrated_four_macro_f1": uncalibrated_metrics["four_class"]["macro_f1"],
        "calibrated_four_macro_f1": selected[6]["four_class"]["macro_f1"],
    }))


if __name__ == "__main__":
    main()
