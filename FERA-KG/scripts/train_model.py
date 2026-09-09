from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fera_kg.benchmark import BENCHMARK_DIR, GRAPH_PATH_DIR, load_graph_path_split, load_labeled_split  # noqa: E402
from fera_kg.deep_data import (  # noqa: E402
    SPECIAL_TOKENS,
    PathVocabulary,
    build_counterfactual_map,
    encode_graph,
    pack_evidence_and_paths,
    pack_evidence_window,
)
from fera_kg.deep_models import FERAKGModel, LABELS  # noqa: E402
from fera_kg.evaluation import full_relation_verification_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a development-only FERA-KG model family.")
    parser.add_argument(
        "--family",
        required=True,
        choices=[
            "text_only",
            "path_text",
            "path_text_hierarchical",
            "graph_only",
            "graph_hierarchical",
            "late_fusion",
            "output_cross_fusion",
            "no_ecrp",
            "fera_kg",
            "fera_flat",
            "graph_residual",
        ],
    )
    parser.add_argument("--checkpoint", default="SIKU-BERT/sikuroberta")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=BENCHMARK_DIR,
        help="Development protocol directory. It must contain train/development JSONL files.",
    )
    parser.add_argument(
        "--graph-path-dir",
        type=Path,
        default=None,
        help="Graph-path directory; defaults to <benchmark-dir>/graph_paths_v1.",
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-paths", type=int, default=16)
    parser.add_argument(
        "--linearized-paths",
        type=int,
        default=6,
        help="Number of evidence-ranked paths serialized into path-text inputs.",
    )
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument(
        "--text-pooling",
        choices=["cls", "mean", "learned_mix", "entity_pair"],
        default="cls",
    )
    parser.add_argument("--lr-text", type=float, default=2.0e-5)
    parser.add_argument("--lr-graph", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--cer-weight", type=float, default=0.50)
    parser.add_argument("--cer-margin", type=float, default=0.20)
    parser.add_argument(
        "--evidence-path-attention",
        action="store_true",
        help="Add a zero-initialized text-conditioned residual to path-attention scores.",
    )
    parser.add_argument(
        "--source-balance-power",
        type=float,
        default=0.0,
        help="Per-item source weight is proportional to source_count**(-power); 0 disables it.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=[
            "four_class_macro_f1",
            "three_decision_macro_f1",
            "source_unseen_three_decision_macro_f1",
            "harmonic_three_decision_macro_f1",
        ],
        default="four_class_macro_f1",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--smoke-items", type=int, default=0)
    parser.add_argument("--initialize-from", type=Path, default=None)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument(
        "--train-scope",
        choices=["all", "ecrp_only", "ecrp_head", "residual_only"],
        default="all",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision. Ignored on non-CUDA devices.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch

    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> Any:
    import torch

    requested = os.environ.get("FERA_DEVICE", "").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def graph_to_dict(encoded: Any) -> dict[str, Any]:
    return {
        "edge_ids": encoded.edge_ids,
        "node_type_ids": encoded.node_type_ids,
        "hop_mask": encoded.hop_mask,
        "path_mask": encoded.path_mask,
        "path_features": encoded.path_features,
        "graph_features": encoded.graph_features,
        "query_relation_id": encoded.query_relation_id,
        "head_type_id": encoded.head_type_id,
        "tail_type_id": encoded.tail_type_id,
    }


class DevelopmentDataset:
    def __init__(
        self,
        items: list[dict[str, Any]],
        path_items: list[dict[str, Any]],
        *,
        tokenizer: Any,
        vocabulary: PathVocabulary,
        max_length: int,
        max_paths: int,
        max_hops: int,
        family: str,
        linearized_paths: int,
        counterfactuals: Optional[dict[str, str]] = None,
        source_weights: Optional[dict[str, float]] = None,
    ) -> None:
        import torch

        if [row["item_id"] for row in items] != [row["item_id"] for row in path_items]:
            raise ValueError("benchmark and graph-path rows are not aligned")
        self.rows = []
        counterfactuals = counterfactuals or {}
        source_weights = source_weights or {}
        for item, path_item in zip(items, path_items):
            uses_linearized_paths = family in {"path_text", "path_text_hierarchical"}
            if uses_linearized_paths:
                text = pack_evidence_and_paths(
                    item,
                    path_item,
                    max_paths=linearized_paths,
                )
            else:
                text = pack_evidence_window(item)
            tokenized = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            counterfactual_passage = counterfactuals.get(str(item["item_id"]))
            if counterfactual_passage and uses_linearized_paths:
                cf_text = pack_evidence_and_paths(
                    item,
                    path_item,
                    passage_override=counterfactual_passage,
                    max_paths=linearized_paths,
                )
            elif counterfactual_passage:
                cf_text = pack_evidence_window(
                    item,
                    passage_override=counterfactual_passage,
                )
            else:
                cf_text = text
            cf_tokenized = tokenizer(
                cf_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            graph = encode_graph(
                item,
                path_item,
                vocabulary,
                max_paths=max_paths,
                max_hops=max_hops,
            )
            self.rows.append({
                "item_id": str(item["item_id"]),
                "passage_id": str(item["passage_id"]),
                "source_name": str(item["source_name"]),
                "relation_group": str(item["relation_group"]),
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "counterfactual_input_ids": cf_tokenized["input_ids"].squeeze(0),
                "counterfactual_attention_mask": cf_tokenized["attention_mask"].squeeze(0),
                "counterfactual_mask": torch.tensor(counterfactual_passage is not None, dtype=torch.bool),
                "label": torch.tensor(LABELS.index(str(item["label"])), dtype=torch.long),
                "sample_weight": torch.tensor(
                    float(source_weights.get(str(item["source_name"]), 1.0)),
                    dtype=torch.float32,
                ),
                "graph": graph_to_dict(graph),
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    graph_keys = list(rows[0]["graph"])
    graph = {}
    for key in graph_keys:
        values = [row["graph"][key] for row in rows]
        if key in {"hop_mask", "path_mask"}:
            graph[key] = torch.tensor(values, dtype=torch.bool)
        elif key in {"path_features", "graph_features"}:
            graph[key] = torch.tensor(values, dtype=torch.float32)
        else:
            graph[key] = torch.tensor(values, dtype=torch.long)
    return {
        "item_ids": [row["item_id"] for row in rows],
        "passage_ids": [row["passage_id"] for row in rows],
        "source_names": [row["source_name"] for row in rows],
        "relation_groups": [row["relation_group"] for row in rows],
        "input_ids": torch.stack([row["input_ids"] for row in rows]),
        "attention_mask": torch.stack([row["attention_mask"] for row in rows]),
        "counterfactual_input_ids": torch.stack([row["counterfactual_input_ids"] for row in rows]),
        "counterfactual_attention_mask": torch.stack([row["counterfactual_attention_mask"] for row in rows]),
        "counterfactual_mask": torch.stack([row["counterfactual_mask"] for row in rows]),
        "labels": torch.stack([row["label"] for row in rows]),
        "sample_weights": torch.stack([row["sample_weight"] for row in rows]),
        "graph": graph,
    }


def move_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    tensor_keys = [
        "input_ids",
        "attention_mask",
        "counterfactual_input_ids",
        "counterfactual_attention_mask",
        "counterfactual_mask",
        "labels",
        "sample_weights",
    ]
    moved = dict(batch)
    for key in tensor_keys:
        moved[key] = batch[key].to(device)
    moved["graph"] = {key: value.to(device) for key, value in batch["graph"].items()}
    return moved


def evaluate(
    model: Any,
    loader: Any,
    device: Any,
    *,
    amp_enabled: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    model.eval()
    gold, predicted, predictions = [], [], []
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
            probabilities = output["probabilities"].detach().float().cpu()
            labels = batch["labels"].detach().cpu()
            for item_id, passage_id, source_name, relation_group, label, probability in zip(
                batch["item_ids"],
                batch["passage_ids"],
                batch["source_names"],
                batch["relation_groups"],
                labels.tolist(),
                probabilities.tolist(),
            ):
                prediction = int(max(range(len(probability)), key=probability.__getitem__))
                gold.append(LABELS[label])
                predicted.append(LABELS[prediction])
                predictions.append({
                    "item_id": item_id,
                    "passage_id": passage_id,
                    "source_name": source_name,
                    "relation_group": relation_group,
                    "gold_label": LABELS[label],
                    "predicted_label": LABELS[prediction],
                    "probabilities": {name: value for name, value in zip(LABELS, probability)},
                    "confidence": max(probability),
                })
    return full_relation_verification_metrics(gold, predicted), predictions


def development_selection_score(
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    selection_metric: str,
    train_sources: set[str],
) -> tuple[float, dict[str, Any]]:
    overall_three = float(metrics["three_decision"]["macro_f1"])
    unseen_rows = [row for row in predictions if str(row["source_name"]) not in train_sources]
    if unseen_rows:
        unseen_metrics = full_relation_verification_metrics(
            [str(row["gold_label"]) for row in unseen_rows],
            [str(row["predicted_label"]) for row in unseen_rows],
        )
        unseen_three = float(unseen_metrics["three_decision"]["macro_f1"])
    else:
        unseen_metrics = None
        unseen_three = overall_three
    if selection_metric == "four_class_macro_f1":
        score = float(metrics["four_class"]["macro_f1"])
    elif selection_metric == "three_decision_macro_f1":
        score = overall_three
    elif selection_metric == "source_unseen_three_decision_macro_f1":
        score = unseen_three
    elif selection_metric == "harmonic_three_decision_macro_f1":
        score = (
            2.0 * overall_three * unseen_three / (overall_three + unseen_three)
            if overall_three + unseen_three > 0.0
            else 0.0
        )
    else:
        raise ValueError(f"unknown selection metric: {selection_metric}")
    return score, {
        "selection_metric": selection_metric,
        "selection_score": score,
        "overall_four_class_macro_f1": float(metrics["four_class"]["macro_f1"]),
        "overall_three_decision_macro_f1": overall_three,
        "source_unseen_items": len(unseen_rows),
        "source_unseen_three_decision_macro_f1": unseen_three,
        "source_unseen_metrics": unseen_metrics,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    device = select_device()
    amp_enabled = bool(args.amp and device.type == "cuda")
    benchmark_dir = args.benchmark_dir.resolve()
    graph_path_dir = (
        args.graph_path_dir.resolve()
        if args.graph_path_dir is not None
        else benchmark_dir / "graph_paths_v1"
    )
    train_items = load_labeled_split("train", benchmark_dir=benchmark_dir)
    development_items = load_labeled_split("development", benchmark_dir=benchmark_dir)
    train_paths = load_graph_path_split("train", graph_path_dir=graph_path_dir)
    development_paths = load_graph_path_split("development", graph_path_dir=graph_path_dir)
    if args.smoke_items:
        train_items, train_paths = train_items[:args.smoke_items], train_paths[:args.smoke_items]
        development_items, development_paths = (
            development_items[:max(8, args.smoke_items // 4)],
            development_paths[:max(8, args.smoke_items // 4)],
        )
    vocabulary = PathVocabulary.fit(train_items, train_paths)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    counterfactuals = (
        build_counterfactual_map(train_items)
        if args.family in {"fera_kg", "fera_flat"} and args.cer_weight > 0.0
        else {}
    )
    source_counts: dict[str, int] = {}
    for item in train_items:
        source_name = str(item["source_name"])
        source_counts[source_name] = source_counts.get(source_name, 0) + 1
    source_weights = {
        source_name: float(count) ** (-args.source_balance_power)
        for source_name, count in source_counts.items()
    }
    if source_weights:
        mean_item_weight = sum(
            source_weights[str(item["source_name"])] for item in train_items
        ) / len(train_items)
        source_weights = {
            source_name: value / mean_item_weight
            for source_name, value in source_weights.items()
        }
    train_dataset = DevelopmentDataset(
        train_items,
        train_paths,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        max_length=args.max_length,
        max_paths=args.max_paths,
        max_hops=args.max_hops,
        family=args.family,
        linearized_paths=args.linearized_paths,
        counterfactuals=counterfactuals,
        source_weights=source_weights,
    )
    development_dataset = DevelopmentDataset(
        development_items,
        development_paths,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        max_length=args.max_length,
        max_paths=args.max_paths,
        max_hops=args.max_hops,
        family=args.family,
        linearized_paths=args.linearized_paths,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        num_workers=0,
    )
    development_loader = DataLoader(
        development_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    counts = [sum(item["label"] == label for item in train_items) for label in LABELS]
    weights = torch.tensor(
        [math.sqrt(len(train_items) / (len(LABELS) * max(1, count))) for count in counts],
        dtype=torch.float32,
    )
    model = FERAKGModel(
        checkpoint=args.checkpoint,
        family=args.family,
        edge_vocab_size=len(vocabulary.edge_to_id),
        type_vocab_size=len(vocabulary.type_to_id),
        tokenizer_size=len(tokenizer),
        hidden_dim=args.hidden_dim,
        max_hops=args.max_hops,
        text_pooling=args.text_pooling,
        marker_token_ids={
            token: int(tokenizer.convert_tokens_to_ids(token))
            for token in ("[REL]", "[/REL]", "[H]", "[/H]", "[T]", "[/T]")
        },
        class_weights=weights,
        cer_margin=args.cer_margin,
        cer_weight=args.cer_weight,
        evidence_path_attention=args.evidence_path_attention,
    ).to(device)
    initialization: dict[str, Any] = {}
    if args.initialize_from is not None:
        source_state = torch.load(args.initialize_from, map_location="cpu")
        target_state = model.state_dict()
        compatible = {
            name: value
            for name, value in source_state.items()
            if name in target_state and tuple(value.shape) == tuple(target_state[name].shape)
        }
        incompatible = sorted(
            name
            for name, value in source_state.items()
            if name not in target_state or tuple(value.shape) != tuple(target_state[name].shape)
        )
        load_result = model.load_state_dict(compatible, strict=False)
        initialization = {
            "source": str(args.initialize_from),
            "source_sha256": sha256(args.initialize_from),
            "compatible_tensors": len(compatible),
            "source_incompatible_tensors": incompatible,
            "target_missing_tensors": sorted(load_result.missing_keys),
        }
    if args.train_scope in {"ecrp_only", "ecrp_head"}:
        if args.family != "fera_kg" or args.initialize_from is None:
            raise ValueError("restricted ECRP training requires a staged FERA-KG run")
        head_prefixes = (
            "fusion.",
            "judgeability_head.",
            "support_head.",
            "reason_head.",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (
                name.startswith("graph_reasoner.evidence_")
                or (args.train_scope == "ecrp_head" and name.startswith(head_prefixes))
            )
    elif args.train_scope == "residual_only":
        if args.family != "graph_residual" or args.initialize_from is None:
            raise ValueError(
                "residual-only training requires graph_residual initialized from graph_only"
            )
        residual_prefixes = (
            "text_encoder.",
            "entity_pair_pooler.",
            "residual_fusion.",
            "residual_gate_logit",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(residual_prefixes)
    teacher = None
    if args.anchor_weight > 0.0:
        if args.family != "fera_kg" or args.initialize_from is None:
            raise ValueError("anchored training requires a staged FERA-KG run")
        teacher = copy.deepcopy(model).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
    text_parameters, other_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (text_parameters if name.startswith("text_encoder.") else other_parameters).append(parameter)
    parameter_groups = []
    if other_parameters:
        parameter_groups.append({"params": other_parameters, "lr": args.lr_graph})
    if text_parameters:
        parameter_groups.append({"params": text_parameters, "lr": args.lr_text})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    update_steps = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = max(1, update_steps * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    run_name = args.run_name or f"{args.family}_s{args.seed}"
    run_dir = PROJECT_ROOT / "results" / "deep_development" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    vocabulary.save(run_dir / "path_vocabulary.json")
    tokenizer.save_pretrained(run_dir / "tokenizer")
    history, best_score, best_epoch, stale = [], -1.0, 0, 0
    best_selection: dict[str, Any] = {}
    train_sources = {str(item["source_name"]) for item in train_items}
    training_started = time.time()
    if args.initialize_from is not None:
        metrics, predictions = evaluate(
            model, development_loader, device, amp_enabled=amp_enabled
        )
        best_score, selection = development_selection_score(
            metrics,
            predictions,
            selection_metric=args.selection_metric,
            train_sources=train_sources,
        )
        best_selection = selection
        history.append({
            "epoch": 0,
            "train_loss": None,
            "development": metrics,
            "selection": selection,
            "elapsed_seconds": time.time() - training_started,
            "stage": "loaded_initialization",
        })
        torch.save(model.state_dict(), run_dir / "best_model.pt")
        write_json(run_dir / "best_development_metrics.json", metrics)
        with (run_dir / "best_development_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        print(
            f"epoch=0 initialized selection_score={best_score:.5f} "
            f"four_macro_f1={metrics['four_class']['macro_f1']:.5f} "
            f"three_macro_f1={metrics['three_decision']['macro_f1']:.5f} "
            f"abstain_recall={metrics['abstain_recall']:.5f}",
            flush=True,
        )
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, batches = 0.0, 0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_to_device(batch, device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp_enabled
            ):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    graph=batch["graph"],
                    labels=batch["labels"],
                    sample_weights=batch["sample_weights"],
                    counterfactual_input_ids=batch["counterfactual_input_ids"],
                    counterfactual_attention_mask=batch["counterfactual_attention_mask"],
                    counterfactual_mask=batch["counterfactual_mask"],
                )
                if teacher is not None:
                    with torch.no_grad():
                        teacher_probabilities = teacher(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            graph=batch["graph"],
                        )["probabilities"]
                    anchor_loss = torch.nn.functional.kl_div(
                        torch.log(output["probabilities"].clamp_min(1.0e-8)),
                        teacher_probabilities,
                        reduction="batchmean",
                    )
                    output["loss"] = output["loss"] + args.anchor_weight * anchor_loss
                    output["anchor_loss"] = anchor_loss.detach()
            loss = output["loss"] / args.grad_accum
            scaler.scale(loss).backward()
            running_loss += float(output["loss"].detach().float().cpu())
            batches += 1
            if batch_index % args.grad_accum == 0 or batch_index == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        metrics, predictions = evaluate(
            model, development_loader, device, amp_enabled=amp_enabled
        )
        score, selection = development_selection_score(
            metrics,
            predictions,
            selection_metric=args.selection_metric,
            train_sources=train_sources,
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, batches),
            "development": metrics,
            "selection": selection,
            "elapsed_seconds": time.time() - training_started,
        }
        history.append(row)
        write_json(run_dir / "history.json", history)
        print(
            f"epoch={epoch} loss={row['train_loss']:.5f} selection_score={score:.5f} "
            f"four_macro_f1={metrics['four_class']['macro_f1']:.5f} "
            f"three_macro_f1={metrics['three_decision']['macro_f1']:.5f} "
            f"abstain_recall={metrics['abstain_recall']:.5f}",
            flush=True,
        )
        if score > best_score + 1.0e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_selection = selection
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            write_json(run_dir / "best_development_metrics.json", metrics)
            with (run_dir / "best_development_predictions.jsonl").open("w", encoding="utf-8") as handle:
                for prediction in predictions:
                    handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        else:
            stale += 1
        if stale >= args.patience:
            break
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    config = {
        "status": "DEVELOPMENT_ONLY_TEST_LABELS_NOT_ACCESSED",
        "family": args.family,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "train_items": len(train_items),
        "development_items": len(development_items),
        "counterfactual_training_pairs": len(counterfactuals),
        "initialization": initialization,
        "best_epoch": best_epoch,
        "best_development_four_class_macro_f1": best_selection.get(
            "overall_four_class_macro_f1"
        ),
        "best_development_three_decision_macro_f1": best_selection.get(
            "overall_three_decision_macro_f1"
        ),
        "best_selection": best_selection,
        "source_balance_power": args.source_balance_power,
        "source_count": len(source_counts),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "elapsed_seconds": time.time() - training_started,
        "input_hashes": {
            "train": sha256(benchmark_dir / "train.jsonl"),
            "development": sha256(benchmark_dir / "development.jsonl"),
            "train_graph_paths": sha256(graph_path_dir / "train.jsonl"),
            "development_graph_paths": sha256(graph_path_dir / "development.jsonl"),
        },
        "test_labels_accessed": False,
    }
    write_json(run_dir / "run_config.json", config)
    write_json(PROJECT_ROOT / "manifests" / f"deep_development_{run_name}.json", {
        "run_directory": str(run_dir),
        "family": args.family,
        "best_epoch": best_epoch,
        "best_development_four_class_macro_f1": best_selection.get(
            "overall_four_class_macro_f1"
        ),
        "best_development_three_decision_macro_f1": best_selection.get(
            "overall_three_decision_macro_f1"
        ),
        "selection_metric": args.selection_metric,
        "best_selection_score": best_score,
        "test_labels_accessed": False,
    })
    print(
        json.dumps(
            {
                "run": run_name,
                "best_epoch": best_epoch,
                "selection_metric": args.selection_metric,
                "best_selection_score": best_score,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
