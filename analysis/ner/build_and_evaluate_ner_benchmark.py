from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DOWNLOADS = Path(os.environ["USERPROFILE"]) / "Downloads"
BENCH_ROOT = ROOT / "paper_revision" / "ner_benchmark"
INPUT_DIR = BENCH_ROOT / "inputs"
TEST_DIR = BENCH_ROOT / "test_set"
OUTPUT_DIR = BENCH_ROOT / "outputs"

SOURCE_FILES = {
    "Dr1.xlsx": DOWNLOADS / "Dr1.xlsx",
    "Dr2.xlsx": DOWNLOADS / "Dr2.xlsx",
    "Dr3.xlsx": DOWNLOADS / "Dr3.xlsx",
    "gold_standard.xlsx": DOWNLOADS / "gold_standard.xlsx",
}

ENTITY_LABELS = [
    "食材",
    "食性",
    "食味",
    "归经",
    "功效",
    "病症",
    "脏腑",
    "食疗方",
    "烹饪方法",
    "禁忌",
    "养生原则",
    "医家",
    "古籍来源",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def read_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    df["record_id"] = df["记录ID"].map(norm_id)
    return df


def read_entities(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=1, engine="openpyxl")
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    rename = {
        "记录ID": "record_id",
        "记录id": "record_id",
        "实体序号": "entity_index",
        "实体类型": "label",
        "标签": "label",
        "起始位置": "start",
        "结束位置": "end",
        "实体文本": "text",
        "文本": "text",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["record_id"] = df["record_id"].map(norm_id)
    if "start" in df.columns:
        df["start"] = pd.to_numeric(df["start"], errors="coerce").astype("Int64")
    if "end" in df.columns:
        df["end"] = pd.to_numeric(df["end"], errors="coerce").astype("Int64")
    return df


def load_gold(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = read_data(path)
    entities = read_entities(path)
    entities = entities[entities["label"].notna()].copy()
    entities["label"] = entities["label"].astype(str).str.strip()
    entities["text"] = entities["text"].astype(str)
    entities = entities[entities["label"].isin(ENTITY_LABELS)]
    return data, entities


def entity_set(df: pd.DataFrame, require_offsets: bool = True) -> set[tuple[Any, ...]]:
    if require_offsets:
        keep = df.dropna(subset=["record_id", "label", "start", "end"])
        return {
            (str(r.record_id), int(r.start), int(r.end), str(r.label))
            for r in keep.itertuples(index=False)
        }
    keep = df.dropna(subset=["record_id", "label", "text"])
    return {
        (str(r.record_id), str(r.label), str(r.text))
        for r in keep.itertuples(index=False)
    }


def prf(pred: set[tuple[Any, ...]], gold: set[tuple[Any, ...]]) -> dict[str, Any]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def split_records(data: pd.DataFrame, seed: int = 20260731) -> dict[str, list[str]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in data.itertuples(index=False):
        groups[(str(row.难度), str(row.类别))].append(str(row.record_id))
    split = {"train": [], "dev": [], "test": []}
    for _, ids in sorted(groups.items()):
        ids = sorted(ids, key=lambda x: int(x) if x.isdigit() else x)
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * 0.20))
        n_dev = max(1, round(n * 0.10))
        split["test"].extend(ids[:n_test])
        split["dev"].extend(ids[n_test : n_test + n_dev])
        split["train"].extend(ids[n_test + n_dev :])
    for key in split:
        split[key] = sorted(split[key], key=lambda x: int(x) if x.isdigit() else x)
    return split


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bio_tags(text: str, ents: pd.DataFrame) -> list[str]:
    tags = ["O"] * len(text)
    spans = []
    for r in ents.dropna(subset=["start", "end", "label"]).itertuples(index=False):
        start = int(r.start)
        end = int(r.end)
        label = str(r.label)
        if start < 0 or end > len(text) or start >= end:
            continue
        spans.append((start, end, label))
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    occupied = [False] * len(text)
    for start, end, label in spans:
        if any(occupied[start:end]):
            continue
        tags[start] = f"B-{label}"
        for i in range(start + 1, end):
            tags[i] = f"I-{label}"
        for i in range(start, end):
            occupied[i] = True
    return tags


def write_bio(path: Path, data: pd.DataFrame, entities: pd.DataFrame, ids: list[str]) -> None:
    ent_by_id = {rid: g for rid, g in entities.groupby("record_id")}
    rows = data.set_index("record_id")
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rid in ids:
            text = str(rows.loc[rid, "正文"])
            empty_entities = pd.DataFrame(columns=["start", "end", "label"])
            tags = bio_tags(text, ent_by_id.get(rid, empty_entities))
            for ch, tag in zip(text, tags):
                if ch in {"\n", "\r"}:
                    ch = " "
                f.write(f"{ch}\t{tag}\n")
            f.write("\n")


def build_lexicon(train_entities: pd.DataFrame) -> list[dict[str, Any]]:
    counts = Counter(
        (str(r.text), str(r.label))
        for r in train_entities.dropna(subset=["text", "label"]).itertuples(index=False)
        if str(r.text).strip()
    )
    items = [
        {"text": text, "label": label, "count": count, "length": len(text)}
        for (text, label), count in counts.items()
        if len(text) >= 1
    ]
    items.sort(key=lambda x: (-x["length"], -x["count"], x["text"], x["label"]))
    return items


def longest_match_predict(text: str, lexicon: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for item in lexicon:
        token = item["text"]
        start = 0
        while True:
            idx = text.find(token, start)
            if idx < 0:
                break
            candidates.append(
                {
                    "start": idx,
                    "end": idx + len(token),
                    "label": item["label"],
                    "text": token,
                    "count": item["count"],
                    "length": item["length"],
                }
            )
            start = idx + 1
    candidates.sort(key=lambda x: (-x["length"], x["start"], -x["count"], x["label"]))
    occupied = [False] * len(text)
    kept = []
    for cand in candidates:
        if any(occupied[cand["start"] : cand["end"]]):
            continue
        kept.append(cand)
        for i in range(cand["start"], cand["end"]):
            occupied[i] = True
    kept.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    return kept


def filter_lexicon(
    lexicon: list[dict[str, Any]],
    min_count: int,
    min_length: int,
    allow_single_char_labels: set[str],
) -> list[dict[str, Any]]:
    return [
        item
        for item in lexicon
        if item["count"] >= min_count
        and (item["length"] >= min_length or item["label"] in allow_single_char_labels)
    ]


def predict_lexicon_dataframe(
    data: pd.DataFrame,
    split_ids: list[str],
    lexicon: list[dict[str, Any]],
    method: str,
) -> pd.DataFrame:
    rows = data.set_index("record_id")
    pred_rows = []
    for rid in split_ids:
        text = str(rows.loc[rid, "正文"])
        for pred in longest_match_predict(text, lexicon):
            pred_rows.append(
                {
                    "record_id": rid,
                    "start": pred["start"],
                    "end": pred["end"],
                    "label": pred["label"],
                    "text": pred["text"],
                    "method": method,
                }
            )
    return pd.DataFrame(pred_rows, columns=["record_id", "start", "end", "label", "text", "method"])


def lexicon_metrics(
    pred_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    n_passages: int,
    method: str,
) -> list[dict[str, Any]]:
    overall = prf(entity_set(pred_df), entity_set(gold_df))
    metrics = [
        {
            "method": method,
            "label": "micro_all",
            "n_test_passages": n_passages,
            "gold_entities": len(gold_df),
            "predicted_entities": len(pred_df),
            **overall,
        }
    ]
    for label in ENTITY_LABELS:
        pred_label = pred_df[pred_df["label"].eq(label)]
        gold_label = gold_df[gold_df["label"].eq(label)]
        row = prf(entity_set(pred_label), entity_set(gold_label))
        metrics.append(
            {
                "method": method,
                "label": label,
                "n_test_passages": n_passages,
                "gold_entities": len(gold_label),
                "predicted_entities": len(pred_label),
                **row,
            }
        )
    return metrics


def evaluate_lexicon(
    data: pd.DataFrame,
    gold_entities: pd.DataFrame,
    split: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_entities = gold_entities[gold_entities["record_id"].isin(split["train"])].copy()
    lexicon = build_lexicon(train_entities)

    pred_df = predict_lexicon_dataframe(
        data,
        split["test"],
        lexicon,
        "train_lexicon_longest_match",
    )
    gold_test = gold_entities[gold_entities["record_id"].isin(split["test"])].copy()

    metrics = lexicon_metrics(
        pred_df,
        gold_test,
        len(split["test"]),
        "train_lexicon_longest_match",
    )

    gold_dev = gold_entities[gold_entities["record_id"].isin(split["dev"])].copy()
    single_char_options = [
        set(),
        {ENTITY_LABELS[1], ENTITY_LABELS[2]},
        {ENTITY_LABELS[1], ENTITY_LABELS[2], ENTITY_LABELS[3], ENTITY_LABELS[6]},
    ]
    tuning_rows = []
    for min_count in [1, 2, 3]:
        for min_length in [1, 2, 3]:
            for option_index, allow_single in enumerate(single_char_options):
                tuned_lexicon = filter_lexicon(lexicon, min_count, min_length, allow_single)
                dev_pred = predict_lexicon_dataframe(
                    data,
                    split["dev"],
                    tuned_lexicon,
                    "tuned_train_lexicon_longest_match",
                )
                dev_score = prf(entity_set(dev_pred), entity_set(gold_dev))
                tuning_rows.append(
                    {
                        "method": "tuned_train_lexicon_longest_match",
                        "min_count": min_count,
                        "min_length": min_length,
                        "single_char_option": option_index,
                        "single_char_labels": ";".join(sorted(allow_single)),
                        "lexicon_size": len(tuned_lexicon),
                        "dev_predicted_entities": len(dev_pred),
                        **dev_score,
                    }
                )
    tuning = pd.DataFrame(tuning_rows).sort_values(
        ["f1", "precision", "recall", "lexicon_size"],
        ascending=[False, False, False, True],
    )
    best = tuning.iloc[0]
    best_single = set(str(best["single_char_labels"]).split(";")) if str(best["single_char_labels"]) else set()
    best_lexicon = filter_lexicon(
        lexicon,
        int(best["min_count"]),
        int(best["min_length"]),
        best_single,
    )
    tuned_pred_df = predict_lexicon_dataframe(
        data,
        split["test"],
        best_lexicon,
        "tuned_train_lexicon_longest_match",
    )
    metrics.extend(
        lexicon_metrics(
            tuned_pred_df,
            gold_test,
            len(split["test"]),
            "tuned_train_lexicon_longest_match",
        )
    )
    return pd.DataFrame(metrics), pred_df, tuned_pred_df, tuning


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCE_FILES.items():
        shutil.copy2(source, INPUT_DIR / name)

    gold_data, gold_entities = load_gold(INPUT_DIR / "gold_standard.xlsx")
    dr1_data, dr1_entities = load_gold(INPUT_DIR / "Dr1.xlsx")
    dr2_data, dr2_entities = load_gold(INPUT_DIR / "Dr2.xlsx")
    dr3_entities = read_entities(INPUT_DIR / "Dr3.xlsx")

    split = split_records(gold_data, args.seed)
    split_records_rows = [
        {"record_id": rid, "split": split_name}
        for split_name, ids in split.items()
        for rid in ids
    ]
    pd.DataFrame(split_records_rows).to_csv(TEST_DIR / "ner_split_763.csv", index=False, encoding="utf-8-sig")

    data_rows = []
    data_by_id = gold_data.set_index("record_id")
    for split_name, ids in split.items():
        for rid in ids:
            row = data_by_id.loc[rid]
            data_rows.append(
                {
                    "record_id": rid,
                    "split": split_name,
                    "text": row["正文"],
                    "difficulty": row["难度"],
                    "category": row["类别"],
                    "book_name": row.get("书名", ""),
                    "entity_count": int(row.get("实体数量", 0) or 0),
                }
            )
    write_jsonl(TEST_DIR / "frozen_passages_763.jsonl", data_rows)

    entity_rows = []
    split_lookup = {rid: s for s, ids in split.items() for rid in ids}
    for r in gold_entities.itertuples(index=False):
        entity_rows.append(
            {
                "record_id": str(r.record_id),
                "split": split_lookup.get(str(r.record_id)),
                "start": int(r.start),
                "end": int(r.end),
                "label": str(r.label),
                "text": str(r.text),
            }
        )
    write_jsonl(TEST_DIR / "gold_ner_entities_763.jsonl", entity_rows)

    for split_name, ids in split.items():
        write_bio(TEST_DIR / f"{split_name}.bio", gold_data, gold_entities, ids)
    (TEST_DIR / "labels.txt").write_text(
        "O\n"
        + "\n".join(f"B-{x}" for x in ENTITY_LABELS)
        + "\n"
        + "\n".join(f"I-{x}" for x in ENTITY_LABELS)
        + "\n",
        encoding="utf-8",
    )

    agreement_rows = []
    gold_span = entity_set(gold_entities)
    for name, ents in [
        ("Dr1", dr1_entities),
        ("Dr2", dr2_entities),
    ]:
        exact = prf(entity_set(ents), gold_span)
        text_label = prf(entity_set(ents, require_offsets=False), entity_set(gold_entities, require_offsets=False))
        agreement_rows.append({"source": name, "comparison": "exact_span_label_vs_gold", **exact})
        agreement_rows.append({"source": name, "comparison": "text_label_vs_gold", **text_label})
    overlap = prf(entity_set(dr1_entities), entity_set(dr2_entities))
    agreement_rows.append({"source": "Dr1_vs_Dr2", "comparison": "exact_span_label", **overlap})
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(OUTPUT_DIR / "human_annotation_agreement.csv", index=False, encoding="utf-8-sig")

    metrics, pred_df, tuned_pred_df, tuning_df = evaluate_lexicon(gold_data, gold_entities, split)
    metrics.to_csv(OUTPUT_DIR / "ner_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_json(
        OUTPUT_DIR / "train_lexicon_longest_match_predictions.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    tuned_pred_df.to_json(
        OUTPUT_DIR / "tuned_train_lexicon_longest_match_predictions.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    tuning_df.to_csv(OUTPUT_DIR / "lexicon_tuning_grid.csv", index=False, encoding="utf-8-sig")

    inventory = {
        "version": "ner_human_gold_763_seed_20260731",
        "source_files": {name: sha256(INPUT_DIR / name) for name in SOURCE_FILES},
        "n_passages": int(gold_data.shape[0]),
        "n_gold_entities": int(gold_entities.shape[0]),
        "split_counts": {k: len(v) for k, v in split.items()},
        "label_counts": dict(Counter(gold_entities["label"])),
        "dr3_text_only_adjudication_rows": int(dr3_entities.shape[0]),
        "outputs": {
            str(p.relative_to(BENCH_ROOT)): sha256(p)
            for p in sorted([*TEST_DIR.iterdir(), *OUTPUT_DIR.iterdir()])
            if p.is_file() and p.name != "ner_benchmark_manifest.json"
        },
        "notes": [
            "Gold standard is read from gold_standard.xlsx with exact offsets.",
            "Dr3.xlsx contains adjudicated label/text rows without offsets and is used for inventory, not exact-span evaluation.",
            "The default baseline is trained only on the train split and evaluated on the frozen test split.",
        ],
    }
    (OUTPUT_DIR / "ner_benchmark_manifest.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
