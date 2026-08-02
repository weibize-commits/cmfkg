from __future__ import annotations

import csv
import gzip
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(relative: str) -> list[dict]:
    path = ROOT / relative
    opener = gzip.open if path.name.endswith(".gz") else path.open
    kwargs = {"mode": "rt", "encoding": "utf-8-sig"}
    with opener(**kwargs) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hashes() -> int:
    checked = 0
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if sha256(path) != expected:
            raise AssertionError(f"SHA-256 mismatch: {relative}")
        checked += 1
    return checked


def verify_structured_files() -> tuple[int, int, int]:
    csv_rows = 0
    jsonl_rows = 0
    json_files = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if not header:
                    raise AssertionError(f"Empty CSV: {path.relative_to(ROOT)}")
                width = len(header)
                for row_number, row in enumerate(reader, 2):
                    if len(row) != width:
                        raise AssertionError(
                            f"CSV width mismatch: {path.relative_to(ROOT)}:{row_number}"
                        )
                    csv_rows += 1
        elif path.suffix.lower() == ".json":
            json.loads(path.read_text(encoding="utf-8-sig"))
            json_files += 1
        elif path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    if line.strip():
                        json.loads(line)
                        jsonl_rows += 1
        elif path.name.endswith(".jsonl.gz"):
            with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
                for line in handle:
                    if line.strip():
                        json.loads(line)
                        jsonl_rows += 1
    return csv_rows, jsonl_rows, json_files


def verify_ner_offset_audit() -> Counter:
    rows = read_csv(
        "data/human_reference/ner/gold_entities_with_offset_integrity_status.csv"
    )
    counts = Counter(row["offset_integrity_status"] for row in rows)
    expected = Counter(
        {
            "exact_match": 4_858,
            "text_found_at_other_position": 12_385,
            "entity_text_not_found_in_current_passage": 809,
        }
    )
    if counts != expected:
        raise AssertionError(f"Unexpected NER offset counts: {counts}")
    return counts


def relation_key_from_label(row: dict[str, str]) -> tuple:
    return tuple(json.loads(row["candidate_key_json"]))


def relation_key_from_prediction(row: dict) -> tuple:
    head = row["head"]
    tail = row["tail"]
    return (
        row["sample_id"],
        int(head["start_char"]),
        int(head["end_char"]),
        head["text"],
        head["type"],
        row["relation"],
        int(tail["start_char"]),
        int(tail["end_char"]),
        tail["text"],
        tail["type"],
    )


def prf(reference: set[tuple], predicted: set[tuple]) -> tuple[float, float, float]:
    tp = len(reference & predicted)
    fp = len(predicted - reference)
    fn = len(reference - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def verify_relation_metrics() -> dict[str, tuple[float, float, float]]:
    labels = read_csv(
        "data/human_reference/relation/human_revalidated_final_labels.csv"
    )
    label_counts = Counter(row["final_decision"] for row in labels)
    expected_counts = Counter(
        {
            "present": 445,
            "absent": 575,
            "insufficient_context": 97,
            "entity_or_type_error": 45,
        }
    )
    if label_counts != expected_counts:
        raise AssertionError(f"Unexpected relation label counts: {label_counts}")
    reference = {
        relation_key_from_label(row) for row in labels if row["final_decision"] == "present"
    }
    files = {
        "typed_cooccurrence": "typed_cooccurrence.jsonl",
        "distance_trigger_heuristic": "distance_trigger_heuristic.jsonl",
        "fixed_charngram_comparator": "fixed_charngram_comparator.jsonl",
        "deepseek_supported_or_plausible": "deepseek_supported_or_plausible.jsonl",
        "deepseek_supported_only": "deepseek_supported_only.jsonl",
        "local_symbolic_text": "local_symbolic_text.jsonl",
        "cmfkg_frozen_hybrid": "cmfkg_frozen_hybrid.jsonl",
    }
    metrics_table = {
        row["method"]: row
        for row in read_csv(
            "data/human_reference/relation/human_revalidated_relation_metrics.csv"
        )
        if row["analysis_set"] == "primary_present_vs_all_nonpresent"
    }
    output = {}
    for method, name in files.items():
        predictions = read_jsonl(f"model_outputs/relation/{name}")
        predicted = {relation_key_from_prediction(row) for row in predictions}
        observed = prf(reference, predicted)
        archived = tuple(
            float(metrics_table[method][metric]) for metric in ("precision", "recall", "f1")
        )
        if any(abs(left - right) > 1e-12 for left, right in zip(observed, archived)):
            raise AssertionError(f"Relation metric mismatch for {method}")
        output[method] = observed
    return output


def verify_rule_counts() -> dict[str, Counter]:
    rows = read_csv(
        "data/human_reference/rule_validation/b_rule_item_level_final.csv"
    )
    by_rule: dict[str, Counter] = {}
    for row in rows:
        by_rule.setdefault(row["rule_id"], Counter())[row["final_decision"]] += 1
    expected = {
        "B1": Counter({"不支持": 99, "不确定": 1}),
        "B2": Counter({"支持": 42, "不确定": 48, "不支持": 10}),
        "B3": Counter({"支持": 8, "不确定": 33, "不支持": 59}),
    }
    if by_rule != expected:
        raise AssertionError(f"Unexpected rule-validation counts: {by_rule}")
    return by_rule


def verify_summary_tables() -> None:
    ocr = {
        row["metric"]: row
        for row in read_csv("data/human_reference/ocr/ocr_cer_summary.csv")
    }
    if int(ocr["raw_ocr_cer_all_adjudicated_pages"]["n_pages"]) != 51:
        raise AssertionError("OCR 51-page summary mismatch")
    if int(ocr["cleaned_ocr_cer_raw_cleaned_paired_subset"]["n_pages"]) != 17:
        raise AssertionError("OCR 17-page paired summary mismatch")
    primary = read_csv("data/user_study/user_study_primary_results.csv")
    if len(primary) != 2 or any(int(row["n_pairs"]) != 100 for row in primary):
        raise AssertionError("User-study primary table mismatch")


def verify_supplementary_archives() -> None:
    path = (
        ROOT
        / "supplementary_submission"
        / "Supplementary_Data_2_Machine_Readable_Resources.zip"
    )
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise AssertionError("Damaged Supplementary Data 2 archive")
        expected = {"Annotation_Guideline.md", "CMFKG_Ontology.ttl", "Operational_Rules.json"}
        if set(archive.namelist()) != expected:
            raise AssertionError("Unexpected Supplementary Data 2 members")


def main() -> None:
    checked_hashes = verify_hashes()
    csv_rows, jsonl_rows, json_files = verify_structured_files()
    ner_counts = verify_ner_offset_audit()
    relation_metrics = verify_relation_metrics()
    rule_counts = verify_rule_counts()
    verify_summary_tables()
    verify_supplementary_archives()
    print(f"Release verification passed: {checked_hashes} file hashes")
    print(f"Structured records parsed: {csv_rows} CSV rows, {jsonl_rows} JSONL rows, {json_files} JSON files")
    print(f"NER offset audit: {dict(ner_counts)}")
    print(f"Relation methods reproduced: {len(relation_metrics)}")
    print(f"Rule-validation samples: {sum(sum(counts.values()) for counts in rule_counts.values())}")


if __name__ == "__main__":
    main()
