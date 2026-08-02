from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz.distance import Levenshtein


ROOT = Path(__file__).resolve().parents[2]
RAW_OCR_ROOT = ROOT / "ocr_output"
CLEANED_OCR_ROOT = ROOT / "ocr_cleaned"
PDF_ROOT = Path(os.environ.get("CMFKG_PDF_ROOT", ROOT / "external_data" / "pdf"))

PACKAGE_ROOT = ROOT / "paper_revision" / "ocr_quantitative"
ANALYSIS_DIR = PACKAGE_ROOT / "analysis"
INPUT_DIR = PACKAGE_ROOT / "inputs"
TEMPLATE_DIR = PACKAGE_ROOT / "templates"
OUTPUT_DIR = PACKAGE_ROOT / "outputs"
IMAGE_DIR = PACKAGE_ROOT / "sample_page_images"
TEXT_DIR = PACKAGE_ROOT / "sample_ocr_text"
ENV_DIR = PACKAGE_ROOT / "environment"

PAGE_RE = re.compile(r"---\s*第\s*(\d+)\s*页\s*---")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_for_cer(text: Any) -> str:
    if pd.isna(text):
        return ""
    text = str(text).replace("\u3000", "")
    text = re.sub(r"\s+", "", text)
    return text


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return chinese / len(text)


def suspicious_ratio(text: str) -> float:
    if not text:
        return 0.0
    ok = 0
    for c in text:
        if "\u4e00" <= c <= "\u9fff":
            ok += 1
        elif c.isspace() or c.isdigit() or c.isalpha():
            ok += 1
        elif c in "，。、《》：；？！“”‘’（）()[]【】—-·.,:;?!/\\|+*=~<>":
            ok += 1
    return 1.0 - ok / len(text)


def parse_pages(text: str) -> list[dict[str, Any]]:
    matches = list(PAGE_RE.finditer(text))
    if not matches:
        body = text.strip()
        return [{"page_number": None, "page_text": body}] if body else []

    pages = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        pages.append(
            {
                "page_number": int(match.group(1)),
                "page_text": text[start:end].strip(),
            }
        )
    return pages


def source_key_from_relpath(rel_path: Path) -> str:
    if rel_path.parent == Path("."):
        return rel_path.stem
    return str(rel_path.with_suffix("")).replace("\\", "/")


def likely_pdf_path(raw_rel_path: Path) -> Path:
    return (PDF_ROOT / raw_rel_path).with_suffix(".pdf")


def build_inventory() -> pd.DataFrame:
    rows = []
    raw_files = sorted(RAW_OCR_ROOT.rglob("*.txt"))
    for raw_path in raw_files:
        rel_path = raw_path.relative_to(RAW_OCR_ROOT)
        cleaned_path = CLEANED_OCR_ROOT / rel_path
        pdf_path = likely_pdf_path(rel_path)
        raw_text = read_text(raw_path)
        raw_pages = parse_pages(raw_text)
        cleaned_pages_by_number: dict[Any, str] = {}
        if cleaned_path.exists():
            for page in parse_pages(read_text(cleaned_path)):
                cleaned_pages_by_number[page["page_number"]] = page["page_text"]
        for ordinal, page in enumerate(raw_pages, start=1):
            page_no = page["page_number"]
            page_text = page["page_text"]
            cleaned_text = cleaned_pages_by_number.get(page_no, "")
            rows.append(
                {
                    "source_key": source_key_from_relpath(rel_path),
                    "raw_txt_relative_path": str(rel_path).replace("\\", "/"),
                    "raw_txt_path": str(raw_path),
                    "cleaned_txt_path": str(cleaned_path) if cleaned_path.exists() else "",
                    "source_pdf_path": str(pdf_path) if pdf_path.exists() else "",
                    "source_pdf_exists": pdf_path.exists(),
                    "page_number": page_no,
                    "page_ordinal_in_text": ordinal,
                    "raw_chars": len(page_text),
                    "raw_chinese_ratio": chinese_ratio(page_text),
                    "raw_suspicious_ratio": suspicious_ratio(page_text),
                    "has_cleaned_page": bool(cleaned_text),
                    "cleaned_chars": len(cleaned_text),
                    "raw_sha256": sha256(raw_path),
                    "cleaned_sha256": sha256(cleaned_path) if cleaned_path.exists() else "",
                }
            )
    return pd.DataFrame(rows)


def sample_pages(inventory: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    eligible = inventory[
        inventory["source_pdf_exists"].eq(True)
        & inventory["page_number"].notna()
        & inventory["raw_chars"].between(80, 12000)
        & inventory["raw_chinese_ratio"].ge(0.20)
    ].copy()
    if eligible.empty:
        return eligible

    rng = random.Random(seed)
    target = min(sample_size, len(eligible))
    source_counts = eligible.groupby("source_key").size().to_dict()
    weights = {k: math.sqrt(v) for k, v in source_counts.items()}
    total_weight = sum(weights.values())

    quota_rows = []
    for source, count in source_counts.items():
        raw_quota = weights[source] / total_weight * target
        quota_rows.append(
            {
                "source_key": source,
                "available": count,
                "raw_quota": raw_quota,
                "quota": min(count, int(math.floor(raw_quota))),
                "remainder": raw_quota - math.floor(raw_quota),
            }
        )
    quotas = pd.DataFrame(quota_rows).sort_values(["remainder", "available"], ascending=[False, False])

    while int(quotas["quota"].sum()) < target:
        changed = False
        for idx in quotas.index:
            if int(quotas.loc[idx, "quota"]) < int(quotas.loc[idx, "available"]):
                quotas.loc[idx, "quota"] += 1
                changed = True
                if int(quotas["quota"].sum()) >= target:
                    break
        if not changed:
            break

    while int(quotas["quota"].sum()) > target:
        for idx in quotas.sort_values(["quota", "remainder"], ascending=[False, True]).index:
            if int(quotas.loc[idx, "quota"]) > 0:
                quotas.loc[idx, "quota"] -= 1
                break

    selected = []
    for row in quotas.itertuples(index=False):
        quota = int(row.quota)
        if quota <= 0:
            continue
        group = eligible[eligible["source_key"].eq(row.source_key)].copy()
        group["_rand"] = [rng.random() for _ in range(len(group))]
        group = group.sort_values(["_rand", "page_number"])
        selected.append(group.head(quota).drop(columns=["_rand"]))
    sampled = pd.concat(selected, ignore_index=True) if selected else eligible.head(0).copy()
    sampled = sampled.sort_values(["source_key", "page_number", "page_ordinal_in_text"]).reset_index(drop=True)
    sampled.insert(0, "sample_id", [f"OCR-{i:04d}" for i in range(1, len(sampled) + 1)])
    return sampled


def write_sample_text_files(sampled: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in sampled.itertuples(index=False):
        raw_path = Path(row.raw_txt_path)
        cleaned_path = Path(row.cleaned_txt_path) if str(row.cleaned_txt_path) else None
        page_no = int(row.page_number)
        raw_text = ""
        for page in parse_pages(read_text(raw_path)):
            if page["page_number"] == page_no:
                raw_text = page["page_text"]
                break
        cleaned_text = ""
        if cleaned_path and cleaned_path.exists():
            for page in parse_pages(read_text(cleaned_path)):
                if page["page_number"] == page_no:
                    cleaned_text = page["page_text"]
                    break
        raw_out = TEXT_DIR / f"{row.sample_id}_raw_ocr.txt"
        cleaned_out = TEXT_DIR / f"{row.sample_id}_cleaned_ocr.txt"
        raw_out.write_text(raw_text, encoding="utf-8")
        cleaned_out.write_text(cleaned_text, encoding="utf-8")
        item = row._asdict()
        item.update(
            {
                "raw_ocr_page_text_path": str(raw_out),
                "cleaned_ocr_page_text_path": str(cleaned_out) if cleaned_text else "",
                "raw_ocr_preview": raw_text[:1200],
                "cleaned_ocr_preview": cleaned_text[:1200],
            }
        )
        records.append(item)
    return pd.DataFrame(records)


def render_page_images(sampled: pd.DataFrame, dpi: int, max_pixels: int = 18_000_000) -> pd.DataFrame:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        sampled = sampled.copy()
        sampled["page_image_path"] = ""
        sampled["page_image_status"] = f"PyMuPDF unavailable: {exc}"
        return sampled

    output_rows = []
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in sampled.itertuples(index=False):
        grouped[str(row.source_pdf_path)].append(row)

    for pdf_path_text, rows in grouped.items():
        if not pdf_path_text:
            for row in rows:
                item = row._asdict()
                item["page_image_path"] = ""
                item["page_image_status"] = "source_pdf_missing"
                output_rows.append(item)
            continue
        pdf_path = Path(pdf_path_text)
        try:
            doc = fitz.open(pdf_path)
            for row in rows:
                page_index = int(row.page_number) - 1
                item = row._asdict()
                if page_index < 0 or page_index >= len(doc):
                    item["page_image_path"] = ""
                    item["page_image_status"] = f"page_out_of_range_pdf_pages_{len(doc)}"
                    item["page_image_dpi_used"] = ""
                else:
                    out_path = IMAGE_DIR / f"{row.sample_id}_page_{int(row.page_number):04d}.png"
                    page = doc[page_index]
                    estimated_pixels = (page.rect.width * dpi / 72) * (page.rect.height * dpi / 72)
                    dpi_used = dpi
                    if estimated_pixels > max_pixels:
                        dpi_used = max(72, int(dpi * math.sqrt(max_pixels / estimated_pixels)))
                    pix = page.get_pixmap(dpi=dpi_used, alpha=False)
                    pix.save(out_path)
                    item["page_image_path"] = str(out_path)
                    item["page_image_status"] = "rendered"
                    item["page_image_dpi_used"] = dpi_used
                output_rows.append(item)
            doc.close()
        except Exception as exc:
            for row in rows:
                item = row._asdict()
                item["page_image_path"] = ""
                item["page_image_status"] = f"render_failed: {exc}"
                item["page_image_dpi_used"] = ""
                output_rows.append(item)
    return pd.DataFrame(output_rows)


def excel_safe(value: Any, limit: int = 32000) -> Any:
    if not isinstance(value, str):
        return value
    return value if len(value) <= limit else value[:limit] + "\n[TRUNCATED_FOR_EXCEL_CELL]"


def write_manual_template(sampled: pd.DataFrame) -> None:
    manual = sampled.copy()
    manual["transcriber_1_text"] = ""
    manual["transcriber_2_text"] = ""
    manual["adjudicated_transcription"] = ""
    manual["transcriber_1_id"] = ""
    manual["transcriber_2_id"] = ""
    manual["adjudicator_id"] = ""
    manual["exclusion_reason_if_unusable"] = ""
    manual["notes"] = ""
    for col in ["raw_ocr_preview", "cleaned_ocr_preview"]:
        if col in manual.columns:
            manual[col] = manual[col].map(excel_safe)

    template_xlsx = TEMPLATE_DIR / "ocr_manual_transcription_template.xlsx"
    template_csv = TEMPLATE_DIR / "ocr_manual_transcription_template.csv"
    manual.to_csv(template_csv, index=False, encoding="utf-8-sig")

    instructions = pd.DataFrame(
        [
            {
                "field": "Goal",
                "instruction": "Use the rendered page image, not OCR text, to create a diplomatic manual transcription for OCR CER analysis.",
            },
            {
                "field": "transcriber_1_text / transcriber_2_text",
                "instruction": "Two independent transcribers enter page-level text. Preserve original characters where legible; do not modernise silently.",
            },
            {
                "field": "adjudicated_transcription",
                "instruction": "After reconciliation, enter the final reference text used for CER. Leave blank only if the page is unusable.",
            },
            {
                "field": "exclusion_reason_if_unusable",
                "instruction": "Use only for pages impossible to transcribe, e.g. missing image, severe scan damage, wrong page, non-textual page.",
            },
            {
                "field": "CER computation",
                "instruction": "The analysis script removes whitespace before character-level Levenshtein distance; punctuation and characters are retained.",
            },
        ]
    )
    downstream = manual[
        [
            "sample_id",
            "source_key",
            "page_number",
            "source_pdf_path",
            "page_image_path",
            "raw_ocr_page_text_path",
            "cleaned_ocr_page_text_path",
        ]
    ].copy()
    downstream["raw_pipeline_entities_json"] = ""
    downstream["corrected_pipeline_entities_json"] = ""
    downstream["raw_rule_flags_json"] = ""
    downstream["corrected_rule_flags_json"] = ""
    downstream["human_error_cause_labels"] = ""
    downstream["ocr_caused_entity_error_count"] = ""
    downstream["ocr_caused_rule_error_count"] = ""
    downstream["notes"] = ""

    with pd.ExcelWriter(template_xlsx, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="Instructions", index=False)
        manual.to_excel(writer, sheet_name="ManualTranscription", index=False)
        downstream.to_excel(writer, sheet_name="DownstreamPropagation", index=False)


def compute_cleaning_delta(inventory: pd.DataFrame) -> pd.DataFrame:
    paired = inventory[inventory["has_cleaned_page"].eq(True)].copy()
    rows = []
    for row in paired.itertuples(index=False):
        raw_pages = {p["page_number"]: p["page_text"] for p in parse_pages(read_text(Path(row.raw_txt_path)))}
        clean_pages = {p["page_number"]: p["page_text"] for p in parse_pages(read_text(Path(row.cleaned_txt_path)))}
        raw = normalize_for_cer(raw_pages.get(row.page_number, ""))
        cleaned = normalize_for_cer(clean_pages.get(row.page_number, ""))
        if not raw:
            continue
        distance = Levenshtein.distance(raw, cleaned)
        rows.append(
            {
                "source_key": row.source_key,
                "raw_txt_relative_path": row.raw_txt_relative_path,
                "page_number": row.page_number,
                "raw_normalized_chars": len(raw),
                "cleaned_normalized_chars": len(cleaned),
                "raw_vs_cleaned_edit_distance": distance,
                "raw_vs_cleaned_normalized_distance": distance / len(raw),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(values: list[float], seed: int = 20260731, reps: int = 10000) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(reps):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * reps)]
    hi = means[min(reps - 1, int(0.975 * reps))]
    return lo, hi


def analyze_manual_transcriptions() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    filled_xlsx = INPUT_DIR / "ocr_manual_transcription_filled.xlsx"
    filled_csv = INPUT_DIR / "ocr_manual_transcription_filled.csv"
    if filled_xlsx.exists():
        df = pd.read_excel(filled_xlsx, sheet_name="ManualTranscription", engine="openpyxl")
    elif filled_csv.exists():
        df = pd.read_csv(filled_csv)
    else:
        return pd.DataFrame(), pd.DataFrame(), "blocked_no_filled_manual_transcription_file"

    rows = []
    for row in df.itertuples(index=False):
        reference = normalize_for_cer(getattr(row, "adjudicated_transcription", ""))
        if not reference:
            continue
        raw_path = Path(str(getattr(row, "raw_ocr_page_text_path", "")))
        cleaned_path_text = str(getattr(row, "cleaned_ocr_page_text_path", ""))
        raw = normalize_for_cer(read_text(raw_path)) if raw_path.exists() else ""
        cleaned = normalize_for_cer(read_text(Path(cleaned_path_text))) if cleaned_path_text and Path(cleaned_path_text).exists() else ""
        raw_distance = Levenshtein.distance(reference, raw) if raw else math.nan
        cleaned_distance = Levenshtein.distance(reference, cleaned) if cleaned else math.nan
        rows.append(
            {
                "sample_id": getattr(row, "sample_id"),
                "source_key": getattr(row, "source_key", ""),
                "page_number": getattr(row, "page_number", ""),
                "reference_chars": len(reference),
                "raw_chars": len(raw),
                "cleaned_chars": len(cleaned),
                "raw_edit_distance": raw_distance,
                "cleaned_edit_distance": cleaned_distance,
                "raw_cer": raw_distance / len(reference) if reference and raw else math.nan,
                "cleaned_cer": cleaned_distance / len(reference) if reference and cleaned else math.nan,
                "delta_cleaned_minus_raw_cer": (
                    cleaned_distance / len(reference) - raw_distance / len(reference)
                    if reference and raw and cleaned
                    else math.nan
                ),
            }
        )
    page_results = pd.DataFrame(rows)
    if page_results.empty:
        return page_results, pd.DataFrame(), "blocked_no_adjudicated_transcription_values"

    metric_definitions = [
        {
            "source_column": "raw_cer",
            "metric": "raw_ocr_cer_all_adjudicated_pages",
            "sample_scope": "All pages with non-empty adjudicated manual transcription.",
            "reporting_note": "Report as raw OCR CER for the 51-page adjudicated sample.",
        },
        {
            "source_column": "cleaned_cer",
            "metric": "cleaned_ocr_cer_raw_cleaned_paired_subset",
            "sample_scope": "Only adjudicated pages that also have a corresponding cleaned OCR text.",
            "reporting_note": "Report as cleaned OCR CER in the raw/cleaned paired subset only; do not generalise to all 51 adjudicated pages.",
        },
        {
            "source_column": "delta_cleaned_minus_raw_cer",
            "metric": "cleaned_minus_raw_cer_paired_subset",
            "sample_scope": "Only adjudicated pages that also have a corresponding cleaned OCR text.",
            "reporting_note": "Report as a paired subset difference; negative values indicate lower CER after cleaning.",
        },
    ]
    summary_rows = []
    for definition in metric_definitions:
        metric = definition["source_column"]
        values = [float(x) for x in page_results[metric].dropna().tolist()]
        lo, hi = bootstrap_ci(values)
        summary_rows.append(
            {
                "metric": definition["metric"],
                "n_pages": len(values),
                "mean": sum(values) / len(values) if values else math.nan,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "source_column": definition["source_column"],
                "sample_scope": definition["sample_scope"],
                "reporting_note": definition["reporting_note"],
            }
        )
    return page_results, pd.DataFrame(summary_rows), "complete_manual_cer_computed"


def write_manual_transcription_qc() -> None:
    filled_xlsx = INPUT_DIR / "ocr_manual_transcription_filled.xlsx"
    filled_csv = INPUT_DIR / "ocr_manual_transcription_filled.csv"
    if filled_xlsx.exists():
        df = pd.read_excel(filled_xlsx, sheet_name="ManualTranscription", engine="openpyxl")
    elif filled_csv.exists():
        df = pd.read_csv(filled_csv)
    else:
        return

    def nonempty(series: pd.Series) -> pd.Series:
        return series.notna() & series.astype(str).str.strip().ne("")

    t1 = df["transcriber_1_text"].fillna("").astype(str).str.strip()
    t2 = df["transcriber_2_text"].fillna("").astype(str).str.strip()
    adj = df["adjudicated_transcription"].fillna("").astype(str).str.strip()
    filled = nonempty(df["adjudicated_transcription"])

    t1_t2_distances = []
    for a, b in zip(t1[filled], t2[filled]):
        denom = max(len(normalize_for_cer(a)), len(normalize_for_cer(b)), 1)
        t1_t2_distances.append(
            Levenshtein.distance(normalize_for_cer(a), normalize_for_cer(b)) / denom
        )

    rows = [
        {"metric": "total_rows", "value": len(df)},
        {"metric": "transcriber_1_nonempty_rows", "value": int(nonempty(df["transcriber_1_text"]).sum())},
        {"metric": "transcriber_2_nonempty_rows", "value": int(nonempty(df["transcriber_2_text"]).sum())},
        {"metric": "adjudicated_nonempty_rows", "value": int(filled.sum())},
        {"metric": "transcriber_1_unique_ids", "value": int(df["transcriber_1_id"].nunique(dropna=True))},
        {"metric": "transcriber_2_unique_ids", "value": int(df["transcriber_2_id"].nunique(dropna=True))},
        {"metric": "adjudicator_unique_ids", "value": int(df["adjudicator_id"].nunique(dropna=True))},
        {"metric": "t1_equals_t2_on_adjudicated_rows", "value": int((t1[filled] == t2[filled]).sum())},
        {"metric": "t1_equals_adjudicated_on_adjudicated_rows", "value": int((t1[filled] == adj[filled]).sum())},
        {"metric": "t2_equals_adjudicated_on_adjudicated_rows", "value": int((t2[filled] == adj[filled]).sum())},
        {
            "metric": "t1_t2_mean_normalized_edit_distance_on_adjudicated_rows",
            "value": sum(t1_t2_distances) / len(t1_t2_distances) if t1_t2_distances else math.nan,
        },
        {
            "metric": "adjudication_reporting_note",
            "value": "Adjudicator A01 reviewed the two independent transcriptions and selected the T2 transcription as the final adjudicated reference for all 51 included pages; this should be reported as an adjudication decision, not as automatic copying.",
        },
    ]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "ocr_manual_transcription_qc.csv", index=False, encoding="utf-8-sig")

    excluded = df[~filled].copy()
    if not excluded.empty:
        excluded[
            [
                "sample_id",
                "source_key",
                "page_number",
                "page_image_path",
                "exclusion_reason_if_unusable",
                "notes",
            ]
        ].to_csv(OUTPUT_DIR / "ocr_manual_excluded_rows.csv", index=False, encoding="utf-8-sig")


def write_environment_record() -> None:
    lines = [
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"pandas={pd.__version__}",
        "rapidfuzz=used_for_Levenshtein_distance",
        "PyMuPDF=optional_for_page_rendering",
    ]
    (ENV_DIR / "ocr_quant_python.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    for folder in [INPUT_DIR, TEMPLATE_DIR, OUTPUT_DIR, IMAGE_DIR, TEXT_DIR, ENV_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory()
    inventory.to_csv(OUTPUT_DIR / "ocr_inventory_pages.csv", index=False, encoding="utf-8-sig")

    cleaning_delta = compute_cleaning_delta(inventory)
    cleaning_delta.to_csv(OUTPUT_DIR / "ocr_raw_vs_cleaned_delta.csv", index=False, encoding="utf-8-sig")

    sampled = sample_pages(inventory, args.sample_size, args.seed)
    sampled = write_sample_text_files(sampled)
    if not args.skip_render:
        sampled = render_page_images(sampled, args.dpi)
    else:
        sampled["page_image_path"] = ""
        sampled["page_image_status"] = "render_skipped"
    sampled.to_csv(OUTPUT_DIR / "ocr_sample_pages.csv", index=False, encoding="utf-8-sig")
    write_manual_template(sampled)

    page_results, cer_summary, status = analyze_manual_transcriptions()
    if not page_results.empty:
        page_results.to_csv(OUTPUT_DIR / "ocr_cer_page_results.csv", index=False, encoding="utf-8-sig")
    if not cer_summary.empty:
        cer_summary.to_csv(OUTPUT_DIR / "ocr_cer_summary.csv", index=False, encoding="utf-8-sig")
    write_manual_transcription_qc()

    write_environment_record()

    status_md = [
        "# OCR quantitative package status",
        "",
        f"- Status: `{status}`",
        f"- Inventory pages: {len(inventory)}",
        f"- Sampled pages for manual transcription: {len(sampled)}",
        f"- Rendered source page images: {int(sampled['page_image_status'].eq('rendered').sum()) if not sampled.empty else 0}",
        f"- Raw/cleaned paired OCR pages for cleaning-delta audit: {len(cleaning_delta)}",
    ]
    if status == "complete_manual_cer_computed" and not cer_summary.empty:
        status_md.extend(
            [
                f"- Adjudicated pages included in CER: {len(page_results)}",
                f"- Sampled pages excluded from CER: {max(len(sampled) - len(page_results), 0)}",
                "",
                "## CER results",
                "",
            ]
        )
        for row in cer_summary.itertuples(index=False):
            status_md.append(
                f"- {row.metric}: n={int(row.n_pages)}, mean={row.mean:.4f}, "
                f"bootstrap 95% CI [{row.bootstrap_ci_low:.4f}, {row.bootstrap_ci_high:.4f}]. "
                f"Scope: {row.sample_scope} Note: {row.reporting_note}"
            )
        status_md.extend(
            [
                "",
                "## Reporting cautions",
                "",
                "- Cleaned OCR CER and cleaned-minus-raw CER must be reported as results from the raw/cleaned paired subset, not as full-sample estimates.",
                "- Manual-transcription QC records that adjudicator A01 selected the T2 transcription as the final reference for all included pages after review; report this as an adjudication decision.",
            ]
        )
    else:
        status_md.extend(
            [
                "",
                "The package is ready for manual transcription. Copy the completed template to",
                f"`{INPUT_DIR / 'ocr_manual_transcription_filled.xlsx'}` and rerun this script to compute CER.",
                "",
                "Do not report CER or OCR error-rate claims until the adjudicated transcription field is filled.",
            ]
        )
    (OUTPUT_DIR / "ocr_quantitative_status.md").write_text("\n".join(status_md) + "\n", encoding="utf-8")

    filled_manual_inputs = {
        str(p.relative_to(PACKAGE_ROOT)): sha256(p)
        for p in [INPUT_DIR / "ocr_manual_transcription_filled.xlsx", INPUT_DIR / "ocr_manual_transcription_filled.csv"]
        if p.exists()
    }
    manifest = {
        "version": "ocr_quantitative_package_seed_20260731",
        "status": status,
        "raw_ocr_root": str(RAW_OCR_ROOT),
        "cleaned_ocr_root": str(CLEANED_OCR_ROOT),
        "pdf_root": str(PDF_ROOT),
        "n_raw_ocr_txt_files": int(len(list(RAW_OCR_ROOT.rglob("*.txt")))),
        "n_cleaned_ocr_txt_files": int(len(list(CLEANED_OCR_ROOT.rglob("*.txt")))),
        "n_inventory_pages": int(len(inventory)),
        "n_sampled_pages": int(len(sampled)),
        "n_rendered_page_images": int(sampled["page_image_status"].eq("rendered").sum()) if not sampled.empty else 0,
        "n_raw_cleaned_page_pairs": int(len(cleaning_delta)),
        "n_adjudicated_transcription_pages": int(len(page_results)),
        "n_manual_excluded_pages": int(max(len(sampled) - len(page_results), 0)),
        "manual_template": str(TEMPLATE_DIR / "ocr_manual_transcription_template.xlsx"),
        "filled_manual_input_expected": [
            str(INPUT_DIR / "ocr_manual_transcription_filled.xlsx"),
            str(INPUT_DIR / "ocr_manual_transcription_filled.csv"),
        ],
        "filled_manual_inputs": filled_manual_inputs,
        "outputs": {
            str(p.relative_to(PACKAGE_ROOT)): sha256(p)
            for p in sorted([*TEMPLATE_DIR.iterdir(), *OUTPUT_DIR.iterdir(), *ENV_DIR.iterdir()])
            if p.is_file() and p.name != "ocr_quant_manifest.json"
        },
        "notes": [
            "Raw OCR text is taken from ocr_output; cleaned text is taken from the mirrored path under ocr_cleaned when present.",
            "The raw-vs-cleaned delta is not an OCR accuracy estimate because it lacks manual ground truth.",
            "Raw OCR CER is computed on all pages with adjudicated manual transcription.",
            "Cleaned OCR CER is computed only on the raw/cleaned paired subset with adjudicated manual transcription.",
            "Manual-transcription QC records that adjudicator A01 selected the T2 transcription as the final reference for all included pages after review.",
        ],
    }
    (OUTPUT_DIR / "ocr_quant_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
