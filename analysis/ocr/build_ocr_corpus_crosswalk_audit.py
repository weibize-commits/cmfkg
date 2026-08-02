from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CER_INPUT = ROOT / "ocr_quantitative" / "outputs" / "ocr_cer_page_results.csv"
CORPUS_INPUT = ROOT / "output" / "doc" / "supplement" / "corpus-source-inventory.csv"
OUTPUT_DIR = ROOT / "ocr_quantitative" / "outputs"
PAGE_OUTPUT = OUTPUT_DIR / "ocr_corpus_crosswalk_page_results.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "ocr_corpus_crosswalk_summary.csv"
MANIFEST_OUTPUT = OUTPUT_DIR / "ocr_corpus_crosswalk_manifest.json"
SEED = 20260731
BOOTSTRAP_REPS = 10000


def normalise_source_key(value: object) -> str:
    key = str(value).replace("\\", "/").strip()
    if key.lower().endswith(".txt"):
        key = key[:-4]
    return key


def bootstrap_mean_interval(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = random.Random(SEED)
    count = len(values)
    means = []
    for _ in range(BOOTSTRAP_REPS):
        means.append(sum(values[rng.randrange(count)] for _ in range(count)) / count)
    means.sort()
    return means[int(0.025 * BOOTSTRAP_REPS)], means[min(BOOTSTRAP_REPS - 1, int(0.975 * BOOTSTRAP_REPS))]


def source_cluster_bootstrap_interval(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    clusters = [group[column].dropna().astype(float).tolist() for _, group in frame.groupby("source_key_normalised")]
    clusters = [cluster for cluster in clusters if cluster]
    if not clusters:
        return math.nan, math.nan
    rng = random.Random(SEED)
    means = []
    for _ in range(BOOTSTRAP_REPS):
        sampled_clusters = [clusters[rng.randrange(len(clusters))] for _ in range(len(clusters))]
        sampled_values = [value for cluster in sampled_clusters for value in cluster]
        means.append(sum(sampled_values) / len(sampled_values))
    means.sort()
    return means[int(0.025 * BOOTSTRAP_REPS)], means[min(BOOTSTRAP_REPS - 1, int(0.975 * BOOTSTRAP_REPS))]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarise(
    name: str,
    frame: pd.DataFrame,
    column: str,
    note: str,
    cluster_by_source: bool = False,
) -> dict[str, object]:
    values = [float(value) for value in frame[column].dropna().tolist()]
    if cluster_by_source:
        low, high = source_cluster_bootstrap_interval(frame, column)
        resampling_unit = "source_key cluster"
    else:
        low, high = bootstrap_mean_interval(values)
        resampling_unit = "page"
    series = pd.Series(values, dtype=float)
    return {
        "analysis": name,
        "n_pages": len(values),
        "n_source_keys": int(frame.loc[frame[column].notna(), "source_key_normalised"].nunique()),
        "mean": float(series.mean()),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "median": float(series.median()),
        "q1": float(series.quantile(0.25)),
        "q3": float(series.quantile(0.75)),
        "minimum": float(series.min()),
        "maximum": float(series.max()),
        "source_column": column,
        "bootstrap_resampling_unit": resampling_unit,
        "scope_note": note,
    }


def main() -> None:
    cer = pd.read_csv(CER_INPUT)
    corpus = pd.read_csv(CORPUS_INPUT)
    cer["source_key_normalised"] = cer["source_key"].map(normalise_source_key)
    corpus["source_key_normalised"] = corpus["archived_source_file"].map(normalise_source_key)
    corpus["extraction_included_bool"] = corpus["extraction_included"].astype(str).str.lower().eq("true")

    crosswalk = corpus[
        [
            "source_key_normalised",
            "source_id",
            "archived_source_file",
            "extraction_included_bool",
            "bibliographic_identity_status",
        ]
    ].drop_duplicates("source_key_normalised")
    merged = cer.merge(crosswalk, on="source_key_normalised", how="left", validate="many_to_one")
    merged["crosswalk_matched"] = merged["source_id"].notna()
    merged["corpus_extraction_included"] = merged["extraction_included_bool"].fillna(False).astype(bool)
    merged.to_csv(PAGE_OUTPUT, index=False, encoding="utf-8-sig")

    corpus_pages = merged[merged["corpus_extraction_included"]].copy()
    paired_pages = corpus_pages[corpus_pages["cleaned_cer"].notna()].copy()
    if len(corpus_pages) != 42:
        raise RuntimeError(f"Expected 42 corpus-matched adjudicated pages, found {len(corpus_pages)}")
    if len(paired_pages) != 17:
        raise RuntimeError(f"Expected 17 corpus-matched paired pages, found {len(paired_pages)}")

    summaries = [
        summarise(
            "raw_cer_corpus_crosswalk_matched",
            corpus_pages,
            "raw_cer",
            "Primary corpus-scoped raw OCR audit: adjudicated pages whose source key crosswalks to one of the 52 extraction-included sources.",
            cluster_by_source=True,
        ),
        summarise(
            "raw_cer_all_archived_ocr_pages",
            merged,
            "raw_cer",
            "Broader technical audit retained for provenance; includes nine adjudicated pages without an extraction-included corpus crosswalk.",
        ),
        summarise(
            "cleaned_cer_corpus_paired_subset",
            paired_pages,
            "cleaned_cer",
            "Only corpus-matched pages with raw OCR, cleaned OCR and adjudicated transcription.",
        ),
        summarise(
            "cleaned_minus_raw_cer_corpus_paired_subset",
            paired_pages,
            "delta_cleaned_minus_raw_cer",
            "Paired corpus-matched subset; negative values indicate lower CER after cleaning.",
        ),
    ]
    pd.DataFrame(summaries).to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")

    source_counts = corpus_pages.groupby("source_key_normalised").size()
    manifest = {
        "version": "ocr_corpus_crosswalk_audit_v1",
        "seed": SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "inputs": {
            str(CER_INPUT): sha256(CER_INPUT),
            str(CORPUS_INPUT): sha256(CORPUS_INPUT),
        },
        "counts": {
            "adjudicated_archive_pages": int(len(merged)),
            "corpus_crosswalk_matched_pages": int(len(corpus_pages)),
            "corpus_crosswalk_matched_source_keys": int(corpus_pages["source_key_normalised"].nunique()),
            "unmatched_or_excluded_pages": int(len(merged) - merged["corpus_extraction_included"].sum()),
            "paired_raw_cleaned_corpus_pages": int(len(paired_pages)),
            "source_keys_with_one_page": int((source_counts == 1).sum()),
            "source_keys_with_two_pages": int((source_counts == 2).sum()),
        },
        "dynasty_stratification_status": (
            "not_estimable: the page audit does not carry a complete dynasty field for all sampled source keys, "
            "and 38 of 40 corpus-matched source keys contribute only one page"
        ),
        "outputs": [str(PAGE_OUTPUT), str(SUMMARY_OUTPUT)],
    }
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(SUMMARY_OUTPUT)
    print(MANIFEST_OUTPUT)


if __name__ == "__main__":
    main()
