# CMFKG data and code release candidate

This repository accompanies the manuscript **A Provenance-Aware Knowledge Graph for Role-Adaptive Access to Traditional Chinese Food Therapy Records**.

Release version: `0.9.0-rc1`  
Release status: public audit release candidate

## Evidence-aware repository structure

The repository separates materials by evidential role so that human reference data, deterministic rule outputs and model predictions cannot be confused.

1. `data/human_reference/` contains de-identified expert annotation, adjudication, transcription and scoring records.
2. `data/user_study/` contains de-identified user-study records and participant-level statistical summaries.
3. `derived_outputs/` contains deterministic rule-generated candidates. These records are not verified facts.
4. `model_outputs/` contains frozen predictions, configurations and model-run metadata. Named model records are retained as computational provenance and are not represented as human annotation.
5. `analysis/` contains the analysis and evaluation scripts available for the NER, relation, rule, OCR and user-study components.
6. `metadata/` contains the corpus inventory, annotation guideline, ontology, operational-rule inventory and quantitative figure/table source data.
7. `supplementary_submission/` mirrors the three Revision 12 files prepared for journal upload.

## Important NER integrity status

The archived NER export contains 763 passages and 18,052 entity rows, but its character offsets are not consistently aligned with the current passage version. The audit found 4,858 exact offset matches, 12,385 entity strings elsewhere in the passage and 809 entity strings absent from the current passage version. The export is therefore included for transparent audit only and is **not** a release-validated strict-span benchmark. See [`NER_OFFSET_INTEGRITY_STATUS.md`](NER_OFFSET_INTEGRITY_STATUS.md).

The relation, OCR, operational-rule and user-study data are not affected by this offset issue. Relation evaluation uses a human-revalidated set of 1,162 candidate pairs. OCR reporting distinguishes the 42-page corpus-linked raw audit from the 17-page raw/cleaned paired subset. Operational-rule outputs remain candidates, flags or presentation policies with distinct meanings.

## Data and rights boundaries

Copyrighted page images and complete source books are excluded. Short source-linked excerpts, derived annotations and adjudicated audit transcriptions are provided only to the extent permitted by applicable rights. User-study identifiers are study codes rather than direct identifiers. See [`LICENSE-DATA.md`](LICENSE-DATA.md) and [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

CMFKG is a heritage-information system. Its records do not establish modern clinical efficacy, safety, dosage or patient-specific recommendations.

## Reproducibility and integrity

Install the recorded Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

The environment records used for the OCR and participant analyses are stored in `environment/`. File-level SHA-256 values are listed in `SHA256SUMS.txt`, and `RELEASE_MANIFEST.csv` provides a machine-readable inventory.

Run the self-contained release verification from the repository root:

```bash
python analysis/verify_release.py
```

Component scripts preserve the revision analyses. Some builders require excluded source books, page images, original workbooks or model checkpoints and are retained for procedural transparency rather than as standalone public entry points. See `analysis/README.md`.

No credential is stored in this release. Scripts read credentials from environment variables, and `.env.example` contains variable names only.

## Citation and archival status

Citation metadata are provided in `CITATION.cff`. This GitHub release candidate does not yet have a DOI or repository accession. A DOI-bearing archival snapshot should be created after the NER offset realignment and expert confirmation are complete.
