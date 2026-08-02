# Analysis and reproducibility notes

Run the release-wide integrity and result check from the repository root:

```bash
python analysis/verify_release.py
```

The verifier checks every SHA-256 value, parses all CSV, JSON and JSONL files, validates the supplementary archive, reproduces the seven primary relation metrics from item-level human labels and frozen predictions, and checks the reported NER offset, rule-validation, OCR and participant-study counts.

The component scripts under `analysis/ner/`, `analysis/relation/`, `analysis/rule_validation/`, `analysis/ocr/` and `analysis/user_study/` preserve the analysis code used during revision. Some builders retain their original project-directory assumptions or require inputs that cannot be redistributed, including complete source books, page images, model checkpoints or the original expert workbooks. They are provided for procedural transparency and are not all standalone entry points in the public package.

The archived NER scripts and outputs must not be used to claim a release-validated strict-span benchmark until the offset realignment described in `../NER_OFFSET_INTEGRITY_STATUS.md` is complete.
