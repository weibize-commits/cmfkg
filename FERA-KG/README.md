# FERA-KG reproducibility release

This directory contains the code, frozen derived predictions, evaluation summaries,
source-work fold manifests, adjudicated label-only references, and figure source data
for **FERA-KG: A Provenance-Aware Framework for Source-Grounded Relation Verification**.

## Repository scope

The release supports inspection of the model definitions, data-processing logic,
source-work-grouped out-of-fold design, fusion analysis, matched module study,
statistical evaluation, and reported figure source data. The frozen evaluation entry point is:

```bash
python -m pip install -e .
python scripts/run_pipeline.py --mode frozen-evaluation
```

Training entry points and task manifests are included. Training requires an authorised
Chinese RoBERTa checkpoint and the registered passage inputs described below.

## Data boundary

The public `data` directory contains label-only references and derived numerical
artifacts. It deliberately excludes the 1,250 exact registered historical passages,
complete source editions, raw scans, model checkpoints, and serialized estimators.
Item-level redistribution permission was not recorded for those passages. This
restriction does not apply to the released predictions, metrics, fold assignments,
configuration files, or figure source data.

The controlled, primary-natural, and difficulty-track records can be audited from
their item identifiers, source-work fields, hashes, labels, frozen predictions, and
evaluation outputs. Exact model-input reconstruction additionally requires access to
the registered source passages through their holding institutions or an authorised
review package.

## External material

The ProVe Web-Text Relation data used for a protocol audit are not redistributed here.
They remain available from the original Figshare record under its stated CC BY 4.0
licence. This repository does not extend that data licence to third-party code.

## Environment

See `ENVIRONMENT.md`, `requirements.txt`, and `pyproject.toml`. The principal training
runs used Python 3.9, PyTorch 2.1.0, and torch-npu 2.1.0.post6 on Ascend 910B3 hardware.

## Integrity

`MANIFEST_SHA256.csv` records a SHA-256 digest for every released file except the
manifest itself.
