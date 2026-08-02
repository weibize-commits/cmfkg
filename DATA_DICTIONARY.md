# Data dictionary

## Human reference data

- `data/human_reference/ner/` contains de-identified expert entity layers, adjudication inventories, the 763-record split and explicit offset-integrity audits. Complete source passages and the original Excel workbooks are excluded. The archived offsets are not release-validated strict spans.
- `data/human_reference/relation/` contains 1,162 item-level decisions from 65 passages, two-expert decisions, adjudication status, final labels, evidence excerpts and frozen evaluation summaries.
- `data/human_reference/rule_validation/` contains 300 sampled B-rule candidates, independent ratings, adjudicated final decisions and agreement summaries.
- `data/human_reference/ocr/` contains the 60-page audit disposition, 51 adjudicated transcriptions, corpus-crosswalk results and character-error summaries. Source page images are excluded.

No LLM, simulation or model-generation marker occurs in `data/human_reference/`. Model names and predictions are retained only in the computational-output directories and analysis records where they form necessary provenance.

## User study

- `data/user_study/task_delivery_logs.csv` contains de-identified task-level records.
- `data/user_study/randomization_v2.1.csv` contains blind system codes and allocation records.
- Files ending in `_model.csv` or `_results.csv` contain participant-level analyses and subgroup summaries.
- `participant_id` values are study codes and are not direct identifiers.

## Derived and model outputs

- `derived_outputs/rule_candidates/` contains deterministic rule-generated candidate populations. These are not verified facts and are not a human reference set.
- `model_outputs/ner_archived/` contains archived lexical and neural benchmark outputs. They remain quarantined from a final strict-span claim until the offset issue is resolved.
- `model_outputs/relation/` contains frozen outputs from named relation-decision methods, including LLM-assisted methods where applicable.
- `model_outputs/ocr_paired_propagation/` contains frozen downstream-sensitivity outputs for the 17 raw/cleaned/adjudicated page pairs.

## Supplementary submission mirror

- `supplementary_submission/CMFKG_Revision_12_Supplementary_Information.pdf` is the merged Supplementary Information PDF.
- `supplementary_submission/Supplementary_Data_1_Verification_Tables.xlsx` contains the verification tables in 15 worksheets.
- `supplementary_submission/Supplementary_Data_2_Machine_Readable_Resources.zip` contains the annotation guideline, ontology and operational-rule inventory.

## Common identifiers

- `record_id` identifies an entity-reference passage without exposing annotator identity.
- `sample_id` identifies an OCR page, relation passage or validation item within its own table.
- `review_id` identifies one relation-candidate decision.
- `candidate_id` identifies one operational-rule candidate.
- `participant_id` is a de-identified study code.

Counts from different output classes must not be summed. Asserted records, rule candidates, review flags and presentation views have different meanings.
