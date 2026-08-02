# OCR quantitative package status

- Status: `complete_manual_cer_computed`
- Inventory pages: 19034
- Sampled pages for manual transcription: 60
- Rendered source page images: 60
- Raw/cleaned paired OCR pages for cleaning-delta audit: 5781
- Adjudicated pages included in CER: 51
- Sampled pages excluded from CER: 9

## CER results

- raw_ocr_cer_all_adjudicated_pages: n=51, mean=0.1491, bootstrap 95% CI [0.1134, 0.1894]. Scope: All pages with non-empty adjudicated manual transcription. Note: Report as raw OCR CER for the 51-page adjudicated sample.
- cleaned_ocr_cer_raw_cleaned_paired_subset: n=17, mean=0.0114, bootstrap 95% CI [0.0022, 0.0259]. Scope: Only adjudicated pages that also have a corresponding cleaned OCR text. Note: Report as cleaned OCR CER in the raw/cleaned paired subset only; do not generalise to all 51 adjudicated pages.
- cleaned_minus_raw_cer_paired_subset: n=17, mean=-0.2639, bootstrap 95% CI [-0.3522, -0.1812]. Scope: Only adjudicated pages that also have a corresponding cleaned OCR text. Note: Report as a paired subset difference; negative values indicate lower CER after cleaning.

## Reporting cautions

- Cleaned OCR CER and cleaned-minus-raw CER must be reported as results from the raw/cleaned paired subset, not as full-sample estimates.
- Manual-transcription QC records that adjudicator A01 selected the T2 transcription as the final reference for all included pages after review; report this as an adjudication decision.
