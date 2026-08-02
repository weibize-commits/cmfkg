# Human-revalidated relation benchmark

- Human-revalidated confirmatory set: 65 passages and 1162 candidate relation pairs.
- Final labels: present=445, absent=575, insufficient_context=97, entity_or_type_error=45.
- Adjudication: 1065 labels were auto-accepted after two-expert agreement; 97 disagreements were resolved by the third expert.
- Primary analysis: present versus all non-present labels; bootstrap unit=passage; resamples=5000.
- Primary CMFKG hybrid: precision=0.3610, recall=0.4989, F1=0.4189 (95% CI 0.3362-0.4996).
- Primary fixed char-ngram comparator: precision=0.3657, recall=0.5506, F1=0.4395 (95% CI 0.3728-0.5079).
- Sensitivity analysis excluding insufficient_context and entity_or_type_error: CMFKG F1=0.4444; fixed comparator F1=0.4653.
