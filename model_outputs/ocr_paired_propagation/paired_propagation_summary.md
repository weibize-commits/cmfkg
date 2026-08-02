# Paired OCR propagation sensitivity

This analysis uses the 17 pages with raw OCR, cleaned OCR, and an adjudicated manual transcription.
The same frozen model is run on all three versions. The adjudicated-text prediction is used only as a stability reference, not as human-annotated truth.

- SikuRoBERTa-CRF_archived / entity: raw stability F1=0.8455; cleaned stability F1=0.9797; delta=0.1342 (bootstrap 95% CI 0.0898 to 0.1825).
- SikuRoBERTa-CRF_archived / relation: raw stability F1=0.6557; cleaned stability F1=0.9516; delta=0.2958 (bootstrap 95% CI 0.1879 to 0.4558).
- SikuRoBERTa-CRF_archived / rule: raw stability F1=0.0687; cleaned stability F1=0.9536; delta=0.8849 (bootstrap 95% CI 0.0000 to 1.0000).
- Chinese-RoBERTa-CRF_archived / entity: raw stability F1=0.8077; cleaned stability F1=0.9879; delta=0.1802 (bootstrap 95% CI 0.1220 to 0.2500).
- Chinese-RoBERTa-CRF_archived / relation: raw stability F1=0.5660; cleaned stability F1=0.9582; delta=0.3922 (bootstrap 95% CI 0.2234 to 0.6183).
- Chinese-RoBERTa-CRF_archived / rule: raw stability F1=0.2387; cleaned stability F1=0.9390; delta=0.7004 (bootstrap 95% CI 0.0000 to 1.0000).

These are paired output-stability results. They do not estimate human-reference accuracy, the number of OCR-caused false inferences in the full graph, C-class flagging precision, or researcher-view OCR-detection precision/recall.
