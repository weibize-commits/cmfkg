# NER offset-integrity status

The four archived source workbooks were produced through independent expert annotation and third-expert adjudication. They contain no LLM, model-generation, simulation, placeholder or API-key markers. The final adjudicated workbook contains 763 records and 18,052 entity rows, and those rows match the archived JSONL entity inventory.

The release audit nevertheless found that the declared character offsets do not consistently select the declared entity text in the current passage version.

| Status | Entity rows |
|---|---:|
| Declared span exactly matches entity text | 4,858 |
| Entity text occurs elsewhere in the passage | 12,385 |
| Entity text is absent from the current passage | 809 |
| Total | 18,052 |

This pattern indicates passage-version or character-offset drift. It prevents the current export from being described as a verified strict-span NER benchmark. Automatically moving each entity to the nearest matching string would not be an acceptable substitute for expert verification, especially when a mention occurs more than once or no exact string remains.

Before a final NER benchmark release, the authors must freeze one passage version, realign every span to that version and obtain expert confirmation for ambiguous or unmatched rows. NER metrics and manuscript statements based on strict-span scoring must then be recomputed. The file `data/human_reference/ner/offset_integrity_by_record.csv` identifies affected records, and `data/human_reference/ner/gold_entities_with_offset_integrity_status.csv` provides the row-level audit without redistributing full passages.

The relation, OCR, operational-rule and user-study release tables are not affected by this offset issue.
