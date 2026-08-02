# CMFKG entity-annotation guideline, consolidated version 3.1

Status: `REVISION_12_ARCHIVED_REFERENCE_GUIDELINE`

This document reconciles the preserved CMFKG guideline versions 1.0 and 2.0
with the 13-class schema represented in the expert annotation workbooks.
Expert identities remain de-identified as Expert 1, Expert 2 and Expert 3.
The archived reference export contains passage-version or character-offset
drift. It must not be treated as a release-validated strict-span benchmark
until the spans have been realigned to one frozen passage version and the
ambiguous or unmatched cases have received expert confirmation.

## 1. Annotation and adjudication workflow

1. The sampling frame contained 800 page-derived passages.
2. Expert 1 and Expert 2 annotated passages independently for entity spans
   and entity classes.
3. The two workbooks contained 767 and 795 annotated passages, respectively.
   Their shared set contained 763 passages.
4. Six shared records were identical and were accepted directly.
5. Expert 3 adjudicated the 757 records containing disagreements.
6. The archived adjudicated reference contains 763 passages and 18,052
   entity rows.
7. Thirty-seven sampling-frame passages lacked paired annotations and were
   excluded from the adjudicated reference. Their disposition is retained in
   `Supplementary_Data_1_Verification_Tables.xlsx`.

## 2. Archived file roles

- `Dr1.xlsx`: Expert 1 independent annotations.
- `Dr2.xlsx`: Expert 2 independent annotations.
- `Dr3.xlsx`: Expert 3 adjudication of disagreements between Experts 1 and 2.
- `gold_standard.xlsx`: legacy filename for the adjudicated expert reference.
  The filename is retained for provenance; the resource is not described as
  a separately generated gold standard.

## 3. Annotation unit and intended offsets

- The sampling unit is a page-derived passage.
- The annotation unit is a continuous source-text entity mention.
- Each mention is intended to record its label, source text, zero-based start
  offset and end-exclusive end offset.
- Repeated mentions are annotated separately at each source position.
- Punctuation is excluded unless it is an integral part of a source title.
- Unsupported normalisation or completion of historical text is not permitted
  at the span-annotation stage.

## 4. Entity classes

1. **Food ingredient (食材)**: edible plant, animal, mineral, seasoning or
   medicine-food material used in a dietary-therapy context.
2. **Food nature (食性)**: traditional thermal properties, including cold,
   hot, warm, cool and neutral.
3. **Food flavour (食味)**: traditional flavours, including sour, bitter,
   sweet, pungent, salty, bland and astringent.
4. **Meridian tropism (归经)**: an explicit meridian-entry or
   meridian-assignment expression.
5. **Effect (功效)**: an effect stated in the historical source passage.
6. **Disease or symptom (病症)**: a disease, symptom, syndrome or
   pathological condition in the historical source terminology.
7. **Organ (脏腑)**: an independently stated traditional organ or
   triple-burner concept.
8. **Dietary formula (食疗方)**: a complete named dietary formula,
   preparation or therapeutic food.
9. **Cooking method (烹饪方法)**: a preparation, processing, cooking or
   administration method.
10. **Contraindication (禁忌)**: a historical restriction, incompatible
    combination, unsuitable population or adverse-use statement.
11. **Health principle (养生原则)**: a general normative principle of
    dietary health maintenance.
12. **Physician (医家)**: a named historical medical practitioner.
13. **Ancient source (古籍来源)**: an explicitly cited or titled historical
    source.

## 5. Boundary and conflict rules

- Entity spans do not overlap or nest by default.
- A compound effect, disease expression, contraindication or formula name is
  retained as the smallest continuous span with complete meaning.
- Clearly coordinated units are separated when each unit is independently
  meaningful.
- An organ term inside a complete effect or disease span is not annotated
  again as an organ.
- A formula name is annotated as a whole and is not automatically split into
  ingredient mentions.
- Repeated appearances of the same surface form are retained as separate
  mentions.
- Canonicalisation, synonym merging and modern terminology mapping occur
  after span annotation and do not change the source span.

## 6. OCR and exclusion rules

- Isolated OCR symbols, page numbers, collation marks and uninterpretable
  fragments are excluded.
- A readable historical surface form is annotated as displayed; uncertain
  restoration is not inserted into the span.
- Modern pharmacological commentary, Latin names, chemical constituents,
  dosage-only strings and vague references without an identifiable entity
  are excluded from the historical entity layer.
- A passage too corrupted to support reliable span identification is left
  unannotated and recorded in the sampling-frame disposition.

## 7. Adjudication rule

Expert 3 compares both candidate annotations against the same source passage.
The decision prioritises source fidelity, semantic completeness, the
non-overlap rule, the class definitions and the OCR exclusions above. Direct
agreements are accepted without adjudication. The separate expert workbooks
retain the two independent inputs.

## 8. Offset-integrity status

The final release audit found 4,858 rows whose declared offsets exactly select
the declared entity text, 12,385 rows whose entity text occurs elsewhere in
the current passage and 809 rows whose entity text is absent from the current
passage version. The row-level and record-level audit is provided in
`Supplementary_Data_1_Verification_Tables.xlsx`. Automatic nearest-string
realignment is not a substitute for expert confirmation.

## 9. Version record

- Version 1.0 established the first entity and relation definitions.
- Version 2.0 expanded OCR, exclusion and boundary guidance.
- Version 3.0 consolidated the final 13 entity classes for revision.
- Version 3.1 adds the verified offset-integrity limitation and synchronises
  terminology with Revision 12.
