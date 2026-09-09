from __future__ import annotations

"""Render publication-ready KBS result blocks from the frozen evaluation JSON.

The renderer is deliberately read-only with respect to the manuscript. It converts
the machine-readable unified evaluation and the separate matched ECRP experiment
into a Markdown drafting report. No numerical result is embedded in this source
file, so the report cannot silently survive a change in the evaluation artifact.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION = (
    ROOT / "data" / "results" / "final_evaluation.json"
)
DEFAULT_MATCHED_ECRP = ROOT / "data" / "results" / "matched_module_analysis.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "result_blocks.md"

INTERNAL_FINAL_KEY = "FERA-KG unified fusion"
FINAL_NAME = "FERA-KG"

DISPLAY_NAMES = {
    INTERNAL_FINAL_KEY: FINAL_NAME,
    "construction-only diagnostic": "Construction-only diagnostic",
    "cohort-majority baseline": "Cohort-majority baseline",
    "constant SUPPORTED prevalence reference": "Constant-SUPPORTED baseline",
    "fixed pre-existing comparator": "Prespecified comparator",
    "component::fera_ecrp": "FERA-ECRP expert",
    "component::fera_ecrp_sourcebal": "Source-balanced FERA-ECRP expert",
    "component::text_entity": "Entity-marked text expert",
    "component::graph_anchor_wide": "Graph-anchor expert",
    "component::same_backbone_no_ecrp": "Same-backbone model without ECRP",
    "fusion_retrained::full": "Refitted full fusion",
    "fusion_retrained::full_130": "Refitted full fusion",
    "fusion_retrained::without_ecrp_experts": "Fusion without ECRP experts",
    "fusion_retrained::without_text_expert": "Fusion without the text expert",
    "fusion_retrained::without_passage_local_evidence": (
        "Fusion without passage-local evidence"
    ),
    "fusion_retrained::without_raw_construction_and_surface_features": (
        "Fusion without raw construction and surface features"
    ),
    "fusion_retrained::without_graph_experts_and_explicit_graph_diagnostics": (
        "Fusion without graph experts and explicit graph diagnostics"
    ),
    "fusion_retrained::deep_expert_outputs_only": "Fusion using deep-expert outputs only",
}

MATCHED_NAMES = {
    "ecrp_correct": "Reference passage-conditioned ECRP",
    "unconditioned_path": "Unconditioned path attention",
    "counterfactual_off": "Counterfactual ranking loss disabled",
    "no_paths": "Path stream removed",
    "shuffled_passage": "Shuffled-passage condition",
    "empty_passage": "Empty-passage condition",
}


def load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(
            f"ERROR: required {description} is missing: {path}\n"
            "Run the unified inference and evaluation pipeline before rendering "
            "the manuscript result blocks. No values were guessed or substituted."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"ERROR: {description} is not valid JSON: {path}\n{exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"ERROR: {description} must contain a JSON object: {path}")
    return payload


def require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise SystemExit(f"ERROR: missing key {key!r} in {context}")
    return mapping[key]


def finite_number(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"ERROR: expected a number for {context}, found {value!r}") from exc
    if not math.isfinite(number):
        raise SystemExit(f"ERROR: non-finite number for {context}: {number!r}")
    return number


def f4(value: Any) -> str:
    return f"{finite_number(value, 'reported metric'):.4f}"


def f3(value: Any) -> str:
    return f"{finite_number(value, 'reported metric'):.3f}"


def percent(value: Any, digits: int = 1) -> str:
    number = 100.0 * finite_number(value, "reported proportion")
    return f"{number:.{digits}f}%"


def display_name(key: str) -> str:
    if key in DISPLAY_NAMES:
        return DISPLAY_NAMES[key]
    if key.startswith("component::"):
        return key.split("::", 1)[1].replace("_", " ").title()
    if key.startswith("fusion_retrained::"):
        return "Refitted fusion: " + key.split("::", 1)[1].replace("_", " ")
    return key.replace("_", " ").strip().title()


def ci_values(comparison: dict[str, Any]) -> tuple[float, float]:
    raw = require(comparison, "ci95_percentile", "bootstrap comparison")
    if not isinstance(raw, list) or len(raw) != 2:
        raise SystemExit("ERROR: ci95_percentile must be a two-element list")
    return finite_number(raw[0], "CI lower bound"), finite_number(
        raw[1], "CI upper bound"
    )


def bootstrap_description(comparison: dict[str, Any]) -> str:
    """Return a manuscript-facing label for the recorded resampling unit."""
    unit = str(comparison.get("resampling_unit", "grouped")).lower()
    if "cluster" in unit:
        return "passage-cluster bootstrap"
    if "source_work" in unit or "source-work" in unit or "source work" in unit:
        return "paired two-stage source-work bootstrap"
    if "passage" in unit:
        return "passage bootstrap"
    return "grouped bootstrap"


def comparison_clause(comparison: dict[str, Any], comparator_name: str) -> str:
    point = finite_number(
        require(comparison, "point_difference", "bootstrap comparison"),
        "point difference",
    )
    lower, upper = ci_values(comparison)
    interval = "excluded zero" if lower > 0.0 or upper < 0.0 else "included zero"
    return (
        f"The paired difference in Macro-F1 (FERA-KG minus {comparator_name}) was "
        f"{point:.4f} (95% CI [{lower:.4f}, {upper:.4f}], estimated by "
        f"{bootstrap_description(comparison)}). The interval {interval}."
    )


def comparison_for(block: dict[str, Any], comparator_key: str) -> dict[str, Any] | None:
    comparisons = block.get("comparisons", {})
    value = comparisons.get(f"full_minus::{comparator_key}")
    return value if isinstance(value, dict) else None


def metric(block: dict[str, Any], system_key: str) -> dict[str, Any]:
    metrics = require(block, "metrics", "evaluation resource")
    value = require(metrics, system_key, "evaluation-resource metrics")
    if not isinstance(value, dict):
        raise SystemExit(f"ERROR: metrics for {system_key!r} are not an object")
    return value


def metric_summary(values: dict[str, Any]) -> str:
    return (
        f"Macro-F1 {f4(values['macro_f1'])}, accuracy {f4(values['accuracy'])}, "
        f"and three-decision Macro-F1 {f4(values['three_decision_macro_f1'])}"
    )


def optional_metric_line(block: dict[str, Any], system_key: str) -> str | None:
    metrics = block.get("metrics", {})
    if system_key not in metrics:
        return None
    return metric_summary(metrics[system_key])


def best_system(
    block: dict[str, Any], prefix: str, *, exclude: Iterable[str] = ()
) -> tuple[str, dict[str, Any]] | None:
    excluded = set(exclude)
    candidates = [
        (key, values)
        for key, values in block.get("metrics", {}).items()
        if key.startswith(prefix) and key not in excluded
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item[1]["macro_f1"]))


def relation_to_zero(comparison: dict[str, Any] | None) -> str:
    if comparison is None:
        return "was not accompanied by a paired bootstrap interval"
    lower, upper = ci_values(comparison)
    if lower > 0.0:
        return "favoured FERA-KG and excluded zero"
    if upper < 0.0:
        return "favoured the comparator and excluded zero"
    return "included zero"


def matched_reference_advantage_resolved(contrast: dict[str, Any]) -> bool:
    """Apply the prespecified three-part rule for a matched module claim."""
    raw_ci = contrast.get("source_work_bootstrap_95_ci")
    seed_differences = contrast.get("paired_seed_differences")
    ensemble_difference = contrast.get(
        "reference_minus_variant_ensemble_macro_f1"
    )
    if not isinstance(raw_ci, list) or len(raw_ci) != 2:
        return False
    if not isinstance(seed_differences, dict) or len(seed_differences) != 5:
        return False
    if ensemble_difference is None:
        return False
    lower = finite_number(raw_ci[0], "matched CI lower bound")
    upper = finite_number(raw_ci[1], "matched CI upper bound")
    ensemble = finite_number(ensemble_difference, "matched ensemble difference")
    seeds = [
        finite_number(value, f"matched seed difference {seed}")
        for seed, value in seed_differences.items()
    ]
    return lower > 0.0 and upper > 0.0 and ensemble > 0.0 and all(
        value > 0.0 for value in seeds
    )


def primary_blocks(evaluation: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    natural = require(evaluation, "natural", "unified evaluation")
    cohorts = require(natural, "cohorts", "natural evaluation")
    primary = require(cohorts, "primary_natural_600", "natural cohorts")
    difficulty = require(cohorts, "difficulty_track_200", "natural cohorts")
    controlled = require(evaluation, "controlled", "unified evaluation")
    return primary, difficulty, controlled


def abstract_block(evaluation: dict[str, Any]) -> str:
    primary, difficulty, controlled = primary_blocks(evaluation)
    p_final = metric(primary, INTERNAL_FINAL_KEY)
    d_final = metric(difficulty, INTERNAL_FINAL_KEY)
    c_final = metric(controlled, INTERNAL_FINAL_KEY)
    diagnostic = metric(primary, "construction-only diagnostic")
    comparison = comparison_for(primary, "construction-only diagnostic")
    if comparison is None:
        raise SystemExit(
            "ERROR: the primary FERA-KG-versus-construction-diagnostic comparison "
            "is missing; refusing to render a final abstract without its paired interval"
        )
    contrast = (
        f"The construction-only diagnostic reached Macro-F1 "
        f"{f4(diagnostic['macro_f1'])}. "
        + comparison_clause(comparison, "the construction-only diagnostic")
    )
    stress = evaluation.get("stress_tests", {}).get("results", {})
    stress_sentence = ""
    if "tiangongkaiwu" in stress:
        tiangong = metric(stress["tiangongkaiwu"], INTERNAL_FINAL_KEY)
        stress_sentence = (
            f" In the *Tiangong Kaiwu* zero-shot stress test, FERA-KG achieved "
            f"Macro-F1 {f4(tiangong['macro_f1'])}. This boundary test is not evidence "
            "of cross-domain generality."
        )
    return (
        "Knowledge graphs derived from historical Chinese food-therapy texts require "
        "verification against their registered sources. The task must distinguish "
        "textual support, lack of support, "
        "insufficient context, and invalid entity or type assignments. We introduce "
        "the Traditional Chinese Food-Therapy Evidence-Grounded Relation Verification "
        "suite (TCFT-EVR) and Food-Therapy Evidence-Aware Relation Assessment for "
        "Knowledge Graphs (FERA-KG), a provenance-aware hybrid verification framework "
        "that combines textual evidence, target-edge-deleted graph paths, passage-local "
        "evidence, uncertainty estimates, and provenance through source-work-grouped "
        "out-of-fold fusion. On the 600-item primary natural cohort, FERA-KG obtained "
        f"Macro-F1 {f4(p_final['macro_f1'])} and accuracy {f4(p_final['accuracy'])}. "
        f"The separate 200-item difficulty track yielded Macro-F1 "
        f"{f4(d_final['macro_f1'])}, while the 450-item controlled challenge yielded "
        f"{f4(c_final['macro_f1'])}. {contrast}{stress_sentence} Together, TCFT-EVR and "
        "the evaluated FERA-KG system provide an auditable within-task reference, but "
        "they do not support a field-wide state-of-the-art claim or an independently "
        "confirmed gain from the passage-conditioned path module."
    )


def main_evaluation_results(evaluation: dict[str, Any]) -> str:
    primary, difficulty, controlled = primary_blocks(evaluation)
    p_final = metric(primary, INTERNAL_FINAL_KEY)
    d_final = metric(difficulty, INTERNAL_FINAL_KEY)
    c_final = metric(controlled, INTERNAL_FINAL_KEY)
    paragraphs: list[str] = []

    p_diag = metric(primary, "construction-only diagnostic")
    p_majority = metric(primary, "cohort-majority baseline")
    text = (
        f"The primary analysis kept the 600-item natural cohort separate from the "
        f"difficulty track. FERA-KG achieved {metric_summary(p_final)} on this "
        f"cohort. The construction-only diagnostic achieved Macro-F1 "
        f"{f4(p_diag['macro_f1'])} and accuracy {f4(p_diag['accuracy'])}, whereas the "
        f"cohort-majority baseline achieved Macro-F1 {f4(p_majority['macro_f1'])} and "
        f"accuracy {f4(p_majority['accuracy'])}."
    )
    comparison = comparison_for(primary, "construction-only diagnostic")
    if comparison:
        text += " " + comparison_clause(comparison, "the construction-only diagnostic")
    majority_comparison = comparison_for(primary, "cohort-majority baseline")
    if majority_comparison:
        text += " " + comparison_clause(
            majority_comparison, "the cohort-majority baseline"
        )
    if float(p_final["accuracy"]) < float(p_majority["accuracy"]):
        text += (
            " The majority baseline had higher accuracy because the primary cohort was "
            "class-imbalanced, while its low Macro-F1 showed that it did not recover the "
            "minority decisions."
        )
    paragraphs.append(text)

    d_diag = metric(difficulty, "construction-only diagnostic")
    d_majority = metric(difficulty, "cohort-majority baseline")
    text = (
        f"The 200-item difficulty track was analysed as a distinct evaluation resource. "
        f"FERA-KG obtained {metric_summary(d_final)}. The construction-only "
        f"diagnostic and cohort-majority baseline reached Macro-F1 "
        f"{f4(d_diag['macro_f1'])} and {f4(d_majority['macro_f1'])}, respectively."
    )
    comparison = comparison_for(difficulty, "construction-only diagnostic")
    if comparison:
        text += " " + comparison_clause(comparison, "the construction-only diagnostic")
    paragraphs.append(text)

    controlled_comparator = optional_metric_line(controlled, "fixed pre-existing comparator")
    text = (
        f"On the separately constructed 450-item controlled challenge, FERA-KG "
        f"reached {metric_summary(c_final)}."
    )
    if controlled_comparator:
        comparator = metric(controlled, "fixed pre-existing comparator")
        text += (
            f" The prespecified comparator reached Macro-F1 "
            f"{f4(comparator['macro_f1'])} and accuracy {f4(comparator['accuracy'])}."
        )
        comparison = comparison_for(controlled, "fixed pre-existing comparator")
        if comparison:
            text += " " + comparison_clause(
                comparison, "the prespecified comparator"
            )
    controlled_diagnostic = optional_metric_line(
        controlled, "construction-only diagnostic"
    )
    if controlled_diagnostic:
        diagnostic = metric(controlled, "construction-only diagnostic")
        text += (
            f" The passage-free construction-only diagnostic reached Macro-F1 "
            f"{f4(diagnostic['macro_f1'])} and accuracy "
            f"{f4(diagnostic['accuracy'])}."
        )
        comparison = comparison_for(controlled, "construction-only diagnostic")
        if comparison:
            text += " " + comparison_clause(
                comparison, "the construction-only diagnostic"
            )
    text += (
        " Because this controlled resource deliberately encodes diagnostic conditions, "
        "it provides an operational audit rather than the primary estimate of natural "
        "relation-verification performance."
    )
    paragraphs.append(text)
    return "\n\n".join(paragraphs)


def ablation_results(
    evaluation: dict[str, Any], matched: dict[str, Any]
) -> str:
    primary, difficulty, _ = primary_blocks(evaluation)
    paragraphs: list[str] = []

    best_component = best_system(primary, "component::")
    components = [
        (key, values)
        for key, values in primary.get("metrics", {}).items()
        if key.startswith("component::")
    ]
    if best_component:
        key, values = best_component
        text = (
            f"The strongest single component on the primary natural cohort was the "
            f"{display_name(key)}, with Macro-F1 {f4(values['macro_f1'])} and accuracy "
            f"{f4(values['accuracy'])}. FERA-KG reached Macro-F1 "
            f"{f4(metric(primary, INTERNAL_FINAL_KEY)['macro_f1'])}."
        )
        comparison = comparison_for(primary, key)
        if comparison:
            text += " " + comparison_clause(comparison, display_name(key))
        if len(components) > 1:
            ordered = sorted(
                components, key=lambda item: float(item[1]["macro_f1"]), reverse=True
            )
            remaining = ", ".join(
                f"{display_name(key)} {f4(values['macro_f1'])}"
                for key, values in ordered[1:]
            )
            text += f" The remaining component Macro-F1 values were {remaining}."
        paragraphs.append(text)

    fusion_rows = [
        (key, values)
        for key, values in primary.get("metrics", {}).items()
        if key.startswith("fusion_retrained::")
        and key not in {"fusion_retrained::full", "fusion_retrained::full_130"}
    ]
    if fusion_rows:
        final_score = float(metric(primary, INTERNAL_FINAL_KEY)["macro_f1"])
        most_reduced = min(
            fusion_rows, key=lambda item: float(item[1]["macro_f1"])
        )
        best_ablation = max(
            fusion_rows, key=lambda item: float(item[1]["macro_f1"])
        )
        key_low, value_low = most_reduced
        key_high, value_high = best_ablation
        low_comparison = comparison_for(primary, key_low)
        low_delta = final_score - float(value_low["macro_f1"])
        text = (
            f"Fusion-layer ablations refitted the tree-based fusion stage while keeping "
            f"the deep encoders fixed. The {display_name(key_low)} condition produced "
            f"the lowest ablated Macro-F1, {f4(value_low['macro_f1'])}. The "
            f"FERA-KG-minus-ablation difference was {low_delta:.4f}."
        )
        if low_comparison:
            lower, upper = ci_values(low_comparison)
            text += (
                f" Its 95% {bootstrap_description(low_comparison)} interval for the "
                f"difference between FERA-KG and the ablation was "
                f"[{lower:.4f}, {upper:.4f}] and "
                f"{relation_to_zero(low_comparison)}."
            )
        if key_high != key_low:
            text += (
                f" The {display_name(key_high)} condition produced the highest "
                f"ablated Macro-F1, {f4(value_high['macro_f1'])}."
            )
        text += (
            " These are fusion-layer retraining ablations, not end-to-end retraining "
            "experiments. They measure the fitted fusion layer's sensitivity to removing "
            "feature groups; they do not isolate the causal contribution of an encoder."
        )
        paragraphs.append(text)

    matched_tasks = require(matched, "tasks", "matched ECRP analysis")
    reference = require(matched_tasks, "ecrp_correct", "matched ECRP tasks")
    contrasts = require(
        matched, "contrasts_against_ecrp_correct", "matched ECRP analysis"
    )
    variant_scores = [
        (key, values)
        for key, values in matched_tasks.items()
        if key != "ecrp_correct"
    ]
    interval_results = []
    for key, _ in variant_scores:
        contrast = contrasts.get(key, {})
        raw_ci = contrast.get("source_work_bootstrap_95_ci")
        if isinstance(raw_ci, list) and len(raw_ci) == 2:
            lower = float(raw_ci[0])
            upper = float(raw_ci[1])
            interval_results.append((key, lower, upper))
    resolved = [
        key
        for key, _ in variant_scores
        if matched_reference_advantage_resolved(contrasts.get(key, {}))
    ]
    values_text = ", ".join(
        f"{MATCHED_NAMES.get(key, key)} {f4(values['ensemble_macro_f1'])}"
        for key, values in variant_scores
    )
    text = (
        f"A separate matched component study evaluated six ECRP conditions on "
        f"{int(matched.get('items', 0))} items with the same five seeds. The reference "
        f"passage-conditioned ECRP ensemble reached Macro-F1 "
        f"{f4(reference['ensemble_macro_f1'])}; the variant ensemble scores were "
        f"{values_text}."
    )
    if interval_results:
        if resolved:
            text += (
                " The following reference-minus-variant contrasts met all three "
                "prespecified requirements for a positive reference-condition advantage: "
                + ", ".join(MATCHED_NAMES.get(key, key) for key in resolved)
                + ". These contrasts support the corresponding matched interventions "
                "within the component study, but they do not isolate a contribution "
                "inside the final fused system."
            )
        else:
            text += (
                " No contrast combined a positive ensemble difference, positive paired "
                "differences for all five seeds, and a two-stage source-work bootstrap "
                "interval above zero. This component experiment therefore tests the "
                "architectural hypothesis but does not independently confirm a performance "
                "contribution from passage conditioning, path attention, or the "
                "counterfactual term."
            )
    else:
        text += (
            " No complete two-stage source-work bootstrap interval was available for the "
            "matched contrasts, so no independent performance contribution is claimed."
        )
    paragraphs.append(text)

    difficulty_components = best_system(difficulty, "component::")
    if difficulty_components:
        key, values = difficulty_components
        paragraphs.append(
            f"On the separate difficulty track, the single component with the highest "
            f"Macro-F1 was the {display_name(key)}, at {f4(values['macro_f1'])}. This "
            "track was not pooled with the primary natural cohort when selecting or "
            "summarising the main result."
        )
    return "\n\n".join(paragraphs)


def intervention_and_selective_results(evaluation: dict[str, Any]) -> str:
    primary, difficulty, controlled = primary_blocks(evaluation)
    interventions = require(
        require(evaluation, "natural", "unified evaluation"),
        "passage_interventions",
        "natural evaluation",
    )
    primary_interventions = require(
        interventions, "primary_natural_600", "passage interventions"
    )
    final_intervention = primary_interventions.get(INTERNAL_FINAL_KEY)
    paragraphs: list[str] = []

    if isinstance(final_intervention, dict):
        scores = require(final_intervention, "metrics", "passage intervention")
        correct = require(scores, "correct", "passage-intervention metrics")
        shuffled = require(scores, "shuffled", "passage-intervention metrics")
        empty = require(scores, "empty", "passage-intervention metrics")
        changed = final_intervention.get("changed_decisions", {})
        mapc = final_intervention.get("mean_absolute_probability_change", {})
        text = (
            f"With the registered passage, FERA-KG reached Macro-F1 "
            f"{f4(correct['macro_f1'])} on the primary natural cohort. Replacing it with "
            f"a passage from another item yielded {f4(shuffled['macro_f1'])}, and an "
            f"empty-passage input yielded {f4(empty['macro_f1'])}. These interventions "
            "were scored against the original labels and therefore quantify information "
            "dependence rather than counterfactual accuracy."
        )
        if "shuffled" in changed and "empty" in changed:
            text += (
                f" The shuffled and empty interventions changed "
                f"{int(changed['shuffled'])} and {int(changed['empty'])} of the 600 "
                "decisions, respectively."
            )
        if "shuffled" in mapc and "empty" in mapc:
            text += (
                f" Their mean absolute probability changes were "
                f"{f4(mapc['shuffled'])} and {f4(mapc['empty'])}."
            )
        for condition, label in (
            ("correct_minus_shuffled", "shuffled passage"),
            ("correct_minus_empty", "empty passage"),
        ):
            comparison = final_intervention.get(condition)
            if isinstance(comparison, dict):
                point = float(comparison["point_difference"])
                lower, upper = ci_values(comparison)
                text += (
                    f" The correct-minus-{label} Macro-F1 difference was {point:.4f} "
                    f"(95% CI [{lower:.4f}, {upper:.4f}], estimated by "
                    f"{bootstrap_description(comparison)})."
                )
        paragraphs.append(text)

    p_final = metric(primary, INTERNAL_FINAL_KEY)
    p_abstain = require(p_final, "abstention", "primary FERA-KG metrics")
    p_calibration = require(
        require(primary, "probability_metrics", "primary cohort"),
        INTERNAL_FINAL_KEY,
        "primary probability metrics",
    )
    text = (
        f"FERA-KG issued PRESENT or ABSENT decisions for "
        f"{percent(p_abstain['coverage'])} of the primary cohort. Accuracy on those "
        f"decisions was {f4(p_abstain['covered_accuracy'])}. Among items whose reference "
        f"label mapped to ABSTAIN, any-abstention recall was "
        f"{f4(p_abstain['any_abstention_recall'])}, and the "
        f"knowledge-admission error among PRESENT outputs was "
        f"{f4(p_abstain['knowledge_admission_error'])}. Probability quality was "
        f"summarised by expected calibration error (ECE) "
        f"{f4(p_calibration['top_label_ece_15_bins'])}, multiclass Brier score "
        f"{f4(p_calibration['multiclass_brier_sum'])}, negative log-likelihood (NLL) "
        f"{f4(p_calibration['negative_log_likelihood'])}, and the reported risk-coverage "
        f"grid area {f4(p_calibration['risk_coverage_grid_area'])}."
    )
    reason_rows = p_abstain.get("by_reason", {})
    reason_phrases = []
    for key, label in (
        ("INSUFFICIENT_CONTEXT", "insufficient-context"),
        ("ENTITY_OR_TYPE_ERROR", "entity-or-type-error"),
    ):
        row = reason_rows.get(key)
        if isinstance(row, dict) and row.get("support", 0):
            reason_phrases.append(
                f"{label} correct-reason recall {f4(row['correct_reason_recall'])} "
                f"(n={int(row['support'])})"
            )
    if reason_phrases:
        text += " Reason-specific results were " + " and ".join(reason_phrases) + "."
    paragraphs.append(text)

    c_abstain = metric(controlled, INTERNAL_FINAL_KEY).get("abstention", {})
    d_abstain = metric(difficulty, INTERNAL_FINAL_KEY).get("abstention", {})
    if c_abstain and d_abstain:
        paragraphs.append(
            f"The difficulty track had coverage {percent(d_abstain['coverage'])} and "
            f"covered accuracy {f4(d_abstain['covered_accuracy'])}; the controlled "
            f"challenge had coverage {percent(c_abstain['coverage'])} and covered "
            f"accuracy {f4(c_abstain['covered_accuracy'])}. Reporting these resources "
            "separately prevents the abundant diagnostic abstentions in the controlled "
            "challenge from masking behaviour on naturally occurring cases."
        )
    return "\n\n".join(paragraphs)


def stress_results(evaluation: dict[str, Any]) -> str:
    stress_root = require(evaluation, "stress_tests", "unified evaluation")
    resources = require(stress_root, "results", "stress-test evaluation")
    paragraphs: list[str] = []

    external = require(
        evaluation, "external_protocol_audit", "unified evaluation"
    )
    if external.get("not_fera_kg_external_validation") is not True:
        raise SystemExit(
            "ERROR: the WTR audit is not explicitly separated from FERA-KG external validation"
        )
    outputs = require(external, "audited_outputs", "WTR protocol audit")
    semantic = require(outputs, "semantic_verifier", "WTR protocol audit")
    artifact = require(outputs, "artifact_only", "WTR protocol audit")
    bootstrap = require(
        external, "source_domain_bootstrap", "WTR protocol audit"
    )
    wtr_interval = bootstrap.get("semantic_minus_artifact_macro_f1_ci95")
    if not isinstance(wtr_interval, list) or len(wtr_interval) != 2:
        raise SystemExit("ERROR: WTR protocol-audit comparison interval is missing")
    paragraphs.append(
        f"A separate protocol audit used {int(external['items'])} public ProVe "
        f"Web-Text Relation items from {int(external['source_domains'])} source "
        f"domains and {int(external['properties'])} properties. The fixed-prompt "
        f"semantic verifier obtained three-class Macro-F1 "
        f"{f4(semantic['macro_f1'])}, compared with {f4(artifact['macro_f1'])} for "
        f"the source-domain-grouped artifact-only diagnostic. The 95% source-domain "
        f"bootstrap interval for their difference was "
        f"[{float(wtr_interval[0]):.4f}, {float(wtr_interval[1]):.4f}]. This audit "
        "uses an approximate three-label mapping on English web evidence and does "
        "not include the entity-or-type-error class. The Chinese FERA-KG encoder "
        "was not run, so this result is not external validation of FERA-KG."
    )

    ic = resources.get("ic_stress")
    if isinstance(ic, dict):
        final = metric(ic, INTERNAL_FINAL_KEY)
        abstention = final.get("abstention", {})
        reason = abstention.get("by_reason", {}).get("INSUFFICIENT_CONTEXT", {})
        text = (
            f"The {int(ic.get('items', final.get('items', 0)))}-item insufficient-context "
            f"stress set yielded FERA-KG Macro-F1 "
            f"{f4(final['macro_f1'])} and accuracy {f4(final['accuracy'])}."
        )
        if reason and reason.get("support", 0):
            text += (
                f" Correct insufficient-context reason recall was "
                f"{f4(reason['correct_reason_recall'])}, and the false-PRESENT rate for "
                f"these cases was {f4(reason['false_present_rate'])}."
            )
        comparator = ic.get("metrics", {}).get("fixed pre-existing comparator")
        if comparator:
            text += (
                f" The prespecified comparator reached Macro-F1 "
                f"{f4(comparator['macro_f1'])}."
            )
            comparison = comparison_for(ic, "fixed pre-existing comparator")
            if comparison:
                text += " " + comparison_clause(
                    comparison, "the prespecified comparator"
                )
        paragraphs.append(text)

    tiangong = resources.get("tiangongkaiwu")
    if isinstance(tiangong, dict):
        final = metric(tiangong, INTERNAL_FINAL_KEY)
        text = (
            f"On the {int(tiangong.get('items', final.get('items', 0)))}-item "
            f"*Tiangong Kaiwu* zero-shot domain-shift stress set, FERA-KG obtained "
            f"Macro-F1 {f4(final['macro_f1'])} and accuracy {f4(final['accuracy'])}."
        )
        comparator = tiangong.get("metrics", {}).get("fixed pre-existing comparator")
        if comparator:
            text += (
                f" The prespecified comparator obtained Macro-F1 "
                f"{f4(comparator['macro_f1'])}."
            )
            comparison = comparison_for(tiangong, "fixed pre-existing comparator")
            if comparison:
                text += " " + comparison_clause(
                    comparison, "the prespecified comparator"
                )
        text += (
            " This experiment evaluates a domain boundary under an approximate task "
            "mapping. It is not an in-domain benchmark and does not establish "
            "cross-domain generality."
        )
        paragraphs.append(text)
    return "\n\n".join(paragraphs)


def discussion_block(
    evaluation: dict[str, Any], matched: dict[str, Any]
) -> str:
    primary, difficulty, controlled = primary_blocks(evaluation)
    p_final = metric(primary, INTERNAL_FINAL_KEY)
    p_diag = metric(primary, "construction-only diagnostic")
    p_comparison = comparison_for(primary, "construction-only diagnostic")
    matched_contrasts = matched.get("contrasts_against_ecrp_correct", {})
    matched_resolved = [
        key
        for key, values in matched_contrasts.items()
        if matched_reference_advantage_resolved(values)
    ]

    if p_comparison is None:
        construction_interpretation = (
            "The available report does not contain the required two-stage source-work "
            "bootstrap interval against "
            "the construction-only diagnostic, so no resolved advantage should be claimed."
        )
    else:
        lower, upper = ci_values(p_comparison)
        point = float(p_comparison["point_difference"])
        bootstrap = bootstrap_description(p_comparison)
        if lower > 0.0:
            construction_interpretation = (
                f"FERA-KG exceeded the construction-only diagnostic by {point:.4f} "
                f"Macro-F1, and the 95% CI estimated by {bootstrap} excluded zero. "
                "This is evidence of a within-task gain beyond the measured construction "
                "cues, although it does not identify which subsystem produced the gain."
            )
        elif upper < 0.0:
            construction_interpretation = (
                f"The construction-only diagnostic exceeded FERA-KG by "
                f"{abs(point):.4f} Macro-F1, and the 95% CI estimated by {bootstrap} "
                "excluded zero. The primary result therefore does not demonstrate added "
                "predictive value from the evidence-aware pipeline."
            )
        else:
            construction_interpretation = (
                f"FERA-KG and the construction-only diagnostic differed by "
                f"{point:.4f} Macro-F1, but the 95% CI estimated by {bootstrap} included "
                "zero. The main experiment therefore does not resolve whether the full "
                "pipeline adds predictive value beyond measured construction cues."
            )

    first = (
        f"The primary natural-cohort result, Macro-F1 {f4(p_final['macro_f1'])}, should "
        f"be interpreted as the performance of the complete FERA-KG system on TCFT-EVR. "
        f"{construction_interpretation} The 200-item difficulty "
        f"track and 450-item controlled challenge answer different questions and remain "
        f"separate from the primary estimate. Their Macro-F1 values were "
        f"{f4(metric(difficulty, INTERNAL_FINAL_KEY)['macro_f1'])} and "
        f"{f4(metric(controlled, INTERNAL_FINAL_KEY)['macro_f1'])}, respectively."
    )
    controlled_diagnostic = controlled.get("metrics", {}).get(
        "construction-only diagnostic"
    )
    if isinstance(controlled_diagnostic, dict):
        first += (
            f" The controlled construction-only diagnostic reached Macro-F1 "
            f"{f4(controlled_diagnostic['macro_f1'])}, which shows that the targeted "
            "construction remains partly predictable without reading the passage."
        )

    if matched_resolved:
        architecture = (
            "At least one matched contrast met the prespecified three-part criterion for "
            "a positive reference-condition advantage, but this separate study does not "
            "isolate the contribution inside the final fused system."
        )
    else:
        architecture = (
            "No matched contrast combined a positive ensemble difference, positive paired "
            "differences for all five seeds, and a two-stage source-work bootstrap interval "
            "above zero. Accordingly, FERA-ECRP is presented as a reproducible "
            "passage-conditioned relational architecture whose independent performance "
            "contribution remains unconfirmed, not as a validated new neural reasoning "
            "mechanism."
        )
    second = (
        architecture
        + " Passage replacement and removal provide complementary information-dependence "
        "diagnostics. Because their labels were not re-adjudicated after intervention, they "
        "cannot be interpreted as counterfactual correctness tests. Fusion-layer "
        "ablations likewise assess how a fixed set of encoder outputs is used, rather than "
        "the end-to-end causal contribution of each encoder."
    )

    stress_resources = evaluation.get("stress_tests", {}).get("results", {})
    limits: list[str] = []
    ic = stress_resources.get("ic_stress")
    if isinstance(ic, dict):
        reason = (
            metric(ic, INTERNAL_FINAL_KEY)
            .get("abstention", {})
            .get("by_reason", {})
            .get("INSUFFICIENT_CONTEXT", {})
        )
        if reason and reason.get("support", 0):
            limits.append(
                f"correct insufficient-context reason recall was "
                f"{f4(reason['correct_reason_recall'])} in the dedicated stress set"
            )
    tiangong = stress_resources.get("tiangongkaiwu")
    if isinstance(tiangong, dict):
        limits.append(
            f"zero-shot *Tiangong Kaiwu* Macro-F1 was "
            f"{f4(metric(tiangong, INTERNAL_FINAL_KEY)['macro_f1'])}"
        )
    boundary = " and ".join(limits)
    third = (
        "The abstention, calibration, and admission analyses describe operational "
        "behaviour but do not constitute a clinical safety evaluation."
    )
    if boundary:
        third += f" In particular, {boundary}."
    third += (
        " These boundary tests motivate more balanced natural error collection, stronger "
        "source-shift controls, and ontology-aware adaptation before the framework is "
        "applied outside historical Chinese food-therapy texts."
    )
    return "\n\n".join((first, second, third))


def conclusion_block(evaluation: dict[str, Any]) -> str:
    primary, _, _ = primary_blocks(evaluation)
    final = metric(primary, INTERNAL_FINAL_KEY)
    comparison = comparison_for(primary, "construction-only diagnostic")
    resolved = relation_to_zero(comparison)
    return (
        "TCFT-EVR provides provenance-linked resources for four-way relation verification "
        "in historical Chinese food-therapy texts, and FERA-KG supplies a fully specified "
        "verification framework. On the primary 600-item natural cohort, the system "
        f"reached Macro-F1 {f4(final['macro_f1'])}. Its comparison with the construction-only "
        f"diagnostic {resolved}. Component, intervention, calibration, and stress analyses "
        "document the system's limitations, but they do not establish an independent "
        "gain from the passage-conditioned path module or generalisation across cultural "
        "heritage domains. The principal contribution is therefore a source-grounded task, "
        "a reproducible evaluation protocol, and a reference implementation for subsequent "
        "work on evidence-sensitive relation verification."
    )


def table_one(evaluation: dict[str, Any]) -> str:
    primary, difficulty, controlled = primary_blocks(evaluation)
    rows = [
        "| Resource | System | n | Macro-F1 | Accuracy | Three-decision Macro-F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    specifications = (
        (
            "Primary natural cohort",
            primary,
            (
                INTERNAL_FINAL_KEY,
                "construction-only diagnostic",
                "cohort-majority baseline",
            ),
        ),
        (
            "Difficulty track",
            difficulty,
            (
                INTERNAL_FINAL_KEY,
                "construction-only diagnostic",
                "cohort-majority baseline",
            ),
        ),
        (
            "Controlled challenge",
            controlled,
            (
                INTERNAL_FINAL_KEY,
                "construction-only diagnostic",
                "fixed pre-existing comparator",
                "constant SUPPORTED prevalence reference",
            ),
        ),
    )
    for resource, block, keys in specifications:
        for key in keys:
            values = block.get("metrics", {}).get(key)
            if not isinstance(values, dict):
                continue
            rows.append(
                f"| {resource} | {display_name(key)} | {int(values['items'])} | "
                f"{f4(values['macro_f1'])} | {f4(values['accuracy'])} | "
                f"{f4(values['three_decision_macro_f1'])} |"
            )
    return "\n".join(rows)


def table_two(evaluation: dict[str, Any], matched: dict[str, Any]) -> str:
    primary, _, _ = primary_blocks(evaluation)
    rows = [
        "| Analysis | Condition | Macro-F1 | Final/reference minus condition | 95% grouped CI |",
        "|---|---|---:|---:|---:|",
    ]
    final_score = float(metric(primary, INTERNAL_FINAL_KEY)["macro_f1"])
    for key, values in primary.get("metrics", {}).items():
        if not (key.startswith("component::") or key.startswith("fusion_retrained::")):
            continue
        if key in {"fusion_retrained::full", "fusion_retrained::full_130"}:
            continue
        comparison = comparison_for(primary, key)
        if comparison:
            delta = f4(comparison["point_difference"])
            lower, upper = ci_values(comparison)
            ci = f"[{lower:.4f}, {upper:.4f}]"
        else:
            delta = f"{final_score - float(values['macro_f1']):.4f}"
            ci = "not reported"
        analysis = "Single component" if key.startswith("component::") else "Fusion refit"
        rows.append(
            f"| {analysis} | {display_name(key)} | {f4(values['macro_f1'])} | "
            f"{delta} | {ci} |"
        )

    tasks = matched.get("tasks", {})
    contrasts = matched.get("contrasts_against_ecrp_correct", {})
    for key, values in tasks.items():
        if key == "ecrp_correct":
            rows.append(
                f"| Matched ECRP study | {MATCHED_NAMES[key]} | "
                f"{f4(values['ensemble_macro_f1'])} | reference | reference |"
            )
            continue
        contrast = contrasts.get(key, {})
        delta = contrast.get("reference_minus_variant_ensemble_macro_f1")
        raw_ci = contrast.get("source_work_bootstrap_95_ci")
        ci = (
            f"[{float(raw_ci[0]):.4f}, {float(raw_ci[1]):.4f}]"
            if isinstance(raw_ci, list) and len(raw_ci) == 2
            else "not reported"
        )
        rows.append(
            f"| Matched ECRP study | {MATCHED_NAMES.get(key, key)} | "
            f"{f4(values['ensemble_macro_f1'])} | "
            f"{f4(delta) if delta is not None else 'not reported'} | {ci} |"
        )
    return "\n".join(rows)


def claim_boundaries(evaluation: dict[str, Any], matched: dict[str, Any]) -> str:
    primary, _, _ = primary_blocks(evaluation)
    diagnostic_comparison = comparison_for(primary, "construction-only diagnostic")
    claims = [
        "Report the 600-item primary natural cohort and 200-item difficulty track separately; do not pool them into a main score.",
        "Describe all rankings as same-task, same-resource comparisons. Do not call the result field-wide state of the art.",
        "Call *Tiangong Kaiwu* a zero-shot domain-shift stress test, not an external validation dataset.",
        "Treat shuffled- and empty-passage results as information-dependence diagnostics because the intervened inputs were not independently re-adjudicated.",
        "Describe the fusion refits as fusion-layer retraining with fixed deep encoders, not complete end-to-end ablations.",
        "Present the matched ECRP experiment as a separate component study; it does not replace evaluation of the final fused system.",
        "Do not interpret abstention metrics as clinical safety or medical decision-support evidence.",
    ]
    if diagnostic_comparison is None or relation_to_zero(diagnostic_comparison) == "included zero":
        claims.append(
            "Do not claim that FERA-KG outperforms the construction-only diagnostic; the two-stage source-work bootstrap comparison is unresolved."
        )
    else:
        lower, upper = ci_values(diagnostic_comparison)
        if upper < 0.0:
            claims.append(
                "State that the construction-only diagnostic performed better on the primary cohort; do not attribute the final score to evidence semantics."
            )
        elif lower > 0.0:
            claims.append(
                "A within-task advantage over the construction-only diagnostic may be reported, but it does not isolate the responsible model component."
            )
    contrasts = matched.get("contrasts_against_ecrp_correct", {})
    if contrasts and not any(
        matched_reference_advantage_resolved(value) for value in contrasts.values()
    ):
        claims.append(
            "Do not claim that passage conditioning, graph paths, or the counterfactual loss has an independently confirmed performance contribution."
        )
    return "\n".join(f"- {claim}" for claim in claims)


def render(evaluation: dict[str, Any], matched: dict[str, Any]) -> str:
    status = evaluation.get("status")
    if status != "UNIFIED_FINAL_EVALUATION_COMPLETE":
        raise SystemExit(
            "ERROR: unified evaluation does not have status "
            f"UNIFIED_FINAL_EVALUATION_COMPLETE; found {status!r}"
        )
    abstract = abstract_block(evaluation)
    abstract_words = len(abstract.replace("-", " ").split())
    if abstract_words > 250:
        raise SystemExit(
            f"ERROR: generated abstract has {abstract_words} words, exceeding the KBS limit"
        )
    sections = [
        "# Unified KBS result blocks",
        "",
        "> Generated directly from the frozen unified evaluation artifact. These blocks use **FERA-KG** throughout and keep the primary natural cohort, difficulty track, and controlled challenge separate.",
        "",
        "## Abstract result block",
        "",
        abstract,
        "",
        f"Abstract word count: {abstract_words}",
        "",
        "## Results 1. Performance on the registered evaluation resources",
        "",
        main_evaluation_results(evaluation),
        "",
        "## Results 2. Component and fusion analyses",
        "",
        ablation_results(evaluation, matched),
        "",
        "## Results 3. Passage dependence, selective prediction, and calibration",
        "",
        intervention_and_selective_results(evaluation),
        "",
        "## Results 4. Stress tests and transfer boundary",
        "",
        stress_results(evaluation),
        "",
        "## Discussion",
        "",
        discussion_block(evaluation, matched),
        "",
        "## Conclusion",
        "",
        conclusion_block(evaluation),
        "",
        "## Table 1. Main same-resource comparisons",
        "",
        table_one(evaluation),
        "",
        "## Table 2. Component and fusion analyses",
        "",
        table_two(evaluation, matched),
        "",
        "## Claim boundaries for final editorial use",
        "",
        claim_boundaries(evaluation, matched),
        "",
    ]
    return "\n".join(sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render KBS manuscript result blocks from frozen unified results."
    )
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--matched-ecrp", type=Path, default=DEFAULT_MATCHED_ECRP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation = load_json(args.evaluation, "unified final evaluation JSON")
    matched = load_json(args.matched_ecrp, "matched ECRP analysis JSON")
    rendered = render(evaluation, matched)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
