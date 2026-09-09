"""Generate final KBS main-text Figures 3--6 from the unified evaluation.

The script deliberately contains no result values.  Every plotted number is
read from ``results/unified_final_v1/unified_final_evaluation_v1.json``.  It
also writes machine-readable figure-source tables and accessibility metadata.

Outputs
-------
* 600-dpi PNG, PDF and SVG files in ``kbs_revision_20260908/figures_final``
* a copy of each PNG in ``manuscript/figures``
* one source-data CSV per figure
* ``unified_kbs_figure_metadata.csv`` with source and alt-text fields
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "results" / "final_evaluation.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "figures"
MANUSCRIPT_FIGURES = ROOT / "outputs" / "figures"

FINAL_SYSTEM = "FERA-KG unified fusion"
TRACKS = (
    ("Primary natural", "primary_natural_600", 600),
    ("Difficulty track", "difficulty_track_200", 200),
    ("Controlled challenge", "controlled", 450),
)
STRESS_TRACKS = (
    ("Insufficient-context stress", "ic_stress", 150),
    ("Tiangong Kaiwu zero-shot", "tiangongkaiwu", 356),
)

# Restrained palette shared with the other final manuscript figures.
BLUE = "#3F72B8"
TEAL = "#27867E"
ORANGE = "#D9822B"
PURPLE = "#7559A6"
RED = "#C84A3A"
GREEN = "#33865A"
NAVY = "#142B4A"
INK = "#202938"
MUTED = "#657184"
GRID = "#DCE3EA"
LIGHT = "#F5F7FA"
WHITE = "#FFFFFF"

LABEL_COLORS = {
    "SUPPORTED": GREEN,
    "NOT_SUPPORTED": RED,
    "INSUFFICIENT_CONTEXT": ORANGE,
    "ENTITY_OR_TYPE_ERROR": PURPLE,
}
LABEL_DISPLAY = {
    "SUPPORTED": "Supported",
    "NOT_SUPPORTED": "Not supported",
    "INSUFFICIENT_CONTEXT": "Insufficient\ncontext",
    "ENTITY_OR_TYPE_ERROR": "Entity/type\nerror",
}
TRACK_COLORS = {
    "Primary natural": BLUE,
    "Difficulty track": TEAL,
    "Controlled challenge": PURPLE,
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9.2,
        "axes.labelsize": 9.6,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": WHITE,
    }
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_unit(value: Any, context: str, *, allow_none: bool = True) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{context} is not a finite number: {value!r}")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{context} is outside [0, 1]: {value!r}")


def assert_metric_block(
    metric: dict[str, Any], expected_items: int, labels: list[str], context: str
) -> None:
    if int(metric.get("items", -1)) != expected_items:
        raise ValueError(
            f"{context}: metric item count {metric.get('items')} != {expected_items}"
        )
    for key in (
        "accuracy",
        "macro_f1",
        "three_decision_macro_f1",
        "three_decision_accuracy",
    ):
        finite_unit(metric.get(key), f"{context}.{key}", allow_none=False)
    distribution = metric.get("label_distribution", {})
    if sum(int(value) for value in distribution.values()) != expected_items:
        raise ValueError(f"{context}: label distribution does not sum to item count")
    per_class = metric.get("per_class", {})
    if not all(label in per_class for label in labels):
        raise ValueError(f"{context}: one or more fixed labels are missing")
    if sum(int(per_class[label].get("support", -1)) for label in labels) != expected_items:
        raise ValueError(f"{context}: per-class supports do not sum to item count")
    matrix = metric.get("confusion_matrix", {})
    matrix_total = sum(
        int(matrix.get(gold, {}).get(predicted, 0))
        for gold in labels
        for predicted in labels
    )
    if matrix_total != expected_items:
        raise ValueError(f"{context}: confusion matrix does not sum to item count")
    for label in labels:
        for key in ("precision", "recall", "f1"):
            finite_unit(
                per_class[label].get(key), f"{context}.per_class.{label}.{key}",
                allow_none=False,
            )
    abstention = metric.get("abstention", {})
    for key in (
        "any_abstention_recall",
        "false_present_rate_on_gold_abstention",
        "coverage",
        "covered_accuracy",
        "knowledge_admission_error",
        "supported_recall",
    ):
        finite_unit(abstention.get(key), f"{context}.abstention.{key}")


def validate_payload(payload: dict[str, Any]) -> list[str]:
    if payload.get("status") != "UNIFIED_FINAL_EVALUATION_COMPLETE":
        raise ValueError("Unified evaluation is not marked complete")
    labels = list(payload.get("inferential_scope", {}).get("fixed_label_set", []))
    expected_labels = set(LABEL_COLORS)
    if set(labels) != expected_labels or len(labels) != len(expected_labels):
        raise ValueError(f"Unexpected fixed label set: {labels!r}")

    natural = payload.get("natural", {}).get("cohorts", {})
    for _, key, count in TRACKS[:2]:
        block = natural.get(key)
        if not isinstance(block, dict) or int(block.get("items", -1)) != count:
            raise ValueError(f"Natural cohort {key} is missing or is not n={count}")
        for name, metric in block.get("metrics", {}).items():
            assert_metric_block(metric, count, labels, f"{key}.{name}")
        if FINAL_SYSTEM not in block.get("metrics", {}):
            raise ValueError(f"{key} lacks {FINAL_SYSTEM}")

    controlled = payload.get("controlled", {})
    if int(controlled.get("items", -1)) != 450:
        raise ValueError("Controlled challenge is missing or is not n=450")
    for name, metric in controlled.get("metrics", {}).items():
        assert_metric_block(metric, 450, labels, f"controlled.{name}")
    if FINAL_SYSTEM not in controlled.get("metrics", {}):
        raise ValueError(f"controlled lacks {FINAL_SYSTEM}")

    stress = payload.get("stress_tests", {}).get("results", {})
    for _, key, count in STRESS_TRACKS:
        block = stress.get(key)
        if not isinstance(block, dict) or int(block.get("items", -1)) != count:
            raise ValueError(f"Stress resource {key} is missing or is not n={count}")
        for name, metric in block.get("metrics", {}).items():
            assert_metric_block(metric, count, labels, f"stress.{key}.{name}")
        if FINAL_SYSTEM not in block.get("metrics", {}):
            raise ValueError(f"stress.{key} lacks {FINAL_SYSTEM}")

    # Probability diagnostics must correspond to an available classifier and
    # retain valid coverage coordinates.
    all_blocks = [natural[key] for _, key, _ in TRACKS[:2]] + [controlled]
    all_blocks += [stress[key] for _, key, _ in STRESS_TRACKS]
    for block in all_blocks:
        for name, calibration in block.get("probability_metrics", {}).items():
            if name not in block.get("metrics", {}):
                raise ValueError(f"Calibration system {name} has no metric block")
            for key in (
                "negative_log_likelihood",
                "multiclass_brier_sum",
                "top_label_ece_15_bins",
                "mean_confidence",
                "risk_coverage_grid_area",
            ):
                value = calibration.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"Invalid calibration value {name}.{key}: {value!r}")
            last_coverage = -1.0
            for point in calibration.get("risk_coverage_grid", []):
                finite_unit(point.get("coverage"), "risk coverage", allow_none=False)
                finite_unit(point.get("accuracy"), "risk accuracy", allow_none=False)
                finite_unit(point.get("risk"), "risk", allow_none=False)
                coverage = float(point["coverage"])
                if coverage <= last_coverage:
                    raise ValueError("Risk-coverage points are not strictly ordered")
                last_coverage = coverage
    return labels


def get_track_blocks(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    cohorts = payload["natural"]["cohorts"]
    return [
        ("Primary natural", cohorts["primary_natural_600"]),
        ("Difficulty track", cohorts["difficulty_track_200"]),
        ("Controlled challenge", payload["controlled"]),
    ]


def short_system(name: str) -> str:
    replacements = {
        FINAL_SYSTEM: "FERA-KG",
        "construction-only diagnostic": "Construction diagnostic",
        "cohort-majority baseline": "Majority baseline",
        "constant SUPPORTED prevalence reference": "Constant supported",
        "fixed pre-existing comparator": "Fixed comparator",
        "component::fera_ecrp": "FERA-ECRP",
        "component::fera_ecrp_sourcebal": "Source-balanced ECRP",
        "component::text_entity": "Entity-marked text",
        "component::graph_anchor_wide": "Graph anchor",
        "component::same_backbone_no_ecrp": "Same backbone, no ECRP",
        "fusion_retrained::full_130": "Full fusion",
        "fusion_retrained::without_ecrp_experts": "Without ECRP experts",
        "fusion_retrained::without_text_expert": "Without text expert",
        "fusion_retrained::without_passage_local_evidence": "Without passage evidence",
        "fusion_retrained::without_raw_construction_and_surface_features": "Without construction/surface",
        "fusion_retrained::without_graph_experts_and_explicit_graph_diagnostics": "Without graph experts",
        "fusion_retrained::deep_expert_outputs_only": "Deep outputs only",
    }
    if name in replacements:
        return replacements[name]
    value = name.replace("component::", "").replace("fusion_retrained::", "")
    return value.replace("_", " ").strip().title()


def strongest_registered_reference(block: dict[str, Any]) -> str:
    metrics = block["metrics"]
    preferred = [
        "fixed pre-existing comparator",
        "construction-only diagnostic",
        "cohort-majority baseline",
        "constant SUPPORTED prevalence reference",
    ]
    candidates = [name for name in preferred if name in metrics]
    if not candidates:
        candidates = [
            name
            for name in metrics
            if name != FINAL_SYSTEM
            and not name.startswith("component::")
            and not name.startswith("fusion_retrained::")
        ]
    if not candidates:
        candidates = [name for name in metrics if name != FINAL_SYSTEM]
    if not candidates:
        raise ValueError("No comparison system is available")
    return max(candidates, key=lambda name: float(metrics[name]["macro_f1"]))


def clean_axis(ax: plt.Axes, *, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="both", colors=INK, width=0.8, length=3.0)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.75, alpha=0.9)


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(title, loc="left", color=NAVY, fontweight="bold", pad=9)
    ax.text(
        -0.13,
        1.035,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13.5,
        fontweight="bold",
        color=NAVY,
        clip_on=False,
    )


def percent_axis(ax: plt.Axes, *, maximum: float = 1.0) -> None:
    ax.set_ylim(0.0, maximum)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def source_row(
    rows: list[dict[str, Any]],
    figure: int,
    panel_letter: str,
    evaluation: str,
    system: str,
    metric: str,
    value: Any,
    *,
    label: str = "",
    n: Any = "",
    ci: Iterable[Any] | None = None,
) -> None:
    lower, upper = ("", "") if ci is None else tuple(ci)
    rows.append(
        {
            "figure": f"Fig. {figure}",
            "panel": panel_letter,
            "evaluation": evaluation,
            "system": system,
            "metric": metric,
            "label_or_condition": label,
            "value": value,
            "n": n,
            "ci95_lower": lower,
            "ci95_upper": upper,
        }
    )


def new_canvas() -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.75))
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.095,
        top=0.905,
        wspace=0.30,
        hspace=0.58,
    )
    return fig, axes


def legend_above(
    ax: plt.Axes,
    *,
    ncol: int,
    fontsize: float,
    handles: list[Any] | None = None,
) -> None:
    kwargs: dict[str, Any] = {
        "loc": "lower right",
        "bbox_to_anchor": (1.0, 1.105),
        "ncol": ncol,
        "frameon": False,
        "fontsize": fontsize,
        "borderaxespad": 0.0,
        "columnspacing": 1.0,
        "handletextpad": 0.45,
    }
    if handles is not None:
        kwargs["handles"] = handles
    ax.legend(**kwargs)


def figure3(payload: dict[str, Any], labels: list[str]) -> tuple[plt.Figure, list[dict[str, Any]], str]:
    fig, axes = new_canvas()
    rows: list[dict[str, Any]] = []
    blocks = get_track_blocks(payload)

    ax = axes[0, 0]
    y = np.arange(len(blocks))
    left = np.zeros(len(blocks), dtype=float)
    for label in labels:
        values = []
        for track, block in blocks:
            metric = block["metrics"][FINAL_SYSTEM]
            count = int(metric["label_distribution"].get(label, 0))
            value = count / int(block["items"])
            values.append(value)
            source_row(rows, 3, "a", track, "Human reference", "label share", value, label=label, n=count)
        ax.barh(y, values, left=left, color=LABEL_COLORS[label], height=0.52, edgecolor=WHITE, linewidth=0.7, label=LABEL_DISPLAY[label].replace("\n", " "))
        left += np.asarray(values)
    ax.set_yticks(y, [name for name, _ in blocks])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of reference labels")
    clean_axis(ax, grid="x")
    panel(ax, "a", "Reference-label composition")
    legend_above(ax, ncol=2, fontsize=7.4)

    ax = axes[0, 1]
    for index, (track, block) in enumerate(blocks):
        reference = strongest_registered_reference(block)
        full = float(block["metrics"][FINAL_SYSTEM]["macro_f1"])
        baseline = float(block["metrics"][reference]["macro_f1"])
        ax.plot([baseline, full], [index, index], color=GRID, linewidth=4.0, solid_capstyle="round", zorder=1)
        ax.scatter(baseline, index, color=ORANGE, marker="s", s=58, edgecolor=WHITE, linewidth=0.8, zorder=3)
        ax.scatter(full, index, color=BLUE, marker="o", s=66, edgecolor=WHITE, linewidth=0.8, zorder=3)
        ax.text(min(1.0, max(full, baseline) + 0.018), index, f"{full-baseline:+.3f}", color=NAVY, va="center", fontsize=8.0, fontweight="bold")
        source_row(rows, 3, "b", track, FINAL_SYSTEM, "Macro-F1", full, n=block["items"])
        source_row(rows, 3, "b", track, reference, "Macro-F1", baseline, n=block["items"])
    ax.set_yticks(np.arange(len(blocks)), [name for name, _ in blocks])
    ax.invert_yaxis()
    ax.set_xlabel("Four-class Macro-F1")
    ax.set_xlim(0, max(0.7, ax.get_xlim()[1]))
    clean_axis(ax, grid="x")
    reference_names = {
        strongest_registered_reference(block) for _, block in blocks
    }
    reference_label = (
        short_system(next(iter(reference_names)))
        if len(reference_names) == 1
        else "Strongest registered reference"
    )
    panel(ax, "b", "Within-track system comparison")
    legend_above(
        ax,
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=WHITE, markersize=8, label="FERA-KG"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=ORANGE, markeredgecolor=WHITE, markersize=8, label=reference_label),
        ],
        ncol=2,
        fontsize=7.5,
    )

    ax = axes[1, 0]
    x = np.arange(len(labels))
    offsets = np.linspace(-0.18, 0.18, len(blocks))
    for offset, (track, block) in zip(offsets, blocks):
        metric = block["metrics"][FINAL_SYSTEM]
        values = [float(metric["per_class"][label]["f1"]) for label in labels]
        ax.plot(x + offset, values, color=TRACK_COLORS[track], marker="o", markersize=5.6, label=track)
        for label, value in zip(labels, values):
            source_row(rows, 3, "c", track, FINAL_SYSTEM, "class F1", value, label=label, n=metric["per_class"][label]["support"])
    ax.set_xticks(x, [LABEL_DISPLAY[label] for label in labels])
    ax.set_ylabel("Class F1")
    ax.set_ylim(0, 1.02)
    clean_axis(ax)
    panel(ax, "c", "FERA-KG performance by decision class")
    legend_above(ax, ncol=3, fontsize=7.2)

    ax = axes[1, 1]
    metric_defs = (
        ("Any-abstention\nrecall", "any_abstention_recall"),
        ("False-present\nrate", "false_present_rate_on_gold_abstention"),
        ("Coverage", "coverage"),
        ("Admission\nerror", "knowledge_admission_error"),
    )
    x = np.arange(len(metric_defs))
    width = 0.22
    for index, (track, block) in enumerate(blocks):
        abstention = block["metrics"][FINAL_SYSTEM]["abstention"]
        values = [abstention.get(key) for _, key in metric_defs]
        plot_values = [np.nan if value is None else float(value) for value in values]
        ax.bar(x + (index - 1) * width, plot_values, width=width, color=TRACK_COLORS[track], edgecolor=WHITE, linewidth=0.6, label=track)
        for (label, key), value in zip(metric_defs, values):
            source_row(rows, 3, "d", track, FINAL_SYSTEM, key, value, label=label.replace("\n", " "), n=block["items"])
    ax.set_xticks(x, [label for label, _ in metric_defs])
    ax.set_ylabel("Rate")
    percent_axis(ax)
    clean_axis(ax)
    panel(ax, "d", "Selective-decision outcomes")
    legend_above(ax, ncol=3, fontsize=7.2)

    alt = (
        "Four-panel comparison of the primary natural cohort, difficulty track and "
        "controlled challenge. Panels show reference-label shares, FERA-KG versus the "
        "strongest registered reference within each track, class-specific F1 scores, "
        "and abstention, coverage and admission-error rates."
    )
    return fig, rows, alt


def figure4(payload: dict[str, Any]) -> tuple[plt.Figure, list[dict[str, Any]], str]:
    fig, axes = new_canvas()
    rows: list[dict[str, Any]] = []
    blocks = get_track_blocks(payload)
    primary = blocks[0][1]

    ax = axes[0, 0]
    components = sorted(
        {name for _, block in blocks for name in block["metrics"] if name.startswith("component::")}
    )
    component_priority = [
        "component::fera_ecrp",
        "component::fera_ecrp_sourcebal",
        "component::text_entity",
        "component::graph_anchor_wide",
        "component::same_backbone_no_ecrp",
    ]
    components = [name for name in component_priority if name in components] + [name for name in components if name not in component_priority]
    components = components[:5]
    x = np.arange(len(components))
    offsets = np.linspace(-0.18, 0.18, len(blocks))
    for offset, (track, block) in zip(offsets, blocks):
        values = [float(block["metrics"][name]["macro_f1"]) if name in block["metrics"] else np.nan for name in components]
        ax.plot(x + offset, values, color=TRACK_COLORS[track], marker="o", markersize=5.4, label=track)
        for name, value in zip(components, values):
            if np.isfinite(value):
                source_row(rows, 4, "a", track, name, "Macro-F1", value, n=block["items"])
    ax.set_xticks(x, [short_system(name).replace(" ", "\n", 1) for name in components])
    ax.set_ylabel("Four-class Macro-F1")
    ax.set_ylim(bottom=0)
    clean_axis(ax)
    panel(ax, "a", "Fixed component models")
    legend_above(ax, ncol=3, fontsize=7.2)

    ax = axes[0, 1]
    ablations = [name for name in primary["metrics"] if name.startswith("fusion_retrained::")]
    ablations.sort(key=lambda name: float(primary["metrics"][name]["macro_f1"]))
    if len(ablations) > 7:
        ablations = ablations[-7:]
    values = [float(primary["metrics"][name]["macro_f1"]) for name in ablations]
    y = np.arange(len(ablations))
    ax.hlines(y, 0, values, color=GRID, linewidth=2.2)
    colors = [BLUE if "full" in name.lower() else TEAL for name in ablations]
    ax.scatter(values, y, color=colors, s=52, edgecolor=WHITE, linewidth=0.7, zorder=3)
    ax.set_yticks(y, [short_system(name) for name in ablations])
    ax.set_xlabel("Macro-F1 on primary natural cohort")
    ax.set_xlim(left=0)
    clean_axis(ax, grid="x")
    for name, value in zip(ablations, values):
        source_row(rows, 4, "b", "Primary natural", name, "Macro-F1", value, n=primary["items"])
    panel(ax, "b", "Fusion-layer retraining ablations")

    interventions = payload["natural"].get("passage_interventions", {})
    ax = axes[1, 0]
    conditions = ("correct", "shuffled", "empty")
    x = np.arange(len(conditions))
    for track, key, _ in TRACKS[:2]:
        systems = interventions.get(key, {})
        if FINAL_SYSTEM not in systems:
            continue
        metrics = systems[FINAL_SYSTEM]["metrics"]
        values = [float(metrics[condition]["macro_f1"]) for condition in conditions]
        ax.plot(x, values, color=TRACK_COLORS[track], marker="o", markersize=6.0, label=track)
        for condition, value in zip(conditions, values):
            source_row(rows, 4, "c", track, FINAL_SYSTEM, "Macro-F1 against original labels", value, label=condition, n=metrics[condition]["items"])
    ax.set_xticks(x, ["Correct passage", "Shuffled passage", "Empty passage"])
    ax.set_ylabel("Macro-F1 against original labels")
    ax.set_ylim(bottom=0)
    clean_axis(ax)
    panel(ax, "c", "Passage information-dependence diagnostic")
    legend_above(ax, ncol=2, fontsize=7.5)

    ax = axes[1, 1]
    forest: list[tuple[str, float, list[float], str]] = []
    for track, key, _ in TRACKS[:2]:
        systems = interventions.get(key, {})
        if FINAL_SYSTEM not in systems:
            continue
        block = systems[FINAL_SYSTEM]
        for condition, comparison_key in (
            ("Shuffled", "correct_minus_shuffled"),
            ("Empty", "correct_minus_empty"),
        ):
            comparison = block[comparison_key]
            point = float(comparison["point_difference"])
            ci = [float(value) for value in comparison["ci95_percentile"]]
            forest.append((f"{track}\ncorrect minus {condition.lower()}", point, ci, track))
            source_row(rows, 4, "d", track, FINAL_SYSTEM, "paired Macro-F1 difference", point, label=f"correct minus {condition.lower()}", n=block["metrics"]["correct"]["items"], ci=ci)
    y = np.arange(len(forest))
    ax.axvline(0, color=INK, linewidth=1.0, linestyle=(0, (3, 3)))
    for index, (label, point, ci, track) in enumerate(forest):
        ax.errorbar(point, index, xerr=np.asarray([[point - ci[0]], [ci[1] - point]]), fmt="o", color=TRACK_COLORS[track], ecolor=TRACK_COLORS[track], elinewidth=1.8, capsize=4, markersize=6.5)
    ax.set_yticks(y, [item[0] for item in forest])
    ax.invert_yaxis()
    ax.set_xlabel("Paired difference in Macro-F1 (95% grouped-bootstrap CI)")
    clean_axis(ax, grid="x")
    panel(ax, "d", "Correct-passage paired differences")

    alt = (
        "Four-panel diagnostic figure showing component-model Macro-F1 across the three "
        "evaluation tracks, fusion-layer retraining ablations on the primary cohort, "
        "correct versus shuffled or empty passage scores, and grouped-bootstrap paired "
        "differences. Intervention scores use the original labels and are information-"
        "dependence diagnostics rather than counterfactual accuracy estimates."
    )
    return fig, rows, alt


def figure5(payload: dict[str, Any]) -> tuple[plt.Figure, list[dict[str, Any]], str]:
    fig, axes = new_canvas()
    rows: list[dict[str, Any]] = []
    blocks = get_track_blocks(payload)

    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle=(0, (3, 3)), linewidth=1.0, label="Ideal")
    for track, block in blocks:
        calibration = block.get("probability_metrics", {}).get(FINAL_SYSTEM)
        if not calibration:
            continue
        points = [point for point in calibration.get("calibration_bins", []) if point.get("count", 0) and point.get("accuracy") is not None]
        confidence = [float(point["mean_confidence"]) for point in points]
        accuracy = [float(point["accuracy"]) for point in points]
        ax.plot(confidence, accuracy, marker="o", markersize=4.7, color=TRACK_COLORS[track], label=track)
        for index, point in enumerate(points):
            source_row(rows, 5, "a", track, FINAL_SYSTEM, "calibration-bin accuracy", point["accuracy"], label=f"bin {index + 1}; mean confidence={point['mean_confidence']}", n=point["count"])
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    clean_axis(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=7.3)
    panel(ax, "a", "Top-label calibration")

    ax = axes[0, 1]
    for track, block in blocks:
        calibration = block.get("probability_metrics", {}).get(FINAL_SYSTEM)
        if not calibration:
            continue
        points = calibration.get("risk_coverage_grid", [])
        coverage = [float(point["coverage"]) for point in points]
        risk = [float(point["risk"]) for point in points]
        ax.plot(coverage, risk, marker="o", markersize=4.8, color=TRACK_COLORS[track], label=track)
        for point in points:
            source_row(rows, 5, "b", track, FINAL_SYSTEM, "selective risk", point["risk"], label=f"coverage={point['coverage']}", n=block["items"])
    ax.set_xlabel("Coverage retained by confidence")
    ax.set_ylabel("Error rate among retained items")
    ax.set_xlim(0.08, 1.02)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    clean_axis(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=7.3)
    panel(ax, "b", "Confidence-ranked risk and coverage")

    ax = axes[1, 0]
    markers = ("o", "s", "D")
    for marker, (track, block) in zip(markers, blocks):
        systems = [FINAL_SYSTEM]
        reference = strongest_registered_reference(block)
        if reference != FINAL_SYSTEM:
            systems.append(reference)
        for name in systems:
            abstention = block["metrics"][name]["abstention"]
            coverage = abstention.get("coverage")
            accuracy = abstention.get("covered_accuracy")
            if coverage is None or accuracy is None:
                continue
            color = TRACK_COLORS[track]
            filled = name == FINAL_SYSTEM
            ax.scatter(float(coverage), float(accuracy), marker=marker, s=75, facecolor=color if filled else WHITE, edgecolor=color, linewidth=1.5, zorder=3)
            ax.annotate("FERA-KG" if filled else short_system(name), (float(coverage), float(accuracy)), xytext=(5, 4 if filled else -11), textcoords="offset points", fontsize=6.7, color=color)
            source_row(rows, 5, "c", track, name, "covered accuracy", accuracy, label=f"coverage={coverage}", n=block["items"])
    ax.set_xlabel("Decision coverage")
    ax.set_ylabel("Accuracy on covered items")
    # Keep the zero-valued markers fully inside the plotting region.  The
    # explicit ticks preserve the intended 0--100% scale while the small
    # negative margin prevents the left half of a marker from being clipped.
    ax.set_xlim(-0.025, 1.03)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_ylim(0, 1.03)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    clean_axis(ax)
    panel(ax, "c", "Operational coverage and covered accuracy")
    legend_above(
        ax,
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=INK, markeredgecolor=INK, markersize=7, label="FERA-KG"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=WHITE, markeredgecolor=INK, markersize=7, label="Track reference"),
        ],
        ncol=2,
        fontsize=7.5,
    )

    ax = axes[1, 1]
    reason_labels = ("INSUFFICIENT_CONTEXT", "ENTITY_OR_TYPE_ERROR")
    x = np.arange(len(reason_labels))
    offsets = np.linspace(-0.18, 0.18, len(blocks))
    for offset, (track, block) in zip(offsets, blocks):
        by_reason = block["metrics"][FINAL_SYSTEM]["abstention"].get("by_reason", {})
        values = []
        for reason in reason_labels:
            info = by_reason.get(reason, {})
            value = info.get("correct_reason_recall")
            values.append(np.nan if value is None else float(value))
            source_row(rows, 5, "d", track, FINAL_SYSTEM, "reason-specific correct-reason recall", value, label=reason, n=info.get("support", 0))
        ax.plot(x + offset, values, marker="o", markersize=6.0, color=TRACK_COLORS[track], label=track)
        for xx, value, reason in zip(x + offset, values, reason_labels):
            if np.isfinite(value):
                support = by_reason.get(reason, {}).get("support", 0)
                ax.text(xx, min(1.0, value + 0.06), f"n={support}", ha="center", va="bottom", fontsize=6.4, color=TRACK_COLORS[track])
    ax.set_xticks(x, [LABEL_DISPLAY[label] for label in reason_labels])
    ax.set_ylabel("Correct-reason recall")
    percent_axis(ax)
    clean_axis(ax)
    panel(ax, "d", "Reason-specific abstention")
    legend_above(ax, ncol=3, fontsize=7.2)

    alt = (
        "Four-panel assessment of FERA-KG probability calibration and selective behavior "
        "across the primary natural, difficulty and controlled tracks. Panels show "
        "reliability curves, confidence-ranked risk-coverage curves, operational covered "
        "accuracy, and separate abstention recall for insufficient-context and entity-or-"
        "type-error cases."
    )
    return fig, rows, alt


def figure6(payload: dict[str, Any], labels: list[str]) -> tuple[plt.Figure, list[dict[str, Any]], str]:
    fig, axes = new_canvas()
    rows: list[dict[str, Any]] = []
    stress = payload["stress_tests"]["results"]
    stress_blocks = [(name, stress[key]) for name, key, _ in STRESS_TRACKS]
    stress_colors = {"Insufficient-context stress": ORANGE, "Tiangong Kaiwu zero-shot": PURPLE}

    metric_defs = (
        ("Macro-F1", "macro_f1"),
        ("Accuracy", "accuracy"),
        ("3-decision\nMacro-F1", "three_decision_macro_f1"),
    )
    for panel_index, (track, block) in enumerate(stress_blocks):
        ax = axes[0, panel_index]
        reference = strongest_registered_reference(block)
        x = np.arange(len(metric_defs))
        width = 0.34
        for offset, name, color in ((-width / 2, FINAL_SYSTEM, BLUE), (width / 2, reference, stress_colors[track])):
            values = [float(block["metrics"][name][key]) for _, key in metric_defs]
            ax.bar(x + offset, values, width=width, color=color, edgecolor=WHITE, linewidth=0.7, label=short_system(name))
            for (label, key), value in zip(metric_defs, values):
                source_row(rows, 6, "a" if panel_index == 0 else "b", track, name, key, value, label=label.replace("\n", " "), n=block["items"])
        ax.set_xticks(x, [label for label, _ in metric_defs])
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.02)
        clean_axis(ax)
        panel(ax, "a" if panel_index == 0 else "b", f"{track} (n={block['items']})")
        legend_above(ax, ncol=2, fontsize=7.5)

    ax = axes[1, 0]
    x = np.arange(len(labels))
    offsets = (-0.08, 0.08)
    for offset, (track, block) in zip(offsets, stress_blocks):
        metric = block["metrics"][FINAL_SYSTEM]
        values = [float(metric["per_class"][label]["f1"]) for label in labels]
        ax.plot(x + offset, values, marker="o", markersize=5.8, color=stress_colors[track], label=track)
        for label, value in zip(labels, values):
            source_row(rows, 6, "c", track, FINAL_SYSTEM, "class F1", value, label=label, n=metric["per_class"][label]["support"])
    ax.set_xticks(x, [LABEL_DISPLAY[label] for label in labels])
    ax.set_ylabel("FERA-KG class F1")
    ax.set_ylim(0, 1.02)
    clean_axis(ax)
    panel(ax, "c", "Class-specific stress-test performance")
    legend_above(ax, ncol=2, fontsize=7.3)

    ax = axes[1, 1]
    diagnostic_rows: list[tuple[str, str, str, float | None]] = []
    for track, block in stress_blocks:
        reference = strongest_registered_reference(block)
        for name in (FINAL_SYSTEM, reference):
            abstention = block["metrics"][name]["abstention"]
            if track == "Insufficient-context stress":
                by_reason = abstention.get("by_reason", {}).get("INSUFFICIENT_CONTEXT", {})
                diagnostic_rows.extend(
                    [
                        (track, name, "IC correct-reason recall", by_reason.get("correct_reason_recall")),
                        (track, name, "IC false-present rate", by_reason.get("false_present_rate")),
                    ]
                )
            else:
                diagnostic_rows.extend(
                    [
                        (track, name, "Coverage", abstention.get("coverage")),
                        (track, name, "Admission error", abstention.get("knowledge_admission_error")),
                    ]
                )
    labels_seen: list[str] = []
    for _, _, metric, _ in diagnostic_rows:
        if metric not in labels_seen:
            labels_seen.append(metric)
    y = np.arange(len(labels_seen))
    for idx, metric_name in enumerate(labels_seen):
        items = [(track, name, value) for track, name, metric, value in diagnostic_rows if metric == metric_name]
        for offset_index, (track, name, value) in enumerate(items):
            if value is None:
                continue
            is_full = name == FINAL_SYSTEM
            color = BLUE if is_full else MUTED
            offset = -0.09 if is_full else 0.09
            ax.scatter(float(value), idx + offset, s=62, marker="o" if is_full else "s", color=color, edgecolor=WHITE, linewidth=0.7, zorder=3)
            source_row(rows, 6, "d", track, name, metric_name, value, n=next(block["items"] for block_track, block in stress_blocks if block_track == track))
    ax.set_yticks(y, labels_seen)
    ax.invert_yaxis()
    # Several stress-test rates are exactly zero.  Leave a narrow plotting
    # margin so their markers remain visible rather than being bisected by
    # the y axis, while retaining percentage ticks from 0 to 100%.
    ax.set_xlim(-0.025, 1.03)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Rate")
    clean_axis(ax, grid="x")
    panel(ax, "d", "Observed failure-boundary diagnostics")
    legend_above(
        ax,
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=WHITE, markersize=8, label="FERA-KG"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=MUTED, markeredgecolor=WHITE, markersize=8, label="Fixed comparator"),
        ],
        ncol=2,
        fontsize=7.5,
    )

    alt = (
        "Four-panel stress-test summary. FERA-KG is compared with the strongest registered "
        "reference on the 150-item insufficient-context resource and the 356-item "
        "Tiangong Kaiwu zero-shot domain-shift resource. Additional panels expose class-"
        "specific F1 and abstention, false-admission and coverage boundaries."
    )
    return fig, rows, alt


def write_source_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "figure",
        "panel",
        "evaluation",
        "system",
        "metric",
        "label_or_condition",
        "value",
        "n",
        "ci95_lower",
        "ci95_upper",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(
    fig: plt.Figure,
    number: int,
    slug: str,
    rows: list[dict[str, Any]],
    alt_text: str,
    input_path: Path,
    output_dir: Path,
    manuscript_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    stem = f"Fig{number}_{slug}"
    png = output_dir / f"{stem}_600dpi.png"
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    source = output_dir / f"{stem}_source_data.csv"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    copied = manuscript_dir / png.name
    shutil.copy2(png, copied)
    write_source_csv(source, rows)
    return {
        "figure": f"Fig. {number}",
        "title": slug.replace("_", " "),
        "alt_text": alt_text,
        "data_source": str(input_path.resolve()),
        "data_source_sha256": sha256(input_path),
        "source_data_csv": str(source.resolve()),
        "png_600dpi": str(png.resolve()),
        "pdf": str(pdf.resolve()),
        "svg": str(svg.resolve()),
        "manuscript_png_copy": str(copied.resolve()),
    }


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "figure",
        "title",
        "alt_text",
        "data_source",
        "data_source_sha256",
        "source_data_csv",
        "png_600dpi",
        "pdf",
        "svg",
        "manuscript_png_copy",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manuscript-figures", type=Path, default=MANUSCRIPT_FIGURES)
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Unified evaluation is not available yet: {args.input}. "
            "Run scripts/evaluate_final.py after unified inference completes."
        )
    payload = read_json(args.input)
    labels = validate_payload(payload)
    generated = []
    specifications = (
        (3, "Unified_evaluation_tracks_KBS", figure3(payload, labels)),
        (4, "Unified_model_diagnostics_KBS", figure4(payload)),
        (5, "Unified_selective_calibration_KBS", figure5(payload)),
        (6, "Unified_stress_boundaries_KBS", figure6(payload, labels)),
    )
    for number, slug, (fig, rows, alt_text) in specifications:
        generated.append(
            save_figure(
                fig,
                number,
                slug,
                rows,
                alt_text,
                args.input,
                args.output_dir,
                args.manuscript_figures,
            )
        )
    metadata = args.output_dir / "unified_kbs_figure_metadata.csv"
    write_metadata(metadata, generated)
    print(
        json.dumps(
            {
                "status": "UNIFIED_KBS_FIGURES_COMPLETE",
                "figures": generated,
                "metadata": str(metadata.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
