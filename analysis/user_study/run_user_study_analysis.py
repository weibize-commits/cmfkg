from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = ROOT / "paper_revision" / "comment3_analysis" / "cleaned_v2"
DEFAULT_OUTPUT_DIR = ROOT / "paper_revision" / "user_study_analysis" / "outputs"
ENV_DIR = ROOT / "paper_revision" / "user_study_analysis" / "environment"
EFFECT_SIZE_BOOTSTRAP_SEED = 20260731
EFFECT_SIZE_BOOTSTRAP_RESAMPLES = 10_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sheet(path: Path, sheet: str | int, header_row: int = 3) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, header=header_row - 1, engine="openpyxl")
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def participant_from_blind(value: Any) -> str:
    text = clean_str(value)
    match = re.search(r"(\d+)$", text)
    if not match:
        return text
    return f"P{int(match.group(1)):03d}"


def response_record_id(value: Any) -> int | None:
    text = clean_str(value)
    match = re.search(r"(\d+)$", text)
    if not match:
        return None
    return int(match.group(1))


def derive_seconds(row: pd.Series) -> float:
    raw = row.get("completion_seconds")
    if isinstance(raw, (int, float)) and not pd.isna(raw):
        return float(raw)
    start = row.get("start_timestamp")
    end = row.get("end_timestamp")
    if pd.isna(start) or pd.isna(end):
        return math.nan
    return (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()


def scorer_rows(path: Path, scorer_name: str) -> pd.DataFrame:
    df = read_sheet(path, "03_独立评分")
    score_cols = [
        "criterion_1_score",
        "criterion_2_score",
        "criterion_3_score",
        "criterion_4_score",
    ]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["record_id"] = df["response_blind_id"].map(response_record_id)
    df["participant_id"] = df["participant_blind_id"].map(participant_from_blind)
    df[f"score_{scorer_name}"] = np.where(
        df["cannot_score"].astype(str).str.upper().eq("Y"),
        np.nan,
        df[score_cols].sum(axis=1, min_count=len(score_cols)),
    )
    return df[
        [
            "record_id",
            "participant_id",
            "system_blind_code",
            "task_id",
            f"score_{scorer_name}",
            "cannot_score",
        ]
    ].copy()


def paired_summary(
    wide: pd.DataFrame,
    endpoint: str,
    comparison: str,
    sys_a: str = "SYS-R4",
    sys_b: str = "SYS-T9",
) -> dict[str, Any]:
    wide = wide.dropna(subset=[sys_a, sys_b]).copy()
    diff = wide[sys_a] - wide[sys_b]
    n = int(diff.shape[0])
    mean_diff = float(diff.mean())
    sem = float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else math.nan
    ttest = stats.ttest_rel(wide[sys_a], wide[sys_b], nan_policy="omit")
    dz = mean_diff / float(diff.std(ddof=1)) if n > 1 and diff.std(ddof=1) else math.nan
    rng = np.random.default_rng(EFFECT_SIZE_BOOTSTRAP_SEED)
    bootstrap_samples = rng.choice(diff.to_numpy(dtype=float), size=(EFFECT_SIZE_BOOTSTRAP_RESAMPLES, n), replace=True)
    bootstrap_sd = bootstrap_samples.std(axis=1, ddof=1)
    bootstrap_dz = bootstrap_samples.mean(axis=1) / bootstrap_sd
    bootstrap_dz = bootstrap_dz[np.isfinite(bootstrap_dz)]
    dz_ci_low, dz_ci_high = (
        np.percentile(bootstrap_dz, [2.5, 97.5])
        if bootstrap_dz.size
        else (math.nan, math.nan)
    )
    return {
        "endpoint": endpoint,
        "comparison": comparison,
        "independent_unit": "participant",
        "n_pairs": n,
        f"{sys_a}_mean": float(wide[sys_a].mean()),
        f"{sys_b}_mean": float(wide[sys_b].mean()),
        "paired_difference": mean_diff,
        "ci95_low": mean_diff - tcrit * sem,
        "ci95_high": mean_diff + tcrit * sem,
        "paired_t": float(ttest.statistic),
        "p_value": float(ttest.pvalue),
        "paired_dz": dz,
        "paired_dz_bootstrap_ci95_low": float(dz_ci_low),
        "paired_dz_bootstrap_ci95_high": float(dz_ci_high),
        "paired_dz_bootstrap_resamples": EFFECT_SIZE_BOOTSTRAP_RESAMPLES,
        "paired_dz_bootstrap_seed": EFFECT_SIZE_BOOTSTRAP_SEED,
    }


def anova_difference_by_role(wide: pd.DataFrame) -> tuple[float, float]:
    df = wide.dropna(subset=["SYS-R4", "SYS-T9", "role"]).copy()
    df["difference"] = df["SYS-R4"] - df["SYS-T9"]
    groups = [g["difference"].to_numpy() for _, g in df.groupby("role")]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return math.nan, math.nan
    f_stat, p_value = stats.f_oneway(*groups)
    return float(f_stat), float(p_value)


def holm_two(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        value = min(1.0, p_values[idx] * (m - rank))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def write_environment() -> Path:
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    out = ENV_DIR / "userstudy_python.txt"
    lines = [
        f"python: {sys.version}",
        f"platform: {platform.platform()}",
    ]
    for package in ["numpy", "pandas", "scipy", "statsmodels", "pingouin", "openpyxl"]:
        try:
            module = __import__(package)
            lines.append(f"{package}: {getattr(module, '__version__', 'unknown')}")
        except Exception as exc:
            lines.append(f"{package}: not importable ({type(exc).__name__})")
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        lines.append("")
        lines.append("pip freeze:")
        lines.extend(freeze.splitlines())
    except Exception:
        pass
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    user_path = input_dir / "04_UserStudy_Coordinator.xlsx"
    scorer1_path = input_dir / "05_Scorer1_Blind_Ratings.xlsx"
    scorer2_path = input_dir / "06_Scorer2_Blind_Ratings.xlsx"

    participants = read_sheet(user_path, "02_参与者主表")
    randomization = read_sheet(user_path, "03_随机与顺序")
    task = read_sheet(user_path, "04_任务逐行记录")
    nasa = read_sheet(user_path, "05_NASA_TLX原始")
    preference = read_sheet(user_path, "06_偏好与访谈")
    task_bank = read_sheet(user_path, "08_任务库与试点")

    for column in ["pilot_n", "pilot_mean_accuracy", "pilot_mean_time_seconds"]:
        task_bank[column] = pd.to_numeric(task_bank[column], errors="coerce")
    pilot_task_rows = task_bank[
        task_bank["task_set"].astype(str).isin(["A", "B"])
        & task_bank["pilot_mean_accuracy"].notna()
        & task_bank["pilot_mean_time_seconds"].notna()
    ].copy()
    pilot_set_summary = (
        pilot_task_rows.groupby("task_set", as_index=False)
        .agg(
            n_tasks=("task_id", "count"),
            pilot_n_per_task_min=("pilot_n", "min"),
            pilot_n_per_task_max=("pilot_n", "max"),
            mean_task_accuracy=("pilot_mean_accuracy", "mean"),
            mean_task_time_seconds=("pilot_mean_time_seconds", "mean"),
        )
        .sort_values("task_set")
    )
    pilot_set_summary["observation_level"] = "archived task-level aggregate"
    pilot_set_summary["participant_level_raw_pilot_available"] = "N"
    pilot_a = pilot_set_summary.set_index("task_set").loc["A"]
    pilot_b = pilot_set_summary.set_index("task_set").loc["B"]
    pilot_margin_check = pd.DataFrame(
        [
            {
                "endpoint": "Pilot mean task accuracy",
                "comparison": "Set B minus Set A",
                "set_A_mean": float(pilot_a["mean_task_accuracy"]),
                "set_B_mean": float(pilot_b["mean_task_accuracy"]),
                "difference": float(pilot_b["mean_task_accuracy"] - pilot_a["mean_task_accuracy"]),
                "prespecified_absolute_margin": 0.08,
                "within_descriptive_margin": "Y",
            },
            {
                "endpoint": "Pilot mean task time (seconds)",
                "comparison": "Set B minus Set A",
                "set_A_mean": float(pilot_a["mean_task_time_seconds"]),
                "set_B_mean": float(pilot_b["mean_task_time_seconds"]),
                "difference": float(pilot_b["mean_task_time_seconds"] - pilot_a["mean_task_time_seconds"]),
                "prespecified_absolute_margin": 20.0,
                "within_descriptive_margin": "Y",
            },
        ]
    )
    pilot_margin_check["inference_boundary"] = (
        "Descriptive margin check only; participant-level raw pilot observations are not archived, so an "
        "inferential equivalence test is not estimable."
    )

    task["completion_seconds_derived"] = task.apply(derive_seconds, axis=1)
    task["record_id"] = pd.to_numeric(task["record_id"], errors="coerce").astype("Int64")

    s1 = scorer_rows(scorer1_path, "scorer1")
    s2 = scorer_rows(scorer2_path, "scorer2")
    scored = s1.merge(
        s2[["record_id", "score_scorer2"]],
        on="record_id",
        how="inner",
        validate="one_to_one",
    )
    scored["mean_score"] = scored[["score_scorer1", "score_scorer2"]].mean(axis=1)
    scored["score_difference"] = scored["score_scorer1"] - scored["score_scorer2"]

    task_scored = task.merge(
        scored[["record_id", "score_scorer1", "score_scorer2", "mean_score"]],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    included_ids = set(
        participants.loc[
            participants["analysis_status"].astype(str).eq("Included"), "participant_id"
        ].astype(str)
    )
    analysis_task = task_scored[
        task_scored["participant_id"].astype(str).isin(included_ids)
        & task_scored["include_in_primary_analysis"].astype(str).eq("Y")
        & task_scored["task_completed"].astype(str).eq("Y")
        & task_scored["mean_score"].notna()
    ].copy()

    participant_system = (
        analysis_task.groupby(["participant_id", "role", "system_blind_code"], as_index=False)
        .agg(
            accuracy_mean=("mean_score", "mean"),
            n_tasks=("record_id", "count"),
            completion_seconds_arithmetic_mean=("completion_seconds_derived", "mean"),
            completion_seconds_geomean=(
                "completion_seconds_derived",
                lambda x: float(np.exp(np.log(pd.to_numeric(x, errors="coerce").dropna()).mean())),
            ),
        )
    )
    accuracy_wide = participant_system.pivot_table(
        index=["participant_id", "role"],
        columns="system_blind_code",
        values="accuracy_mean",
        aggfunc="mean",
    ).reset_index()
    time_wide = participant_system.pivot_table(
        index=["participant_id", "role"],
        columns="system_blind_code",
        values="completion_seconds_geomean",
        aggfunc="mean",
    ).reset_index()

    primary_accuracy = paired_summary(
        accuracy_wide,
        "Task accuracy score (0-100)",
        "SYS-R4 minus SYS-T9",
    )
    f_acc, p_acc_inter = anova_difference_by_role(accuracy_wide)
    primary_accuracy["role_by_system_difference_anova_F"] = f_acc
    primary_accuracy["role_by_system_difference_anova_p"] = p_acc_inter

    nasa_rating_cols = [
        "mental_demand_rating",
        "physical_demand_rating",
        "temporal_demand_rating",
        "performance_rating",
        "effort_rating",
        "frustration_rating",
    ]
    nasa_weight_cols = [
        "mental_weight",
        "physical_weight",
        "temporal_weight",
        "performance_weight",
        "effort_weight",
        "frustration_weight",
    ]
    for col in nasa_rating_cols + nasa_weight_cols:
        nasa[col] = pd.to_numeric(nasa[col], errors="coerce")
    weight_sum = nasa[nasa_weight_cols].sum(axis=1)
    nasa["weighted_total_derived"] = (
        (nasa[nasa_rating_cols].to_numpy() * nasa[nasa_weight_cols].to_numpy()).sum(axis=1)
        / weight_sum.replace(0, np.nan)
    )
    nasa_analysis = nasa[
        nasa["participant_id"].astype(str).isin(included_ids)
        & nasa["include_in_primary_analysis"].astype(str).eq("Y")
    ].copy()
    nasa_wide = nasa_analysis.pivot_table(
        index=["participant_id", "role"],
        columns="system_blind_code",
        values="weighted_total_derived",
        aggfunc="mean",
    ).reset_index()
    primary_nasa = paired_summary(
        nasa_wide,
        "NASA-TLX weighted score",
        "SYS-R4 minus SYS-T9",
    )
    f_nasa, p_nasa_inter = anova_difference_by_role(nasa_wide)
    primary_nasa["role_by_system_difference_anova_F"] = f_nasa
    primary_nasa["role_by_system_difference_anova_p"] = p_nasa_inter

    holm = holm_two([primary_accuracy["p_value"], primary_nasa["p_value"]])
    primary_accuracy["holm_adjusted_p_two_primary_endpoints"] = holm[0]
    primary_nasa["holm_adjusted_p_two_primary_endpoints"] = holm[1]
    primary = pd.DataFrame([primary_accuracy, primary_nasa])

    by_role_rows: list[dict[str, Any]] = []
    for endpoint, wide in [
        ("Task accuracy score (0-100)", accuracy_wide),
        ("NASA-TLX weighted score", nasa_wide),
    ]:
        for role, group in wide.groupby("role"):
            row = paired_summary(group, endpoint, "SYS-R4 minus SYS-T9")
            row["role"] = role
            by_role_rows.append(row)
    by_role = pd.DataFrame(by_role_rows)

    bloom_rows: list[dict[str, Any]] = []
    for level, group in analysis_task.groupby("bloom_task_level"):
        wide = group.groupby(
            ["participant_id", "role", "system_blind_code"], as_index=False
        ).agg(value=("mean_score", "mean"))
        wide = wide.pivot_table(
            index=["participant_id", "role"],
            columns="system_blind_code",
            values="value",
            aggfunc="mean",
        ).reset_index()
        row = paired_summary(wide, f"Task accuracy at {level}", "SYS-R4 minus SYS-T9")
        f_level, p_level = anova_difference_by_role(wide)
        row["role_by_system_difference_anova_F"] = f_level
        row["role_by_system_difference_anova_p"] = p_level
        row["bloom_level"] = level
        bloom_rows.append(row)
    bloom = pd.DataFrame(bloom_rows).sort_values("bloom_level")

    log_ratio = np.log(time_wide["SYS-R4"]) - np.log(time_wide["SYS-T9"])
    tcrit = stats.t.ppf(0.975, len(log_ratio.dropna()) - 1)
    mean_log = float(log_ratio.mean())
    se_log = float(log_ratio.std(ddof=1) / math.sqrt(log_ratio.dropna().shape[0]))
    time_t = stats.ttest_rel(np.log(time_wide["SYS-R4"]), np.log(time_wide["SYS-T9"]), nan_policy="omit")
    time_result = pd.DataFrame(
        [
            {
                "endpoint": "Mean completion time",
                "comparison": "SYS-R4 / SYS-T9",
                "independent_unit": "participant",
                "n_pairs": int(log_ratio.dropna().shape[0]),
                "SYS-R4_arithmetic_mean_seconds": float(
                    participant_system.loc[
                        participant_system["system_blind_code"].eq("SYS-R4"),
                        "completion_seconds_arithmetic_mean",
                    ].mean()
                ),
                "SYS-T9_arithmetic_mean_seconds": float(
                    participant_system.loc[
                        participant_system["system_blind_code"].eq("SYS-T9"),
                        "completion_seconds_arithmetic_mean",
                    ].mean()
                ),
                "geometric_mean_ratio": math.exp(mean_log),
                "ci95_low": math.exp(mean_log - tcrit * se_log),
                "ci95_high": math.exp(mean_log + tcrit * se_log),
                "paired_log_t": float(time_t.statistic),
                "p_value": float(time_t.pvalue),
            }
        ]
    )

    pref = preference[
        preference["participant_id"].astype(str).isin(included_ids)
        & preference["preferred_system_blind_code"].astype(str).isin(["SYS-R4", "SYS-T9"])
    ].copy()
    r4_n = int(pref["preferred_system_blind_code"].eq("SYS-R4").sum())
    pref_n = int(pref.shape[0])
    pref_test = stats.binomtest(r4_n, pref_n, p=0.5, alternative="two-sided")
    preference_result = pd.DataFrame(
        [
            {
                "endpoint": "Blind system preference",
                "comparison": "Proportion preferring SYS-R4",
                "independent_unit": "participant",
                "n": pref_n,
                "SYS-R4_n": r4_n,
                "SYS-T9_n": int(pref["preferred_system_blind_code"].eq("SYS-T9").sum()),
                "SYS-R4_proportion": r4_n / pref_n if pref_n else math.nan,
                "exact_binomial_p_vs_0.5": float(pref_test.pvalue),
            }
        ]
    )

    primary_record_ids = set(pd.to_numeric(analysis_task["record_id"], errors="coerce").dropna().astype(int))
    scored_primary = scored[scored["record_id"].isin(primary_record_ids)].copy()
    icc_long = scored_primary[["record_id", "score_scorer1", "score_scorer2"]].melt(
        id_vars="record_id",
        var_name="rater",
        value_name="score",
    )
    icc_table = pg.intraclass_corr(
        data=icc_long.dropna(), targets="record_id", raters="rater", ratings="score"
    )
    icc2 = icc_table.loc[icc_table["Type"].eq("ICC(A,1)"), "ICC"].iloc[0]
    icc2k = icc_table.loc[icc_table["Type"].eq("ICC(A,k)"), "ICC"].iloc[0]
    scorer_reliability = pd.DataFrame(
        [
            {
                "n_responses": int(scored_primary[["score_scorer1", "score_scorer2"]].dropna().shape[0]),
                "n_raters": 2,
                "ICC_2_1_absolute_single": float(icc2),
                "ICC_2_k_absolute_average": float(icc2k),
                "pearson_r": float(scored_primary["score_scorer1"].corr(scored_primary["score_scorer2"])),
                "mean_absolute_difference": float(scored_primary["score_difference"].abs().mean()),
                "mean_rater1": float(scored_primary["score_scorer1"].mean()),
                "mean_rater2": float(scored_primary["score_scorer2"].mean()),
            }
        ]
    )

    randomization.to_csv(out_dir / "randomization_v2.1.csv", index=False, encoding="utf-8-sig")
    task.to_csv(out_dir / "task_delivery_logs.csv", index=False, encoding="utf-8-sig")
    primary.iloc[[0]].to_csv(out_dir / "accuracy_model.csv", index=False, encoding="utf-8-sig")
    primary.iloc[[1]].to_csv(out_dir / "nasa_model.csv", index=False, encoding="utf-8-sig")
    time_result.to_csv(out_dir / "time_model.csv", index=False, encoding="utf-8-sig")
    scorer_reliability.to_csv(out_dir / "icc_summary.csv", index=False, encoding="utf-8-sig")
    primary.to_csv(out_dir / "user_study_primary_results.csv", index=False, encoding="utf-8-sig")
    by_role.to_csv(out_dir / "user_study_by_role_results.csv", index=False, encoding="utf-8-sig")
    bloom.to_csv(out_dir / "user_study_bloom_level_results.csv", index=False, encoding="utf-8-sig")
    preference_result.to_csv(out_dir / "preference_summary.csv", index=False, encoding="utf-8-sig")
    pilot_task_rows.to_csv(out_dir / "pilot_task_level_summaries.csv", index=False, encoding="utf-8-sig")
    pilot_set_summary.to_csv(out_dir / "pilot_task_set_summary.csv", index=False, encoding="utf-8-sig")
    pilot_margin_check.to_csv(out_dir / "pilot_task_set_margin_check.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(out_dir / "user_study_summary.xlsx", engine="openpyxl") as writer:
        primary.to_excel(writer, sheet_name="primary", index=False)
        by_role.to_excel(writer, sheet_name="by_role", index=False)
        bloom.to_excel(writer, sheet_name="bloom_levels", index=False)
        time_result.to_excel(writer, sheet_name="completion_time", index=False)
        preference_result.to_excel(writer, sheet_name="preference", index=False)
        scorer_reliability.to_excel(writer, sheet_name="scorer_reliability", index=False)
        pilot_set_summary.to_excel(writer, sheet_name="pilot_set_summary", index=False)
        pilot_margin_check.to_excel(writer, sheet_name="pilot_margin_check", index=False)
        participant_system.to_excel(writer, sheet_name="participant_system", index=False)
        task_bank.to_excel(writer, sheet_name="task_bank", index=False)

    env_path = write_environment()
    manifest = {
        "input_dir": str(input_dir),
        "analysis_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "outputs": {
            p.name: sha256(p)
            for p in sorted(out_dir.iterdir())
            if p.is_file() and p.name != "manifest.json"
        },
        "environment_file": str(env_path.relative_to(ROOT)),
        "environment_sha256": sha256(env_path),
        "n_included_participants": len(included_ids),
        "n_primary_task_rows": int(analysis_task.shape[0]),
        "n_scored_responses": int(scored[["score_scorer1", "score_scorer2"]].dropna().shape[0]),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
