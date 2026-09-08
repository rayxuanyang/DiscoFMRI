#!/usr/bin/env python3
"""Compare HCPex parcel-wise LMEr results from lemma-mean and lemma-sum EFA.

The exported tables are matched one-to-one on factor number and HCPex ROI.
The script does not recompute FDR; it compares the ``significant_FDR`` decisions
and adjusted p-values already exported by the two interpretation pipelines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


DEFAULT_MEAN = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/FactorAnalysis/github/"
    "FactorAnalysis_fMRI/data/HCPex_p005FDR/"
    "df_ROIvalue_HCPex_p005FDR_unsmoothed_LMEr_r1.4_lemmaMean.csv"
)
DEFAULT_SUM = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/FactorAnalysis/github/"
    "FactorAnalysis_fMRI/data/HCPex_p005FDR/"
    "df_ROIvalue_HCPex_p005FDR_unsmoothed_LMEr_r1.4.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "lemma_mean_vs_lemma_sum_comparison"
KEY = ["iFA", "ROI_idx"]
META = ["#No.", "Region", "RegionLongName", "Cortical Division", "LR", "idx"]
MEASURES = [
    "ROI_value",
    "p_value",
    "p_value_FDR",
    "significant_FDR",
    "label_FDR",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lemma-mean", type=Path, default=DEFAULT_MEAN)
    parser.add_argument("--lemma-sum", type=Path, default=DEFAULT_SUM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_and_validate(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(KEY + META + MEASURES)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    if frame.duplicated(KEY).any():
        duplicates = frame.loc[frame.duplicated(KEY, keep=False), KEY].head()
        raise ValueError(f"{name} has duplicate factor/ROI keys:\n{duplicates}")
    if set(frame["iFA"]) != set(range(1, 9)):
        raise ValueError(f"{name} does not contain factors 1 through 8")
    if frame["ROI_idx"].nunique() != 360:
        raise ValueError(f"{name} does not contain 360 HCPex ROIs")
    return frame


def correlation(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return np.nan, np.nan
    return pearsonr(x[finite], y[finite]).statistic, spearmanr(x[finite], y[finite]).statistic


def add_comparison_columns(matched: pd.DataFrame) -> pd.DataFrame:
    matched["delta_mean_minus_sum"] = matched["ROI_value_mean"] - matched["ROI_value_sum"]
    matched["abs_delta"] = matched["delta_mean_minus_sum"].abs()
    matched["effect_sign_mean"] = np.sign(matched["ROI_value_mean"]).astype(int)
    matched["effect_sign_sum"] = np.sign(matched["ROI_value_sum"]).astype(int)
    matched["effect_sign_reversal"] = matched["effect_sign_mean"] != matched["effect_sign_sum"]

    mean_sig = matched["significant_FDR_mean"].astype(bool)
    sum_sig = matched["significant_FDR_sum"].astype(bool)
    shared = mean_sig & sum_sig
    shared_opposite = shared & matched["effect_sign_reversal"]
    conditions = [
        shared_opposite,
        shared & ~shared_opposite,
        mean_sig & ~sum_sig,
        sum_sig & ~mean_sig,
    ]
    labels = ["shared_opposite_direction", "shared_same_direction", "lemma_mean_only", "lemma_sum_only"]
    matched["FDR_comparison"] = np.select(conditions, labels, default="neither")
    return matched


def summarize(group: pd.DataFrame, factor: str | int) -> dict[str, float | int | str]:
    mean_sig = group["significant_FDR_mean"].astype(bool)
    sum_sig = group["significant_FDR_sum"].astype(bool)
    shared = mean_sig & sum_sig
    union = mean_sig | sum_sig
    pearson, spearman = correlation(group["ROI_value_sum"], group["ROI_value_mean"])
    shared_opposite = shared & group["effect_sign_reversal"]
    return {
        "iFA": factor,
        "n_ROI": len(group),
        "lemma_sum_significant": int(sum_sig.sum()),
        "lemma_mean_significant": int(mean_sig.sum()),
        "shared_significant": int(shared.sum()),
        "shared_same_direction": int((shared & ~group["effect_sign_reversal"]).sum()),
        "shared_opposite_direction": int(shared_opposite.sum()),
        "lemma_mean_only": int((mean_sig & ~sum_sig).sum()),
        "lemma_sum_only": int((sum_sig & ~mean_sig).sum()),
        "FDR_jaccard": float(shared.sum() / union.sum()) if union.any() else np.nan,
        "sum_result_retained": float(shared.sum() / sum_sig.sum()) if sum_sig.any() else np.nan,
        "mean_result_shared": float(shared.sum() / mean_sig.sum()) if mean_sig.any() else np.nan,
        "pearson_r_effect": pearson,
        "spearman_rho_effect": spearman,
        "mean_delta": float(group["delta_mean_minus_sum"].mean()),
        "mean_absolute_delta": float(group["abs_delta"].mean()),
        "rmse": float(np.sqrt(np.mean(group["delta_mean_minus_sum"] ** 2))),
        "all_ROI_sign_reversals": int(group["effect_sign_reversal"].sum()),
    }


def scatter_figure(matched: pd.DataFrame, output: Path) -> None:
    colors = {
        "neither": "#B8B8B8",
        "shared_same_direction": "#6A3D9A",
        "shared_opposite_direction": "#E31A1C",
        "lemma_mean_only": "#FF7F00",
        "lemma_sum_only": "#1F78B4",
    }
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), constrained_layout=True)
    for factor, ax in enumerate(axes.flat, start=1):
        frame = matched.loc[matched["iFA"] == factor]
        low = min(frame["ROI_value_sum"].min(), frame["ROI_value_mean"].min())
        high = max(frame["ROI_value_sum"].max(), frame["ROI_value_mean"].max())
        margin = max((high - low) * 0.05, 0.05)
        limits = (low - margin, high + margin)
        ax.plot(limits, limits, color="black", linewidth=0.8, linestyle="--", zorder=1)
        for status in colors:
            subset = frame.loc[frame["FDR_comparison"] == status]
            ax.scatter(
                subset["ROI_value_sum"], subset["ROI_value_mean"],
                s=13, alpha=0.75, color=colors[status], edgecolors="none", label=status,
            )
        r, rho = correlation(frame["ROI_value_sum"], frame["ROI_value_mean"])
        ax.set(xlim=limits, ylim=limits, title=f"FA{factor}  r={r:.3f}, ρ={rho:.3f}")
        ax.grid(alpha=0.15, linewidth=0.5)
        if factor > 4:
            ax.set_xlabel("LemmaSum ROI effect")
        if factor in (1, 5):
            ax.set_ylabel("LemmaMean ROI effect")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    clean_labels = [label.replace("_", " ") for label in labels]
    fig.legend(handles, clean_labels, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("HCPex parcel-wise LMEr effect agreement", fontsize=15)
    fig.savefig(output / "effect_scatter_by_factor.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "effect_scatter_by_factor.pdf", bbox_inches="tight")
    plt.close(fig)


def overlap_figure(summary: pd.DataFrame, output: Path) -> None:
    per_factor = summary.loc[summary["iFA"] != "All"].copy()
    x = np.arange(len(per_factor))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)

    bottom = np.zeros(len(per_factor))
    for column, label, color in [
        ("shared_significant", "Shared", "#6A3D9A"),
        ("lemma_mean_only", "LemmaMean only", "#FF7F00"),
        ("lemma_sum_only", "LemmaSum only", "#1F78B4"),
    ]:
        values = per_factor[column].to_numpy(dtype=float)
        axes[0].bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    axes[0].set(xticks=x, xticklabels=[f"FA{i}" for i in range(1, 9)], ylabel="Number of FDR-significant ROIs")
    axes[0].set_title("FDR-significant set overlap")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].plot(x, per_factor["pearson_r_effect"], marker="o", label="Effect Pearson r", color="#33A02C")
    axes[1].plot(x, per_factor["FDR_jaccard"], marker="s", label="FDR Jaccard", color="#6A3D9A")
    axes[1].plot(x, per_factor["sum_result_retained"], marker="^", label="LemmaSum retained", color="#1F78B4")
    axes[1].set(xticks=x, xticklabels=[f"FA{i}" for i in range(1, 9)], ylim=(0, 1.03), ylabel="Agreement")
    axes[1].set_title("Continuous and thresholded robustness")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    fig.savefig(output / "FDR_overlap_and_agreement.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "FDR_overlap_and_agreement.pdf", bbox_inches="tight")
    plt.close(fig)


def report(summary: pd.DataFrame, matched: pd.DataFrame, mean_path: Path, sum_path: Path) -> str:
    overall = summary.loc[summary["iFA"] == "All"].iloc[0]
    report_columns = [
        "iFA", "lemma_sum_significant", "lemma_mean_significant",
        "shared_significant", "lemma_mean_only", "lemma_sum_only",
        "FDR_jaccard", "sum_result_retained", "pearson_r_effect",
        "spearman_rho_effect", "mean_absolute_delta", "rmse",
    ]
    table = summary[report_columns]
    header = "| " + " | ".join(report_columns) + " |"
    separator = "| " + " | ".join(["---"] * len(report_columns)) + " |"
    table_rows = []
    for row in table.itertuples(index=False, name=None):
        cells = [f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        table_rows.append("| " + " | ".join(cells) + " |")
    markdown_table = "\n".join([header, separator] + table_rows)
    lines = [
        "# LemmaMean versus LemmaSum parcel-wise comparison",
        "",
        f"- LemmaMean source: `{mean_path}`",
        f"- LemmaSum source: `{sum_path}`",
        f"- Matched observations: {len(matched):,} (8 factors × 360 HCPex ROIs)",
        "- FDR decisions were read from each source table and were not recomputed.",
        "",
        "## Overall",
        "",
        f"- Continuous effects: Pearson r = {overall['pearson_r_effect']:.4f}; Spearman ρ = {overall['spearman_rho_effect']:.4f}.",
        f"- FDR-significant pairs: LemmaSum = {int(overall['lemma_sum_significant'])}; LemmaMean = {int(overall['lemma_mean_significant'])}.",
        f"- Shared significant pairs = {int(overall['shared_significant'])}; LemmaMean-only = {int(overall['lemma_mean_only'])}; LemmaSum-only = {int(overall['lemma_sum_only'])}.",
        f"- FDR-set Jaccard = {overall['FDR_jaccard']:.4f}; proportion of LemmaSum findings retained = {overall['sum_result_retained']:.4f}.",
        f"- Shared significant pairs with opposite effect direction = {int(overall['shared_opposite_direction'])}.",
        f"- Mean absolute effect difference = {overall['mean_absolute_delta']:.4f}; RMSE = {overall['rmse']:.4f}.",
        "",
        "## Per-factor summary",
        "",
        markdown_table,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = arguments()
    mean = read_and_validate(args.lemma_mean, "LemmaMean")
    lemma_sum = read_and_validate(args.lemma_sum, "LemmaSum")

    mean_columns = KEY + META + MEASURES
    sum_columns = KEY + MEASURES
    matched = mean[mean_columns].merge(
        lemma_sum[sum_columns], on=KEY, suffixes=("_mean", "_sum"), validate="one_to_one"
    )
    if len(matched) != len(mean) or len(matched) != len(lemma_sum):
        raise ValueError("The source tables do not have identical factor/ROI keys")
    for column in META:
        # Anatomy comes from LemmaMean; source-table equality was verified above by key.
        if column not in matched.columns:
            raise RuntimeError(f"Metadata column lost during merge: {column}")

    # Explicitly confirm anatomical metadata agree before keeping a single copy.
    meta_check = mean[KEY + META].merge(lemma_sum[KEY + META], on=KEY, suffixes=("_mean", "_sum"))
    for column in META:
        left, right = meta_check[f"{column}_mean"], meta_check[f"{column}_sum"]
        equal = left.eq(right) | (left.isna() & right.isna())
        if not equal.all():
            raise ValueError(f"Anatomical metadata differ for column {column}")

    matched = add_comparison_columns(matched)
    factor_rows = [summarize(group, factor) for factor, group in matched.groupby("iFA", sort=True)]
    summary = pd.DataFrame(factor_rows + [summarize(matched, "All")])

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    matched.sort_values(KEY).to_csv(output / "matched_ROI_results.csv", index=False)
    summary.to_csv(output / "comparison_summary_by_factor.csv", index=False)
    matched.loc[matched["FDR_comparison"] != "neither"].sort_values(KEY).to_csv(
        output / "FDR_union_ROIs.csv", index=False
    )
    changes = matched.loc[
        matched["FDR_comparison"].isin(["lemma_mean_only", "lemma_sum_only"])
    ].sort_values(KEY)
    changes.to_csv(output / "FDR_significance_changes.csv", index=False)
    changes.groupby(["Cortical Division", "FDR_comparison"]).size().unstack(fill_value=0).reset_index().to_csv(
        output / "FDR_changes_by_cortical_division.csv", index=False
    )
    matched.loc[matched["effect_sign_reversal"]].sort_values(KEY).to_csv(
        output / "all_effect_direction_reversals.csv", index=False
    )
    matched.sort_values("abs_delta", ascending=False).head(100).to_csv(
        output / "top100_absolute_effect_changes.csv", index=False
    )
    scatter_figure(matched, output)
    overlap_figure(summary, output)
    (output / "comparison_report.md").write_text(
        report(summary, matched, args.lemma_mean, args.lemma_sum)
    )
    print(summary.to_string(index=False))
    print(f"\nOutputs written to {output}")


if __name__ == "__main__":
    main()
