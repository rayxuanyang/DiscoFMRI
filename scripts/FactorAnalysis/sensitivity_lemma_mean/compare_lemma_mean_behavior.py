#!/usr/bin/env python3
"""Compare original and lemma-mean factor associations with behavior.

This reproduces the direct and mutually adjusted Spearman correlations and
the Fisher-z aggregation in Batch_MakeFigures_rfirst_fisherz.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


FACTOR_COLUMNS = [f"FA{i}" for i in range(1, 9)]
BEHAVIOR_GROUPS = {
    "RT_LD_V": ["LexicalD_RT_V_ELP_z", "LexicalD_RT_V_ECP_z", "LexicalD_RT_V_BLP_z"],
    "RT_LD_A": ["LexicalD_RT_A_MALD_z", "LexicalD_RT_A_AELP_z"],
    "ACC_LD_V": ["LexicalD_ACC_V_ELP", "LexicalD_ACC_V_ECP", "LexicalD_ACC_V_BLP"],
    "ACC_LD_A": ["LexicalD_ACC_A_MALD", "LexicalD_ACC_A_AELP"],
    "ACC_Naming": ["Naming_ACC_ELP"],
    "RT_Naming": ["Naming_RT_ELP_z"],
    "ACC_SD": ["SemanticD_ACC_Calgary"],
    "RT_SD": ["SemanticD_RT_Calgary_z"],
    "ACC_Recog": ["Recog_Memory"],
}
COMBINED_GROUPS = {
    "ACC(3tasks)": ["ACC_LD_V", "ACC_LD_A", "ACC_Naming"],
    "ACC(4tasks)": ["ACC_LD_V", "ACC_LD_A", "ACC_Naming", "ACC_SD"],
    "ACC(5tasks)": ["ACC_LD_V", "ACC_LD_A", "ACC_Naming", "ACC_SD", "ACC_Recog"],
    "RT(3tasks)": ["RT_LD_V", "RT_LD_A", "RT_Naming"],
    "RT(4tasks)": ["RT_LD_V", "RT_LD_A", "RT_Naming", "RT_SD"],
}
DISPLAY_ORDER = [
    "RT_LD_V", "RT_LD_A", "RT_Naming", "RT_SD", "RT(3tasks)", "RT(4tasks)",
    "ACC_LD_V", "ACC_LD_A", "ACC_Naming", "ACC_SD", "ACC_Recog",
    "ACC(3tasks)", "ACC(4tasks)", "ACC(5tasks)",
]


def partial_spearman(
    data: pd.DataFrame, x: str, y: str, covariates: list[str]
) -> tuple[float, float]:
    """Partial Spearman correlation via residualized column ranks."""
    ranked = data[[x, y, *covariates]].rank(method="average").to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(ranked)), ranked[:, 2:]])
    x_residual = ranked[:, 0] - design @ np.linalg.lstsq(
        design, ranked[:, 0], rcond=None
    )[0]
    y_residual = ranked[:, 1] - design @ np.linalg.lstsq(
        design, ranked[:, 1], rcond=None
    )[0]
    r = float(np.corrcoef(x_residual, y_residual)[0, 1])
    dof = len(ranked) - len(covariates) - 2
    t_value = r * np.sqrt(dof / ((1.0 - r) * (1.0 + r)))
    p_value = float(2 * stats.t.sf(abs(t_value), dof))
    return r, p_value


def raw_correlations(data: pd.DataFrame, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = [item for values in BEHAVIOR_GROUPS.values() for item in values]
    correlations = pd.DataFrame(index=outcomes, columns=[*FACTOR_COLUMNS, "N"], dtype=float)
    p_values = pd.DataFrame(index=outcomes, columns=FACTOR_COLUMNS, dtype=float)
    for outcome in outcomes:
        complete = data[[*FACTOR_COLUMNS, outcome]].dropna()
        correlations.loc[outcome, "N"] = len(complete)
        if method == "spearman_r":
            for factor in FACTOR_COLUMNS:
                result = stats.spearmanr(complete[factor], complete[outcome])
                correlations.loc[outcome, factor] = result.statistic
                p_values.loc[outcome, factor] = result.pvalue
        elif method == "spearman_partial_r":
            for factor in FACTOR_COLUMNS:
                covariates = [c for c in FACTOR_COLUMNS if c != factor]
                r, p_value = partial_spearman(
                    complete, factor, outcome, covariates
                )
                correlations.loc[outcome, factor] = r
                p_values.loc[outcome, factor] = p_value
        else:
            raise ValueError(method)
    return correlations, p_values


def add_aggregate(
    names: list[str],
    members: dict[str, list[str]],
    fisher: pd.DataFrame,
) -> None:
    for name in names:
        sources = members[name]
        if len(sources) == 1:
            fisher.loc[name, [*FACTOR_COLUMNS, "N"]] = fisher.loc[
                sources[0], [*FACTOR_COLUMNS, "N"]
            ]
            continue
        weights = fisher.loc[sources, "N"] - 3
        for factor in FACTOR_COLUMNS:
            fisher.loc[name, factor] = np.sum(fisher.loc[sources, factor] * weights) / np.sum(weights)
        fisher.loc[name, "N"] = stats.hmean(fisher.loc[sources, "N"].to_numpy())


def aggregate_correlations(data: pd.DataFrame, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    correlations, p_values = raw_correlations(data, method)
    fisher = np.arctanh(correlations[FACTOR_COLUMNS])
    fisher["N"] = correlations["N"]
    add_aggregate(list(BEHAVIOR_GROUPS), BEHAVIOR_GROUPS, fisher)
    add_aggregate(list(COMBINED_GROUPS), COMBINED_GROUPS, fisher)

    aggregate_names = [*BEHAVIOR_GROUPS, *COMBINED_GROUPS]
    aggregate_r = np.tanh(fisher.loc[aggregate_names, FACTOR_COLUMNS])
    aggregate_r["N"] = fisher.loc[aggregate_names, "N"]
    correlations = pd.concat([correlations, aggregate_r])

    aggregate_z = fisher.loc[aggregate_names, FACTOR_COLUMNS].mul(
        np.sqrt(fisher.loc[aggregate_names, "N"] - 3), axis=0
    )
    aggregate_p = aggregate_z.map(lambda z: 2 * stats.norm.sf(abs(z)))
    p_values = pd.concat([p_values, aggregate_p])
    return correlations, p_values


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    factor_dir = repo_root / "data" / "FactorAnalysis"
    output_dir = factor_dir / "sensitivity_lemma_mean"
    original = pd.read_csv(factor_dir / "df_FA8_behav_merged.csv", low_memory=False)
    mean_scores = pd.read_csv(output_dir / "data_FA_n8_lemma_mean_aligned.csv")
    mean_score_columns = [f"F{i}_n8" for i in range(1, 9)]
    score_lookup = mean_scores[["idx", *mean_score_columns]].rename(
        columns={f"F{i}_n8": f"FA{i}" for i in range(1, 9)}
    )
    # Match Batch_mergeFAbehav.ipynb, which reverses F7 before behavioral
    # analyses so that its interpretation runs from sparse to dense.
    score_lookup["FA7"] *= -1
    mean_data = original.drop(columns=FACTOR_COLUMNS).merge(
        score_lookup, on="idx", how="left", validate="one_to_one"
    )
    if mean_data[FACTOR_COLUMNS].isna().any().any():
        raise RuntimeError("Some behavioral rows did not receive lemma-mean factor scores")

    all_comparisons = []
    summaries = []
    for method in ["spearman_r", "spearman_partial_r"]:
        original_r, original_p = aggregate_correlations(original, method)
        mean_r, mean_p = aggregate_correlations(mean_data, method)
        original_r.loc[DISPLAY_ORDER].to_csv(output_dir / f"behavior_{method}_original.csv")
        mean_r.loc[DISPLAY_ORDER].to_csv(output_dir / f"behavior_{method}_lemma_mean.csv")

        for outcome in DISPLAY_ORDER:
            for factor in FACTOR_COLUMNS:
                old_r = float(original_r.loc[outcome, factor])
                new_r = float(mean_r.loc[outcome, factor])
                old_p = float(original_p.loc[outcome, factor])
                new_p = float(mean_p.loc[outcome, factor])
                all_comparisons.append(
                    {
                        "method": method,
                        "outcome": outcome,
                        "factor": factor,
                        "original_r": old_r,
                        "lemma_mean_r": new_r,
                        "change_r": new_r - old_r,
                        "absolute_change_r": abs(new_r - old_r),
                        "original_p": old_p,
                        "lemma_mean_p": new_p,
                        "original_p_lt_0.001": old_p < 0.001,
                        "lemma_mean_p_lt_0.001": new_p < 0.001,
                        "same_direction": np.sign(old_r) == np.sign(new_r),
                    }
                )
        subset = pd.DataFrame(all_comparisons)
        subset = subset[subset["method"] == method]
        originally_significant = subset[subset["original_p_lt_0.001"]]
        summaries.append(
            {
                "method": method,
                "n_comparisons": len(subset),
                "median_absolute_change_r": subset["absolute_change_r"].median(),
                "max_absolute_change_r": subset["absolute_change_r"].max(),
                "n_same_direction": int(subset["same_direction"].sum()),
                "n_original_p_lt_0.001": int(subset["original_p_lt_0.001"].sum()),
                "n_mean_p_lt_0.001": int(subset["lemma_mean_p_lt_0.001"].sum()),
                "n_original_significant_retained": int(
                    originally_significant["lemma_mean_p_lt_0.001"].sum()
                ),
            }
        )

    comparison = pd.DataFrame(all_comparisons)
    comparison.to_csv(output_dir / "behavior_association_comparison_long.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "behavior_association_comparison_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nLargest changes:")
    print(
        comparison.nlargest(10, "absolute_change_r")[
            ["method", "outcome", "factor", "original_r", "lemma_mean_r", "change_r"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
