#!/usr/bin/env python3
"""Sensitivity analysis replacing lemma-summed WF/CD with lemma means.

Only predictors beginning with ``Freq_`` or ``CD_`` are changed. Means are
computed over the linear-scale values already created by the original lemma
pipeline, preserving its lemma assignments, missing-value filling, analysis
sample, remaining predictors, and eight-factor EFA specification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr


N_FACTORS = 8
LOADING_CUTOFF = 0.4
EXCLUDED_PREDICTORS = {
    "Orth_N_Freq_L",
    "Orth_N_Freq_G",
    "Orth_N_Freq",
    "Orth_N_Freq_L_Mean",
    "Orth_N_Freq_G_Mean",
    "Phonographic_N_Freq",
    "Phon_N_Freq",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_intermediate = Path(
        "/work/desai-lab/xuanyang/Project/Semantic/analysis/"
        "FactorAnalysis/github/data/SCOPE_lemma.csv"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--lemma-intermediate",
        type=Path,
        default=default_intermediate,
        help="Archived SCOPE_lemma.csv produced by Batch_SCOPE_FillinLemmas.ipynb",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: data/FactorAnalysis/sensitivity_lemma_mean",
    )
    return parser.parse_args()


def tucker_congruence(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else np.nan


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repo_root / "data" / "FactorAnalysis" / "sensitivity_lemma_mean"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scope_file = repo_root / "data" / "SCOPE" / "SCOPE_lemma_var106.csv"
    variable_file = repo_root / "data" / "SCOPE" / "List_113variables_sorted.csv"
    original_data_file = repo_root / "data" / "FactorAnalysis" / "data_FactorAnalysis_var106.csv"
    original_scores_file = repo_root / "data" / "FactorAnalysis" / "scores_FA_n8.csv"
    original_loadings_file = repo_root / "data" / "FactorAnalysis" / "FA_n8_loadings_full.csv"

    scope = pd.read_csv(scope_file, low_memory=False)
    original_x = pd.read_csv(original_data_file)
    variable_table = pd.read_csv(variable_file)

    selected = []
    for variable in variable_table["variable"]:
        if variable in EXCLUDED_PREDICTORS:
            continue
        selected.append(
            f"{variable}_LemmaSum"
            if variable.startswith(("Freq_", "CD_"))
            else variable
        )
    if selected != original_x.columns.tolist():
        raise RuntimeError("Reconstructed predictor order differs from the original analysis table")
    if len(selected) != 106:
        raise RuntimeError(f"Expected 106 predictors, found {len(selected)}")

    sum_columns = [c for c in selected if c.startswith(("Freq_", "CD_"))]
    base_columns = [c.removesuffix("_LemmaSum") for c in sum_columns]
    expo_columns = [f"{c}_expo" for c in base_columns]

    intermediate = pd.read_csv(
        args.lemma_intermediate,
        usecols=["idx", "Lemma", *expo_columns, *sum_columns],
        low_memory=False,
    )
    if intermediate["idx"].duplicated().any():
        raise RuntimeError("The archived lemma intermediate has duplicate idx values")

    target_lookup = scope[["idx", "Word", "Lemma"]].merge(
        intermediate,
        on="idx",
        how="left",
        validate="one_to_one",
        suffixes=("_target", "_intermediate"),
    )
    lemma_matches = target_lookup["Lemma_target"].eq(target_lookup["Lemma_intermediate"])
    if not lemma_matches.all():
        raise RuntimeError(f"Lemma mismatch for {(~lemma_matches).sum()} target rows")

    # Verify that this is exactly the intermediate used for the published sums.
    for column in sum_columns:
        if not np.allclose(
            target_lookup[column].to_numpy(),
            scope[column].to_numpy(),
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise RuntimeError(f"Archived and published values differ for {column}")

    # This is the controlled change: mean rather than sum of linear-scale
    # frequency/contextual-diversity values within each lemma.
    grouped_means = intermediate.groupby("Lemma", sort=False)[expo_columns].mean()
    grouped_counts = intermediate.groupby("Lemma", sort=False)[expo_columns].count()
    target_lemmas = target_lookup["Lemma_target"]

    mean_values = grouped_means.reindex(target_lemmas).reset_index(drop=True)
    valid_counts = grouped_counts.reindex(target_lemmas).reset_index(drop=True)
    if mean_values.isna().any().any():
        missing = int(mean_values.isna().sum().sum())
        raise RuntimeError(f"Mean aggregation generated {missing} missing values")

    new_x = original_x.copy()
    rename_map = {}
    aggregation_rows = []
    for sum_column, base_column, expo_column in zip(sum_columns, base_columns, expo_columns):
        mean_column = f"{base_column}_LemmaMean"
        mean_log10 = np.log10(mean_values[expo_column].to_numpy())
        new_x[sum_column] = mean_log10
        rename_map[sum_column] = mean_column

        original_values = original_x[sum_column].to_numpy()
        counts = valid_counts[expo_column].to_numpy()
        expected = original_values - np.log10(counts)
        max_identity_error = float(np.max(np.abs(mean_log10 - expected)))
        aggregation_rows.append(
            {
                "original_variable": sum_column,
                "sensitivity_variable": mean_column,
                "n_words": len(mean_log10),
                "n_unique_lemmas": int(target_lemmas.nunique()),
                "min_valid_forms": int(counts.min()),
                "median_valid_forms": float(np.median(counts)),
                "max_valid_forms": int(counts.max()),
                "mean_change_log10": float(np.mean(mean_log10 - original_values)),
                "median_change_log10": float(np.median(mean_log10 - original_values)),
                "max_abs_sum_mean_identity_error": max_identity_error,
            }
        )

    new_x = new_x.rename(columns=rename_map)
    new_predictor_order = [rename_map.get(c, c) for c in selected]
    new_x = new_x[new_predictor_order]
    pd.DataFrame(aggregation_rows).to_csv(
        output_dir / "lemma_mean_aggregation_summary.csv", index=False
    )

    unchanged = [c for c in selected if c not in sum_columns]
    max_other_difference = float(
        np.max(
            np.abs(
                new_x[unchanged].to_numpy(dtype=float)
                - original_x[unchanged].to_numpy(dtype=float)
            )
        )
    )
    if max_other_difference != 0:
        raise RuntimeError("A non-frequency/contextual-diversity predictor changed")

    new_x.to_csv(output_dir / "data_FactorAnalysis_var106_lemma_mean.csv", index=False)

    model = FactorAnalyzer(
        rotation="oblimin", method="principal", n_factors=N_FACTORS
    )
    model.fit(new_x)
    raw_scores = model.transform(new_x)
    factor_names = [f"F{i}_n{N_FACTORS}" for i in range(1, N_FACTORS + 1)]
    raw_scores_df = pd.DataFrame(raw_scores, columns=factor_names)
    raw_scores_df.to_csv(output_dir / "scores_FA_n8_lemma_mean_raw.csv", index=False)

    raw_loadings = pd.DataFrame(
        model.loadings_,
        index=new_predictor_order,
        columns=[f"Factor {i}" for i in range(1, N_FACTORS + 1)],
    )
    raw_loadings.reset_index(names="Variable").to_csv(
        output_dir / "FA_n8_loadings_full_lemma_mean_raw.csv", index=False
    )

    original_loadings = pd.read_csv(original_loadings_file)
    original_variable_column = original_loadings.columns[0]
    original_loadings = original_loadings.set_index(original_variable_column)
    original_factor_columns = [f"Factor {i}" for i in range(1, N_FACTORS + 1)]
    original_loadings = original_loadings[original_factor_columns].reindex(selected)

    canonical_new_loadings = raw_loadings.rename(index={v: k for k, v in rename_map.items()})
    canonical_new_loadings = canonical_new_loadings.reindex(selected)
    congruence = np.empty((N_FACTORS, N_FACTORS))
    for old_idx in range(N_FACTORS):
        for new_idx in range(N_FACTORS):
            congruence[old_idx, new_idx] = tucker_congruence(
                original_loadings.iloc[:, old_idx].to_numpy(),
                canonical_new_loadings.iloc[:, new_idx].to_numpy(),
            )
    old_indices, new_indices = linear_sum_assignment(-np.abs(congruence))
    mapping = dict(zip(old_indices, new_indices))

    original_scores = pd.read_csv(original_scores_file)[factor_names]
    aligned_loadings = pd.DataFrame(index=canonical_new_loadings.index)
    aligned_scores = pd.DataFrame(index=new_x.index)
    comparison_rows = []
    for old_idx in range(N_FACTORS):
        new_idx = mapping[old_idx]
        signed_congruence = congruence[old_idx, new_idx]
        sign = 1.0 if signed_congruence >= 0 else -1.0
        old_name = factor_names[old_idx]
        aligned_loadings[f"Factor {old_idx + 1}"] = (
            canonical_new_loadings.iloc[:, new_idx] * sign
        )
        aligned_scores[old_name] = raw_scores[:, new_idx] * sign
        pearson_r = pearsonr(original_scores[old_name], aligned_scores[old_name]).statistic
        spearman_r = spearmanr(original_scores[old_name], aligned_scores[old_name]).statistic
        comparison_rows.append(
            {
                "original_factor": old_idx + 1,
                "matched_raw_mean_factor": new_idx + 1,
                "sign_applied": int(sign),
                "tucker_congruence": abs(float(signed_congruence)),
                "score_pearson_r": float(pearson_r),
                "score_spearman_r": float(spearman_r),
                "score_mean_absolute_difference": float(
                    np.mean(np.abs(original_scores[old_name] - aligned_scores[old_name]))
                ),
            }
        )

    aligned_scores.to_csv(output_dir / "scores_FA_n8_lemma_mean_aligned.csv", index=False)
    aligned_loadings.reset_index(names="Variable").to_csv(
        output_dir / "FA_n8_loadings_full_lemma_mean_aligned.csv", index=False
    )
    thresholded = aligned_loadings.mask(aligned_loadings.abs() <= LOADING_CUTOFF)
    thresholded.reset_index(names="Variable").to_csv(
        output_dir / "FA_n8_loadings_lemma_mean_aligned_cutoff_0.4.csv", index=False
    )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "factor_solution_comparison.csv", index=False)

    metadata = scope[["idx", "Word", "Lemma"]].reset_index(drop=True)
    combined = pd.concat([metadata, new_x.reset_index(drop=True), aligned_scores], axis=1)
    combined.to_csv(output_dir / "data_FA_n8_lemma_mean_aligned.csv", index=False)

    variance = model.get_factor_variance()
    pd.DataFrame(
        {
            "raw_mean_factor": np.arange(1, N_FACTORS + 1),
            "ss_loadings": variance[0],
            "proportion_variance": variance[1],
            "cumulative_variance": variance[2],
        }
    ).to_csv(output_dir / "factor_variance_lemma_mean_raw.csv", index=False)

    manifest = {
        "n_words": int(len(new_x)),
        "n_predictors": int(new_x.shape[1]),
        "n_changed_predictors": int(len(sum_columns)),
        "n_unchanged_predictors": int(len(unchanged)),
        "changed_original_predictors": sum_columns,
        "max_absolute_difference_other_predictors": max_other_difference,
        "factor_method": "principal",
        "rotation": "oblimin",
        "n_factors": N_FACTORS,
        "loading_cutoff_for_display_only": LOADING_CUTOFF,
        "source_intermediate": str(args.lemma_intermediate.resolve()),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote sensitivity analysis to {output_dir}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
