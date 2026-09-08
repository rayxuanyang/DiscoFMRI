#!/usr/bin/env python3
"""Rebuild lemma-mean visualization inputs and execute the retained notebook.

The merge uses ``idx`` to replace the original FA1-FA8 scores and the 14
lemma-summed WF/CD predictors with outputs from the lemma-mean EFA. All other
behavioral and psycholinguistic columns are preserved from the original merged
table. The plotting cells themselves live only in
``Batch_MakeFigures_rfirst_fisherz_lemma_mean.ipynb``.
"""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd


N_FACTORS = 8


def prepare_compatibility_inputs(repo_root: Path, output_dir: Path) -> None:
    factor_dir = repo_root / "data" / "FactorAnalysis"
    sensitivity_data = pd.read_csv(
        output_dir / "data_FA_n8_lemma_mean_aligned.csv", low_memory=False
    )
    original_merged = pd.read_csv(
        factor_dir / "df_FA8_behav_merged.csv", low_memory=False
    )

    sum_columns = [
        column
        for column in original_merged.columns
        if column.startswith(("Freq_", "CD_")) and column.endswith("_LemmaSum")
    ]
    if len(sum_columns) != 14:
        raise RuntimeError(f"Expected 14 WF/CD lemma-sum columns, found {len(sum_columns)}")
    mean_column_map = {
        column: f"{column.removesuffix('_LemmaSum')}_LemmaMean"
        for column in sum_columns
    }
    factor_columns = [f"F{i}_n8" for i in range(1, N_FACTORS + 1)]
    source_columns = ["idx", *mean_column_map.values(), *factor_columns]
    missing_sources = set(source_columns) - set(sensitivity_data.columns)
    if missing_sources:
        raise RuntimeError(f"Lemma-mean EFA file lacks: {sorted(missing_sources)}")

    # Keep the notebook-compatible historical names while replacing the values.
    lookup = sensitivity_data[source_columns].copy().rename(
        columns={mean: summed for summed, mean in mean_column_map.items()}
    )
    merged = original_merged.drop(
        columns=[*sum_columns, *[f"FA{i}" for i in range(1, N_FACTORS + 1)]]
    )
    if "FA7_raw" in merged:
        merged = merged.drop(columns="FA7_raw")
    merged = merged.merge(lookup, on="idx", how="left", validate="one_to_one")
    required_values = [*sum_columns, *factor_columns]
    if merged[required_values].isna().any().any():
        raise RuntimeError("Some rows did not receive lemma-mean predictors or scores")

    # Match Batch_mergeFAbehav.ipynb: retain raw F7 and reverse displayed FA7.
    merged["FA7_raw"] = merged["F7_n8"]
    for factor in range(1, N_FACTORS + 1):
        merged[f"FA{factor}"] = merged[f"F{factor}_n8"]
    merged["FA7"] *= -1
    merged = merged.drop(columns=factor_columns)
    merged = merged[original_merged.columns]
    merged.to_csv(output_dir / "df_FA8_behav_merged.csv", index=False)

    full_loadings = pd.read_csv(
        output_dir / "FA_n8_loadings_full_lemma_mean_aligned.csv"
    ).rename(columns={"Variable": "index"})
    full_loadings.to_csv(output_dir / "FA_n8_loadings_full.csv", index=False)

    thresholded = full_loadings.copy()
    loading_columns = [f"Factor {i}" for i in range(1, N_FACTORS + 1)]
    thresholded[loading_columns] = thresholded[loading_columns].mask(
        thresholded[loading_columns].abs() <= 0.4
    )
    thresholded.to_csv(output_dir / "FA_n8_loadings.csv", index=False)

    predictors = pd.read_csv(output_dir / "data_FactorAnalysis_var106_lemma_mean.csv")
    eigenvalues = np.linalg.eigvalsh(predictors.corr().to_numpy())[::-1]
    factors = pd.DataFrame({"ev": eigenvalues})
    factors["explained%"] = factors["ev"] / factors["ev"].sum() * 100
    factors["explained%_cum"] = factors["explained%"].cumsum()
    factors["idx"] = factors.index
    factors.to_csv(output_dir / "df_factorloadings_eigenvalues.csv", index=False)


def validate_notebook_path(notebook_path: Path, output_dir: Path) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    expected = f"Dir_output = r'{output_dir}'"
    assignments = [
        line
        for cell in notebook.cells
        if cell.cell_type == "code"
        for line in cell.source.splitlines()
        if line.startswith("Dir_output =")
    ]
    if assignments != [expected]:
        raise RuntimeError(
            f"Notebook must contain exactly this output assignment: {expected!r}; "
            f"found {assignments!r}"
        )


def execute_notebook_cells(notebook_path: Path) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        try:
            exec(
                compile(cell.source, f"{notebook_path.name}:cell-{index}", "exec"),
                namespace,
            )
        except Exception as error:
            raise RuntimeError(f"Visualization notebook cell {index} failed") from error


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = repo_root / "data" / "FactorAnalysis" / "sensitivity_lemma_mean"
    notebook_path = (
        repo_root
        / "scripts/FactorAnalysis/sensitivity_lemma_mean/"
        "Batch_MakeFigures_rfirst_fisherz_lemma_mean.ipynb"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_compatibility_inputs(repo_root, output_dir)
    validate_notebook_path(notebook_path, output_dir)
    execute_notebook_cells(notebook_path)
    print(f"Rebuilt inputs and executed: {notebook_path}")


if __name__ == "__main__":
    main()
