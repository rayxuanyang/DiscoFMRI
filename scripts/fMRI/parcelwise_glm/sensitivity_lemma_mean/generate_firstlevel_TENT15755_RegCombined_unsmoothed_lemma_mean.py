#!/usr/bin/env python3
"""Generate the published unsmoothed FA8 first level using lemma-mean scores.

This reproduces ``Batch_firstlevel_TENT15755_RegCombined_unsmoothed.ipynb``.
The only model change is the source of the eight factor-score amplitudes.
Outputs use a separate ``unsmoothed_lemmaMean`` model directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


NVAR = 113
NFA = 8
FLAG_MODEL = f"Nvar{NVAR}NFA{NFA}_LPAC_multipleReg_unsmoothed_lemmaMean"
REPO_ROOT = Path(__file__).resolve().parents[4]
MASTER_FILE = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/"
    "Nastase/allstories/master/codes/master_subject_10stories_highacc.csv"
)
FA_REPO = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/FactorAnalysis/"
    "github/FactorAnalysis_fMRI"
)
WORK_DIR = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/"
    "Nastase/allstories/FactorAnalysis"
)
ORIGINAL_FACTOR_FILE = FA_REPO / "scripts/FactorAnalysis/output/data_FA_n8.csv"
MEAN_FACTOR_FILE = (
    REPO_ROOT
    / "data/FactorAnalysis/sensitivity_lemma_mean/data_FA_n8_lemma_mean_aligned.csv"
)
TIMING_FILE = FA_REPO / "scripts/fMRI/timings/df_timings_15transcripts.csv"
PAC_LEFT_MASK = Path(
    "/work/desai-lab/xuanyang/Project/dataset/Nastase/codes/mask/final/"
    "rs_HCP_AuditoryMask_L_sm6_100.nii.gz"
)
LIBPNG12_DIR = (
    WORK_DIR
    / "models/Nvar113NFA8_LPAC_multipleReg_lemmaMean/runtime/libpng12/lib"
)

CONTENT_POS = [
    "JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "POS", "PRP",
    "PRP$", "RB", "RBR", "RBS", "VB", "VBG", "VBN", "VBD", "VBP", "VBZ",
]
FACTOR_VARS = [f"F{x}_n{NFA}" for x in range(1, NFA + 1)]
DEFAULT_VARS = ["word", "onset", "transcript", "PoS", "functional"]


def prepare_timestamps() -> pd.DataFrame:
    timestamps = pd.read_csv(TIMING_FILE)
    timestamps["word_lower"] = timestamps["word"].str.lower()
    timestamps["idx"] = range(len(timestamps))
    timestamps["functional"] = 1
    timestamps.loc[timestamps["PoS"].isin(CONTENT_POS), "functional"] = 0
    return timestamps


def merge_factors(timestamps: pd.DataFrame, factor_file: Path) -> pd.DataFrame:
    factors = pd.read_csv(factor_file)
    if factors["Word"].duplicated().any():
        raise RuntimeError(f"Duplicate Word values in {factor_file}")
    by_word = timestamps.merge(factors, on="Word", how="left")
    by_gentle = timestamps.merge(
        factors, left_on="word_gentle", right_on="Word", how="left"
    )
    by_lower = timestamps.merge(
        factors, left_on="word_lower", right_on="Word", how="left"
    )
    # Match the notebook: Word, then word_gentle, then lowercase; no lemma fallback.
    return by_word.combine_first(by_gentle).combine_first(by_lower)


def write_lines(path: Path, values: pd.Series) -> None:
    path.write_text("\n".join(values.astype(str)), encoding="utf-8")


def compare_factor_timings(
    timestamps: pd.DataFrame, original: pd.DataFrame, mean: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for transcript in timestamps["transcript"].drop_duplicates():
        old = original.loc[original["transcript"] == transcript, FACTOR_VARS]
        new = mean.loc[mean["transcript"] == transcript, FACTOR_VARS]
        old_complete = old.notna().all(axis=1)
        new_complete = new.notna().all(axis=1)
        if not old_complete.equals(new_complete):
            raise RuntimeError(f"Matched-token set changed for {transcript}")
        row = {
            "transcript": transcript,
            "n_tokens": len(old),
            "n_factor_tokens": int(old_complete.sum()),
            "n_missing_tokens": int((~old_complete).sum()),
        }
        for factor in FACTOR_VARS:
            row[f"{factor}_pearson_r"] = old.loc[old_complete, factor].corr(
                new.loc[new_complete, factor]
            )
            row[f"{factor}_mean_abs_change"] = (
                old.loc[old_complete, factor] - new.loc[new_complete, factor]
            ).abs().mean()
        rows.append(row)
    return pd.DataFrame(rows)


def validate_inputs(master: pd.DataFrame) -> None:
    required = [
        "subID", "task", "label", "event", "confounding", "nosmooth_clean_rs"
    ]
    missing_columns = set(required) - set(master.columns)
    if missing_columns:
        raise RuntimeError(f"Master sheet lacks columns: {sorted(missing_columns)}")
    for column in ["event", "confounding", "nosmooth_clean_rs"]:
        missing = [path for path in master[column] if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"{column}: {len(missing)} inputs do not exist")
    for path in [ORIGINAL_FACTOR_FILE, MEAN_FACTOR_FILE, TIMING_FILE, PAC_LEFT_MASK]:
        if not path.exists():
            raise FileNotFoundError(path)
    if not (LIBPNG12_DIR / "libpng12.so.0").exists():
        raise FileNotFoundError(LIBPNG12_DIR / "libpng12.so.0")


def main() -> None:
    master = pd.read_csv(MASTER_FILE)
    validate_inputs(master)
    timestamps = prepare_timestamps()
    original = merge_factors(timestamps, ORIGINAL_FACTOR_FILE)
    mean = merge_factors(timestamps, MEAN_FACTOR_FILE)

    model_dir = WORK_DIR / "models" / FLAG_MODEL / f"FA{NFA}"
    firstlevel_dir = model_dir / "results/firstlevel"
    scripts_dir = model_dir / "codes/scriptsToRun"
    log_dir = model_dir / "codes/logs"
    for directory in [firstlevel_dir, scripts_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    comparison = compare_factor_timings(timestamps, original, mean)
    comparison.to_csv(model_dir / "factor_timing_comparison.csv", index=False)

    for row_index, row in master.iterrows():
        array_index = f"{row_index + 1:03d}"
        subject, task, label = row["subID"], row["task"], row["label"]
        events = pd.read_csv(row["event"], sep="\t")
        blank_seconds = events.loc[0, "onset"]
        timing = mean.loc[
            mean["transcript"] == label, DEFAULT_VARS + FACTOR_VARS
        ].reset_index(drop=True)
        timing["onset"] += blank_seconds
        timing["missing"] = 1
        timing.loc[timing[FACTOR_VARS].notna().all(axis=1), "missing"] = 0

        task_dir = firstlevel_dir / subject / task
        task_dir.mkdir(parents=True, exist_ok=True)

        missing = timing.loc[timing["missing"] == 1]
        missing.to_csv(task_dir / f"{subject}_{task}_missing.csv", index=False)
        path_missing = task_dir / f"{subject}_{task}_missing.1D"
        write_lines(path_missing, missing["onset"])

        content = timing.loc[
            (timing["functional"] == 0) & (timing["missing"] == 0)
        ]
        content.to_csv(task_dir / f"{subject}_{task}_cont.csv", index=False)
        amplitudes = [",".join(values) for values in content[FACTOR_VARS].astype(str).values]
        path_content = task_dir / f"{subject}_{task}_cont.1D"
        path_content.write_text(
            "\n".join(
                onset + "*" + amplitude
                for onset, amplitude in zip(content["onset"].astype(str), amplitudes)
            ),
            encoding="utf-8",
        )

        function = timing.loc[
            (timing["functional"] == 1) & (timing["missing"] == 0)
        ]
        function.to_csv(task_dir / f"{subject}_{task}_func.csv", index=False)
        path_function = task_dir / f"{subject}_{task}_func.1D"
        write_lines(path_function, function["onset"])

        confounds = pd.read_csv(row["confounding"], sep="\t")
        motion_columns = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
        path_motion = task_dir / "motions_6.1D"
        confounds[motion_columns].to_csv(
            path_motion, index=False, header=False, sep=" "
        )
        outlier_columns = [column for column in confounds if "motion_outlier" in column]
        confounds["motion_outlier"] = 1 - confounds[outlier_columns].sum(axis=1)
        path_outlier = task_dir / "motions_outlier.1D"
        confounds["motion_outlier"].to_csv(
            path_outlier, index=False, header=False, sep=" "
        )

        firstlevel_script = task_dir / f"Batch_firstlevel_{subject}_{task}.sh"
        lines = [
            "#!/bin/tcsh",
            f"/work/apps/AFNI/linux_openmp_64/afni_proc.py -subj_id {subject} \\",
            "-scr_overwrite -blocks mask regress \\",
            f"-mask_import PAC_L {PAC_LEFT_MASK} \\",
            f"-dsets {row['nosmooth_clean_rs']} \\",
            f"-regress_stim_times {path_content} {path_function} {path_missing} \\",
            "-regress_basis 'TENT(1.5,7.5,5)' \\",
            "-regress_ROI PAC_L \\",
            "-regress_stim_types AM2 times times \\",
            "-regress_stim_labels cont func missing \\",
            "-regress_opts_3dD -jobs 6 \\",
        ]
        for factor_index in range(1, NFA + 1):
            tent_indices = " +".join(
                f"cont[{offset + 5 * factor_index}]" for offset in range(1, 5)
            )
            lines.append(
                f"-gltsym 'SYM: {tent_indices}' "
                f"-glt_label {factor_index} FA{factor_index}_bin2345 \\")
        lines.extend([
            "-regress_polort 0 \\",
            "-regress_3dD_stop \\",
            "-regress_reml_exec \\",
            "-regress_make_ideal_sum sum_ideal.1D \\",
            "-regress_run_clustsim no",
            "",
        ])
        firstlevel_script.write_text("\n".join(lines), encoding="utf-8")

        run_script = scripts_dir / f"run_Batch_firstlevel_{array_index}.sh"
        run_script.write_text(
            "\n".join([
                "#!/bin/tcsh",
                f"cd {task_dir}",
                f"tcsh {firstlevel_script}",
                f"tcsh proc.{subject}",
                f"rm {task_dir}/{subject}.results/pb00.*",
                f"rm {task_dir}/{subject}.results/all_runs.*",
                f"rm {task_dir}/{subject}.results/fitts.*",
                f"rm {task_dir}/{subject}.results/errts.*",
                "",
            ]),
            encoding="utf-8",
        )

    slurm_script = model_dir / "codes/Batch_slurm_firstlevel.sh"
    slurm_script.write_text(
        "\n".join([
            "#!/bin/sh",
            f"#SBATCH --job-name={FLAG_MODEL}",
            "#SBATCH -n 6",
            "#SBATCH -N 1",
            f"#SBATCH --output='{log_dir / FLAG_MODEL}-%A_%a.log'",
            f"#SBATCH --error='{log_dir / FLAG_MODEL}-%A_%a.err'",
            f"#SBATCH --array=001-{len(master):03d}",
            "module load afni",
            "module load python3/anaconda/2021.11",
            "export LC_ALL=C",
            f'export LD_LIBRARY_PATH="{LIBPNG12_DIR}:${{LD_LIBRARY_PATH:-}}"',
            'echo "Slurm job ID: " $SLURM_JOB_ID',
            'echo "Slurm array task ID: " $SLURM_ARRAY_TASK_ID',
            'printf -v subj "%03d" $SLURM_ARRAY_TASK_ID',
            'echo "Running first-level regression on array item $subj"',
            f"cd {scripts_dir}",
            "tcsh run_Batch_firstlevel_$subj.sh",
            'echo "Finished first-level regression array item $subj"',
            "date",
            "",
        ]),
        encoding="utf-8",
    )

    manifest = {
        "flag_model": FLAG_MODEL,
        "n_jobs": len(master),
        "source_notebook": str(
            REPO_ROOT
            / "scripts/fMRI/parcelwise_glm/Batch_firstlevel_TENT15755_RegCombined_unsmoothed.ipynb"
        ),
        "factor_source": str(MEAN_FACTOR_FILE),
        "timing_source": str(TIMING_FILE),
        "master_source": str(MASTER_FILE),
        "input_image_column": "nosmooth_clean_rs",
        "pac_regression": "left PAC",
        "factor_score_orientation": "aligned to original raw EFA factors",
        "only_model_change": "eight factor scores from WF/CD lemma-mean EFA",
        "matched_token_sets_identical_to_original": True,
        "afni_compatibility_library": str(LIBPNG12_DIR / "libpng12.so.0"),
    }
    (model_dir / "lemma_mean_firstlevel_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(master)} first-level jobs in {model_dir}")
    print(f"Submit with: sbatch {slurm_script}")


if __name__ == "__main__":
    main()
