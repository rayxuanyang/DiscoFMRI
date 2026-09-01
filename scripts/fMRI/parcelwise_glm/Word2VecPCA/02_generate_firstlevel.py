#!/usr/bin/env python3
"""Generate the matched Word2Vec-PCA first-level AFNI and Slurm scripts."""

import argparse
from pathlib import Path

import pandas as pd


MASTER = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/"
    "Nastase/allstories/master/codes/master_subject_10stories_highacc.csv"
)
TIMESTAMPS = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/FactorAnalysis/github/"
    "FactorAnalysis_fMRI/scripts/fMRI/timings/df_timings_15transcripts.csv"
)
PC_TABLE = Path(__file__).resolve().parent / "output/data_W2V_PCA_n8.csv"
WORKING = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/"
    "Nastase/allstories/Word2VecPCA"
)
MASK_PAC_L = Path(
    "/work/desai-lab/xuanyang/Project/dataset/Nastase/codes/mask/final/"
    "rs_HCP_AuditoryMask_L_sm6_100.nii.gz"
)
AFNI = Path("/work/apps/AFNI/26.0.08")
MODEL_NAME = "SCOPE13850_W2V300_PCA8_L2_LPAC_multipleReg"
N_PC = 8
PC_COLUMNS = ["PC{}_n{}".format(i, N_PC) for i in range(1, N_PC + 1)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=MASTER)
    parser.add_argument("--timestamps", type=Path, default=TIMESTAMPS)
    parser.add_argument("--pc-table", type=Path, default=PC_TABLE)
    parser.add_argument("--working-dir", type=Path, default=WORKING)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--image-column",
        choices=("smooth_clean_rs", "nosmooth_clean_rs"),
        default="smooth_clean_rs",
        help="Master-table image column (default: smooth_clean_rs).",
    )
    parser.add_argument(
        "--default-partition",
        action="store_true",
        help="Comment out desailab-48core and use Slurm's default partition.",
    )
    return parser.parse_args()


def attach_pc_scores(timestamps, scores):
    """Match scores using the same Word -> gentle -> lowercase order as EFA."""
    result = timestamps.copy()
    score_table = scores[["Word"] + PC_COLUMNS].copy()
    for source in ("Word", "word_gentle", "word_lower"):
        matched = timestamps[[source]].merge(
            score_table,
            left_on=source,
            right_on="Word",
            how="left",
            sort=False,
        )
        for column in PC_COLUMNS:
            if column not in result:
                result[column] = matched[column]
            else:
                result[column] = result[column].combine_first(matched[column])
    return result


def write_lines(path, values):
    with open(path, "w") as stream:
        stream.write("\n".join(values))


def write_afni_proc_script(path, sub_id, image, path_cont, path_func, path_missing):
    with open(path, "w") as stream:
        stream.write("#!/bin/tcsh\n")
        stream.write("setenv LC_ALL C\n")
        stream.write("{} -subj_id {} \\\n".format(AFNI / "afni_proc.py", sub_id))
        stream.write("-scr_overwrite -blocks mask regress \\\n")
        stream.write("-mask_import PAC_L {} \\\n".format(MASK_PAC_L))
        stream.write("-dsets {} \\\n".format(image))
        stream.write(
            "-regress_stim_times {} {} {} \\\n".format(
                path_cont, path_func, path_missing
            )
        )
        stream.write("-regress_basis 'TENT(1.5,7.5,5)' \\\n")
        stream.write("-regress_ROI PAC_L \\\n")
        stream.write("-regress_stim_types AM2 times times \\\n")
        stream.write("-regress_stim_labels cont func missing \\\n")
        stream.write("-regress_opts_3dD -jobs 6 \\\n")
        for pc in range(1, N_PC + 1):
            indices = [1 + 5 * pc, 2 + 5 * pc, 3 + 5 * pc, 4 + 5 * pc]
            expression = " +".join("cont[{}]".format(value) for value in indices)
            stream.write(
                "-gltsym 'SYM: {}' -glt_label {} PC{}_bin2345 \\\n".format(
                    expression, pc, pc
                )
            )
        stream.write("-regress_polort 0 \\\n")
        stream.write("-regress_3dD_stop \\\n")
        stream.write("-regress_reml_exec \\\n")
        stream.write("-regress_compute_fitts \\\n")
        stream.write("-regress_make_ideal_sum sum_ideal.1D \\\n")
        stream.write("-regress_run_clustsim no\n")


def write_run_script(path, sub_id, task, sub_task, proc_script, image):
    result_dir = sub_task / "{}.results".format(sub_id)
    errts = result_dir / "errts.{}_REML+tlrc".format(sub_id)
    r2_map = result_dir / "R2_fullmodel.{}_{}.nii.gz".format(sub_id, task)
    data_sd = result_dir / "_tmp_data_stdev.nii.gz"
    resid_sd = result_dir / "_tmp_resid_stdev.nii.gz"

    with open(path, "w") as stream:
        stream.write("#!/bin/tcsh\n")
        stream.write("setenv LC_ALL C\n")
        stream.write("setenv AFNI_DECONFLICT OVERWRITE\n")
        stream.write("cd {}\n".format(sub_task))
        stream.write("tcsh {}\n".format(proc_script))
        stream.write("tcsh proc.{}\n".format(sub_id))
        stream.write("if ( -f '{}.HEAD' ) then\n".format(errts))
        stream.write(
            "  {}/3dTstat -stdev -prefix {} {}\n".format(AFNI, data_sd, image)
        )
        stream.write(
            "  {}/3dTstat -stdev -prefix {} {}\n".format(AFNI, resid_sd, errts)
        )
        stream.write(
            "  {}/3dcalc -a {} -b {} -expr "
            "'step(a-1e-10)*(1-(b*b)/max(a*a,1e-20))' "
            "-datum float -prefix {}\n".format(AFNI, data_sd, resid_sd, r2_map)
        )
        stream.write("  rm -f {} {}\n".format(data_sd, resid_sd))
        stream.write("else\n")
        stream.write("  echo 'ERROR: missing REML residual dataset {}'\n".format(errts))
        stream.write("  exit 1\n")
        stream.write("endif\n")
        stream.write("rm -f {}/pb00.*\n".format(result_dir))
        stream.write("rm -f {}/all_runs.*\n".format(result_dir))
        stream.write("rm -f {}/fitts.*\n".format(result_dir))
        stream.write("rm -f {}/errts.*\n".format(result_dir))


def main():
    args = parse_args()
    master = pd.read_csv(args.master)
    timestamps = pd.read_csv(args.timestamps)
    scores = pd.read_csv(args.pc_table, keep_default_na=False)

    if len(scores) != 13_850:
        raise ValueError("PC table has {} rows; expected 13,850".format(len(scores)))
    missing_columns = [column for column in PC_COLUMNS if column not in scores]
    if missing_columns:
        raise ValueError("PC table is missing columns {}".format(missing_columns))

    timestamps["word_lower"] = timestamps["word"].str.lower()
    timestamps["idx"] = range(len(timestamps))
    timestamps["functional"] = 1
    content_pos = [
        "JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "POS", "PRP",
        "PRP$", "RB", "RBR", "RBS", "VB", "VBG", "VBN", "VBD", "VBP", "VBZ",
    ]
    timestamps.loc[timestamps["PoS"].isin(content_pos), "functional"] = 0
    final = attach_pc_scores(timestamps, scores)

    coverage = (
        final.assign(has_all_pc=final[PC_COLUMNS].notna().all(axis=1))
        .groupby("transcript")
        .agg(tokens=("word", "size"), tokens_with_pc=("has_all_pc", "sum"))
        .reset_index()
    )
    coverage["tokens_with_pc_percent"] = (
        100 * coverage["tokens_with_pc"] / coverage["tokens"]
    ).round(2)

    model_dir = args.working_dir / "models" / args.model_name / "PC8"
    firstlevel_dir = model_dir / "results/firstlevel"
    scripts_dir = model_dir / "codes/scriptsToRun"
    logs_dir = model_dir / "codes/logs"
    for directory in (firstlevel_dir, scripts_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(model_dir / "token_coverage_by_transcript.csv", index=False)

    default_columns = ["word", "onset", "transcript", "PoS", "functional"]
    for row_number, row in master.iterrows():
        sub_id = row["subID"]
        task = row["task"]
        event = pd.read_csv(row["event"], sep="\t")
        blank_seconds = event.loc[0, "onset"]
        run_tokens = final.loc[
            final["transcript"] == row["label"], default_columns + PC_COLUMNS
        ].reset_index(drop=True)
        run_tokens["onset"] = run_tokens["onset"] + blank_seconds
        run_tokens["missing"] = (~run_tokens[PC_COLUMNS].notna().all(axis=1)).astype(int)

        sub_task = firstlevel_dir / sub_id / task
        sub_task.mkdir(parents=True, exist_ok=True)

        missing = run_tokens.loc[run_tokens["missing"] == 1]
        missing.to_csv(sub_task / "{}_{}_missing.csv".format(sub_id, task), index=False)
        path_missing = sub_task / "{}_{}_missing.1D".format(sub_id, task)
        write_lines(path_missing, missing["onset"].astype(str).tolist())

        content = run_tokens.loc[
            (run_tokens["functional"] == 0) & (run_tokens["missing"] == 0)
        ]
        content.to_csv(sub_task / "{}_{}_cont.csv".format(sub_id, task), index=False)
        path_cont = sub_task / "{}_{}_cont.1D".format(sub_id, task)
        amplitudes = content[PC_COLUMNS].astype(str).agg(",".join, axis=1)
        write_lines(
            path_cont,
            ["{}*{}".format(onset, amplitude) for onset, amplitude in zip(content["onset"], amplitudes)],
        )

        function = run_tokens.loc[
            (run_tokens["functional"] == 1) & (run_tokens["missing"] == 0)
        ]
        function.to_csv(sub_task / "{}_{}_func.csv".format(sub_id, task), index=False)
        path_func = sub_task / "{}_{}_func.1D".format(sub_id, task)
        write_lines(path_func, function["onset"].astype(str).tolist())

        confounds = pd.read_csv(row["confounding"], sep="\t")
        motions = confounds[["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]]
        motions.to_csv(sub_task / "motions_6.1D", index=False, header=False, sep=" ")
        outlier_columns = [column for column in confounds if "motion_outlier" in column]
        confounds["motion_outlier"] = 1 - confounds[outlier_columns].sum(axis=1)
        confounds["motion_outlier"].to_csv(
            sub_task / "motions_outlier.1D", index=False, header=False
        )

        proc_script = sub_task / "Batch_firstlevel_{}_{}.sh".format(sub_id, task)
        write_afni_proc_script(
            proc_script,
            sub_id,
            row[args.image_column],
            path_cont,
            path_func,
            path_missing,
        )
        run_script = scripts_dir / "run_Batch_firstlevel_{:03d}.sh".format(row_number + 1)
        write_run_script(
            run_script, sub_id, task, sub_task, proc_script, row[args.image_column]
        )

    slurm = model_dir / "codes/Batch_slurm_firstlevel.sh"
    with open(slurm, "w") as stream:
        stream.write("#!/bin/sh\n")
        stream.write("#SBATCH --job-name={}\n".format(args.model_name))
        stream.write("#SBATCH -n 6\n#SBATCH -N 1\n")
        if args.default_partition:
            stream.write("# SBATCH -p desailab-48core\n")
        else:
            stream.write("#SBATCH -p desailab-48core\n")
        stream.write("#SBATCH --output='{}-%A_%a.log'\n".format(logs_dir / args.model_name))
        stream.write("#SBATCH --error='{}-%A_%a.err'\n".format(logs_dir / args.model_name))
        stream.write("#SBATCH --array=001-{:03d}\n".format(len(master)))
        stream.write("export PATH={}:$PATH\n".format(AFNI))
        stream.write('printf -v run_id "%03d" "$SLURM_ARRAY_TASK_ID"\n')
        stream.write("cd {}\n".format(scripts_dir))
        stream.write('tcsh "run_Batch_firstlevel_${run_id}.sh"\n')

    print("Generated {} first-level run scripts".format(len(master)))
    print("Slurm array: {}".format(slurm))
    print("Model directory: {}".format(model_dir))


if __name__ == "__main__":
    main()
