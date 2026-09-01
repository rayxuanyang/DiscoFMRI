#!/usr/bin/env python3
"""Generate the eight matched 3dLMEr analyses after first-level completion.

This script does not run a group model. It verifies first-level outputs,
resolves coefficient sub-bricks by AFNI label, and writes LMER, Slurm, ACF,
ClustSim, and descriptive group-mean R-squared scripts.
"""

import argparse
import subprocess
from pathlib import Path

import pandas as pd


MASTER = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/"
    "Nastase/allstories/master/codes/master_subject_10stories_highacc.csv"
)
WORKING = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/"
    "Nastase/allstories/Word2VecPCA"
)
PROTOCOL_REFERENCE = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/"
    "Nastase/allstories/FactorAnalysis/models/Nvar113NFA8_LPAC_multipleReg/FA8/"
    "results/secondlevel/threshold_GM_LMEr/"
    "dataTable_FA1_Nvar113NFA8_LPAC_multipleReg_213highacc.txt"
)
GM_MASK = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/FactorAnalysis/github/"
    "FactorAnalysis_fMRI/scripts/fMRI/masks/rs_mask_GM_33.nii.gz"
)
AFNI = Path("/work/apps/AFNI/26.0.08")
MODEL_NAME = "SCOPE13850_W2V300_PCA8_L2_LPAC_multipleReg"
N_PC = 8
EXPECTED_RUNS = 213


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=MASTER)
    parser.add_argument("--working-dir", type=Path, default=WORKING)
    parser.add_argument("--protocol-reference", type=Path, default=PROTOCOL_REFERENCE)
    parser.add_argument("--gm-mask", type=Path, default=GM_MASK)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Generate from completed runs instead of requiring all 213.",
    )
    return parser.parse_args()


def firstlevel_paths(model_dir, row):
    result_dir = (
        model_dir
        / "results/firstlevel"
        / row.subID
        / row.task
        / "{}.results".format(row.subID)
    )
    stats = result_dir / "stats.{}_REML+tlrc".format(row.subID)
    r2_map = result_dir / "R2_fullmodel.{}_{}.nii.gz".format(row.subID, row.task)
    return stats, r2_map


def coefficient_index(stats, pc):
    label = "PC{}_bin2345#0_Coef".format(pc)
    completed = subprocess.run(
        [str(AFNI / "3dinfo"), "-label2index", label, str(stats)],
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip().splitlines()[-1]
    try:
        index = int(value)
    except ValueError as error:
        raise RuntimeError(
            "Could not resolve {!r} in {}: {!r}".format(label, stats, value)
        ) from error
    if index < 0:
        raise RuntimeError("Missing {!r} in {}".format(label, stats))
    return index


def write_lmer_script(path, prefix, data_table, mask):
    with open(path, "w") as stream:
        stream.write("#!/bin/tcsh -xef\n")
        stream.write("setenv AFNI_DECONFLICT OVERWRITE\n")
        stream.write("{} -prefix {} \\\n".format(AFNI / "3dLMEr", prefix))
        stream.write("-resid {}_resid \\\n".format(prefix))
        stream.write("-mask {} \\\n".format(mask))
        stream.write("-model 'protocol+(1|Subj)+(1|transcript)' \\\n")
        stream.write("-IF InputFile \\\n")
        stream.write("-SS_type 3 \\\n")
        stream.write(
            "-gltCode mean 'protocol : 0.333333*skyra +0.333333*Prisma_MB3 "
            "+0.333333*Prisma_MB4' \\\n"
        )
        stream.write("-dataTable @{}\n".format(data_table))


def write_slurm(path, group_dir, run_count):
    with open(path, "w") as stream:
        stream.write("#!/bin/sh\n")
        stream.write("#SBATCH --job-name=w2v_pca_lmer\n")
        stream.write("#SBATCH -n 6\n#SBATCH -N 1\n")
        stream.write("#SBATCH --output='{}'\n".format(group_dir / "lmer-%A_%a.log"))
        stream.write("#SBATCH --error='{}'\n".format(group_dir / "lmer-%A_%a.err"))
        stream.write("#SBATCH --array=1-8\n")
        stream.write("source ~/.bashrc\n")
        stream.write("module load slurm/16.05.8\n")
        stream.write("start_env_afni_lmer\n")
        stream.write("cd {}\n".format(group_dir))
        stream.write(
            'tcsh "Batch_secondlevel_PC${{SLURM_ARRAY_TASK_ID}}_'
            '{}_{}highacc.sh"\n'.format(MODEL_NAME, run_count)
        )


def write_cluster_script(path, group_dir, mask, run_count):
    with open(path, "w") as stream:
        stream.write("#!/bin/bash\nset -euo pipefail\n")
        stream.write("cd {}\n".format(group_dir))
        stream.write("for pc in {1..8}; do\n")
        stream.write(
            '  prefix="PC${{pc}}_{}_{}highacc"\n'.format(MODEL_NAME, run_count)
        )
        stream.write(
            '  "{}/3dFWHMx" -mask "{}" -ACF NULL '
            '-input "${{prefix}}_resid+tlrc" > "${{prefix}}_ACF.txt"\n'.format(
                AFNI, mask
            )
        )
        stream.write(
            "  read -r acf_a acf_b acf_c < <(awk "
            "'NF >= 4 && $1 !~ /^#/ {a=$1; b=$2; c=$3} "
            "END {print a, b, c}' \"${prefix}_ACF.txt\")\n"
        )
        stream.write(
            '  "{}/3dClustSim" -mask "{}" -acf "$acf_a" "$acf_b" '
            '"$acf_c" -pthr 0.001 -athr 0.05 -LOTS '
            '-prefix "${{prefix}}.CSimA"\n'.format(AFNI, mask)
        )
        stream.write("done\n")


def write_group_mean_r2(path, output, r2_maps):
    with open(path, "w") as stream:
        stream.write("#!/bin/tcsh -xef\n")
        stream.write("setenv AFNI_DECONFLICT OVERWRITE\n")
        stream.write("{} -prefix {} \\\n".format(AFNI / "3dMean", output))
        for index, r2_map in enumerate(r2_maps):
            ending = " \\\n" if index < len(r2_maps) - 1 else "\n"
            stream.write("{}{}".format(r2_map, ending))


def main():
    args = parse_args()
    master = pd.read_csv(args.master)
    if len(master) != EXPECTED_RUNS:
        raise ValueError("Master table has {} rows; expected 213".format(len(master)))

    protocol = pd.read_csv(
        args.protocol_reference,
        sep="\t",
        usecols=["Subj", "protocol", "transcript"],
    )
    conflicting = protocol.groupby(["Subj", "transcript"])["protocol"].nunique()
    if (conflicting > 1).any():
        raise ValueError("Protocol reference contains conflicting scanner assignments")
    protocol = protocol.drop_duplicates(["Subj", "transcript"])
    master = master.merge(
        protocol,
        left_on=["subID", "transcript"],
        right_on=["Subj", "transcript"],
        how="left",
        validate="many_to_one",
    )
    if master["protocol"].isna().any():
        missing = master.loc[
            master["protocol"].isna(), ["subID", "transcript"]
        ]
        raise ValueError(
            "Missing protocol assignments:\n{}".format(missing.to_string(index=False))
        )

    model_dir = args.working_dir / "models" / MODEL_NAME / "PC8"
    available_rows = []
    missing_stats = []
    missing_r2 = []
    for row in master.itertuples(index=False):
        stats, r2_map = firstlevel_paths(model_dir, row)
        if Path(str(stats) + ".HEAD").is_file():
            available_rows.append((row, stats, r2_map))
            if not r2_map.is_file():
                missing_r2.append(r2_map)
        else:
            missing_stats.append(stats)

    if missing_stats and not args.allow_incomplete:
        raise RuntimeError(
            "Only {}/{} first-level stats datasets exist. Finish the Slurm array "
            "before generating LMER scripts. First missing: {}".format(
                len(available_rows), len(master), missing_stats[0]
            )
        )
    if not available_rows:
        raise RuntimeError("No completed first-level stats datasets were found")

    run_count = len(available_rows)
    group_dir = model_dir / "results/secondlevel/threshold_GM_LMEr"
    group_dir.mkdir(parents=True, exist_ok=True)
    representative_stats = available_rows[0][1]

    for pc in range(1, N_PC + 1):
        subbrick = coefficient_index(representative_stats, pc)
        table_rows = []
        for row, stats, _ in available_rows:
            table_rows.append(
                {
                    "Subj": row.subID,
                    "protocol": row.protocol,
                    "transcript": row.transcript,
                    "InputFile": "{}[{}]".format(stats, subbrick),
                }
            )
        prefix = "PC{}_{}_{}highacc".format(pc, MODEL_NAME, run_count)
        data_table = group_dir / "dataTable_{}.txt".format(prefix)
        pd.DataFrame(table_rows).to_csv(data_table, sep="\t", index=False)
        lmer_script = group_dir / "Batch_secondlevel_{}.sh".format(prefix)
        write_lmer_script(lmer_script, prefix, data_table, args.gm_mask)

    write_slurm(group_dir / "slurm_lmer.sh", group_dir, run_count)
    write_cluster_script(
        group_dir / "Batch_estimateACF_and_ClustSim.sh",
        group_dir,
        args.gm_mask,
        run_count,
    )

    if missing_r2:
        print(
            "Warning: {} full-model R-squared maps are missing; no group-mean "
            "R-squared script was written.".format(len(missing_r2))
        )
    else:
        group_r2 = group_dir / "R2_fullmodel_groupMean_{}highacc.nii.gz".format(
            run_count
        )
        write_group_mean_r2(
            group_dir / "Batch_groupMean_R2.sh",
            group_r2,
            [item[2] for item in available_rows],
        )

    print("Generated eight LMER scripts from {} first-level runs".format(run_count))
    print("LMER model: protocol + (1|Subj) + (1|transcript)")
    print("Group directory: {}".format(group_dir))


if __name__ == "__main__":
    main()
