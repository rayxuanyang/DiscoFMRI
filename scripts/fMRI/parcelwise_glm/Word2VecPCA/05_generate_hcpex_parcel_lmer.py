#!/usr/bin/env python3
"""Generate the authoritative HCPex parcel-wise 3dLMEr analysis for PC1-PC8."""

import argparse
from pathlib import Path

import pandas as pd


MASTER = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/dissemination/github/"
    "DiscoFMRI/scripts/fMRI/master_subject_10stories_highacc.csv"
)
WORKING = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/"
    "Nastase/allstories/Word2VecPCA"
)
MODEL_NAME = "SCOPE13850_W2V300_PCA8_L2_LPAC_multipleReg_unsmoothed"
GM_MASK = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/FactorAnalysis/github/"
    "FactorAnalysis_fMRI/scripts/fMRI/masks/rs_mask_GM_33.nii.gz"
)
AFNI = Path("/work/apps/AFNI/26.0.08")
N_PC = 8
EXPECTED_RUNS = 213
TRANSCRIPT_GLT = (
    "transcript : "
    "0.046948*21styear "
    "+0.187793*black "
    "+0.107981*bronx "
    "+0.107981*forgot "
    "+0.065728*milkywayoriginal "
    "+0.079812*milkywayvodka "
    "+0.140845*piemanpni "
    "+0.164319*prettymouth "
    "+0.084507*shapessocial "
    "+0.014085*slumlordreach"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=MASTER)
    parser.add_argument("--working-dir", type=Path, default=WORKING)
    parser.add_argument("--gm-mask", type=Path, default=GM_MASK)
    return parser.parse_args()


def parcel_file(parcel_root, row, pc):
    return (
        parcel_root
        / "results/firstlevel"
        / row.subID
        / row.task
        / "{}_{}_HCPex_PC{}.nii.gz".format(row.subID, MODEL_NAME, pc)
    )


def write_lmer_script(path, pc, data_table, mask):
    with open(path, "w") as stream:
        stream.write("#!/bin/tcsh -xef\n")
        stream.write("setenv LC_ALL C\n")
        stream.write("setenv AFNI_DECONFLICT OVERWRITE\n")
        stream.write("{} -prefix PC{} \\\n".format(AFNI / "3dLMEr", pc))
        stream.write("-resid PC{}_resid \\\n".format(pc))
        stream.write("-mask {} \\\n".format(mask))
        stream.write(
            "-model 'transcript+age_c+sex+comprehension+(1|Subj)' \\\n"
        )
        stream.write("-qVars 'age_c,comprehension' \\\n")
        stream.write("-IF InputFile \\\n")
        stream.write("-SS_type 3 \\\n")
        stream.write("-gltCode mean '{}' \\\n".format(TRANSCRIPT_GLT))
        stream.write("-gltCode age 'age_c :' \\\n")
        stream.write("-gltCode comprehension 'comprehension :' \\\n")
        stream.write("-gltCode male_vs_female 'sex : 1*M -1*F' \\\n")
        stream.write("-dataTable @{}\n".format(data_table))


def write_slurm(path, result_dir):
    with open(path, "w") as stream:
        stream.write("#!/bin/sh\n")
        stream.write("#SBATCH --job-name=w2v_pc_hcpex_lmer\n")
        stream.write("#SBATCH -n 6\n#SBATCH -N 1\n")
        stream.write("# SBATCH -p desailab-48core\n")
        stream.write("#SBATCH --output='{}'\n".format(result_dir / "lmer-%A_%a.log"))
        stream.write("#SBATCH --error='{}'\n".format(result_dir / "lmer-%A_%a.err"))
        stream.write("#SBATCH --array=1-8\n")
        stream.write("source ~/.bashrc\n")
        stream.write("module load slurm/16.05.8\n")
        stream.write("start_env_afni_lmer\n")
        stream.write("cd {}\n".format(result_dir))
        stream.write('tcsh "Batch_secondlevel_PC${SLURM_ARRAY_TASK_ID}"\n')


def write_acf_script(path, result_dir, mask):
    with open(path, "w") as stream:
        stream.write("#!/bin/bash\nset -euo pipefail\n")
        stream.write("export LC_ALL=C\n")
        stream.write("cd {}\n".format(result_dir))
        stream.write("for pc in {1..8}; do\n")
        stream.write(
            '  "{}/3dFWHMx" -mask "{}" -ACF NULL '
            '-input "PC${{pc}}_resid+orig" > "PC${{pc}}_ACF.txt"\n'.format(
                AFNI, mask
            )
        )
        stream.write("done\n")


def main():
    args = parse_args()
    master = pd.read_csv(args.master)
    if len(master) != EXPECTED_RUNS:
        raise ValueError("Master has {} rows; expected 213".format(len(master)))

    master["age"] = pd.to_numeric(master["age"], errors="raise")
    master["comprehension"] = pd.to_numeric(
        master["comprehension"], errors="raise"
    )
    age_reference = master.groupby("subID")["age"].mean().mean()
    master["age_c"] = master["age"] - age_reference

    if set(master["sex"].dropna().unique()) != {"F", "M"}:
        raise ValueError("Expected sex categories F and M")
    transcript_weights = {
        "21styear": 0.046948,
        "black": 0.187793,
        "bronx": 0.107981,
        "forgot": 0.107981,
        "milkywayoriginal": 0.065728,
        "milkywayvodka": 0.079812,
        "piemanpni": 0.140845,
        "prettymouth": 0.164319,
        "shapessocial": 0.084507,
        "slumlordreach": 0.014085,
    }
    if set(master["transcript"].unique()) != set(transcript_weights):
        raise ValueError("Transcript levels differ from the authoritative GLT")
    if abs(sum(transcript_weights.values()) - 1.0) > 2e-6:
        raise ValueError("Transcript GLT weights do not sum to one")

    parcel_root = args.working_dir / "models" / MODEL_NAME / "PC8_HCPex"
    result_dir = parcel_root / "results/secondlevel/nonthreshold_LMEr_r1.4"
    result_dir.mkdir(parents=True, exist_ok=True)

    for pc in range(1, N_PC + 1):
        paths = [
            parcel_file(parcel_root, row, pc)
            for row in master.itertuples(index=False)
        ]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "PC{} is missing {} parcel maps; first: {}".format(
                    pc, len(missing), missing[0]
                )
            )

        table = master[
            ["subID", "transcript", "age_c", "sex", "comprehension"]
        ].copy()
        table.rename(columns={"subID": "Subj"}, inplace=True)
        table["InputFile"] = [str(path) for path in paths]
        data_table = result_dir / "dataTable_PC{}.txt".format(pc)
        table.to_csv(data_table, sep="\t", index=False)

        write_lmer_script(
            result_dir / "Batch_secondlevel_PC{}".format(pc),
            pc,
            data_table,
            args.gm_mask,
        )

    write_slurm(result_dir / "slurm_lmer.sh", result_dir)
    write_acf_script(result_dir / "Batch_estimateACF.sh", result_dir, args.gm_mask)
    pd.DataFrame(
        [{"age_reference_unique_participant_mean": age_reference}]
    ).to_csv(result_dir / "model_metadata.csv", index=False)

    print("Generated eight HCPex parcel LMER scripts")
    print("Runs per PC: {}".format(len(master)))
    print("Age reference: {:.12f}".format(age_reference))
    print("Model: transcript + age_c + sex + comprehension + (1|Subj)")
    print("Result directory: {}".format(result_dir))


if __name__ == "__main__":
    main()
