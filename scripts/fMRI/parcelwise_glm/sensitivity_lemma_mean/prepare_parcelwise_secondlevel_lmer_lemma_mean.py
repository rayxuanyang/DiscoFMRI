#!/usr/bin/env python3
"""Prepare the eight parcel-wise lemma-mean 3dLMEr analyses."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MASTER = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/dissemination/github/DiscoFMRI/"
    "scripts/fMRI/master_subject_10stories_highacc.csv"
)
WORKING = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/"
    "Nastase/allstories/FactorAnalysis"
)
MODEL = "Nvar113NFA8_LPAC_multipleReg_unsmoothed_lemmaMean"
MODEL_ROOT = WORKING / "models" / MODEL / "FA8_HCPex" / "results"
FIRSTLEVEL = MODEL_ROOT / "firstlevel"
SECONDLEVEL = MODEL_ROOT / "secondlevel" / "nonthreshold_LMEr_r1.4"
GROUP_MASK = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/FactorAnalysis/github/"
    "FactorAnalysis_fMRI/scripts/fMRI/masks/rs_mask_GM_33.nii.gz"
)
AFNI = Path("/work/apps/AFNI/26.0.08")
TRANSCRIPT_GLT = (
    "transcript : 0.046948*21styear +0.187793*black +0.107981*bronx "
    "+0.107981*forgot +0.065728*milkywayoriginal +0.079812*milkywayvodka "
    "+0.140845*piemanpni +0.164319*prettymouth +0.084507*shapessocial "
    "+0.014085*slumlordreach"
)


def main() -> None:
    master = pd.read_csv(MASTER)
    master["age"] = pd.to_numeric(master["age"], errors="raise")
    age_reference = master.groupby("subID")["age"].mean().mean()
    master["age_c"] = master["age"] - age_reference
    if set(master["sex"].dropna().unique()) != {"F", "M"}:
        raise ValueError("Expected sex levels F and M")

    SECONDLEVEL.mkdir(parents=True, exist_ok=True)
    for factor in range(1, 9):
        inputs = [
            FIRSTLEVEL
            / str(row.subID)
            / str(row.task)
            / f"{row.subID}_{MODEL}_HCPex_FA{factor}.nii.gz"
            for row in master.itertuples()
        ]
        missing = [path for path in inputs if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"FA{factor}: missing {len(missing)} parcel maps; first is {missing[0]}"
            )
        table = master[["subID", "transcript", "age_c", "sex", "comprehension"]].copy()
        table.rename(columns={"subID": "Subj"}, inplace=True)
        table["InputFile"] = [str(path) for path in inputs]
        table.to_csv(SECONDLEVEL / f"dataTable_FA{factor}.txt", index=False, sep="\t")

        script = SECONDLEVEL / f"Batch_secondlevel_FA{factor}"
        script.write_text(
            "#!/bin/tcsh -xef\n"
            f"{AFNI / '3dLMEr'} -prefix FA{factor} \\\n"
            f"-resid FA{factor}_resid \\\n"
            f"-mask {GROUP_MASK} \\\n"
            "-model 'transcript+age_c+sex+comprehension+(1|Subj)' \\\n"
            "-qVars 'age_c,comprehension' \\\n"
            "-IF InputFile \\\n"
            "-SS_type 3 \\\n"
            f"-gltCode mean '{TRANSCRIPT_GLT}' \\\n"
            "-gltCode age 'age_c :' \\\n"
            "-gltCode comprehension 'comprehension :' \\\n"
            "-gltCode male_vs_female 'sex : 1*M -1*F' \\\n"
            f"-dataTable @/{SECONDLEVEL}/dataTable_FA{factor}.txt\n"
        )
        script.chmod(0o755)

    slurm = SECONDLEVEL / "submit_parcelwise_lmer_desailab_48core.sh"
    slurm.write_text(
        "#!/usr/bin/env bash\n"
        "#SBATCH --job-name=lemmaMean_ROI_LMEr\n"
        "#SBATCH --partition=desailab-48core\n"
        "#SBATCH --array=1-8\n"
        "#SBATCH --cpus-per-task=4\n"
        "#SBATCH --mem=24G\n"
        "#SBATCH --time=12:00:00\n"
        f"#SBATCH --output={SECONDLEVEL}/slurm_LMEr_%A_%a.out\n"
        "set -euo pipefail\n"
        "export TMPDIR=/work/xy6/tmp\n"
        "export PATH=/work/apps/AFNI/26.0.08:/work/xy6/ENVS/AFNI_LMEr/bin:${PATH}\n"
        "export LD_LIBRARY_PATH=/work/xy6/ENVS/AFNI_LMEr/lib/R/lib:/work/xy6/ENVS/AFNI_LMEr/lib:${LD_LIBRARY_PATH:-}\n"
        "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}\n"
        f"cd {SECONDLEVEL}\n"
        "tcsh ./Batch_secondlevel_FA${SLURM_ARRAY_TASK_ID}\n"
    )
    slurm.chmod(0o755)
    print(f"age_reference={age_reference:.12g}")
    print(f"rows={len(master)} factors=8 output={SECONDLEVEL}")
    print(slurm)


if __name__ == "__main__":
    main()
