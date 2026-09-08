#!/usr/bin/env python3
"""Validate lemma-mean parcel maps and parcel-wise 3dLMEr outputs."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


MASTER = Path("/work/desai-lab/xuanyang/Project/Semantic/dissemination/github/DiscoFMRI/scripts/fMRI/master_subject_10stories_highacc.csv")
BASE = Path("/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/Nastase/allstories/FactorAnalysis/models")
MODEL = "Nvar113NFA8_LPAC_multipleReg_unsmoothed_lemmaMean"
FIRST = BASE / MODEL / "FA8_HCPex/results/firstlevel"
SECOND = BASE / MODEL / "FA8_HCPex/results/secondlevel/nonthreshold_LMEr_r1.4"
MASK = Path("/work/desai-lab/xuanyang/Project/Semantic/analysis/FactorAnalysis/github/FactorAnalysis_fMRI/scripts/fMRI/masks/rs_mask_GM_33.nii.gz")


def main() -> None:
    master = pd.read_csv(MASTER)
    mask_img = nib.load(MASK)
    expected = []
    for row in master.itertuples():
        for factor in range(1, 9):
            expected.append(FIRST / str(row.subID) / str(row.task) / f"{row.subID}_{MODEL}_HCPex_FA{factor}.nii.gz")
    missing = [path for path in expected if not path.exists()]
    bad = []
    for path in expected:
        if not path.exists():
            continue
        img = nib.load(path)
        if img.shape != mask_img.shape or not np.allclose(img.affine, mask_img.affine):
            bad.append(str(path))
    lmer = [SECOND / f"FA{factor}+tlrc.HEAD" for factor in range(1, 9)]
    present_lmer = sum(path.exists() for path in lmer)
    print(f"parcel_expected={len(expected)} parcel_present={len(expected)-len(missing)} parcel_bad_geometry={len(bad)}")
    print(f"lmer_expected=8 lmer_present={present_lmer}")
    if missing:
        print(f"first_missing={missing[0]}")
    if bad:
        print(f"first_bad_geometry={bad[0]}")
    if missing or bad or present_lmer != 8:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
