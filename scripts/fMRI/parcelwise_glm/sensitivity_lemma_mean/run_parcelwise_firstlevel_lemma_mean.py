#!/usr/bin/env python3
"""Create HCPex parcel-mean maps from lemma-mean first-level AFNI results.

This reproduces Batch_ROI_firstlevel.ipynb.  The only analysis change is the
model root: the input coefficients are the eight lemma-mean EFA regressors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
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
INPUT_ROOT = WORKING / "models" / MODEL / "FA8" / "results" / "firstlevel"
OUTPUT_ROOT = WORKING / "models" / MODEL / "FA8_HCPex" / "results" / "firstlevel"
MASK_ROOT = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/FactorAnalysis/github/"
    "FactorAnalysis_fMRI/scripts/fMRI/masks"
)
ATLAS = MASK_ROOT / "rs_HCPex.nii.gz"
GM_MASK = MASK_ROOT / "rs_mask_GM_33.nii.gz"
FA_SUBBRICKS = (114, 117, 120, 123, 126, 129, 132, 135)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--row",
        type=int,
        help="One-based master-sheet row to process (default: all 213 rows).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="With --row, also process row+stride, row+2*stride, etc.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing parcel maps."
    )
    return parser.parse_args()


def parcel_means(data: np.ndarray, labels: np.ndarray, in_mask: np.ndarray) -> np.ndarray:
    """Assign each in-mask voxel its parcel's mean (including atlas label 0).

    The notebook loops over ``np.unique(data_atlas)`` after setting atlas labels
    outside the GM mask to zero.  Restricting assignment to ``in_mask`` and
    retaining label zero reproduces that behavior efficiently.
    """
    valid_labels = labels[in_mask]
    if not np.all(valid_labels == np.rint(valid_labels)):
        raise ValueError("HCPex atlas contains non-integer labels")
    valid_labels = valid_labels.astype(np.int64)
    valid_data = np.asarray(data[in_mask], dtype=np.float64)
    if not np.isfinite(valid_data).all():
        raise ValueError("Input coefficient map contains non-finite in-mask values")
    sums = np.bincount(valid_labels, weights=valid_data)
    counts = np.bincount(valid_labels)
    means = np.zeros_like(sums, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts != 0)
    output = np.zeros(data.shape, dtype=np.float32)
    output[in_mask] = means[valid_labels].astype(np.float32)
    return output


def main() -> None:
    args = arguments()
    master = pd.read_csv(MASTER)
    if args.row is not None:
        if not 1 <= args.row <= len(master):
            raise SystemExit(f"--row must be between 1 and {len(master)}")
        if args.stride < 1:
            raise SystemExit("--stride must be positive")
        master = master.iloc[args.row - 1 :: args.stride]

    atlas_img = nib.load(ATLAS)
    mask_img = nib.load(GM_MASK)
    labels = np.asarray(atlas_img.dataobj)
    in_mask = np.asarray(mask_img.dataobj) == 1
    labels = labels.copy()
    labels[~in_mask] = 0

    if labels.shape != mask_img.shape or not np.allclose(atlas_img.affine, mask_img.affine):
        raise ValueError("Atlas and GM mask geometries differ")

    written = skipped = 0
    for master_index, row in master.iterrows():
        subject, task = str(row.subID), str(row.task)
        stats = (
            INPUT_ROOT
            / subject
            / task
            / f"{subject}.results"
            / f"stats.{subject}_REML+tlrc.HEAD"
        )
        if not stats.exists():
            raise FileNotFoundError(stats)
        stats_img = nib.load(stats)
        if stats_img.shape[:3] != labels.shape or not np.allclose(stats_img.affine, mask_img.affine):
            raise ValueError(f"Geometry mismatch: {stats}")

        output_dir = OUTPUT_ROOT / subject / task
        output_dir.mkdir(parents=True, exist_ok=True)
        for factor, brick in enumerate(FA_SUBBRICKS, start=1):
            output = output_dir / f"{subject}_{MODEL}_HCPex_FA{factor}.nii.gz"
            if output.exists() and not args.overwrite:
                skipped += 1
                continue
            data = np.asanyarray(stats_img.dataobj[..., brick])
            parcel_data = parcel_means(data, labels, in_mask)
            header = mask_img.header.copy()
            header.set_data_dtype(np.float32)
            nib.save(nib.Nifti1Image(parcel_data, mask_img.affine, header), output)
            written += 1
        print(f"row={master_index + 1:03d} subject={subject} task={task}", flush=True)

    print(f"written={written} skipped={skipped} output_root={OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
