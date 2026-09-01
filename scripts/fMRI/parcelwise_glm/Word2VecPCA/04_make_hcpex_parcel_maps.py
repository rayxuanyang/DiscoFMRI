#!/usr/bin/env python3
"""Average unsmoothed PC coefficient maps within the authoritative HCPex atlas."""

import argparse
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


MASTER = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/"
    "Nastase/allstories/master/codes/master_subject_10stories_highacc.csv"
)
WORKING = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/"
    "Nastase/allstories/Word2VecPCA"
)
MODEL_NAME = "SCOPE13850_W2V300_PCA8_L2_LPAC_multipleReg_unsmoothed"
ATLAS = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/FactorAnalysis/github/"
    "FactorAnalysis_fMRI/scripts/fMRI/masks/rs_HCPex.nii.gz"
)
GM_MASK = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/FactorAnalysis/github/"
    "FactorAnalysis_fMRI/scripts/fMRI/masks/rs_mask_GM_33.nii.gz"
)
AFNI_INFO = Path("/work/apps/AFNI/26.0.08/3dinfo")
N_PC = 8
EXPECTED_RUNS = 213


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=MASTER)
    parser.add_argument("--working-dir", type=Path, default=WORKING)
    parser.add_argument("--atlas", type=Path, default=ATLAS)
    parser.add_argument("--gm-mask", type=Path, default=GM_MASK)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def firstlevel_stats(model_dir, row):
    return (
        model_dir
        / "PC8/results/firstlevel"
        / row.subID
        / row.task
        / "{}.results".format(row.subID)
        / "stats.{}_REML+tlrc.HEAD".format(row.subID)
    )


def output_file(parcel_root, row, pc):
    return (
        parcel_root
        / "results/firstlevel"
        / row.subID
        / row.task
        / "{}_{}_HCPex_PC{}.nii.gz".format(row.subID, MODEL_NAME, pc)
    )


def coefficient_index(stats, pc):
    label = "PC{}_bin2345#0_Coef".format(pc)
    result = subprocess.run(
        [str(AFNI_INFO), "-label2index", label, str(stats)],
        check=True,
        capture_output=True,
        text=True,
    )
    index = int(result.stdout.strip().splitlines()[-1])
    if index < 0:
        raise RuntimeError("Missing {} in {}".format(label, stats))
    return index


def main():
    args = parse_args()
    master = pd.read_csv(args.master)
    if len(master) != EXPECTED_RUNS:
        raise ValueError("Master has {} rows; expected 213".format(len(master)))

    model_root = args.working_dir / "models" / MODEL_NAME
    parcel_root = model_root / "PC8_HCPex"
    parcel_root.mkdir(parents=True, exist_ok=True)

    sources = []
    for row in master.itertuples(index=False):
        stats = firstlevel_stats(model_root, row)
        if not stats.is_file():
            raise FileNotFoundError(stats)
        sources.append(stats)

    subbricks = {
        pc: coefficient_index(sources[0], pc) for pc in range(1, N_PC + 1)
    }

    atlas_img = nib.load(str(args.atlas))
    mask_img = nib.load(str(args.gm_mask))
    atlas_float = np.asarray(atlas_img.dataobj)
    mask = np.asarray(mask_img.dataobj) > 0

    if atlas_float.shape != mask.shape:
        raise ValueError("Atlas and GM mask shapes differ")
    if not np.allclose(atlas_img.affine, mask_img.affine):
        raise ValueError("Atlas and GM mask affines differ")
    if not np.allclose(atlas_float, np.rint(atlas_float)):
        raise ValueError("HCPex atlas contains non-integer labels")

    atlas = np.rint(atlas_float).astype(np.int32)
    masked_labels = atlas[mask]
    if np.any(masked_labels < 0):
        raise ValueError("HCPex atlas contains negative labels")
    max_label = int(masked_labels.max())
    counts = np.bincount(masked_labels, minlength=max_label + 1)
    valid_labels = counts > 0

    pd.DataFrame(
        {
            "ROI_idx": np.flatnonzero(valid_labels),
            "voxel_count_in_GM_mask": counts[valid_labels],
        }
    ).to_csv(parcel_root / "HCPex_parcel_voxel_counts.csv", index=False)

    manifest = []
    completed = 0
    skipped = 0
    for run_index, (row, stats) in enumerate(
        zip(master.itertuples(index=False), sources), start=1
    ):
        outputs = [
            output_file(parcel_root, row, pc) for pc in range(1, N_PC + 1)
        ]
        if not args.overwrite and all(path.is_file() for path in outputs):
            skipped += 1
            for pc, path in enumerate(outputs, start=1):
                manifest.append(
                    {
                        "run_index": run_index,
                        "Subj": row.subID,
                        "task": row.task,
                        "transcript": row.transcript,
                        "PC": pc,
                        "subbrick": subbricks[pc],
                        "InputFile": str(path),
                    }
                )
            continue

        stats_img = nib.load(str(stats))
        if stats_img.shape[:3] != mask.shape:
            raise ValueError("Grid shape mismatch: {}".format(stats))
        if not np.allclose(stats_img.affine, mask_img.affine):
            raise ValueError("Grid affine mismatch: {}".format(stats))

        for pc, path in enumerate(outputs, start=1):
            values = np.asarray(
                stats_img.dataobj[..., subbricks[pc]], dtype=np.float64
            )
            parcel_values = values[mask]
            if not np.isfinite(parcel_values).all():
                raise ValueError("Non-finite PC{} values in {}".format(pc, stats))

            sums = np.bincount(
                masked_labels,
                weights=parcel_values,
                minlength=max_label + 1,
            )
            means = np.zeros(max_label + 1, dtype=np.float64)
            means[valid_labels] = sums[valid_labels] / counts[valid_labels]

            output_data = np.zeros(mask.shape, dtype=np.float32)
            output_data[mask] = means[masked_labels].astype(np.float32)
            output_img = nib.Nifti1Image(
                output_data,
                mask_img.affine,
                header=mask_img.header.copy(),
            )
            output_img.set_data_dtype(np.float32)
            path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(output_img, str(path))

            manifest.append(
                {
                    "run_index": run_index,
                    "Subj": row.subID,
                    "task": row.task,
                    "transcript": row.transcript,
                    "PC": pc,
                    "subbrick": subbricks[pc],
                    "InputFile": str(path),
                }
            )

        completed += 1
        if run_index % 10 == 0 or run_index == len(master):
            print(
                "Parcelized {}/{} runs ({} newly written, {} skipped)".format(
                    run_index, len(master), completed, skipped
                ),
                flush=True,
            )

    manifest_table = pd.DataFrame(manifest).sort_values(["run_index", "PC"])
    manifest_table.to_csv(parcel_root / "parcel_map_manifest.csv", index=False)
    print("Wrote {} parcel maps under {}".format(len(manifest_table), parcel_root))
    print("PC sub-bricks: {}".format(subbricks))


if __name__ == "__main__":
    main()
