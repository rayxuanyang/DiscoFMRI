#!/usr/bin/env python3
"""Validate all parcel-wise lemma-mean unsmoothed first-level outputs."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd


MODEL_ROOT = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/"
    "Nastase/allstories/FactorAnalysis/models/"
    "Nvar113NFA8_LPAC_multipleReg_unsmoothed_lemmaMean"
)
FA_ROOT = MODEL_ROOT / "FA8"
FIRSTLEVEL_ROOT = FA_ROOT / "results/firstlevel"
MASTER_FILE = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/"
    "Nastase/allstories/master/codes/master_subject_10stories_highacc.csv"
)
AFNI_3DINFO = Path("/work/apps/AFNI/linux_openmp_64/3dinfo")
LIBPNG_DIR = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/"
    "Nastase/allstories/FactorAnalysis/models/"
    "Nvar113NFA8_LPAC_multipleReg_lemmaMean/runtime/libpng12/lib"
)
REPORT_JSON = FA_ROOT / "firstlevel_validation_report.json"
REPORT_CSV = FA_ROOT / "firstlevel_validation_by_run.csv"


def main() -> int:
    master = pd.read_csv(MASTER_FILE)
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LD_LIBRARY_PATH"] = str(LIBPNG_DIR) + ":" + env.get("LD_LIBRARY_PATH", "")
    rows = []
    for _, source in master.iterrows():
        subject, task = source["subID"], source["task"]
        result_dir = FIRSTLEVEL_ROOT / subject / task / f"{subject}.results"
        stem = result_dir / f"stats.{subject}_REML+tlrc"
        head = Path(str(stem) + ".HEAD")
        brik = Path(str(stem) + ".BRIK")
        brik_gz = Path(str(stem) + ".BRIK.gz")
        head_ok = head.is_file() and head.stat().st_size > 0
        brik_ok = (
            (brik.is_file() and brik.stat().st_size > 0)
            or (brik_gz.is_file() and brik_gz.stat().st_size > 0)
        )
        labels_ok = False
        label_error = ""
        if head_ok:
            checked = subprocess.run(
                [str(AFNI_3DINFO), "-label", str(head)],
                env=env,
                text=True,
                capture_output=True,
            )
            labels_ok = checked.returncode == 0 and all(
                f"FA{index}_bin2345" in checked.stdout for index in range(1, 9)
            )
            if checked.returncode:
                label_error = checked.stderr.strip()
        rows.append({
            "subject": subject,
            "task": task,
            "head_ok": head_ok,
            "brik_ok": brik_ok,
            "eight_factor_contrasts_ok": labels_ok,
            "label_error": label_error,
            "result_dir": str(result_dir),
        })

    details = pd.DataFrame(rows)
    details["valid"] = details[
        ["head_ok", "brik_ok", "eight_factor_contrasts_ok"]
    ].all(axis=1)
    details.to_csv(REPORT_CSV, index=False)
    invalid = details.loc[~details["valid"], ["subject", "task"]]
    report = {
        "expected_runs": len(details),
        "valid_runs": int(details["valid"].sum()),
        "invalid_runs": int((~details["valid"]).sum()),
        "all_valid": bool(details["valid"].all()),
        "invalid_subject_tasks": invalid.to_dict(orient="records"),
        "details_file": str(REPORT_CSV),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["all_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
