#!/bin/sh
# Submit runs 191-213 after the first chunk, then queue full output validation.
set -eu

MODEL_DIR=/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/Nastase/allstories/FactorAnalysis/models/Nvar113NFA8_LPAC_multipleReg_unsmoothed_lemmaMean/FA8
RUN_SCRIPT="$MODEL_DIR/codes/Batch_slurm_firstlevel.sh"
LOG_DIR="$MODEL_DIR/codes/logs"
VALIDATOR=/work/desai-lab/xuanyang/Project/Semantic/dissemination/github/DiscoFMRI/scripts/fMRI/parcelwise_glm/sensitivity_lemma_mean/validate_firstlevel_unsmoothed_lemma_mean.py
PYTHON=/work/apps/python3/anaconda/2021.11/bin/python
JOB_RECORD="$MODEL_DIR/default_partition_job_ids.txt"

SECOND_JOB=$(sbatch --parsable --array=191-213 "$RUN_SCRIPT")
VALIDATION_JOB=$(sbatch --parsable \
    --dependency="afterany:$SECOND_JOB" \
    --job-name=validate_unsmoothed_lemmaMean \
    --output="$LOG_DIR/validate-%j.log" \
    --error="$LOG_DIR/validate-%j.err" \
    --wrap="$PYTHON $VALIDATOR")

{
    echo "second_array=$SECOND_JOB"
    echo "validation=$VALIDATION_JOB"
} >> "$JOB_RECORD"
echo "Submitted second array: $SECOND_JOB"
echo "Submitted validation: $VALIDATION_JOB"
