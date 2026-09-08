#!/usr/bin/env bash
# Prepare and submit the eight 3dLMEr models after all parcel maps exist.
set -euo pipefail

SCRIPT_DIR=/work/desai-lab/xuanyang/Project/Semantic/dissemination/github/DiscoFMRI/scripts/fMRI/parcelwise_glm/sensitivity_lemma_mean
PYTHONNOUSERSITE=1 /work/xy6/ENVS/AFNI_LMEr/bin/python \
  "$SCRIPT_DIR/prepare_parcelwise_secondlevel_lmer_lemma_mean.py"

SECONDLEVEL=/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/Nastase/allstories/FactorAnalysis/models/Nvar113NFA8_LPAC_multipleReg_unsmoothed_lemmaMean/FA8_HCPex/results/secondlevel/nonthreshold_LMEr_r1.4
lmer_job=$(sbatch --parsable "$SECONDLEVEL/submit_parcelwise_lmer_desailab_48core.sh")
printf 'Submitted parcel-wise 3dLMEr array: %s\n' "$lmer_job"

validation_job=$(sbatch --parsable --dependency="afterany:$lmer_job" "$SCRIPT_DIR/validate_parcelwise_pipeline.sbatch")
printf 'Submitted validation job: %s\n' "$validation_job"
