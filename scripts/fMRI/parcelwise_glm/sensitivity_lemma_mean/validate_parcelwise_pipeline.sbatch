#!/usr/bin/env bash
#SBATCH --job-name=lemmaMean_ROI_check
#SBATCH --partition=desailab-48core
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/work/desai-lab/xuanyang/Project/Semantic/dissemination/github/DiscoFMRI/scripts/fMRI/parcelwise_glm/sensitivity_lemma_mean/slurm_validate_parcelwise_%j.out

set -euo pipefail
PYTHONNOUSERSITE=1 /work/xy6/ENVS/AFNI_LMEr/bin/python \
  /work/desai-lab/xuanyang/Project/Semantic/dissemination/github/DiscoFMRI/scripts/fMRI/parcelwise_glm/sensitivity_lemma_mean/validate_parcelwise_lemma_mean.py
