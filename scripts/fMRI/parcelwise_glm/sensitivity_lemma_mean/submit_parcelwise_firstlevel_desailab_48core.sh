#!/usr/bin/env bash
#SBATCH --job-name=lemmaMean_HCPex
#SBATCH --partition=desailab-48core
#SBATCH --array=1-16%8
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/work/desai-lab/xuanyang/Project/Semantic/dissemination/github/DiscoFMRI/scripts/fMRI/parcelwise_glm/sensitivity_lemma_mean/slurm_HCPex_%A_%a.out

set -euo pipefail

PYTHONNOUSERSITE=1 /work/xy6/ENVS/AFNI_LMEr/bin/python \
  /work/desai-lab/xuanyang/Project/Semantic/dissemination/github/DiscoFMRI/scripts/fMRI/parcelwise_glm/sensitivity_lemma_mean/run_parcelwise_firstlevel_lemma_mean.py \
  --row "${SLURM_ARRAY_TASK_ID}" --stride 16
