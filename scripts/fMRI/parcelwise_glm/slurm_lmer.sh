#!/bin/sh
### name your job
#SBATCH --job-name=afni_lmer
### specify number of cores to use
#SBATCH -n 6
#SBATCH -N 1
### which partition do you want to use?
# SBATCH -p desailab-48core
### %a specifies that the output file for each subject will be named according to their array ID

# Where to output log files?
#SBATCH --output='/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/Nastase/allstories/FactorAnalysis/models/Nvar113NFA8_LPAC_multipleReg_unsmoothed/FA8_HCPex/results/secondlevel/nonthreshold_LMEr_r1.3/lmer-%A_%a.log'
#SBATCH --error='/work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/Nastase/allstories/FactorAnalysis/models/Nvar113NFA8_LPAC_multipleReg_unsmoothed/FA8_HCPex/results/secondlevel/nonthreshold_LMEr_r1.3/lmer-%A_%a.err'

### list the subjects that you want to submit in this array
#SBATCH --array=1-8
echo "Start load modules"
#source /share/apps/Modules/3.2.10/init/modules.sh

# module load singularity/3.11.2 freesurfer/7.2.0
source ~/.bashrc

module load slurm/16.05.8
start_env_afni_lmer

echo "Running afni lmer"

cd /work/desai-lab/xuanyang/Project/Semantic/analysis/ParametricModulation/Nastase/allstories/FactorAnalysis/models/Nvar113NFA8_LPAC_multipleReg_unsmoothed/FA8_HCPex/results/secondlevel/nonthreshold_LMEr_r1.3
tcsh Batch_secondlevel_FA${SLURM_ARRAY_TASK_ID}

date



