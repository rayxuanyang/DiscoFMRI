#!/bin/sh
### name your job
#SBATCH --job-name=spacyonly
### specify number of cores to use
#SBATCH -n 6
#SBATCH -N 1
### which partition do you want to use?
# SBATCH -p desailab-48core
### %a specifies that the output file for each subject will be named according to their array ID

# Where to output log files?
#SBATCH --output='/work/desai-lab/xuanyang/Project/Semantic/dissemination/github/DiscoFMRI/data/SCOPE/spacyonly-%A_%a.log'
#SBATCH --error='/work/desai-lab/xuanyang/Project/Semantic/dissemination/github/DiscoFMRI/data/SCOPE/spacyonly-%A_%a.err'

### list the subjects that you want to submit in this array
#SBATCH --array=1-1
echo "Start load modules"
#source /share/apps/Modules/3.2.10/init/modules.sh

# module load singularity/3.11.2 freesurfer/7.2.0
source ~/.bashrc

module load slurm/16.05.8
module load python3/anaconda/2023.9
start_env_hyperalignment
conda deactivate
conda activate /work/xy6/ENVS/spacy

echo "Running spacy only"

cd /work/desai-lab/xuanyang/Project/Semantic/dissemination/github/DiscoFMRI/scripts/SCOPE/
python Batch_SCOPE_FillinLemmas_spacyonly.py

date



