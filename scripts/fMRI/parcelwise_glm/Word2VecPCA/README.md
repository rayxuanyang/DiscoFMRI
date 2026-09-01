# Word2Vec PCA control analysis

This is a matched control for the existing eight-factor fMRI analysis. It uses
the same 13,850 SCOPE words, reduces the 300-dimensional SCOPE Word2Vec vectors
to eight PCs, and keeps the existing first-level AFNI model unchanged apart
from replacing `F1_n8`-`F8_n8` with `PC1_n8`-`PC8_n8`.

Run scripts in order after loading the project Python environment:

```bash
module load python3/anaconda/2021.11
python3 01_make_word2vec_pca.py
python3 02_generate_firstlevel.py
```

The second command generates, but does not submit, the first-level Slurm array.
Submit the generated `Batch_slurm_firstlevel.sh`, wait for completion, and then
generate the group-level mixed-effects analysis:

```bash
python3 03_generate_secondlevel_lmer.py
```

The group script reproduces the existing model:

```text
protocol + (1|Subj) + (1|transcript)
```

and its equal-weight mean across `skyra`, `Prisma_MB3`, and `Prisma_MB4`.

Each first-level run also creates `R2_fullmodel.<sub>_<task>.nii.gz` before the
temporary residual time series is deleted. This is descriptive in-sample full-
model R-squared:

```text
R2 = 1 - Var(REML residuals) / Var(input BOLD)
```

It includes PCs, word-event regressors, PAC, and all other terms in the model;
it is not partial R-squared attributable only to Word2Vec. The group generator
also writes a cheap arithmetic mean R-squared command. That map is a descriptive
summary only; it is not a t-test. All PC coefficient inference uses `3dLMEr`,
with one LMER for each of PC1-PC8. The generator also writes the matching
residual ACF and `3dClustSim` script.
