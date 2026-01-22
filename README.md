# DiscoFMRI
Code repositry for the Multifaceted neural representation of words in naturalistic language.

### SCOPE
This folder contains the scripts used for extracting pscholinguistic valeus from the SCOPE metabase
* `Batch_SCOPE_FillinLemmas.ipynb`: Lemmatize all the words provided in SCOPE, fill in the missing values with lemma values.
* `gBatch_lemmatize.py`: Lemmatize the raw SCOPE metabase using a 4-step procedure.

### FactorAnalylsis
This folder contains the scripts used running Exploratory Factor Analysis (EFA) and examining the behavioral correspondance.
* `Batch_factorAnalysis.ipynb`: Run EFA analysis.
* `Batch_mergeFAbehav.ipynb`: Integrate the factor scores with the behavioral outcome variables from SCOPE.
* `Batch_MakeFigures_rfirst_fisherz.ipynb`: Make Figures 1-3.
* `Batch_MakeTableBestPredictor.ipynb`: Compare the predictive performance between factor scores and single variables. 
* `Batch_makeTables_S1.ipynb`: Make Table S1. 


### Data
* `./data/SCOPE/List_113variables_sorted.csv`: The original 113 variables selected from `Gao et al., 2022`.
* `./data/SCOPE/SCOPE_lemma_var106.csv`: The final dataset containing 13,850 words and 106 variables.
* `./data/SCOPE/SUBTLEX-UK.xlsx`: The file downloaded from the SUTLEX-UK database containing the POS tags and lemmas.
* `./data/SCOPE/SUBTLEX-US frequency list with PoS and Zipf information.xlsx`: he file downloaded from the SUTLEX-US database containing the POS tags.
* `./data/FactorAnalysis/data_FactorAnalysis_var106.csv`: The final dataset used for Factor Analysis.
* `./data/FactorAnalysis/df_FA8_behav_merged.csv`: The final dataset used for examining the behavioral correspondance.
