# DiscoFMRI
Code repositry for the Multifaceted neural representation of words in naturalistic language.

### SCOPE
This folder contains the scripts used for extracting pscholinguistic valeus from the SCOPE metabase
* `Batch_SCOPE_FillinLemmas.ipynb`: Lemmatize all the words provided in SCOPE, fill in the missing values with lemma values.
* `gBatch_lemmatize.py`: Lemmatize the raw SCOPE metabase using a 4-step procedure.



### Data
* `List_113variables_sorted.csv`: The original 113 variables selected from `Gao et al., 2022`.
* `SCOPE_lemma_var106.csv`: The final dataset containing 13,850 words and 106 variables.
* `SUBTLEX-UK.xlsx`: The file downloaded from the SUTLEX-UK database containing the POS tags and lemmas.
* `SUBTLEX-US frequency list with PoS and Zipf information.xlsx`: he file downloaded from the SUTLEX-US database containing the POS tags.