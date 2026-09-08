# LemmaMean versus LemmaSum parcel-wise comparison

- LemmaMean source: `/work/desai-lab/xuanyang/Project/Semantic/analysis/FactorAnalysis/github/FactorAnalysis_fMRI/data/HCPex_p005FDR/df_ROIvalue_HCPex_p005FDR_unsmoothed_LMEr_r1.4_lemmaMean.csv`
- LemmaSum source: `/work/desai-lab/xuanyang/Project/Semantic/analysis/FactorAnalysis/github/FactorAnalysis_fMRI/data/HCPex_p005FDR/df_ROIvalue_HCPex_p005FDR_unsmoothed_LMEr_r1.4.csv`
- Matched observations: 2,880 (8 factors × 360 HCPex ROIs)
- FDR decisions were read from each source table and were not recomputed.

## Overall

- Continuous effects: Pearson r = 0.9922; Spearman ρ = 0.9911.
- FDR-significant pairs: LemmaSum = 320; LemmaMean = 325.
- Shared significant pairs = 295; LemmaMean-only = 30; LemmaSum-only = 25.
- FDR-set Jaccard = 0.8429; proportion of LemmaSum findings retained = 0.9219.
- Shared significant pairs with opposite effect direction = 0.
- Mean absolute effect difference = 0.1877; RMSE = 0.2632.

## Per-factor summary

| iFA | lemma_sum_significant | lemma_mean_significant | shared_significant | lemma_mean_only | lemma_sum_only | FDR_jaccard | sum_result_retained | pearson_r_effect | spearman_rho_effect | mean_absolute_delta | rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 32 | 31 | 25 | 6 | 7 | 0.6579 | 0.7812 | 0.9753 | 0.9689 | 0.3518 | 0.4533 |
| 2 | 44 | 42 | 38 | 4 | 6 | 0.7917 | 0.8636 | 0.9901 | 0.9889 | 0.2219 | 0.2867 |
| 3 | 27 | 28 | 26 | 2 | 1 | 0.8966 | 0.9630 | 0.9987 | 0.9984 | 0.0679 | 0.0882 |
| 4 | 17 | 15 | 15 | 0 | 2 | 0.8824 | 0.8824 | 0.9968 | 0.9949 | 0.1064 | 0.1356 |
| 5 | 18 | 16 | 16 | 0 | 2 | 0.8889 | 0.8889 | 0.9962 | 0.9947 | 0.1208 | 0.1517 |
| 6 | 133 | 138 | 129 | 9 | 4 | 0.9085 | 0.9699 | 0.9958 | 0.9952 | 0.2249 | 0.2852 |
| 7 | 19 | 17 | 16 | 1 | 3 | 0.8000 | 0.8421 | 0.9938 | 0.9938 | 0.1513 | 0.1964 |
| 8 | 30 | 38 | 30 | 8 | 0 | 0.7895 | 1.0000 | 0.9880 | 0.9850 | 0.2565 | 0.3124 |
| All | 320 | 325 | 295 | 30 | 25 | 0.8429 | 0.9219 | 0.9922 | 0.9911 | 0.1877 | 0.2632 |
