LOMO repeated20x5 GHPI sensitivity analysis. Same stratified folds as archived baseline, seeds 42-61.
Each model removal updates median model-specific GD_global and refits pooled Logit coefficients using only five remaining model error outcomes; training-derived P75 is refit in each fold.
Held-out model errors are excluded from calibrating GHPI but are evaluated on same sentence corpus: not an external validation.
GHPI coefficients use only GD_global, Semantic_Depth and Negation_Norm; intercept intentionally omitted from score, following source code.
Repeated OOF score and threshold are each aggregated by observation-level median.
Positive recall tests: two-sided Fisher if any Pearson expected count <5, otherwise uncorrected Pearson chi-square; BH across six evaluated models (baseline-compatible); separately BH across the five calibration models; and BH across six omitted-model tests (exploratory).
All tests and model set sensitivity analyses are exploratory; BH per condition does not control FDR across all LOMO families.
Original raw data, notebooks and results have not been overwritten.
