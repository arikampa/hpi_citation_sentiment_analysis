# Leave-One-Model-Out (LOMO) sensitivity analysis of the GHPI

## Reproducibility and scope

- Source: `Final_Master_results_with_features.csv` from the HPI paper.
- Mixed single-citation subset: **2,156** sentences; real-only single-citation subset: **1,128** sentences, identified with the same IEEE-style citation-pattern rule used in the original notebooks.
- All six original sentiment model predictions and SHAP gap-distance columns are reused (no model retraining, no new SHAP extraction).
- The full six-model baseline and six leave-one-model-out recalibrations were computed within *each* subset. Across conditions the sentence-level stratified folds are held fixed (five folds, 20 repetitions, seeds 42–61).
- In each condition, `GD_global` is the **median of model-specific SHAP gap distances** for the models retained. For each training fold, a pooled logistic error regression is fitted using **only the retained models' error outcomes**, and the coefficients for `GD_global`, `Semantic_Depth`, and `Negation_Norm` are used to compute a GHPI score (intercept excluded, as in the original notebooks). The 75th percentile of training GHPI scores is the low/high pressure threshold.
- A held-out sentence contributes no model-specific error outcome to the calibration of its own OOF score; scores and thresholds for each sentence are separately aggregated by medians over 20 repetitions.
- A model excluded from the calibration is also separately evaluated on the resulting pressure regions, but this is *within-corpus, cross-model sensitivity*, **not external validation**.
- For each sentiment model, the two-sided Fisher test is used if a Pearson chi-square table's expected minimum count is <5; otherwise the uncorrected Pearson chi-square test is used. Benjamini–Hochberg correction is reported in distinct, explicitly identified families: all six evaluated models per condition (matching the paper's original convention); the five remaining calibration models per LOMO condition; and the six excluded-model diagnostic comparisons within each subset. Results of different families are **not interchangeable**, and the LOMO tests are exploratory.

## Baseline verification

Independent reproduction of the six-model reference from the supplied FINAL archive was exact to numerical precision: the largest absolute score difference was `4.0e-15` in the mixed subset and `3.3e-15` in the real-only subset. Thresholds likewise agreed within `3.6e-15`, and all low/high pressure classifications and positive-class recall 2×2 tables matched exactly.

| Subset | Baseline high pressure |
|---|---:|
| Mixed | 530/2,156 |
| Real-only | 282/1,128 |

## Rank and pressure-region stability

Across six exclusions relative to the full six-model baseline:

| Subset | Spearman correlation of GHPI ranks (min–max) | Pressure-region agreement (min–max) | Flipped sentences (min–max) |
|---|---|---|---|
| Mixed | 0.976–0.978 | 92.58%–94.20% | 125–160 |
| Real-only | 0.903–0.956 | 91.40%–94.41% | 63–97 |

The largest real-only perturbation arose after excluding RoBERTa (Spearman 0.903; 97/1,128 pressure assignments changed). The GHPI is **model-set dependent**, yet rankings are comparatively stable under each single-model exclusion; classification of individual borderline instances is not invariant.

## Positive-class recall

- In *every* LOMO condition in both subsets, the positive-class recall in the high-pressure region was below the positive-class recall in the low-pressure region for **all six evaluated models** (including the excluded model). This is directional consistency, not blanket significance.
- **Mixed, 5 remaining calibration models:** 2/5 to 5/5 comparisons remained significant using a within-condition BH correction over those five. **Mixed, all six evaluated models:** 2/6 to 6/6 remained BH-significant depending on the exclusion.
- **Real-only, 5 remaining calibration models:** 3/5 to 5/5 comparisons remained significant with BH over five. **Real-only, all six evaluated models:** 0/6 to 6/6 remained BH-significant, with zero observed under exclusion of T5. This demonstrates sensitivity of inferential strength to the model composition and multiplicity family.
- For the **excluded model tested against GHPI calibrated on the other five models**, all twelve recall differences were negative. After BH correction across the six such excluded-model checks within each subset, 3/6 were significant in mixed and 5/6 in real-only. These are exploratory in-corpus checks, not estimates of transfer to unseen architectures.

## Interpretation

The central *directional* pattern (lower positive-class recall under higher GHPI pressure) survives all six model exclusions in both subsets. However, the GHPI score, high/low assignment and statistical strength are not invariant to the evaluated model set. Therefore, the evidence supports a **within-corpus LOMO sensitivity result**, not model-set independence or external generalizability.

## Files in this repository

- [`scripts/lomo/LOMO_repeated20x5.py`](../../scripts/lomo/LOMO_repeated20x5.py): command-line implementation; default paths use the checked-in HPI source data and OOF reference outputs.
- [`notebooks/lomo/16_lomo_repeated20x5.ipynb`](../../notebooks/lomo/16_lomo_repeated20x5.ipynb): Jupyter implementation using the same algorithm and repository-relative inputs.
- [`outputs/lomo/results/`](../../outputs/lomo/results): original supplied CSV/TXT result files, preserved as an archival copy.
- [`docs/lomo/VALIDATION_REPORT.md`](VALIDATION_REPORT.md): comparison to the archived baseline, source repository, and independent rerun.

## Command (from repository root)

```bash
pip install -r requirements-analysis.txt
python scripts/lomo/LOMO_repeated20x5.py
python scripts/lomo/verify_lomo_outputs.py
```

By default, newly generated results are saved to `outputs/lomo/regenerated/`. The original supplied CSV/TXT files under `outputs/lomo/results/` are not overwritten. All inputs are taken from `data/processed/` and `outputs/oof/results/` already present in the original repository.
