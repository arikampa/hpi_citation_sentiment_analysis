# LOMO addendum — repeated 20 × 5-fold GHPI sensitivity analysis

## Purpose

The original paper's GHPI uses six sentiment-classification architectures: BART, T5, SciBERT, RoBERTa, DeepSeek and LLaMA3. Leave-One-Model-Out (LOMO) sensitivity analysis assesses dependence on this architecture set. It is *not* leave-one-sentence-out cross-validation, independent external validation, or evidence of generalization to a novel architecture.

The supplied analysis evaluates the full six-model baseline and six conditions in which one architecture is omitted from *calibration*. An omitted architecture can still be assessed diagnostically using pressure regions derived entirely from the other five models. Both mixed and real-only single-citation subsets are analyzed separately.

## Already present in the base repository (do not duplicate)

- `data/processed/Final_Master_results_with_features.csv` — all required sentence-level predictions, model-specific SHAP-distance features and linguistic features.
- `outputs/oof/results/mixed_GHPI_REPEATED20x5_FINAL_dataset.csv`
- `outputs/oof/results/real_only_GHPI_REPEATED20x5_FINAL_dataset.csv`
- `outputs/oof/results/mixed_GHPI_REPEATED20x5_FINAL_positive_recall_low_vs_high.csv`
- `outputs/oof/results/real_only_GHPI_REPEATED20x5_FINAL_positive_recall_low_vs_high.csv`

The input bytes supplied with the LOMO experiment were checked against the repository's Git blob hashes for all five files; they matched exactly.

## Newly added files

- `scripts/lomo/LOMO_repeated20x5.py`: self-contained command-line source. It recomputes and asserts the original six-model baseline before writing LOMO outputs.
- `scripts/lomo/verify_lomo_outputs.py`: compares regenerated outputs to the original supplied CSVs, using exact non-numeric comparisons and tight numerical tolerance.
- `notebooks/lomo/16_lomo_repeated20x5.ipynb`: self-contained notebook; adjusted to use repository-relative inputs and a non-destructive output directory.
- `outputs/lomo/results/`: seven original CSV result files plus the accompanying `README.txt`.
- `docs/lomo/RESULTS_AND_METHOD.md`: experiment design, observed effect sizes and inferential interpretation.
- `docs/lomo/VALIDATION_REPORT.md`: integrity and rerun results.
- `docs/lomo/INSTALL_WINDOWS_POWERSHELL.md`: safe import commands for Windows Git/PowerShell.
- `docs/lomo/ADDITION_MANIFEST.csv`: SHA-256 hashes for the newly added files.

## Experimental design

1. Filter to single-citation observations; mixed has 2,156 and real-only has 1,128 sentences.
2. For every condition, construct `GD_global` as the **median** of the available model-specific SHAP gap distances. It is not the mean.
3. With the same sentence-stratified folds for all conditions (20 seeds: 42–61; five folds each), estimate a pooled logistic error model on training sentences using only the *retained* architectures' labels and errors.
4. Obtain GHPI from the coefficients on `GD_global`, `Semantic_Depth` and `Negation_Norm`; omit the intercept as in the original HPI computation. Define the pressure threshold from the training-score 75th percentile.
5. Compute held-out scores and thresholds, then aggregate each observation across repetitions with the median. Evaluate recall differences and pressure assignment stability.
6. Use two-sided Fisher exact testing when expected chi-square contingency counts are small, otherwise uncorrected Pearson chi-square. Report Benjamini–Hochberg corrections in *separately labeled* groups: six evaluated architectures, five retained architectures, and excluded-model diagnostics across six omissions.

**Interpretation limitation:** The same sentences and ground-truth annotations are used across all conditions. BH correction within a condition does not control error rates across the entire ensemble of LOMO analyses. Pressure labels and statistical significance change under model omission even when the direction of the recall effect remains consistent.

## Reproduction from the repository root

```bash
python -m pip install -r requirements-analysis.txt
python scripts/lomo/LOMO_repeated20x5.py
python scripts/lomo/verify_lomo_outputs.py
```

The first script creates `outputs/lomo/regenerated/`, leaving the committed results unchanged. The verifier compares every regenerated CSV to the committed source CSVs (ordering, column names, categorical entries and numeric values).

To use Jupyter instead:

```bash
jupyter lab
```

Open `notebooks/lomo/16_lomo_repeated20x5.ipynb` and run all cells. The notebook writes to `outputs/lomo/regenerated_notebook/`.

See [the full methodology and observed results](RESULTS_AND_METHOD.md) and [validation](VALIDATION_REPORT.md). Do not use `input_data/` or the original Google Drive absolute path — the required files are already in this repository.
