# HPI / GHPI — Out-of-Fold (OOF) Validation Addendum

This addendum contains additional cross-fitted validation code, complete supplied result files, and figures for the HPI citation-sentiment reliability study. The original training notebooks, data, documentation, and previous experimental results are unchanged.

## Structure

- `notebooks/oof/13_hpi_cross_fitted_validation_final.ipynb` — final 5-fold OOF analysis and repeated 20 × 5-fold GHPI robustness.
- `notebooks/oof/14_oof_positive_recall_dumbbell.ipynb` — plotting notebook for real-only positive recall (values embedded in notebook, retained from supplied code).
- `notebooks/oof/15_oof_pressure_assignment_stability.ipynb` — pressure-assignment stability visualizations from repeated OOF scores.
- `scripts/oof/HPI_cross_fitted_validation.py` — stand-alone original cross-fitted analysis pipeline with repository-relative input/output defaults.
- `outputs/oof/results/` — 60 supplied CSV/TXT results files, including mixed/real-only OOF and repeated 20×5 outputs.
- `outputs/oof/figures/` — four supplied original PDF figures.
- `docs/oof/HPI_CrossFitted_Validation_SUMMARY.md` — author's methodological summary and findings (preliminary document).
- `docs/oof/ADDITION_MANIFEST.csv` — paths, sizes, and SHA-256 digests for files in this addendum.

## Data and method

The cross-fitted pipeline expects `data/processed/Final_Master_results_with_features.csv`, which already exists in the base repository. To run the standalone script from the repository root:

```bash
python scripts/oof/HPI_cross_fitted_validation.py
```

Install relevant dependencies from `requirements-analysis.txt`. The plotting notebooks additionally use Matplotlib and Seaborn.

The **original supplied results** are committed as evidence in `outputs/oof/results/`. By default, regenerated files are written to `outputs/oof/regenerated/` (not to the supplied result folder). Plot notebooks default to `outputs/oof/regenerated_figures/`, preserving the original PDFs.

The scripts/notebooks have been adjusted **only for file paths and notebook-output clearing**; dataset CSV/TXT content and supplied figures are not modified. The original computational rules, model lists, folds, seeds, and statistical tests are retained. Running the full final notebook provides the repeated 20×5 analysis; the stand-alone `.py` script implements the original cross-fitted core and shorter repeated 5-fold sensitivity, not the appended final 20×5 stage.

## Analytical choices noted by supplied materials

- Five sentence-level folds and seed 42 for the primary cross-fitting.
- Common folds across six model architectures.
- 2,156 mixed and 1,128 real-only single-citation contexts in the original outputs.
- Repeated 20×5 robustness uses seeds 42–61 and aggregates observation-level OOF scores by median.
- The compatibility definition of `Global_Gap` is **the median of six model-specific gap distances**, not the arithmetic mean. The supplied summary notes that this might differ from wording in the manuscript; resolve that discrepancy in the paper, not by silently changing the code.
- The author's summary states that five models are BH-significant for real-only positive recall in the seed-42 OOF analysis, with T5 directional but not significant. Treat the supplied summaries and p-values as research outputs rather than independently revalidated claims; no experiments were rerun during this packaging.

## Original-file integrity and upload safety

`ADDITION_MANIFEST.csv` records every new file's SHA-256 digest. The upload script distributed with the addendum only stages paths under `notebooks/oof/`, `scripts/oof/`, `outputs/oof/`, or `docs/oof/` and checks that all staged changes are *additions*; existing paths cause an abort, never an overwrite. No original repository file is modified or deleted.
