# Validation Report

This report records the checks performed on the packaged repository.

## Structural validation

- Confirmed that the repository contains exactly 12 numbered notebooks (`01` through `12`).
- Confirmed that all required source and processed CSV inputs are present.
- Confirmed that no `sample/` directory exists.
- Confirmed that every notebook is valid notebook JSON and contains stable cell identifiers.
- Cleared notebook outputs and execution counters.
- Checked notebook code cells for Python/IPython syntax after transformation.
- Checked notebooks and Markdown documentation for emoji characters, private Google Drive paths, and token-like credential strings.
- Verified SHA-256 checksums for all CSV files listed in `data/DATA_MANIFEST.csv`.

## Notebook 11 execution test

`notebooks/11_hpi_logistic_regression_and_global_hpi.ipynb` was executed end to end from the repository root using the included CSV files.

The execution completed without an exception after correcting one dataframe-shadowing issue inherited from the supplied source notebook. The stacked-bar visualization now uses a dedicated dataframe instead of replacing the main HPI dataframe.

The regenerated versions of the following core outputs were compared with the corresponding supplied Paper 3 files:

```text
Step1_General_Coefficients.csv
Step2_Specific_Coefficients.csv
Step3_Final_Dataset_with_HI_Scores.csv
dataset_with_HPI_global_sentence_level.csv
longformat_with_HPI_global.csv
longformat_with_globalHPI_cases.csv
longformat_with_globalHPI_cases_COMMON_THRESHOLD.csv
globalHPI_thresholds_per_model.csv
globalHPI_case_summary.csv
globalHPI_case_summary_COMMON_THRESHOLD.csv
positive_low_vs_high_hpi_stats.csv
negative_low_vs_high_hpi_stats.csv
neutral_low_vs_high_hpi_stats.csv
```

For these files, row counts, column order, non-numeric values, and numeric values matched. Numeric comparisons used tight floating-point tolerances.

## Validation environment for notebook 11

```text
Python 3.13.5
pandas 2.2.3
NumPy 2.3.5
SciPy 1.17.0
statsmodels 0.14.6
scikit-learn 1.8.0
Matplotlib 3.10.8
Seaborn 0.13.2
nbclient 0.10.4
nbformat 5.10.4
```

These versions are recorded in `requirements-analysis.txt`.

## Scope limitation

The large-model training, decoder inference, and full SHAP recomputation stages were not executed during packaging because they require external fine-tuned model weights, gated-model access, substantial GPU memory, and long runtimes. Their notebooks were structurally validated, cleaned, and checked for syntax and path consistency.
