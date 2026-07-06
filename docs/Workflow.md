# Workflow

This document describes the execution order, inputs, outputs, and two supported reproduction modes.

## Mode A: Reproduce the HPI and statistical analysis

This is the practical route for reviewers and researchers who do not need to retrain every language model.

### A1. Validate the repository

```bash
python scripts/validate_repository.py
```

### A2. Install the analysis environment

```bash
pip install -r requirements-analysis.txt
```

### A3. Run notebook 11

```text
notebooks/11_hpi_logistic_regression_and_global_hpi.ipynb
```

Primary input:

```text
data/processed/Final_Master_results_with_features.csv
```

The table contains model predictions, correctness indicators, static language features, and the six model-specific SHAP positional-distance features.

Additional sensitivity-analysis input:

```text
data/processed/dataset_with_HPI_global_sentence_level_only_real_citations.csv
```

Primary generated intermediate datasets:

```text
data/processed/Step3_Final_Dataset_with_HI_Scores.csv
data/processed/dataset_with_HPI_global_sentence_level.csv
data/processed/longformat_with_HPI_global.csv
data/processed/longformat_with_globalHPI_cases.csv
data/processed/longformat_with_globalHPI_cases_COMMON_THRESHOLD.csv
```

Primary result tables:

```text
outputs/results/Step1_General_Coefficients.csv
outputs/results/Step2_Specific_Coefficients.csv
outputs/results/globalHPI_thresholds_per_model.csv
outputs/results/globalHPI_case_summary.csv
outputs/results/globalHPI_case_summary_COMMON_THRESHOLD.csv
outputs/results/positive_low_vs_high_hpi_stats.csv
outputs/results/negative_low_vs_high_hpi_stats.csv
outputs/results/neutral_low_vs_high_hpi_stats.csv
outputs/results/*_only_real_citations.csv
outputs/results/global_common_threshold_shap_analysis/
outputs/results/global_common_threshold_shap_stats/
```

Generated figures are written under `outputs/figures/`.

### A4. Recreate loss-curve figures, when required

Run:

```text
notebooks/12_loss_curves.ipynb
```

It uses the included CSV files in `data/processed/logs/`.

## Mode B: Reproduce the complete model-to-HPI pipeline

This route requires pretrained model downloads, substantial disk space, gated-model access for some models, and a suitable GPU environment.

### 1. Train encoder transformer models

```text
notebooks/01_train_encoder_transformer_models.ipynb
```

Input:

```text
data/external/sentences_dataset_45269.csv
```

Outputs include trained model/tokenizer folders under `external_materials/model_weights/`, loss logs, and classification reports.

### 2. Train BART

```text
notebooks/02_train_bart_classification_model.ipynb
```

### 3. Train T5

```text
notebooks/03_train_t5_seq2seq_model.ipynb
```

### 4. Train LLaMA 3

```text
notebooks/04_train_llama3_model.ipynb
```

A Hugging Face token with access to the configured gated model may be required.

### 5. Train DeepSeek

```text
notebooks/05_train_deepseek_model.ipynb
```

### 6. Generate the initial prediction table

```text
notebooks/06_generate_master_results_t5_bart.ipynb
```

Output:

```text
data/processed/Master_results.csv
```

### 7. Add encoder-model predictions

```text
notebooks/07_update_master_results_encoder_models.ipynb
```

This updates `data/processed/Master_results.csv`.

### 8. Add decoder-only and CNN predictions

```text
notebooks/08_update_master_results_decoder_cnn_models.ipynb
```

This updates the same master prediction table.

### 9. Extract static linguistic features

```text
notebooks/09_extract_static_language_features.ipynb
```

Outputs:

```text
data/processed/Master_results_with_features.csv
data/processed/Final_Master_results_with_features.csv
```

### 10. Compute SHAP positional-distance features

```text
notebooks/10_shap_analysis_positional_feature_fast.ipynb
```

Run the configured analysis for every HPI model so that the following columns are present:

```text
BART_shap_pos_abs_dist
T5_shap_pos_abs_dist
SCIBERT_shap_pos_abs_dist
ROBERTA_shap_pos_abs_dist
DEEPSEEK_shap_pos_abs_dist
LLAMA3_shap_pos_abs_dist
```

This step updates `data/processed/Final_Master_results_with_features.csv`.

### 11. Construct HPI and run statistical analyses

```text
notebooks/11_hpi_logistic_regression_and_global_hpi.ipynb
```

Notebook 11 was rebuilt specifically from the supplied `General Logistic Regression + Specific Logistic Regression + HPI.ipynb`. It replaces the earlier notebook 11 while retaining the original analysis logic and using repository-relative paths.

### 12. Generate loss-curve figures

```text
notebooks/12_loss_curves.ipynb
```

## Reproducibility controls

- The notebooks use the original random seed settings where they were defined.
- The source and processed CSV files are included and checksummed in `data/DATA_MANIFEST.csv`.
- Notebook outputs are cleared to distinguish source code from regenerated results.
- Paths are relative to the repository root.
- Tokens and API keys are read from environment variables.
- The exact pretrained model snapshots are not vendored; record model revisions locally when performing a new full retraining run.

## Important limitation

The included processed CSV files permit direct reproduction of the HPI and statistical stages. Bit-for-bit reproduction of neural-model training can still vary across GPU hardware, CUDA/cuDNN versions, library versions, and remote model revisions. Researchers should archive their exact environment and model revision identifiers when rerunning the complete pipeline.

## Packaging validation

The rebuilt notebook 11 was executed end to end using the included CSV files. See `docs/Validation_Report.md` for the tested environment, output comparisons, and scope limitations.
