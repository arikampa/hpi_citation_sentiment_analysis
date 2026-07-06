# File Selection Report

This report documents the construction of the GitHub-ready Paper 3 repository.

## Notebook retention rule

Every file from the `notebooks/` directory of the supplied `hpi-citation-sentiment-analysis` ZIP was retained:

```text
01_train_encoder_transformer_models.ipynb
02_train_bart_classification_model.ipynb
03_train_t5_seq2seq_model.ipynb
04_train_llama3_model.ipynb
05_train_deepseek_model.ipynb
06_generate_master_results_t5_bart.ipynb
07_update_master_results_encoder_models.ipynb
08_update_master_results_decoder_cnn_models.ipynb
09_extract_static_language_features.ipynb
10_shap_analysis_positional_feature_fast.ipynb
11_hpi_logistic_regression_and_global_hpi.ipynb
12_loss_curves.ipynb
```

The only notebook replaced at the source level was:

```text
11_hpi_logistic_regression_and_global_hpi.ipynb
```

It was rebuilt from the separately supplied:

```text
General Logistic Regression + Specific Logistic Regression + HPI.ipynb
```

The rebuilt notebook retains that notebook's HPI, global-HPI, common-threshold, SHAP-token, class-recall, real-citation sensitivity, case-analysis, and visualization sections while using the repository directory structure.

## Included data

The repository includes:

- the complete original `sentences_dataset_45269.csv`,
- the master prediction table,
- the static-feature table,
- the SHAP-enriched final feature table,
- HPI/global-HPI intermediate datasets,
- the real-citation-only HPI subset required by notebook 11,
- SHAP sample indices,
- model loss-history CSV files,
- coefficient, threshold, class-comparison, SHAP-category, and case-summary result tables.

No `sample/` folder was created.

## Cleaning and portability actions

- Removed notebook execution outputs and reset execution counters.
- Removed emoji and pictographic characters from notebooks and Markdown documentation.
- Preserved CSV dataset cell values without emoji filtering or semantic modification.
- Removed private Google Drive paths.
- Replaced filesystem paths with repository-relative paths.
- Added a repository-root resolver to every notebook.
- Removed embedded or interactive credential assumptions.
- Retained Hugging Face and Weights & Biases credentials only as environment-variable placeholders.
- Added data checksums and schemas in `data/DATA_MANIFEST.csv`.
- Added `scripts/validate_repository.py` for automated structural checks.

## Excluded material from the broad Paper 3 working ZIP

The following categories were excluded because they are not required by the retained 12-notebook workflow or are unsuitable for a public code repository:

- manuscript and cover-letter DOCX files,
- reference-paper PDFs,
- duplicate and `OLD` CSV versions,
- duplicate figure exports,
- exploratory or superseded notebooks outside the old repository's `notebooks/` folder,
- raw Google Drive organization artifacts,
- model weights and tokenizer binaries,
- credentials and private tokens.

Examples of excluded exploratory notebooks include the baseline logistic-regression notebook, older SHAP notebook, Meta-SHAP notebook, and visualization variants. Their essential production workflow is already represented by the retained notebooks and the rebuilt notebook 11.

## Known external dependency

Full model retraining and SHAP recomputation require large pretrained and fine-tuned model artifacts that are not included. The included processed CSVs allow notebook 11 and the core paper analysis to be reproduced without those artifacts.

## Execution validation

The rebuilt notebook 11 was executed end to end. Its regenerated core CSV outputs matched the supplied Paper 3 result files in structure and values. Details are recorded in `docs/Validation_Report.md`.
