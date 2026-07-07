# HPI - Citation Sentiment Analysis (CSA)

**Authors:** Aristotelis Kampatzis, Antonis Sidiropoulos

**Affiliation:** Department of Information and Electronic Engineering, International Hellenic University (IHU), Greece

**Manuscript:** [preprint version](https://www.researchsquare.com/article/rs-9462261/v1)

This repository contains the cleaned and reorganized code and data package for the study on the reliability of large language model based citation sentiment analysis under structural and semantic pressure.

The package supports two reproducibility levels:

1. **Analysis reproduction:** run the HPI, statistical, SHAP-token, and visualization analyses using the included processed CSV files.
2. **Full pipeline reproduction:** retrain the models, regenerate predictions and features, compute SHAP positional distances, and then rerun the HPI analysis. Full retraining requires substantial GPU resources and externally downloaded pretrained model checkpoints.

## What is included

- Every notebook from the original `notebooks/` directory is retained.
- The complete source dataset is included as `data/external/sentences_dataset_45269.csv`.
- The processed CSV files required by the included notebooks are present in `data/processed/`.
- Notebook outputs and execution counts were cleared.
- Emoji characters, private absolute paths, and embedded credentials were removed from code and documentation.
- Training notebooks used prior to the HPI analysis workflow.
- Conclusion/summary notebooks that update Master_results.csv.
- Extraction of static language features.
- The optimized FAST SHAP notebook required to calculate the position-distance capability used by the HPI framework.
- HPI/logistic regression analysis code.
- Loss curve visualization notebook.
- Tables of statistical results obtained and used in the analysis of the Consumer Price Index (HPI).
- Documentation describing the workflow and required external files.

## What is intentionally excluded

The following files are intentionally excluded from the public GitHub package:

- Full trained model weights and tokenizers (`.pth`, `.pt`, `.bin`, `.safetensors`, `.h5`, etc.).
- Reference PDFs and manuscript drafts.
- Private Google Drive paths and local machine paths.
- API keys and access tokens.


## Repository structure

```text
.
├── notebooks/
│   ├── 01_train_encoder_transformer_models.ipynb
│   ├── 02_train_bart_classification_model.ipynb
│   ├── 03_train_t5_seq2seq_model.ipynb
│   ├── 04_train_llama3_model.ipynb
│   ├── 05_train_deepseek_model.ipynb
│   ├── 06_generate_master_results_t5_bart.ipynb
│   ├── 07_update_master_results_encoder_models.ipynb
│   ├── 08_update_master_results_decoder_cnn_models.ipynb
│   ├── 09_extract_static_language_features.ipynb
│   ├── 10_shap_analysis_positional_feature_fast.ipynb
│   ├── 11_hpi_logistic_regression_and_global_hpi.ipynb
│   └── 12_loss_curves.ipynb
├── data/
│   ├── external/
│   ├── processed/
│   │   └── logs/
│   └── DATA_MANIFEST.csv
├── outputs/
│   ├── results/
│   └── figures/
├── external_materials/
│   └── model_weights/
├── docs/
│   ├── Workflow.md
│   ├── External_Materials.md
│   ├── File_Selection_Report.md
│   └── Validation_Report.md
├── scripts/
│   └── validate_repository.py
├── requirements.txt
└── requirements-analysis.txt
```

## Important workflow note

The training notebooks come first because they produce the trained models/tokenizers used by the inference and SHAP notebooks. The actual model weights are not included in this repository and must be placed under `external_materials/model_weights/`.

Core order:

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

The HPI notebook 11, requires the SHAP positional-distance features. Therefore, the optimized FAST SHAP notebook must run before the HPI/logistic-regression notebook, unless `Final_Master_results_with_features.csv` already contains all required `*_shap_pos_abs_dist` columns.

## Setup

For HPI/statistical analysis only:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-analysis.txt
jupyter lab
```

For the complete training and SHAP workflow:

```bash
pip install -r requirements.txt
```

Open Jupyter from the repository root. The notebooks also resolve the correct root when opened from `notebooks/`.

## Recommended starting point

To reproduce the HPI and statistical analyses without retraining the models, run:

```text
notebooks/11_hpi_logistic_regression_and_global_hpi.ipynb
```

The required `Final_Master_results_with_features.csv` and real-citation-only sensitivity subset are included.

For the full workflow, follow [docs/Workflow.md](docs/Workflow.md).

## Data integrity

`data/DATA_MANIFEST.csv` records the row count, column count, SHA-256 checksum, and schema of every included dataset and log CSV. Run:

```bash
python scripts/validate_repository.py
```

before analysis to check the package structure, required columns, notebook validity, forbidden absolute paths, exposed token patterns, and emoji characters in code/documentation. The completed packaging checks are summarized in [docs/Validation_Report.md](docs/Validation_Report.md).

## Credentials and gated models

No credentials are included. Set gated-model credentials through environment variables or a local `.env` file that is not committed:

```bash
export HF_TOKEN="your_token"
export WANDB_API_KEY="your_key"  # optional
```

See [docs/External_Materials.md](docs/External_Materials.md).

## Dataset acknowledgement
This study uses citation contexts derived from the Citation Sentiment Corpus created by Awais Athar.
The corpus consists of 8,736 citation sentences manually annotated for citation sentiment and extracted from the ACL Anthology Network corpus. The original corpus and related information are publicly available at:
https://awaisathar.com/citation-sentiment-corpus/. 
The dataset is publicly available. Users should verify and comply with the license terms of the version they use. The version used in this project was accessed from a public source reported with an Apache-2.0 license.
If you use this repository or the derived experiments, please also cite the original dataset paper:

```text
@InProceedings{athar:2011:SS,
  author    = {Athar, Awais},
  title     = {Sentiment Analysis of Citations using Sentence Structure-Based Features},
  booktitle = {Proceedings of the ACL 2011 Student Session},
  month     = {June},
  year      = {2011},
  address   = {Portland, OR, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {81--87},
  url       = {http://www.aclweb.org/anthology/P11-3015}
}
```
This study also used the public Hugging Face dataset `Sp1786/multiclass-sentiment-analysis-dataset` as an auxiliary multiclass sentiment dataset.
Sp1786. (2023). `multiclass-sentiment-analysis-dataset` [Dataset]. Hugging Face. 
https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset
The dataset is distributed on Hugging Face with an Apache-2.0 license. Users should verify and comply with the license terms of the version they use.

## Third-party citation contexts
In addition to the public citation sentiment datasets acknowledged above, some experiments may involve citation-context sentences extracted from open access third-party scholarly articles across different journals and scientific domains.

## License
The MIT License applies to the repository code. Dataset records, pretrained models, model weights, and third-party materials retain their original terms and are not relicensed by this repository.
