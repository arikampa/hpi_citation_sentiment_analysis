# External Materials

All CSV datasets required by the included notebooks are present in this repository. There is no `sample/` directory and the source dataset is not treated as an external download.

## Materials not included

The following items remain external because of size, access restrictions, or third-party licensing:

- pretrained base-model downloads from Hugging Face,
- fine-tuned model weights and tokenizers,
- gated-model access credentials,
- optional Weights & Biases account credentials,
- CUDA drivers and GPU-specific runtime components,
- third-party scholarly PDFs and manuscript drafts.

## Expected model-weight location

Fine-tuned artifacts are read from or written to:

```text
external_materials/model_weights/
```

Depending on the selected model configuration, expected subfolders include encoder models, BART, T5, LLaMA 3, DeepSeek, GPT-family models, Gemma, Falcon, and CNN tokenizers/models. The exact folder names are defined in notebooks 01-10.

Typical examples:

```text
external_materials/model_weights/BERT/saved_model/
external_materials/model_weights/BERT/saved_tokenizer/
external_materials/model_weights/SciBERT/saved_model/
external_materials/model_weights/SciBERT/saved_tokenizer/
external_materials/model_weights/facebook_bart-large/saved_model/
external_materials/model_weights/facebook_bart-large/saved_tokenizer/
external_materials/model_weights/t5-large/saved_model/
external_materials/model_weights/t5-large/saved_tokenizer/
external_materials/model_weights/LLaMA_3/saved_model/
external_materials/model_weights/LLaMA_3/saved_tokenizer/
external_materials/model_weights/DeepSeek/saved_model/
external_materials/model_weights/DeepSeek/saved_tokenizer/
```

Folder naming must match the corresponding notebook configuration.

## Downloaded pretrained models

The training notebooks download base models through the Transformers library. For stronger future reproducibility, record and archive the exact Hugging Face revision or commit hash used for each model. The original notebooks identify model repositories but do not consistently pin immutable revisions.

## Authentication

No token is embedded in the repository.

Use environment variables:

```bash
export HF_TOKEN="<token>"
export WANDB_API_KEY="<key>"
```

On Windows PowerShell:

```powershell
$env:HF_TOKEN="<token>"
$env:WANDB_API_KEY="<key>"
```

Weights & Biases is optional. The cleaned notebooks do not perform an interactive login when `WANDB_API_KEY` is absent.

## Dataset licensing

The included CSV files contain derived research data assembled from multiple sources. Their inclusion does not override the licenses or rights of the original data providers. Users are responsible for reviewing and complying with the terms of the Citation Sentiment Corpus, auxiliary sentiment datasets, and any third-party citation-context sources.

## Recommended archival practice

For a publication release, archive the following separately in a DOI-bearing repository when licensing permits:

- exact fine-tuned weights and tokenizers,
- `pip freeze` or Conda environment export,
- CUDA and GPU metadata,
- immutable base-model revision identifiers,
- regenerated figure files,
- a checksum manifest for external artifacts.
