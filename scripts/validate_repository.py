#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, json, re, sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ERRORS=[]

required_notebooks=[f"{i:02d}_" for i in range(1,13)]
notebooks=sorted((ROOT/'notebooks').glob('*.ipynb'))
if len(notebooks)!=12:
    ERRORS.append(f"Expected 12 notebooks, found {len(notebooks)}")
for prefix in required_notebooks:
    if not any(p.name.startswith(prefix) for p in notebooks):
        ERRORS.append(f"Missing notebook prefix {prefix}")

required_files=[
    'data/external/sentences_dataset_45269.csv',
    'data/processed/Master_results.csv',
    'data/processed/Master_results_with_features.csv',
    'data/processed/Final_Master_results_with_features.csv',
    'data/processed/dataset_with_HPI_global_sentence_level_only_real_citations.csv',
    'data/DATA_MANIFEST.csv',
    'docs/Workflow.md','docs/External_Materials.md','docs/File_Selection_Report.md','docs/Validation_Report.md'
]
for rel in required_files:
    if not (ROOT/rel).is_file(): ERRORS.append(f"Missing required file: {rel}")

schemas={
'data/external/sentences_dataset_45269.csv': {'sentence','is_citation','polarity'},
'data/processed/Final_Master_results_with_features.csv': {
    'sentence','is_citation','Multi_Citation_Flag','ground_truth','Semantic_Depth','Negation_Count',
    'BART_shap_pos_abs_dist','T5_shap_pos_abs_dist','SCIBERT_shap_pos_abs_dist',
    'ROBERTA_shap_pos_abs_dist','DEEPSEEK_shap_pos_abs_dist','LLAMA3_shap_pos_abs_dist'
},
'data/processed/dataset_with_HPI_global_sentence_level_only_real_citations.csv': {'sentence','HPI_global','ground_truth'}
}
for rel,cols in schemas.items():
    p=ROOT/rel
    if p.is_file():
        got=set(pd.read_csv(p,nrows=2).columns)
        missing=cols-got
        if missing: ERRORS.append(f"{rel} missing columns: {sorted(missing)}")

# Manifest checksum verification
manifest=ROOT/'data/DATA_MANIFEST.csv'
if manifest.is_file():
    with manifest.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            p=ROOT/row['path']
            if not p.is_file():
                ERRORS.append(f"Manifest file missing: {row['path']}")
                continue
            sha=hashlib.sha256(p.read_bytes()).hexdigest()
            if sha != row['sha256']: ERRORS.append(f"Checksum mismatch: {row['path']}")

emoji_re=re.compile('[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]')
secret_re=re.compile(r'(hf_[A-Za-z0-9]{20,}|[A-Fa-f0-9]{40})')
absolute_markers=['/content/drive/MyDrive/','ΔΙΔΑΚΤΟΡΙΚΟ PhD','3o PAPER/']

for p in notebooks:
    try: nb=json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        ERRORS.append(f"Invalid notebook JSON {p.name}: {e}")
        continue
    for cell in nb.get('cells',[]):
        if not cell.get('id'):
            ERRORS.append(f"Missing cell id in {p.name}")
        src=''.join(cell.get('source',[]))
        if emoji_re.search(src): ERRORS.append(f"Emoji found in {p.name}")
        if secret_re.search(src): ERRORS.append(f"Possible credential found in {p.name}")
        for marker in absolute_markers:
            if marker in src: ERRORS.append(f"Forbidden path marker {marker!r} in {p.name}")
        if cell.get('cell_type')=='code':
            if cell.get('execution_count') is not None: ERRORS.append(f"Execution count not cleared in {p.name}")
            if cell.get('outputs'): ERRORS.append(f"Outputs not cleared in {p.name}")

for p in list(ROOT.glob('*.md'))+list((ROOT/'docs').glob('*.md')):
    text=p.read_text(encoding='utf-8')
    if emoji_re.search(text): ERRORS.append(f"Emoji found in {p.relative_to(ROOT)}")
    if secret_re.search(text): ERRORS.append(f"Possible credential found in {p.relative_to(ROOT)}")

if ERRORS:
    print('Repository validation FAILED')
    for e in sorted(set(ERRORS)): print('-',e)
    sys.exit(1)
print('Repository validation PASSED')
print(f'Notebooks: {len(notebooks)}')
print(f'CSV files: {len(list((ROOT/"data").rglob("*.csv")))}')
