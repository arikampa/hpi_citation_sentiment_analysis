#!/usr/bin/env python3
"""Leave-one-model-out repeated 20x5 OOF GHPI sensitivity experiment.

Faithful extension of the user-supplied HPI_cross_fitted_validation.py and
repeated_aggregated_ghpi notebook cell.  Baseline is independently reproduced,
validated against archived FINAL files, and each five-model subset is refitted.

Usage from repository root:
 python scripts/lomo/LOMO_repeated20x5.py

Override any paths with --input, --reference, and --out.

Dependencies: numpy, pandas, scipy, scikit-learn, statsmodels.
No sentiment models are retrained and no SHAP values are recomputed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr, chi2_contingency, fisher_exact
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import adjusted_rand_score
from statsmodels.stats.multitest import multipletests

MODELS = ["BART", "T5", "SciBERT", "RoBERTa", "DeepSeek", "LLaMA3"]
DIST = {"BART":"BART_shap_pos_abs_dist", "T5":"T5_shap_pos_abs_dist",
        "SciBERT":"SCIBERT_shap_pos_abs_dist", "RoBERTa":"ROBERTA_shap_pos_abs_dist",
        "DeepSeek":"DEEPSEEK_shap_pos_abs_dist", "LLaMA3":"LLAMA3_shap_pos_abs_dist"}
SEEDS = tuple(range(42,62)); N_FOLDS=5
IEEE_PATTERN = re.compile(r"\[\s*\d+\s*(?:[,\-–;]\s*\d+\s*)*\]")


def prepare_dataset(raw, real_only):
    # As in supplied legacy code, the 95th percentile is based on *all* test-set rows.
    d = raw.copy()
    q95 = d['Negation_Count'].quantile(.95)
    d['Negation_Norm'] = (d['Negation_Count']/q95).clip(0,1) if pd.notna(q95) and q95 else 0.
    d = d[(d.is_citation == 1) & (d.Multi_Citation_Flag == 0)].copy()
    d['is_synthetic_ieee'] = d['sentence'].fillna('').astype(str).str.contains(IEEE_PATTERN,regex=True)
    if real_only:
        d = d[~d.is_synthetic_ieee].copy()
    # Retain original 5,661-row positional indexes; reset to match archived files.
    d = d.reset_index().rename(columns={'index':'original_index'})
    return d


def fit_single_scenario(d, kept, fold_assignments):
    """Fit a five-/six-model pooled logit in each training fold and OOF score heldouts."""
    n = len(d)
    gap = d[[DIST[m] for m in kept]].median(axis=1, skipna=True).to_numpy(float)
    sd = d['Semantic_Depth'].to_numpy(float)
    neg = d['Negation_Norm'].to_numpy(float)
    # Original formula is Logit(error ~ 1 + Global_Gap + Semantic_Depth + Negation_Norm).
    X_base = np.column_stack([np.ones(n), gap, sd, neg])
    errors = np.column_stack([(d[f'{m}_pred'].to_numpy() != d.ground_truth.to_numpy()).astype(int) for m in kept])
    scores = np.empty((len(SEEDS), n),dtype=float)
    thresholds = np.empty((len(SEEDS),n),dtype=float)
    coefs=[]
    for ri, fids in enumerate(fold_assignments):
        for fold in range(N_FOLDS):
            train = fids != fold
            test = ~train
            X_tr = X_base[train]
            y_long = errors[train].T.reshape(-1)  # column-major stacking, as pandas.melt
            X_long = np.tile(X_tr, (len(kept),1))
            # Use statsmodels MLE to match the original experiment (not sklearn regularization).
            res = sm.Logit(y_long,X_long).fit(disp=0,maxiter=100)
            p = res.params
            tr_s = X_tr[:,1:]@p[1:]
            scores[ri,test] = X_base[test,1:]@p[1:]
            thresholds[ri,test] = np.quantile(tr_s,.75)
            coefs.append({'repeat_id':ri+1,'seed':SEEDS[ri],'fold':fold,
                          'n_train_sentences':int(train.sum()),'n_train_long_rows':int(len(y_long)),
                          'Intercept_B':p[0],'Global_Gap_B':p[1], 'Depth_B':p[2],
                          'Negation_B':p[3],'train_P75_common':np.quantile(tr_s,.75)})
    s=np.median(scores,axis=0)
    t=np.median(thresholds,axis=0)
    return {'gap':gap, 'score':s, 'threshold':t, 'high':s>t,
            'repeat_scores':scores,'repeat_thresholds':thresholds,
            'coefficients':pd.DataFrame(coefs)}


def positive_recall(d, high, subset, excluded):
    """Identical Fisher vs Pearson chi-squared rule and BH to archived notebook."""
    true = d.ground_truth.to_numpy() == 2
    rows=[]
    for m in MODELS:
        correct = d[f'{m}_pred'].to_numpy() == 2
        a=int(np.sum(true & ~high & correct))
        b=int(np.sum(true & ~high & ~correct))
        c=int(np.sum(true & high & correct))
        dd=int(np.sum(true & high & ~correct))
        ct=np.array([[a,b],[c,dd]])
        chi,chi_p,_,exp=chi2_contingency(ct,correction=False)
        _,f_p=fisher_exact(ct,alternative='two-sided')
        test = 'Fisher' if exp.min()<5 else 'Chi-square'
        p=f_p if test=='Fisher' else chi_p
        rows.append({'subset':subset,'excluded_model':excluded,'model':m,
                     'model_in_ghpi':excluded=='none' or m!=excluded,
                     'low_correct':a,'low_wrong':b,'high_correct':c,'high_wrong':dd,
                     'low_total':a+b,'high_total':c+dd,
                     'low_recall':a/(a+b),'high_recall':c/(c+dd),
                     'delta_high_minus_low':c/(c+dd)-a/(a+b),
                     'recommended_test':test,'min_expected':exp.min(),
                     'p_raw':p})
    df=pd.DataFrame(rows)
    df['p_bh_six_evaluated']=multipletests(df.p_raw,method='fdr_bh')[1]
    df['significant_bh_six_evaluated']=df.p_bh_six_evaluated<.05
    retained=df.model_in_ghpi
    df['p_bh_retained_five']=np.nan
    if excluded!='none':
        df.loc[retained,'p_bh_retained_five']=multipletests(df.loc[retained,'p_raw'],method='fdr_bh')[1]
    return df


def compare_baseline(base, archived, subset):
    f=archived/f'{subset}_GHPI_REPEATED20x5_FINAL_dataset.csv'
    assert f.is_file(),f'Cannot verify baseline without {f}'
    a=pd.read_csv(f)
    assert a.original_index.tolist()==base['original_index'].tolist(), 'Baseline sentence ordering differs!'
    dif_score=np.max(np.abs(base['score']-a.GHPI_REPEATED_MEDIAN.to_numpy(float)))
    dif_t=np.max(np.abs(base['threshold']-a.GHPI_REPEATED_THRESHOLD_MEDIAN.to_numpy(float)))
    region_agreement=np.mean(base['high']==a.pressure_region_repeated.eq('High').to_numpy())
    assert dif_score<1e-7 and dif_t<1e-7 and region_agreement==1., (dif_score,dif_t,region_agreement)
    return {'subset':subset,'max_abs_GHPI_difference':dif_score,'max_abs_P75_difference':dif_t,
            'pressure_agreement':region_agreement,'n_high':int(base['high'].sum())}


def run(input_csv, reference, out):
    out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(input_csv,encoding='utf-8-sig',low_memory=False)
    checks=[]; cases=[]; recalls=[]; coef=[]; per_row=[]; omitted=[]
    for subset,real_only in [('mixed',False),('real_only',True)]:
        d=prepare_dataset(raw,real_only)
        expected=1128 if real_only else 2156
        assert len(d)==expected,(subset,len(d))
        assert not d[[DIST[m] for m in MODELS]+['Semantic_Depth','Negation_Norm']].isna().any().any(), 'Missing features'
        fold_assignments=[]
        for seed in SEEDS:
            skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=seed)
            folds=np.empty(len(d),dtype=int)
            for j,(_,te) in enumerate(skf.split(d,d.ground_truth)):
                folds[te]=j
            fold_assignments.append(folds)
        scenario_results={}
        for excluded in ['none']+MODELS:
            kept=MODELS if excluded=='none' else [m for m in MODELS if m!=excluded]
            print(f'{subset}: exclude={excluded}; calibration models={kept}',flush=True)
            r=fit_single_scenario(d,kept,fold_assignments)
            scenario_results[excluded]=r
            cf=r['coefficients'].copy()
            cf.insert(0,'subset',subset)
            cf.insert(1,'excluded_model',excluded)
            coef.append(cf)
            recall=positive_recall(d,r['high'],subset,excluded)
            recalls.append(recall)
            if excluded=='none':
                bmeta=d[['original_index']].copy()
                bmeta['original_index']=bmeta['original_index'].astype(int)
                checks.append(compare_baseline({**r,**{'original_index':bmeta.original_index}},reference,subset))
                f=reference/f'{subset}_GHPI_REPEATED20x5_FINAL_positive_recall_low_vs_high.csv'
                ref=pd.read_csv(f)
                for m in MODELS:
                    ours=recall.loc[recall.model==m].iloc[0]
                    theirs=ref.loc[ref.model==m].iloc[0]
                    assert (ours.low_correct,ours.low_wrong,ours.high_correct,ours.high_wrong)==(theirs.low_correct,theirs.low_wrong,theirs.high_correct,theirs.high_wrong),f'Baseline recall does not match archived {subset}/{m}'
        baseline=scenario_results['none']
        for excluded,r in scenario_results.items():
            gd_rho=spearmanr(baseline['gap'],r['gap']).statistic
            ghpi_rho=spearmanr(baseline['score'],r['score']).statistic
            same=(baseline['high']==r['high'])
            high_intersection=np.sum(baseline['high'] & r['high'])
            high_union=np.sum(baseline['high'] | r['high'])
            cases.append({'subset':subset,'excluded_model':excluded,'n_models':6 if excluded=='none' else 5,
                          'n_citations':len(d),'n_high':int(r['high'].sum()),
                          'n_high_baseline':int(baseline['high'].sum()),
                          'gd_global_spearman':gd_rho,'ghpi_spearman':ghpi_rho,
                          'pressure_agreement':same.mean(), 'pressure_flip_n':int((~same).sum()),
                          'pressure_ARI':adjusted_rand_score(baseline['high'],r['high']),
                          'high_jaccard':high_intersection/high_union if high_union else np.nan,
                          'mean_score':np.mean(r['score']),'median_score':np.median(r['score']),
                          'median_threshold':np.median(r['threshold']),
                          'median_global_gap':np.median(r['gap'])})
            row=pd.DataFrame({'subset':subset,'original_index':d.original_index,
                             'ground_truth':d.ground_truth,'excluded_model':excluded,
                             'GD_global':r['gap'],'GHPI_repeated_median':r['score'],
                             'threshold_repeated_median':r['threshold'],'high_pressure':r['high']})
            per_row.append(row)
        print('baseline checked',checks[-1],flush=True)
    cases=pd.DataFrame(cases); recall=pd.concat(recalls,ignore_index=True); coefs=pd.concat(coef,ignore_index=True)
    full=pd.concat(per_row,ignore_index=True)
    all_omitted = recall[(recall.excluded_model!='none') & (recall.excluded_model==recall.model)].copy()
    all_omitted['p_bh_six_omissions']=all_omitted.groupby('subset').p_raw.transform(lambda x: multipletests(x,method='fdr_bh')[1])
    all_omitted['significant_bh_six_omissions']=all_omitted.p_bh_six_omissions<.05
    base_diff=recall[recall.excluded_model=='none'][['subset','model','delta_high_minus_low']].rename(columns={'delta_high_minus_low':'baseline_delta'})
    recall=recall.merge(base_diff,on=['subset','model'])
    recall['delta_change_vs_baseline']=recall.delta_high_minus_low-recall.baseline_delta
    recall.to_csv(out/'LOMO_positive_recall_all_models.csv',index=False)
    cases.to_csv(out/'LOMO_GHPI_stability_summary.csv',index=False)
    coefs.to_csv(out/'LOMO_fold_coefficients.csv',index=False)
    full.to_csv(out/'LOMO_per_citation_scores.csv',index=False)
    all_omitted.to_csv(out/'LOMO_heldout_model_evaluation.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'baseline_validation.csv',index=False)
    # compact paper-ready summaries
    top=recall[(recall.excluded_model!='none') & (recall.model_in_ghpi)].groupby(['subset','excluded_model']).agg(
        n_remaining=('model','count'),all_positive_deltas=('delta_high_minus_low',lambda x:bool((x<0).all())),
        n_significant_BH_five=('p_bh_retained_five',lambda x:int((x<.05).sum())),
        min_delta=('delta_high_minus_low','min'),max_delta=('delta_high_minus_low','max')).reset_index()
    summary=cases[cases.excluded_model!='none'].merge(top,on=['subset','excluded_model'])
    summary.to_csv(out/'LOMO_paper_summary.csv',index=False)
    with open(out/'README.txt','w') as f:
        f.write('LOMO repeated20x5 GHPI sensitivity analysis. Same stratified folds as archived baseline, seeds 42-61.\n'
                'Each model removal updates median model-specific GD_global and refits pooled Logit coefficients using only five remaining model error outcomes; training-derived P75 is refit in each fold.\n'
                'Held-out model errors are excluded from calibrating GHPI but are evaluated on same sentence corpus: not an external validation.\n'
                'GHPI coefficients use only GD_global, Semantic_Depth and Negation_Norm; intercept intentionally omitted from score, following source code.\n'
                'Repeated OOF score and threshold are each aggregated by observation-level median.\n'
                'Positive recall tests: two-sided Fisher if any Pearson expected count <5, otherwise uncorrected Pearson chi-square; BH across six evaluated models (baseline-compatible); separately BH across the five calibration models; and BH across six omitted-model tests (exploratory).\n'
                'All tests and model set sensitivity analyses are exploratory; BH per condition does not control FDR across all LOMO families.\n'
                'Original raw data, notebooks and results have not been overwritten.\n')
    print('\nBASELINE VERIFICATION:\n',pd.DataFrame(checks).to_string(index=False),flush=True)
    print('\nSTABILITY:\n',summary[['subset','excluded_model','ghpi_spearman','pressure_agreement','pressure_flip_n','all_positive_deltas','n_significant_BH_five']].to_string(index=False),flush=True)
    print('\nHELD OUT:\n',all_omitted[['subset','model','low_recall','high_recall','delta_high_minus_low','p_raw','p_bh_six_omissions']].to_string(index=False),flush=True)
    print('Done! Output:',out,flush=True)

if __name__=='__main__':
    repo_root = Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=repo_root/'data/processed/Final_Master_results_with_features.csv',
                        help='Source features CSV already committed to the HPI repository')
    parser.add_argument('--reference', type=Path, default=repo_root/'outputs/oof/results',
                        help='Existing repeated 20x5 OOF baseline reference CSV directory')
    parser.add_argument('--out', type=Path, default=repo_root/'outputs/lomo/regenerated',
                        help='Output directory (separate from original published results)')
    a=parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',category=FutureWarning)
        run(a.input,a.reference,a.out)
