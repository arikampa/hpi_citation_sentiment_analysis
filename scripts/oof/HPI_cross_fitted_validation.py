"""
Cross-fitted validation pipeline for the HPI/GHPI analyses.

Purpose
-------
1. Preserve all previously generated results unchanged.
2. Remove in-sample outcome circularity by estimating HPI/GHPI coefficients
   only on training folds and scoring held-out sentences out-of-fold (OOF).
3. Re-estimate the HPI/GHPI on real citations only, so synthetic citations do
   not contribute to coefficient estimation in the real-only sensitivity analysis.
4. Test an extended GHPI that also includes Sentence_Length and is_narrative.

Important compatibility choice
------------------------------
The legacy notebook computes Global_Gap as the MEDIAN of the six model-specific
Gap Distances. This script preserves that exact implementation by default so the
new validation isolates the circularity issue without redefining prior calculations.
Set GLOBAL_GAP_AGG = "mean" only for an additional sensitivity analysis.

The script writes only to a NEW output directory and never overwrites legacy files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests

# ============================================================
# CONFIGURATION
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = str(REPO_ROOT / "data" / "processed")
BASE_DIR = Path(os.environ.get("HPI_BASE_DIR", DEFAULT_BASE_DIR))
INPUT_CSV = BASE_DIR / "Final_Master_results_with_features.csv"
OUTPUT_DIR = Path(os.environ.get(
    "HPI_OUTPUT_DIR",
    str(REPO_ROOT / "outputs" / "oof" / "regenerated")
))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42
GLOBAL_GAP_AGG = "median"  # preserves the legacy notebook implementation

MODELS = ["BART", "T5", "SciBERT", "RoBERTa", "DeepSeek", "LLaMA3"]
PRED_COLS = {m: f"{m}_pred" for m in MODELS}
DIST_MAP = {
    "BART": "BART_shap_pos_abs_dist",
    "T5": "T5_shap_pos_abs_dist",
    "SciBERT": "SCIBERT_shap_pos_abs_dist",
    "RoBERTa": "ROBERTA_shap_pos_abs_dist",
    "DeepSeek": "DEEPSEEK_shap_pos_abs_dist",
    "LLaMA3": "LLAMA3_shap_pos_abs_dist",
}
DIST_COLS = list(DIST_MAP.values())
ERROR_COLS = [f"{m}_is_error" for m in MODELS]

IEEE_PATTERN = re.compile(r"\[\s*\d+\s*(?:[,\-–;]\s*\d+\s*)*\]")

# ============================================================
# HELPERS
# ============================================================
def add_common_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce legacy error columns and Negation_Norm exactly."""
    df = df.copy()
    for m in MODELS:
        df[f"{m}_is_error"] = (df[PRED_COLS[m]] != df["ground_truth"]).astype(int)

    # Legacy notebook: P95 is computed on the full 5,661-row test set.
    neg_p95 = df["Negation_Count"].quantile(0.95)
    if pd.isna(neg_p95) or neg_p95 == 0:
        df["Negation_Norm"] = 0.0
    else:
        df["Negation_Norm"] = (df["Negation_Count"] / neg_p95).clip(0, 1)
    return df


def add_real_synthetic_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the existing IEEE-regex rule used to remove synthetic citations."""
    df = df.copy()
    sentences = df["sentence"].fillna("").astype(str)
    df["is_synthetic_ieee"] = sentences.str.contains(IEEE_PATTERN, regex=True)
    return df


def make_single_citation_subset(df: pd.DataFrame, real_only: bool = False,
                                seed: int = RANDOM_STATE) -> pd.DataFrame:
    d = df[(df["is_citation"] == 1) & (df["Multi_Citation_Flag"] == 0)].copy()
    d = add_real_synthetic_flag(d)
    if real_only:
        d = d[~d["is_synthetic_ieee"]].copy()
    d = d.reset_index().rename(columns={"index": "original_index"})

    # One common sentence-level fold assignment across all six models.
    # Stratification uses only the ground-truth sentiment class.
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    d["fold"] = -1
    for fold, (_, test_idx) in enumerate(skf.split(d, d["ground_truth"])):
        d.loc[test_idx, "fold"] = fold
    return d


def global_gap(df: pd.DataFrame, agg: str = GLOBAL_GAP_AGG) -> pd.Series:
    if agg == "median":
        return df[DIST_COLS].median(axis=1, skipna=True)
    if agg == "mean":
        return df[DIST_COLS].mean(axis=1, skipna=True)
    raise ValueError("GLOBAL_GAP_AGG must be 'median' or 'mean'.")


def crossfit_model_hpi(d: pd.DataFrame, extended: bool = False):
    """
    OOF model-calibrated HPI.

    The logistic specification exactly preserves the legacy regression:
    error ~ Gap * Semantic_Depth + Negation_Norm + is_narrative + Sentence_Length

    The basic HPI score continues to use only Gap, Semantic_Depth, Negation_Norm.
    The extended sensitivity score additionally includes is_narrative and Sentence_Length.
    """
    out = d.copy()
    coef_rows = []

    for m in MODELS:
        gap = DIST_MAP[m]
        score = np.full(len(out), np.nan)
        train_p75 = np.full(len(out), np.nan)

        for fold in sorted(out["fold"].unique()):
            tr = out[out["fold"] != fold].copy()
            te = out[out["fold"] == fold].copy()

            formula = (
                f"{m}_is_error ~ {gap} * Semantic_Depth + "
                "Negation_Norm + is_narrative + Sentence_Length"
            )
            res = smf.logit(formula, data=tr).fit(disp=0)
            p = res.params

            def calc(dd):
                s = (
                    p[gap] * dd[gap]
                    + p["Semantic_Depth"] * dd["Semantic_Depth"]
                    + p["Negation_Norm"] * dd["Negation_Norm"]
                )
                if extended:
                    s = (
                        s
                        + p["is_narrative"] * dd["is_narrative"]
                        + p["Sentence_Length"] * dd["Sentence_Length"]
                    )
                return s

            tr_score = calc(tr)
            te_score = calc(te)
            score[te.index] = te_score
            train_p75[te.index] = tr_score.quantile(0.75)

            coef_rows.append({
                "model": m,
                "fold": int(fold),
                "extended": bool(extended),
                "n_train": len(tr),
                "Intercept_B": p["Intercept"],
                "Gap_B": p[gap],
                "Depth_B": p["Semantic_Depth"],
                "Interaction_B": p[f"{gap}:Semantic_Depth"],
                "Negation_B": p["Negation_Norm"],
                "Narrative_B": p["is_narrative"],
                "Length_B": p["Sentence_Length"],
                "train_P75_score": tr_score.quantile(0.75),
            })

        suffix = "_EXT" if extended else ""
        out[f"{m}_HPI_OOF{suffix}"] = score
        out[f"{m}_HPI_OOF_trainP75{suffix}"] = train_p75

    return out, pd.DataFrame(coef_rows)


def crossfit_ghpi(d: pd.DataFrame, extended: bool = False,
                  agg: str = GLOBAL_GAP_AGG):
    """
    OOF Global HPI at SENTENCE level.

    All rows for a held-out sentence are excluded from the pooled training regression.
    This removes sentence-level leakage from the six repeated model-error rows.

    Two thresholds are learned inside each training fold:
    (a) common P75 of the training GHPI distribution (does not use error labels),
    (b) model-specific P75 of GHPI among training errors (uses error labels only in train).
    """
    out = d.copy()
    out["Global_Gap"] = global_gap(out, agg=agg)

    score = np.full(len(out), np.nan)
    common_train_p75 = np.full(len(out), np.nan)
    model_train_p75_error = {m: np.full(len(out), np.nan) for m in MODELS}
    coef_rows = []

    for fold in sorted(out["fold"].unique()):
        tr = out[out["fold"] != fold].copy()
        te = out[out["fold"] == fold].copy()

        id_vars = [
            "Global_Gap", "Semantic_Depth", "Negation_Norm",
            "Sentence_Length", "is_narrative"
        ]
        tr_long = tr.melt(
            id_vars=id_vars,
            value_vars=ERROR_COLS,
            var_name="model_error_col",
            value_name="is_error",
        )

        xcols = ["Global_Gap", "Semantic_Depth", "Negation_Norm"]
        if extended:
            xcols += ["Sentence_Length", "is_narrative"]

        X = sm.add_constant(tr_long[xcols])
        y = tr_long["is_error"].astype(int)
        res = sm.Logit(y, X).fit(disp=0)
        p = res.params

        def calc(dd):
            s = (
                p["Global_Gap"] * dd["Global_Gap"]
                + p["Semantic_Depth"] * dd["Semantic_Depth"]
                + p["Negation_Norm"] * dd["Negation_Norm"]
            )
            if extended:
                s = (
                    s
                    + p["Sentence_Length"] * dd["Sentence_Length"]
                    + p["is_narrative"] * dd["is_narrative"]
                )
            return s

        tr_score = calc(tr)
        te_score = calc(te)
        score[te.index] = te_score

        # Strict out-of-fold common threshold: learned from TRAIN features/scores only.
        t_common = tr_score.quantile(0.75)
        common_train_p75[te.index] = t_common

        # Strict out-of-fold model-specific error threshold: TRAIN errors only.
        for m in MODELS:
            err_scores = tr_score.loc[tr[f"{m}_is_error"] == 1]
            t_m = err_scores.quantile(0.75) if len(err_scores) else np.nan
            model_train_p75_error[m][te.index] = t_m

        coef_rows.append({
            "fold": int(fold),
            "extended": bool(extended),
            "global_gap_aggregation": agg,
            "n_train_sentences": len(tr),
            "n_train_long_rows": len(tr_long),
            "Intercept_B": p["const"],
            "Global_Gap_B": p["Global_Gap"],
            "Depth_B": p["Semantic_Depth"],
            "Negation_B": p["Negation_Norm"],
            "Length_B": p.get("Sentence_Length", np.nan),
            "Narrative_B": p.get("is_narrative", np.nan),
            "train_P75_common": t_common,
        })

    suffix = "_EXT" if extended else ""
    out[f"GHPI_OOF{suffix}"] = score
    out[f"GHPI_OOF_trainP75{suffix}"] = common_train_p75
    for m in MODELS:
        out[f"{m}_GHPI_trainP75error{suffix}"] = model_train_p75_error[m]

    return out, pd.DataFrame(coef_rows)


def bh_selected_p(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["selected_p"] = np.where(
        results["recommended_test"].eq("Fisher"),
        results["fisher_p"], results["chi2_p"]
    )
    reject, p_bh, _, _ = multipletests(results["selected_p"], method="fdr_bh")
    results["selected_p_bh"] = p_bh
    results["selected_significant_bh"] = reject
    return results


def class_recall_stats(df: pd.DataFrame, score_col: str, threshold_col: str,
                       class_label: int) -> pd.DataFrame:
    """Low-vs-high pressure recall tests using one common sentence-level pressure split."""
    sub = df[df["ground_truth"] == class_label].copy()
    low = sub[score_col] <= sub[threshold_col]
    rows = []

    for m in MODELS:
        pred_col = PRED_COLS[m]
        a = int(((sub[pred_col] == class_label) & low).sum())
        b = int(((sub[pred_col] != class_label) & low).sum())
        c = int(((sub[pred_col] == class_label) & (~low)).sum())
        d = int(((sub[pred_col] != class_label) & (~low)).sum())
        table = np.array([[a, b], [c, d]])

        chi2, chi2_p, _, expected = chi2_contingency(table, correction=False)
        fisher_or, fisher_p = fisher_exact(table, alternative="two-sided")
        min_expected = float(expected.min())

        low_recall = a / (a + b)
        high_recall = c / (c + d)

        aa, bb, cc, dd = (a, b, c, d)
        if 0 in (a, b, c, d):
            aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        corrected_or = (aa * dd) / (bb * cc)

        rows.append({
            "model": m,
            "class_label": class_label,
            "low_correct": a,
            "low_wrong": b,
            "high_correct": c,
            "high_wrong": d,
            "low_total": a + b,
            "high_total": c + d,
            "low_recall": low_recall,
            "high_recall": high_recall,
            "recall_difference_high_minus_low": high_recall - low_recall,
            "chi2": chi2,
            "chi2_p": chi2_p,
            "min_expected_count": min_expected,
            "fisher_odds_ratio": fisher_or,
            "fisher_p": fisher_p,
            "odds_ratio_correct_low_vs_high": corrected_or,
            "recommended_test": "Fisher" if min_expected < 5 else "Chi-square",
        })

    return bh_selected_p(pd.DataFrame(rows))


def model_hpi_case_summary(df: pd.DataFrame, extended: bool = False) -> pd.DataFrame:
    suffix = "_EXT" if extended else ""
    rows = []
    for m in MODELS:
        score_col = f"{m}_HPI_OOF{suffix}"
        err_col = f"{m}_is_error"
        rules = {
            "Case 1 (Unexpected Error)": (df[score_col] <= 0) & (df[err_col] == 1),
            "Case 2 (Protective)": (df[score_col] <= 0) & (df[err_col] == 0),
            "Case 3 (Pressure-induced Failure)": (df[score_col] > 0) & (df[err_col] == 1),
            "Case 4 (Robust under Pressure)": (df[score_col] > 0) & (df[err_col] == 0),
        }
        for case, mask in rules.items():
            vals = df.loc[mask, score_col]
            rows.append({
                "model": m, "case": case, "count": len(vals),
                "mean_hpi": vals.mean() if len(vals) else np.nan,
                "p75_hpi": vals.quantile(0.75) if len(vals) else np.nan,
                "p95_hpi": vals.quantile(0.95) if len(vals) else np.nan,
                "max_hpi": vals.max() if len(vals) else np.nan,
            })
    return pd.DataFrame(rows)


def ghpi_common_case_long(df: pd.DataFrame, extended: bool = False) -> pd.DataFrame:
    suffix = "_EXT" if extended else ""
    score_col = f"GHPI_OOF{suffix}"
    threshold_col = f"GHPI_OOF_trainP75{suffix}"
    rows = []

    for m in MODELS:
        err_col = f"{m}_is_error"
        low = df[score_col] <= df[threshold_col]
        is_error = df[err_col] == 1
        case = np.select(
            [low & is_error, low & ~is_error, ~low & is_error, ~low & ~is_error],
            ["Case 1", "Case 2", "Case 3", "Case 4"],
            default="Unknown"
        )
        temp = pd.DataFrame({
            "original_index": df["original_index"].values,
            "fold": df["fold"].values,
            "model": m,
            "is_error": df[err_col].values,
            "GHPI_OOF": df[score_col].values,
            "threshold_used": df[threshold_col].values,
            "case": case,
        })
        rows.append(temp)

    return pd.concat(rows, ignore_index=True)


def ghpi_model_specific_case_long(df: pd.DataFrame, extended: bool = False) -> pd.DataFrame:
    """OOF GHPI with model-specific P75(error) thresholds learned ONLY in training folds."""
    suffix = "_EXT" if extended else ""
    score_col = f"GHPI_OOF{suffix}"
    rows = []

    for m in MODELS:
        err_col = f"{m}_is_error"
        threshold_col = f"{m}_GHPI_trainP75error{suffix}"
        low = df[score_col] <= df[threshold_col]
        is_error = df[err_col] == 1
        case = np.select(
            [low & is_error, low & ~is_error, ~low & is_error, ~low & ~is_error],
            ["Case 1", "Case 2", "Case 3", "Case 4"],
            default="Unknown"
        )
        rows.append(pd.DataFrame({
            "original_index": df["original_index"].values,
            "fold": df["fold"].values,
            "model": m,
            "is_error": df[err_col].values,
            "GHPI_OOF": df[score_col].values,
            "threshold_used": df[threshold_col].values,
            "case": case,
        }))
    return pd.concat(rows, ignore_index=True)


def case_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    return (
        long_df.groupby(["model", "case"])["GHPI_OOF"]
        .agg(
            count="count", mean="mean", median="median",
            p75=lambda x: x.quantile(0.75),
            p95=lambda x: x.quantile(0.95), max="max"
        )
        .reset_index()
    )


def save_recall_suite(df: pd.DataFrame, prefix: str,
                      score_col: str, threshold_col: str):
    label_names = {0: "negative", 1: "neutral", 2: "positive"}
    all_tables = []
    for label, name in label_names.items():
        tab = class_recall_stats(df, score_col, threshold_col, label)
        tab.to_csv(OUTPUT_DIR / f"{prefix}_{name}_recall_low_vs_high.csv", index=False)
        all_tables.append(tab.assign(class_name=name))
    pd.concat(all_tables, ignore_index=True).to_csv(
        OUTPUT_DIR / f"{prefix}_all_class_recall_tests.csv", index=False
    )


def repeated_real_positive_sensitivity(df_all: pd.DataFrame,
                                       seeds=(42, 43, 44, 45, 46)) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        d = make_single_citation_subset(df_all, real_only=True, seed=seed)
        g, _ = crossfit_ghpi(d, extended=False, agg=GLOBAL_GAP_AGG)
        st = class_recall_stats(g, "GHPI_OOF", "GHPI_OOF_trainP75", 2)
        st["seed"] = seed
        rows.append(st)
    return pd.concat(rows, ignore_index=True)

# ============================================================
# RUN
# ============================================================
print(f"Input:  {INPUT_CSV}")
print(f"Output: {OUTPUT_DIR}")
print(f"Global_Gap aggregation: {GLOBAL_GAP_AGG}")

raw = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)
raw = add_common_features(raw)

mixed = make_single_citation_subset(raw, real_only=False, seed=RANDOM_STATE)
real = make_single_citation_subset(raw, real_only=True, seed=RANDOM_STATE)

print(f"Single-citation mixed subset: {len(mixed)}")
print(f"Single-citation real-only subset: {len(real)}")
assert len(mixed) == 2156, f"Expected 2,156 mixed single-citation rows; got {len(mixed)}"
assert len(real) == 1128, f"Expected 1,128 real-only rows; got {len(real)}"

# A. Model-calibrated HPI (basic OOF)
mixed_hpi, mixed_hpi_coef = crossfit_model_hpi(mixed, extended=False)
real_hpi, real_hpi_coef = crossfit_model_hpi(real, extended=False)
mixed_hpi.to_csv(OUTPUT_DIR / "mixed_model_HPI_OOF_dataset.csv", index=False)
real_hpi.to_csv(OUTPUT_DIR / "real_only_model_HPI_OOF_dataset.csv", index=False)
mixed_hpi_coef.to_csv(OUTPUT_DIR / "mixed_model_HPI_OOF_fold_coefficients.csv", index=False)
real_hpi_coef.to_csv(OUTPUT_DIR / "real_only_model_HPI_OOF_fold_coefficients.csv", index=False)
model_hpi_case_summary(mixed_hpi).to_csv(OUTPUT_DIR / "mixed_model_HPI_OOF_case_summary.csv", index=False)
model_hpi_case_summary(real_hpi).to_csv(OUTPUT_DIR / "real_only_model_HPI_OOF_case_summary.csv", index=False)

# B. GHPI (basic OOF)
mixed_ghpi, mixed_ghpi_coef = crossfit_ghpi(mixed, extended=False)
real_ghpi, real_ghpi_coef = crossfit_ghpi(real, extended=False)
mixed_ghpi.to_csv(OUTPUT_DIR / "mixed_GHPI_OOF_dataset.csv", index=False)
real_ghpi.to_csv(OUTPUT_DIR / "real_only_GHPI_OOF_dataset.csv", index=False)
mixed_ghpi_coef.to_csv(OUTPUT_DIR / "mixed_GHPI_OOF_fold_coefficients.csv", index=False)
real_ghpi_coef.to_csv(OUTPUT_DIR / "real_only_GHPI_OOF_fold_coefficients.csv", index=False)

# Strict OOF common threshold cases and class tests
for name, d in [("mixed", mixed_ghpi), ("real_only", real_ghpi)]:
    long_common = ghpi_common_case_long(d, extended=False)
    long_common.to_csv(OUTPUT_DIR / f"{name}_GHPI_OOF_common_threshold_cases.csv", index=False)
    case_summary(long_common).to_csv(OUTPUT_DIR / f"{name}_GHPI_OOF_common_threshold_case_summary.csv", index=False)
    save_recall_suite(d, f"{name}_GHPI_OOF", "GHPI_OOF", "GHPI_OOF_trainP75")

    # Model-specific P75(error) thresholds, learned in training folds only
    long_ms = ghpi_model_specific_case_long(d, extended=False)
    long_ms.to_csv(OUTPUT_DIR / f"{name}_GHPI_OOF_model_specific_threshold_cases.csv", index=False)
    case_summary(long_ms).to_csv(OUTPUT_DIR / f"{name}_GHPI_OOF_model_specific_threshold_case_summary.csv", index=False)

# C. Extended GHPI sensitivity (Length + Narrative included)
mixed_ext, mixed_ext_coef = crossfit_ghpi(mixed, extended=True)
real_ext, real_ext_coef = crossfit_ghpi(real, extended=True)
mixed_ext.to_csv(OUTPUT_DIR / "mixed_extended_GHPI_OOF_dataset.csv", index=False)
real_ext.to_csv(OUTPUT_DIR / "real_only_extended_GHPI_OOF_dataset.csv", index=False)
mixed_ext_coef.to_csv(OUTPUT_DIR / "mixed_extended_GHPI_OOF_fold_coefficients.csv", index=False)
real_ext_coef.to_csv(OUTPUT_DIR / "real_only_extended_GHPI_OOF_fold_coefficients.csv", index=False)
save_recall_suite(mixed_ext, "mixed_extended_GHPI_OOF", "GHPI_OOF_EXT", "GHPI_OOF_trainP75_EXT")
save_recall_suite(real_ext, "real_only_extended_GHPI_OOF", "GHPI_OOF_EXT", "GHPI_OOF_trainP75_EXT")

# D. Repeated-fold sensitivity for the key real-only positive-class finding
rep = repeated_real_positive_sensitivity(raw)
rep.to_csv(OUTPUT_DIR / "real_only_positive_repeated_5fold_sensitivity.csv", index=False)
rep_summary = (
    rep.groupby("model")
    .agg(
        mean_recall_difference=("recall_difference_high_minus_low", "mean"),
        min_recall_difference=("recall_difference_high_minus_low", "min"),
        max_recall_difference=("recall_difference_high_minus_low", "max"),
        significant_seeds_BH=("selected_significant_bh", "sum"),
        median_BH_p=("selected_p_bh", "median"),
    )
    .reset_index()
)
rep_summary.to_csv(OUTPUT_DIR / "real_only_positive_repeated_5fold_summary.csv", index=False)

# Run metadata / safeguards
with open(OUTPUT_DIR / "README.txt", "w", encoding="utf-8") as f:
    f.write(
        "Cross-fitted HPI/GHPI validation outputs.\n\n"
        "Legacy files are not modified.\n"
        f"N_SPLITS={N_SPLITS}, RANDOM_STATE={RANDOM_STATE}\n"
        f"Global_Gap aggregation={GLOBAL_GAP_AGG}\n"
        "Mixed single-citation N=2156; real-only N=1128.\n"
        "Real-only identification reproduces the existing IEEE-style regex removal rule.\n"
        "Common OOF threshold = training-fold P75 of GHPI scores (no held-out errors used).\n"
        "Model-specific OOF threshold = training-fold P75 of GHPI among errors for that model.\n"
        "The extended GHPI sensitivity additionally includes Sentence_Length and is_narrative.\n"
    )

print("\nDONE. No legacy outputs were overwritten.")
print(f"New files written to: {OUTPUT_DIR}")
