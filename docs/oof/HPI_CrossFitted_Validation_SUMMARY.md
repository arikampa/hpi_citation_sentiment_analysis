# HPI/GHPI Cross-Fitted Validation — Preliminary Results

## Design
- 5-fold sentence-level cross-fitting, seed 42.
- Common folds across all six architectures.
- HPI/GHPI coefficients for each held-out sentence are estimated without that sentence's error outcome.
- Real-only sensitivity re-estimates coefficients using only the 1,128 naturally occurring single-citation contexts.
- Common pressure threshold is estimated inside each training fold as P75 of training GHPI scores; held-out errors are not used.
- Extended GHPI sensitivity adds Sentence_Length and is_narrative.
- Global_Gap uses the **median** across six model-specific Gap Distances to preserve the implementation in the legacy notebook.

## Main positive-class result — mixed real + synthetic (N=2,156)
All six models retain lower positive recall in high-pressure regions and all six remain BH-significant in the seed-42 OOF analysis.

## Main positive-class result — real-only (N=1,128)
Positive recall is lower in the high-pressure region for all six models. BH-corrected significance is retained for BART, SciBERT, RoBERTa, DeepSeek, and LLaMA3; T5 remains directional but non-significant.

## Repeated 5-fold sensitivity (seeds 42–46)
The real-only positive recall difference remains negative for all six models across all tested seeds. BART, SciBERT, RoBERTa, DeepSeek, and LLaMA3 are BH-significant in 4/5 seeds; T5 is non-significant in all five seeds but retains the same direction.

## Extended GHPI sensitivity
When Sentence_Length and is_narrative are included in the cross-fitted GHPI, the real-only positive recall decrease remains BH-significant for all six architectures in the seed-42 analysis.

## Important implementation note
The manuscript currently describes GDglobal as an average, whereas the legacy code calculates Global_Gap using the median across model-specific distances. This discrepancy should be resolved explicitly before the next manuscript revision. The cross-fitted primary outputs preserve the legacy median implementation so that the validation does not silently redefine prior calculations.
