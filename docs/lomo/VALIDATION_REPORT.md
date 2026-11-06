# LOMO addendum — integrity and reproduction check

Review date: **2026-11-06**. This report records checks performed on the supplied `HPI_LOMO_Repeated20x5_Results_and_Code` archive before GitHub packaging. These checks verify numerical reproduction of the supplied implementation; they are not independent external validation of the scientific conclusions.

## Input integrity

The source feature CSV and four mixed/real-only repeated-OOF reference CSVs included in the archive were verified as **byte-for-byte identical** to files already committed in `arikampa/hpi_citation_sentiment_analysis`, main branch at tree SHA `ade062c3ac72d47ddf0528247a245b13ace540fc`. Therefore the five input/reference copies are **not duplicated** in this addendum.

## End-to-end reproduction

- The original `LOMO_repeated20x5.py` was run from its archive with the supplied source CSV and four reference baseline CSVs.
- Execution completed normally for the mixed and real-only subsets and all seven calibration conditions per subset (full baseline plus six omissions).
- The recalculated six-model baseline reproduced the archived reference, with maximum GHPI differences of `3.997e-15` (mixed) and `3.331e-15` (real-only), maximum P75 differences of `3.553e-15` and `3.109e-15`, and **100% agreement** on high/low-pressure labels.
- Each of the seven original result CSVs matched the independently regenerated CSV in shape, column sequence, all nonnumeric entries, and all numeric values to absolute tolerance `1e-10`. The additional `notebook_rerun_results` archive contained duplicates of the same data within this numerical tolerance; they are not committed a second time.
- Original result table sizes: 14 GHPI stability rows, 12 LOMO paper summary rows, 84 positive-recall rows, 12 held-out-model rows, 1,400 fitted-fold coefficient rows, 22,988 per-citation condition rows, 2 baseline-validation rows.

## Packaging changes

1. Preserved all original supplied result CSVs.
2. Removed copies of the five already committed input/reference CSVs from this addendum.
3. Kept the original numerical methodology unchanged and added repository-relative **default** CLI paths to the script, with command-line overrides still supported.
4. Changed only the notebook path/mounting boilerplate: removed the old Google Drive mount and fixed absolute Colab path, while retaining the computational implementation. Notebook output directory now defaults to `outputs/lomo/regenerated_notebook/`.
5. Added an explicit CSV result verifier and documentation. Generated files are excluded by the local `outputs/lomo/.gitignore`; original LOMO outputs are explicitly unignored because the repository's root `.gitignore` has a generic `results/` pattern.
6. Executed the new package's script and the revised notebook end to end. Both matched all seven archived result CSVs at `atol=1e-10`.

## Dependencies used for reproduction

Python 3, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, statsmodels 0.14.6, nbformat 5.10.4.

## Caution

The LOMO procedure establishes sensitivity of GHPI to the composition of the *six tested architectures* using the same sentence corpus. It does not establish invariance across model collections, transfer performance on unseen models, or independent validation on a fresh dataset. Multiplicity-adjusted significance depends on the named family of tests.
