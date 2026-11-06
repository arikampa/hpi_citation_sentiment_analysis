#!/usr/bin/env python3
"""Check reproducibility of all LOMO CSVs against supplied original outputs.

Run after scripts/lomo/LOMO_repeated20x5.py from any working directory.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def verify(reference_dir: Path, regenerated_dir: Path, atol: float=1e-10) -> None:
    originals = sorted(reference_dir.glob('*.csv'))
    if not originals:
        raise FileNotFoundError(f'No reference CSVs in {reference_dir}')
    for original in originals:
        rerun = regenerated_dir / original.name
        if not rerun.is_file():
            raise FileNotFoundError(f'Missing regenerated CSV: {rerun}')
        a = pd.read_csv(original, low_memory=False)
        b = pd.read_csv(rerun, low_memory=False)
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            raise AssertionError(f'{original.name}: shape/column mismatch')
        for column in a.columns:
            x = a[column]
            y = b[column]
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                xv = x.to_numpy(dtype=float)
                yv = y.to_numpy(dtype=float)
                if not np.allclose(xv, yv, atol=atol, rtol=0, equal_nan=True):
                    raise AssertionError(f'{original.name}: numeric mismatch in {column}')
            else:
                if not x.fillna('<MISSING>').equals(y.fillna('<MISSING>')):
                    raise AssertionError(f'{original.name}: mismatch in {column}')
        print(f'PASS {original.name}: {len(a)} rows')
    print(f'All {len(originals)} LOMO CSV files match the archived results (atol={atol}).')


if __name__=='__main__':
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, default=root/'outputs/lomo/results')
    p.add_argument('--regenerated', type=Path, default=root/'outputs/lomo/regenerated')
    args = p.parse_args()
    verify(args.reference, args.regenerated)
