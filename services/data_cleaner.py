"""
services/data_cleaner.py
Module 4: Clean Data Repository

Applies fixes for the 4 issue types found in Module 3:
    - Duplicates -> keep first occurrence, drop the rest
    - Nulls      -> numeric columns filled with median, text columns filled with mode (or 'Unknown')
    - Unknowns   -> placeholder strings (n/a, na, ?, etc.) converted to NaN, then handled like nulls
    - Outliers   -> capped at the IQR lower/upper boundary (not removed, to preserve row count)
"""

import pandas as pd

UNKNOWN_VALUES = ['unknown', 'n/a', 'na', 'none', '?', '-', 'undefined', 'null', '']


def clean_dataset(df: pd.DataFrame) -> dict:
    """
    Runs the full cleaning pipeline and returns both the cleaned
    DataFrame and a summary of what was fixed.
    """
    original_count = len(df)
    summary = {}

    # ---- Step 1: Remove duplicates ----
    before = len(df)
    df = df.drop_duplicates(keep='first').reset_index(drop=True)
    summary['duplicates_removed'] = before - len(df)

    # ---- Step 2: Convert unknown-style strings to real NaN ----
    unknowns_fixed = 0
    for col in df.select_dtypes(include='object').columns:
        col_lower = df[col].astype(str).str.strip().str.lower()
        mask = col_lower.isin(UNKNOWN_VALUES)
        unknowns_fixed += int(mask.sum())
        df.loc[mask, col] = None
    summary['unknowns_fixed'] = unknowns_fixed

    # ---- Step 3: Handle nulls (includes the unknowns we just converted) ----
    nulls_fixed = int(df.isnull().sum().sum())
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            mode_series = df[col].mode()
            fill_val = mode_series[0] if not mode_series.empty else 'Unknown'
            df[col] = df[col].fillna(fill_val)
    summary['nulls_fixed'] = nulls_fixed

    # ---- Step 4: Cap outliers (numeric columns only, using IQR) ----
    outliers_capped = 0
    for col in df.select_dtypes(include='number').columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        below = df[col] < lower_bound
        above = df[col] > upper_bound
        outliers_capped += int(below.sum() + above.sum())

        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    summary['outliers_capped'] = outliers_capped

    summary['original_record_count'] = original_count
    summary['clean_record_count'] = len(df)

    return {
        'dataframe': df,
        'summary': summary
    }