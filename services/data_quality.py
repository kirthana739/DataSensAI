"""
services/data_quality.py
Module 3: Data Quality Analysis

Contains 4 detection functions:
    - detect_duplicates()
    - detect_nulls()
    - detect_unknown_values()
    - detect_outliers()  (IQR method)

Each function returns a dictionary summary so the Flask route
can pass everything straight into the template.
"""

import pandas as pd

# Common placeholder strings that mean "missing" but aren't real NaN
UNKNOWN_VALUES = ['unknown', 'n/a', 'na', 'none', '?', '-', 'undefined', 'null', '']


def detect_duplicates(df: pd.DataFrame) -> dict:
    """Finds fully duplicated rows."""
    duplicate_mask = df.duplicated(keep=False)  # keep=False marks ALL copies, not just repeats
    duplicate_rows = df[duplicate_mask]

    return {
        'count': int(df.duplicated().sum()),  # number of extra copies (excludes first occurrence)
        'rows': duplicate_rows.head(50)         # cap preview to 50 rows for display
    }


def detect_nulls(df: pd.DataFrame) -> dict:
    """Finds null/empty cells per column."""
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]  # only columns that actually have nulls

    return {
        'count': int(df.isnull().sum().sum()),
        'by_column': null_counts.to_dict()
    }


def detect_unknown_values(df: pd.DataFrame) -> dict:
    """Finds cells containing placeholder 'unknown' style text."""
    unknown_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.select_dtypes(include='object').columns:
        col_lower = df[col].astype(str).str.strip().str.lower()
        unknown_mask[col] = col_lower.isin(UNKNOWN_VALUES)

    total_unknown = int(unknown_mask.values.sum())
    by_column = unknown_mask.sum()
    by_column = by_column[by_column > 0].to_dict()

    return {
        'count': total_unknown,
        'by_column': by_column
    }


def detect_outliers(df: pd.DataFrame) -> dict:
    """
    Finds outliers in numeric columns using the IQR method:
        Q1 = 25th percentile
        Q3 = 75th percentile
        IQR = Q3 - Q1
        Outlier = value < Q1 - 1.5*IQR  OR  value > Q3 + 1.5*IQR
    """
    numeric_cols = df.select_dtypes(include='number').columns
    outlier_summary = {}
    total_outliers = 0

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        count = len(outliers)

        if count > 0:
            outlier_summary[col] = count
            total_outliers += count

    return {
        'count': total_outliers,
        'by_column': outlier_summary
    }


def run_full_analysis(df: pd.DataFrame) -> dict:
    """Runs all 4 checks and returns a combined summary for the analysis page."""
    duplicates = detect_duplicates(df)
    nulls = detect_nulls(df)
    unknowns = detect_unknown_values(df)
    outliers = detect_outliers(df)

    return {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'duplicates': duplicates,
        'nulls': nulls,
        'unknowns': unknowns,
        'outliers': outliers
    }


# --------------------------------------------------------------
# Module 5: Data Issues Repository
# Returns the ACTUAL flagged rows (not just counts) for each
# issue type, so they can be displayed in tabs and exported.
# --------------------------------------------------------------

def get_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Every row that is part of a duplicate set (all copies included)."""
    mask = df.duplicated(keep=False)
    return df[mask].copy()


def get_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Every row that has at least one null value, with a column
    listing which fields were null."""
    mask = df.isnull().any(axis=1)
    result = df[mask].copy()
    if not result.empty:
        result['_issue_columns'] = df[mask].apply(
            lambda row: ', '.join(row[row.isnull()].index), axis=1
        )
    return result


def get_unknown_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Every row that has at least one placeholder 'unknown' style value."""
    unknown_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.select_dtypes(include='object').columns:
        col_lower = df[col].astype(str).str.strip().str.lower()
        unknown_mask[col] = col_lower.isin(UNKNOWN_VALUES)

    row_mask = unknown_mask.any(axis=1)
    result = df[row_mask].copy()
    if not result.empty:
        result['_issue_columns'] = unknown_mask[row_mask].apply(
            lambda row: ', '.join(row[row].index), axis=1
        )
    return result


def get_outlier_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Every row that has at least one outlier value in any numeric column."""
    numeric_cols = df.select_dtypes(include='number').columns
    row_mask = pd.Series(False, index=df.index)
    issue_cols_per_row = pd.Series([[] for _ in range(len(df))], index=df.index)

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        col_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        row_mask = row_mask | col_mask

        for idx in df[col_mask].index:
            issue_cols_per_row[idx].append(col)

    result = df[row_mask].copy()
    if not result.empty:
        result['_issue_columns'] = issue_cols_per_row[row_mask].apply(lambda cols: ', '.join(cols))
    return result


def get_issue_report(df: pd.DataFrame) -> dict:
    """Bundles all 4 issue-row DataFrames together for the Issues Repository page."""
    return {
        'duplicates': get_duplicate_rows(df),
        'nulls': get_null_rows(df),
        'unknowns': get_unknown_rows(df),
        'outliers': get_outlier_rows(df),
    }


# --------------------------------------------------------------
# Data Quality Score - a single composite metric (0-100 + letter
# grade) summarizing overall dataset health at a glance.
# --------------------------------------------------------------

def compute_quality_score(results: dict) -> dict:
    """
    Computes a 0-100 quality score and letter grade from the output
    of run_full_analysis(). Each issue type contributes up to 25 points
    of penalty, scaled by how much of the dataset it affects.
    """
    total_records = results['total_records'] or 1
    total_columns = results['total_columns'] or 1
    total_cells = total_records * total_columns

    duplicate_rate = min(results['duplicates']['count'] / total_records, 1.0)
    null_rate = min(results['nulls']['count'] / total_cells, 1.0)
    unknown_rate = min(results['unknowns']['count'] / total_cells, 1.0)
    outlier_rate = min(results['outliers']['count'] / total_records, 1.0)

    penalty = (duplicate_rate * 25) + (null_rate * 25) + (unknown_rate * 25) + (outlier_rate * 25)
    score = max(0, round(100 - penalty))

    if score >= 90:
        grade = 'A'
    elif score >= 75:
        grade = 'B'
    elif score >= 60:
        grade = 'C'
    elif score >= 40:
        grade = 'D'
    else:
        grade = 'F'

    return {
        'score': score,
        'grade': grade,
        'breakdown': {
            'Duplicates': round(duplicate_rate * 25, 1),
            'Nulls': round(null_rate * 25, 1),
            'Unknowns': round(unknown_rate * 25, 1),
            'Outliers': round(outlier_rate * 25, 1),
        }
    }