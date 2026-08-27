"""
services/feature_engineering.py
Module 5: Feature Engineering

Automatically creates new, more useful columns from whatever raw data
exists - date parts, growth rates, margins, purchase frequency, and
encoded categories. Works generically since uploaded datasets vary.
"""

import pandas as pd
from services.rfm_analysis import detect_customer_column, detect_amount_column, detect_date_column


def extract_date_features(df: pd.DataFrame, date_col: str) -> tuple:
    """Adds Year, Month, Quarter, DayOfWeek columns from a date column."""
    working = df.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors='coerce')

    added = []
    if working[date_col].notna().sum() > 0:
        working[f'{date_col}_year'] = working[date_col].dt.year
        working[f'{date_col}_month'] = working[date_col].dt.month
        working[f'{date_col}_quarter'] = working[date_col].dt.quarter
        working[f'{date_col}_day_of_week'] = working[date_col].dt.day_name()
        added = [f'{date_col}_year', f'{date_col}_month', f'{date_col}_quarter', f'{date_col}_day_of_week']

    return working, added


def calculate_revenue_growth(df: pd.DataFrame, date_col: str, amount_col: str) -> tuple:
    """Adds a month-over-month growth % column based on monthly totals."""
    working = df.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors='coerce')
    working = working.dropna(subset=[date_col])

    if working.empty:
        return df, []

    monthly = working.groupby(working[date_col].dt.to_period('M'))[amount_col].sum()
    growth = monthly.pct_change().fillna(0) * 100
    growth_map = growth.to_dict()

    working['revenue_growth_pct'] = working[date_col].dt.to_period('M').map(growth_map)
    working['revenue_growth_pct'] = working['revenue_growth_pct'].round(2)

    return working, ['revenue_growth_pct']


def calculate_profit_margin(df: pd.DataFrame) -> tuple:
    """If both a revenue-like and cost/profit-like column exist, adds a profit_margin_pct column."""
    numeric_cols = list(df.select_dtypes(include='number').columns)
    revenue_col = next((c for c in numeric_cols if any(k in c.lower() for k in ['revenue', 'sales', 'amount', 'price'])), None)
    cost_col = next((c for c in numeric_cols if any(k in c.lower() for k in ['cost', 'expense'])), None)
    profit_col = next((c for c in numeric_cols if 'profit' in c.lower()), None)

    working = df.copy()

    if profit_col and revenue_col and profit_col != revenue_col:
        working['profit_margin_pct'] = (working[profit_col] / working[revenue_col].replace(0, pd.NA) * 100).round(2)
        return working, ['profit_margin_pct']

    if revenue_col and cost_col:
        working['profit_margin_pct'] = ((working[revenue_col] - working[cost_col]) / working[revenue_col].replace(0, pd.NA) * 100).round(2)
        return working, ['profit_margin_pct']

    return df, []


def calculate_purchase_frequency(df: pd.DataFrame, customer_col: str) -> tuple:
    """Adds a column showing how many total transactions each customer has."""
    working = df.copy()
    freq_map = working[customer_col].value_counts().to_dict()
    working['customer_purchase_frequency'] = working[customer_col].map(freq_map)
    return working, ['customer_purchase_frequency']


def encode_categories(df: pd.DataFrame, max_columns: int = 2) -> tuple:
    """Label-encodes up to `max_columns` categorical columns into new _encoded columns."""
    working = df.copy()
    categorical_cols = list(df.select_dtypes(include='object').columns)[:max_columns]
    added = []

    for col in categorical_cols:
        codes, _ = pd.factorize(working[col])
        working[f'{col}_encoded'] = codes
        added.append(f'{col}_encoded')

    return working, added


def run_feature_engineering(df: pd.DataFrame) -> dict:
    """
    Auto-detects what's possible and applies all applicable feature
    engineering steps. Returns the enhanced dataframe plus a log of
    what was added and why.
    """
    working = df.copy()
    features_added = []

    customer_col = detect_customer_column(df)
    amount_col = detect_amount_column(df)
    date_col = detect_date_column(df)

    if date_col:
        working, added = extract_date_features(working, date_col)
        if added:
            features_added.append({
                'name': 'Date Parts (Year, Month, Quarter, Day of Week)',
                'columns': added,
                'reason': f"Extracted from '{date_col}' to enable time-based analysis and forecasting"
            })

        if amount_col:
            working, added = calculate_revenue_growth(working, date_col, amount_col)
            if added:
                features_added.append({
                    'name': 'Revenue Growth %',
                    'columns': added,
                    'reason': f"Month-over-month % change in '{amount_col}', useful for trend analysis"
                })

    working, added = calculate_profit_margin(working)
    if added:
        features_added.append({
            'name': 'Profit Margin %',
            'columns': added,
            'reason': "Computed from revenue/cost or revenue/profit columns detected in the data"
        })

    if customer_col:
        working, added = calculate_purchase_frequency(working, customer_col)
        if added:
            features_added.append({
                'name': 'Customer Purchase Frequency',
                'columns': added,
                'reason': f"Count of transactions per '{customer_col}', useful for loyalty/segmentation analysis"
            })

    working, added = encode_categories(working)
    if added:
        features_added.append({
            'name': 'Encoded Categories',
            'columns': added,
            'reason': "Numeric-encoded versions of text columns, required as input for machine learning models"
        })

    return {
        'dataframe': working,
        'features_added': features_added,
        'original_column_count': len(df.columns),
        'new_column_count': len(working.columns),
        'preview_html': working.head(15).to_html(classes='feature-table', index=False)
    }