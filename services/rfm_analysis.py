"""
services/rfm_analysis.py
Module 10 (Bonus): Customer Segmentation - RFM Analysis

RFM = Recency, Frequency, Monetary. A classic technique used in banking,
CRM, and retail to segment customers by value and engagement:
    - Recency:  how recently did they transact?
    - Frequency: how often do they transact?
    - Monetary: how much do they spend in total?

Works on any dataset that has a customer identifier + a numeric amount
column. A date column is optional - if present, full RFM is computed;
if not, segmentation falls back to Frequency + Monetary only.
"""

import pandas as pd


def detect_customer_column(df: pd.DataFrame):
    """Looks for a column that identifies individual customers."""
    keywords = ['customer', 'client', 'user', 'account_holder', 'name']
    for col in df.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in keywords):
            return col
    # Fallback: first object/text column
    text_cols = list(df.select_dtypes(include='object').columns)
    return text_cols[0] if text_cols else None


def detect_amount_column(df: pd.DataFrame):
    """Looks for a numeric column representing money/value."""
    keywords = ['amount', 'value', 'price', 'revenue', 'sales', 'salary', 'payment', 'total']
    numeric_cols = list(df.select_dtypes(include='number').columns)
    for col in numeric_cols:
        if any(k in col.lower() for k in keywords):
            return col
    return numeric_cols[0] if numeric_cols else None


def detect_date_column(df: pd.DataFrame):
    """Looks for a date column (optional - enables true Recency scoring)."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                pd.to_datetime(df[col], errors='raise')
                return col
            except Exception:
                continue
    return None


def _segment(row) -> str:
    """Rule-based segment assignment using High/Low splits (robust for small datasets)."""
    r_high = row.get('R_high', None)
    f_high = row['F_high']
    m_high = row['M_high']

    if r_high is None:
        # No recency data available - segment on Frequency + Monetary only
        if f_high and m_high:
            return 'Loyal High-Value'
        if m_high:
            return 'Big Spenders'
        if f_high:
            return 'Frequent, Low-Value'
        return 'Low Engagement'

    if r_high and f_high and m_high:
        return 'Champions'
    if f_high and m_high:
        return 'Loyal Customers'
    if r_high and not f_high and not m_high:
        return 'New / Recent'
    if not r_high and (f_high or m_high):
        return 'At Risk'
    if not r_high and not f_high and not m_high:
        return 'Lost'
    return 'Others'


def compute_rfm(df: pd.DataFrame, customer_col: str, amount_col: str, date_col: str = None) -> dict:
    """
    Computes RFM scores and segments per customer.
    Returns a dict with the full RFM table (as records) and segment summary counts.
    """
    working = df.copy()
    working = working.dropna(subset=[customer_col, amount_col])
    working[amount_col] = pd.to_numeric(working[amount_col], errors='coerce')
    working = working.dropna(subset=[amount_col])

    if working.empty:
        return {'error': 'No valid rows after removing missing customer/amount values.'}

    has_date = False
    if date_col and date_col in working.columns:
        try:
            working[date_col] = pd.to_datetime(working[date_col], errors='coerce')
            if working[date_col].notna().sum() > 0:
                has_date = True
        except Exception:
            has_date = False

    agg_dict = {amount_col: ['count', 'sum']}
    if has_date:
        agg_dict[date_col] = 'max'

    grouped = working.groupby(customer_col).agg(agg_dict)
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns]
    grouped = grouped.reset_index()

    freq_col = f'{amount_col}_count'
    monetary_col = f'{amount_col}_sum'
    grouped = grouped.rename(columns={freq_col: 'Frequency', monetary_col: 'Monetary'})

    # Median-based High/Low split - robust even with small customer counts
    freq_median = grouped['Frequency'].median()
    monetary_median = grouped['Monetary'].median()
    grouped['F_high'] = grouped['Frequency'] >= freq_median
    grouped['M_high'] = grouped['Monetary'] >= monetary_median

    if has_date:
        date_agg_col = f'{date_col}_max'
        most_recent_overall = grouped[date_agg_col].max()
        grouped['Recency_days'] = (most_recent_overall - grouped[date_agg_col]).dt.days
        recency_median = grouped['Recency_days'].median()
        grouped['R_high'] = grouped['Recency_days'] <= recency_median  # lower days = more recent = "high" recency score
    else:
        grouped['R_high'] = None
        grouped['Recency_days'] = None

    grouped['Segment'] = grouped.apply(_segment, axis=1)

    segment_counts = grouped['Segment'].value_counts().to_dict()

    display_cols = [customer_col, 'Frequency', 'Monetary', 'Segment']
    if has_date:
        display_cols.insert(1, 'Recency_days')

    result_table = grouped[display_cols].sort_values('Monetary', ascending=False)

    return {
        'has_date': has_date,
        'total_customers': len(grouped),
        'segment_counts': segment_counts,
        'table_html': result_table.head(100).to_html(classes='rfm-table', index=False, float_format='%.2f'),
        'full_table': result_table,  # for CSV export
        'top_segment': max(segment_counts, key=segment_counts.get) if segment_counts else None,
        'avg_monetary': round(float(grouped['Monetary'].mean()), 2),
        'champions_count': segment_counts.get('Champions', 0) + segment_counts.get('Loyal High-Value', 0),
        'at_risk_count': segment_counts.get('At Risk', 0) + segment_counts.get('Lost', 0)
    }