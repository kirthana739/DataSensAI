"""
services/pii_detector.py
Bonus Feature: PII (Personally Identifiable Information) Detection & Masking

Scans text columns for patterns that look like sensitive data:
    - Email addresses
    - Phone numbers (10-digit Indian mobile format)
    - PAN card numbers (Indian tax ID: 5 letters + 4 digits + 1 letter)
    - Aadhaar numbers (12-digit Indian national ID)
    - Credit/debit card numbers (13-16 digits)

This matters especially for banking/CRM data, where accidentally
exposing customer PII in reports or downloads is a real compliance risk.
"""

import re
import pandas as pd

PATTERNS = {
    'Email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
    'Phone Number': re.compile(r'^[6-9]\d{9}$'),
    'PAN Card': re.compile(r'^[A-Z]{5}\d{4}[A-Z]$'),
    'Aadhaar Number': re.compile(r'^\d{4}\s?\d{4}\s?\d{4}$'),
    'Card Number': re.compile(r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$'),
}


def _clean_str(value) -> str:
    """Converts a cell to a clean string for PII pattern matching.

    Pandas silently infers purely-numeric columns (phone numbers, Aadhaar
    numbers, card numbers with no spaces/dashes) as int64/float64 rather
    than text. A plain str(value) on those turns e.g. 9876543210 into
    '9876543210.0' (float, when the column also has a missing value) which
    breaks exact-match regexes. This normalizes both cases back to the
    clean digit string a human would have typed.
    """
    if pd.isna(value):
        return ''
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def detect_pii(df: pd.DataFrame) -> dict:
    """
    Scans every column (text AND numeric) against all PII patterns.
    Numeric columns are included because pandas silently infers phone
    numbers, Aadhaar numbers, and card numbers as int64/float64 whenever
    they're stored without spaces or dashes -- the most common real-world
    case -- so restricting the scan to text columns would miss them entirely.
    Returns: {column_name: {pii_type: match_count}}
    """
    findings = {}

    for col in df.columns:
        col_values = df[col].apply(_clean_str)
        col_findings = {}

        for pii_type, pattern in PATTERNS.items():
            match_count = col_values.apply(lambda v: bool(pattern.match(v))).sum()
            if match_count > 0:
                col_findings[pii_type] = int(match_count)

        if col_findings:
            findings[col] = col_findings

    return findings


def _mask_value(value: str, pii_type: str) -> str:
    """Masks a single value based on its detected PII type."""
    value = str(value).strip()

    if pii_type == 'Email':
        parts = value.split('@')
        if len(parts) == 2:
            name, domain = parts
            visible = name[:2] if len(name) > 2 else name[:1]
            return f"{visible}{'*' * max(len(name) - len(visible), 2)}@{domain}"
        return '****@****'

    if pii_type == 'Phone Number':
        return f"{'*' * 6}{value[-4:]}" if len(value) >= 4 else '**********'

    if pii_type == 'PAN Card':
        return f"{value[:2]}{'*' * 6}{value[-1:]}" if len(value) >= 3 else '**********'

    if pii_type == 'Aadhaar Number':
        digits = value.replace(' ', '')
        return f"{'*' * 8}{digits[-4:]}" if len(digits) >= 4 else '************'

    if pii_type == 'Card Number':
        digits = value.replace(' ', '').replace('-', '')
        return f"**** **** **** {digits[-4:]}" if len(digits) >= 4 else '**** **** **** ****'

    return '****'


def mask_dataframe(df: pd.DataFrame, findings: dict) -> pd.DataFrame:
    """Returns a copy of the dataframe with all detected PII columns masked."""
    masked = df.copy()

    for col, pii_types in findings.items():
        # Use the PII type with the most matches in this column as the masking rule
        primary_type = max(pii_types, key=pii_types.get)
        pattern = PATTERNS[primary_type]

        def mask_cell(v):
            v_str = _clean_str(v)
            if pattern.match(v_str):
                return _mask_value(v_str, primary_type)
            return v

        masked[col] = masked[col].apply(mask_cell)

    return masked


def build_pii_report(df: pd.DataFrame) -> dict:
    """Returns everything the PII scan template needs."""
    findings = detect_pii(df)
    total_pii_cells = sum(sum(types.values()) for types in findings.values())

    masked_df = mask_dataframe(df, findings) if findings else df
    preview_html = masked_df.head(20).to_html(classes='pii-table', index=False)

    return {
        'findings': findings,
        'total_pii_columns': len(findings),
        'total_pii_cells': total_pii_cells,
        'preview_html': preview_html,
        'masked_df': masked_df
    }