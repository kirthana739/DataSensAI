"""
services/excel_loader.py

Multi-sheet Excel workbooks often have more than just the raw data on
them - PivotTable summaries, changelogs, dashboard tabs, notes, etc.
Blindly reading the first sheet (the pandas/Excel default) can grab one
of those instead of the actual data, which is exactly what was
happening with the Superstore file: the first sheet was an 8-row
PivotTable summary, not the 9,995-row raw data sheet.

This picks the sheet that looks most like a real data table - the one
with the most non-empty rows x columns after dropping fully-blank rows
and columns (which trims out trailing empty cells Excel often leaves
behind, so those don't inflate a sheet's apparent size).
"""

import pandas as pd


def load_best_sheet(filepath: str) -> pd.DataFrame:
    """Reads every sheet in an .xlsx/.xls workbook and returns the one
    that looks most like a real, row-level data table. Falls back to
    the only sheet if there's just one."""
    xls = pd.ExcelFile(filepath)

    if len(xls.sheet_names) == 1:
        return pd.read_excel(xls, sheet_name=xls.sheet_names[0])

    best_df = None
    best_score = -1
    best_name = None

    for name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=name)
        except Exception:
            continue
        # drop fully-empty rows/columns before scoring, so trailing
        # blank cells (common in Excel exports) don't inflate the size
        trimmed = df.dropna(how='all').dropna(axis=1, how='all')
        score = trimmed.shape[0] * trimmed.shape[1]
        if score > best_score:
            best_score = score
            best_df = trimmed
            best_name = name

    print(f"[excel_loader] workbook sheets: {xls.sheet_names} -> "
          f"picked '{best_name}' ({best_df.shape[0]} rows x {best_df.shape[1]} cols)")

    return best_df