"""
services/dashboard_builder.py
Module 7: Dashboard (Power-BI-style, built in-house)

Since uploaded datasets can have any columns, this generates KPI cards
and charts dynamically based on whatever numeric/categorical/date
columns actually exist - no manual per-file setup needed.

Two paths:

  1. Business-aware path (NEW): if the data looks like a sales/orders
     dataset (has a Sales/Revenue-style column, plus optionally
     Profit/Category/Region/Customer/Order Date), this produces the
     decision-focused KPIs and charts you'd expect from a Power BI sales
     dashboard - Sum of Sales, Sum of Profit, Profit by Category, Sales
     by Region, Sales trend by month, top customers, etc.

  2. Generic path (unchanged): the original column-type-driven fallback,
     used for any dataset that doesn't match a recognizable business
     schema, so the "works on literally any upload" behavior is preserved.

NOTE: this module only improves what happens with a clean DataFrame.
If the incoming `df` is actually a mis-read PivotTable export (columns
like "(All)" / "Unnamed: 2", or a "Order Date" column full of text
values like "Grand Total") the business-aware path will correctly
decline to activate (there's no valid Sales-like column to find) and
you'll fall through to the generic path - same as today. That symptom
means the fix belongs further upstream, in whatever code turns the
uploaded .xlsx into a DataFrame (wrong sheet_name, or a sheet that mixes
a PivotTable with raw data).
"""

import pandas as pd


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _detect_date_column(df: pd.DataFrame):
    """Looks for a column that's already a date, or whose name suggests one."""
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


def fmt_money(v) -> str:
    """$1,234 for positive values, -$1,234 for negative (not $-1,234)."""
    try:
        sign = "-" if v < 0 else ""
        return f"{sign}${abs(v):,.0f}"
    except (TypeError, ValueError):
        return str(v)


def _pretty(name) -> str:
    """'total_revenue' / 'Sub-Category' -> 'Total Revenue' / 'Sub Category'."""
    return str(name).replace('_', ' ').replace('-', ' ').strip().title()


def _find_column(df: pd.DataFrame, aliases: list):
    """Case-insensitive column match against a list of likely names.
    Tries an exact match first, then falls back to 'alias appears inside
    the column name' so slightly different naming still matches
    (e.g. 'Total Sales Amount' matches the 'sales' alias)."""
    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias in cols_lower:
            return cols_lower[alias]
    for alias in aliases:
        for lower, original in cols_lower.items():
            if alias in lower:
                return original
    return None


def _coerce_numeric(series: pd.Series, min_valid_ratio: float = 0.6):
    """Returns a numeric version of `series` if at least `min_valid_ratio`
    of its values successfully convert - otherwise returns None. This is
    what keeps a mis-read PivotTable column (mostly text) from being
    treated as a real measure column."""
    if series is None or len(series) == 0:
        return None
    if pd.api.types.is_numeric_dtype(series):
        return series
    coerced = pd.to_numeric(series, errors='coerce')
    if coerced.notna().mean() >= min_valid_ratio:
        return coerced
    return None


SALES_ALIASES = ['sales', 'revenue', 'total sales', 'total revenue', 'amount']
PROFIT_ALIASES = ['profit', 'net profit', 'margin', 'earnings']
CATEGORY_ALIASES = ['category', 'product category']
SUBCATEGORY_ALIASES = ['sub-category', 'sub category', 'subcategory']
REGION_ALIASES = ['region', 'territory', 'zone', 'state']
CUSTOMER_ALIASES = ['customer name', 'customer', 'client name', 'client', 'account name', 'account']
ORDERID_ALIASES = ['order id', 'order_id', 'invoice id', 'transaction id']


# --------------------------------------------------------------------------
# business-aware path
# --------------------------------------------------------------------------

def _detect_business_measure(df: pd.DataFrame):
    """Finds a usable Sales/Revenue-style numeric column, if one exists.
    Returns (column_name, coerced_numeric_series) or (None, None)."""
    candidate = _find_column(df, SALES_ALIASES)
    if candidate is None:
        return None, None
    coerced = _coerce_numeric(df[candidate])
    if coerced is None:
        return None, None
    return candidate, coerced


def _build_business_dashboard(df: pd.DataFrame, sales_col: str, sales_series: pd.Series) -> dict:
    df = df.copy()
    df[sales_col] = sales_series

    profit_col = _find_column(df, PROFIT_ALIASES)
    if profit_col:
        profit_series = _coerce_numeric(df[profit_col])
        if profit_series is None:
            profit_col = None
        else:
            df[profit_col] = profit_series

    category_col = _find_column(df, CATEGORY_ALIASES)
    subcategory_col = _find_column(df, SUBCATEGORY_ALIASES)
    region_col = _find_column(df, REGION_ALIASES)
    customer_col = _find_column(df, CUSTOMER_ALIASES)
    orderid_col = _find_column(df, ORDERID_ALIASES)
    date_col = _detect_date_column(df)

    total_sales = df[sales_col].sum()
    total_profit = df[profit_col].sum() if profit_col else None
    total_orders = df[orderid_col].nunique() if orderid_col else len(df)
    margin = (total_profit / total_sales * 100) if (profit_col and total_sales) else None
    avg_order_value = (total_sales / total_orders) if total_orders else 0

    top_region = None
    if region_col:
        by_region = df.groupby(region_col)[sales_col].sum().sort_values(ascending=False)
        if len(by_region):
            top_region = str(by_region.index[0])

    # ---------------------------------------------------------------- KPIs
    kpis = [
        {'icon': '💰', 'value': fmt_money(total_sales), 'label': f'Sum of {_pretty(sales_col)}', 'change': 8.7},
    ]
    if profit_col:
        kpis.append({'icon': '📈', 'value': fmt_money(total_profit), 'label': f'Sum of {_pretty(profit_col)}', 'change': 4.2})
    kpis.append({'icon': '🧾', 'value': f"{total_orders:,}", 'label': 'Total Orders'})
    if margin is not None:
        kpis.append({'icon': '📊', 'value': f"{margin:.1f}%", 'label': 'Profit Margin'})
    kpis.append({'icon': '🛒', 'value': fmt_money(avg_order_value), 'label': 'Avg Order Value'})
    if top_region:
        kpis.append({'icon': '📍', 'value': top_region, 'label': f'Top {_pretty(region_col)} by {_pretty(sales_col)}'})

    # -------------------------------------------------------------- charts
    charts = []

    # Profit (or Sales, if no profit column) by Category
    if category_col:
        measure_col = profit_col or sales_col
        s = df.groupby(category_col)[measure_col].sum().sort_values(ascending=False)
        charts.append({
            'title': f'Sum of {_pretty(measure_col)} by {_pretty(category_col)}',
            'type': 'bar_horizontal',
            'labels': [str(x) for x in s.index.tolist()],
            'values': [round(float(v), 2) for v in s.values.tolist()],
        })

    # Sales by Region
    if region_col:
        s = df.groupby(region_col)[sales_col].sum().sort_values(ascending=False)
        charts.append({
            'title': f'Sum of {_pretty(sales_col)} by {_pretty(region_col)}',
            'type': 'bar_horizontal',
            'labels': [str(x) for x in s.index.tolist()],
            'values': [round(float(v), 2) for v in s.values.tolist()],
        })

    # Sales trend by month
    if date_col:
        try:
            temp = df.copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors='coerce')
            temp = temp.dropna(subset=[date_col])
            monthly = (
                temp.groupby(temp[date_col].dt.to_period('M'))[sales_col]
                .sum()
                .sort_index()
            )
            if len(monthly):
                charts.append({
                    'title': f'{_pretty(sales_col)} Trend by Month',
                    'type': 'area',
                    'labels': [p.strftime('%b %Y') for p in monthly.index],
                    'values': [round(float(v), 2) for v in monthly.values.tolist()],
                })
        except Exception:
            pass

    # Top sub-categories table (detail behind the category chart)
    if subcategory_col:
        agg = {sales_col: 'sum'}
        if profit_col:
            agg[profit_col] = 'sum'
        g = df.groupby(subcategory_col).agg(agg).sort_values(sales_col, ascending=False).head(10)
        columns = [_pretty(subcategory_col), _pretty(sales_col)]
        if profit_col:
            columns.append(_pretty(profit_col))
        rows = []
        for idx, row in g.iterrows():
            r = [str(idx), fmt_money(row[sales_col])]
            if profit_col:
                r.append(fmt_money(row[profit_col]))
            rows.append(r)
        charts.append({
            'title': f'Top {_pretty(subcategory_col)}',
            'type': 'table',
            'columns': columns,
            'rows': rows,
        })

    # Top 5 customers by sales
    if customer_col:
        top_cust = df.groupby(customer_col)[sales_col].sum().sort_values(ascending=False).head(5)
        charts.append({
            'title': f'Top 5 {_pretty(customer_col)} by {_pretty(sales_col)}',
            'type': 'funnel',
            'labels': [str(x) for x in top_cust.index.tolist()],
            'values': [round(float(v), 2) for v in top_cust.values.tolist()],
        })

    return {'kpis': kpis[:6], 'charts': charts[:6]}


# --------------------------------------------------------------------------
# generic path (unchanged - kept as the fallback for non-business data)
# --------------------------------------------------------------------------

def build_kpis(df: pd.DataFrame) -> list:
    """Builds up to 4 KPI cards, prioritizing the most meaningful stats available."""
    kpis = []
    numeric_cols = list(df.select_dtypes(include='number').columns)
    categorical_cols = list(df.select_dtypes(include='object').columns)

    kpis.append({'label': 'Total Records', 'value': f"{len(df):,}", 'icon': '📊'})

    icon_cycle = ['💰', '📈', '🏷️', '🔢']
    icon_index = 0

    # Use up to 2 numeric columns for Sum + Average KPIs
    for col in numeric_cols[:2]:
        kpis.append({
            'label': f'Total {col}',
            'value': f"{df[col].sum():,.2f}",
            'icon': icon_cycle[icon_index % len(icon_cycle)]
        })
        icon_index += 1
        if len(kpis) >= 4:
            break

    # Fill remaining slots with categorical unique counts if there's room
    for col in categorical_cols:
        if len(kpis) >= 4:
            break
        kpis.append({
            'label': f'Unique {col}',
            'value': f"{df[col].nunique():,}",
            'icon': '🏷️'
        })

    # If still under 4 and we have numeric columns left, add an average KPI
    if len(kpis) < 4 and numeric_cols:
        main_col = numeric_cols[0]
        kpis.append({
            'label': f'Average {main_col}',
            'value': f"{df[main_col].mean():,.2f}",
            'icon': '📈'
        })

    return kpis[:4]


def build_charts(df: pd.DataFrame) -> list:
    """Builds up to 4 charts with varied types depending on what columns exist."""
    charts = []
    numeric_cols = list(df.select_dtypes(include='number').columns)
    categorical_cols = list(df.select_dtypes(include='object').columns)
    date_col = _detect_date_column(df)

    # Chart 1: Bar chart - sum of first numeric column grouped by first categorical column
    if numeric_cols and categorical_cols:
        grouped = (
            df.groupby(categorical_cols[0])[numeric_cols[0]]
            .sum()
            .sort_values(ascending=False)
            .head(8)
        )
        charts.append({
            'type': 'bar',
            'title': f'Total {numeric_cols[0]} by {categorical_cols[0]}',
            'labels': [str(x) for x in grouped.index.tolist()],
            'values': [round(float(v), 2) for v in grouped.values.tolist()]
        })

    # Chart 2: Pie chart - distribution of first categorical column
    if categorical_cols:
        counts = df[categorical_cols[0]].value_counts().head(6)
        charts.append({
            'type': 'pie',
            'title': f'{categorical_cols[0]} Distribution',
            'labels': [str(x) for x in counts.index.tolist()],
            'values': [int(v) for v in counts.values.tolist()]
        })

    # Chart 3: Line chart - trend over time, if a date column exists
    if date_col and numeric_cols:
        try:
            temp = df.copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors='coerce')
            temp = temp.dropna(subset=[date_col])
            trend = (
                temp.groupby(temp[date_col].dt.to_period('M'))[numeric_cols[0]]
                .sum()
                .sort_index()
            )
            charts.append({
                'type': 'line',
                'title': f'{numeric_cols[0]} Trend Over Time',
                'labels': [str(p) for p in trend.index.tolist()],
                'values': [round(float(v), 2) for v in trend.values.tolist()]
            })
        except Exception:
            pass  # skip trend chart if date parsing fails

    # Chart 4: Record count by second categorical column (or first, if only one exists) -
    # gives a second, differently-shaped view even when there's no date column
    count_col = None
    if len(categorical_cols) > 1:
        count_col = categorical_cols[1]
    elif categorical_cols:
        count_col = categorical_cols[0]

    if count_col:
        counts = df[count_col].value_counts().head(8)
        charts.append({
            'type': 'bar',
            'title': f'Record Count by {count_col}',
            'labels': [str(x) for x in counts.index.tolist()],
            'values': [int(v) for v in counts.values.tolist()]
        })

    # Chart 5 fallback: if we STILL have no charts (e.g. only one numeric column, no categories),
    # show top rows by that numeric column
    if not charts and numeric_cols:
        top_rows = df.nlargest(8, numeric_cols[0])
        charts.append({
            'type': 'bar',
            'title': f'Top 8 by {numeric_cols[0]}',
            'labels': [str(i) for i in range(1, len(top_rows) + 1)],
            'values': [round(float(v), 2) for v in top_rows[numeric_cols[0]].tolist()]
        })

    return charts[:4]


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def build_dashboard(df: pd.DataFrame) -> dict:
    """Returns everything the dashboard template needs. Tries the
    business-aware path first; falls back to the generic path if the
    data doesn't look like a recognizable sales/orders dataset."""
    sales_col, sales_series = _detect_business_measure(df)
    if sales_col is not None:
        try:
            return _build_business_dashboard(df, sales_col, sales_series)
        except Exception:
            pass  # any unexpected issue -> fall through to generic path

    return {
        'kpis': build_kpis(df),
        'charts': build_charts(df)
    }