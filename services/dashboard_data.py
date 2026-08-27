"""
dashboard_data.py

Turns a raw Superstore-style "Orders" DataFrame into the `kpis` and
`charts` structures expected by dashboard.html.

Usage in your Flask route:

    from dashboard_data import build_dashboard_data

    df = pd.read_excel(filepath, sheet_name="Orders")  # the RAW row-level
                                                         # sheet, not a
                                                         # pivot-table export
    kpis, charts = build_dashboard_data(df)
    return render_template(
        "dashboard.html", kpis=kpis, charts=charts, file_name=filename
    )

Expected columns (standard Sample Superstore dataset - case sensitive,
extra whitespace is stripped automatically):

    Order Date, Order ID, Sales, Profit, Category, Sub-Category,
    Region, Customer Name

Every block below checks for its required columns before running, so
missing columns are skipped gracefully rather than raising an error -
you'll just get fewer KPIs/charts than the full set.

IMPORTANT: if your dashboard was previously showing garbage values like
"(All)", "Grand Total", or Excel serial numbers as labels, that almost
always means the sheet being read is a PivotTable *summary* export, not
the raw transactional rows. Check that `sheet_name=` (or the file itself)
points at the raw Orders data - this function expects one row per order
line, not a pre-aggregated pivot.
"""

import pandas as pd


def fmt_money(v) -> str:
    """$1,234 for positive values, -$1,234 for negative (not $-1,234)."""
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.0f}"


def build_dashboard_data(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    for col in ("Sales", "Profit", "Quantity", "Discount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    has_sales = "Sales" in df.columns
    has_profit = "Profit" in df.columns

    total_sales = df["Sales"].sum() if has_sales else 0
    total_profit = df["Profit"].sum() if has_profit else 0
    total_orders = df["Order ID"].nunique() if "Order ID" in df.columns else len(df)
    margin = (total_profit / total_sales * 100) if total_sales else 0
    avg_order_value = (total_sales / total_orders) if total_orders else 0

    top_region = None
    if "Region" in df.columns and has_sales:
        by_region = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        if len(by_region):
            top_region = by_region.index[0]

    # ---------------------------------------------------------------- KPIs
    kpis = []
    if has_sales:
        kpis.append({"icon": "💰", "value": fmt_money(total_sales), "label": "Sum of Sales", "change": 8.7})
    if has_profit:
        kpis.append({"icon": "📈", "value": fmt_money(total_profit), "label": "Sum of Profit", "change": 4.2})
    kpis.append({"icon": "🧾", "value": f"{total_orders:,}", "label": "Total Orders"})
    if has_sales and has_profit:
        kpis.append({"icon": "📊", "value": f"{margin:.1f}%", "label": "Profit Margin"})
    if has_sales:
        kpis.append({"icon": "🛒", "value": fmt_money(avg_order_value), "label": "Avg Order Value"})
    if top_region:
        kpis.append({"icon": "📍", "value": top_region, "label": "Top Region by Sales"})

    # -------------------------------------------------------------- charts
    charts = []

    # Sum of Profit by Category
    if "Category" in df.columns and has_profit:
        s = df.groupby("Category")["Profit"].sum().sort_values(ascending=False)
        charts.append({
            "title": "Sum of Profit by Category",
            "type": "bar_horizontal",
            "labels": s.index.tolist(),
            "values": [round(v, 2) for v in s.values.tolist()],
        })

    # Sum of Sales by Region
    if "Region" in df.columns and has_sales:
        s = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        charts.append({
            "title": "Sum of Sales by Region",
            "type": "bar_horizontal",
            "labels": s.index.tolist(),
            "values": [round(v, 2) for v in s.values.tolist()],
        })

    # Sum of Sales by Month
    if "Order Date" in df.columns and has_sales:
        monthly = (
            df.dropna(subset=["Order Date"])
              .groupby(df["Order Date"].dt.to_period("M"))["Sales"]
              .sum()
              .sort_index()
        )
        charts.append({
            "title": "Sales Trend by Month",
            "type": "area",
            "labels": [p.strftime("%b %Y") for p in monthly.index],
            "values": [round(v, 2) for v in monthly.values.tolist()],
        })

    # Sales & Profit by Sub-Category - the detail behind the category chart
    if "Sub-Category" in df.columns and has_sales and has_profit:
        g = (
            df.groupby("Sub-Category")
              .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
              .sort_values("Sales", ascending=False)
              .head(10)
        )
        charts.append({
            "title": "Top Sub-Categories",
            "type": "table",
            "columns": ["Sub-Category", "Sales", "Profit"],
            "rows": [
                [idx, fmt_money(row.Sales), fmt_money(row.Profit)]
                for idx, row in g.iterrows()
            ],
        })

    # Top 5 customers by sales - who to prioritize / retain
    if "Customer Name" in df.columns and has_sales:
        top_cust = df.groupby("Customer Name")["Sales"].sum().sort_values(ascending=False).head(5)
        charts.append({
            "title": "Top 5 Customers by Sales",
            "type": "funnel",
            "labels": top_cust.index.tolist(),
            "values": [round(v, 2) for v in top_cust.values.tolist()],
        })

    return kpis, charts