"""
services/forecasting.py
Module 7: Sales Forecasting

Aggregates a numeric column (e.g. revenue) by month from a date
column, fits a simple linear regression on the time trend, and
projects the next few months forward. Requires a date column and
a numeric amount column - if missing, the caller should show a
clear message rather than fail silently.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def _evaluate_on_holdout(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Holds out the most recent months as a test set, fits the same
    Linear Regression model on the remaining (earlier) months only,
    and reports how well it predicts the held-out months. This is a
    genuine out-of-sample evaluation, not just a fit-quality number
    on data the model has already seen.
    """
    n = len(y)
    test_size = min(3, max(1, n // 4))
    train_size = n - test_size

    if train_size < 2:
        return {'available': False, 'reason': 'Not enough months for a train/test split (need 5+).'}

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    holdout_model = LinearRegression()
    holdout_model.fit(X_train, y_train)
    y_pred = holdout_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    # R^2 needs at least 2 test points to be meaningful; with 1 point it's undefined
    r2 = r2_score(y_test, y_pred) if test_size >= 2 else None

    return {
        'available': True,
        'test_months': test_size,
        'train_months': train_size,
        'mae': round(float(mae), 2),
        'rmse': round(float(rmse), 2),
        'r2': round(float(r2), 3) if r2 is not None else None,
    }


def forecast_sales(df: pd.DataFrame, date_col: str, amount_col: str, periods: int = 3) -> dict:
    """
    Fits a linear trend on monthly totals and forecasts `periods` months ahead.
    Returns historical + forecasted values ready for charting.
    """
    working = df.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors='coerce')
    working = working.dropna(subset=[date_col, amount_col])

    if working.empty:
        return {'error': 'No valid rows with both a date and amount value.'}

    monthly = working.groupby(working[date_col].dt.to_period('M'))[amount_col].sum().sort_index()

    if len(monthly) < 3:
        return {'error': 'Need at least 3 months of historical data to forecast a trend.'}

    # Fit linear regression: X = month index (0, 1, 2, ...), y = monthly totals
    X = np.arange(len(monthly)).reshape(-1, 1)
    y = monthly.values

    model = LinearRegression()
    model.fit(X, y)

    # Out-of-sample evaluation: how well would this model have predicted
    # the most recent months if it had only seen the earlier ones?
    evaluation = _evaluate_on_holdout(X, y)

    # Forecast next `periods` months
    future_X = np.arange(len(monthly), len(monthly) + periods).reshape(-1, 1)
    future_y = model.predict(future_X)
    future_y = np.maximum(future_y, 0)  # don't predict negative sales

    last_period = monthly.index[-1]
    future_periods = [last_period + i for i in range(1, periods + 1)]

    historical_labels = [str(p) for p in monthly.index]
    historical_values = [round(float(v), 2) for v in monthly.values]

    forecast_labels = [str(p) for p in future_periods]
    forecast_values = [round(float(v), 2) for v in future_y]

    # Growth rate: comparing predicted next month to the last actual month
    last_actual = historical_values[-1]
    next_predicted = forecast_values[0]
    growth_rate = round(((next_predicted - last_actual) / last_actual) * 100, 2) if last_actual != 0 else 0

    return {
        'historical_labels': historical_labels,
        'historical_values': historical_values,
        'forecast_labels': forecast_labels,
        'forecast_values': forecast_values,
        'next_month_prediction': round(next_predicted, 2),
        'growth_rate_pct': growth_rate,
        'trend_direction': 'increasing' if model.coef_[0] > 0 else 'decreasing',
        'amount_col': amount_col,
        'evaluation': evaluation
    }