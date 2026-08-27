"""
services/anomaly_detection.py
Module 8: Anomaly Detection (real ML - Isolation Forest)

Unlike Module 3's IQR method (a simple statistical rule per column),
this uses scikit-learn's Isolation Forest - an actual unsupervised
machine learning model that considers ALL numeric columns together
to find rows that are unusual as a combination, not just single
out-of-range values.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> dict:
    """
    Runs Isolation Forest on all numeric columns.
    contamination = expected proportion of anomalies (default 5%).
    Returns the dataframe with anomaly flags/scores plus a summary.
    """
    numeric_cols = list(df.select_dtypes(include='number').columns)

    if len(numeric_cols) == 0:
        return {'error': 'No numeric columns found - Isolation Forest needs at least one numeric column.'}

    working = df.copy()
    feature_data = working[numeric_cols].fillna(working[numeric_cols].median())

    if len(feature_data) < 10:
        return {'error': 'Need at least 10 rows for reliable anomaly detection.'}

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_data)

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    predictions = model.fit_predict(scaled_features)  # -1 = anomaly, 1 = normal
    scores = model.decision_function(scaled_features)  # lower = more anomalous

    working['anomaly_flag'] = predictions
    working['anomaly_score'] = scores.round(4)
    working['is_anomaly'] = working['anomaly_flag'] == -1

    anomalies = working[working['is_anomaly']].sort_values('anomaly_score')
    normal_count = int((working['anomaly_flag'] == 1).sum())
    anomaly_count = int((working['anomaly_flag'] == -1).sum())
    total = len(working)

    display_cols = numeric_cols + ['anomaly_score']
    anomaly_table_html = anomalies[display_cols].head(50).to_html(classes='anomaly-table', index=True, float_format='%.2f')

    return {
        'total_records': total,
        'anomaly_count': anomaly_count,
        'normal_count': normal_count,
        'anomaly_percentage': round((anomaly_count / total) * 100, 2),
        'numeric_columns_used': numeric_cols,
        'anomaly_table_html': anomaly_table_html,
        'full_dataframe': working,
        'score_distribution': {
            'labels': [str(i) for i in range(len(working))][:200],
            'scores': working['anomaly_score'].tolist()[:200],
            'flags': working['is_anomaly'].tolist()[:200]
        }
    }