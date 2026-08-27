"""
services/kmeans_segmentation.py
Module 6: Customer Segmentation (real ML - K-Means Clustering)

This is the ML-based counterpart to the RFM rule-based segmentation
already in the project. K-Means finds natural groupings in the data
based on Frequency and Monetary value (and Recency, if a date column
exists) using actual unsupervised learning, then labels each cluster
based on its characteristics (Premium, Regular, At-Risk, New).
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def _compute_elbow_curve(scaled_features, max_k: int) -> dict:
    """
    Runs K-Means for k = 2..max_k and records inertia (within-cluster
    sum of squares) for each, so the UI can render an elbow chart to
    justify the chosen number of clusters instead of a hardcoded value.
    """
    k_values = []
    inertia_values = []
    for k in range(2, max_k + 1):
        m = KMeans(n_clusters=k, random_state=42, n_init=10)
        m.fit(scaled_features)
        k_values.append(k)
        inertia_values.append(round(float(m.inertia_), 2))
    return {'k_values': k_values, 'inertia_values': inertia_values}


def _label_clusters(cluster_profile: pd.DataFrame) -> dict:
    """
    Assigns a human-readable label to each cluster based on its
    average Monetary and Frequency values relative to other clusters.
    """
    monetary_rank = cluster_profile['Monetary'].rank(ascending=False)
    frequency_rank = cluster_profile['Frequency'].rank(ascending=False)
    combined_rank = monetary_rank + frequency_rank

    labels = {}
    sorted_clusters = combined_rank.sort_values().index.tolist()
    label_names = ['Premium Customers', 'Regular Customers', 'At-Risk Customers', 'New Customers']

    for i, cluster_id in enumerate(sorted_clusters):
        labels[cluster_id] = label_names[i] if i < len(label_names) else f'Segment {i + 1}'

    return labels


def run_kmeans_segmentation(df: pd.DataFrame, customer_col: str, amount_col: str,
                             date_col: str = None, n_clusters: int = 4) -> dict:
    """
    Clusters customers into n_clusters groups using K-Means on
    Frequency + Monetary (+ Recency if available).
    """
    working = df.copy()
    working = working.dropna(subset=[customer_col, amount_col])
    working[amount_col] = pd.to_numeric(working[amount_col], errors='coerce')
    working = working.dropna(subset=[amount_col])

    if working.empty:
        return {'error': 'No valid rows after removing missing customer/amount values.'}

    agg_dict = {amount_col: ['count', 'sum']}
    has_date = False

    if date_col and date_col in working.columns:
        try:
            working[date_col] = pd.to_datetime(working[date_col], errors='coerce')
            if working[date_col].notna().sum() > 0:
                agg_dict[date_col] = 'max'
                has_date = True
        except Exception:
            has_date = False

    grouped = working.groupby(customer_col).agg(agg_dict)
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns]
    grouped = grouped.reset_index()
    grouped = grouped.rename(columns={f'{amount_col}_count': 'Frequency', f'{amount_col}_sum': 'Monetary'})

    feature_cols = ['Frequency', 'Monetary']

    if has_date:
        date_agg_col = f'{date_col}_max'
        most_recent = grouped[date_agg_col].max()
        grouped['Recency'] = (most_recent - grouped[date_agg_col]).dt.days
        feature_cols.append('Recency')

    n_customers = len(grouped)
    actual_clusters = min(n_clusters, max(2, n_customers // 3)) if n_customers >= 6 else 2

    if n_customers < 4:
        return {'error': f'Need at least 4 unique customers for clustering (found {n_customers}).'}

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(grouped[feature_cols])

    model = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    grouped['cluster'] = model.fit_predict(scaled_features)

    # --- Model evaluation: silhouette score ---
    # Measures how well-separated the clusters are (range -1 to 1;
    # higher is better). Needs at least 2 clusters and fewer clusters
    # than samples, otherwise it's undefined.
    try:
        if 2 <= actual_clusters < n_customers:
            sil_score = round(float(silhouette_score(scaled_features, grouped['cluster'])), 3)
        else:
            sil_score = None
    except Exception:
        sil_score = None

    # --- Model evaluation: elbow curve (justifies cluster count) ---
    max_k_for_elbow = min(8, n_customers - 1)
    elbow_curve = _compute_elbow_curve(scaled_features, max_k_for_elbow) if max_k_for_elbow >= 2 else None

    cluster_profile = grouped.groupby('cluster')[feature_cols].mean().round(2)
    cluster_labels = _label_clusters(cluster_profile)
    grouped['segment'] = grouped['cluster'].map(cluster_labels)

    segment_counts = grouped['segment'].value_counts().to_dict()

    cluster_profile_display = cluster_profile.copy()
    cluster_profile_display['segment'] = cluster_profile_display.index.map(cluster_labels)
    cluster_profile_display['customer_count'] = grouped.groupby('cluster').size()

    display_cols = [customer_col, 'Frequency', 'Monetary', 'segment']
    if has_date:
        display_cols.insert(1, 'Recency')

    result_table = grouped[display_cols].sort_values('Monetary', ascending=False)

    return {
        'has_date': has_date,
        'total_customers': n_customers,
        'clusters_used': actual_clusters,
        'silhouette_score': sil_score,
        'elbow_curve': elbow_curve,
        'segment_counts': segment_counts,
        'cluster_profile_html': cluster_profile_display.reset_index()[['segment', 'customer_count'] + feature_cols].to_html(classes='cluster-table', index=False, float_format='%.2f'),
        'table_html': result_table.head(100).to_html(classes='cluster-table', index=False, float_format='%.2f'),
        'full_table': result_table
    }