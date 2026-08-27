"""
services/query_engine.py
Module 9: Analytics Query Assistant

Since uploaded datasets can have any column names/types (not a fixed
revenue/category/region schema), this works in two ways:

    1. Quick queries - generic operations that adapt to whatever
       columns actually exist (top N by first numeric column, etc.)
    2. Free-text queries - sent to Claude, which returns a small JSON
       "query plan" (operation + column + optional group_by) that we
       then execute safely with pandas. Claude never runs code directly;
       it only picks from a fixed set of safe operations.
"""

import json
import pandas as pd
import anthropic
import config

# The only operations Claude is allowed to choose from - keeps execution safe
ALLOWED_OPERATIONS = ['top_n', 'group_sum', 'group_count', 'group_mean', 'describe', 'value_counts']


def get_quick_queries(df: pd.DataFrame) -> list:
    """Builds a few ready-made query buttons based on the columns that actually exist."""
    numeric_cols = list(df.select_dtypes(include='number').columns)
    categorical_cols = list(df.select_dtypes(include='object').columns)

    quick_queries = []

    if numeric_cols and categorical_cols:
        quick_queries.append({
            'label': f'Top 10 by {numeric_cols[0]}',
            'operation': 'top_n',
            'column': numeric_cols[0],
            'group_by': None
        })
        quick_queries.append({
            'label': f'Total {numeric_cols[0]} by {categorical_cols[0]}',
            'operation': 'group_sum',
            'column': numeric_cols[0],
            'group_by': categorical_cols[0]
        })

    if categorical_cols:
        quick_queries.append({
            'label': f'Most common {categorical_cols[0]}',
            'operation': 'value_counts',
            'column': categorical_cols[0],
            'group_by': None
        })

    if numeric_cols:
        quick_queries.append({
            'label': f'Summary stats for {numeric_cols[0]}',
            'operation': 'describe',
            'column': numeric_cols[0],
            'group_by': None
        })

    return quick_queries


def _run_operation(df: pd.DataFrame, operation: str, column: str, group_by: str = None) -> dict:
    """Safely executes one of the ALLOWED_OPERATIONS. Returns a dict with
    a result DataFrame (as HTML) and a short text insight."""

    if operation not in ALLOWED_OPERATIONS:
        return {'error': f'Unknown operation: {operation}'}

    if column and column not in df.columns:
        return {'error': f"Column '{column}' not found in this dataset"}

    if group_by and group_by not in df.columns:
        return {'error': f"Column '{group_by}' not found in this dataset"}

    try:
        if operation == 'top_n':
            result_df = df.sort_values(by=column, ascending=False).head(10)
            insight = f"Top 10 rows ranked by highest {column}."

        elif operation == 'group_sum':
            result_df = df.groupby(group_by)[column].sum().sort_values(ascending=False).reset_index()
            insight = f"Total {column} grouped by {group_by}, highest first."

        elif operation == 'group_count':
            result_df = df.groupby(group_by).size().sort_values(ascending=False).reset_index(name='count')
            insight = f"Record count grouped by {group_by}."

        elif operation == 'group_mean':
            result_df = df.groupby(group_by)[column].mean().sort_values(ascending=False).reset_index()
            insight = f"Average {column} grouped by {group_by}."

        elif operation == 'describe':
            result_df = df[column].describe().reset_index()
            result_df.columns = ['Statistic', 'Value']
            insight = f"Summary statistics for {column}."

        elif operation == 'value_counts':
            result_df = df[column].value_counts().head(10).reset_index()
            result_df.columns = [column, 'count']
            insight = f"Most common values in {column}."

        return {
            'table_html': result_df.to_html(classes='query-table', index=False),
            'insight': insight,
            'chart_labels': result_df.iloc[:, 0].astype(str).tolist()[:10],
            'chart_values': result_df.iloc[:, 1].tolist()[:10] if result_df.shape[1] > 1 else []
        }

    except Exception as e:
        return {'error': f'Could not run this query: {str(e)}'}


def run_quick_query(df: pd.DataFrame, query: dict) -> dict:
    """Runs one of the pre-built quick query dicts from get_quick_queries()."""
    return _run_operation(df, query['operation'], query['column'], query.get('group_by'))


def _find_matching_column(question_lower: str, candidate_columns: list) -> str:
    """Finds which column (if any) is mentioned in the question text."""
    for col in candidate_columns:
        if col.lower() in question_lower:
            return col
    return None


def parse_question_free(df: pd.DataFrame, question: str) -> dict:
    """
    FREE fallback - matches keywords in the question to a safe operation,
    with no API call and no cost. Handles common phrasings like:
        "average salary", "total revenue by city", "most common city",
        "top 10 by salary", "how many records per region"
    """
    question_lower = question.lower()
    numeric_cols = list(df.select_dtypes(include='number').columns)
    categorical_cols = list(df.select_dtypes(include='object').columns)

    matched_numeric = _find_matching_column(question_lower, numeric_cols)
    matched_categorical = _find_matching_column(question_lower, categorical_cols)

    has_grouping_word = any(w in question_lower for w in ['by ', 'per ', 'each ', 'group'])

    # "average" / "mean"
    if any(w in question_lower for w in ['average', 'mean', 'avg']):
        if matched_numeric and matched_categorical and has_grouping_word:
            return _run_operation(df, 'group_mean', matched_numeric, matched_categorical)
        if matched_numeric:
            return _run_operation(df, 'describe', matched_numeric)

    # "total" / "sum"
    if any(w in question_lower for w in ['total', 'sum']):
        if matched_numeric and matched_categorical:
            return _run_operation(df, 'group_sum', matched_numeric, matched_categorical)
        if matched_numeric:
            return _run_operation(df, 'describe', matched_numeric)

    # "top" / "highest" / "best" / "largest"
    if any(w in question_lower for w in ['top', 'highest', 'best', 'largest', 'maximum']):
        if matched_numeric:
            return _run_operation(df, 'top_n', matched_numeric)

    # "most common" / "most frequent" / "popular"
    if any(w in question_lower for w in ['most common', 'most frequent', 'popular', 'which']):
        if matched_categorical:
            return _run_operation(df, 'value_counts', matched_categorical)

    # "how many" / "count"
    if any(w in question_lower for w in ['how many', 'count', 'number of']):
        if matched_categorical:
            return _run_operation(df, 'group_count', None, matched_categorical)
        if matched_categorical is None and categorical_cols:
            return _run_operation(df, 'value_counts', categorical_cols[0])

    # Fallback: if we matched ANY column, show summary/counts for it
    if matched_numeric:
        return _run_operation(df, 'describe', matched_numeric)
    if matched_categorical:
        return _run_operation(df, 'value_counts', matched_categorical)

    return {
        'error': "Couldn't match that question to a column in this dataset. "
                 f"Try mentioning a column name directly, e.g. one of: "
                 f"{', '.join(numeric_cols + categorical_cols)}"
    }


def run_natural_language_query(df: pd.DataFrame, question: str) -> dict:
    """
    Tries Claude API translation first (if configured), falls back
    automatically to the free keyword-based parser on any failure -
    so the query box always works, even without API billing set up.
    """
    if not getattr(config, 'ANTHROPIC_API_KEY', None):
        return parse_question_free(df, question)

    numeric_cols = list(df.select_dtypes(include='number').columns)
    categorical_cols = list(df.select_dtypes(include='object').columns)

    prompt = f"""You translate a user's plain-English question into a JSON query plan for a pandas dataset.

Available numeric columns: {numeric_cols}
Available categorical columns: {categorical_cols}

Allowed operations ONLY: {ALLOWED_OPERATIONS}
- top_n: needs "column" (numeric)
- group_sum / group_count / group_mean: needs "column" (numeric, except group_count) and "group_by" (categorical)
- describe: needs "column" (numeric)
- value_counts: needs "column" (categorical)

User question: "{question}"

Respond with ONLY a JSON object, no other text, in this exact format:
{{"operation": "...", "column": "...", "group_by": "..." or null}}

If the question can't be answered with the allowed operations and available columns, respond with:
{{"error": "short explanation"}}
"""

    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        raw_text = message.content[0].text.strip()

        if raw_text.startswith('```'):
            raw_text = raw_text.strip('`').replace('json', '', 1).strip()

        plan = json.loads(raw_text)

        if 'error' in plan:
            return {'error': plan['error']}

        return _run_operation(df, plan.get('operation'), plan.get('column'), plan.get('group_by'))

    except Exception:
        # Silently fall back to the free keyword parser instead of showing an API error
        return parse_question_free(df, question)