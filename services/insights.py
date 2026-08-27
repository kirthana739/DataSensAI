"""
services/insights.py
Module 8: AI Insight Engine

Two layers:
    1. Rule-based insights - always works, computed directly from the data
    2. AI-powered insights   - sends a compact stats summary (not raw rows)
                                to the Claude API for natural-language
                                business recommendations

NOTE: requires an ANTHROPIC_API_KEY in config.py. Get one from
https://console.anthropic.com -> API Keys
"""

import pandas as pd
import anthropic
import config


def generate_rule_based_insights(df: pd.DataFrame) -> dict:
    """
    Computes concrete stats from the data:
        - numeric column summaries (sum, mean, min, max)
        - top value for each categorical column
        - overall shape of the dataset
    This works on ANY dataset shape, not just a fixed revenue/category/region schema.
    """
    insights = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'numeric_summary': {},
        'categorical_summary': {}
    }

    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        insights['numeric_summary'][col] = {
            'sum': round(float(df[col].sum()), 2),
            'mean': round(float(df[col].mean()), 2),
            'min': round(float(df[col].min()), 2),
            'max': round(float(df[col].max()), 2)
        }

    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        if not value_counts.empty:
            insights['categorical_summary'][col] = {
                'top_value': str(value_counts.index[0]),
                'top_count': int(value_counts.iloc[0]),
                'unique_count': int(df[col].nunique())
            }

    return insights


def _build_summary_text(rule_based: dict) -> str:
    """Turns the rule-based stats dict into a compact text block for the AI prompt."""
    lines = [
        f"Dataset has {rule_based['total_records']} rows and {rule_based['total_columns']} columns.",
        ""
    ]

    if rule_based['numeric_summary']:
        lines.append("Numeric columns:")
        for col, stats in rule_based['numeric_summary'].items():
            lines.append(
                f"- {col}: sum={stats['sum']}, avg={stats['mean']}, "
                f"min={stats['min']}, max={stats['max']}"
            )
        lines.append("")

    if rule_based['categorical_summary']:
        lines.append("Categorical columns:")
        for col, stats in rule_based['categorical_summary'].items():
            lines.append(
                f"- {col}: {stats['unique_count']} unique values, "
                f"most common is '{stats['top_value']}' ({stats['top_count']} occurrences)"
            )

    return "\n".join(lines)


def generate_template_insights(rule_based: dict) -> str:
    """
    FREE fallback - writes natural-language style insights directly from the
    stats using templates, with no API call and no cost. This matches the
    'Basic Version' rule-based approach from the project spec.
    """
    lines = []
    total_records = rule_based['total_records']

    lines.append(f"📊 This dataset contains {total_records} records across {rule_based['total_columns']} columns.")

    for col, stats in rule_based['numeric_summary'].items():
        spread = stats['max'] - stats['min']
        lines.append(
            f"💰 '{col}' ranges from {stats['min']} to {stats['max']} "
            f"(average: {stats['mean']}). A spread of {round(spread, 2)} suggests "
            f"{'wide variation - worth checking for outliers' if spread > (stats['mean'] * 5 if stats['mean'] else 1) else 'fairly consistent values'}."
        )

    for col, stats in rule_based['categorical_summary'].items():
        pct = round((stats['top_count'] / total_records) * 100, 1) if total_records else 0
        lines.append(
            f"🏆 For '{col}', the most common value is '{stats['top_value']}', "
            f"appearing in {stats['top_count']} records ({pct}% of the dataset), "
            f"out of {stats['unique_count']} unique values total."
        )

    if not rule_based['numeric_summary'] and not rule_based['categorical_summary']:
        lines.append("⚠️ No numeric or categorical columns were detected to summarize.")

    return "\n".join(lines)


def generate_ai_insights(rule_based: dict) -> str:
    """
    Attempts Claude API insights first (if a key + billing credit are available).
    Falls back automatically to the free template-based insights on any failure,
    so the page never shows a raw API error to the user.
    """
    if not getattr(config, 'ANTHROPIC_API_KEY', None):
        return generate_template_insights(rule_based)

    summary_text = _build_summary_text(rule_based)

    prompt = f"""You are a data analyst reviewing a business dataset summary below.
Write 3-5 short, specific, actionable business insights based on this data.
Use plain language, no headers, one insight per line, each starting with a relevant emoji.

DATA SUMMARY:
{summary_text}
"""

    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception:
        # Silently fall back to free template insights instead of showing a raw error
        return generate_template_insights(rule_based)


def generate_full_insights(df: pd.DataFrame) -> dict:
    """Runs both layers and returns everything the route/template needs."""
    rule_based = generate_rule_based_insights(df)
    ai_text = generate_ai_insights(rule_based)

    return {
        'rule_based': rule_based,
        'ai_text': ai_text
    }