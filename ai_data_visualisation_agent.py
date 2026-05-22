import ast
import base64
import difflib
import glob
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()  # Load .env from the current working directory (silent no-op if file is absent)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from groq import APIConnectionError, APIError, AuthenticationError, Groq, RateLimitError
from PIL import Image

try:
    import plotly.express as px
except ImportError:
    px = None


warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

CODE_BLOCK_PATTERN = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
SAFE_IMPORT_ROOTS = {"pandas", "numpy", "matplotlib", "seaborn", "plotly", "math", "statistics"}
BANNED_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "__import__",
    "breakpoint",
    "help",
    "exit",
    "quit",
}
BANNED_IMPORT_ROOTS = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "pathlib",
    "glob",
    "tempfile",
    "pickle",
    "importlib",
}
BANNED_ATTRS = {"system", "popen", "remove", "unlink", "rmdir", "rmtree", "rename", "replace"}


def normalize_column_name(name: Any) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "column"


def make_unique_columns(columns: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    unique = []
    for col in columns:
        base = normalize_column_name(col)
        count = seen.get(base, 0)
        unique.append(base if count == 0 else f"{base}_{count + 1}")
        seen[base] = count + 1
    return unique


def text_columns(df: pd.DataFrame) -> List[str]:
    return [
        col for col in df.columns
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
    ]


def categorical_columns(df: pd.DataFrame) -> List[str]:
    return [
        col for col in df.columns
        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(df[col])
        )
    ]


def load_csv_file(uploaded_file) -> Tuple[str, pd.DataFrame]:
    uploaded_file.seek(0)
    return uploaded_file.name, pd.read_csv(uploaded_file)


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = list(df.select_dtypes(include="number").columns)
    categorical_cols = categorical_columns(df)
    datetime_candidates = []

    for col in df.columns:
        if col in text_columns(df):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                datetime_candidates.append(col)

    missing = (
        pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": df.isna().sum().values,
                "missing_percent": (df.isna().mean().values * 100).round(2),
            }
        )
        .sort_values("missing_percent", ascending=False)
        .reset_index(drop=True)
    )

    overview = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_candidates": datetime_candidates,
    }

    numeric_summary = df[numeric_cols].describe().T.reset_index().rename(columns={"index": "column"}) if numeric_cols else pd.DataFrame()
    categorical_summary_rows = []
    for col in categorical_cols:
        top_value = df[col].mode(dropna=True)
        categorical_summary_rows.append(
            {
                "column": col,
                "unique_values": int(df[col].nunique(dropna=True)),
                "top_value": "" if top_value.empty else str(top_value.iloc[0]),
            }
        )

    return {
        "overview": overview,
        "missing": missing,
        "numeric_summary": numeric_summary,
        "categorical_summary": pd.DataFrame(categorical_summary_rows),
        "correlation": df[numeric_cols].corr(numeric_only=True) if len(numeric_cols) > 1 else pd.DataFrame(),
    }


def preprocess_dataset(
    df: pd.DataFrame,
    normalize_columns: bool,
    remove_duplicates: bool,
    fill_missing: bool,
    parse_dates: bool,
    clip_outliers: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    processed = df.copy()
    steps = []

    if normalize_columns:
        processed.columns = make_unique_columns(list(processed.columns))
        steps.append("Normalized column names to snake_case.")

    if remove_duplicates:
        before = len(processed)
        processed = processed.drop_duplicates()
        steps.append(f"Removed {before - len(processed)} duplicate rows.")

    if parse_dates:
        converted = []
        for col in text_columns(processed):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(processed[col], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                processed[col] = parsed
                converted.append(col)
        if converted:
            steps.append(f"Converted likely date columns: {', '.join(converted)}.")

    if fill_missing:
        numeric_cols = list(processed.select_dtypes(include="number").columns)
        categorical_cols = categorical_columns(processed)
        for col in numeric_cols:
            processed[col] = processed[col].fillna(processed[col].median())
        for col in categorical_cols:
            mode = processed[col].mode(dropna=True)
            fill_value = "Unknown" if mode.empty else mode.iloc[0]
            processed[col] = processed[col].fillna(fill_value)
        steps.append("Filled numeric missing values with medians and categorical missing values with modes.")

    if clip_outliers:
        clipped_cols = []
        for col in processed.select_dtypes(include="number").columns:
            q1 = processed[col].quantile(0.25)
            q3 = processed[col].quantile(0.75)
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                before = processed[col].copy()
                processed[col] = processed[col].clip(lower, upper)
                if not before.equals(processed[col]):
                    clipped_cols.append(col)
        if clipped_cols:
            steps.append(f"Clipped outliers with the IQR rule in: {', '.join(clipped_cols)}.")

    if not steps:
        steps.append("No preprocessing steps were applied.")

    return processed, steps


def suggest_charts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    numeric = list(df.select_dtypes(include="number").columns)
    categorical = [col for col in categorical_columns(df) if df[col].nunique() <= 30]
    datetime_cols = list(df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
    outcome_words = [
        "score",
        "target",
        "label",
        "result",
        "outcome",
        "rating",
        "sales",
        "revenue",
        "profit",
        "price",
        "cost",
        "conversion",
        "churn",
        "retention",
    ]
    likely_outcomes = [
        col for col in numeric
        if any(word in normalize_column_name(col) for word in outcome_words)
    ]

    if len(numeric) >= 2:
        corr = df[numeric].corr(numeric_only=True).abs()
        pairs = []
        for i, x in enumerate(numeric):
            for y in numeric[i + 1:]:
                value = corr.loc[x, y]
                if pd.notna(value):
                    pairs.append((float(value), x, y))
        pairs.sort(reverse=True)

        if likely_outcomes:
            target = likely_outcomes[0]
            target_pairs = []
            for x in numeric:
                if x == target:
                    continue
                value = corr.loc[x, target]
                if pd.notna(value):
                    target_pairs.append((float(value), x, target))
            target_pairs.sort(reverse=True)
            for value, x, y in target_pairs[:2]:
                strength = "weak" if value < 0.25 else "clear"
                suggestions.append(
                    {
                        "name": "Outcome relationship",
                        "chart": "scatter",
                        "x": x,
                        "y": y,
                        "score": value + 0.25,
                        "reason": (
                            f"{pretty_label(y)} looks like an outcome column; {pretty_label(x)} has the "
                            f"strongest {strength} relationship with it (|r| = {value:.2f})."
                        ),
                    }
                )

        for value, x, y in pairs[:2]:
            if value >= 0.25:
                suggestions.append(
                    {
                        "name": "Strongest numeric relationship",
                        "chart": "scatter",
                        "x": x,
                        "y": y,
                        "score": value,
                        "reason": f"These two numeric columns have the strongest relationship detected (|r| = {value:.2f}).",
                    }
                )
        if pairs and pairs[0][0] >= 0.1 and len(numeric) >= 3:
            suggestions.append(
                {
                    "name": "Correlation map",
                    "chart": "heatmap",
                    "x": "",
                    "y": "",
                    "score": pairs[0][0] - 0.02,
                    "reason": "Several numeric columns are available, so a correlation map helps spot related measures quickly.",
                }
            )

    for category in categorical:
        valid = df[[category] + numeric].dropna() if numeric else pd.DataFrame()
        if valid.empty or valid[category].nunique() < 2:
            continue
        counts = valid[category].value_counts()
        usable_categories = counts[counts >= 3].index
        if len(usable_categories) < 2:
            continue
        valid = valid[valid[category].isin(usable_categories)]
        for number in numeric:
            grouped = valid.groupby(category, dropna=False)[number]
            means = grouped.mean()
            overall_std = valid[number].std()
            if pd.isna(overall_std) or overall_std == 0 or means.empty:
                continue
            spread = float((means.max() - means.min()) / overall_std)
            if spread >= 0.6:
                suggestions.append(
                    {
                        "name": "Most different groups",
                        "chart": "bar",
                        "x": category,
                        "y": number,
                        "score": spread,
                        "reason": f"Average {pretty_label(number)} differs meaningfully across {pretty_label(category)} groups.",
                    }
                )
                suggestions.append(
                    {
                        "name": "Group spread and outliers",
                        "chart": "box",
                        "x": category,
                        "y": number,
                        "score": spread - 0.05,
                        "reason": f"This checks whether {pretty_label(category)} groups differ consistently or only because of outliers.",
                    }
                )

    for date_col in datetime_cols:
        for number in numeric:
            trend_data = df[[date_col, number]].dropna().sort_values(date_col)
            if len(trend_data) < 6 or trend_data[date_col].nunique() < 4:
                continue
            ordinal = trend_data[date_col].map(pd.Timestamp.toordinal)
            trend_strength = abs(float(pd.Series(ordinal).corr(trend_data[number])))
            if pd.notna(trend_strength) and trend_strength >= 0.25:
                suggestions.append(
                    {
                        "name": "Clearest trend",
                        "chart": "line",
                        "x": date_col,
                        "y": number,
                        "score": trend_strength,
                        "reason": f"{pretty_label(number)} shows the clearest time-based movement in the dataset.",
                    }
                )

    for number in numeric:
        series = df[number].dropna()
        if len(series) < 5 or series.nunique() < 3:
            continue
        skew = abs(float(series.skew())) if pd.notna(series.skew()) else 0
        cv = abs(float(series.std() / series.mean())) if series.mean() not in (0, None) and pd.notna(series.mean()) else 0
        score = max(skew / 2, min(cv, 2) / 2)
        suggestions.append(
            {
                "name": "Most informative distribution",
                "chart": "histogram",
                "x": number,
                "y": "",
                "score": score,
                "reason": f"{pretty_label(number)} has enough variation to make its distribution worth checking.",
            }
        )

    deduped = {}
    for suggestion in sorted(suggestions, key=lambda item: item.get("score", 0), reverse=True):
        key = (suggestion["chart"], suggestion.get("x", ""), suggestion.get("y", ""))
        deduped.setdefault(key, suggestion)

    return list(deduped.values())[:6]


def validate_chart_suggestions(suggestions: Any, df: pd.DataFrame) -> List[Dict[str, Any]]:
    if not isinstance(suggestions, list):
        return []

    valid = []
    columns = set(map(str, df.columns))
    numeric = set(map(str, df.select_dtypes(include="number").columns))
    categorical = set(map(str, categorical_columns(df)))
    datetime_cols = set(map(str, df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns))
    allowed_charts = {"histogram", "scatter", "bar", "box", "line", "heatmap"}

    for item in suggestions:
        if not isinstance(item, dict):
            continue
        chart = str(item.get("chart", "")).strip().lower()
        x = str(item.get("x", "") or "").strip()
        y = str(item.get("y", "") or "").strip()
        if chart not in allowed_charts:
            continue
        if chart == "histogram" and x not in numeric:
            continue
        if chart == "scatter" and (x not in numeric or y not in numeric or x == y):
            continue
        if chart in {"bar", "box"} and (x not in categorical or y not in numeric):
            continue
        if chart == "line" and (x not in datetime_cols or y not in numeric):
            continue
        if chart == "heatmap" and len(numeric) < 2:
            continue
        if x and x not in columns:
            continue
        if y and y not in columns:
            continue

        valid.append(
            {
                "name": str(item.get("name") or chart.title())[:80],
                "chart": chart,
                "x": x,
                "y": y,
                "score": 1.0 - (len(valid) * 0.01),
                "reason": str(item.get("reason") or "Recommended from the dataset profile.")[:240],
            }
        )

    return valid[:6]


def _build_deep_profile_context(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Build an enriched dataset context for AI chart recommendations."""
    overview = profile["overview"]
    numeric_cols = overview["numeric_columns"]
    categorical_cols = overview["categorical_columns"]

    # Correlation pairs ranked by strength
    correlation_insights = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True).abs()
        pairs = []
        for i, x in enumerate(numeric_cols):
            for y in numeric_cols[i + 1:]:
                value = corr.loc[x, y]
                if pd.notna(value):
                    pairs.append({"col_a": x, "col_b": y, "abs_correlation": round(float(value), 3)})
        pairs.sort(key=lambda p: p["abs_correlation"], reverse=True)
        correlation_insights = pairs[:10]

    # Distribution shape for each numeric column
    distribution_insights = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 5:
            continue
        skew_val = float(series.skew()) if pd.notna(series.skew()) else 0
        distribution_insights.append({
            "column": col,
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4),
            "skewness": round(skew_val, 3),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "unique_values": int(series.nunique()),
            "shape": "highly skewed" if abs(skew_val) > 1.5 else "moderately skewed" if abs(skew_val) > 0.5 else "roughly symmetric",
        })

    # Categorical column breakdowns
    categorical_insights = []
    for col in categorical_cols:
        nunique = int(df[col].nunique(dropna=True))
        value_counts = df[col].value_counts(dropna=True).head(5)
        top_values = {str(k): int(v) for k, v in value_counts.items()}
        categorical_insights.append({
            "column": col,
            "unique_count": nunique,
            "top_values": top_values,
            "is_binary": nunique == 2,
            "is_high_cardinality": nunique > 30,
        })

    # Group difference signals: which categorical×numeric combos show meaningful variation
    group_difference_signals = []
    for cat in categorical_cols:
        if df[cat].nunique() < 2 or df[cat].nunique() > 30:
            continue
        for num in numeric_cols:
            valid = df[[cat, num]].dropna()
            if len(valid) < 10:
                continue
            grouped = valid.groupby(cat)[num]
            means = grouped.mean()
            overall_std = valid[num].std()
            if pd.isna(overall_std) or overall_std == 0:
                continue
            spread = float((means.max() - means.min()) / overall_std)
            if spread >= 0.5:
                group_difference_signals.append({
                    "categorical": cat,
                    "numeric": num,
                    "spread_score": round(spread, 3),
                    "interpretation": f"Average {num} differs meaningfully across {cat} groups",
                })
    group_difference_signals.sort(key=lambda g: g["spread_score"], reverse=True)

    # Temporal patterns
    datetime_cols = list(df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
    temporal_signals = []
    for date_col in datetime_cols:
        for num in numeric_cols:
            trend_data = df[[date_col, num]].dropna().sort_values(date_col)
            if len(trend_data) < 6:
                continue
            ordinal = trend_data[date_col].map(pd.Timestamp.toordinal)
            corr_val = abs(float(pd.Series(ordinal).corr(trend_data[num])))
            if pd.notna(corr_val) and corr_val >= 0.15:
                temporal_signals.append({
                    "date_column": date_col,
                    "value_column": num,
                    "trend_strength": round(corr_val, 3),
                    "date_range": f"{trend_data[date_col].min()} to {trend_data[date_col].max()}",
                })
    temporal_signals.sort(key=lambda t: t["trend_strength"], reverse=True)

    # Detect likely outcome/target columns
    outcome_words = [
        "score", "target", "label", "result", "outcome", "rating", "sales",
        "revenue", "profit", "price", "cost", "conversion", "churn",
        "retention", "count", "amount", "total", "value",
    ]
    likely_outcomes = [
        col for col in numeric_cols
        if any(word in normalize_column_name(col) for word in outcome_words)
    ]

    return {
        "dataset_shape": {"rows": overview["rows"], "columns": overview["columns"]},
        "column_names": list(map(str, df.columns)),
        "numeric_columns": list(map(str, numeric_cols)),
        "categorical_columns": list(map(str, categorical_cols)),
        "datetime_columns": list(map(str, datetime_cols)),
        "likely_outcome_columns": likely_outcomes,
        "correlation_pairs": correlation_insights,
        "distribution_profiles": distribution_insights,
        "categorical_breakdowns": categorical_insights[:8],
        "group_difference_signals": group_difference_signals[:8],
        "temporal_signals": temporal_signals[:5],
        "sample_rows": df.head(5).to_dict(orient="records"),
    }


def suggest_charts_with_ai(
    df: pd.DataFrame,
    profile: Dict[str, Any],
    api_key: str,
    model_name: str,
) -> Tuple[List[Dict[str, Any]], str]:
    fallback = suggest_charts(df)
    if not api_key:
        return fallback, "Using local data-driven recommendations. Add a Groq API key for AI recommendations."

    deep_context = _build_deep_profile_context(df, profile)

    system_prompt = """You are a senior data analyst recommending the most insightful charts for a dataset.

Your goal is NOT to show arbitrary graphs. Your goal is to surface the most valuable analytical insights hidden in the data.

You will receive a detailed data profile including:
- Column types, distributions, skewness, and summary statistics
- Correlation pairs ranked by strength
- Group difference signals (which categorical groups show meaningful numeric variation)
- Temporal trend signals (which metrics trend over time)
- Likely outcome/target columns (based on column naming patterns)

Follow this analytical reasoning process:
1. IDENTIFY the likely analytical questions this dataset can answer (e.g., "What drives sales?", "How do customer segments differ?", "Is there a trend over time?")
2. PRIORITIZE charts that answer the most important questions first
3. For each chart, explain the specific analytical insight it reveals

Rules:
- Return ONLY a JSON array with 3 to 6 objects
- Each object must have: name, chart, x, y, reason
- Allowed chart values: histogram, scatter, bar, box, line, heatmap
- Use ONLY column names from the provided context — never invent column names
- For histogram: x = numeric column, y = "" (empty string)
- For heatmap: x = "", y = "" (empty strings)
- For scatter: both x and y must be numeric columns
- For bar and box: x = categorical column (with ≤30 unique values), y = numeric column
- For line: x = datetime column, y = numeric column
- The "reason" must explain WHAT ANALYTICAL QUESTION this chart answers and WHY it matters for this specific dataset
- Prefer strongest correlations, biggest group differences, clearest trends, and most skewed distributions
- If outcome columns are detected, prioritize charts showing what drives/predicts those outcomes
- Do NOT suggest charts for columns with no meaningful variation or weak signals
- Do NOT suggest redundant charts (e.g., two scatter plots of the same pair)"""

    try:
        client = Groq(api_key=api_key, timeout=20.0, max_retries=0)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(deep_context, default=str)},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r"\[.*\]", content, re.DOTALL)
        parsed = json.loads(match.group(0) if match else content)
        ai_suggestions = validate_chart_suggestions(parsed, df)
        if ai_suggestions:
            return ai_suggestions, "✨ AI-powered recommendations based on deep dataset analysis."
        return fallback, "AI recommendations were not valid for this dataset, so local data-driven recommendations are shown."
    except Exception as exc:
        return fallback, f"AI recommendations are unavailable, so local data-driven recommendations are shown. Details: {exc}"


def _generate_query_suggestions_local(df: pd.DataFrame, profile: Dict[str, Any]) -> List[str]:
    """Generate helpful query suggestions based on local dataset analysis."""
    suggestions = []
    overview = profile["overview"]
    numeric_cols = overview["numeric_columns"]
    categorical_cols = overview["categorical_columns"]
    datetime_candidates = overview["datetime_candidates"]

    # Distribution queries
    for col in numeric_cols[:2]:
        suggestions.append(f"Show the distribution of {pretty_label(col)}")

    # Correlation queries
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True).abs()
        pairs = []
        for i, x in enumerate(numeric_cols):
            for y in numeric_cols[i + 1:]:
                value = corr.loc[x, y]
                if pd.notna(value):
                    pairs.append((float(value), x, y))
        pairs.sort(reverse=True)
        if pairs:
            _, x, y = pairs[0]
            suggestions.append(f"What is the relationship between {pretty_label(x)} and {pretty_label(y)}?")
        if len(numeric_cols) >= 3:
            suggestions.append("Show a correlation heatmap of all numeric columns")

    # Category comparison queries
    usable_cats = [col for col in categorical_cols if 2 <= df[col].nunique() <= 20]
    if usable_cats and numeric_cols:
        cat = usable_cats[0]
        num = numeric_cols[0]
        suggestions.append(f"Compare average {pretty_label(num)} across different {pretty_label(cat)} groups")

    # Outlier / spread queries
    if usable_cats and numeric_cols:
        suggestions.append(f"Show box plots of {pretty_label(numeric_cols[0])} by {pretty_label(usable_cats[0])}")

    # Time trend queries
    if datetime_candidates and numeric_cols:
        suggestions.append(f"Show the trend of {pretty_label(numeric_cols[0])} over time")

    # General insight queries
    if numeric_cols:
        outcome_words = ["score", "target", "sales", "revenue", "profit", "price", "conversion", "churn", "rating"]
        likely_outcomes = [col for col in numeric_cols if any(w in normalize_column_name(col) for w in outcome_words)]
        if likely_outcomes:
            suggestions.append(f"What factors most influence {pretty_label(likely_outcomes[0])}?")

    suggestions.append("Give me a complete statistical summary of the dataset")

    return suggestions[:8]


def generate_query_suggestions(
    df: pd.DataFrame,
    profile: Dict[str, Any],
    api_key: str,
    model_name: str,
) -> Tuple[List[str], str]:
    """Generate AI-powered query suggestions by analyzing the uploaded dataset."""
    local_suggestions = _generate_query_suggestions_local(df, profile)
    if not api_key:
        return local_suggestions, "💡 Suggestions based on dataset analysis. Add a Groq API key for smarter recommendations."

    deep_context = _build_deep_profile_context(df, profile)

    system_prompt = """You are a senior data analyst. Given a detailed dataset profile, suggest 6 to 8 specific, actionable questions a user can ask to gain the most valuable insights from their data through visualizations.

Your questions should:
1. Be specific to the actual columns and patterns in this dataset
2. Target the most interesting relationships, trends, distributions, and group comparisons
3. Be phrased as natural language questions or requests that a user would type into a chat
4. Cover a variety of chart types (distributions, comparisons, correlations, trends)
5. Prioritize questions that would reveal non-obvious insights

Rules:
- Return ONLY a JSON array of strings (each string is a question/query)
- Use actual column names from the dataset (use human-readable versions with spaces and title case)
- Questions should be self-contained and directly answerable with a visualization
- Include a mix of simple exploratory questions and deeper analytical questions
- Do NOT suggest generic questions like "show me a chart" — be specific"""

    try:
        client = Groq(api_key=api_key, timeout=15.0, max_retries=0)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(deep_context, default=str)},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r"\[.*\]", content, re.DOTALL)
        parsed = json.loads(match.group(0) if match else content)
        if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed) and len(parsed) >= 3:
            return parsed[:8], "✨ AI-powered query suggestions based on deep dataset analysis."
        return local_suggestions, "AI suggestions were not valid, showing data-driven suggestions instead."
    except Exception:
        return local_suggestions, "💡 Suggestions based on dataset analysis."


def pretty_label(column_name: str) -> str:
    return str(column_name).replace("_", " ").strip().title()


def style_easy_chart(fig, title: str, x_title: str = "", y_title: str = ""):
    fig.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 22, "color": "#111827"},
        },
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"size": 14, "color": "#111827"},
        margin={"l": 35, "r": 25, "t": 80, "b": 55},
        hoverlabel={"bgcolor": "#111827", "font_size": 13, "font_color": "#ffffff"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(
        title_text=x_title,
        showgrid=False,
        zeroline=False,
        title_font={"size": 15},
        tickfont={"size": 13},
    )
    fig.update_yaxes(
        title_text=y_title,
        gridcolor="#e5e7eb",
        zeroline=False,
        title_font={"size": 15},
        tickfont={"size": 13},
    )
    return fig


def chart_takeaway(df: pd.DataFrame, suggestion: Dict[str, Any]) -> str:
    if suggestion.get("reason"):
        return suggestion["reason"]

    chart = suggestion["chart"]
    x = suggestion.get("x", "")
    y = suggestion.get("y", "")

    if chart == "histogram":
        series = df[x].dropna()
        if series.empty:
            return "No values are available for this column."
        return (
            f"Most values for {pretty_label(x)} are around {series.median():.2f}. "
            f"The average is {series.mean():.2f}, based on {len(series):,} rows."
        )

    if chart == "scatter":
        corr = df[[x, y]].corr(numeric_only=True).iloc[0, 1]
        direction = "positive" if corr > 0 else "negative"
        strength = "strong" if abs(corr) >= 0.7 else "moderate" if abs(corr) >= 0.4 else "weak"
        return f"This shows a {strength} {direction} relationship between {pretty_label(x)} and {pretty_label(y)}."

    if chart == "bar":
        grouped = df.groupby(x, dropna=False)[y].mean().sort_values(ascending=False)
        if grouped.empty:
            return "No category summary is available."
        return f"The highest average {pretty_label(y)} is for {grouped.index[0]} ({grouped.iloc[0]:.2f})."

    if chart == "box":
        return f"Use this to compare the typical {pretty_label(y)} and outliers across {pretty_label(x)} groups."

    if chart == "line":
        return f"This shows how {pretty_label(y)} changes as {pretty_label(x)} increases over time/order."

    if chart == "heatmap":
        return "Darker cells show stronger relationships. Values close to 1 or -1 are more important."

    return "This chart summarizes the selected dataset columns."


def render_recommended_chart(df: pd.DataFrame, suggestion: Dict[str, Any]):
    if px is None:
        st.info("Install plotly to enable interactive recommended charts.")
        return

    chart = suggestion["chart"]
    title = suggestion["name"]
    x = suggestion.get("x", "")
    y = suggestion.get("y", "")
    color = "#2563eb"
    accent = "#f97316"

    st.info(chart_takeaway(df, suggestion))

    if chart == "histogram":
        series = df[x].dropna()
        fig = px.histogram(
            df,
            x=x,
            nbins=min(12, max(5, int(series.nunique() ** 0.5) if not series.empty else 8)),
            title=f"How {pretty_label(x)} Is Distributed",
            labels={x: pretty_label(x), "count": "Number of rows"},
            color_discrete_sequence=[color],
            opacity=0.82,
        )
        if not series.empty:
            mean_value = series.mean()
            median_value = series.median()
            fig.add_vline(
                x=mean_value,
                line_dash="dash",
                line_color=accent,
                annotation_text=f"Average: {mean_value:.2f}",
                annotation_position="top right",
            )
            fig.add_vline(
                x=median_value,
                line_dash="dot",
                line_color="#16a34a",
                annotation_text=f"Middle: {median_value:.2f}",
                annotation_position="top left",
            )
        fig.update_traces(marker_line_width=1, marker_line_color="#ffffff", hovertemplate=f"{pretty_label(x)}: %{{x}}<br>Rows: %{{y}}<extra></extra>")
        fig.update_layout(bargap=0.08)
        fig = style_easy_chart(fig, f"How {pretty_label(x)} Is Distributed", pretty_label(x), "Number of rows")
    elif chart == "scatter":
        fig = px.scatter(
            df,
            x=x,
            y=y,
            title=f"{pretty_label(y)} vs {pretty_label(x)}",
            labels={x: pretty_label(x), y: pretty_label(y)},
            color_discrete_sequence=[color],
        )
        fig.update_traces(marker={"size": 9, "opacity": 0.7, "line": {"width": 1, "color": "#ffffff"}})
        fig = style_easy_chart(fig, f"{pretty_label(y)} vs {pretty_label(x)}", pretty_label(x), pretty_label(y))
    elif chart == "bar":
        grouped = df.groupby(x, dropna=False)[y].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(
            grouped,
            x=x,
            y=y,
            text=y,
            title=f"Average {pretty_label(y)} by {pretty_label(x)}",
            labels={x: pretty_label(x), y: f"Average {pretty_label(y)}"},
            color_discrete_sequence=[color],
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_width=0)
        fig = style_easy_chart(fig, f"Average {pretty_label(y)} by {pretty_label(x)}", pretty_label(x), f"Average {pretty_label(y)}")
    elif chart == "box":
        fig = px.box(
            df,
            x=x,
            y=y,
            points="outliers",
            title=f"Spread of {pretty_label(y)} by {pretty_label(x)}",
            labels={x: pretty_label(x), y: pretty_label(y)},
            color_discrete_sequence=[color],
        )
        fig = style_easy_chart(fig, f"Spread of {pretty_label(y)} by {pretty_label(x)}", pretty_label(x), pretty_label(y))
    elif chart == "line":
        grouped = df.sort_values(x)
        fig = px.line(
            grouped,
            x=x,
            y=y,
            markers=True,
            title=f"{pretty_label(y)} Trend over {pretty_label(x)}",
            labels={x: pretty_label(x), y: pretty_label(y)},
            color_discrete_sequence=[color],
        )
        fig.update_traces(line={"width": 3}, marker={"size": 7})
        fig = style_easy_chart(fig, f"{pretty_label(y)} Trend over {pretty_label(x)}", pretty_label(x), pretty_label(y))
    elif chart == "heatmap":
        corr = df.select_dtypes(include="number").corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Correlation between Numeric Columns",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        fig.update_layout(coloraxis_colorbar={"title": "Relationship"})
        fig = style_easy_chart(fig, "Correlation between Numeric Columns", "", "")
    else:
        return

    st.plotly_chart(fig, width="stretch")


def pick_column_from_query(query: str, columns: List[str]) -> Optional[str]:
    query_norm = normalize_column_name(query)
    for col in columns:
        if normalize_column_name(col) in query_norm:
            return col

    words = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    scored = []
    for col in columns:
        col_words = set(re.findall(r"[a-zA-Z0-9]+", pretty_label(col).lower()))
        overlap = len(words & col_words)
        if overlap:
            scored.append((overlap, col))

    if scored:
        return sorted(scored, reverse=True)[0][1]

    matches = difflib.get_close_matches(query_norm, [normalize_column_name(col) for col in columns], n=1, cutoff=0.35)
    if matches:
        reverse_lookup = {normalize_column_name(col): col for col in columns}
        return reverse_lookup[matches[0]]
    return None


def build_offline_suggestion(query: str, df: pd.DataFrame) -> Optional[Dict[str, str]]:
    query_lower = query.lower()
    numeric = list(df.select_dtypes(include="number").columns)
    categorical = [col for col in categorical_columns(df) if df[col].nunique() <= 30]
    datetime_cols = list(df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)

    if not numeric:
        return None

    chosen_numeric = pick_column_from_query(query, numeric) or numeric[0]
    second_numeric = next((col for col in numeric if col != chosen_numeric), "")
    chosen_category = pick_column_from_query(query, categorical) if categorical else None
    chosen_date = pick_column_from_query(query, datetime_cols) if datetime_cols else None

    if any(word in query_lower for word in ["correlation", "correlate", "relationship", "related"]) and second_numeric:
        return {"name": "Relationship", "chart": "scatter", "x": chosen_numeric, "y": second_numeric}

    if any(word in query_lower for word in ["compare", "category", "group", "average", "mean", "by"]) and chosen_category:
        return {"name": "Category comparison", "chart": "bar", "x": chosen_category, "y": chosen_numeric}

    if any(word in query_lower for word in ["spread", "outlier", "box"]) and chosen_category:
        return {"name": "Spread by category", "chart": "box", "x": chosen_category, "y": chosen_numeric}

    if any(word in query_lower for word in ["trend", "time", "date", "over time"]) and chosen_date:
        return {"name": "Trend", "chart": "line", "x": chosen_date, "y": chosen_numeric}

    if any(word in query_lower for word in ["heatmap", "correlation"]) and len(numeric) >= 2:
        return {"name": "Correlation", "chart": "heatmap", "x": "", "y": ""}

    return {"name": "Distribution", "chart": "histogram", "x": chosen_numeric, "y": ""}


def run_offline_analysis(query: str, df: pd.DataFrame) -> Tuple[Optional[List[Any]], str, Dict[str, Any], Optional[Dict[str, str]]]:
    suggestion = build_offline_suggestion(query, df)
    if not suggestion:
        return None, "Offline mode could not find enough numeric data to create a chart.", {
            "success": False,
            "error": "Offline mode needs at least one numeric column for automatic charting.",
            "duration": 0,
            "results_count": 0,
            "offline": True,
        }, None

    response = (
        "Groq is not reachable, so the app used offline analysis instead.\n\n"
        f"Selected chart: {suggestion['name']} ({suggestion['chart']})\n\n"
        f"Takeaway: {chart_takeaway(df, suggestion)}"
    )
    metadata = {
        "success": True,
        "error": "",
        "duration": 0,
        "results_count": 1,
        "offline": True,
        "code": "Offline rule-based chart recommendation; no LLM code was executed.",
    }
    return [{"type": "offline_chart", "suggestion": suggestion}], response, metadata, suggestion


def dataset_context(datasets: Dict[str, pd.DataFrame], profile: Dict[str, Any], preprocessing_steps: List[str]) -> str:
    lines = ["Uploaded CSV schema:"]
    for name, df in datasets.items():
        lines.append(f"- File: {name}")
        lines.append(f"- Shape: {len(df)} rows x {len(df.columns)} columns")
        lines.append(f"- Columns: {', '.join(map(str, df.columns))}")

    overview = profile["overview"]
    lines.append("Dataset profile:")
    lines.append(f"- Rows: {overview['rows']}, Columns: {overview['columns']}, Duplicates: {overview['duplicates']}")
    lines.append(f"- Numeric columns: {', '.join(overview['numeric_columns']) or 'None'}")
    lines.append(f"- Categorical columns: {', '.join(overview['categorical_columns']) or 'None'}")
    lines.append(f"- Date-like columns: {', '.join(overview['datetime_candidates']) or 'None'}")
    lines.append("Preprocessing already applied:")
    lines.extend(f"- {step}" for step in preprocessing_steps)
    return "\n".join(lines)


def validate_generated_code(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Generated code has a syntax error: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif node.module:
                modules = [node.module.split(".")[0]]
            for module in modules:
                if module in BANNED_IMPORT_ROOTS or module not in SAFE_IMPORT_ROOTS:
                    return False, f"Blocked unsafe import: {module}"

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_NAMES:
                return False, f"Blocked unsafe function call: {node.func.id}"
            if isinstance(node.func, ast.Attribute) and node.func.attr in BANNED_ATTRS:
                return False, f"Blocked unsafe method call: {node.func.attr}"

        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, "Blocked dunder attribute access."

    return True, ""


def remap_columns_in_code(code: str, dataset_path: str) -> str:
    columns = list(pd.read_csv(dataset_path, nrows=0).columns)
    lowered = {col.lower(): col for col in columns}
    patterns = [
        r"df\s*\[\s*['\"]([^'\"]+)['\"]\s*\]",
        r"groupby\(\s*['\"]([^'\"]+)['\"]\s*\)",
        r"by\s*=\s*['\"]([^'\"]+)['\"]",
        r"x\s*=\s*['\"]([^'\"]+)['\"]",
        r"y\s*=\s*['\"]([^'\"]+)['\"]",
    ]

    referenced = set()
    for pat in patterns:
        referenced.update(re.findall(pat, code))

    replacements = {}
    for name in referenced:
        if name in columns:
            continue
        lowered_match = lowered.get(name.lower())
        if lowered_match:
            replacements[name] = lowered_match
            continue
        close = difflib.get_close_matches(name, columns, n=1, cutoff=0.8)
        if close:
            replacements[name] = close[0]

    remapped = code
    for src, dest in replacements.items():
        remapped = re.sub(rf"(['\"])({re.escape(src)})(['\"])", rf"\1{dest}\3", remapped)
    return remapped


def score_output_quality(
    query: str,
    code: str,
    results: Optional[List[Any]],
    execution: Dict[str, Any],
    df: pd.DataFrame,
    api_key: str,
    model_name: str,
) -> Dict[str, Any]:
    """Compute per-query output quality metrics."""
    scores: Dict[str, Any] = {}
    actual_cols = list(df.columns)

    # ── 1. Execution error rate (this run: 0 = error, 1 = clean) ─────────
    scores["exec_ok"] = 1 if execution.get("success") else 0
    scores["exec_error"] = execution.get("error", "") or ""

    # ── 2. Column relevance: referenced cols that don't exist ────────────
    patterns = [
        r"df\s*\[\s*['\"]([^'\"]+)['\"]\s*\]",
        r"groupby\(\s*['\"]([^'\"]+)['\"]\s*\)",
        r"x\s*=\s*['\"]([^'\"]+)['\"]",
        r"y\s*=\s*['\"]([^'\"]+)['\"]",
        r"hue\s*=\s*['\"]([^'\"]+)['\"]",
        r"by\s*=\s*['\"]([^'\"]+)['\"]",
    ]
    referenced = set()
    for pat in patterns:
        referenced.update(re.findall(pat, code or ""))
    valid_refs   = [c for c in referenced if c in actual_cols]
    invalid_refs = [c for c in referenced if c not in actual_cols]
    col_relevance = round(100 * len(valid_refs) / max(len(referenced), 1), 0) if referenced else 100
    scores["col_relevance"]  = int(col_relevance)
    scores["hallucinated_cols"] = invalid_refs

    # ── 3. Data coverage: rows actually used vs total ─────────────────────
    # Infer from code: if filtering/slicing found, coverage may be partial
    has_filter = bool(re.search(r"\[.*[<>=!]=?\s*\d", code or ""))
    has_head   = bool(re.search(r"\.head\s*\(\s*\d", code or ""))
    has_sample = bool(re.search(r"\.sample\s*\(", code or ""))
    if has_head:
        # Extract the n from .head(n) to estimate coverage
        m = re.search(r"\.head\s*\(\s*(\d+)", code or "")
        n = int(m.group(1)) if m else 10
        coverage = round(100 * min(n, len(df)) / max(len(df), 1), 1)
    elif has_sample:
        m = re.search(r"\.sample\s*\(\s*(\d+)", code or "")
        n = int(m.group(1)) if m else len(df)
        coverage = round(100 * min(n, len(df)) / max(len(df), 1), 1)
    elif has_filter:
        coverage = None  # unknown — filter applied
    else:
        coverage = 100.0
    scores["data_coverage"] = coverage  # None = filtered/unknown

    # ── 4. Chart type appropriateness (rule-based) ────────────────────────
    q_lower = query.lower()
    code_lower = (code or "").lower()
    chart_score = None
    chart_note  = ""
    time_keywords    = ["trend", "over time", "time series", "monthly", "daily", "yearly", "temporal"]
    corr_keywords    = ["correlation", "relationship", "scatter", "vs ", "versus"]
    dist_keywords    = ["distribution", "spread", "histogram", "skew"]
    compare_keywords = ["compare", "difference", "group", "by category", "per ", "breakdown"]
    if any(k in q_lower for k in time_keywords):
        used_line = any(k in code_lower for k in ["plot_date", "lineplot", "line(", ".plot(", "kind='line'", 'kind="line"'])
        chart_score = 100 if used_line else 40
        chart_note  = "line/time chart" if used_line else "expected line chart for time query"
    elif any(k in q_lower for k in corr_keywords):
        used_scatter = any(k in code_lower for k in ["scatter", "regplot", "lmplot", "pairplot"])
        chart_score  = 100 if used_scatter else 50
        chart_note   = "scatter/regression" if used_scatter else "expected scatter for correlation query"
    elif any(k in q_lower for k in dist_keywords):
        used_dist = any(k in code_lower for k in ["hist", "histplot", "distplot", "kdeplot", "boxplot", "violinplot"])
        chart_score = 100 if used_dist else 50
        chart_note  = "histogram/KDE/box" if used_dist else "expected distribution chart"
    elif any(k in q_lower for k in compare_keywords):
        used_bar = any(k in code_lower for k in ["barplot", "bar(", "kind='bar'", 'kind="bar"', "countplot"])
        chart_score = 100 if used_bar else 60
        chart_note  = "bar/count chart" if used_bar else "expected bar chart for comparison query"
    scores["chart_type_score"] = chart_score
    scores["chart_type_note"]  = chart_note

    # ── 5. Answer relevance via LLM-as-judge (Groq call) ─────────────────
    scores["relevance_score"] = None
    scores["relevance_reason"] = ""
    if api_key and execution.get("success") and code:
        try:
            client = Groq(api_key=api_key, timeout=15.0, max_retries=0)
            judge_prompt = (
                f"You are an expert data analyst evaluating an AI-generated visualization.\n\n"
                f"User question: \"{query}\"\n\n"
                f"Generated Python code summary (first 800 chars):\n{code[:800]}\n\n"
                f"Score how well this code answers the user's question on a scale of 1-5:\n"
                f"5 = directly and completely answers the question\n"
                f"4 = mostly answers with minor gaps\n"
                f"3 = partially answers or uses a suboptimal approach\n"
                f"2 = loosely related but misses the core intent\n"
                f"1 = does not answer the question at all\n\n"
                f"Reply with ONLY a JSON object: {{\"score\": <1-5>, \"reason\": \"<one sentence>\"}}"
            )
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                max_tokens=120,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)
            scores["relevance_score"]  = int(parsed.get("score", 0))
            scores["relevance_reason"] = str(parsed.get("reason", ""))
        except Exception:
            pass

    return scores


def save_datasets_to_temp(datasets: Dict[str, pd.DataFrame]) -> Tuple[str, Dict[str, str]]:
    tmp_dir = tempfile.mkdtemp(prefix="ai_viz_data_")
    paths = {}
    for name, df in datasets.items():
        safe_name = os.path.basename(name)
        path = os.path.join(tmp_dir, safe_name)
        df.to_csv(path, index=False)
        paths[name] = path
    return tmp_dir, paths


def code_interpret(code: str, datasets: Dict[str, pd.DataFrame], primary_name: str) -> Tuple[Optional[List[Any]], Dict[str, Any]]:
    started = time.time()
    validation_ok, validation_message = validate_generated_code(code)
    if not validation_ok:
        return None, {
            "success": False,
            "error": validation_message,
            "duration": 0,
            "code": code,
            "results_count": 0,
        }

    with st.spinner("Executing validated analysis code locally..."):
        tmp_dir, dataset_paths = save_datasets_to_temp(datasets)
        try:
            primary_path = dataset_paths[primary_name]
            dataset_basename = os.path.basename(primary_path)
            code = remap_columns_in_code(code, primary_path)

            augmented_code = code + "\n\n"
            augmented_code += "# Auto-save all open matplotlib figures.\n"
            augmented_code += "import matplotlib\n"
            augmented_code += "matplotlib.use('Agg')\n"
            augmented_code += "import matplotlib.pyplot as plt\n"
            augmented_code += "_figs = [plt.figure(n) for n in plt.get_fignums()]\n"
            augmented_code += "for _i, _fig in enumerate(_figs):\n"
            augmented_code += f"    _fig.savefig(r'{tmp_dir}' + f'/plot_{{_i}}.png', dpi=150, bbox_inches='tight')\n"
            augmented_code += "plt.close('all')\n"
            augmented_code = augmented_code.replace("'./", f"r'{tmp_dir}/")
            augmented_code = augmented_code.replace('"./', f'r"{tmp_dir}/')

            script_path = os.path.join(tmp_dir, "analysis_script.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(augmented_code)

            result = subprocess.run(
                [sys.executable, script_path],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            results = []
            error_text = ""
            if result.returncode != 0:
                error_lines = [
                    line for line in result.stderr.strip().split("\n")
                    if not any(w in line.lower() for w in ["userwarning", "futurewarning", "deprecationwarning"])
                ]
                error_text = "\n".join(error_lines[-10:])

            if result.stdout.strip():
                results.append({"type": "text", "content": result.stdout.strip()})

            for plot_file in sorted(glob.glob(os.path.join(tmp_dir, "plot_*.png"))):
                with open(plot_file, "rb") as image_file:
                    encoded = base64.b64encode(image_file.read()).decode("utf-8")
                results.append({"type": "image", "path": plot_file, "base64": encoded})

            duration = round(time.time() - started, 2)
            success = result.returncode == 0 and bool(results)
            metadata = {
                "success": success,
                "error": error_text,
                "duration": duration,
                "code": code,
                "results_count": len(results),
                "primary_file": dataset_basename,
            }
            return (results if results else None), metadata

        except subprocess.TimeoutExpired:
            return None, {
                "success": False,
                "error": "Code execution timed out after 60 seconds.",
                "duration": round(time.time() - started, 2),
                "code": code,
                "results_count": 0,
            }
        except Exception as exc:
            return None, {
                "success": False,
                "error": f"Execution error: {exc}",
                "duration": round(time.time() - started, 2),
                "code": code,
                "results_count": 0,
            }
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def match_code_blocks(llm_response: str) -> str:
    match = CODE_BLOCK_PATTERN.search(llm_response)
    return match.group(1) if match else ""


def build_messages(
    user_message: str,
    datasets: Dict[str, pd.DataFrame],
    primary_name: str,
    profile: Dict[str, Any],
    preprocessing_steps: List[str],
    retry_error: str = "",
) -> List[Dict[str, str]]:
    primary_file = os.path.basename(primary_name)
    context = dataset_context(datasets, profile, preprocessing_steps)
    history = st.session_state.get("chat_history", [])[-6:]
    history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history)

    system_prompt = f"""You are a careful Python data scientist and visualization expert.
You will answer the user by writing one Python code block that analyzes the uploaded CSV file.

Uploaded dataset: {primary_file}

{context}

Rules:
1. Load the uploaded dataset with pd.read_csv('./{primary_file}').
2. Analyze only this uploaded CSV. Do not assume any other local or uploaded files exist.
3. Use pandas, numpy, matplotlib, seaborn, or plotly only.
4. Print concise statistical findings and interpretation with print().
5. Prefer useful charts: histograms, bar charts, box plots, scatter plots, line charts, and heatmaps.
6. Always use exact column names from the schema. If uncertain, print available columns and choose the closest reasonable column.
7. Avoid file operations except pd.read_csv for the uploaded CSV file.
8. Do not use os, sys, subprocess, requests, eval, exec, open, or shell commands.
9. Keep code linear, explicit, and robust to missing values.
10. Return ONLY one Python code block wrapped in ```python ... ```.
"""

    if history_text:
        system_prompt += f"\nRecent conversation for follow-up context:\n{history_text}\n"
    if retry_error:
        system_prompt += f"\nThe previous code failed with this error. Fix it:\n{retry_error}\n"

    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]


def describe_connection_error(exc: BaseException) -> str:
    details = []
    current: Optional[BaseException] = exc
    while current is not None and len(details) < 4:
        message = str(current).strip()
        if message and message not in details:
            details.append(message)
        current = current.__cause__ or current.__context__

    detail_text = " | ".join(details) if details else exc.__class__.__name__
    guidance = "Check your internet connection, VPN/proxy, firewall, or security software."
    if "10013" in detail_text:
        guidance = (
            "Windows blocked the Python process from opening the socket. "
            "Allow python.exe/streamlit.exe in Windows Firewall or your antivirus/web protection."
        )
    elif "timed out" in detail_text.lower() or "timeout" in detail_text.lower():
        guidance = "The request timed out. Check VPN/proxy/firewall rules or try again on another network."

    return f"Could not reach Groq. Details: {detail_text}. {guidance}"


def chat_with_llm(
    user_message: str,
    datasets: Dict[str, pd.DataFrame],
    primary_name: str,
    profile: Dict[str, Any],
    preprocessing_steps: List[str],
) -> Tuple[Optional[List[Any]], str, Dict[str, Any]]:
    try:
        client = Groq(api_key=st.session_state.groq_api_key, timeout=30.0, max_retries=1)
    except Exception as exc:
        return None, "", {
            "success": False,
            "error": f"Could not initialize Groq client: {exc}",
            "duration": 0,
            "results_count": 0,
        }

    last_response = ""
    last_metadata = {"success": False, "error": "No response generated.", "duration": 0, "results_count": 0}

    for attempt in range(2):
        retry_error = last_metadata.get("error", "") if attempt else ""
        messages = build_messages(user_message, datasets, primary_name, profile, preprocessing_steps, retry_error)
        with st.spinner("Asking Groq to generate analysis code..." if attempt == 0 else "Retrying with the execution error..."):
            try:
                response = client.chat.completions.create(
                    model=st.session_state.model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=4096,
                )
            except APIConnectionError as exc:
                results, response_text, metadata, _ = run_offline_analysis(user_message, datasets[primary_name])
                metadata["error"] = (
                    "Groq connection failed, so offline analysis was used. "
                    f"{describe_connection_error(exc)}"
                )
                return results, response_text, metadata
            except AuthenticationError:
                return None, "", {
                    "success": False,
                    "error": "Groq rejected the API key. Please check the key in the sidebar and try again.",
                    "duration": 0,
                    "results_count": 0,
                }
            except RateLimitError:
                return None, "", {
                    "success": False,
                    "error": "Groq rate limit reached. Wait a moment or switch to a smaller model, then try again.",
                    "duration": 0,
                    "results_count": 0,
                }
            except APIError as exc:
                return None, "", {
                    "success": False,
                    "error": f"Groq API error: {exc}",
                    "duration": 0,
                    "results_count": 0,
                }
            except Exception as exc:
                return None, "", {
                    "success": False,
                    "error": f"Unexpected Groq error: {exc}",
                    "duration": 0,
                    "results_count": 0,
                }

        last_response = response.choices[0].message.content
        python_code = match_code_blocks(last_response)
        if not python_code:
            last_metadata = {"success": False, "error": "The model did not return a Python code block.", "duration": 0, "results_count": 0}
            continue

        results, metadata = code_interpret(python_code, datasets, primary_name)
        last_metadata = metadata
        if metadata["success"]:
            return results, last_response, metadata

    return None, last_response, last_metadata


def dataframe_to_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def image_result_html(result: Dict[str, Any]) -> str:
    return f'<img src="data:image/png;base64,{result["base64"]}" style="max-width:100%;margin:16px 0;border:1px solid #ddd;" />'


def build_html_report(
    file_name: str,
    query: str,
    profile: Dict[str, Any],
    preprocessing_steps: List[str],
    llm_response: str,
    results: Optional[List[Any]],
    execution: Dict[str, Any],
) -> bytes:
    overview = profile["overview"]
    preprocessing_note = (
        "The generated code was executed against the app's temporary preprocessed CSV. "
        "That temporary file keeps the uploaded filename for pd.read_csv(...), but its "
        "columns, rows, and missing values reflect the preprocessing steps listed above."
    )
    display_code = execution.get("code", "")
    if display_code:
        display_code = (
            "# NOTE: This code was executed against the app's temporary preprocessed CSV.\n"
            "# See the Preprocessing section above for row, column, and missing-value changes.\n\n"
            f"{display_code}"
        )
    result_blocks = []
    for result in results or []:
        if result["type"] == "text":
            result_blocks.append(f"<pre>{html.escape(result['content'])}</pre>")
        elif result["type"] == "image" and result.get("base64"):
            result_blocks.append(image_result_html(result))

    body = f"""
    <html>
    <head>
        <meta charset="utf-8" />
        <title>AI Data Visualization Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.5; margin: 40px; color: #222; }}
            h1, h2 {{ color: #0f172a; }}
            pre {{ background: #f6f8fa; padding: 12px; overflow-x: auto; }}
            .meta {{ color: #475569; }}
            .note {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 10px 12px; }}
        </style>
    </head>
    <body>
        <h1>AI Data Visualization Report</h1>
        <p class="meta">Dataset: {html.escape(file_name)} | Rows: {overview['rows']} | Columns: {overview['columns']}</p>
        <h2>Question</h2>
        <p>{html.escape(query)}</p>
        <h2>Preprocessing</h2>
        <ul>{''.join(f'<li>{html.escape(step)}</li>' for step in preprocessing_steps)}</ul>
        <h2>Execution</h2>
        <p class="note">{html.escape(preprocessing_note)}</p>
        <p>Success: {execution.get('success')} | Duration: {execution.get('duration')}s | Results: {execution.get('results_count')}</p>
        <h2>Results</h2>
        {''.join(result_blocks) or '<p>No result blocks were produced.</p>'}
        <h2>Generated Code Used for Execution</h2>
        <pre>{html.escape(display_code)}</pre>
        <h2>Model Response</h2>
        <pre>{html.escape(llm_response)}</pre>
    </body>
    </html>
    """
    return body.encode("utf-8")


def initialize_state():
    defaults = {
        "groq_api_key": os.environ.get("GROQ_API_KEY", ""),
        "model_name": "llama-3.3-70b-versatile",
        "chat_history": [],
        "execution_log": [],
        "latest_report": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_profile(profile: Dict[str, Any]):
    overview = profile["overview"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{overview['rows']:,}")
    c2.metric("Columns", overview["columns"])
    c3.metric("Duplicates", overview["duplicates"])
    c4.metric("Missing Cells", int(profile["missing"]["missing_count"].sum()))

    tab_missing, tab_numeric, tab_categorical = st.tabs(["🔍 Missing", "🔢 Numeric", "🏷 Categorical"])
    with tab_missing:
        st.dataframe(profile["missing"].head(5), use_container_width=True, hide_index=True)
        st.caption("Top 5 columns by missing %")
    with tab_numeric:
        if profile["numeric_summary"].empty:
            st.info("No numeric columns detected.")
        else:
            st.dataframe(profile["numeric_summary"].head(5), use_container_width=True, hide_index=True)
            st.caption("Showing the first 5 numeric columns.")
    with tab_categorical:
        if profile["categorical_summary"].empty:
            st.info("No categorical columns detected.")
        else:
            st.dataframe(profile["categorical_summary"].head(5), use_container_width=True, hide_index=True)
            st.caption("Showing the first 5 categorical columns.")


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ─────────────────────────────────────────── */
:root {
    --bg:        #0d0f14;
    --surface:   #13161d;
    --border:    #1e2330;
    --accent:    #5b8dee;
    --accent2:   #38e8c4;
    --muted:     #6b7280;
    --text:      #e4e8f0;
    --text-dim:  #9aa3b2;
    --danger:    #f87171;
    --success:   #4ade80;
    --warn:      #fb923c;
}

/* ── Global base ──────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stMain"] {
    background: var(--bg) !important;
}

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em !important;
}

[data-testid="stSidebarNav"] { display: none; }

/* ── Main headings ────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.03em !important;
    color: var(--text) !important;
}

/* ── Page title hero ──────────────────────────────────────── */
.hero-block {
    background: linear-gradient(135deg, #131929 0%, #0d1520 60%, #0a0f1a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 36px 40px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-block::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(91,141,238,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-block::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(56,232,196,0.07) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(90deg, #e4e8f0 0%, #5b8dee 60%, #38e8c4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    color: var(--text-dim);
    margin: 0;
    font-weight: 300;
    letter-spacing: 0.01em;
}
.hero-badge {
    display: inline-block;
    background: rgba(91,141,238,0.15);
    border: 1px solid rgba(91,141,238,0.35);
    color: #7aaaf4;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Metric cards ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.03em !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
[data-testid="stButton"] > button,
[data-testid="stFormSubmitButton"] > button {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.18s ease !important;
}
[data-testid="stButton"] > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(91,141,238,0.06) !important;
}

/* Suggestion buttons get a subtle teal tint */
[data-testid="stButton"][key^="query_suggestion"] > button {
    background: rgba(56,232,196,0.04) !important;
    border-color: rgba(56,232,196,0.25) !important;
    color: #7fe8d4 !important;
    font-size: 0.8rem !important;
    text-align: left !important;
}
[data-testid="stButton"][key^="query_suggestion"] > button:hover {
    background: rgba(56,232,196,0.1) !important;
    border-color: var(--accent2) !important;
}

/* Download button accent */
[data-testid="stDownloadButton"] > button {
    background: rgba(91,141,238,0.12) !important;
    border: 1px solid rgba(91,141,238,0.4) !important;
    color: #7aaaf4 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: all 0.18s ease !important;
    width: 100% !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(91,141,238,0.22) !important;
    border-color: var(--accent) !important;
}

/* ── Inputs & selects ─────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stFileUploader"] {
    background: var(--bg) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(91,141,238,0.15) !important;
}

/* ── Chat messages ────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
}
[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(91,141,238,0.12) !important;
}

/* ── Expanders ────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: var(--text) !important;
    letter-spacing: -0.01em !important;
}

/* ── Tabs ─────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--muted) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Dataframes ───────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Alerts / info / warning / error ─────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 3px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.87rem !important;
}

/* ── Divider ──────────────────────────────────────────────── */
hr {
    border-color: var(--border) !important;
    margin: 16px 0 !important;
}

/* ── Section label helper ─────────────────────────────────── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--muted);
    margin-bottom: 10px;
}

/* ── Subheaders ───────────────────────────────────────────── */
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3 {
    color: var(--text) !important;
    font-weight: 700 !important;
}

/* ── Spinner ──────────────────────────────────────────────── */
[data-testid="stSpinner"] {
    color: var(--accent) !important;
}

/* ── Checkbox ─────────────────────────────────────────────── */
[data-testid="stCheckbox"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    color: var(--text-dim) !important;
}
[data-testid="stCheckbox"] input[type="checkbox"]:checked + div {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── Caption / small text ─────────────────────────────────── */
[data-testid="stCaptionContainer"], small, .stCaption {
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
}

/* ── Code blocks ──────────────────────────────────────────── */
code, pre {
    font-family: 'DM Mono', monospace !important;
    background: #0a0d13 !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
}

/* ── Success / error messages in sidebar ─────────────────── */
[data-testid="stAlert"][data-baseweb="notification"] {
    border-radius: 8px !important;
}

/* ── File uploader ────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(91,141,238,0.03) !important;
    border: 1.5px dashed rgba(91,141,238,0.3) !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(91,141,238,0.07) !important;
    border-color: var(--accent) !important;
}

/* ── Markdown body text ───────────────────────────────────── */
[data-testid="stMarkdown"] p,
[data-testid="stMarkdown"] li {
    color: var(--text-dim) !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
}

/* ── Hide Streamlit deploy button & top-right toolbar ─────── */
[data-testid="stDeployButton"],
[data-testid="stToolbar"],
[data-testid="stSidebarHeader"],
[data-testid="stSidebarNavItems"],
#MainMenu,
header[data-testid="stHeader"] {
    display: none !important;
}

/* ── Select dropdown ──────────────────────────────────────── */
[data-testid="stSelectbox"] svg { color: var(--muted) !important; }

/* ── Sidebar: widget labels (Groq API Key, Model, etc.) ───── */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: 0.68rem !important;
    font-family: 'DM Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
    margin-bottom: 3px !important;
}

/* ── Sidebar: input & select values ──────────────────────── */
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    font-size: 0.82rem !important;
    padding: 5px 10px !important;
    min-height: 36px !important;
}

/* ── Sidebar: compact the success/info/warning alert ─────── */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    padding: 6px 10px !important;
    border-radius: 7px !important;
    margin-bottom: 6px !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] p,
[data-testid="stSidebar"] [data-testid="stAlert"] div {
    font-size: 0.73rem !important;
    margin: 0 !important;
    line-height: 1.4 !important;
}

/* ── Sidebar: markdown text and links ────────────────────── */
[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
[data-testid="stSidebar"] [data-testid="stMarkdown"] a {
    font-size: 0.74rem !important;
    line-height: 1.5 !important;
}

/* ── Sidebar: caption ─────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    font-size: 0.68rem !important;
    color: var(--muted) !important;
    margin-top: 2px !important;
}

/* ── Sidebar: checkbox labels ─────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stCheckbox"] label p {
    font-size: 0.79rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    color: var(--text-dim) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Sidebar: button text ─────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stButton"] > button,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button {
    font-size: 0.77rem !important;
    padding: 5px 10px !important;
}

/* ── Sidebar: file uploader — hide the glitched button text,
       keep the dropzone itself clean ───────────────────────── */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] {
    color: transparent !important;
    position: relative !important;
    min-height: 28px !important;
    padding: 4px 16px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] * {
    visibility: hidden !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]::before {
    content: "Upload" !important;
    visibility: visible !important;
    position: absolute !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    font-size: 0.74rem !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-dim) !important;
    white-space: nowrap !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    padding: 10px 10px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span {
    font-size: 0.72rem !important;
    color: var(--muted) !important;
}

/* ── Sidebar: section-label spacing ──────────────────────── */
[data-testid="stSidebar"] .section-label {
    margin-top: 2px !important;
    margin-bottom: 5px !important;
}

/* ── Sidebar: tighten vertical gaps between elements ─────── */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
    gap: 0.4rem !important;
}
[data-testid="stSidebar"] hr {
    margin: 10px 0 !important;
}
</style>
"""


def main():
    st.set_page_config(
        page_title="AI Data Visualization Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_state()

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Hero header
    st.markdown("""
    <div class="hero-block">
        <div class="hero-badge">✦ Powered by Groq</div>
        <h1 class="hero-title">AI Data Visualization Agent</h1>
        <p class="hero-sub">Conversational CSV analysis — profiling, preprocessing, code execution, smart query suggestions & HTML reports.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<p class="section-label">⚙ Configuration</p>', unsafe_allow_html=True)
        env_key = os.environ.get("GROQ_API_KEY", "")
        if env_key:
            st.markdown("""
<div style="display:flex;align-items:center;gap:7px;background:rgba(74,222,128,0.08);
border:1px solid rgba(74,222,128,0.25);border-radius:7px;padding:6px 10px;margin-bottom:6px">
  <span style="font-size:0.75rem">✅</span>
  <span style="font-size:0.7rem;font-family:'DM Mono',monospace;color:#4ade80;letter-spacing:0.04em">
    API key loaded from <code style="background:rgba(74,222,128,0.12);padding:1px 4px;border-radius:3px;font-size:0.68rem;color:#86efac;border:none">.env</code>
  </span>
</div>
<p style="font-size:0.68rem;color:var(--muted);margin:0 0 6px;font-family:'DM Sans',sans-serif">
  Override below for this session.
</p>""", unsafe_allow_html=True)
        else:
            st.markdown('<p style="font-size:0.68rem;color:var(--muted);margin:0 0 6px;font-family:\'DM Sans\',sans-serif">No <code style="font-size:0.65rem">.env</code> key — enter manually.</p>', unsafe_allow_html=True)

        manual_key = st.text_input(
            "Groq API Key",
            type="password",
            value="" if env_key else st.session_state.groq_api_key,
            placeholder="Paste key here to override .env" if env_key else "sk-...",
        )
        # Manual entry always wins; fall back to env key
        st.session_state.groq_api_key = manual_key if manual_key else env_key
        st.markdown("[Get a Groq API key →](https://console.groq.com/)")

        model_options = {
            "Llama 3.3 70B (Recommended)": "llama-3.3-70b-versatile",
            "Llama 4 Scout 17B": "meta-llama/llama-4-scout-17b-16e-instruct",
            "Qwen 2.5 Coder 32B": "qwen-2.5-coder-32b",
            "Gemma 2 9B": "gemma2-9b-it",
        }
        selected_model = st.selectbox("Model", options=list(model_options.keys()), index=0)
        st.session_state.model_name = model_options[selected_model]

        st.divider()
        st.markdown('<p class="section-label">📂 Dataset</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv", accept_multiple_files=False)
        if uploaded_file is not None:
            st.caption(f"📄 {uploaded_file.name}")

    if uploaded_file is None:
        with st.sidebar:
            st.divider()
            st.markdown('<p class="section-label">🔧 Preprocessing</p>', unsafe_allow_html=True)
            st.caption("Upload a dataset to configure preprocessing.")

        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown("""
            <div style="background:#13161d;border:1px solid #1e2330;border-radius:14px;padding:36px 32px;text-align:center;margin-top:20px">
                <div style="font-size:2.4rem;margin-bottom:14px">📊</div>
                <p style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#e4e8f0;margin:0 0 8px;letter-spacing:-0.02em">Upload your dataset</p>
                <p style="font-family:'DM Sans',sans-serif;font-size:0.84rem;color:#6b7280;margin:0">Drop a CSV in the sidebar to start exploring your data with AI.</p>
            </div>
            """, unsafe_allow_html=True)
        with col_r:
            st.markdown("""
            <div style="background:#13161d;border:1px solid #1e2330;border-radius:14px;padding:28px 32px;margin-top:20px">
                <p style="font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:700;color:#e4e8f0;margin:0 0 16px;letter-spacing:-0.02em">What you can do</p>
                <div style="display:flex;flex-direction:column;gap:10px">
                    <div style="display:flex;align-items:center;gap:10px;font-family:'DM Sans',sans-serif;font-size:0.83rem;color:#9aa3b2">
                        <span style="color:#5b8dee;font-size:1rem">🔍</span> Auto-profile your dataset structure
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;font-family:'DM Sans',sans-serif;font-size:0.83rem;color:#9aa3b2">
                        <span style="color:#38e8c4;font-size:1rem">🧹</span> Clean &amp; preprocess with one click
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;font-family:'DM Sans',sans-serif;font-size:0.83rem;color:#9aa3b2">
                        <span style="color:#5b8dee;font-size:1rem">💬</span> Ask questions in plain English
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;font-family:'DM Sans',sans-serif;font-size:0.83rem;color:#9aa3b2">
                        <span style="color:#38e8c4;font-size:1rem">📈</span> Get AI-powered chart suggestions
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;font-family:'DM Sans',sans-serif;font-size:0.83rem;color:#9aa3b2">
                        <span style="color:#5b8dee;font-size:1rem">📥</span> Export full HTML reports
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        return

    primary_name, raw_primary = load_csv_file(uploaded_file)

    # Reserve the Export container in sidebar — visually above Preprocessing
    with st.sidebar:
        export_container = st.container()
        st.divider()
        st.markdown('<p class="section-label">🔧 Preprocessing</p>', unsafe_allow_html=True)
        normalize_columns = st.checkbox("Normalize column names", value=True)
        remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
        fill_missing = st.checkbox("Fill missing values", value=True)
        parse_dates = st.checkbox("Parse date-like columns", value=True)
        clip_outliers = st.checkbox("Clip numeric outliers", value=False)

    processed_primary, preprocessing_steps = preprocess_dataset(
        raw_primary,
        normalize_columns,
        remove_duplicates,
        fill_missing,
        parse_dates,
        clip_outliers,
    )

    processed_datasets = {primary_name: processed_primary}
    profile = profile_dataset(processed_primary)

    # Fill the reserved Export container with the download button
    with export_container:
        st.divider()
        st.markdown('<p class="section-label">📤 Export & Tools</p>', unsafe_allow_html=True)
        st.download_button(
            "📥 Download preprocessed CSV",
            dataframe_to_download(processed_primary),
            file_name=f"preprocessed_{os.path.basename(primary_name)}",
            mime="text/csv",
            use_container_width=True,
        )

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑 Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.execution_log = []
                st.session_state.latest_report = None
                st.rerun()
        with col_b:
            if st.button("🔌 Test API", use_container_width=True):
                if not st.session_state.groq_api_key:
                    st.warning("Enter your Groq API key first.")
                else:
                    try:
                        client = Groq(api_key=st.session_state.groq_api_key, timeout=15.0, max_retries=0)
                        client.models.list()
                        st.success("Connected ✓")
                    except APIConnectionError as exc:
                        st.error(describe_connection_error(exc))
                    except AuthenticationError:
                        st.error("Invalid API key.")
                    except RateLimitError:
                        st.error("Rate limit reached.")
                    except APIError as exc:
                        st.error(f"API error: {exc}")
                    except Exception as exc:
                        st.error(f"Failed: {exc}")

    main_col, metrics_col = st.columns([3, 1], gap="medium")

    # ── Right panel: Dataset Insights ─────────────────────────────────────
    with metrics_col:
        st.markdown('<p class="section-label" style="margin-top:8px">📊 Dataset Insights</p>', unsafe_allow_html=True)

        ov            = profile["overview"]
        num_summary   = profile["numeric_summary"]
        num_cols      = ov.get("numeric_columns", [])
        cat_cols      = ov.get("categorical_columns", [])
        dt_cols       = ov.get("datetime_candidates", [])
        total_cells   = ov["rows"] * ov["columns"] if ov["columns"] > 0 else 1
        missing_cells = int(profile["missing"]["missing_count"].sum()) if not profile["missing"].empty else 0
        dup_rows      = ov.get("duplicates", 0)

        # ── Quality metrics ──────────────────────────────────────────────
        completeness = round(100 * (1 - missing_cells / max(total_cells, 1)), 1)
        uniqueness   = round(100 * (1 - dup_rows / max(ov["rows"], 1)), 1)
        health       = round((completeness + uniqueness) / 2, 1)

        # ── Skewness: % of numeric cols that are highly skewed (|skew|>1) ─
        skewed_cols = 0
        if not num_summary.empty and "50%" in num_summary.columns and "mean" in num_summary.columns:
            for _, row in num_summary.iterrows():
                try:
                    mean, p50 = float(row["mean"]), float(row["50%"])
                    std = float(row["std"]) if "std" in row else 0
                    if std > 0:
                        skew_approx = 3 * (mean - p50) / std
                        if abs(skew_approx) > 1:
                            skewed_cols += 1
                except Exception:
                    pass
        skew_pct = round(100 * skewed_cols / max(len(num_cols), 1), 0) if num_cols else 0

        # ── Outliers: avg % of values beyond 3 std per numeric col ───────
        outlier_pct = 0.0
        if not num_summary.empty and "mean" in num_summary.columns and "std" in num_summary.columns:
            outlier_fracs = []
            for _, row in num_summary.iterrows():
                try:
                    col_name = row["column"] if "column" in row.index else None
                    if col_name and col_name in processed_primary.columns:
                        m, s = float(row["mean"]), float(row["std"])
                        if s > 0:
                            frac = ((processed_primary[col_name] - m).abs() > 3 * s).mean()
                            outlier_fracs.append(frac)
                except Exception:
                    pass
            if outlier_fracs:
                outlier_pct = round(100 * sum(outlier_fracs) / len(outlier_fracs), 1)

        # ── High-cardinality & visualizable columns ───────────────────────
        cat_summary  = profile["categorical_summary"]
        high_card = 0
        if not cat_summary.empty and "unique_values" in cat_summary.columns:
            high_card = int((cat_summary["unique_values"] > max(ov["rows"] * 0.5, 10)).sum())
        low_card_cat = len(cat_cols) - high_card
        viz_cols     = len(num_cols) + low_card_cat

        # ── Card rendering helpers ────────────────────────────────────────
        def section_label(text):
            st.markdown(
                f'<div style="font-size:0.62rem;font-family:\'DM Mono\',monospace;text-transform:uppercase;'
                f'letter-spacing:0.1em;color:#4b5563;margin:14px 0 5px">{text}</div>',
                unsafe_allow_html=True,
            )

        def metric_card(label, value, sub=None, color="#4ade80"):
            sub_html = f'<div style="font-size:0.62rem;color:#6b7280;margin-top:3px;line-height:1.4">{sub}</div>' if sub else ""
            st.markdown(f"""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
border-radius:10px;padding:11px 13px;margin-bottom:7px">
  <div style="font-size:0.6rem;font-family:'DM Mono',monospace;text-transform:uppercase;
  letter-spacing:0.08em;color:#6b7280;margin-bottom:3px">{label}</div>
  <div style="font-size:1.35rem;font-weight:700;color:{color};line-height:1.1">{value}</div>
  {sub_html}
</div>""", unsafe_allow_html=True)

        def progress_card(label, value, decimals=0):
            bar_color = "#4ade80" if value >= 80 else "#facc15" if value >= 50 else "#f87171"
            fmt = f"{value:.{decimals}f}%"
            st.markdown(f"""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
border-radius:10px;padding:11px 13px;margin-bottom:7px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
    <span style="font-size:0.6rem;font-family:'DM Mono',monospace;text-transform:uppercase;
    letter-spacing:0.08em;color:#6b7280">{label}</span>
    <span style="font-size:0.82rem;font-weight:700;color:{bar_color}">{fmt}</span>
  </div>
  <div style="background:rgba(255,255,255,0.07);border-radius:99px;height:4px;overflow:hidden">
    <div style="width:{min(value,100)}%;height:100%;background:{bar_color};border-radius:99px"></div>
  </div>
</div>""", unsafe_allow_html=True)

        def badge_card(label, items, color="#60a5fa"):
            badges = "".join(
                f'<span style="background:rgba(96,165,250,0.1);border:1px solid rgba(96,165,250,0.2);'
                f'border-radius:5px;padding:2px 7px;font-size:0.65rem;color:{color};margin:2px 2px 0 0;'
                f'display:inline-block">{item}</span>'
                for item in items
            )
            st.markdown(f"""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
border-radius:10px;padding:11px 13px;margin-bottom:7px">
  <div style="font-size:0.6rem;font-family:'DM Mono',monospace;text-transform:uppercase;
  letter-spacing:0.08em;color:#6b7280;margin-bottom:6px">{label}</div>
  <div style="line-height:1.8">{badges}</div>
</div>""", unsafe_allow_html=True)

        # ── 1. Dataset Quality ────────────────────────────────────────────
        section_label("① Dataset Quality")
        progress_card("Completeness", completeness)
        progress_card("Row Uniqueness", uniqueness)
        progress_card("Health Score", health)

        # ── 2. Data Structure ─────────────────────────────────────────────
        section_label("② Data Structure")
        metric_card(
            "Column Breakdown",
            f"{len(num_cols)}N · {len(cat_cols)}C · {len(dt_cols)}T",
            f"{len(num_cols)} numeric · {len(cat_cols)} categorical · {len(dt_cols)} datetime",
            "#a78bfa",
        )
        metric_card(
            "Visualizable Cols",
            str(viz_cols),
            f"{len(num_cols)} numeric + {low_card_cat} low-cardinality categorical",
            "#60a5fa",
        )
        metric_card(
            "High-Cardinality Cols",
            str(high_card),
            "categorical cols with >50% unique values — poor for grouping" if high_card else "none — all categoricals are groupable",
            "#f87171" if high_card > 0 else "#4ade80",
        )
        metric_card(
            "Skewed Numeric Cols",
            f"{skewed_cols} / {len(num_cols)}",
            f"{int(skew_pct)}% of numeric cols are heavily skewed — affects chart scales",
            "#facc15" if skewed_cols > 0 else "#4ade80",
        )
        metric_card(
            "Avg Outlier Rate",
            f"{outlier_pct}%",
            "values beyond 3σ per numeric column",
            "#f87171" if outlier_pct > 5 else "#facc15" if outlier_pct > 1 else "#4ade80",
        )

        # ── 3. Output Quality (per-query, from execution log) ─────────────
        section_label("③ Output Quality")
        logs = st.session_state.execution_log
        total_q = len(logs)

        if total_q == 0:
            st.markdown(
                '<div style="font-size:0.72rem;color:#4b5563;font-style:italic;padding:6px 0 10px">'
                'Run a query to see output quality metrics.</div>',
                unsafe_allow_html=True,
            )
        else:
            # Execution error rate
            exec_ok_rate = round(100 * sum(l.get("q_exec_ok", 1) for l in logs) / total_q, 0)
            progress_card("Execution Success Rate", exec_ok_rate)

            # Column hallucination rate (avg % of valid refs across runs)
            col_rel_scores = [l.get("q_col_relevance") for l in logs if l.get("q_col_relevance") is not None]
            if col_rel_scores:
                avg_col_rel = round(sum(col_rel_scores) / len(col_rel_scores), 0)
                progress_card("Column Relevance", avg_col_rel)
                # Show last run's hallucinated cols if any
                last_hallucinated = logs[-1].get("q_hallucinated_cols", [])
                if last_hallucinated:
                    badge_card("Hallucinated Cols (last run)", last_hallucinated, "#f87171")

            # Data coverage — last run
            last_coverage = logs[-1].get("q_data_coverage")
            if last_coverage is not None:
                cov_color = "#4ade80" if last_coverage >= 80 else "#facc15" if last_coverage >= 30 else "#f87171"
                metric_card(
                    "Data Coverage (last run)",
                    f"{last_coverage}%",
                    "rows used vs total dataset size",
                    cov_color,
                )
            else:
                metric_card("Data Coverage (last run)", "filtered", "query applied row filter — coverage unknown", "#facc15")

            # Chart type appropriateness — last run
            last_chart_score = logs[-1].get("q_chart_type_score")
            last_chart_note  = logs[-1].get("q_chart_type_note", "")
            if last_chart_score is not None:
                progress_card("Chart Type Match (last run)", last_chart_score)
                if last_chart_note:
                    st.markdown(
                        f'<div style="font-size:0.62rem;color:#6b7280;margin:-4px 0 8px;padding:0 2px">'
                        f'{last_chart_note}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                metric_card("Chart Type Match", "—", "query intent unclear for rule check", "#6b7280")

            # Answer relevance — LLM judge score
            rel_scores = [l.get("q_relevance_score") for l in logs if l.get("q_relevance_score") is not None]
            if rel_scores:
                avg_rel = round(sum(rel_scores) / len(rel_scores), 1)
                rel_color = "#4ade80" if avg_rel >= 4 else "#facc15" if avg_rel >= 3 else "#f87171"
                last_reason = logs[-1].get("q_relevance_reason", "")
                metric_card(
                    "Answer Relevance (LLM judge)",
                    f"{avg_rel} / 5",
                    f'Last: "{last_reason}"' if last_reason else f"avg across {len(rel_scores)} scored runs",
                    rel_color,
                )
            else:
                metric_card("Answer Relevance", "—", "scoring in progress…", "#6b7280")

    # ── Left: main content ──────────────────────────────────────────────────
    with main_col:

        # Dataset preview with styled header
        st.markdown('<p class="section-label" style="margin-top:8px">📋 Dataset Preview</p>', unsafe_allow_html=True)
        st.dataframe(processed_primary.head(5), use_container_width=True)

        with st.expander("📊 Dataset Profile", expanded=True):
            render_profile(profile)

        # --- AI Query Suggestions (replaces AI Recommended Interactive Charts) ---
        with st.expander("💡 Suggested Queries", expanded=True):
            data_hash = int(pd.util.hash_pandas_object(processed_primary, index=True).sum())
            suggestion_key = (
                primary_name,
                data_hash,
                tuple(map(str, processed_primary.columns)),
                st.session_state.model_name,
                bool(st.session_state.groq_api_key),
            )
            if (
                st.session_state.get("query_suggestion_key") != suggestion_key
                or "query_suggestions" not in st.session_state
            ):
                with st.spinner("Analyzing dataset to suggest queries..."):
                    query_suggestions, suggestion_note = generate_query_suggestions(
                        processed_primary,
                        profile,
                        st.session_state.groq_api_key,
                        st.session_state.model_name,
                    )
                st.session_state.query_suggestion_key = suggestion_key
                st.session_state.query_suggestions = query_suggestions
                st.session_state.query_suggestion_note = suggestion_note

            query_suggestions = st.session_state.query_suggestions
            st.caption(st.session_state.get("query_suggestion_note", ""))
            if not query_suggestions:
                st.info("No query suggestions available for this dataset yet.")
            else:
                st.markdown("Click any suggestion below to use it as your query:")
                cols = st.columns(2)
                for idx, suggestion_text in enumerate(query_suggestions):
                    col = cols[idx % 2]
                    with col:
                        if st.button(
                            f"🔍 {suggestion_text}",
                            key=f"query_suggestion_{idx}",
                            use_container_width=True,
                        ):
                            st.session_state["prefilled_query"] = suggestion_text
                            st.rerun()

        st.markdown('<p class="section-label" style="margin-top:8px">💬 Ask the Agent</p>', unsafe_allow_html=True)
        for item in st.session_state.chat_history:
            with st.chat_message(item["role"]):
                st.write(item["content"])
                # Replay persisted results (charts / tables)
                if item.get("results"):
                    st.markdown('<p class="section-label" style="margin-top:8px">📈 Results</p>', unsafe_allow_html=True)
                    for result in item["results"]:
                        if result["type"] == "image":
                            image = Image.open(BytesIO(base64.b64decode(result["base64"])))
                            st.image(image, caption="Generated visualization", width="stretch")
                        elif result["type"] == "text":
                            st.code(result["content"], language="text")
                        elif result["type"] == "offline_chart":
                            render_recommended_chart(processed_primary, result["suggestion"])
                if item.get("llm_response"):
                    with st.expander("🤖 AI Response & Generated Code", expanded=False):
                        st.markdown(item["llm_response"])


        # Check for prefilled query from suggestion buttons
        prefilled = st.session_state.pop("prefilled_query", None)
        query = st.chat_input("Ask a follow-up or request a new visualization")
        if prefilled:
            query = prefilled
        if query:
            if not st.session_state.groq_api_key:
                st.error("Please enter your Groq API key in the sidebar.")
                return

            st.session_state.chat_history.append({"role": "user", "content": query})
            results, llm_response, execution = chat_with_llm(query, processed_datasets, primary_name, profile, preprocessing_steps)

            # Score output quality
            quality = score_output_quality(
                query=query,
                code=execution.get("code", ""),
                results=results,
                execution=execution,
                df=processed_primary,
                api_key=st.session_state.groq_api_key,
                model_name=st.session_state.model_name,
            )

            st.session_state.execution_log.append(
                {
                    "query": query,
                    "success": execution.get("success"),
                    "duration": execution.get("duration"),
                    "results_count": execution.get("results_count"),
                    "error": execution.get("error", ""),
                    **{f"q_{k}": v for k, v in quality.items()},
                }
            )

            if execution.get("success"):
                msg = ("Groq was unavailable, so I used offline analysis. Results are shown below."
                       if execution.get("offline") else "Analysis completed. Results are shown below.")
            else:
                msg = f"Analysis failed: {execution.get('error')}"

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": msg,
                "results": results if execution.get("success") and results else [],
                "llm_response": llm_response,
            })

            if execution.get("error") and not execution.get("success"):
                st.session_state["_last_error"] = execution["error"]

            st.session_state.latest_report = build_html_report(
                primary_name,
                query,
                profile,
                preprocessing_steps,
                llm_response,
                results,
                execution,
            )
            st.rerun()

        # Show any error from last run
        if st.session_state.pop("_last_error", None):
            st.error(st.session_state.get("_last_error", ""))

        if st.session_state.latest_report:
            st.download_button(
                "📄 Download latest HTML report",
                st.session_state.latest_report,
                file_name="ai_data_visualization_report.html",
                mime="text/html",
            )

        with st.expander("🗂 Execution Log"):
            if st.session_state.execution_log:
                st.dataframe(pd.DataFrame(st.session_state.execution_log), use_container_width=True, hide_index=True)
            else:
                st.info("No analysis runs yet.")


if __name__ == "__main__":
    main()