"""
helpers.py — Data loading, feature engineering, and shared utilities.

All heavy transformations live here so app.py stays clean and declarative.
ABCflix Subscriber Growth Analytics Dashboard.
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ── Constants ──────────────────────────────────────────────────────────────────
ANALYSIS_DATE = pd.Timestamp("2025-03-01")

# ── Semantic Color System ──────────────────────────────────────────────────────
# Designed around color psychology for product analytics:
#   Blue   = neutral / analytical (default for most charts)
#   Green  = positive / healthy / active
#   Red    = negative / churn / risk (ONLY for warning states)
#   Purple = premium / monetization / LTV
#   Amber  = cautionary / secondary emphasis
#   Teal   = referral / organic growth
#   Gray   = neutral text / low emphasis

COLORS = {
    # ── Semantic roles ─────────────────────────────────────
    "primary":        "#4F8EF7",   # Primary analytical blue
    "primary_light":  "#7AABFF",   # Lighter blue for hover / secondary
    "secondary":      "#2EC5FF",   # Cyan-blue for secondary analytical
    "positive":       "#2ECC71",   # Green — active, healthy, strong
    "positive_dim":   "#27AE60",   # Darker green for contrast
    "negative":       "#FF5A5F",   # Red — churn, risk, drop-off ONLY
    "negative_dark":  "#D94045",   # Darker red for emphasis
    "premium":        "#A66CFF",   # Purple — LTV, monetization, premium
    "premium_dim":    "#8B5CF6",   # Deeper purple
    "caution":        "#F5B041",   # Amber — cautionary, secondary
    "teal":           "#2EC4B6",   # Teal — referral, organic
    "coral":          "#FF8A65",   # Soft coral — warm accent
    "slate":          "#7C8DB5",   # Muted blue-gray

    # ── Neutral tones ──────────────────────────────────────
    "text_primary":   "#F5F7FA",   # Off-white for KPI values (default)
    "text_secondary": "#9AA4B2",   # Muted gray for labels / subtext
    "text_muted":     "#636E7B",   # Dimmer gray for sub-labels
    "surface":        "#141414",   # Card backgrounds
    "border":         "#2A2A2A",   # Borders
    "accent_border":  "#3A3F4B",   # Hover / accent border (subtle)

    # ── Legacy aliases (for any remaining references) ──────
    "red":       "#FF5A5F",
    "dark_red":  "#D94045",
    "green":     "#2ECC71",
    "amber":     "#F5B041",
    "blue":      "#4F8EF7",
    "purple":    "#A66CFF",
    "cyan":      "#2EC5FF",
}

# ── Categorical palettes ──────────────────────────────────────────────────────
# General-purpose: blue-anchored with balanced variety (NO red in neutral charts)
PALETTE = [
    "#4F8EF7",  # blue
    "#2EC4B6",  # teal
    "#A66CFF",  # purple
    "#F5B041",  # amber
    "#2ECC71",  # green
    "#7AABFF",  # light blue
    "#FF8A65",  # coral
    "#7C8DB5",  # slate
]

# Plan tier palette: each tier gets a semantically appropriate color
PLAN_TIER_COLORS = {
    "Mobile":   "#F5B041",  # amber — entry / value
    "Basic":    "#2EC5FF",  # cyan — standard entry
    "Standard": "#4F8EF7",  # blue — core product
    "Premium":  "#A66CFF",  # purple — premium
}

# Diverging scale for rate metrics (good → bad)
SCALE_POSITIVE = ["#D94045", "#F5B041", "#2ECC71"]  # red → amber → green
SCALE_NEGATIVE = ["#2ECC71", "#F5B041", "#FF5A5F"]  # green → amber → red (for churn)

# Consistent Plotly layout — softened gridlines, polished hover, refined typography
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9EA3AE", family="Inter, -apple-system, sans-serif", size=12),
    title_font=dict(color="#C0C4CC", size=15, family="Inter, sans-serif"),
    margin=dict(l=56, r=24, t=52, b=48),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.035)",
        zerolinecolor="rgba(255,255,255,0.06)",
        title_font=dict(size=11, color="#6B7080"),
        tickfont=dict(size=10, color="#6B7080"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.035)",
        zerolinecolor="rgba(255,255,255,0.06)",
        title_font=dict(size=11, color="#6B7080"),
        tickfont=dict(size=10, color="#6B7080"),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
        font=dict(size=11, color="#8890A0"),
    ),
    hoverlabel=dict(
        bgcolor="#1A1A22",
        bordercolor="rgba(255,255,255,0.08)",
        font_size=12,
        font_color="#D0D4DC",
        font_family="Inter, sans-serif",
    ),
)


# ── Data Loading & Feature Engineering ─────────────────────────────────────────

def load_data(path: str = "./netflix_synthetic_subscription_data.csv") -> pd.DataFrame:
    """Load CSV and engineer all analytical features."""
    df = pd.read_csv(path)

    # Parse dates
    df["created_date"]  = pd.to_datetime(df["created_date"], errors="coerce")
    df["canceled_date"] = pd.to_datetime(df["canceled_date"], errors="coerce")

    # ── Core lifecycle fields ──────────────────────────────────────────────
    df["is_churned"]    = df["canceled_date"].notna().astype(int)
    df["event_observed"] = df["is_churned"]  # alias for survival analysis

    end_date = df["canceled_date"].fillna(ANALYSIS_DATE)
    df["tenure_days"]   = (end_date - df["created_date"]).dt.days.clip(lower=0)
    df["tenure_months"] = (df["tenure_days"] / 30.44).round(2)

    # ── Revenue estimation ─────────────────────────────────────────────────
    # Monthly plans: subscription_cost × months active
    # Annual plans:  subscription_cost × years active (cost is annual price)
    df["billing_periods"] = np.where(
        df["subscription_interval"] == "annual",
        (df["tenure_days"] / 365.25).clip(lower=0),
        (df["tenure_days"] / 30.44).clip(lower=0),
    )
    df["estimated_lifetime_revenue"] = (df["subscription_cost"] * df["billing_periods"]).round(2)

    # Monthly-normalized revenue for cross-plan comparison
    df["monthly_rate"] = np.where(
        df["subscription_interval"] == "annual",
        df["subscription_cost"] / 12,
        df["subscription_cost"],
    )
    df["realized_revenue_to_date"] = (df["monthly_rate"] * df["tenure_months"]).round(2)

    # ── Cohort fields ──────────────────────────────────────────────────────
    df["created_month"]   = df["created_date"].dt.to_period("M").astype(str)
    df["created_quarter"] = df["created_date"].dt.to_period("Q").astype(str)

    # ── Engagement scoring ─────────────────────────────────────────────────
    for col in ["first_30d_watch_hours", "sessions_first_30d", "recommendation_click_rate"]:
        lo, hi = df[col].min(), df[col].max()
        df[f"_{col}_norm"] = (df[col] - lo) / (hi - lo + 1e-9)

    df["engagement_score"] = (
        df["_first_30d_watch_hours_norm"] * 0.45
        + df["_sessions_first_30d_norm"] * 0.35
        + df["_recommendation_click_rate_norm"] * 0.20
    ).round(3)

    df["engagement_band"] = pd.cut(
        df["first_30d_watch_hours"],
        bins=[-1, 5, 15, 30, 999],
        labels=["Very Low (<5h)", "Low (5–15h)", "Medium (15–30h)", "High (30h+)"],
    )

    # ── Referral flags ─────────────────────────────────────────────────────
    df["was_referred"] = (df["referral_source"] != "None").astype(int)
    df["is_referrer"]  = (df["referral_sent"] == 1).astype(int)

    # ── Early churn flag ───────────────────────────────────────────────────
    df["early_churn_90d"] = ((df["is_churned"] == 1) & (df["tenure_days"] <= 90)).astype(int)

    # Clean up temp columns
    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True)

    return df


# ── Filter helpers ─────────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filter selections to the dataframe."""
    out = df.copy()

    for col, vals in filters.items():
        if col == "date_range":
            start, end = vals
            out = out[(out["created_date"] >= pd.Timestamp(start)) & (out["created_date"] <= pd.Timestamp(end))]
        elif vals:  # non-empty list of selected values
            out = out[out[col].isin(vals)]

    return out


# ── KPI formatting ─────────────────────────────────────────────────────────────

def fmt_pct(val: float) -> str:
    return f"{val:.1%}"

def fmt_dollar(val: float) -> str:
    return f"${val:,.0f}"

def fmt_int(val) -> str:
    return f"{int(val):,}"

def fmt_days(val: float) -> str:
    return f"{val:,.0f} days"


# ── Survival / Retention curve builder ─────────────────────────────────────────

def build_retention_curve(df: pd.DataFrame, group_col: str = None, max_days: int = 365) -> pd.DataFrame:
    """
    Build Kaplan-Meier–style retention curves.

    Returns a long-format dataframe with columns:
      day, retention_rate, [group_col]
    """
    days = list(range(0, max_days + 1, 7))  # weekly granularity for smoothness
    records = []

    groups = df[group_col].unique() if group_col else [None]

    for grp in groups:
        subset = df[df[group_col] == grp] if group_col else df
        n = len(subset)
        if n < 10:
            continue

        for d in days:
            survived = (subset["tenure_days"] >= d).sum()
            records.append({
                "day": d,
                "retention_rate": survived / n,
                "group": grp if grp is not None else "All",
                "n": n,
            })

    return pd.DataFrame(records)


# ── Cohort retention matrix ────────────────────────────────────────────────────

def build_cohort_matrix(df: pd.DataFrame, max_months: int = 12) -> pd.DataFrame:
    """Build M0–M12 cohort retention matrix."""
    cohorts = df.groupby("created_month")
    retention = {}

    for coh, grp in cohorts:
        coh_start = pd.Timestamp(coh)
        n = len(grp)
        if n < 15:
            continue

        end_dates = grp["canceled_date"].fillna(ANALYSIS_DATE)

        for m in range(0, max_months + 1):
            boundary = coh_start + pd.DateOffset(months=m)
            if boundary > ANALYSIS_DATE:
                break
            survived = (end_dates >= boundary).sum()
            retention.setdefault(coh, {})[f"M{m}"] = survived / n

    rdf = pd.DataFrame(retention).T.sort_index()
    return rdf
