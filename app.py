"""
ABCflix Subscriber Growth Analytics Dashboard
──────────────────────────────────────────────
A stakeholder-friendly lifecycle analytics tool for product data scientists.

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from helpers import (
    load_data, apply_filters, build_retention_curve, build_cohort_matrix,
    ANALYSIS_DATE, COLORS, PALETTE, PLAN_TIER_COLORS, SCALE_POSITIVE, SCALE_NEGATIVE,
    CHART_LAYOUT,
    fmt_pct, fmt_dollar, fmt_int, fmt_days,
)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="ABCflix Growth Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Dark Theme CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Inter for typography ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ──────────────────────────────────────────────────────────── */
    .stApp {
        background-color: #0B0B0F;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, h4 { color: #EDEDF0 !important; font-family: 'Inter', sans-serif; }
    h1 { font-size: 28px !important; font-weight: 700 !important; letter-spacing: -0.5px !important; }
    h3 { font-size: 16px !important; font-weight: 600 !important; }
    h4 { font-size: 14px !important; font-weight: 600 !important; color: #C0C4CC !important; }

    .stMarkdown p { color: #9EA3AE; font-size: 13.5px; line-height: 1.6; }
    hr { border-color: #1E1E24 !important; }

    /* ── Sidebar — lighter, breathable ───────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #111116;
        border-right: 1px solid #1A1A20;
        padding-top: 1.5rem;
    }
    section[data-testid="stSidebar"] .stMarkdown p { color: #7A7F8A; font-size: 12.5px; }
    section[data-testid="stSidebar"] h3 { font-size: 14px !important; color: #C0C4CC !important; }
    section[data-testid="stSidebar"] .stDivider { margin: 0.8rem 0; }

    /* Sidebar button — subtle ghost style */
    section[data-testid="stSidebar"] .stButton > button {
        background: transparent;
        border: 1px solid #2A2A32;
        color: #7A7F8A;
        font-size: 12px;
        font-weight: 500;
        border-radius: 6px;
        padding: 6px 12px;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #4F8EF7;
        color: #B0B5C0;
    }

    /* ── Tab Navigation — segmented control style ────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #111116;
        border-radius: 10px;
        padding: 3px;
        border: 1px solid #1E1E24;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #6B7080;
        padding: 8px 18px;
        font-size: 13px;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        border: none;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #B0B5C0;
        background-color: #18181F;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1C2333 !important;
        color: #7AABFF !important;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(79,142,247,0.08);
    }
    /* Remove the blue underline indicator */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* ── KPI Cards — minimal, clean ──────────────────────────────────────── */
    .kpi-card {
        background: #12121A;
        border: 1px solid #1E1E26;
        border-radius: 12px;
        padding: 20px 16px 16px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #2A2E3A;
        background: #14141D;
    }
    .kpi-label {
        font-size: 10.5px;
        color: #6B7080;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        font-family: 'Inter', sans-serif;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 10px;
        color: #4D5260;
        margin-top: 4px;
        font-family: 'Inter', sans-serif;
    }
    .kpi-delta {
        font-size: 11px;
        margin-top: 4px;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }

    /* ── Section Headers — refined ───────────────────────────────────────── */
    .section-header {
        font-size: 20px;
        font-weight: 600;
        color: #EDEDF0;
        margin: 32px 0 4px;
        padding-bottom: 10px;
        border-bottom: 1px solid #1E1E24;
        display: block;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.3px;
    }
    .section-desc {
        font-size: 13px;
        color: #6B7080;
        margin-bottom: 28px;
        max-width: 720px;
        line-height: 1.55;
    }

    /* ── Insight Annotations ─────────────────────────────────────────────── */
    .chart-insight {
        background: #111118;
        border-left: 3px solid #2A3A5C;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 8px 0 24px;
        font-size: 12.5px;
        color: #8890A0;
        line-height: 1.5;
        font-family: 'Inter', sans-serif;
    }
    .chart-insight strong { color: #B0B8CC; font-weight: 600; }

    /* ── Recommendation Cards — refined ──────────────────────────────────── */
    .rec-card {
        background: #111118;
        border: 1px solid #1E1E26;
        border-left: 3px solid #4F8EF7;
        border-radius: 0 10px 10px 0;
        padding: 20px 22px;
        margin-bottom: 14px;
        transition: all 0.15s ease;
    }
    .rec-card:hover {
        background: #14141D;
        border-color: #252530;
    }
    .rec-card h4 { margin: 0 0 8px; font-size: 14px; color: #DDDFE4 !important; }
    .rec-card p { margin: 2px 0; font-size: 12.5px; color: #7A7F8A; line-height: 1.5; }
    .rec-num {
        font-size: 20px;
        font-weight: 700;
        color: #4A5678;
        margin-bottom: 6px;
        font-family: 'Inter', sans-serif;
    }

    /* ── Expander styling ────────────────────────────────────────────────── */
    .streamlit-expanderHeader { color: #8890A0 !important; font-size: 13px; }

    /* ── Spacing utility ─────────────────────────────────────────────────── */
    .spacer-sm { height: 12px; }
    .spacer-md { height: 24px; }
    .spacer-lg { height: 36px; }

    /* ── Hide default metric styling ─────────────────────────────────────── */
    [data-testid="stMetricValue"] { font-size: 22px !important; }

    /* ── Hide hamburger & footer ─────────────────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DATA LOADING                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@st.cache_data(show_spinner="Loading subscriber data...")
def get_data():
    return load_data()

df_raw = get_data()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR — GLOBAL FILTERS                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("### ABCflix Analytics")
    st.caption("Global filters apply to all sections.")

    # Date range
    min_date = df_raw["created_date"].min().date()
    max_date = df_raw["created_date"].max().date()
    date_range = st.date_input(
        "Signup Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        dr = date_range
    else:
        dr = (min_date, max_date)

    st.divider()

    f_channel  = st.multiselect("Channel", sorted(df_raw["acquisition_channel"].unique()))
    f_plan     = st.multiselect("Plan Tier", sorted(df_raw["plan_tier"].unique()))
    f_geo      = st.multiselect("Geography", sorted(df_raw["geography"].unique()))
    f_source   = st.multiselect("Signup Source", sorted(df_raw["signup_source"].unique()))
    f_segment  = st.multiselect("Segment", sorted(df_raw["user_segment"].unique()))
    f_interval = st.multiselect("Billing", sorted(df_raw["subscription_interval"].unique()))

    st.divider()

    if st.button("Reset Filters", use_container_width=True):
        st.rerun()

    st.divider()
    st.caption(f"Analysis: {ANALYSIS_DATE.strftime('%b %d, %Y')}  ·  {len(df_raw):,} subs")

# Build filter dict and apply
filters = {
    "date_range": dr,
    "acquisition_channel": f_channel,
    "plan_tier": f_plan,
    "geography": f_geo,
    "signup_source": f_source,
    "user_segment": f_segment,
    "subscription_interval": f_interval,
}

df = apply_filters(df_raw, filters)

if len(df) == 0:
    st.warning("No subscribers match the current filter selection. Please adjust your filters.")
    st.stop()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def styled_fig(fig, height=420):
    """Apply consistent dark styling to any Plotly figure."""
    fig.update_layout(**CHART_LAYOUT, height=height)
    return fig


def kpi_html(value, label, sub="", color="#EDEDF0"):
    """Render a single KPI card as HTML."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def section_header(icon, title, desc=""):
    """Render a section header with optional description."""
    st.markdown(f'<div class="section-header">{icon}  {title}</div>', unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="section-desc">{desc}</div>', unsafe_allow_html=True)


def chart_insight(text):
    """Render a short insight annotation below a chart."""
    st.markdown(f'<div class="chart-insight">{text}</div>', unsafe_allow_html=True)


def spacer(size="md"):
    """Add vertical breathing room."""
    st.markdown(f'<div class="spacer-{size}"></div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  NAVIGATION                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

st.markdown("# ABCflix · Subscriber Growth Analytics")
st.caption("Lifecycle-level product growth dashboard  ·  Interactive filters in sidebar")

spacer("sm")

tabs = st.tabs(["Overview", "Acquisition", "Retention", "Revenue", "Referral", "Recommendations"])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — OVERVIEW                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with tabs[0]:
    section_header("📈", "Overview",
        "High-level health check across the subscriber base. All metrics reflect your current filter selection.")

    # ── KPI Row ────────────────────────────────────────────────────────────
    total   = len(df)
    active  = int(df["is_active_as_of_2025_03_01"].sum())
    churned = int(df["is_churned"].sum())
    paid_rate = df["was_subscription_paid"].mean()
    med_tenure = df["tenure_days"].median()
    avg_rev = df["estimated_lifetime_revenue"].mean()
    ref_rate = df["is_referrer"].mean()

    cols = st.columns(7)
    kpis = [
        (fmt_int(total),    "Total Subs",     "",                              COLORS["text_primary"]),
        (fmt_int(active),   "Active",         f"{active/total:.0%} of total",  COLORS["positive"]),
        (fmt_int(churned),  "Churned",        f"{churned/total:.0%} of total", COLORS["negative"]),
        (fmt_pct(paid_rate),"Paid Conv.",      "paid / total",                 COLORS["positive"]),
        (fmt_days(med_tenure),"Med. Tenure",   "",                             COLORS["text_primary"]),
        (fmt_dollar(avg_rev),"Avg LTV",        "estimated",                    COLORS["premium"]),
        (fmt_pct(ref_rate), "Referral Rate",   "sent ≥1 referral",            COLORS["teal"]),
    ]
    for col, (val, label, sub, color) in zip(cols, kpis):
        col.markdown(kpi_html(val, label, sub, color), unsafe_allow_html=True)

    spacer("lg")

    # ── Row 1: Volume by channel + Plan mix (horizontal bar) ─────────────
    c1, c2 = st.columns([3, 2])

    with c1:
        ch_counts = df["acquisition_channel"].value_counts().reset_index()
        ch_counts.columns = ["channel", "subscribers"]
        fig = px.bar(
            ch_counts.sort_values("subscribers"),
            x="subscribers", y="channel", orientation="h",
            color_discrete_sequence=[COLORS["primary"]],
            title="Subscriber Volume by Channel",
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(styled_fig(fig, 380), use_container_width=True)

        # Dynamic insight
        top_ch = ch_counts.sort_values("subscribers", ascending=False).iloc[0]
        chart_insight(
            f"<strong>{top_ch['channel']}</strong> leads with {top_ch['subscribers']:,} subscribers "
            f"({top_ch['subscribers']/total:.0%} of filtered base)."
        )

    with c2:
        # Horizontal bar chart instead of donut — cleaner, more precise
        tier_counts = df["plan_tier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "subscribers"]
        tier_counts = tier_counts.sort_values("subscribers")
        tier_counts["pct"] = tier_counts["subscribers"] / tier_counts["subscribers"].sum()

        fig = px.bar(
            tier_counts,
            x="subscribers", y="tier", orientation="h",
            color="tier",
            color_discrete_map=PLAN_TIER_COLORS,
            title="Subscribers by Plan Tier",
            text=tier_counts["pct"].apply(lambda x: f"{x:.0%}"),
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(showlegend=False)
        st.plotly_chart(styled_fig(fig, 380), use_container_width=True)

        top_tier = tier_counts.iloc[-1]
        chart_insight(
            f"<strong>{top_tier['tier']}</strong> is the most popular tier "
            f"at {top_tier['pct']:.0%} of subscribers."
        )

    spacer("md")

    # ── Row 2: Signup trend (waterfall) — muted tones ────────────────────
    monthly = df.groupby("created_month").agg(
        new_subs=("customer_id", "count"),
        churns=("is_churned", "sum"),
    ).reset_index()
    monthly["net_adds"] = monthly["new_subs"] - monthly["churns"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["created_month"], y=monthly["new_subs"],
        name="New Subs",
        marker_color="rgba(46,204,113,0.55)",   # muted green
    ))
    fig.add_trace(go.Bar(
        x=monthly["created_month"], y=-monthly["churns"],
        name="Churns",
        marker_color="rgba(255,90,95,0.45)",     # muted red
    ))
    fig.add_trace(go.Scatter(
        x=monthly["created_month"], y=monthly["net_adds"],
        name="Net Adds", mode="lines+markers",
        line=dict(color=COLORS["caution"], width=2),
        marker=dict(size=4),
    ))
    fig.update_layout(
        title="Monthly Subscriber Waterfall",
        barmode="relative",
        xaxis_title="Month", yaxis_title="Subscribers",
    )
    st.plotly_chart(styled_fig(fig, 400), use_container_width=True)

    # Waterfall insight
    if len(monthly) > 1:
        latest = monthly.iloc[-1]
        prev   = monthly.iloc[-2]
        delta  = latest["net_adds"] - prev["net_adds"]
        direction = "up" if delta > 0 else "down"
        chart_insight(
            f"Latest month: <strong>{int(latest['new_subs'])}</strong> new subs, "
            f"<strong>{int(latest['churns'])}</strong> churns → "
            f"<strong>{int(latest['net_adds'])}</strong> net adds "
            f"({direction} {abs(int(delta))} vs prior month)."
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — ACQUISITION                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with tabs[1]:
    section_header("🎯", "Acquisition Quality",
        "Not all subscribers are equal. This section evaluates channel and source quality "
        "beyond raw volume — looking at paid conversion, onboarding completion, and early engagement.")

    # ── Paid conversion by channel ─────────────────────────────────────────
    ch_qual = df.groupby("acquisition_channel").agg(
        volume=("customer_id", "count"),
        paid_rate=("was_subscription_paid", "mean"),
        onboarding_rate=("onboarding_completed", "mean"),
        avg_watch_30d=("first_30d_watch_hours", "mean"),
        avg_sessions_30d=("sessions_first_30d", "mean"),
        churn_rate=("is_churned", "mean"),
    ).reset_index().sort_values("volume", ascending=False)

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            ch_qual.sort_values("paid_rate"),
            x="paid_rate", y="acquisition_channel", orientation="h",
            title="Paid Conversion Rate by Channel",
            color="paid_rate",
            color_continuous_scale=SCALE_POSITIVE,
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False)
        st.plotly_chart(styled_fig(fig, 380), use_container_width=True)

    with c2:
        fig = px.bar(
            ch_qual.sort_values("onboarding_rate"),
            x="onboarding_rate", y="acquisition_channel", orientation="h",
            title="Onboarding Completion Rate by Channel",
            color="onboarding_rate",
            color_continuous_scale=SCALE_POSITIVE,
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False)
        st.plotly_chart(styled_fig(fig, 380), use_container_width=True)

    # Insight for acquisition rates
    best_paid = ch_qual.sort_values("paid_rate", ascending=False).iloc[0]
    worst_paid = ch_qual.sort_values("paid_rate").iloc[0]
    chart_insight(
        f"<strong>{best_paid['acquisition_channel']}</strong> converts at "
        f"{best_paid['paid_rate']:.0%} (highest), while "
        f"<strong>{worst_paid['acquisition_channel']}</strong> lags at "
        f"{worst_paid['paid_rate']:.0%} — a {best_paid['paid_rate'] - worst_paid['paid_rate']:.0%} spread."
    )

    spacer("md")

    # ── Volume vs Quality scatter ──────────────────────────────────────────
    st.markdown("#### Channel Quality Matrix")
    fig = px.scatter(
        ch_qual,
        x="volume", y="churn_rate",
        size="avg_watch_30d", color="acquisition_channel",
        color_discrete_sequence=PALETTE,
        title="Volume vs. Churn Rate  (bubble = avg 30d watch hours)",
        hover_data=["paid_rate", "onboarding_rate"],
    )
    fig.update_layout(yaxis_tickformat=".0%", xaxis_title="Signup Volume", yaxis_title="Churn Rate")
    fig.update_traces(marker=dict(line=dict(width=1, color="rgba(255,255,255,0.1)")))
    st.plotly_chart(styled_fig(fig, 420), use_container_width=True)

    chart_insight(
        "Ideal channels sit in the <strong>lower-right</strong> quadrant — high volume, low churn. "
        "Bubble size indicates first-30-day engagement depth."
    )

    spacer("md")

    # ── Signup source analysis ─────────────────────────────────────────────
    st.markdown("#### Signup Source Deep Dive")
    src_qual = df.groupby("signup_source").agg(
        volume=("customer_id", "count"),
        paid_rate=("was_subscription_paid", "mean"),
        onboarding_rate=("onboarding_completed", "mean"),
        avg_watch_30d=("first_30d_watch_hours", "mean"),
        avg_sessions_30d=("sessions_first_30d", "mean"),
    ).reset_index().sort_values("volume", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            src_qual, x="signup_source", y="paid_rate",
            color="signup_source", color_discrete_sequence=PALETTE,
            title="Paid Conversion by Signup Source",
        )
        fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(styled_fig(fig, 360), use_container_width=True)

    with c2:
        fig = px.bar(
            src_qual, x="signup_source", y="avg_watch_30d",
            color="signup_source", color_discrete_sequence=PALETTE,
            title="Avg Watch Hours (First 30 Days) by Source",
        )
        fig.update_layout(showlegend=False, yaxis_title="Hours")
        st.plotly_chart(styled_fig(fig, 360), use_container_width=True)

    spacer("sm")

    # ── Channel ranking table ──────────────────────────────────────────────
    with st.expander("Full Channel Quality Scorecard"):
        display_df = ch_qual.copy()
        display_df.columns = ["Channel", "Volume", "Paid Conv %", "Onboarding %",
                              "Avg Watch (30d)", "Avg Sessions (30d)", "Churn Rate"]
        for c in ["Paid Conv %", "Onboarding %", "Churn Rate"]:
            display_df[c] = display_df[c].apply(lambda x: f"{x:.1%}")
        display_df["Avg Watch (30d)"] = display_df["Avg Watch (30d)"].apply(lambda x: f"{x:.1f}h")
        display_df["Avg Sessions (30d)"] = display_df["Avg Sessions (30d)"].apply(lambda x: f"{x:.1f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — RETENTION                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with tabs[2]:
    section_header("🔒", "Retention & Churn",
        "The single biggest driver of subscriber growth economics. "
        "This section examines when and why subscribers leave, and which segments retain best.")

    # ── Churn rate by segment ──────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        seg_churn = df.groupby("user_segment")["is_churned"].mean().reset_index()
        seg_churn.columns = ["segment", "churn_rate"]
        seg_churn = seg_churn.sort_values("churn_rate")
        fig = px.bar(
            seg_churn, x="churn_rate", y="segment", orientation="h",
            title="Churn Rate by User Segment",
            color="churn_rate",
            color_continuous_scale=SCALE_NEGATIVE,
        )
        fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False)
        st.plotly_chart(styled_fig(fig, 350), use_container_width=True)

    with c2:
        band_churn = df.groupby("engagement_band", observed=True)["is_churned"].mean().reset_index()
        band_churn.columns = ["band", "churn_rate"]
        fig = px.bar(
            band_churn, x="band", y="churn_rate",
            title="Churn Risk by First-30-Day Engagement",
            color="churn_rate",
            color_continuous_scale=SCALE_NEGATIVE,
        )
        fig.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False,
                          xaxis_title="Engagement Band", yaxis_title="Churn Rate")
        st.plotly_chart(styled_fig(fig, 350), use_container_width=True)

    # Churn insight
    highest_churn_seg = seg_churn.iloc[-1]
    lowest_churn_seg = seg_churn.iloc[0]
    chart_insight(
        f"<strong>{highest_churn_seg['segment']}</strong> has the highest churn at "
        f"{highest_churn_seg['churn_rate']:.0%}, while "
        f"<strong>{lowest_churn_seg['segment']}</strong> retains best at "
        f"{lowest_churn_seg['churn_rate']:.0%}."
    )

    spacer("md")

    # ── Retention curves ───────────────────────────────────────────────────
    st.markdown("#### Retention Curves")
    ret_by = st.selectbox("View retention by:", ["plan_tier", "acquisition_channel", "user_segment", "geography"], index=0)

    ret_data = build_retention_curve(df, group_col=ret_by, max_days=365)
    fig = px.line(
        ret_data, x="day", y="retention_rate", color="group",
        color_discrete_sequence=PALETTE,
        title=f"Subscriber Retention Curve by {ret_by.replace('_', ' ').title()}",
        hover_data=["n"],
    )
    fig.update_layout(
        yaxis_tickformat=".0%",
        xaxis_title="Days Since Signup",
        yaxis_title="Retention Rate",
        yaxis_range=[0, 1.05],
    )
    fig.update_traces(line=dict(width=2.5))
    st.plotly_chart(styled_fig(fig, 450), use_container_width=True)

    spacer("md")

    # ── Cohort retention heatmap ───────────────────────────────────────────
    st.markdown("#### Cohort Retention Heatmap")
    coh_matrix = build_cohort_matrix(df, max_months=12)

    if not coh_matrix.empty:
        coh_display = coh_matrix.iloc[-14:]

        fig = go.Figure(data=go.Heatmap(
            z=coh_display.values,
            x=coh_display.columns.tolist(),
            y=coh_display.index.tolist(),
            colorscale=[[0, COLORS["negative"]], [0.5, COLORS["caution"]], [1, COLORS["positive"]]],
            zmin=0.25, zmax=1.0,
            text=[[f"{v:.0%}" if pd.notna(v) else "" for v in row] for row in coh_display.values],
            texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
            hovertemplate="Cohort: %{y}<br>Month: %{x}<br>Retention: %{z:.1%}<extra></extra>",
        ))
        fig.update_layout(
            title="Monthly Cohort Retention (M0–M12)",
            xaxis_title="Months Since Signup",
            yaxis_title="Signup Cohort",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(styled_fig(fig, 500), use_container_width=True)

        chart_insight(
            "Read left-to-right to see how each cohort retains over time. "
            "Consistent vertical patterns suggest systemic retention levers, "
            "not cohort-specific events."
        )
    else:
        st.info("Not enough data for cohort retention view with current filters.")

    spacer("md")

    # ── Churn tenure distribution ──────────────────────────────────────────
    st.markdown("#### When Do Subscribers Churn?")
    churned_df = df[df["is_churned"] == 1].copy()
    if len(churned_df) > 0:
        churned_df["tenure_bucket"] = pd.cut(
            churned_df["tenure_months"],
            bins=[0, 1, 3, 6, 12, 24, 100],
            labels=["0–1m", "1–3m", "3–6m", "6–12m", "12–24m", "24m+"],
        )
        bucket_counts = churned_df["tenure_bucket"].value_counts().sort_index().reset_index()
        bucket_counts.columns = ["bucket", "count"]
        bucket_counts["pct"] = bucket_counts["count"] / bucket_counts["count"].sum()

        fig = px.bar(
            bucket_counts, x="bucket", y="count",
            title="Churn Distribution by Tenure at Cancellation",
            color="bucket",
            color_discrete_sequence=[COLORS["negative"], COLORS["negative_dark"], COLORS["caution"],
                                     COLORS["primary"], COLORS["premium"], COLORS["slate"]],
            text=bucket_counts["pct"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="Tenure at Cancellation", yaxis_title="Cancelled Subs")
        st.plotly_chart(styled_fig(fig, 380), use_container_width=True)

        early_pct = bucket_counts[bucket_counts["bucket"].isin(["0–1m", "1–3m"])]["pct"].sum()
        chart_insight(
            f"<strong>{early_pct:.0%}</strong> of all churn happens within the first 3 months — "
            f"the critical window for onboarding and habit-formation interventions."
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — REVENUE                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with tabs[3]:
    section_header("💰", "Revenue Analytics",
        "Understanding not just how many subscribers we have, but how much value they generate. "
        "Revenue is estimated as subscription_cost × billing periods active.")

    # ── Revenue KPIs ───────────────────────────────────────────────────────
    total_rev   = df["estimated_lifetime_revenue"].sum()
    avg_ltv     = df["estimated_lifetime_revenue"].mean()
    median_ltv  = df["estimated_lifetime_revenue"].median()
    avg_monthly = df["monthly_rate"].mean()

    cols = st.columns(4)
    rev_kpis = [
        (fmt_dollar(total_rev),   "Total Est. Revenue", "", COLORS["positive"]),
        (fmt_dollar(avg_ltv),     "Avg LTV",           "",  COLORS["premium"]),
        (fmt_dollar(median_ltv),  "Median LTV",        "",  COLORS["premium_dim"]),
        (fmt_dollar(avg_monthly), "Avg Monthly Rate",  "",  COLORS["text_primary"]),
    ]
    for col, (val, label, sub, color) in zip(cols, rev_kpis):
        col.markdown(kpi_html(val, label, sub, color), unsafe_allow_html=True)

    spacer("lg")

    # ── Revenue by plan tier ───────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        tier_rev = df.groupby("plan_tier").agg(
            total_rev=("estimated_lifetime_revenue", "sum"),
            avg_ltv=("estimated_lifetime_revenue", "mean"),
            count=("customer_id", "count"),
        ).reset_index()

        fig = px.bar(
            tier_rev.sort_values("total_rev"),
            x="total_rev", y="plan_tier", orientation="h",
            title="Total Revenue by Plan Tier",
            color="plan_tier",
            color_discrete_map=PLAN_TIER_COLORS,
            text=tier_rev.sort_values("total_rev")["total_rev"].apply(lambda x: f"${x:,.0f}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="Revenue ($)")
        st.plotly_chart(styled_fig(fig, 360), use_container_width=True)

    with c2:
        fig = px.bar(
            tier_rev.sort_values("avg_ltv"),
            x="avg_ltv", y="plan_tier", orientation="h",
            title="Avg Lifetime Value by Plan Tier",
            color="plan_tier",
            color_discrete_map=PLAN_TIER_COLORS,
            text=tier_rev.sort_values("avg_ltv")["avg_ltv"].apply(lambda x: f"${x:,.0f}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="Avg LTV ($)")
        st.plotly_chart(styled_fig(fig, 360), use_container_width=True)

    # Revenue tier insight
    top_rev_tier = tier_rev.sort_values("total_rev", ascending=False).iloc[0]
    top_ltv_tier = tier_rev.sort_values("avg_ltv", ascending=False).iloc[0]
    chart_insight(
        f"<strong>{top_rev_tier['plan_tier']}</strong> drives the most total revenue "
        f"(${top_rev_tier['total_rev']:,.0f}), while <strong>{top_ltv_tier['plan_tier']}</strong> "
        f"has the highest per-subscriber LTV (${top_ltv_tier['avg_ltv']:,.0f})."
    )

    spacer("md")

    # ── Revenue by channel: volume vs value ────────────────────────────────
    st.markdown("#### Channel: Volume vs. Value")
    ch_rev = df.groupby("acquisition_channel").agg(
        volume=("customer_id", "count"),
        total_rev=("estimated_lifetime_revenue", "sum"),
        avg_ltv=("estimated_lifetime_revenue", "mean"),
    ).reset_index()
    ch_rev["pct_subs"] = ch_rev["volume"] / ch_rev["volume"].sum()
    ch_rev["pct_rev"]  = ch_rev["total_rev"] / ch_rev["total_rev"].sum()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ch_rev["acquisition_channel"], y=ch_rev["pct_subs"],
        name="% of Subscribers", marker_color=COLORS["primary"], opacity=0.75,
    ))
    fig.add_trace(go.Bar(
        x=ch_rev["acquisition_channel"], y=ch_rev["pct_rev"],
        name="% of Revenue", marker_color=COLORS["premium"], opacity=0.75,
    ))
    fig.update_layout(
        title="Subscriber Share vs. Revenue Share by Channel",
        barmode="group", yaxis_tickformat=".0%",
        xaxis_title="Channel", yaxis_title="Share",
    )
    st.plotly_chart(styled_fig(fig, 400), use_container_width=True)

    chart_insight(
        "Channels where <strong>revenue share > subscriber share</strong> indicate high-value acquisition. "
        "The reverse signals volume-heavy but low-monetizing channels."
    )

    spacer("md")

    # ── Revenue by segment ─────────────────────────────────────────────────
    st.markdown("#### Revenue by User Segment")
    seg_rev = df.groupby("user_segment").agg(
        avg_ltv=("estimated_lifetime_revenue", "mean"),
        total_rev=("estimated_lifetime_revenue", "sum"),
        count=("customer_id", "count"),
    ).reset_index().sort_values("avg_ltv", ascending=False)

    fig = px.bar(
        seg_rev, x="user_segment", y="avg_ltv",
        color="user_segment", color_discrete_sequence=PALETTE,
        title="Avg Lifetime Value by User Segment",
        text=seg_rev["avg_ltv"].apply(lambda x: f"${x:,.0f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="Avg LTV ($)")
    st.plotly_chart(styled_fig(fig, 380), use_container_width=True)

    spacer("md")

    # ── Cohort revenue trend ───────────────────────────────────────────────
    st.markdown("#### Monthly Cohort Revenue")
    coh_rev = df.groupby("created_month").agg(
        cohort_size=("customer_id", "count"),
        total_rev=("estimated_lifetime_revenue", "sum"),
        avg_ltv=("estimated_lifetime_revenue", "mean"),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=coh_rev["created_month"], y=coh_rev["total_rev"],
               name="Total Revenue", marker_color=COLORS["primary"], opacity=0.6),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=coh_rev["created_month"], y=coh_rev["avg_ltv"],
                   name="Avg LTV", mode="lines+markers",
                   line=dict(color=COLORS["premium"], width=2.5)),
        secondary_y=True,
    )
    fig.update_layout(title="Revenue by Signup Cohort")
    fig.update_yaxes(title_text="Total Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="Avg LTV ($)", secondary_y=True)
    st.plotly_chart(styled_fig(fig, 400), use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 5 — REFERRAL                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with tabs[4]:
    section_header("🔗", "Referral Analysis",
        "Referrals are the highest-quality acquisition channel at near-zero marginal cost. "
        "This section quantifies referral behavior and identifies the best segments to activate.")

    # ── Referral KPIs ──────────────────────────────────────────────────────
    referrers     = df[df["referral_sent"] == 1]
    referred      = df[df["was_referred"] == 1]
    ref_send_rate = df["referral_sent"].mean()
    avg_ref_count = referrers["referred_signup_count"].mean() if len(referrers) > 0 else 0

    cols = st.columns(4)
    ref_kpis = [
        (fmt_pct(ref_send_rate), "Referral Send Rate", f"{len(referrers):,} referrers", COLORS["teal"]),
        (f"{avg_ref_count:.1f}",  "Avg Referrals Sent", "per referrer", COLORS["text_primary"]),
        (fmt_int(len(referred)),  "Referred Users",     f"{len(referred)/len(df):.1%} of base", COLORS["positive"]),
        (fmt_pct(referred["is_churned"].mean()) if len(referred) > 0 else "N/A",
         "Referred Churn Rate", "", COLORS["negative"]),
    ]
    for col, (val, label, sub, color) in zip(cols, ref_kpis):
        col.markdown(kpi_html(val, label, sub, color), unsafe_allow_html=True)

    spacer("lg")

    # ── Referral rate by segment ───────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        seg_ref = df.groupby("user_segment")["referral_sent"].mean().reset_index()
        seg_ref.columns = ["segment", "referral_rate"]
        seg_ref = seg_ref.sort_values("referral_rate", ascending=True)
        fig = px.bar(
            seg_ref, x="referral_rate", y="segment", orientation="h",
            title="Referral Send Rate by User Segment",
            color="referral_rate",
            color_continuous_scale=SCALE_POSITIVE,
        )
        fig.update_layout(xaxis_tickformat=".1%", coloraxis_showscale=False)
        st.plotly_chart(styled_fig(fig, 350), use_container_width=True)

    with c2:
        ch_ref = df.groupby("acquisition_channel")["referral_sent"].mean().reset_index()
        ch_ref.columns = ["channel", "referral_rate"]
        ch_ref = ch_ref.sort_values("referral_rate", ascending=True)
        fig = px.bar(
            ch_ref, x="referral_rate", y="channel", orientation="h",
            title="Referral Send Rate by Channel",
            color="referral_rate",
            color_continuous_scale=SCALE_POSITIVE,
        )
        fig.update_layout(xaxis_tickformat=".1%", coloraxis_showscale=False)
        st.plotly_chart(styled_fig(fig, 350), use_container_width=True)

    spacer("md")

    # ── Referred vs Non-referred comparison ────────────────────────────────
    st.markdown("#### Referred vs. Non-Referred Subscribers")

    if len(referred) > 10:
        non_referred = df[df["was_referred"] == 0]
        comp = pd.DataFrame({
            "Metric": ["Churn Rate", "Avg LTV ($)", "Avg Tenure (days)", "Avg Watch Hours (30d)", "Onboarding %"],
            "Referred": [
                referred["is_churned"].mean(),
                referred["estimated_lifetime_revenue"].mean(),
                referred["tenure_days"].mean(),
                referred["first_30d_watch_hours"].mean(),
                referred["onboarding_completed"].mean(),
            ],
            "Non-Referred": [
                non_referred["is_churned"].mean(),
                non_referred["estimated_lifetime_revenue"].mean(),
                non_referred["tenure_days"].mean(),
                non_referred["first_30d_watch_hours"].mean(),
                non_referred["onboarding_completed"].mean(),
            ],
        })
        comp["Delta"] = comp["Referred"] - comp["Non-Referred"]

        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=comp["Metric"], y=comp["Referred"],
                name="Referred", marker_color=COLORS["teal"],
            ))
            fig.add_trace(go.Bar(
                x=comp["Metric"], y=comp["Non-Referred"],
                name="Non-Referred", marker_color=COLORS["slate"], opacity=0.7,
            ))
            fig.update_layout(title="Head-to-Head: Referred vs. Non-Referred", barmode="group")
            st.plotly_chart(styled_fig(fig, 380), use_container_width=True)

        with c2:
            st.markdown("##### Key Comparisons")
            for _, row in comp.iterrows():
                better = "✅" if (row["Delta"] < 0 and row["Metric"] == "Churn Rate") or \
                                 (row["Delta"] > 0 and row["Metric"] != "Churn Rate") else "⚠️"
                if row["Metric"] == "Churn Rate":
                    st.markdown(f"{better} **{row['Metric']}**: {row['Referred']:.1%} vs {row['Non-Referred']:.1%}")
                elif "LTV" in row["Metric"] or "Tenure" in row["Metric"]:
                    st.markdown(f"{better} **{row['Metric']}**: {row['Referred']:,.0f} vs {row['Non-Referred']:,.0f}")
                else:
                    st.markdown(f"{better} **{row['Metric']}**: {row['Referred']:.1f} vs {row['Non-Referred']:.1f}")

        chart_insight(
            "Referred subscribers consistently outperform on retention and LTV — "
            "evidence that referral programs drive higher-quality acquisition."
        )
    else:
        st.info("Not enough referred users in the filtered dataset for comparison.")

    spacer("md")

    # ── Referral volume by segment ─────────────────────────────────────────
    st.markdown("#### Referred Signup Volume by Segment")
    seg_ref_vol = df.groupby("user_segment")["referred_signup_count"].sum().reset_index()
    seg_ref_vol.columns = ["segment", "referred_signups"]
    seg_ref_vol = seg_ref_vol.sort_values("referred_signups", ascending=True)

    fig = px.bar(
        seg_ref_vol, x="referred_signups", y="segment", orientation="h",
        title="Total Referred Signups Generated by Segment",
        color_discrete_sequence=[COLORS["teal"]],
    )
    st.plotly_chart(styled_fig(fig, 350), use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 6 — RECOMMENDATIONS                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

with tabs[5]:
    section_header("💡", "Data-Driven Recommendations",
        "Synthesized from the filtered dataset. Each recommendation is grounded in a specific "
        "analytical finding with estimated impact.")

    spacer("sm")

    # ── Generate dynamic recommendations from filtered data ────────────────

    # 1. Onboarding gap
    onb_complete   = df[df["onboarding_completed"] == 1]["is_churned"].mean()
    onb_incomplete = df[df["onboarding_completed"] == 0]["is_churned"].mean()
    onb_pct        = df["onboarding_completed"].mean()
    onb_gap        = onb_incomplete - onb_complete

    # 2. Worst retention channel
    ch_churn = df.groupby("acquisition_channel")["is_churned"].mean()
    worst_ch = ch_churn.idxmax()
    worst_ch_rate = ch_churn.max()
    best_ch  = ch_churn.idxmin()
    best_ch_rate = ch_churn.min()

    # 3. Low engagement risk
    low_eng = df[df["engagement_band"].isin(["Very Low (<5h)", "Low (5–15h)"])]
    low_eng_churn = low_eng["is_churned"].mean() if len(low_eng) > 0 else 0

    # 4. Referral opportunity
    high_eng_ref = df[df["engagement_score"] > 0.6]["referral_sent"].mean()
    overall_ref  = df["referral_sent"].mean()

    # 5. Early churn
    early_churn_rate = df["early_churn_90d"].mean()
    early_churn_n    = df["early_churn_90d"].sum()

    recs = [
        {
            "title": "Fix the Onboarding Funnel",
            "finding": f"Subscribers who don't complete onboarding churn at {onb_incomplete:.0%} "
                       f"vs {onb_complete:.0%} for those who do — a <b>{onb_gap:.0%} gap</b>. "
                       f"Currently {1-onb_pct:.0%} of subscribers never finish onboarding.",
            "action": "Redesign onboarding to be progressive (profile → genre → first watch). "
                      "A/B test reducing steps vs. gamifying completion. Target 95% completion rate.",
            "impact": f"Closing half the onboarding gap could retain ~{int(len(df) * (1-onb_pct) * onb_gap * 0.5):,} "
                      f"additional subscribers.",
            "color": COLORS["negative"],
        },
        {
            "title": "Deploy Day-14 Early Churn Intervention",
            "finding": f"{early_churn_rate:.0%} of all subscribers ({early_churn_n:,} users) churn within 90 days. "
                       f"The first two weeks are the highest-risk window.",
            "action": "Build a predictive trigger at day 14 for subscribers with <5 watch hours and <3 sessions. "
                      "Deliver personalized content recommendations and engagement nudges.",
            "impact": f"Preventing 20% of early churns saves ~{int(early_churn_n * 0.2):,} subscribers/cycle.",
            "color": COLORS["caution"],
        },
        {
            "title": "Shift Acquisition Budget to Higher-Retention Channels",
            "finding": f"<b>{worst_ch}</b> has {worst_ch_rate:.0%} churn rate vs "
                       f"<b>{best_ch}</b> at {best_ch_rate:.0%} — "
                       f"a {worst_ch_rate - best_ch_rate:.0%} spread.",
            "action": "Reallocate 15–20% of paid budget from high-churn to high-retention channels. "
                      "Implement channel-level LTV tracking in the growth model.",
            "impact": "Even a modest mix shift improves blended retention by 2–5pp, "
                      "compounding monthly into significant subscriber base growth.",
            "color": COLORS["primary"],
        },
        {
            "title": "Activate Referral Programs in High-Engagement Segments",
            "finding": f"High-engagement subscribers send referrals at {high_eng_ref:.1%} vs "
                       f"the overall rate of {overall_ref:.1%}.",
            "action": "Prompt referral sharing during peak engagement moments (binge sessions, "
                      "profile creation). Add in-app share CTAs after high-engagement content finishes.",
            "impact": "Doubling the referral rate from high-engagement users could drive 50–100+ "
                      "incremental organic signups per cycle at near-zero CAC.",
            "color": COLORS["teal"],
        },
        {
            "title": "Target Low-Engagement Monthly Subs for Habit Formation",
            "finding": f"Subscribers with low first-30-day engagement churn at {low_eng_churn:.0%}. "
                       f"These are subscribers who signed up but never formed a viewing habit.",
            "action": "Deploy a 'first 7 days' engagement sequence: curated playlists, push "
                      "notifications for trending content, and download prompts for mobile users.",
            "impact": "Moving 30% of low-engagement subs to medium-engagement could reduce "
                      "their churn rate by 15–20pp.",
            "color": COLORS["premium"],
        },
    ]

    for i, rec in enumerate(recs):
        st.markdown(f"""
        <div class="rec-card" style="border-left-color: {rec['color']};">
            <div class="rec-num">0{i+1}</div>
            <h4>{rec['title']}</h4>
            <p><b>Finding:</b> {rec['finding']}</p>
            <p><b>Action:</b> {rec['action']}</p>
            <p><b>Est. Impact:</b> {rec['impact']}</p>
        </div>
        """, unsafe_allow_html=True)

    spacer("md")
    st.info(
        "These recommendations update dynamically based on the filters selected in the sidebar. "
        "Try filtering to a specific geography or segment to see targeted insights."
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FOOTER                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

st.divider()
st.caption(
    f"ABCflix Subscriber Growth Analytics  ·  Synthetic Data  ·  "
    f"Analysis Date: {ANALYSIS_DATE.strftime('%B %d, %Y')}  ·  "
    f"Filtered: {len(df):,} / {len(df_raw):,} subscribers"
)
