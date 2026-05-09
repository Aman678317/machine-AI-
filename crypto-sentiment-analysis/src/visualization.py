"""
visualization.py
================
All static (matplotlib/seaborn) and interactive (plotly) chart generators.
Each function returns a figure object so callers can save or display it.

Author: PrimeTrade.ai Analytics Team
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List

from src.utils import sentiment_label_color

logger = logging.getLogger(__name__)

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
SENTIMENT_PALETTE = {
    "Extreme Fear": "#c0392b",
    "Fear": "#e74c3c",
    "Neutral": "#f39c12",
    "Greed": "#27ae60",
    "Extreme Greed": "#1abc9c",
}

# ── Global style ───────────────────────────────────────────────────────────────
plt.style.use("dark_background")
DARK_BG = "#0d1117"
CARD_BG = "#161b22"
ACCENT = "#58a6ff"
TEXT_COLOR = "#e6edf3"


def _dark_fig(figsize=(14, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    return fig, ax


# ── ANALYSIS 1: Profitability vs Sentiment ─────────────────────────────────────

def plot_pnl_boxplot(df: pd.DataFrame, pnl_col: str = "closedpnl") -> plt.Figure:
    """
    Box plot of PnL distribution per sentiment classification.
    Filters to inter-quartile range for visual clarity.
    """
    df_plot = df[df["sentiment_classification"].isin(SENTIMENT_ORDER)].copy()
    q01 = df_plot[pnl_col].quantile(0.01)
    q99 = df_plot[pnl_col].quantile(0.99)
    df_plot = df_plot[(df_plot[pnl_col] >= q01) & (df_plot[pnl_col] <= q99)]

    fig, ax = _dark_fig(figsize=(14, 7))
    colors = [SENTIMENT_PALETTE.get(s, ACCENT) for s in SENTIMENT_ORDER]
    sentiment_data = [
        df_plot[df_plot["sentiment_classification"] == s][pnl_col].dropna()
        for s in SENTIMENT_ORDER
    ]
    bp = ax.boxplot(
        sentiment_data,
        labels=SENTIMENT_ORDER,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#8b949e"),
        capprops=dict(color="#8b949e"),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.axhline(0, color="#8b949e", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title("PnL Distribution by Market Sentiment", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Sentiment Classification", fontsize=12)
    ax.set_ylabel("Closed PnL (USD)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    logger.info("PnL boxplot generated ✓")
    return fig


def plot_pnl_violin(df: pd.DataFrame, pnl_col: str = "closedpnl") -> plt.Figure:
    """Violin plot of PnL per sentiment."""
    df_plot = df[df["sentiment_classification"].isin(SENTIMENT_ORDER)].copy()
    q01 = df_plot[pnl_col].quantile(0.02)
    q99 = df_plot[pnl_col].quantile(0.98)
    df_plot = df_plot[(df_plot[pnl_col] >= q01) & (df_plot[pnl_col] <= q99)]

    fig, ax = _dark_fig(figsize=(14, 7))
    sns.violinplot(
        data=df_plot,
        x="sentiment_classification",
        y=pnl_col,
        order=SENTIMENT_ORDER,
        palette=SENTIMENT_PALETTE,
        inner="quartile",
        ax=ax,
    )
    ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title("PnL Violin Plot by Market Sentiment", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Closed PnL (USD)", fontsize=12)
    plt.tight_layout()
    return fig


def plot_win_rate_bar(summary_df: pd.DataFrame) -> plt.Figure:
    """Bar chart of win rate per sentiment."""
    fig, ax = _dark_fig(figsize=(10, 6))
    sentiments = summary_df["sentiment_classification"].tolist()
    win_rates = summary_df["win_rate"].tolist()
    colors = [SENTIMENT_PALETTE.get(s, ACCENT) for s in sentiments]

    bars = ax.bar(sentiments, [r * 100 for r in win_rates], color=colors, edgecolor="none", width=0.6)
    for bar, rate in zip(bars, win_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{rate * 100:.1f}%",
            ha="center", va="bottom", color=TEXT_COLOR, fontsize=11, fontweight="bold"
        )
    ax.axhline(50, color="#8b949e", linestyle="--", linewidth=1.5, label="50% baseline")
    ax.set_title("Win Rate by Market Sentiment", fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(facecolor=CARD_BG, edgecolor="none", labelcolor=TEXT_COLOR)
    plt.tight_layout()
    return fig


# ── ANALYSIS 2: Leverage ───────────────────────────────────────────────────────

def plot_leverage_histogram(df: pd.DataFrame) -> plt.Figure:
    """Histogram of leverage distribution."""
    fig, ax = _dark_fig(figsize=(12, 6))
    fear_lev = df[df["sentiment_classification"].isin(["Fear", "Extreme Fear"])]["leverage"].dropna()
    greed_lev = df[df["sentiment_classification"].isin(["Greed", "Extreme Greed"])]["leverage"].dropna()

    ax.hist(fear_lev, bins=40, alpha=0.65, color="#e74c3c", label="Fear", density=True)
    ax.hist(greed_lev, bins=40, alpha=0.65, color="#27ae60", label="Greed", density=True)
    ax.set_title("Leverage Distribution: Fear vs Greed", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Leverage", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(facecolor=CARD_BG, edgecolor="none", labelcolor=TEXT_COLOR, fontsize=12)
    plt.tight_layout()
    return fig


def plot_leverage_vs_pnl(df: pd.DataFrame, pnl_col: str = "closedpnl") -> go.Figure:
    """Interactive scatter: leverage vs PnL coloured by sentiment."""
    df_plot = df.dropna(subset=["leverage", pnl_col, "sentiment_classification"]).copy()
    q01 = df_plot[pnl_col].quantile(0.01)
    q99 = df_plot[pnl_col].quantile(0.99)
    df_plot = df_plot[(df_plot[pnl_col] >= q01) & (df_plot[pnl_col] <= q99)]
    df_plot = df_plot.sample(min(5000, len(df_plot)), random_state=42)

    fig = px.scatter(
        df_plot,
        x="leverage",
        y=pnl_col,
        color="sentiment_classification",
        color_discrete_map=SENTIMENT_PALETTE,
        opacity=0.5,
        title="Leverage vs Closed PnL by Sentiment",
        labels={"leverage": "Leverage", pnl_col: "Closed PnL (USD)"},
        template="plotly_dark",
        category_orders={"sentiment_classification": SENTIMENT_ORDER},
    )
    fig.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        legend_title="Sentiment",
    )
    return fig


# ── ANALYSIS 3: Buy/Sell Behavior ─────────────────────────────────────────────

def plot_side_distribution(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: buy vs sell per sentiment."""
    side_col = "side" if "side" in df.columns else "direction"
    if side_col not in df.columns:
        raise ValueError("No side/direction column found.")

    ct = (
        df.groupby(["sentiment_classification", side_col])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        ct,
        x="sentiment_classification",
        y="count",
        color=side_col,
        barmode="stack",
        category_orders={"sentiment_classification": SENTIMENT_ORDER},
        color_discrete_map={"buy": "#27ae60", "sell": "#e74c3c"},
        title="Buy vs Sell Trade Frequency by Sentiment",
        labels={"count": "Number of Trades", "sentiment_classification": "Sentiment"},
        template="plotly_dark",
    )
    fig.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
    )
    return fig


# ── ANALYSIS 4: Trader Leaderboard ────────────────────────────────────────────

def plot_top_traders_bar(leaderboard: pd.DataFrame, n: int = 15) -> go.Figure:
    """Horizontal bar chart of top N traders by total PnL."""
    top = leaderboard.head(n).copy()
    top["account_short"] = top["account"].str[:12] + "..."
    colors = ["#27ae60" if p >= 0 else "#e74c3c" for p in top["total_pnl"]]

    fig = go.Figure(go.Bar(
        x=top["total_pnl"],
        y=top["account_short"],
        orientation="h",
        marker_color=colors,
        text=[f"${p:,.0f}" for p in top["total_pnl"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top {n} Traders by Total PnL",
        xaxis_title="Total PnL (USD)",
        yaxis_title="Trader",
        template="plotly_dark",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        height=500,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_win_rate_vs_sharpe(leaderboard: pd.DataFrame) -> go.Figure:
    """Bubble chart: Win rate vs Sharpe coloured by total PnL."""
    df_plot = leaderboard.head(50).copy()
    df_plot["account_short"] = df_plot["account"].str[:10] + "…"

    fig = px.scatter(
        df_plot,
        x="win_rate",
        y="sharpe",
        size="total_trades",
        color="total_pnl",
        hover_name="account_short",
        color_continuous_scale="RdYlGn",
        title="Win Rate vs Sharpe Ratio (Top 50 Traders)",
        labels={"win_rate": "Win Rate", "sharpe": "Sharpe Ratio", "total_pnl": "Total PnL"},
        template="plotly_dark",
    )
    fig.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
    )
    return fig


# ── ANALYSIS 5: Time Series ────────────────────────────────────────────────────

def plot_cumulative_pnl(daily_df: pd.DataFrame) -> go.Figure:
    """Interactive cumulative PnL line chart with sentiment background."""
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    # Shade sentiment regions
    for _, row in daily_df.iterrows():
        color = SENTIMENT_PALETTE.get(row.get("sentiment", "Neutral"), "#8b949e")
        fig.add_vrect(
            x0=str(row["trade_date"]),
            x1=str(row["trade_date"]),
            fillcolor=color, opacity=0.1, line_width=0,
        )

    fig.add_trace(go.Scatter(
        x=daily_df["trade_date"],
        y=daily_df["cumulative_pnl"],
        mode="lines",
        name="Cumulative PnL",
        line=dict(color=ACCENT, width=2),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.1)",
    ))
    fig.update_layout(
        title="Cumulative PnL Over Time",
        xaxis_title="Date",
        yaxis_title="Cumulative PnL (USD)",
        template="plotly_dark",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        hovermode="x unified",
    )
    return fig


def plot_daily_pnl_bar(daily_df: pd.DataFrame) -> go.Figure:
    """Daily PnL bar chart colored by positive/negative."""
    colors = ["#27ae60" if p >= 0 else "#e74c3c" for p in daily_df["total_pnl"]]
    fig = go.Figure(go.Bar(
        x=daily_df["trade_date"],
        y=daily_df["total_pnl"],
        marker_color=colors,
        name="Daily PnL",
    ))
    fig.update_layout(
        title="Daily Total PnL",
        xaxis_title="Date",
        yaxis_title="PnL (USD)",
        template="plotly_dark",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
    )
    return fig


def plot_monthly_heatmap(heatmap_df: pd.DataFrame) -> go.Figure:
    """Monthly PnL heatmap (Year × Month)."""
    fig = go.Figure(go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns.tolist(),
        y=heatmap_df.index.astype(str).tolist(),
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="PnL (USD)"),
    ))
    fig.update_layout(
        title="Monthly PnL Heatmap",
        xaxis_title="Month",
        yaxis_title="Year",
        template="plotly_dark",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
    )
    return fig


# ── ANALYSIS 6: Sentiment Distribution ────────────────────────────────────────

def plot_sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart of sentiment distribution in the data."""
    counts = df["sentiment_classification"].value_counts().reindex(SENTIMENT_ORDER).dropna()
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.5,
        marker_colors=[SENTIMENT_PALETTE[s] for s in counts.index],
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Trade Distribution by Market Sentiment",
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Seaborn correlation heatmap for numeric trading features."""
    numeric_cols = ["closedpnl", "leverage", "size", "sentiment_score"]
    available = [c for c in numeric_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = _dark_fig(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig


def plot_symbol_heatmap(pivot_df: pd.DataFrame) -> go.Figure:
    """Interactive heatmap: top symbols × sentiment → mean PnL."""
    fig = go.Figure(go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns.tolist(),
        y=pivot_df.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Mean PnL"),
    ))
    fig.update_layout(
        title="Symbol Performance by Market Sentiment",
        xaxis_title="Sentiment",
        yaxis_title="Symbol",
        template="plotly_dark",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
    )
    return fig
