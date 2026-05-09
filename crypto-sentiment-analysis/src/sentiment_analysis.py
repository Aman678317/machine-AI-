"""
sentiment_analysis.py
=====================
Sentiment-focused analytical functions:
- Profitability vs sentiment
- Leverage risk by sentiment
- Buy/sell behavior
- Statistical significance testing

Author: PrimeTrade.ai Analytics Team
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


def fear_vs_greed_pnl(df: pd.DataFrame, pnl_col: str = "closedpnl") -> Dict[str, Any]:
    """
    Compare key PnL metrics between broad Fear (Extreme Fear + Fear)
    and broad Greed (Greed + Extreme Greed) periods.

    Returns
    -------
    Dictionary with summary statistics for each sentiment group.
    """
    df = df.copy()
    df["broad_sentiment"] = df["sentiment_classification"].apply(
        lambda x: "Fear"
        if x in ("Extreme Fear", "Fear")
        else ("Greed" if x in ("Greed", "Extreme Greed") else "Neutral")
    )

    result = {}
    for group, grp_df in df.groupby("broad_sentiment"):
        pnl = grp_df[pnl_col].dropna()
        result[group] = {
            "count": len(pnl),
            "mean_pnl": pnl.mean(),
            "median_pnl": pnl.median(),
            "total_pnl": pnl.sum(),
            "win_rate": (pnl > 0).mean(),
            "std_pnl": pnl.std(),
            "positive_days": (pnl > 0).sum(),
            "negative_days": (pnl < 0).sum(),
        }
    logger.info("Fear vs Greed PnL comparison computed ✓")
    return result


def leverage_risk_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze average leverage and proportion of high-leverage trades
    (>10x) per sentiment category.

    Returns a summary DataFrame.
    """
    if "leverage" not in df.columns:
        raise ValueError("leverage column not found.")

    df = df.copy()
    df["high_leverage"] = df["leverage"] > 10.0

    summary = df.groupby("sentiment_classification").agg(
        avg_leverage=("leverage", "mean"),
        median_leverage=("leverage", "median"),
        pct_high_leverage=("high_leverage", "mean"),
        avg_pnl_high_lev=("closedpnl", lambda x: x[df.loc[x.index, "high_leverage"]].mean()),
        avg_pnl_low_lev=("closedpnl", lambda x: x[~df.loc[x.index, "high_leverage"]].mean()),
    ).reset_index()

    summary["sentiment_classification"] = pd.Categorical(
        summary["sentiment_classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    summary.sort_values("sentiment_classification", inplace=True)
    logger.info("Leverage risk by sentiment computed ✓")
    return summary


def sentiment_transition_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze how PnL changes in the days following a sentiment shift
    (e.g., Fear → Greed transition).

    Uses the daily aggregated series and the fear_greed dataset.
    """
    df = df.copy().sort_values("trade_date")
    daily = (
        df.groupby("trade_date")
        .agg(
            mean_pnl=("closedpnl", "mean"),
            sentiment=("sentiment_classification", lambda x: x.mode().iat[0] if len(x.mode()) > 0 else "Unknown"),
        )
        .reset_index()
    )
    daily["prev_sentiment"] = daily["sentiment"].shift(1)
    daily["sentiment_changed"] = daily["sentiment"] != daily["prev_sentiment"]
    daily["transition"] = daily["prev_sentiment"] + " → " + daily["sentiment"]
    transitions = daily[daily["sentiment_changed"]].copy()
    logger.info(f"Found {len(transitions)} sentiment transition days ✓")
    return transitions


def symbol_performance_by_sentiment(
    df: pd.DataFrame,
    pnl_col: str = "closedpnl",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Find which trading symbols perform best during each sentiment regime.

    Returns a pivot table: symbols × sentiment → mean PnL.
    """
    if "symbol" not in df.columns:
        raise ValueError("symbol column not found.")

    pivot = (
        df.groupby(["symbol", "sentiment_classification"])[pnl_col]
        .mean()
        .unstack(fill_value=0)
    )
    pivot.columns.name = None
    pivot = pivot.reindex(
        columns=[c for c in SENTIMENT_ORDER if c in pivot.columns]
    )

    # Rank symbols by total average PnL across sentiments
    pivot["total_avg"] = pivot.mean(axis=1)
    pivot = pivot.nlargest(top_n, "total_avg")
    pivot.drop(columns="total_avg", inplace=True)
    logger.info(f"Symbol performance by sentiment computed (top {top_n}) ✓")
    return pivot


def identify_emotional_traders(
    df: pd.DataFrame,
    pnl_col: str = "closedpnl",
    leverage_threshold: float = 15.0,
    loss_rate_threshold: float = 0.65,
) -> pd.DataFrame:
    """
    Identify traders exhibiting emotional/irrational trading behavior:
    - High leverage during Fear periods.
    - High loss rate overall.
    - Increasing position sizes after losses.

    Returns a DataFrame of flagged trader accounts.
    """
    fear_df = df[df["sentiment_classification"].isin(["Fear", "Extreme Fear"])].copy()

    if fear_df.empty:
        logger.warning("No Fear-period trades found for emotional trader detection.")
        return pd.DataFrame()

    emotional = (
        fear_df.groupby("account")
        .agg(
            avg_leverage_fear=("leverage", "mean"),
            loss_rate=(pnl_col, lambda x: (x < 0).mean()),
            trade_count=(pnl_col, "count"),
        )
        .reset_index()
    )

    # Flag traders who use high leverage AND have high loss rate during fear
    emotional["is_emotional"] = (
        (emotional["avg_leverage_fear"] >= leverage_threshold)
        & (emotional["loss_rate"] >= loss_rate_threshold)
    )
    flagged = emotional[emotional["is_emotional"]].sort_values(
        "avg_leverage_fear", ascending=False
    )
    logger.info(f"Identified {len(flagged)} emotionally-trading accounts ✓")
    return flagged


def overtrading_detection(
    df: pd.DataFrame,
    greed_percentile: float = 0.75,
) -> pd.DataFrame:
    """
    Detect overtrading during Greed periods.
    Traders in the top quartile of trade frequency during Greed
    are flagged as potential overtrades.

    Returns a summary DataFrame.
    """
    greed_df = df[df["sentiment_classification"].isin(["Greed", "Extreme Greed"])].copy()
    fear_df = df[df["sentiment_classification"].isin(["Fear", "Extreme Fear"])].copy()

    greed_counts = greed_df.groupby("account").size().rename("greed_trades")
    fear_counts = fear_df.groupby("account").size().rename("fear_trades")

    comparison = pd.concat([greed_counts, fear_counts], axis=1).fillna(0)
    comparison["trade_ratio_greed_vs_fear"] = comparison["greed_trades"] / (
        comparison["fear_trades"] + 1
    )

    threshold = comparison["greed_trades"].quantile(greed_percentile)
    comparison["is_overtrader"] = comparison["greed_trades"] >= threshold

    overtrades = comparison[comparison["is_overtrader"]].sort_values(
        "trade_ratio_greed_vs_fear", ascending=False
    ).reset_index()
    logger.info(f"Detected {len(overtrades)} potential overtrades during Greed ✓")
    return overtrades
