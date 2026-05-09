"""
metrics.py
==========
Computes all quantitative performance metrics for traders and sentiment groups.
Provides functions consumed by both the Jupyter notebook and the Streamlit dashboard.

Author: PrimeTrade.ai Analytics Team
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from src.utils import sharpe_ratio, max_drawdown, profit_factor

logger = logging.getLogger(__name__)

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


# ── Sentiment-Level Metrics ────────────────────────────────────────────────────

def sentiment_pnl_summary(df: pd.DataFrame, pnl_col: str = "closedpnl") -> pd.DataFrame:
    """
    Aggregate PnL statistics grouped by sentiment classification.

    Returns a DataFrame with columns:
    [sentiment, count, mean_pnl, median_pnl, total_pnl,
     win_rate, std_pnl, skewness, kurtosis]
    """
    grouped = df.groupby("sentiment_classification")[pnl_col]
    stats = grouped.agg(
        count="count",
        mean_pnl="mean",
        median_pnl="median",
        total_pnl="sum",
        std_pnl="std",
    ).reset_index()

    # Win rate per sentiment
    win_rates = (
        df[df[pnl_col] > 0]
        .groupby("sentiment_classification")
        .size()
        .div(df.groupby("sentiment_classification").size())
        .rename("win_rate")
        .reset_index()
    )
    stats = stats.merge(win_rates, on="sentiment_classification", how="left")

    # Skewness & kurtosis
    skew = grouped.skew().rename("skewness").reset_index()
    kurt = grouped.apply(lambda x: x.kurt()).rename("kurtosis").reset_index()
    stats = stats.merge(skew, on="sentiment_classification", how="left")
    stats = stats.merge(kurt, on="sentiment_classification", how="left")

    # Sort by sentiment order
    stats["sentiment_classification"] = pd.Categorical(
        stats["sentiment_classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    stats.sort_values("sentiment_classification", inplace=True)
    stats.reset_index(drop=True, inplace=True)
    return stats


def leverage_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate leverage statistics grouped by sentiment classification.
    """
    if "leverage" not in df.columns:
        raise ValueError("Column 'leverage' not found in DataFrame.")

    stats = (
        df.groupby("sentiment_classification")["leverage"]
        .agg(
            mean_leverage="mean",
            median_leverage="median",
            max_leverage="max",
            std_leverage="std",
        )
        .reset_index()
    )
    stats["sentiment_classification"] = pd.Categorical(
        stats["sentiment_classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    stats.sort_values("sentiment_classification", inplace=True)
    return stats


def side_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count buy vs sell trades per sentiment classification.
    """
    side_col = "side" if "side" in df.columns else "direction"
    if side_col not in df.columns:
        raise ValueError("Neither 'side' nor 'direction' column found.")

    ct = (
        df.groupby(["sentiment_classification", side_col])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    ct["sentiment_classification"] = pd.Categorical(
        ct["sentiment_classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    ct.sort_values("sentiment_classification", inplace=True)
    return ct


# ── Trader-Level Metrics ───────────────────────────────────────────────────────

def compute_trader_metrics(df: pd.DataFrame, pnl_col: str = "closedpnl") -> pd.DataFrame:
    """
    Compute per-trader performance metrics.

    Returns a leaderboard DataFrame with columns:
    [account, total_trades, total_pnl, mean_pnl, win_rate,
     sharpe, max_drawdown, profit_factor, avg_leverage, consistency_score]
    """
    records: List[Dict[str, Any]] = []

    for account, grp in df.groupby("account"):
        pnl = grp[pnl_col].dropna()
        if len(pnl) < 2:
            continue

        wins = (pnl > 0).sum()
        losses = (pnl < 0).sum()
        total = len(pnl)
        win_rate = wins / total if total > 0 else 0.0

        cum_pnl = pnl.cumsum()
        mdd = max_drawdown(cum_pnl)
        pf = profit_factor(pnl)
        sr = sharpe_ratio(pnl)

        avg_lev = grp["leverage"].mean() if "leverage" in grp.columns else np.nan

        # Consistency score: win_rate * (1 - std/abs(mean)) penalised by leverage
        if pnl.mean() != 0:
            volatility_penalty = pnl.std() / abs(pnl.mean()) if pnl.mean() != 0 else 1
            consistency = win_rate * max(0, 1 - volatility_penalty / 10)
        else:
            consistency = 0.0

        records.append(
            {
                "account": account,
                "total_trades": total,
                "wins": wins,
                "losses": losses,
                "total_pnl": pnl.sum(),
                "mean_pnl": pnl.mean(),
                "median_pnl": pnl.median(),
                "win_rate": win_rate,
                "sharpe": sr,
                "max_drawdown": mdd,
                "profit_factor": pf,
                "avg_leverage": avg_lev,
                "consistency_score": consistency,
            }
        )

    leaderboard = pd.DataFrame(records)
    leaderboard.sort_values("total_pnl", ascending=False, inplace=True)
    leaderboard.reset_index(drop=True, inplace=True)
    leaderboard.index += 1   # 1-based rank
    leaderboard.index.name = "rank"
    return leaderboard


def top_traders(
    leaderboard: pd.DataFrame,
    n: int = 10,
    by: str = "total_pnl",
) -> pd.DataFrame:
    """
    Return top N traders from leaderboard by a given metric.

    Parameters
    ----------
    leaderboard : Output of compute_trader_metrics().
    n           : Number of traders.
    by          : Column to sort by.
    """
    return leaderboard.nlargest(n, by).reset_index(drop=True)


def trader_sentiment_behavior(
    df: pd.DataFrame, account: str, pnl_col: str = "closedpnl"
) -> pd.DataFrame:
    """
    For a specific trader, compute their PnL broken down by sentiment.
    """
    grp = df[df["account"] == account]
    if grp.empty:
        raise ValueError(f"Account '{account}' not found in DataFrame.")

    return sentiment_pnl_summary(grp, pnl_col)


# ── Time-Series Metrics ────────────────────────────────────────────────────────

def daily_pnl_series(df: pd.DataFrame, pnl_col: str = "closedpnl") -> pd.DataFrame:
    """
    Aggregate total PnL and trade count per day.
    """
    daily = (
        df.groupby("trade_date")
        .agg(
            total_pnl=(pnl_col, "sum"),
            trade_count=(pnl_col, "count"),
            mean_pnl=(pnl_col, "mean"),
            sentiment=("sentiment_classification", lambda x: x.mode().iat[0] if not x.empty else "Unknown"),
        )
        .reset_index()
    )
    daily.sort_values("trade_date", inplace=True)
    daily["cumulative_pnl"] = daily["total_pnl"].cumsum()
    return daily


def monthly_pnl_heatmap_data(df: pd.DataFrame, pnl_col: str = "closedpnl") -> pd.DataFrame:
    """
    Returns pivoted data suitable for a monthly PnL heatmap.
    Rows = Year, Columns = Month.
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df["trade_date"]).dt.year
    df["month_num"] = pd.to_datetime(df["trade_date"]).dt.month
    monthly = df.groupby(["year", "month_num"])[pnl_col].sum().unstack(fill_value=0)
    monthly.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ][:len(monthly.columns)]
    return monthly


# ── Statistical Testing ────────────────────────────────────────────────────────

def t_test_sentiment_pnl(
    df: pd.DataFrame,
    group_a: str = "Fear",
    group_b: str = "Greed",
    pnl_col: str = "closedpnl",
) -> Dict[str, Any]:
    """
    Independent samples t-test comparing PnL between two sentiment groups.

    Returns
    -------
    Dict with t-statistic, p-value, and interpretation.
    """
    from scipy import stats as sp_stats

    a = df[df["sentiment_classification"] == group_a][pnl_col].dropna()
    b = df[df["sentiment_classification"] == group_b][pnl_col].dropna()

    if len(a) < 5 or len(b) < 5:
        return {"error": "Insufficient data for t-test."}

    t_stat, p_val = sp_stats.ttest_ind(a, b, equal_var=False)
    significant = p_val < 0.05

    return {
        "group_a": group_a,
        "group_b": group_b,
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": a.mean(),
        "mean_b": b.mean(),
        "t_statistic": t_stat,
        "p_value": p_val,
        "significant": significant,
        "interpretation": (
            f"PnL difference between {group_a} and {group_b} is "
            + ("statistically significant (p < 0.05)." if significant
               else "NOT statistically significant (p ≥ 0.05).")
        ),
    }


def anova_sentiment_pnl(df: pd.DataFrame, pnl_col: str = "closedpnl") -> Dict[str, Any]:
    """
    One-way ANOVA across all sentiment groups.
    """
    from scipy import stats as sp_stats

    groups = [
        df[df["sentiment_classification"] == s][pnl_col].dropna()
        for s in SENTIMENT_ORDER
        if s in df["sentiment_classification"].values
    ]
    valid_groups = [g for g in groups if len(g) >= 5]

    if len(valid_groups) < 2:
        return {"error": "Not enough groups for ANOVA."}

    f_stat, p_val = sp_stats.f_oneway(*valid_groups)
    return {
        "f_statistic": f_stat,
        "p_value": p_val,
        "significant": p_val < 0.05,
        "interpretation": (
            "Sentiment has a statistically significant effect on PnL (p < 0.05)."
            if p_val < 0.05
            else "No statistically significant effect of sentiment on PnL found (p ≥ 0.05)."
        ),
    }
