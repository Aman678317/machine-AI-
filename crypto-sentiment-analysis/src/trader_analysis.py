"""
trader_analysis.py
==================
Trader-focused analysis:
- Leaderboard generation
- Risk-adjusted performance
- Clustering traders into behavioral segments
- Consistent vs emotional trader detection

Author: PrimeTrade.ai Analytics Team
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from src.metrics import compute_trader_metrics, top_traders
from src.utils import sharpe_ratio, max_drawdown, profit_factor, winsorize

logger = logging.getLogger(__name__)

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


def build_leaderboard(
    df: pd.DataFrame,
    n: int = 20,
    pnl_col: str = "closedpnl",
) -> pd.DataFrame:
    """
    Build a top-N trader leaderboard ranked by total PnL.

    Parameters
    ----------
    df      : Merged cleaned DataFrame.
    n       : Number of top traders to include.
    pnl_col : Column for profit/loss.

    Returns
    -------
    Styled leaderboard DataFrame.
    """
    leaderboard = compute_trader_metrics(df, pnl_col)
    top = top_traders(leaderboard, n=n, by="total_pnl")
    logger.info(f"Leaderboard built with top {len(top)} traders ✓")
    return top


def segment_traders(
    df: pd.DataFrame,
    pnl_col: str = "closedpnl",
    n_clusters: int = 4,
) -> pd.DataFrame:
    """
    Cluster traders into behavioral segments using K-Means on:
    - Win rate
    - Average leverage
    - Sharpe ratio
    - Total trade count

    Returns
    -------
    Leaderboard DataFrame with a 'segment' column added.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    leaderboard = compute_trader_metrics(df, pnl_col)

    features = ["win_rate", "avg_leverage", "sharpe", "total_trades"]
    available = [f for f in features if f in leaderboard.columns]

    X = leaderboard[available].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    leaderboard["cluster"] = kmeans.fit_predict(X_scaled)

    # Label clusters based on mean PnL
    cluster_pnl = leaderboard.groupby("cluster")["total_pnl"].mean()
    cluster_rank = cluster_pnl.rank(ascending=False).astype(int)
    segment_labels = {
        1: "Elite Performers",
        2: "Consistent Traders",
        3: "Average Traders",
        4: "High-Risk Gamblers",
    }
    leaderboard["segment"] = leaderboard["cluster"].map(
        lambda c: segment_labels.get(cluster_rank[c], f"Cluster {c}")
    )

    logger.info(f"Trader segmentation complete ({n_clusters} clusters) ✓")
    return leaderboard


def consistent_trader_analysis(
    df: pd.DataFrame,
    min_trades: int = 20,
    pnl_col: str = "closedpnl",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify consistent traders (win_rate > 55%, Sharpe > 0.5)
    vs high-risk traders (avg_leverage > 15x, max_drawdown < -1000).

    Parameters
    ----------
    df         : Merged cleaned DataFrame.
    min_trades : Minimum trades required to qualify.
    pnl_col    : PnL column name.

    Returns
    -------
    Tuple of (consistent_traders_df, high_risk_traders_df).
    """
    lb = compute_trader_metrics(df, pnl_col)
    qualified = lb[lb["total_trades"] >= min_trades]

    consistent = qualified[
        (qualified["win_rate"] >= 0.55) & (qualified["sharpe"] >= 0.5)
    ].copy()

    high_risk = qualified[
        (qualified["avg_leverage"] >= 15.0) | (qualified["max_drawdown"] <= -1000)
    ].copy()

    logger.info(
        f"Consistent traders: {len(consistent)} | High-risk traders: {len(high_risk)} ✓"
    )
    return consistent, high_risk


def sentiment_behavior_heatmap(
    df: pd.DataFrame,
    pnl_col: str = "closedpnl",
    top_n_traders: int = 20,
) -> pd.DataFrame:
    """
    Create a heatmap-ready pivot: top traders × sentiment → mean PnL.

    Useful for identifying which traders perform well in each sentiment.
    """
    lb = compute_trader_metrics(df, pnl_col)
    top_accounts = lb.head(top_n_traders)["account"].tolist()

    subset = df[df["account"].isin(top_accounts)]
    pivot = subset.pivot_table(
        index="account",
        columns="sentiment_classification",
        values=pnl_col,
        aggfunc="mean",
        fill_value=0,
    )
    pivot = pivot.reindex(
        columns=[c for c in SENTIMENT_ORDER if c in pivot.columns]
    )
    logger.info(f"Sentiment behavior heatmap created for top {top_n_traders} traders ✓")
    return pivot


def drawdown_analysis(
    df: pd.DataFrame,
    account: str,
    pnl_col: str = "closedpnl",
) -> pd.DataFrame:
    """
    Compute running drawdown series for a single trader.

    Parameters
    ----------
    df      : Merged DataFrame.
    account : Trader account string.
    pnl_col : PnL column.

    Returns
    -------
    DataFrame with columns: [trade_date, pnl, cumulative_pnl, drawdown].
    """
    trader_df = df[df["account"] == account].sort_values("trade_date").copy()
    trader_df["cumulative_pnl"] = trader_df[pnl_col].cumsum()
    running_max = trader_df["cumulative_pnl"].cummax()
    trader_df["drawdown"] = trader_df["cumulative_pnl"] - running_max
    return trader_df[["trade_date", pnl_col, "cumulative_pnl", "drawdown"]]


def compute_win_streaks(
    df: pd.DataFrame,
    account: str,
    pnl_col: str = "closedpnl",
) -> dict:
    """
    Compute longest winning and losing streaks for a given trader.

    Break-even trades (PnL == 0) are excluded: they are neither wins
    nor losses and must not extend or reset either streak.

    Returns
    -------
    Dict with max_win_streak and max_loss_streak.
    """
    trader_df = df[df["account"] == account].sort_values("trade_date")
    # Exclude break-even trades before classifying outcomes
    pnl_series = trader_df[pnl_col].dropna()
    pnl_series = pnl_series[pnl_series != 0]
    outcomes = (pnl_series > 0).astype(int).tolist()

    max_win = max_loss = curr_win = curr_loss = 0
    for outcome in outcomes:
        if outcome:
            curr_win += 1
            curr_loss = 0
        else:
            curr_loss += 1
            curr_win = 0
        max_win = max(max_win, curr_win)
        max_loss = max(max_loss, curr_loss)

    return {"max_win_streak": max_win, "max_loss_streak": max_loss}
