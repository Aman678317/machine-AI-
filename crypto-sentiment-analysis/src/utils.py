"""
utils.py
========
General-purpose utility functions used across the project.

Author: PrimeTrade.ai Analytics Team
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"


def ensure_dirs() -> None:
    """Create output/chart directories if they don't exist."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename: str, dpi: int = 150) -> str:
    """
    Save a matplotlib figure to the charts directory.

    Parameters
    ----------
    fig      : matplotlib Figure object.
    filename : Output filename (without path).
    dpi      : Resolution.

    Returns
    -------
    Absolute path of saved figure.
    """
    ensure_dirs()
    path = CHARTS_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Chart saved: {path}")
    return str(path)


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a float as USD currency string."""
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{decimals}f}"


def format_pct(value: float, decimals: int = 1) -> str:
    """Format a float ratio as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Clip extreme values at given quantile bounds (Winsorization).

    Parameters
    ----------
    series : Numeric pandas Series.
    lower  : Lower quantile (default 1st percentile).
    upper  : Upper quantile (default 99th percentile).

    Returns
    -------
    Winsorized Series.
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def compute_rolling_stats(
    df: pd.DataFrame,
    value_col: str,
    date_col: str,
    window: int = 7,
) -> pd.DataFrame:
    """
    Compute rolling mean and std for a time-series column.

    Parameters
    ----------
    df        : DataFrame with date_col and value_col.
    value_col : Column to compute rolling stats on.
    date_col  : Date column for sorting.
    window    : Rolling window size in days.

    Returns
    -------
    DataFrame with new rolling_mean and rolling_std columns.
    """
    df = df.sort_values(date_col).copy()
    df[f"{value_col}_rolling_mean"] = (
        df[value_col].rolling(window=window, min_periods=1).mean()
    )
    df[f"{value_col}_rolling_std"] = (
        df[value_col].rolling(window=window, min_periods=1).std()
    )
    return df


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = 252) -> float:
    """
    Annualized Sharpe Ratio.

    Parameters
    ----------
    returns    : Series of daily or per-trade returns.
    risk_free  : Risk-free rate (annualized).
    periods    : Number of periods per year.

    Returns
    -------
    Sharpe ratio as float.
    """
    excess = returns - risk_free / periods
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(periods))


def max_drawdown(cumulative_pnl: pd.Series) -> float:
    """
    Compute maximum drawdown from a cumulative PnL series.

    Parameters
    ----------
    cumulative_pnl : Series of cumulative profit/loss values.

    Returns
    -------
    Maximum drawdown as a float (negative or zero).
    """
    roll_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - roll_max
    return float(drawdown.min())


def profit_factor(pnl: pd.Series) -> float:
    """
    Profit Factor = gross_profit / abs(gross_loss).

    Parameters
    ----------
    pnl : Series of trade PnL values.

    Returns
    -------
    Profit factor (>1 is profitable overall).
    """
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    return float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")


def sentiment_label_color(sentiment: str) -> str:
    """Return a hex color string for a given sentiment classification."""
    palette = {
        "Extreme Fear": "#c0392b",
        "Fear": "#e74c3c",
        "Neutral": "#f39c12",
        "Greed": "#27ae60",
        "Extreme Greed": "#1abc9c",
    }
    return palette.get(sentiment, "#95a5a6")
