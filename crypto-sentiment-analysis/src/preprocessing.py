"""
preprocessing.py
================
Production-grade data cleaning and preprocessing pipeline.
Handles missing values, type conversion, outlier treatment,
and date normalization for both datasets.

Author: PrimeTrade.ai Analytics Team
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
VALID_SIDES = {"buy", "sell", "long", "short", "b", "s"}
MAX_LEVERAGE = 200.0      # Hyperliquid max leverage cap
PNL_OUTLIER_ZSCORE = 5.0  # z-score threshold for PnL outlier flagging


# ── Fear & Greed Preprocessing ─────────────────────────────────────────────────

def clean_fear_greed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the Fear & Greed Index dataset.

    Steps:
    1. Parse dates robustly.
    2. Drop rows with invalid/missing dates.
    3. Standardize classification labels.
    4. Add ordinal sentiment score.
    5. Remove duplicates (keep last entry per date).

    Parameters
    ----------
    df : Raw Fear & Greed DataFrame from data_loader.

    Returns
    -------
    Cleaned DataFrame indexed by normalized date.
    """
    logger.info("Cleaning Fear & Greed dataset...")
    df = df.copy()

    # ── 1. Parse dates ──────────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_bad = df["date"].isna().sum()
    if n_bad:
        logger.warning(f"  Dropped {n_bad} rows with unparseable dates.")
    df.dropna(subset=["date"], inplace=True)
    df["date"] = df["date"].dt.normalize()  # strip time component

    # ── 2. Clean classification ────────────────────────────────────────────────
    df["classification"] = (
        df["classification"]
        .astype(str)
        .str.strip()
        .str.title()   # "extreme fear" → "Extreme Fear"
    )
    known = set(SENTIMENT_ORDER)
    unknown_mask = ~df["classification"].isin(known)
    if unknown_mask.any():
        logger.warning(f"  Unknown sentiment labels: {df.loc[unknown_mask, 'classification'].unique()}")

    # ── 3. Add ordinal score ───────────────────────────────────────────────────
    score_map = {s: i for i, s in enumerate(SENTIMENT_ORDER)}
    df["sentiment_score"] = df["classification"].map(score_map)

    # Numeric value column if present (some versions of the index include it)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # ── 4. Deduplicate ─────────────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(subset="date", keep="last", inplace=True)
    logger.info(f"  Removed {before - len(df)} duplicate date rows.")

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"  Fear & Greed cleaned → {df.shape[0]:,} rows ✓")
    return df


# ── Historical Trader Preprocessing ───────────────────────────────────────────

def _parse_timestamp(series: pd.Series) -> pd.Series:
    """
    Parse timestamps that may be Unix millis, Unix seconds, or ISO strings.
    """
    # If numeric, check magnitude for ms vs s
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.9:
        # Likely Unix timestamp
        if numeric.median() > 1e12:
            return pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)
        else:
            return pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
    # Fallback: ISO string
    return pd.to_datetime(series, errors="coerce", utc=True)


def clean_historical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize Hyperliquid historical trade data.

    Steps:
    1. Parse timestamps.
    2. Clean closedPnL column.
    3. Clean leverage column.
    4. Standardize side/direction.
    5. Handle missing values.
    6. Flag PnL outliers.
    7. Remove duplicates.

    Parameters
    ----------
    df : Raw historical trade DataFrame.

    Returns
    -------
    Cleaned DataFrame with typed columns and derived features.
    """
    logger.info("Cleaning Hyperliquid historical data...")
    df = df.copy()

    # ── 1. Timestamps ──────────────────────────────────────────────────────────
    if "time" in df.columns:
        df["timestamp"] = _parse_timestamp(df["time"])
        n_bad = df["timestamp"].isna().sum()
        if n_bad:
            logger.warning(f"  {n_bad} unparseable timestamps dropped.")
        df.dropna(subset=["timestamp"], inplace=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
        df["trade_date"] = df["timestamp"].dt.normalize().dt.tz_localize(None)
        df["trade_date"] = df["trade_date"].dt.date
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.day_name()
        df["month"] = df["timestamp"].dt.to_period("M").astype(str)

    # ── 2. closedPnL ──────────────────────────────────────────────────────────
    if "closedpnl" in df.columns:
        df["closedpnl"] = (
            df["closedpnl"]
            .astype(str)
            .str.replace(r"[,$\s]", "", regex=True)
        )
        df["closedpnl"] = pd.to_numeric(df["closedpnl"], errors="coerce")
        pnl_null = df["closedpnl"].isna().sum()
        if pnl_null:
            logger.info(f"  Imputing {pnl_null} null PnL values with 0.")
            df["closedpnl"].fillna(0.0, inplace=True)
        df["is_profitable"] = df["closedpnl"] > 0
        df["is_loss"] = df["closedpnl"] < 0

    # ── 3. Leverage ────────────────────────────────────────────────────────────
    if "leverage" in df.columns:
        df["leverage"] = pd.to_numeric(df["leverage"], errors="coerce")
        invalid_lev = (df["leverage"] <= 0) | (df["leverage"] > MAX_LEVERAGE)
        n_invalid = invalid_lev.sum()
        if n_invalid:
            logger.warning(f"  {n_invalid} invalid leverage values → set to NaN.")
            df.loc[invalid_lev, "leverage"] = np.nan
        df["leverage"].fillna(df["leverage"].median(), inplace=True)
        df["leverage_bucket"] = pd.cut(
            df["leverage"],
            bins=[0, 2, 5, 10, 25, 50, MAX_LEVERAGE],
            labels=["1-2x", "3-5x", "6-10x", "11-25x", "26-50x", "50x+"],
            right=True,
        )

    # ── 4. Size ────────────────────────────────────────────────────────────────
    if "size" in df.columns:
        df["size"] = pd.to_numeric(df["size"], errors="coerce").abs()

    # ── 5. Side / direction ────────────────────────────────────────────────────
    for col in ["side", "direction"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"b": "buy", "s": "sell", "long": "buy", "short": "sell"})

    # ── 6. Symbol cleanup ─────────────────────────────────────────────────────
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    # ── 7. Account cleanup ────────────────────────────────────────────────────
    if "account" in df.columns:
        df["account"] = df["account"].astype(str).str.strip()

    # ── 8. PnL outlier flag ───────────────────────────────────────────────────
    if "closedpnl" in df.columns:
        mu = df["closedpnl"].mean()
        sigma = df["closedpnl"].std()
        if sigma > 0:
            z = (df["closedpnl"] - mu) / sigma
            df["is_pnl_outlier"] = z.abs() > PNL_OUTLIER_ZSCORE
            n_out = df["is_pnl_outlier"].sum()
            logger.info(f"  Flagged {n_out} PnL outliers (|z| > {PNL_OUTLIER_ZSCORE}).")
        else:
            df["is_pnl_outlier"] = False

    # ── 9. Duplicates ─────────────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"  Removed {before - len(df)} duplicate rows.")

    df.reset_index(drop=True, inplace=True)
    logger.info(f"  Historical data cleaned → {df.shape[0]:,} rows ✓")
    return df


# ── Merge Pipeline ─────────────────────────────────────────────────────────────

def merge_datasets(
    historical_df: pd.DataFrame,
    fear_greed_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge cleaned historical trade data with Fear & Greed sentiment on trade date.

    Each trade row will gain:
    - sentiment_classification : str
    - sentiment_score          : int (0=Extreme Fear … 4=Extreme Greed)
    - fear_greed_value         : float (if available)

    Parameters
    ----------
    historical_df : Cleaned Hyperliquid data with 'trade_date' column.
    fear_greed_df : Cleaned Fear & Greed data with 'date' column.

    Returns
    -------
    Merged DataFrame.
    """
    logger.info("Merging datasets on trade_date ↔ date...")

    fg = fear_greed_df[["date", "classification", "sentiment_score"]].copy()
    if "value" in fear_greed_df.columns:
        fg["fear_greed_value"] = fear_greed_df["value"]
    fg.rename(columns={"date": "trade_date", "classification": "sentiment_classification"}, inplace=True)

    hist = historical_df.copy()
    hist["trade_date"] = pd.to_datetime(hist["trade_date"])
    fg["trade_date"] = pd.to_datetime(fg["trade_date"])

    merged = hist.merge(fg, on="trade_date", how="left")

    match_rate = merged["sentiment_classification"].notna().mean() * 100
    logger.info(f"  Sentiment match rate: {match_rate:.1f}% of trades ✓")

    unmatched = merged["sentiment_classification"].isna().sum()
    if unmatched:
        logger.warning(f"  {unmatched:,} trades could not be matched to a sentiment date.")

    merged.sort_values("trade_date", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    logger.info(f"  Merged dataset shape: {merged.shape} ✓")
    return merged


def run_full_pipeline(
    fear_greed_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline.

    Parameters
    ----------
    fear_greed_df : Raw Fear & Greed DataFrame.
    historical_df : Raw Historical Trades DataFrame.
    save_path     : Optional CSV path to persist cleaned output.

    Returns
    -------
    Final merged and cleaned DataFrame.
    """
    fg_clean = clean_fear_greed(fear_greed_df)
    hist_clean = clean_historical_data(historical_df)
    merged = merge_datasets(hist_clean, fg_clean)

    if save_path:
        merged.to_csv(save_path, index=False)
        logger.info(f"Cleaned data saved to: {save_path}")

    return merged
