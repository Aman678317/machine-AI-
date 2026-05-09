"""
data_loader.py
==============
Production-grade data loading module for the Crypto Sentiment Analysis project.
Handles loading, schema validation, and initial inspection of both datasets.

Author: PrimeTrade.ai Analytics Team
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

FEAR_GREED_FILE = DATA_DIR / "fear_greed.csv"
HISTORICAL_FILE = DATA_DIR / "historical_data.csv"

FEAR_GREED_REQUIRED_COLS = {"date", "classification"}
HISTORICAL_REQUIRED_COLS = {"account", "symbol", "closedpnl", "time"}


# ── Internal Helpers ───────────────────────────────────────────────────────────

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names: strip whitespace, lowercase, replace spaces with underscores.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with normalized column names.
    """
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "_", regex=True)
    )
    return df


def _validate_columns(df: pd.DataFrame, required: set, dataset_name: str) -> None:
    """
    Raise ValueError if required columns are missing.

    Parameters
    ----------
    df            : DataFrame to validate.
    required      : Set of required lowercase column names.
    dataset_name  : Human-readable name for logging.
    """
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{dataset_name}] Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    logger.info(f"[{dataset_name}] Column validation passed ✓")


def _inspect_dataframe(df: pd.DataFrame, name: str) -> None:
    """Print concise schema + null summary."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset : {name}")
    logger.info(f"Shape   : {df.shape[0]:,} rows × {df.shape[1]} cols")
    logger.info(f"Columns : {list(df.columns)}")
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if null_cols.empty:
        logger.info("Nulls   : None ✓")
    else:
        logger.info(f"Nulls   :\n{null_cols.to_string()}")
    logger.info(f"Dtypes  :\n{df.dtypes.to_string()}")
    logger.info(f"{'='*60}\n")


# ── Public API ─────────────────────────────────────────────────────────────────

def load_fear_greed(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load and validate the Bitcoin Fear & Greed Index dataset.

    Parameters
    ----------
    filepath : Optional path override. Defaults to data/fear_greed.csv.

    Returns
    -------
    pd.DataFrame with columns: [date, classification, value (if present)]

    Raises
    ------
    FileNotFoundError if file does not exist.
    ValueError if required columns are absent.
    """
    path = Path(filepath) if filepath else FEAR_GREED_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Fear & Greed file not found: {path}\n"
            f"Expected location: {FEAR_GREED_FILE}"
        )

    logger.info(f"Loading Fear & Greed data from: {path}")

    # Try multiple encodings for robustness
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"Could not decode file {path} with any known encoding.")

    df = _normalize_columns(df)
    _validate_columns(df, FEAR_GREED_REQUIRED_COLS, "FearGreed")
    _inspect_dataframe(df, "Fear & Greed Index")
    return df


def load_historical_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load and validate the Hyperliquid historical trader dataset.

    Parameters
    ----------
    filepath : Optional path override. Defaults to data/historical_data.csv.

    Returns
    -------
    pd.DataFrame with all trading columns normalized.

    Raises
    ------
    FileNotFoundError if file does not exist.
    ValueError if required columns are absent.
    """
    path = Path(filepath) if filepath else HISTORICAL_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Historical data file not found: {path}\n"
            f"Expected location: {HISTORICAL_FILE}"
        )

    logger.info(f"Loading Hyperliquid historical data from: {path}")

    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"Could not decode file {path} with any known encoding.")

    df = _normalize_columns(df)
    _validate_columns(df, HISTORICAL_REQUIRED_COLS, "HistoricalTrader")
    _inspect_dataframe(df, "Hyperliquid Historical Trades")
    return df


def load_all_datasets(
    fg_path: Optional[str] = None,
    hist_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: load both datasets and return as a tuple.

    Returns
    -------
    (fear_greed_df, historical_df)
    """
    fg_df = load_fear_greed(fg_path)
    hist_df = load_historical_data(hist_path)
    logger.info("Both datasets loaded successfully ✓")
    return fg_df, hist_df


if __name__ == "__main__":
    fg, hist = load_all_datasets()
    print(f"\nFear & Greed shape : {fg.shape}")
    print(f"Historical shape   : {hist.shape}")
