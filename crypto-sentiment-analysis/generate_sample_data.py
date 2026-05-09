"""
generate_sample_data.py
=======================
Generates realistic synthetic sample datasets for testing the pipeline
when real Hyperliquid/Fear&Greed data is not yet available.

Run:  python generate_sample_data.py

Produces:
  data/historical_data.csv   (~5 000 synthetic trade rows)
  data/fear_greed.csv        (~500 days of sentiment data)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

np.random.seed(42)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Fear & Greed ───────────────────────────────────────────────────────────────
def generate_fear_greed(n_days: int = 500) -> pd.DataFrame:
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # Markov-chain-like sentiment transitions
    sentiments = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
    transition = np.array([
        [0.50, 0.35, 0.10, 0.04, 0.01],
        [0.20, 0.45, 0.25, 0.08, 0.02],
        [0.05, 0.20, 0.40, 0.25, 0.10],
        [0.02, 0.08, 0.25, 0.45, 0.20],
        [0.01, 0.04, 0.10, 0.35, 0.50],
    ])
    
    current = 2  # start Neutral
    labels = []
    for _ in dates:
        labels.append(sentiments[current])
        current = np.random.choice(5, p=transition[current])
    
    score_map = {"Extreme Fear": 15, "Fear": 35, "Neutral": 50, "Greed": 65, "Extreme Greed": 85}
    values = [score_map[l] + np.random.randint(-10, 10) for l in labels]
    values = np.clip(values, 0, 100)
    
    return pd.DataFrame({"date": [d.date() for d in dates], "classification": labels, "value": values})


# ── Historical Trades ──────────────────────────────────────────────────────────
def generate_historical(n_trades: int = 5000) -> pd.DataFrame:
    accounts = [f"0x{np.random.randint(1000, 9999)}{np.random.randint(1000,9999)}" for _ in range(80)]
    symbols = ["BTC", "ETH", "SOL", "ARB", "OP", "AVAX", "LINK", "MATIC", "SUI", "APT"]
    
    rows = []
    base_time = datetime(2023, 1, 1)
    
    for _ in range(n_trades):
        account = np.random.choice(accounts)
        symbol = np.random.choice(symbols, p=[0.3, 0.25, 0.15, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02])
        
        days_offset = np.random.randint(0, 480)
        ts = base_time + timedelta(days=days_offset, hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60))
        
        leverage = float(np.random.choice([1, 2, 3, 5, 10, 15, 20, 25, 50], p=[0.1, 0.15, 0.15, 0.2, 0.2, 0.1, 0.05, 0.03, 0.02]))
        side = np.random.choice(["buy", "sell"], p=[0.55, 0.45])
        size = round(abs(np.random.lognormal(2, 1.5)), 4)
        exec_price = round(np.random.uniform(0.5, 50000), 4)
        
        # PnL depends on leverage (higher lever → wider distribution)
        mu = 5 if side == "buy" else -5
        sigma = 50 * leverage / 5
        pnl = round(np.random.normal(mu, sigma), 4)
        
        event = np.random.choice(["CLOSE", "LIQUIDATION", "PARTIAL_CLOSE"], p=[0.80, 0.10, 0.10])
        if event == "LIQUIDATION":
            pnl = -abs(pnl) * 2  # liquidations are always losses
        
        rows.append({
            "account": account,
            "symbol": symbol,
            "execution_price": exec_price,
            "size": size,
            "side": side,
            "time": int(ts.timestamp() * 1000),
            "start_position": round(size * np.random.uniform(0.5, 2), 4),
            "event": event,
            "closedPnL": pnl,
            "leverage": leverage,
            "direction": side,
        })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating Fear & Greed data...")
    fg = generate_fear_greed(500)
    fg.to_csv(DATA_DIR / "fear_greed.csv", index=False)
    print(f"  Saved → data/fear_greed.csv  ({len(fg)} rows)")

    print("Generating historical trade data...")
    hist = generate_historical(5000)
    hist.to_csv(DATA_DIR / "historical_data.csv", index=False)
    print(f"  Saved → data/historical_data.csv  ({len(hist)} rows)")

    print("\n✅ Sample datasets ready. Run the dashboard or notebook now.")
