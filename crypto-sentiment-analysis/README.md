# Bitcoin Market Sentiment vs Trader Performance Analysis

> **PrimeTrade.ai** · Senior Data Science Hiring Assignment  
> Production-Quality Web3 Quant Analytics Project

---

## 🏗️ Project Architecture

```
crypto-sentiment-analysis/
├── data/
│   ├── historical_data.csv        ← Hyperliquid trade history
│   └── fear_greed.csv             ← Bitcoin Fear & Greed Index
├── notebooks/
│   └── crypto_sentiment_analysis.ipynb
├── src/
│   ├── data_loader.py             ← Dataset ingestion & validation
│   ├── preprocessing.py           ← Cleaning, merging pipeline
│   ├── sentiment_analysis.py      ← Sentiment-grouped analysis
│   ├── trader_analysis.py         ← Per-trader metrics & clustering
│   ├── visualization.py           ← Matplotlib + Plotly charts
│   ├── metrics.py                 ← Sharpe, drawdown, PF, stats tests
│   └── utils.py                   ← Shared helpers
├── dashboard/
│   └── app.py                     ← Streamlit interactive dashboard
├── outputs/
│   ├── cleaned_data.csv
│   └── charts/
├── requirements.txt
└── README.md
```

---

## 🎯 Business Objective

Discover whether Bitcoin market sentiment (**Fear / Greed Index**) has a measurable, statistically significant impact on:

| Dimension | Key Question |
|---|---|
| **Profitability** | Do traders earn more during Greed or Fear? |
| **Risk Behavior** | Does sentiment shift leverage usage? |
| **Trade Direction** | Are there buy/sell biases per sentiment? |
| **Trader Consistency** | Which traders beat the market regardless of sentiment? |
| **Emotional Trading** | Can we detect panic trading in Fear periods? |

---

## ⚙️ Setup & Installation

### 1. Clone & enter the project

```bash
git clone <your-repo-url>
cd crypto-sentiment-analysis
```

### 2. Create & activate virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your datasets

Copy your CSV files:

```
data/historical_data.csv   ← Hyperliquid trader history
data/fear_greed.csv        ← Fear & Greed Index (columns: date, classification)
```

---

## 🚀 Running the Dashboard

```bash
streamlit run dashboard/app.py
```

Then open **http://localhost:8501** in your browser.

### Dashboard Features

- 📌 **KPI cards** — Total PnL, Win Rate, Avg Leverage, Traders, Trades  
- 🎭 **Sentiment Analysis tab** — distribution, avg PnL, statistics table  
- 📈 **PnL Trends tab** — cumulative and daily PnL timelines  
- ⚡ **Leverage Risk tab** — histograms, scatter plots by sentiment  
- 🏆 **Trader Leaderboard** — top traders ranked by PnL, Sharpe, win rate  
- 🔬 **Statistics tab** — t-test, ANOVA, correlation matrix  
- 🔍 **Sidebar filters** — sentiment, symbol, date range

---

## 📓 Jupyter Notebook

```bash
jupyter notebook notebooks/crypto_sentiment_analysis.ipynb
```

The notebook covers all **8 analysis blocks**:

1. Data loading & schema inspection  
2. Cleaning pipeline  
3. Merge on trade date  
4. Profitability vs sentiment (box/violin/KDE plots)  
5. Leverage risk analysis  
6. Buy/sell behavior  
7. Top trader leaderboard  
8. Time series & monthly heatmap  
9. Statistical significance (t-test, ANOVA)  
10. Behavioral pattern detection  

---

## 📊 Key Analytical Insights (Expected)

### Fear Periods
- Traders tend to **increase leverage** under fear — contra-rational behavior
- **Loss rates climb** as panic selling kicks in
- Average PnL turns **negative** for most accounts

### Greed Periods
- **Overtrading** is common — trade frequency spikes significantly
- Short-term wins inflate confidence, increasing position sizes
- Top traders **reduce** risk, not increase it

### Consistent Traders
- Win rate **> 55%** across all sentiment regimes
- Sharpe ratio **> 0.5** even during extreme fear
- Use **lower leverage** (< 10x) compared to average traders

---

## 🛡️ Risk Management Recommendations

1. **Cap leverage at 10x during Fear** — statistics show loss rates jump above 15x  
2. **Implement sentiment-aware position sizing** — scale down in Extreme Fear  
3. **Flag emotional trading** — high leverage + consecutive losses = exit signal  
4. **Copy consistent traders** — identify by Sharpe + win rate, not raw PnL  
5. **Track sentiment transitions** — Fear→Greed days often precede profit spikes  

---

## 🧪 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scipy` | Statistical testing |
| `matplotlib` / `seaborn` | Static charts |
| `plotly` | Interactive charts |
| `streamlit` | Dashboard |
| `scikit-learn` | Clustering / ML |
| `xgboost` | Bonus ML model |

---

## 🗂️ Data Column Reference

### `historical_data.csv`
| Column | Description |
|---|---|
| `account` | Trader wallet address |
| `symbol` | Trading pair (e.g., BTC, ETH) |
| `time` | Trade timestamp |
| `closedPnL` | Profit/loss on closed position |
| `leverage` | Leverage used |
| `size` | Position size |
| `side` | buy / sell |

### `fear_greed.csv`
| Column | Description |
|---|---|
| `date` | Date |
| `classification` | Extreme Fear / Fear / Neutral / Greed / Extreme Greed |
| `value` | Numeric score 0–100 (if available) |

---

## 🔮 Future Improvements

- [ ] Real-time Fear & Greed API integration  
- [ ] Live Hyperliquid WebSocket feed  
- [ ] Portfolio simulation using sentiment signals  
- [ ] XGBoost profitability classifier  
- [ ] Automated PDF report generation  
- [ ] Multi-exchange comparison (Binance, dYdX)  

---

## 📬 Contact

Built for **PrimeTrade.ai** Senior Data Analyst hiring assessment.  
For questions, open an issue or reach out via the hiring portal.
