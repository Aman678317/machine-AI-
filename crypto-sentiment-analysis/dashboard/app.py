"""
app.py  —  PrimeTrade.ai Streamlit Dashboard
============================================
Interactive dashboard: sentiment filters, KPI cards,
PnL charts, leverage analysis, trader leaderboard.

Run:  streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PrimeTrade.ai | Sentiment Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d1117; color: #e6edf3; }

.kpi-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.kpi-label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.kpi-value { font-size: 28px; font-weight: 700; color: #58a6ff; }
.kpi-delta { font-size: 13px; margin-top: 4px; }
.kpi-pos { color: #3fb950; } .kpi-neg { color: #f85149; }

.section-header {
    font-size: 22px; font-weight: 700; color: #e6edf3;
    border-left: 4px solid #58a6ff; padding-left: 14px;
    margin: 32px 0 16px 0;
}

section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}

div[data-testid="stMetric"] { background: #161b22; border-radius: 10px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
SENTIMENT_PALETTE = {
    "Extreme Fear": "#c0392b", "Fear": "#e74c3c",
    "Neutral": "#f39c12", "Greed": "#27ae60", "Extreme Greed": "#1abc9c",
}


@st.cache_data(show_spinner="Loading datasets…")
def load_data():
    cleaned_path = OUTPUTS_DIR / "cleaned_data.csv"
    
    # Automatically generate sample datasets if they are missing
    raw_fg = DATA_DIR / "fear_greed.csv"
    raw_hist = DATA_DIR / "historical_data.csv"
    if not raw_fg.exists() or not raw_hist.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        from generate_sample_data import generate_fear_greed, generate_historical
        fg_df = generate_fear_greed(500)
        hist_df = generate_historical(5000)
        fg_df.to_csv(raw_fg, index=False)
        hist_df.to_csv(raw_hist, index=False)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path, low_memory=False)
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        return df

    # Fallback: run pipeline on raw files
    from src.data_loader import load_all_datasets
    from src.preprocessing import run_full_pipeline

    fg, hist = load_all_datasets()
    df = run_full_pipeline(fg, hist, save_path=str(cleaned_path))
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    return df


def fmt_usd(v): return f"${v:,.2f}"
def fmt_pct(v): return f"{v*100:.1f}%"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌐 PrimeTrade.ai")
    st.markdown("**Sentiment × Performance Analytics**")
    st.markdown("---")

    try:
        df_full = load_data()
        data_loaded = True
    except FileNotFoundError as e:
        st.error(f"Data not found:\n{e}\n\nPlace `historical_data.csv` and `fear_greed.csv` in the `data/` folder.")
        data_loaded = False
        df_full = pd.DataFrame()

    if data_loaded and not df_full.empty:
        all_sentiments = [s for s in SENTIMENT_ORDER if s in df_full.get("sentiment_classification", pd.Series()).unique()]
        selected_sentiments = st.multiselect("🎭 Sentiment Filter", options=all_sentiments, default=all_sentiments)

        all_symbols = sorted(df_full["symbol"].dropna().unique()) if "symbol" in df_full.columns else []
        selected_symbols = st.multiselect("💎 Symbol Filter", options=all_symbols, default=all_symbols[:10] if len(all_symbols) > 10 else all_symbols)

        date_min = df_full["trade_date"].min()
        date_max = df_full["trade_date"].max()
        date_range = st.date_input("📅 Date Range", value=(date_min, date_max), min_value=date_min, max_value=date_max)

        st.markdown("---")
        st.caption(f"📈 {len(df_full):,} total trades loaded")
        st.caption("Data: Hyperliquid + Fear & Greed Index")

# ── Filter Data ────────────────────────────────────────────────────────────────
if data_loaded and not df_full.empty:
    df = df_full.copy()
    if selected_sentiments:
        df = df[df["sentiment_classification"].isin(selected_sentiments)]
    if selected_symbols:
        df = df[df["symbol"].isin(selected_symbols)]
    if len(date_range) == 2:
        df = df[(df["trade_date"] >= pd.Timestamp(date_range[0])) &
                (df["trade_date"] <= pd.Timestamp(date_range[1]))]

    pnl_col = "closedpnl"

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 32px 0 16px 0;">
      <h1 style="font-size:2.4rem; font-weight:800; color:#58a6ff; margin:0;">
        📊 Bitcoin Sentiment × Trader Performance
      </h1>
      <p style="color:#8b949e; font-size:1rem; margin-top:8px;">
        PrimeTrade.ai · Hyperliquid Historical Data · Fear & Greed Index
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📌 Key Performance Indicators</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

    total_pnl = df[pnl_col].sum() if pnl_col in df.columns else 0
    win_rate = (df[pnl_col] > 0).mean() if pnl_col in df.columns else 0
    avg_lev = df["leverage"].mean() if "leverage" in df.columns else 0
    unique_traders = df["account"].nunique() if "account" in df.columns else 0
    total_trades = len(df)

    def kpi_html(label, value, delta_class=""):
        return f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {delta_class}">{value}</div>
        </div>"""

    pnl_cls = "kpi-pos" if total_pnl >= 0 else "kpi-neg"
    with k1: st.markdown(kpi_html("Total PnL", fmt_usd(total_pnl), pnl_cls), unsafe_allow_html=True)
    with k2: st.markdown(kpi_html("Win Rate", fmt_pct(win_rate)), unsafe_allow_html=True)
    with k3: st.markdown(kpi_html("Avg Leverage", f"{avg_lev:.1f}x"), unsafe_allow_html=True)
    with k4: st.markdown(kpi_html("Unique Traders", f"{unique_traders:,}"), unsafe_allow_html=True)
    with k5: st.markdown(kpi_html("Total Trades", f"{total_trades:,}"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎭 Sentiment Analysis", "📈 PnL Trends",
        "⚡ Leverage Risk", "🏆 Trader Leaderboard", "🔬 Statistics"
    ])

    # ────────────────────────────────────────────────────────────────────────────
    # TAB 1 — Sentiment Analysis
    # ────────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Sentiment Distribution & PnL Impact</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            counts = df["sentiment_classification"].value_counts().reindex(SENTIMENT_ORDER).dropna()
            fig_donut = go.Figure(go.Pie(
                labels=counts.index.tolist(), values=counts.values.tolist(), hole=0.55,
                marker_colors=[SENTIMENT_PALETTE[s] for s in counts.index],
                textinfo="label+percent",
            ))
            fig_donut.update_layout(title="Trade Count by Sentiment", template="plotly_dark",
                                    paper_bgcolor="#0d1117", font=dict(color="#e6edf3"), height=380)
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_b:
            if pnl_col in df.columns:
                sent_pnl = df.groupby("sentiment_classification")[pnl_col].mean().reindex(SENTIMENT_ORDER).dropna()
                colors = [SENTIMENT_PALETTE.get(s, "#8b949e") for s in sent_pnl.index]
                fig_bar = go.Figure(go.Bar(
                    x=sent_pnl.index.tolist(), y=sent_pnl.values,
                    marker_color=colors, text=[fmt_usd(v) for v in sent_pnl.values],
                    textposition="outside",
                ))
                fig_bar.update_layout(title="Avg PnL by Sentiment", template="plotly_dark",
                                      paper_bgcolor="#0d1117", font=dict(color="#e6edf3"),
                                      height=380, yaxis_title="Mean Closed PnL (USD)")
                st.plotly_chart(fig_bar, use_container_width=True)

        # Win rate table
        st.markdown('<div class="section-header">Sentiment Statistics Table</div>', unsafe_allow_html=True)
        if pnl_col in df.columns:
            stats_rows = []
            for s in SENTIMENT_ORDER:
                grp = df[df["sentiment_classification"] == s][pnl_col].dropna()
                if len(grp) == 0: continue
                stats_rows.append({
                    "Sentiment": s, "Trades": len(grp),
                    "Mean PnL": fmt_usd(grp.mean()), "Median PnL": fmt_usd(grp.median()),
                    "Win Rate": fmt_pct((grp > 0).mean()),
                    "Total PnL": fmt_usd(grp.sum()), "Std Dev": fmt_usd(grp.std()),
                })
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    # ────────────────────────────────────────────────────────────────────────────
    # TAB 2 — PnL Trends
    # ────────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Cumulative & Daily PnL</div>', unsafe_allow_html=True)
        if pnl_col in df.columns and "trade_date" in df.columns:
            daily = (df.groupby("trade_date").agg(
                total_pnl=(pnl_col, "sum"),
                trade_count=(pnl_col, "count"),
                sentiment=("sentiment_classification", lambda x: x.mode().iat[0] if len(x) else "Unknown"),
            ).reset_index().sort_values("trade_date"))
            daily["cumulative_pnl"] = daily["total_pnl"].cumsum()

            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=daily["trade_date"], y=daily["cumulative_pnl"],
                mode="lines", name="Cumulative PnL",
                line=dict(color="#58a6ff", width=2),
                fill="tozeroy", fillcolor="rgba(88,166,255,0.1)",
            ))
            fig_cum.update_layout(title="Cumulative PnL Over Time", template="plotly_dark",
                                  paper_bgcolor="#0d1117", font=dict(color="#e6edf3"),
                                  xaxis_title="Date", yaxis_title="USD", height=380)
            st.plotly_chart(fig_cum, use_container_width=True)

            colors_bar = ["#3fb950" if p >= 0 else "#f85149" for p in daily["total_pnl"]]
            fig_daily = go.Figure(go.Bar(
                x=daily["trade_date"], y=daily["total_pnl"], marker_color=colors_bar, name="Daily PnL"
            ))
            fig_daily.update_layout(title="Daily Total PnL", template="plotly_dark",
                                    paper_bgcolor="#0d1117", font=dict(color="#e6edf3"),
                                    xaxis_title="Date", yaxis_title="USD", height=320)
            st.plotly_chart(fig_daily, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────────
    # TAB 3 — Leverage Risk
    # ────────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Leverage Risk by Sentiment</div>', unsafe_allow_html=True)
        if "leverage" in df.columns:
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                lev_sent = df.groupby("sentiment_classification")["leverage"].mean().reindex(SENTIMENT_ORDER).dropna()
                fig_lev = go.Figure(go.Bar(
                    x=lev_sent.index.tolist(), y=lev_sent.values,
                    marker_color=[SENTIMENT_PALETTE.get(s, "#8b949e") for s in lev_sent.index],
                    text=[f"{v:.1f}x" for v in lev_sent.values], textposition="outside",
                ))
                fig_lev.update_layout(title="Avg Leverage by Sentiment", template="plotly_dark",
                                      paper_bgcolor="#0d1117", font=dict(color="#e6edf3"),
                                      yaxis_title="Avg Leverage", height=360)
                st.plotly_chart(fig_lev, use_container_width=True)

            with col_l2:
                fear_lev = df[df["sentiment_classification"].isin(["Fear", "Extreme Fear"])]["leverage"].dropna()
                greed_lev = df[df["sentiment_classification"].isin(["Greed", "Extreme Greed"])]["leverage"].dropna()
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(x=fear_lev, name="Fear", marker_color="#e74c3c", opacity=0.7, nbinsx=40))
                fig_h.add_trace(go.Histogram(x=greed_lev, name="Greed", marker_color="#27ae60", opacity=0.7, nbinsx=40))
                fig_h.update_layout(barmode="overlay", title="Leverage Distribution: Fear vs Greed",
                                    template="plotly_dark", paper_bgcolor="#0d1117",
                                    font=dict(color="#e6edf3"), height=360,
                                    xaxis_title="Leverage", yaxis_title="Count")
                st.plotly_chart(fig_h, use_container_width=True)

            # Scatter
            df_sc = df.dropna(subset=["leverage", pnl_col]).copy()
            q01, q99 = df_sc[pnl_col].quantile(0.01), df_sc[pnl_col].quantile(0.99)
            df_sc = df_sc[(df_sc[pnl_col].between(q01, q99))].sample(min(3000, len(df_sc)), random_state=42)
            fig_sc = px.scatter(df_sc, x="leverage", y=pnl_col,
                                color="sentiment_classification",
                                color_discrete_map=SENTIMENT_PALETTE,
                                opacity=0.5, title="Leverage vs PnL by Sentiment",
                                template="plotly_dark",
                                category_orders={"sentiment_classification": SENTIMENT_ORDER})
            fig_sc.update_layout(paper_bgcolor="#0d1117", font=dict(color="#e6edf3"))
            st.plotly_chart(fig_sc, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────────
    # TAB 4 — Trader Leaderboard
    # ────────────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">🏆 Top Trader Leaderboard</div>', unsafe_allow_html=True)

        @st.cache_data(show_spinner="Computing trader metrics…")
        def get_leaderboard(data_hash):
            from src.metrics import compute_trader_metrics
            return compute_trader_metrics(df)

        lb = get_leaderboard(len(df))
        top_n = st.slider("Show top N traders", 5, 50, 15)
        top = lb.head(top_n).copy()
        top["account_short"] = top["account"].str[:14] + "…"

        colors_lb = ["#3fb950" if p >= 0 else "#f85149" for p in top["total_pnl"]]
        fig_lb = go.Figure(go.Bar(
            x=top["total_pnl"], y=top["account_short"], orientation="h",
            marker_color=colors_lb, text=[fmt_usd(p) for p in top["total_pnl"]], textposition="outside",
        ))
        fig_lb.update_layout(title=f"Top {top_n} Traders by Total PnL", template="plotly_dark",
                             paper_bgcolor="#0d1117", font=dict(color="#e6edf3"),
                             height=max(400, top_n * 28), yaxis=dict(autorange="reversed"),
                             xaxis_title="Total PnL (USD)")
        st.plotly_chart(fig_lb, use_container_width=True)

        display_cols = ["account", "total_trades", "total_pnl", "win_rate", "sharpe",
                        "avg_leverage", "profit_factor", "max_drawdown"]
        display_cols = [c for c in display_cols if c in top.columns]
        st.dataframe(top[display_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

    # ────────────────────────────────────────────────────────────────────────────
    # TAB 5 — Statistics
    # ────────────────────────────────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">📐 Statistical Significance Tests</div>', unsafe_allow_html=True)
        from src.metrics import t_test_sentiment_pnl, anova_sentiment_pnl

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            g_a = st.selectbox("Group A", SENTIMENT_ORDER, index=1)
        with col_s2:
            g_b = st.selectbox("Group B", SENTIMENT_ORDER, index=3)

        if st.button("Run t-Test"):
            result = t_test_sentiment_pnl(df, g_a, g_b)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(result["interpretation"])
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("t-Statistic", f"{result['t_statistic']:.4f}")
                rc2.metric("p-Value", f"{result['p_value']:.6f}")
                rc3.metric("Significant?", "✅ Yes" if result["significant"] else "❌ No")

        st.markdown("---")
        if st.button("Run ANOVA (all sentiment groups)"):
            anova = anova_sentiment_pnl(df)
            if "error" in anova:
                st.error(anova["error"])
            else:
                st.success(anova["interpretation"])
                ca, cb = st.columns(2)
                ca.metric("F-Statistic", f"{anova['f_statistic']:.4f}")
                cb.metric("p-Value", f"{anova['p_value']:.6f}")

        # Correlation matrix
        st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
        numeric_cols = [c for c in ["closedpnl", "leverage", "size", "sentiment_score"] if c in df.columns]
        if numeric_cols:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                 title="Feature Correlation Matrix", template="plotly_dark")
            fig_corr.update_layout(paper_bgcolor="#0d1117", font=dict(color="#e6edf3"))
            st.plotly_chart(fig_corr, use_container_width=True)

else:
    st.info("👈  Place `historical_data.csv` and `fear_greed.csv` in the `data/` folder, then refresh.")
