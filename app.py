# app.py
# -------------------------------------------------------
# A simple Streamlit stock analysis dashboard.
# Run with:  uv run streamlit run app.py
# This encorporates all bonus features brought up
# ------------------------------------------------------


import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
from scipy import stats
import math
import numpy as np

st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")

st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()

default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=date.today())

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=4.5,
    step=0.1
) / 100

ma_window = st.sidebar.slider(
    "Moving Average Window (days)",
    min_value=5,
    max_value=200,
    value=50,
    step=5
)

vol_window = st.sidebar.slider(
    "Rolling Volatility Window (days)",
    min_value=10,
    max_value=120,
    value=30,
    step=5
)

# -- Data download ----------------------------------------
# We wrap the download in st.cache_data so repeated runs with
# the same inputs don't re-download every time. The ttl (time-to-live)
# ensures the cache expires after one hour so data stays fresh.
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    return df

@st.cache_data
def convert_df_to_csv(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=True).encode("utf-8")

# NEW: cached benchmark computation (avoids recomputing each run)
@st.cache_data
def compute_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    df["Daily Return"] = df["Close"].pct_change()
    df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod() - 1
    return df

# -- Benchmark data (S&P 500) ----------------------------
benchmark = load_data("^GSPC", start_date, end_date)

if isinstance(benchmark.columns, pd.MultiIndex):
    benchmark.columns = benchmark.columns.get_level_values(0)

benchmark = compute_benchmark(benchmark)

# -- Main logic -------------------------------------------
if ticker:
    try:
        df = load_data(ticker, start_date, end_date)

        if df.empty:
            st.error(f"No data for {ticker}")
            st.stop()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["Daily Return"] = df["Close"].pct_change()
        df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod() - 1

    except Exception as e:
        st.error(f"Failed to load {ticker}: {e}")
        st.stop()

    df[f"{ma_window}-Day MA"] = df["Close"].rolling(window=ma_window).mean()
    df["Rolling Volatility"] = (
        df["Daily Return"].rolling(window=vol_window).std() * math.sqrt(252)
    )

    # Bollinger Bands (20-day MA ± 2 std)
    bb_window = 20
    df["BB_MA"] = df["Close"].rolling(window=bb_window).mean()
    df["BB_STD"] = df["Close"].rolling(window=bb_window).std()
    df["BB_Upper"] = df["BB_MA"] + 2 * df["BB_STD"]
    df["BB_Lower"] = df["BB_MA"] - 2 * df["BB_STD"]

    # Align benchmark to main ticker dates
    benchmark = benchmark.reindex(df.index).ffill().bfill()

    if ma_window > len(df):
        st.warning(
            f"The selected {ma_window}-day window is longer than the "
            f"available data ({len(df)} trading days). The moving average "
            "line won't appear — try a shorter window or a wider date range."
        )

    # -- Key metrics --------------------------------------
    latest_close = float(df["Close"].iloc[-1])
    total_return = float(df["Cumulative Return"].iloc[-1])
    avg_daily_ret = float(df["Daily Return"].mean())
    volatility = float(df["Daily Return"].std())
    ann_volatility = volatility * math.sqrt(252)

    # FIX: compounded annual return
    ann_return = (1 + avg_daily_ret) ** 252 - 1

    # FIX: safe Sharpe
    sharpe = (ann_return - risk_free_rate) / ann_volatility if ann_volatility != 0 else np.nan

    skewness = float(df["Daily Return"].skew())
    kurtosis = float(df["Daily Return"].kurtosis())
    max_close = float(df["Close"].max())
    min_close = float(df["Close"].min())

    st.subheader(f"{ticker} — Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close", f"${latest_close:,.2f}")
    col2.metric("Total Return", f"{total_return:.2%}")
    col3.metric("Annualized Return", f"{ann_return:.2%}")
    col4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Annualized Volatility (sigma)", f"{ann_volatility:.2%}")
    col6.metric("Skewness", f"{skewness:.2f}")
    col7.metric("Excess Kurtosis", f"{kurtosis:.2f}")
    col8.metric("Avg Daily Return", f"{avg_daily_ret:.4%}")

    col9, col10, _, _ = st.columns(4)
    col9.metric("Period High", f"${max_close:,.2f}")
    col10.metric("Period Low", f"${min_close:,.2f}")

    st.divider()

    csv = convert_df_to_csv(df)

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{ticker}_analysis.csv",
        mime="text/csv",
    )

    # -- Price chart --------------------------------------
    st.subheader("Price & Moving Average")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close Price",
            line=dict(width=1.5)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f"{ma_window}-Day MA"],
            mode="lines",
            name=f"{ma_window}-Day MA",
            line=dict(width=2, dash="dash")
        )
    )

    # Bollinger Upper Band
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_Upper"],
            mode="lines",
            name="BB Upper",
            line=dict(width=1, dash="dot", color="gray")
        )
    )

    # Bollinger Lower Band
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_Lower"],
            mode="lines",
            name="BB Lower",
            line=dict(width=1, dash="dot", color="gray"),
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.1)"
        )
    )

    # Optional clarity improvement (kept minimal)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_MA"],
            mode="lines",
            name="20-Day BB Midline",
            line=dict(width=1, color="gray")
        )
    )

    fig.update_layout(
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig, width="stretch")

    # -- Volume chart -------------------------------------
    st.subheader("Daily Trading Volume")

    fig_vol = go.Figure()
    fig_vol.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color="steelblue",
            opacity=0.7
        )
    )
    fig_vol.update_layout(
        yaxis_title="Shares Traded",
        xaxis_title="Date",
        template="plotly_white",
        height=350
    )
    st.plotly_chart(fig_vol, width="stretch")

    # -- Daily returns distribution -----------------------
    st.subheader("Distribution of Daily Returns")

    returns_clean = df["Daily Return"].dropna()

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=returns_clean,
            nbinsx=60,
            marker_color="mediumpurple",
            opacity=0.75,
            name="Daily Returns",
            histnorm="probability density"
        )
    )

    x_range = np.linspace(float(returns_clean.min()), float(returns_clean.max()), 200)
    mu = float(returns_clean.mean())
    sigma = float(returns_clean.std())

    # FIX: avoid sigma = 0 crash
    if sigma > 0:
        fig_hist.add_trace(
            go.Scatter(
                x=x_range,
                y=stats.norm.pdf(x_range, mu, sigma),
                mode="lines",
                name="Normal Distribution",
                line=dict(color="red", width=2)
            )
        )

    fig_hist.update_layout(
        xaxis_title="Daily Return",
        yaxis_title="Density",
        template="plotly_white",
        height=350
    )
    st.plotly_chart(fig_hist, width="stretch")

    # FIX: small sample guard
    if len(returns_clean) > 10:
        jb_stat, jb_pvalue = stats.jarque_bera(returns_clean)
        st.caption(
            f"**Jarque-Bera test:** statistic = {jb_stat:.2f}, p-value = {jb_pvalue:.4f} — "
            f"{'Fail to reject normality (p > 0.05)' if jb_pvalue > 0.05 else 'Reject normality (p <= 0.05)'}"
        )
    else:
        st.caption("Not enough data for Jarque-Bera test.")

    # -- Cumulative return comparison ----------------------
    st.subheader("Cumulative Return Comparison")

    comparison_tickers = st.multiselect(
        "Add stocks to compare",
        options=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"],
        default=[]
    )

    comparison_tickers = [t.upper().strip() for t in comparison_tickers if t.upper().strip() != ticker]

    fig_cum = go.Figure()

    fig_cum.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Cumulative Return"],
            mode="lines",
            name=ticker,
            line=dict(width=2)
        )
    )

    for comp in comparison_tickers:
        try:
            df_temp = load_data(comp, start_date, end_date)

            if df_temp.empty:
                st.warning(f"No data for {comp}")
                continue

            if isinstance(df_temp.columns, pd.MultiIndex):
                df_temp.columns = df_temp.columns.get_level_values(0)

            df_temp["Daily Return"] = df_temp["Close"].pct_change()
            df_temp["Cumulative Return"] = (1 + df_temp["Daily Return"]).cumprod() - 1

            # Align with main index
            df_temp = df_temp.reindex(df.index).ffill().bfill()

            fig_cum.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df_temp["Cumulative Return"],
                    mode="lines",
                    name=comp,
                    line=dict(width=2)
                )
            )

        except Exception as e:
            st.warning(f"Failed for {comp}: {e}")

    fig_cum.add_trace(
        go.Scatter(
            x=benchmark.index,
            y=benchmark["Cumulative Return"],
            mode="lines",
            name="S&P 500 (^GSPC)",
            line=dict(dash="dash", width=2)
        )
    )

    fig_cum.update_layout(
        yaxis_title="Cumulative Return",
        yaxis_tickformat=".0%",
        xaxis_title="Date",
        template="plotly_white",
        height=400
    )

    st.plotly_chart(fig_cum, width="stretch")

    # -- Rolling volatility chart -------------------------
    st.subheader("Rolling Annualized Volatility")

    fig_roll_vol = go.Figure()
    fig_roll_vol.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Rolling Volatility"],
            mode="lines",
            name=f"{vol_window}-Day Rolling Vol",
            line=dict(color="crimson", width=1.5)
        )
    )
    fig_roll_vol.update_layout(
        yaxis_title="Annualized Volatility",
        yaxis_tickformat=".0%",
        xaxis_title="Date",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig_roll_vol, width="stretch")

    # -- Raw data (expandable) ----------------------------
    with st.expander("View Raw Data"):
        st.dataframe(df.tail(60), width="stretch")

else:
    st.info("Enter a stock ticker in the sidebar to get started.")