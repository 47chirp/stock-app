import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import math
from datetime import date, timedelta
from helpers import load_data, render_top_nav

st.set_page_config(page_title="Analysis", layout="wide")
render_top_nav()
st.title("Analysis")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=date.today())
vol_window = st.sidebar.slider("Rolling Volatility Window (days)", 10, 120, 30, 5)

if ticker:
    df = load_data(ticker, start_date, end_date)
    if df.empty:
        st.error(f"No data for {ticker}")
    else:
        if "Close" not in df.columns:
            st.error("Price data is missing a 'Close' column and cannot be analyzed.")
            st.stop()

        df["Daily Return"] = df["Close"].pct_change()
        df["Rolling Volatility"] = df["Daily Return"].rolling(window=vol_window).std() * math.sqrt(252)

        st.subheader("Distribution of Daily Returns")
        returns_clean = df["Daily Return"].dropna()

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=returns_clean, nbinsx=60, marker_color="mediumpurple", opacity=0.75, name="Daily Returns", histnorm="probability density"))

        x_range = np.linspace(float(returns_clean.min()), float(returns_clean.max()), 200)
        mu = float(returns_clean.mean())
        sigma = float(returns_clean.std())

        if sigma > 0:
            fig_hist.add_trace(go.Scatter(x=x_range, y=stats.norm.pdf(x_range, mu, sigma), mode="lines", name="Normal Distribution", line=dict(color="red", width=2)))

        fig_hist.update_layout(xaxis_title="Daily Return", yaxis_title="Density", template="plotly_white", height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

        if len(returns_clean) > 10:
            jb_stat, jb_pvalue = stats.jarque_bera(returns_clean)
            st.caption(f"**Jarque-Bera test:** stat={jb_stat:.2f}, p={jb_pvalue:.4f} — {'Normal (p>0.05)' if jb_pvalue > 0.05 else 'Non-normal (p≤0.05)'}")

        st.subheader("Rolling Annualized Volatility")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=df.index, y=df["Rolling Volatility"], mode="lines", name=f"{vol_window}-Day Rolling Vol", line=dict(color="crimson")))
        fig_vol.update_layout(yaxis_title="Annualized Volatility", yaxis_tickformat=".0%", template="plotly_white", height=400)
        st.plotly_chart(fig_vol, use_container_width=True)
