import yfinance as yf
import pandas as pd
from datetime import date
import streamlit as st


def render_top_nav() -> None:
    # Keep sidebar widgets, but hide default vertical page list.
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    nav_items = [
        ("app.py", "Home", "🏠"),
        ("pages/1_Main.py", "Main", "📊"),
        ("pages/2_Portfolio.py", "Portfolio", "💼"),
        ("pages/3_Analysis.py", "Analysis", "📈"),
        ("pages/4_Comparison.py", "Comparison", "⚖️"),
        ("pages/5_Data.py", "Data", "🗂️"),
        ("pages/6_Optimizer.py", "Optimizer", "🔬"),
    ]

    cols = st.columns(len(nav_items))
    for col, (path, label, icon) in zip(cols, nav_items):
        with col:
            st.page_link(path, label=label, icon=icon, use_container_width=True)

    st.markdown("---")

@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data
def convert_df_to_csv(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=True).encode("utf-8")

@st.cache_data
def compute_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Close" in df.columns:
        df["Daily Return"] = df["Close"].pct_change()
        df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod() - 1
    return df

@st.cache_data(show_spinner="Fetching portfolio data...", ttl=3600)
def load_portfolio_data(tickers: tuple, start: date, end: date) -> pd.DataFrame:
    price_dict = {}
    for t in tickers:
        temp = yf.download(t, start=start, end=end, progress=False)
        if temp.empty:
            continue
        if isinstance(temp.columns, pd.MultiIndex):
            temp.columns = temp.columns.get_level_values(0)
        if "Close" in temp.columns:
            price_dict[t] = temp["Close"]
    if not price_dict:
        return pd.DataFrame()
    prices = pd.DataFrame(price_dict)
    prices = prices.sort_index().ffill().dropna(how="all")
    return prices
