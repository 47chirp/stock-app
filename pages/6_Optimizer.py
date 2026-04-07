# pages/6_Optimizer.py
# -------------------------------------------------------
# Portfolio Optimizer — Section 2.1: User Inputs & Data Retrieval
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta

from helpers import render_top_nav

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
render_top_nav()
st.title("Portfolio Optimizer")

# ── Sidebar ──────────────────────────────────────────────────────────────────

# Todo 3: Risk-free rate input (default 2%)
rfr_pct = st.sidebar.number_input(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.1,
    help="Annualized risk-free rate used in Sharpe and Sortino ratio calculations.",
)
risk_free_rate = rfr_pct / 100.0

# Todo 2: Date range with minimum 2-year enforcement
MIN_YEARS = 2
today = date.today()
default_start = today - timedelta(days=365 * 3)  # default to 3 years back

start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=today)

date_span_days = (end_date - start_date).days
if date_span_days < MIN_YEARS * 365:
    st.sidebar.error(f"Date range must be at least {MIN_YEARS} years. Current range: {date_span_days} days.")
    st.info("Adjust the date range in the sidebar to at least 2 years to proceed.")
    st.stop()

# ── Ticker Entry ─────────────────────────────────────────────────────────────

# Todo 1: Ticker entry — 3 to 10 symbols, validated before proceeding
st.subheader("Step 1 — Enter Tickers")
st.caption("Enter between 3 and 10 stock ticker symbols, separated by commas.")

raw_ticker_input = st.text_input(
    "Tickers",
    value="AAPL, MSFT, GOOGL, AMZN, TSLA",
    placeholder="e.g. AAPL, MSFT, GOOGL, AMZN, TSLA",
)

tickers = [t.strip().upper() for t in raw_ticker_input.split(",") if t.strip()]
tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order

if len(tickers) < 3:
    st.warning(f"Please enter at least 3 tickers. You currently have {len(tickers)}.")
    st.stop()

if len(tickers) > 10:
    st.warning(f"Please enter no more than 10 tickers. You currently have {len(tickers)}.")
    st.stop()

st.success(f"Tickers accepted: {', '.join(tickers)}")

# ── Data Download ─────────────────────────────────────────────────────────────

# Todo 4: Download adjusted closing prices per ticker + ^GSPC benchmark
# Each ticker is fetched individually so we can report failures by name.

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_single(ticker: str, start: date, end: date) -> pd.Series:
    """Return a daily Close series for one ticker, or an empty Series on failure."""
    try:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty or "Close" not in raw.columns:
            return pd.Series(dtype=float, name=ticker)
        series = raw["Close"].rename(ticker)
        series.index = pd.to_datetime(series.index)
        return series
    except Exception:
        return pd.Series(dtype=float, name=ticker)


@st.cache_data(show_spinner="Downloading price data…", ttl=3600)
def load_prices(tickers: tuple, start: date, end: date) -> tuple:
    """
    Download adjusted closing prices for each ticker individually.
    Returns (prices_df, failed_tickers) where prices_df has one column per
    successfully downloaded ticker (outer-joined on dates).
    """
    series_list = []
    failed = []
    for t in tickers:
        s = _fetch_single(t, start, end)
        if s.empty:
            failed.append(t)
        else:
            series_list.append(s)

    if not series_list:
        return pd.DataFrame(), failed

    prices = pd.concat(series_list, axis=1).sort_index()
    return prices, failed


@st.cache_data(show_spinner="Downloading S&P 500 benchmark…", ttl=3600)
def load_benchmark(start: date, end: date) -> pd.Series:
    """
    Download S&P 500 (^GSPC) adjusted closing prices.
    Returns a Series named '^GSPC', or an empty Series on failure.
    The benchmark is used for comparison only — never included in optimization.
    """
    s = _fetch_single("^GSPC", start, end)
    return s


# ── Run download ──────────────────────────────────────────────────────────────

st.subheader("Step 2 — Data Retrieval")

with st.spinner("Downloading price data…"):
    prices_raw, failed_tickers = load_prices(tuple(tickers), start_date, end_date)
    benchmark = load_benchmark(start_date, end_date)

# Report any download failures
if failed_tickers:
    st.error(
        f"Failed to download data for the following ticker(s): "
        f"**{', '.join(failed_tickers)}**. "
        "Check that the symbols are valid and try again."
    )

if prices_raw.empty:
    st.error("No valid price data was returned. Please check your tickers and date range.")
    st.stop()

# ── Partial Data Handling ─────────────────────────────────────────────────────

# Todo 5: Drop tickers with >5% missing values in the full date range, then
# truncate to the overlapping (inner) date range of the remaining tickers.

MISSING_THRESHOLD = 0.05

missing_pct = prices_raw.isna().mean()
dropped_tickers = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()

if dropped_tickers:
    details = ", ".join(
        f"{t} ({missing_pct[t]:.1%} missing)" for t in dropped_tickers
    )
    st.warning(
        f"The following ticker(s) had more than {MISSING_THRESHOLD:.0%} missing data "
        f"and were removed: {details}"
    )
    prices_raw = prices_raw.drop(columns=dropped_tickers)

if prices_raw.empty:
    st.error("No tickers remain after removing those with excessive missing data.")
    st.stop()

# Truncate to the overlapping date range (drop any row with any NaN)
prices = prices_raw.dropna(how="any")

if prices.empty:
    st.error(
        "No overlapping date range found across the selected tickers. "
        "Try a shorter date range or different tickers."
    )
    st.stop()

# Validate that at least 3 tickers remain after all filtering
valid_tickers = prices.columns.tolist()
if len(valid_tickers) < 3:
    st.error(
        f"Only {len(valid_tickers)} ticker(s) remain after data validation. "
        "At least 3 are required. Please add more tickers or adjust the date range."
    )
    st.stop()

# Report truncation if the date range shrank
if len(prices) < len(prices_raw):
    st.info(
        f"Data aligned to overlapping trading days: "
        f"**{prices.index[0].date()}** → **{prices.index[-1].date()}** "
        f"({len(prices)} trading days across {len(valid_tickers)} tickers)."
    )

# ── Summary ───────────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tickers", len(valid_tickers))
col2.metric("Trading Days", len(prices))
col3.metric(
    "Date Range",
    f"{prices.index[0].date()} → {prices.index[-1].date()}",
)
col4.metric("Risk-Free Rate", f"{rfr_pct:.2f}%")

st.markdown("#### Adjusted Closing Prices (last 10 rows)")
st.dataframe(prices.tail(10).style.format("${:,.2f}"), use_container_width=True)

if not benchmark.empty:
    st.markdown("#### S&P 500 Benchmark (last 10 rows — comparison only)")
    st.dataframe(
        benchmark.tail(10).rename("^GSPC (Close)").to_frame().style.format("${:,.2f}"),
        use_container_width=True,
    )
else:
    st.warning("S&P 500 benchmark data could not be loaded.")

# Store cleaned data in session state for downstream optimization pages
st.session_state["opt_prices"] = prices
st.session_state["opt_benchmark"] = benchmark
st.session_state["opt_risk_free_rate"] = risk_free_rate
st.session_state["opt_tickers"] = valid_tickers
