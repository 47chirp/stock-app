import math
from datetime import date, timedelta

import plotly.graph_objects as go
import streamlit as st

from helpers import convert_df_to_csv, load_data, render_top_nav

st.set_page_config(page_title="Main - Stock Analyzer", layout="wide")
render_top_nav()
st.title("Main — Stock Overview")

STOCK_SEARCH_OPTIONS = [
    "AAPL - Apple",
    "MSFT - Microsoft",
    "GOOGL - Alphabet/Google",
    "AMZN - Amazon",
    "TSLA - Tesla",
    "META - Meta",
    "NVDA - NVIDIA",
    "JPM - JPMorgan Chase",
    "BAC - Bank of America",
    "UNH - UnitedHealth",
    "JNJ - Johnson & Johnson",
    "XOM - ExxonMobil",
    "CVX - Chevron",
    "WMT - Walmart",
    "COST - Costco",
    "MCD - McDonald's",
    "NFLX - Netflix",
    "DIS - Disney",
    "SPY - SPDR S&P 500 ETF",
    "VOO - Vanguard S&P 500 ETF",
    "QQQ - Invesco QQQ Trust",
    "VTI - Vanguard Total Stock Market ETF",
    "BND - Vanguard Total Bond Market ETF",
    "AGG - iShares Core U.S. Aggregate Bond ETF",
    "TLT - iShares 20+ Year Treasury Bond ETF",
    "GLD - SPDR Gold Shares",
]

COMPARE_CATEGORIES = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"],
    "Financials": ["JPM", "BAC"],
    "Healthcare": ["UNH", "JNJ"],
    "Consumer & Media": ["WMT", "COST", "MCD", "NFLX", "DIS"],
    "Energy": ["XOM", "CVX"],
    "ETFs": ["SPY", "VOO", "QQQ", "VTI", "BND", "AGG", "TLT", "GLD"],
}

METRIC_HELP = {
    "Latest Close": "Most recent closing price in the selected date range. Delta shows the change from the prior trading day.",
    "Total Return": "Overall percentage gain/loss from first close to latest close in the selected date range.",
    "Annualized Return": "Return scaled to a one-year rate from average daily return: (1 + avg daily return)^252 - 1.",
    "Sharpe Ratio": "Risk-adjusted return. Higher is better. Calculated as annualized return divided by annualized volatility (risk-free rate assumed 0).",
    "Annualized Volatility": "Estimated yearly volatility (standard deviation) of daily returns: daily std × sqrt(252).",
    "Skewness": "Asymmetry of daily returns. Positive means more upside tail, negative means more downside tail.",
    "Excess Kurtosis": "Tail heaviness vs normal distribution. Higher values indicate more extreme moves.",
    "Avg Daily Return": "Average daily percentage return over the selected period.",
    "Period High": "Highest closing price observed in the selected date range.",
    "Period Low": "Lowest closing price observed in the selected date range.",
}


def render_metric(col, label, value, delta=None):
    col.metric(label, value, delta=delta, help=METRIC_HELP.get(label, ""))


def prepare_returns(dataframe):
    if "Close" not in dataframe.columns:
        return dataframe
    dataframe = dataframe.copy()
    dataframe["Daily Return"] = dataframe["Close"].pct_change()
    dataframe["Cumulative Return"] = (1 + dataframe["Daily Return"]).cumprod() - 1
    return dataframe


def compute_metrics(dataframe):
    latest_close = float(dataframe["Close"].iloc[-1])
    previous_close = float(dataframe["Close"].iloc[-2]) if len(dataframe) > 1 else latest_close
    daily_change = latest_close - previous_close
    daily_change_pct = (daily_change / previous_close) if previous_close != 0 else 0.0

    if daily_change > 0:
        arrow = "↑"
    elif daily_change < 0:
        arrow = "↓"
    else:
        arrow = "→"

    total_return = float(dataframe["Cumulative Return"].iloc[-1])
    avg_daily_ret = float(dataframe["Daily Return"].mean())
    volatility = float(dataframe["Daily Return"].std())
    ann_volatility = volatility * math.sqrt(252)
    ann_return = (1 + avg_daily_ret) ** 252 - 1
    sharpe = (ann_return / ann_volatility) if ann_volatility != 0 else float("nan")

    return {
        "latest_close": latest_close,
        "arrow": arrow,
        "daily_change_pct": daily_change_pct,
        "total_return": total_return,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "ann_volatility": ann_volatility,
        "skewness": float(dataframe["Daily Return"].skew()),
        "kurtosis": float(dataframe["Daily Return"].kurtosis()),
        "avg_daily_ret": avg_daily_ret,
        "max_close": float(dataframe["Close"].max()),
        "min_close": float(dataframe["Close"].min()),
    }


# Sidebar inputs
if "default_start" not in st.session_state:
    st.session_state.default_start = date.today() - timedelta(days=365)

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()
start_date = st.sidebar.date_input("Start Date", value=st.session_state.default_start)
end_date = st.sidebar.date_input("End Date", value=date.today())
ma_window = st.sidebar.slider("Moving Average Window (days)", 5, 200, 50, 5)

if ticker:
    df = load_data(ticker, start_date, end_date)
    if df.empty:
        st.error("No data for " + ticker)
        st.stop()

    df = prepare_returns(df)
    metrics = compute_metrics(df)

    st.subheader(f"{ticker} — Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    render_metric(c1, "Latest Close", f"${metrics['latest_close']:,.2f}", f"{metrics['arrow']} {metrics['daily_change_pct']:.2%}")
    render_metric(c2, "Total Return", f"{metrics['total_return']:.2%}")
    render_metric(c3, "Annualized Return", f"{metrics['ann_return']:.2%}")
    render_metric(c4, "Sharpe Ratio", f"{metrics['sharpe']:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    render_metric(c5, "Annualized Volatility", f"{metrics['ann_volatility']:.2%}")
    render_metric(c6, "Skewness", f"{metrics['skewness']:.2f}")
    render_metric(c7, "Excess Kurtosis", f"{metrics['kurtosis']:.2f}")
    render_metric(c8, "Avg Daily Return", f"{metrics['avg_daily_ret']:.4%}")

    c9, c10, _, _ = st.columns(4)
    render_metric(c9, "Period High", f"${metrics['max_close']:,.2f}")
    render_metric(c10, "Period Low", f"${metrics['min_close']:,.2f}")

    st.divider()
    st.subheader("Compare Stocks (Max 2)")

    option_by_ticker = {opt.split(" - ")[0]: opt for opt in STOCK_SEARCH_OPTIONS}

    if "compare_picker_main" not in st.session_state:
        matching_default = next((opt for opt in STOCK_SEARCH_OPTIONS if opt.startswith(f"{ticker} - ")), None)
        st.session_state.compare_picker_main = [matching_default] if matching_default else []

    selected_compare = st.multiselect(
        "Search by ticker or company name",
        options=STOCK_SEARCH_OPTIONS,
        key="compare_picker_main",
        max_selections=2,
        placeholder="Try 'Apple' or 'AAPL'",
    )

    with st.expander("Browse compare stocks by category", expanded=False):
        for category, tickers in COMPARE_CATEGORIES.items():
            st.markdown(f"**{category}**")
            cols = st.columns(4)
            for i, sym in enumerate(tickers):
                if sym not in option_by_ticker:
                    continue
                option_value = option_by_ticker[sym]
                is_selected = option_value in st.session_state.compare_picker_main
                button_label = f"{'✅ ' if is_selected else ''}{sym}"

                with cols[i % 4]:
                    if st.button(button_label, key=f"main_compare_pick_{category}_{sym}", use_container_width=True):
                        if is_selected:
                            st.session_state.compare_picker_main.remove(option_value)
                        elif len(st.session_state.compare_picker_main) < 2:
                            st.session_state.compare_picker_main.append(option_value)
                        else:
                            st.warning("You can compare a maximum of 2 stocks.")
                        st.rerun()
            st.divider()

    compare_tickers = [item.split(" - ")[0] for item in selected_compare]

    if len(compare_tickers) == 2:
        left_symbol, right_symbol = compare_tickers[0], compare_tickers[1]
        left_df = prepare_returns(load_data(left_symbol, start_date, end_date))
        right_df = prepare_returns(load_data(right_symbol, start_date, end_date))

        if left_df.empty or right_df.empty:
            st.warning("Unable to load one of the selected comparison stocks.")
        else:
            left_metrics = compute_metrics(left_df)
            right_metrics = compute_metrics(right_df)

            lcol, rcol = st.columns(2)
            with lcol:
                st.markdown(f"### {left_symbol}")
                render_metric(st, "Latest Close", f"${left_metrics['latest_close']:,.2f}", f"{left_metrics['arrow']} {left_metrics['daily_change_pct']:.2%}")
                render_metric(st, "Total Return", f"{left_metrics['total_return']:.2%}")
                render_metric(st, "Annualized Return", f"{left_metrics['ann_return']:.2%}")
                render_metric(st, "Sharpe Ratio", f"{left_metrics['sharpe']:.2f}")
                render_metric(st, "Annualized Volatility", f"{left_metrics['ann_volatility']:.2%}")
                render_metric(st, "Period High", f"${left_metrics['max_close']:,.2f}")
                render_metric(st, "Period Low", f"${left_metrics['min_close']:,.2f}")

            with rcol:
                st.markdown(f"### {right_symbol}")
                render_metric(st, "Latest Close", f"${right_metrics['latest_close']:,.2f}", f"{right_metrics['arrow']} {right_metrics['daily_change_pct']:.2%}")
                render_metric(st, "Total Return", f"{right_metrics['total_return']:.2%}")
                render_metric(st, "Annualized Return", f"{right_metrics['ann_return']:.2%}")
                render_metric(st, "Sharpe Ratio", f"{right_metrics['sharpe']:.2f}")
                render_metric(st, "Annualized Volatility", f"{right_metrics['ann_volatility']:.2%}")
                render_metric(st, "Period High", f"${right_metrics['max_close']:,.2f}")
                render_metric(st, "Period Low", f"${right_metrics['min_close']:,.2f}")

            st.subheader("Comparison Price & Moving Average")
            fig_compare_price = go.Figure()
            fig_compare_price.add_trace(
                go.Scatter(
                    x=left_df.index,
                    y=left_df["Close"],
                    mode="lines",
                    name=f"{left_symbol} Close",
                    line=dict(width=2),
                )
            )
            fig_compare_price.add_trace(
                go.Scatter(
                    x=left_df.index,
                    y=left_df["Close"].rolling(window=ma_window).mean(),
                    mode="lines",
                    name=f"{left_symbol} {ma_window}-Day MA",
                    line=dict(width=2, dash="dash"),
                )
            )
            fig_compare_price.add_trace(
                go.Scatter(
                    x=right_df.index,
                    y=right_df["Close"],
                    mode="lines",
                    name=f"{right_symbol} Close",
                    line=dict(width=2),
                )
            )
            fig_compare_price.add_trace(
                go.Scatter(
                    x=right_df.index,
                    y=right_df["Close"].rolling(window=ma_window).mean(),
                    mode="lines",
                    name=f"{right_symbol} {ma_window}-Day MA",
                    line=dict(width=2, dash="dash"),
                )
            )
            fig_compare_price.update_layout(
                template="plotly_white",
                height=450,
                hovermode="x unified",
                yaxis_title="Price (USD)",
                xaxis_title="Date",
            )
            st.plotly_chart(fig_compare_price, use_container_width=True)
    elif len(compare_tickers) == 1:
        st.info("Select one more stock to show side-by-side comparison.")

    st.divider()
    if len(compare_tickers) != 2:
        st.subheader("Price & Moving Average")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"].rolling(window=ma_window).mean(),
                mode="lines",
                name=f"{ma_window}-Day MA",
                line=dict(dash="dash"),
            )
        )
        fig.update_layout(template="plotly_white", height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Trading Volume")
    fig_vol = go.Figure()

    if len(compare_tickers) == 2:
        # Use the main ticker as background bars and overlay one compared ticker on top.
        overlay_symbol = next((sym for sym in compare_tickers if sym != ticker), compare_tickers[0])
        overlay_df = load_data(overlay_symbol, start_date, end_date)

        if not overlay_df.empty and "Volume" in overlay_df.columns and "Volume" in df.columns:
            overlay_df = overlay_df.reindex(df.index).ffill().bfill()

            fig_vol.add_trace(
                go.Bar(
                    x=df.index,
                    y=df["Volume"],
                    name=f"{ticker} Volume",
                    marker_color="lightsteelblue",
                    opacity=0.35,
                )
            )
            fig_vol.add_trace(
                go.Bar(
                    x=df.index,
                    y=overlay_df["Volume"],
                    name=f"{overlay_symbol} Volume",
                    marker_color="royalblue",
                    opacity=0.8,
                )
            )
            fig_vol.update_layout(template="plotly_white", height=300, barmode="overlay", hovermode="x unified")
        else:
            fig_vol.add_trace(go.Bar(x=df.index, y=df.get("Volume", []), name=f"{ticker} Volume"))
            fig_vol.update_layout(template="plotly_white", height=300)
    else:
        fig_vol.add_trace(go.Bar(x=df.index, y=df.get("Volume", []), name=f"{ticker} Volume"))
        fig_vol.update_layout(template="plotly_white", height=300)

    st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")
    csv = convert_df_to_csv(df)
    st.download_button("Download CSV (full)", data=csv, file_name=f"{ticker}_full.csv")
