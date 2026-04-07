import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math
from datetime import date, timedelta
from helpers import load_portfolio_data, render_top_nav

st.set_page_config(page_title="Portfolio Builder", layout="wide")
render_top_nav()
st.title("Portfolio Builder")

PORTFOLIO_METRIC_HELP = {
    "Starting Value": "Initial simulated cash amount used to build the portfolio.",
    "Current Portfolio Value": "Latest total value of all selected holdings based on current prices in the chosen date range.",
    "Total Return": "Overall percentage gain/loss from initial portfolio value to latest portfolio value.",
    "Number of Holdings": "Count of selected tickers with usable price data in the period.",
    "Annualized Return": "Return scaled to a one-year rate from average daily portfolio return.",
    "Annualized Volatility": "Estimated yearly volatility of daily portfolio returns (daily std × sqrt(252)).",
    "Sharpe Ratio": "Risk-adjusted return; annualized return divided by annualized volatility (risk-free rate assumed 0).",
    "Avg Daily Return": "Average daily percentage return of the portfolio over the selected period.",
    "Skewness": "Asymmetry of portfolio daily returns. Positive suggests more upside tail events.",
    "Excess Kurtosis": "Tail heaviness of portfolio returns versus a normal distribution. Higher suggests more extreme moves.",
    "Period High": "Highest portfolio value reached during the selected date range.",
    "Period Low": "Lowest portfolio value reached during the selected date range.",
    "Max Drawdown": "Largest peak-to-trough decline in portfolio value during the selected period.",
    "Unrealized P&L": "Current dollar profit or loss versus starting value.",
    "Total Invested": "Total dollars actually deployed into positions after allocation calculations.",
    "Uninvested Cash": "Cash left unallocated or unused based on your selected weights.",
}


def render_portfolio_metric(col, label, value, delta=None):
    col.metric(label, value, delta=delta, help=PORTFOLIO_METRIC_HELP.get(label, ""))


def risk_profile_label(risk_slider_value):
    if risk_slider_value <= 20:
        return "Very Aggressive"
    if risk_slider_value <= 40:
        return "Aggressive"
    if risk_slider_value <= 60:
        return "Balanced"
    if risk_slider_value <= 80:
        return "Conservative"
    return "Very Conservative"


def interpolate_model_weights(slider_value, anchor_models, return_contributions=False):
    anchors = [0, 25, 50, 75, 100]
    model_order = ["Very Aggressive", "Aggressive", "Balanced", "Conservative", "Very Conservative"]
    contribution_by_preset = {preset: {} for preset in model_order}

    if slider_value <= anchors[0]:
        blended = anchor_models["Very Aggressive"].copy()
        contribution_by_preset["Very Aggressive"] = blended.copy()
        if return_contributions:
            return blended, contribution_by_preset
        return blended

    if slider_value >= anchors[-1]:
        blended = anchor_models["Very Conservative"].copy()
        contribution_by_preset["Very Conservative"] = blended.copy()
        if return_contributions:
            return blended, contribution_by_preset
        return blended

    left_index = 0
    for i in range(len(anchors) - 1):
        if anchors[i] <= slider_value <= anchors[i + 1]:
            left_index = i
            break

    right_index = left_index + 1
    left_anchor = anchors[left_index]
    right_anchor = anchors[right_index]
    mix = (slider_value - left_anchor) / (right_anchor - left_anchor)

    left_model = anchor_models[model_order[left_index]]
    right_model = anchor_models[model_order[right_index]]

    tickers = sorted(set(left_model.keys()) | set(right_model.keys()))
    blended = {}
    for ticker in tickers:
        lw = left_model.get(ticker, 0.0)
        rw = right_model.get(ticker, 0.0)
        left_contribution = (1 - mix) * lw
        right_contribution = mix * rw
        blended[ticker] = left_contribution + right_contribution

        if left_contribution > 0:
            contribution_by_preset[model_order[left_index]][ticker] = left_contribution
        if right_contribution > 0:
            contribution_by_preset[model_order[right_index]][ticker] = right_contribution

    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items() if v > 0}

    if return_contributions:
        normalized_contributions = {}
        for preset, preset_contributions in contribution_by_preset.items():
            normalized_contributions[preset] = {
                ticker: value / total for ticker, value in preset_contributions.items()
            } if total > 0 else {}
        return blended, normalized_contributions

    return blended

# Initialize saved portfolios in session state
if "saved_portfolios" not in st.session_state:
    st.session_state.saved_portfolios = {}

# Sidebar inputs
starting_cash = st.sidebar.number_input("Starting Fake Cash ($)", min_value=100.0, value=10000.0, step=500.0)
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=date.today())

st.subheader("Choose portfolio stocks / ETFs")

# Define categories with company names
categories = {
    "🔷 TECHNOLOGY": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Alphabet/Google",
        "AMZN": "Amazon",
        "TSLA": "Tesla",
        "META": "Meta",
        "NVDA": "NVIDIA",
        "AMD": "AMD",
        "INTC": "Intel",
        "QCOM": "Qualcomm",
        "CRM": "Salesforce",
        "ADBE": "Adobe",
        "ORCL": "Oracle",
        "AVGO": "Broadcom"
    },
    "🟦 BLUE CHIP / INDUSTRIALS": {
        "BRK-B": "Berkshire Hathaway B",
        "JNJ": "Johnson & Johnson",
        "PG": "Procter & Gamble",
        "KO": "Coca-Cola",
        "PEP": "PepsiCo",
        "WMT": "Walmart",
        "COST": "Costco",
        "CAT": "Caterpillar",
        "DE": "Deere",
        "GE": "General Electric",
        "HON": "Honeywell",
        "BA": "Boeing",
        "MMM": "3M"
    },
    "🏦 FINANCIALS / BANKING": {
        "JPM": "JPMorgan Chase",
        "BAC": "Bank of America",
        "GS": "Goldman Sachs",
        "MS": "Morgan Stanley",
        "WFC": "Wells Fargo",
        "USB": "U.S. Bancorp",
        "TFC": "Truist"
    },
    "💊 HEALTHCARE / PHARMA": {
        "PFE": "Pfizer",
        "UNH": "UnitedHealth",
        "LLY": "Eli Lilly",
        "MRK": "Merck",
        "AZN": "AstraZeneca",
        "AMGN": "Amgen",
    },
    "⛽ ENERGY": {
        "XOM": "ExxonMobil",
        "CVX": "Chevron",
        "SLB": "Schlumberger",
        "EOG": "EOG Resources",
        "COP": "ConocoPhillips",
        "MPC": "Marathon Petroleum"
    },
    "🛒 CONSUMER": {
        "MCD": "McDonald's",
        "SBUX": "Starbucks",
        "NKE": "Nike",
        "HD": "Home Depot",
        "LOW": "Lowe's",
        "TGT": "Target",
        "CL": "Colgate-Palmolive"
    },
    "📺 MEDIA / ENTERTAINMENT": {
        "DIS": "Disney",
        "NFLX": "Netflix",
        "UBER": "Uber",
        "LYFT": "Lyft",
        "VNQ": "Vanguard Real Estate ETF"
    },
    "📈 BROAD EQUITY ETFs": {
        "SPY": "SPDR S&P 500 ETF",
        "VOO": "Vanguard S&P 500 ETF",
        "IVV": "iShares Core S&P 500 ETF",
        "QQQ": "Invesco QQQ Trust",
        "VTI": "Vanguard Total Stock Market ETF",
        "IWM": "iShares Russell 2000 ETF"
    },
    "🌍 INTL / EMERGING MARKETS / SECTORS": {
        "VEA": "Vanguard FTSE Developed Markets ETF",
        "VXUS": "Vanguard Total International Stock ETF",
        "EFA": "iShares MSCI EAFE ETF",
        "EEM": "iShares MSCI Emerging Markets ETF",
        "XLF": "Financial Select Sector SPDR ETF",
        "XLK": "Technology Select Sector SPDR ETF",
        "XLE": "Energy Select Sector SPDR ETF",
        "XLV": "Health Care Select Sector SPDR ETF",
        "XLI": "Industrial Select Sector SPDR ETF",
        "XLP": "Consumer Staples Select Sector SPDR ETF",
        "XLY": "Consumer Discretionary Select Sector SPDR ETF",
        "XLU": "Utilities Select Sector SPDR ETF"
    },
    "💰 FIXED INCOME / BONDS": {
        "BND": "Vanguard Total Bond Market ETF",
        "AGG": "iShares Core U.S. Aggregate Bond ETF",
        "TLT": "iShares 20+ Year Treasury Bond ETF",
        "IEF": "iShares 7-10 Year Treasury Bond ETF",
        "SHY": "iShares 1-3 Year Treasury Bond ETF",
        "BIL": "SPDR Bloomberg 1-3 Month T-Bill ETF",
        "TIP": "iShares TIPS Bond ETF",
        "LQD": "iShares Investment Grade Corporate Bond ETF",
        "HYG": "iShares High Yield Corporate Bond ETF",
        "MUB": "iShares National Muni Bond ETF",
        "BSV": "Vanguard Short-Term Bond ETF",
        "BIV": "Vanguard Intermediate-Term Bond ETF",
        "BLV": "Vanguard Long-Term Bond ETF"
    },
    "🪙 COMMODITIES / REAL ASSETS": {
        "GLD": "SPDR Gold Shares",
        "SLV": "iShares Silver Trust",
        "DBC": "Commodities ETF",
        "USO": "Oil ETF"
    }
}

risk_model_portfolios = {
    "Very Aggressive": {
        "QQQ": 0.35,
        "XLK": 0.20,
        "EEM": 0.15,
        "XLE": 0.10,
        "SPY": 0.15,
        "BND": 0.05,
    },
    "Aggressive": {
        "QQQ": 0.25,
        "SPY": 0.25,
        "VXUS": 0.15,
        "XLF": 0.10,
        "BND": 0.15,
        "GLD": 0.10,
    },
    "Balanced": {
        "SPY": 0.25,
        "VXUS": 0.15,
        "XLV": 0.10,
        "BND": 0.30,
        "IEF": 0.10,
        "GLD": 0.10,
    },
    "Conservative": {
        "SPY": 0.15,
        "VXUS": 0.10,
        "BND": 0.35,
        "AGG": 0.20,
        "SHY": 0.10,
        "GLD": 0.10,
    },
    "Very Conservative": {
        "SPY": 0.10,
        "BND": 0.35,
        "AGG": 0.25,
        "SHY": 0.20,
        "BIL": 0.05,
        "GLD": 0.05,
    },
}

ticker_to_name = {}
for stocks in categories.values():
    for ticker, name in stocks.items():
        ticker_to_name[ticker] = name

reference_metrics = {
    "Very Aggressive": {
        "Annualized Return": "12-15%",
        "Annualized Volatility": "18-22%",
        "Sharpe Ratio": "0.6-0.8",
        "Max Drawdown": "-35% to -45%",
    },
    "Aggressive": {
        "Annualized Return": "9-12%",
        "Annualized Volatility": "12-16%",
        "Sharpe Ratio": "0.7-0.9",
        "Max Drawdown": "-25% to -35%",
    },
    "Balanced": {
        "Annualized Return": "6-8%",
        "Annualized Volatility": "8-12%",
        "Sharpe Ratio": "0.8-1.0",
        "Max Drawdown": "-15% to -25%",
    },
    "Conservative": {
        "Annualized Return": "3-5%",
        "Annualized Volatility": "4-8%",
        "Sharpe Ratio": "0.7-0.9",
        "Max Drawdown": "-8% to -15%",
    },
    "Very Conservative": {
        "Annualized Return": "1-3%",
        "Annualized Volatility": "2-4%",
        "Sharpe Ratio": "0.5-0.8",
        "Max Drawdown": "-3% to -8%",
    },
}

# Create options list as "TICKER - Company Name"
all_options = []
for stocks in categories.values():
    for ticker, name in stocks.items():
        all_options.append(f"{ticker} - {name}")

if "portfolio_picker" not in st.session_state:
    st.session_state.portfolio_picker = []

portfolio_tickers = st.multiselect(
    "Search and select stocks / ETFs",
    options=sorted(all_options),
    key="portfolio_picker",
    placeholder="Search by ticker (AAPL) or name (Apple)..."
)

# Extract just the tickers from the selected options
portfolio_tickers = [t.split(" - ")[0] for t in portfolio_tickers]

# Display categories as reference
with st.expander("📚 Browse available stocks by category", expanded=False):
    for category, stocks in categories.items():
        st.markdown(f"**{category}**")
        cols = st.columns(4)
        for i, (ticker, name) in enumerate(stocks.items()):
            option_value = f"{ticker} - {name}"
            selected = option_value in st.session_state.portfolio_picker
            button_label = f"{'✅ ' if selected else ''}{ticker}"
            with cols[i % 4]:
                if st.button(button_label, key=f"pick_{category}_{ticker}", use_container_width=True):
                    if option_value not in st.session_state.portfolio_picker:
                        st.session_state.portfolio_picker.append(option_value)
                    else:
                        st.session_state.portfolio_picker.remove(option_value)
                    st.rerun()
        st.divider()
weight_mode = st.radio(
    "Allocation Method",
    options=["Equal Weight", "Manual Weight", "Risk Model Portfolios"],
)

weights = {}
preset_portfolio_name = None
if portfolio_tickers or weight_mode == "Risk Model Portfolios":
    if weight_mode == "Equal Weight":
        w = 1/len(portfolio_tickers)
        weights = {t: w for t in portfolio_tickers}
    elif weight_mode == "Manual Weight":
        st.markdown("#### Enter Manual Weights")
        cols = st.columns(min(len(portfolio_tickers),4))
        total = 0.0
        for i, t in enumerate(portfolio_tickers):
            with cols[i % len(cols)]:
                v = st.number_input(f"{t} weight (%)", min_value=0.0, max_value=100.0, value=float(round(100/len(portfolio_tickers),2)), step=1.0, key=f"w_{t}")
                weights[t] = v/100
                total += v
        
        if total > 100.0:
            st.error(f"⚠️ Weights exceed 100% (current: {total:.1f}%). Reduce allocations before analyzing.")
            st.stop()
        else:
            remaining = 100.0 - total
            if remaining > 0:
                st.info(f"📊 Allocated: {total:.1f}% | Unallocated: {remaining:.1f}%")
            else:
                st.success(f"✅ Fully allocated: {total:.1f}%")
    else:
        st.markdown("#### Risk-Based Portfolio Presets")
        st.caption("📊 *Historical reference metrics (actual results will vary based on market conditions and timeframe)*")
        
        metrics_rows = []
        for profile_name in ["Very Aggressive", "Aggressive", "Balanced", "Conservative", "Very Conservative"]:
            metrics = reference_metrics[profile_name]
            metrics_rows.append({
                "Profile": profile_name,
                "Typical Return": metrics["Annualized Return"],
                "Typical Volatility": metrics["Annualized Volatility"],
                "Typical Sharpe": metrics["Sharpe Ratio"],
                "Typical Max DD": metrics["Max Drawdown"],
            })
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)
        
        st.markdown("")
        risk_slider_value = st.slider(
            "Risk posture (0 = Risky, 100 = Extremely Stable)",
            min_value=0,
            max_value=100,
            value=50,
            help="Builds a full model portfolio and blends between preset risk profiles.",
        )
        weights = interpolate_model_weights(risk_slider_value, risk_model_portfolios)
        portfolio_tickers = list(weights.keys())

        profile_name = risk_profile_label(risk_slider_value)
        preset_portfolio_name = f"{profile_name} Model ({risk_slider_value}%)"
        st.success(f"Model generated: {profile_name}")

        with st.expander("View all risk presets", expanded=False):
            model_weight_matrix = pd.DataFrame(risk_model_portfolios).T.fillna(0.0)
            model_weight_matrix = model_weight_matrix.reindex(
                columns=sorted(model_weight_matrix.columns)
            )
            preset_risk_tickers = sorted(set().union(*[set(w.keys()) for w in risk_model_portfolios.values()]))
            preset_risk_prices = load_portfolio_data(tuple(preset_risk_tickers), start_date, end_date)
            preset_ticker_volatility = {}
            preset_ticker_sharpe = {}
            if not preset_risk_prices.empty:
                for t in preset_risk_tickers:
                    if t not in preset_risk_prices.columns:
                        preset_ticker_volatility[t] = np.nan
                        preset_ticker_sharpe[t] = np.nan
                        continue

                    series_returns = preset_risk_prices[t].pct_change().dropna()
                    if series_returns.empty:
                        preset_ticker_volatility[t] = np.nan
                        preset_ticker_sharpe[t] = np.nan
                        continue

                    ann_vol = float(series_returns.std() * math.sqrt(252))
                    ann_return = float((1 + series_returns.mean()) ** 252 - 1)
                    sharpe = np.nan if ann_vol == 0 else ann_return / ann_vol
                    preset_ticker_volatility[t] = ann_vol
                    preset_ticker_sharpe[t] = sharpe

            st.markdown("**Risk Preset Weight Matrix (%):**")
            st.dataframe((model_weight_matrix * 100).round(2), use_container_width=True)

            st.markdown("**Individual Weight Contributions by Preset:**")
            preset_tabs = st.tabs(list(risk_model_portfolios.keys()))
            for tab, (preset_label, model_weights) in zip(preset_tabs, risk_model_portfolios.items()):
                with tab:
                    breakdown_df = pd.DataFrame(
                        {
                            "Ticker": list(model_weights.keys()),
                            "Name": [ticker_to_name.get(t, t) for t in model_weights.keys()],
                            "Weight (%)": [w * 100 for w in model_weights.values()],
                            "Annualized Volatility": [
                                preset_ticker_volatility.get(t, np.nan)
                                for t in model_weights.keys()
                            ],
                            "Sharpe Ratio": [
                                preset_ticker_sharpe.get(t, np.nan)
                                for t in model_weights.keys()
                            ],
                            "Contribution ($)": [starting_cash * w for w in model_weights.values()],
                        }
                    ).sort_values("Weight (%)", ascending=False)
                    breakdown_df["Weight (%)"] = breakdown_df["Weight (%)"].round(2)
                    breakdown_df["Annualized Volatility"] = breakdown_df["Annualized Volatility"].map(
                        lambda x: "N/A" if pd.isna(x) else f"{x:.2%}"
                    )
                    breakdown_df["Sharpe Ratio"] = breakdown_df["Sharpe Ratio"].map(
                        lambda x: "N/A" if pd.isna(x) else f"{x:.2f}"
                    )
                    breakdown_df["Contribution ($)"] = breakdown_df["Contribution ($)"].round(2)
                    st.dataframe(breakdown_df, use_container_width=True)

        preview_df = pd.DataFrame(
            {
                "Ticker": portfolio_tickers,
                "Name": [ticker_to_name.get(t, t) for t in portfolio_tickers],
                "Auto Weight": [weights[t] for t in portfolio_tickers],
            }
        )

        # Add per-ticker risk metrics for the selected window.
        risk_prices = load_portfolio_data(tuple(portfolio_tickers), start_date, end_date)
        ticker_volatility = {}
        ticker_sharpe = {}
        if not risk_prices.empty:
            for t in portfolio_tickers:
                if t not in risk_prices.columns:
                    ticker_volatility[t] = np.nan
                    ticker_sharpe[t] = np.nan
                    continue

                series_returns = risk_prices[t].pct_change().dropna()
                if series_returns.empty:
                    ticker_volatility[t] = np.nan
                    ticker_sharpe[t] = np.nan
                    continue

                ann_vol = float(series_returns.std() * math.sqrt(252))
                ann_return = float((1 + series_returns.mean()) ** 252 - 1)
                sharpe = np.nan if ann_vol == 0 else ann_return / ann_vol
                ticker_volatility[t] = ann_vol
                ticker_sharpe[t] = sharpe

        preview_df["Annualized Volatility"] = [ticker_volatility.get(t, np.nan) for t in portfolio_tickers]
        preview_df["Sharpe Ratio"] = [ticker_sharpe.get(t, np.nan) for t in portfolio_tickers]

        preview_df = preview_df.sort_values("Auto Weight", ascending=False)
        preview_df["Auto Weight"] = preview_df["Auto Weight"].map(lambda x: f"{x:.2%}")
        preview_df["Annualized Volatility"] = preview_df["Annualized Volatility"].map(
            lambda x: "N/A" if pd.isna(x) else f"{x:.2%}"
        )
        preview_df["Sharpe Ratio"] = preview_df["Sharpe Ratio"].map(
            lambda x: "N/A" if pd.isna(x) else f"{x:.2f}"
        )
        st.dataframe(preview_df, use_container_width=True)

    # Save portfolio section
    st.divider()
    st.subheader("📁 Save Portfolio Allocation")
    save_col1, save_col2 = st.columns([3, 1])
    with save_col1:
        default_placeholder = "e.g., Growth Portfolio, Tech Heavy, Balanced"
        if preset_portfolio_name:
            default_placeholder = f"Auto suggestion: {preset_portfolio_name}"
        portfolio_name = st.text_input("Portfolio Name", placeholder=default_placeholder)
    with save_col2:
        save_button = st.button("Save Portfolio", type="primary")
    
    if save_button:
        final_portfolio_name = portfolio_name.strip() if portfolio_name else ""
        if not final_portfolio_name and preset_portfolio_name:
            final_portfolio_name = preset_portfolio_name

        if not final_portfolio_name:
            st.warning("Please enter a portfolio name before saving.")
        else:
            st.session_state.saved_portfolios[final_portfolio_name] = {
            "tickers": portfolio_tickers,
            "weights": weights,
            "starting_cash": starting_cash
            }
            st.success(f"✅ Saved: {final_portfolio_name}")
    
    if st.session_state.saved_portfolios:
        st.subheader("Saved Portfolios")
        for name in list(st.session_state.saved_portfolios.keys()):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                port = st.session_state.saved_portfolios[name]
                st.markdown(f"**{name}** — {', '.join(port['tickers'])} (${port['starting_cash']:,.0f})")
            with col2:
                if st.button("Load", key=f"load_{name}"):
                    st.session_state.portfolio_to_compare = name
                    st.info(f"Use '{name}' in Comparison page")
            with col3:
                if st.button("🗑️", key=f"delete_{name}"):
                    del st.session_state.saved_portfolios[name]
                    st.rerun()

if portfolio_tickers:
    prices = load_portfolio_data(tuple(portfolio_tickers), start_date, end_date)
    if prices.empty:
        st.error("No data for selected tickers")
    else:
        avail = [t for t in portfolio_tickers if t in prices.columns]
        first_prices = prices.iloc[0]
        shares = {}
        summary = []
        total_actually_invested = 0.0
        
        for t in avail:
            allocated = starting_cash * weights[t]
            p = float(first_prices[t])
            s = 0.0 if p <= 0 else allocated / p
            shares[t] = s
            
            # Calculate actual amount invested (after converting to shares)
            actual_invested = s * p
            total_actually_invested += actual_invested
            
            summary.append({
                "Ticker": t,
                "Weight": f"{weights[t]:.2%}",
                "Initial Price": f"${round(p,2):,.2f}",
                "Allocated": f"${round(allocated,2):,.2f}",
                "Actual Invested": f"${round(actual_invested,2):,.2f}",
                "Shares Purchased": round(s,4)
            })

        # Calculate uninvested cash
        uninvested_cash = starting_cash - total_actually_invested
        
        # Add uninvested cash as a summary row
        summary.append({
            "Ticker": "CASH",
            "Weight": "—",
            "Initial Price": "—",
            "Allocated": f"${round(starting_cash - total_actually_invested,2):,.2f}",
            "Actual Invested": "—",
            "Shares Purchased": "—"
        })

        port_val = pd.Series(0.0, index=prices.index)
        for t in avail:
            port_val += prices[t] * shares[t]
        port_ret = port_val.pct_change().dropna()
        port_cum = (port_val / port_val.iloc[0]) - 1

        # Display allocation table at bottom of save section
        st.markdown("#### Allocation Details")
        st.dataframe(pd.DataFrame(summary), use_container_width=True)

        st.divider()
        st.subheader("Portfolio Growth")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_val.index, y=port_val, mode="lines", name="Portfolio Value"))
        
        # Add markers for high and low points
        max_idx = port_val.idxmax()
        min_idx = port_val.idxmin()
        max_val = float(port_val.max())
        min_val = float(port_val.min())
        
        fig.add_trace(go.Scatter(
            x=[max_idx], 
            y=[max_val], 
            mode="markers", 
            name="High",
            marker=dict(size=12, color="green", symbol="star"),
            hovertemplate=f"<b>Peak</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: ${max_val:,.2f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=[min_idx], 
            y=[min_val], 
            mode="markers", 
            name="Low",
            marker=dict(size=12, color="red", symbol="star"),
            hovertemplate=f"<b>Trough</b><br>Date: %{{x|%Y-%m-%d}}<br>Value: ${min_val:,.2f}<extra></extra>"
        ))
        
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Portfolio Metrics")
        
        # Calculate additional metrics
        port_total_return = float(port_cum.iloc[-1])
        port_avg_daily = float(port_ret.mean())
        port_volatility = float(port_ret.std())
        port_ann_volatility = port_volatility * math.sqrt(252)
        port_ann_return = (1 + port_avg_daily) ** 252 - 1
        port_sharpe = (port_ann_return - 0.0) / port_ann_volatility if port_ann_volatility != 0 else np.nan
        port_skewness = float(port_ret.skew())
        port_kurtosis = float(port_ret.kurtosis())
        port_high = float(port_val.max())
        port_low = float(port_val.min())
        
        # Calculate max drawdown
        running_max = port_val.expanding().max()
        drawdown = (port_val - running_max) / running_max
        port_max_drawdown = float(drawdown.min())
        
        col1, col2, col3, col4 = st.columns(4)
        render_portfolio_metric(col1, "Starting Value", f"${starting_cash:,.2f}")
        render_portfolio_metric(col2, "Current Portfolio Value", f"${float(port_val.iloc[-1]):,.2f}")
        render_portfolio_metric(col3, "Total Return", f"{port_total_return:.2%}")
        render_portfolio_metric(col4, "Number of Holdings", f"{len(avail)}")
        
        col5, col6, col7, col8 = st.columns(4)
        render_portfolio_metric(col5, "Annualized Return", f"{port_ann_return:.2%}")
        render_portfolio_metric(col6, "Annualized Volatility", f"{port_ann_volatility:.2%}")
        render_portfolio_metric(col7, "Sharpe Ratio", f"{port_sharpe:.2f}")
        render_portfolio_metric(col8, "Avg Daily Return", f"{port_avg_daily:.4%}")
        
        col9, col10, col11, col12 = st.columns(4)
        render_portfolio_metric(col9, "Skewness", f"{port_skewness:.2f}")
        render_portfolio_metric(col10, "Excess Kurtosis", f"{port_kurtosis:.2f}")
        render_portfolio_metric(col11, "Period High", f"${port_high:,.2f}")
        render_portfolio_metric(col12, "Period Low", f"${port_low:,.2f}")
        
        col13, col14, col15, col16 = st.columns(4)
        render_portfolio_metric(col13, "Max Drawdown", f"{port_max_drawdown:.2%}")
        render_portfolio_metric(col14, "Unrealized P&L", f"${float(port_val.iloc[-1] - starting_cash):,.2f}")
        render_portfolio_metric(col15, "Total Invested", f"${total_actually_invested:,.2f}")
        render_portfolio_metric(col16, "Uninvested Cash", f"${uninvested_cash:,.2f}")
