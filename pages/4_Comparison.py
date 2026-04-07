import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import date, timedelta
from helpers import load_data, compute_benchmark, load_portfolio_data, render_top_nav

st.set_page_config(page_title="Comparison & Stats", layout="wide")
render_top_nav()
st.title("Cumulative Return Comparison")

# Initialize saved portfolios in session state
if "saved_portfolios" not in st.session_state:
    st.session_state.saved_portfolios = {}

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=date.today())

if ticker:
    df = load_data(ticker, start_date, end_date)
    if df.empty:
        st.error(f"No data for {ticker}")
    else:
        if "Close" in df.columns:
            df["Daily Return"] = df["Close"].pct_change()
            df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod() - 1

        st.subheader("Add stocks / ETFs to compare")
        
        # Define categories with company names
        comparison_categories = {
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
                "AMGN": "Amgen"
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
        
        # Create options list as "TICKER - Company Name"
        all_comparison_options = []
        for stocks in comparison_categories.values():
            for sym, name in stocks.items():
                all_comparison_options.append(f"{sym} - {name}")

        if "comparison_picker" not in st.session_state:
            st.session_state.comparison_picker = []

        option_by_ticker = {opt.split(" - ")[0]: opt for opt in all_comparison_options}
        
        comparison_tickers = st.multiselect(
            "Search and select stocks / ETFs to compare",
            options=sorted(all_comparison_options),
            key="comparison_picker",
            placeholder="Search by ticker (MSFT) or name (Microsoft)..."
        )
        
        # Extract just the tickers from the selected options
        comparison_tickers = [t.split(" - ")[0] for t in comparison_tickers]
        
        # Filter out the main ticker from comparison list
        comparison_tickers = [t.upper().strip() for t in comparison_tickers if t.upper().strip() != ticker]
        
        # Display categories as clickable browser
        with st.expander("📚 Browse available stocks by category", expanded=False):
            for category, stocks in comparison_categories.items():
                st.markdown(f"**{category}**")
                cols = st.columns(4)
                for i, (sym, name) in enumerate(stocks.items()):
                    option_value = option_by_ticker[sym]
                    selected = option_value in st.session_state.comparison_picker
                    button_label = f"{'✅ ' if selected else ''}{sym}"

                    with cols[i % 4]:
                        if st.button(button_label, key=f"compare_pick_{category}_{sym}", use_container_width=True):
                            if sym == ticker:
                                st.warning("Main ticker is already on the chart. Pick another stock to compare.")
                            elif option_value not in st.session_state.comparison_picker:
                                st.session_state.comparison_picker.append(option_value)
                            else:
                                st.session_state.comparison_picker.remove(option_value)
                            st.rerun()
                st.divider()
        
        # Saved portfolios section
        st.subheader("Compare Saved Portfolio Allocations")
        if st.session_state.saved_portfolios:
            saved_port_names = list(st.session_state.saved_portfolios.keys())
            selected_portfolios = st.multiselect("Add saved portfolios", options=saved_port_names, default=[])
        else:
            selected_portfolios = []
            st.info("No saved portfolios yet. Create and save one on the Portfolio Builder page.")

        benchmark = load_data("^GSPC", start_date, end_date)
        benchmark = compute_benchmark(benchmark)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Cumulative Return"], mode="lines", name=ticker, line=dict(width=2)))

        for comp in comparison_tickers:
            try:
                df_temp = load_data(comp, start_date, end_date)
                if not df_temp.empty and "Close" in df_temp.columns:
                    df_temp["Daily Return"] = df_temp["Close"].pct_change()
                    df_temp["Cumulative Return"] = (1 + df_temp["Daily Return"]).cumprod() - 1
                    df_temp = df_temp.reindex(df.index).ffill().bfill()
                    fig.add_trace(go.Scatter(x=df.index, y=df_temp["Cumulative Return"], mode="lines", name=comp, line=dict(width=2)))
            except Exception as e:
                st.warning(f"Failed to load {comp}")
        
        # Add saved portfolios to comparison
        for port_name in selected_portfolios:
            try:
                port_config = st.session_state.saved_portfolios[port_name]
                port_tickers = tuple(port_config["tickers"])
                port_weights = port_config["weights"]
                
                prices = load_portfolio_data(port_tickers, start_date, end_date)
                if not prices.empty:
                    # Calculate portfolio value
                    port_val = pd.Series(0.0, index=prices.index)
                    for pt in port_tickers:
                        if pt in prices.columns:
                            first_price = float(prices[pt].iloc[0])
                            shares = (port_config["starting_cash"] * port_weights[pt]) / first_price if first_price > 0 else 0
                            port_val += prices[pt] * shares
                    
                    # Calculate cumulative return
                    port_cum_ret = (port_val / port_val.iloc[0]) - 1
                    port_cum_ret = port_cum_ret.reindex(df.index).ffill().bfill()
                    fig.add_trace(go.Scatter(x=df.index, y=port_cum_ret, mode="lines", name=f"📊 {port_name}", line=dict(width=2.5)))
            except Exception as e:
                st.warning(f"Failed to load portfolio '{port_name}': {e}")

        if not benchmark.empty and "Cumulative Return" in benchmark.columns:
            fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark["Cumulative Return"], mode="lines", name="S&P 500", line=dict(dash="dash", width=2)))
        fig.update_layout(yaxis_title="Cumulative Return", yaxis_tickformat=".0%", template="plotly_white", height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
