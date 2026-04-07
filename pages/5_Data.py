import streamlit as st
from datetime import date, timedelta
from helpers import load_data, convert_df_to_csv, render_top_nav

st.set_page_config(page_title="Data & Download", layout="wide")
render_top_nav()
st.title("Data & Download")

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=date.today())

if ticker:
    df = load_data(ticker, start_date, end_date)
    if df.empty:
        st.error(f"No data for {ticker}")
    else:
        # Remove cumulative columns for display
        df_display = df.copy()
        cols_to_drop = [c for c in df_display.columns if "cumul" in c.lower()]
        if cols_to_drop:
            df_display = df_display.drop(columns=cols_to_drop)

        # Create CSV without cumulative columns
        df_for_download = df.copy()
        cols_to_drop = [c for c in df_for_download.columns if "cumul" in c.lower()]
        if cols_to_drop:
            df_for_download = df_for_download.drop(columns=cols_to_drop)

        csv_data = convert_df_to_csv(df_for_download)

        st.subheader("Data & Download")
        st.download_button("Download CSV (no cumulative)", data=csv_data, file_name=f"{ticker}_analysis_no_cum.csv", mime="text/csv")

        st.markdown("---")
        with st.expander("View Raw Data"):
            st.dataframe(df_display, use_container_width=True)
