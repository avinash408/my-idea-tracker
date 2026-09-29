import streamlit as st
import pandas as pd

st.set_page_config(page_title="NSE Sector & Stake Tracker", layout="wide")

@st.cache_data(ttl="1h")
def load_data():
    day_df = pd.read_parquet("data/processed/stock_day.parquet")
    shp_df = pd.read_parquet("data/processed/shareholding.parquet")
    return day_df, shp_df

try:
    day_df, shp_df = load_data()
    if "turnover_cr" not in day_df.columns:
        if "turnover" in day_df.columns:
            day_df["turnover_cr"] = pd.to_numeric(day_df["turnover"], errors="coerce") / 1e7
        elif "close" in day_df.columns and "volume" in day_df.columns:
            day_df["turnover_cr"] = (pd.to_numeric(day_df["close"], errors="coerce") * pd.to_numeric(day_df["volume"], errors="coerce")) / 1e7
        else:
            day_df["turnover_cr"] = 0.0
    if "rvol" not in day_df.columns:
        if "vol_surge_ratio" in day_df.columns:
            day_df["rvol"] = pd.to_numeric(day_df["vol_surge_ratio"], errors="coerce").fillna(1.0)
        elif "vol_sma20" in day_df.columns and "volume" in day_df.columns:
            day_df["rvol"] = (pd.to_numeric(day_df["volume"], errors="coerce") / pd.to_numeric(day_df["vol_sma20"], errors="coerce")).fillna(1.0)
        else:
            # Fallback calculation if rolling average was not precomputed
            day_df["vol_sma20"] = day_df.groupby("isin")["volume"].transform(
                lambda s: s.shift(1).rolling(20, min_periods=1).mean()
            )
            day_df["rvol"] = (day_df["volume"] / day_df["vol_sma20"]).fillna(1.0)
except Exception as e:
    st.error(f"Error loading Parquet files: {e}")
    st.stop()

st.title("📊 NSE Sector Tracker & Institutional Flow")

# Sidebar Filters
sectors = sorted(day_df["custom_sector"].dropna().unique())
selected_sector = st.sidebar.selectbox("Filter Sector", ["All"] + sectors)

sub_df = day_df if selected_sector == "All" else day_df[day_df["custom_sector"] == selected_sector]
industries = sorted(sub_df["custom_industry"].dropna().unique())
selected_industry = st.sidebar.selectbox("Filter Industry", ["All"] + industries)

target_df = sub_df if selected_industry == "All" else sub_df[sub_df["custom_industry"] == selected_industry]
latest_date = target_df["trade_date"].max()
latest_stocks = target_df[target_df["trade_date"] == latest_date].copy()

# Three main tabs
tab1, tab2, tab3 = st.tabs([
    "🚀 Price Returns", 
    "⚡ Volume Surge & Breakouts", 
    "🏛️ Institutional Stakes (FII/DII)"
])

# ----------------- TAB 1: RETURNS -----------------
with tab1:
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Daily Avg Return", f"{target_df['return_1d'].mean():.2f}%")
    col_m2.metric("Weekly Avg Return", f"{target_df['return_1w'].mean():.2f}%")
    col_m3.metric("Monthly Avg Return", f"{target_df['return_1m'].mean():.2f}%")

    st.markdown(f"#### Top 3 Gainers by Industry ({latest_date.strftime('%d-%b-%Y')})")
    timeframe = st.radio("Select Horizon", ["return_1d", "return_1w", "return_1m"], horizontal=True,
                         format_func=lambda x: {"return_1d": "Daily", "return_1w": "Weekly", "return_1m": "Monthly"}[x])

    latest_stocks["rank"] = latest_stocks.groupby("custom_industry")[timeframe].rank(ascending=False, method="dense")
    top_3 = latest_stocks[latest_stocks["rank"] <= 3].sort_values(["custom_industry", "rank"])
    
    st.dataframe(
        top_3[["custom_industry", "rank", "symbol", "company_name", "close", timeframe]]
        .style.format({"close": "₹{:.2f}", timeframe: "{:+.2f}%"}),
        use_container_width=True
    )

# ----------------- TAB 2: VOLUME SPIKES -----------------
with tab2:
    st.markdown(f"### ⚡ Volume Spikes & Unusual Activity ({latest_date.strftime('%d-%b-%Y')})")
    
    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        min_rvol = st.slider("Minimum Relative Volume (RVOL)", min_value=1.5, max_value=10.0, value=2.0, step=0.5,
                             help="2.0x means double the 20-day average volume")
        min_turnover = st.number_input("Min Turnover (₹ Cr)", min_value=0.0, value=1.0, step=0.5,
                                      help="Filters out illiquid micro-caps")

    # 1. Industry Aggregate Turnover Spike
    # Group by industry and date to calculate total industry turnover
    ind_turnover = day_df.groupby(["trade_date", "custom_industry"])["turnover_cr"].sum().reset_index()
    ind_turnover = ind_turnover.sort_values(["custom_industry", "trade_date"])
    ind_turnover["ind_turnover_sma20"] = ind_turnover.groupby("custom_industry")["turnover_cr"].rolling(20, min_periods=5).mean().reset_index(drop=True)
    ind_turnover["industry_rvol"] = (ind_turnover["turnover_cr"] / ind_turnover["ind_turnover_sma20"]).round(2)

    latest_ind = ind_turnover[ind_turnover["trade_date"] == latest_date].sort_values("industry_rvol", ascending=False)
    
    st.markdown("#### 🏭 Sectors / Industries with Largest Turnover Spike")
    st.dataframe(
        latest_ind[["custom_industry", "turnover_cr", "ind_turnover_sma20", "industry_rvol"]]
        .rename(columns={"turnover_cr": "Turnover Today (₹ Cr)", "ind_turnover_sma20": "20D Avg Turnover (₹ Cr)", "industry_rvol": "Volume Surge Multiple (RVOL)"})
        .head(10)
        .style.format({"Turnover Today (₹ Cr)": "₹{:.1f} Cr", "20D Avg Turnover (₹ Cr)": "₹{:.1f} Cr", "Volume Surge Multiple (RVOL)": "{:.2f}x"}),
        use_container_width=True
    )

    # 2. Stock Level Volume Spike Leaderboard
    st.markdown("#### 🚀 Stocks with Large Volume Spikes")
    spike_stocks = latest_stocks[(latest_stocks["rvol"] >= min_rvol) & (latest_stocks["turnover_cr"] >= min_turnover)].copy()
    spike_stocks["vol_rank"] = spike_stocks.groupby("custom_industry")["rvol"].rank(ascending=False, method="dense")
    spike_stocks = spike_stocks.sort_values(["rvol"], ascending=False)

    st.dataframe(
        spike_stocks[["custom_industry", "symbol", "company_name", "close", "return_1d", "volume", "vol_sma_20", "rvol", "turnover_cr"]]
        .rename(columns={
            "return_1d": "1D Return",
            "vol_sma_20": "20D Avg Volume",
            "rvol": "RVOL Multiple",
            "turnover_cr": "Turnover (₹ Cr)"
        })
        .style.format({
            "close": "₹{:.2f}",
            "1D Return": "{:+.2f}%",
            "volume": "{:,.0f}",
            "20D Avg Volume": "{:,.0f}",
            "RVOL Multiple": "{:.2f}x",
            "Turnover (₹ Cr)": "₹{:.2f} Cr"
        }),
        use_container_width=True
    )

# ----------------- TAB 3: SHAREHOLDING -----------------
with tab3:
    st.markdown("### Institutional Stake Changes (QoQ)")
    latest_quarter = shp_df["quarter_dt"].max()
    quarter_shp = shp_df[shp_df["quarter_dt"] == latest_quarter].copy()
    
    master_meta = day_df[["isin", "custom_sector", "custom_industry", "symbol"]].drop_duplicates("isin")
    quarter_shp = quarter_shp.merge(master_meta, on="isin", how="inner")
    
    if selected_sector != "All":
        quarter_shp = quarter_shp[quarter_shp["custom_sector"] == selected_sector]
    if selected_industry != "All":
        quarter_shp = quarter_shp[quarter_shp["custom_industry"] == selected_industry]

    stake_metric = st.selectbox("Rank By", ["FIIs_delta", "DIIs_delta", "Promoters_delta"])
    
    quarter_shp["stake_rank"] = quarter_shp.groupby("custom_industry")[stake_metric].rank(ascending=False, method="dense")
    top_stake = quarter_shp[quarter_shp["stake_rank"] <= 3].sort_values(["custom_industry", "stake_rank"])
    
    st.dataframe(
        top_stake[["custom_industry", "stake_rank", "symbol", "FIIs", "FIIs_delta", "DIIs", "DIIs_delta", "Promoters", "Promoters_delta"]]
        .style.format({
            "FIIs": "{:.2f}%", "FIIs_delta": "{:+.2f}%",
            "DIIs": "{:.2f}%", "DIIs_delta": "{:+.2f}%",
            "Promoters": "{:.2f}%", "Promoters_delta": "{:+.2f}%"
        }),
        use_container_width=True
    )
