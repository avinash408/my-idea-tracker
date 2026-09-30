from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="NSE Sector & Stake Tracker", layout="wide")

MASTER_PATH = Path("data/master/stock_master_final.csv")

@st.cache_data(ttl="1h")
def load_data():
    day_df = pd.read_parquet("data/processed/stock_day.parquet")
    shp_df = pd.read_parquet("data/processed/shareholding.parquet")
    return day_df, shp_df

try:
    day_df, shp_df = load_data()

    # 1. Recover index if 'isin' is set as the DataFrame index
    if "isin" not in day_df.columns and ("isin" in str(day_df.index.name).lower() or day_df.index.name == "isin"):
        day_df = day_df.reset_index()
    if "isin" not in shp_df.columns and ("isin" in str(shp_df.index.name).lower() or shp_df.index.name == "isin"):
        shp_df = shp_df.reset_index()

    # 2. Normalize 'symbol' column if merge created suffixes (symbol_x, symbol_master, etc.)
    if "symbol" not in day_df.columns:
        for alt in ["symbol_x", "symbol_master", "symbol_y", "TckrSymb"]:
            if alt in day_df.columns:
                day_df["symbol"] = day_df[alt]
                break

    # 3. Normalize 'isin' column if merge created suffixes
    if "isin" not in day_df.columns:
        for alt in ["isin_x", "isin_master", "isin_y", "ISIN"]:
            if alt in day_df.columns:
                day_df["isin"] = day_df[alt]
                break

    # 4. Standardize dates
    if "trade_date" in day_df.columns:
        day_df["trade_date"] = pd.to_datetime(day_df["trade_date"], errors="coerce")
    if "quarter_dt" in shp_df.columns:
        shp_df["quarter_dt"] = pd.to_datetime(shp_df["quarter_dt"], errors="coerce")

    # 5. Ensure numeric types for core trading metrics
    for col in ["close", "prev_close", "volume", "turnover"]:
        if col in day_df.columns:
            day_df[col] = pd.to_numeric(day_df[col], errors="coerce")

    # 6. Turnover standardization (in ₹ Crores: 1 Cr = 1e7 INR)
    if "turnover_cr" not in day_df.columns:
        if "turnover" in day_df.columns:
            day_df["turnover_cr"] = day_df["turnover"] / 1e7
        elif "close" in day_df.columns and "volume" in day_df.columns:
            day_df["turnover_cr"] = (day_df["close"] * day_df["volume"]) / 1e7
        else:
            day_df["turnover_cr"] = 0.0

    # 7. Safe trailing return calculations (1D, 1W, 1M)
    day_df = day_df.sort_values(["isin", "trade_date"])
    if "return_1d" not in day_df.columns and "close" in day_df.columns:
        day_df["return_1d"] = day_df.groupby("isin")["close"].pct_change(1) * 100
    if "return_1w" not in day_df.columns and "close" in day_df.columns:
        day_df["return_1w"] = day_df.groupby("isin")["close"].pct_change(5) * 100
    if "return_1m" not in day_df.columns and "close" in day_df.columns:
        day_df["return_1m"] = day_df.groupby("isin")["close"].pct_change(21) * 100

    for ret_col in ["return_1d", "return_1w", "return_1m"]:
        if ret_col not in day_df.columns:
            day_df[ret_col] = 0.0
        else:
            day_df[ret_col] = day_df[ret_col].fillna(0.0)

    # 8. Volume baseline & RVOL calculations
    if "vol_sma_20" not in day_df.columns:
        if "vol_sma20" in day_df.columns:
            day_df["vol_sma_20"] = day_df["vol_sma20"]
        elif "volume" in day_df.columns:
            day_df["vol_sma_20"] = day_df.groupby("isin")["volume"].transform(
                lambda s: s.shift(1).rolling(20, min_periods=1).mean()
            )
        else:
            day_df["vol_sma_20"] = 0.0
    day_df["vol_sma20"] = day_df["vol_sma_20"]

    if "rvol" not in day_df.columns:
        if "vol_surge_ratio" in day_df.columns:
            day_df["rvol"] = pd.to_numeric(day_df["vol_surge_ratio"], errors="coerce").fillna(1.0)
        elif "volume" in day_df.columns:
            day_df["rvol"] = (day_df["volume"] / day_df["vol_sma_20"].replace(0, np.nan)).fillna(1.0)
        else:
            day_df["rvol"] = 1.0

except Exception as e:
    st.error(f"Error loading or initializing Parquet files: {e}")
    st.stop()

st.title("📊 NSE Sector Tracker & Institutional Flow")

# Sidebar Filters
sectors = sorted(day_df["custom_sector"].dropna().unique()) if "custom_sector" in day_df.columns else []
selected_sector = st.sidebar.selectbox("Filter Sector", ["All"] + sectors)

sub_df = day_df if selected_sector == "All" else day_df[day_df["custom_sector"] == selected_sector]
industries = sorted(sub_df["custom_industry"].dropna().unique()) if "custom_industry" in sub_df.columns else []
selected_industry = st.sidebar.selectbox("Filter Industry", ["All"] + industries)

target_df = sub_df if selected_industry == "All" else sub_df[sub_df["custom_industry"] == selected_industry]

latest_date = target_df["trade_date"].max()
latest_stocks = target_df[target_df["trade_date"] == latest_date].copy()
date_str = latest_date.strftime("%d-%b-%Y") if pd.notna(latest_date) else "Latest"

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

    st.markdown(f"#### Top 3 Gainers by Industry ({date_str})")
    timeframe = st.radio(
        "Select Return Horizon", 
        ["return_1d", "return_1w", "return_1m"], 
        horizontal=True,
        format_func=lambda x: {"return_1d": "Daily", "return_1w": "Weekly", "return_1m": "Monthly"}.get(x, x)
    )

    if "custom_industry" in latest_stocks.columns:
        latest_stocks["rank"] = latest_stocks.groupby("custom_industry")[timeframe].rank(ascending=False, method="dense")
        top_3 = latest_stocks[latest_stocks["rank"] <= 3].sort_values(["custom_industry", "rank"]).copy()

        desired_cols = ["custom_industry", "rank", "symbol", "company_name", "close", timeframe]
        avail_cols = [c for c in desired_cols if c in top_3.columns]

        fmt_dict = {}
        if "close" in avail_cols:
            fmt_dict["close"] = "₹{:.2f}"
        if timeframe in avail_cols:
            fmt_dict[timeframe] = "{:+.2f}%"

        st.dataframe(top_3[avail_cols].style.format(fmt_dict), use_container_width=True)

# ----------------- TAB 2: VOLUME SPIKES -----------------
with tab2:
    st.markdown(f"### ⚡ Volume Spikes & Unusual Activity ({date_str})")

    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        min_rvol = st.slider("Minimum Relative Volume (RVOL)", min_value=1.5, max_value=10.0, value=2.0, step=0.5,
                             help="2.0x means double the 20-day average volume")
        min_turnover = st.number_input("Min Turnover (₹ Cr)", min_value=0.0, value=1.0, step=0.5,
                                       help="Filters out illiquid micro-caps")

    # 1. Industry Aggregate Turnover Surge
    if "custom_industry" in day_df.columns:
        ind_turnover = day_df.groupby(["trade_date", "custom_industry"])["turnover_cr"].sum().reset_index()
        ind_turnover = ind_turnover.sort_values(["custom_industry", "trade_date"])
        ind_turnover["ind_turnover_sma20"] = (
            ind_turnover.groupby("custom_industry")["turnover_cr"]
            .transform(lambda s: s.shift(1).rolling(20, min_periods=3).mean())
        )
        ind_turnover["industry_rvol"] = (ind_turnover["turnover_cr"] / ind_turnover["ind_turnover_sma20"]).round(2)

        latest_ind = ind_turnover[ind_turnover["trade_date"] == latest_date].sort_values("industry_rvol", ascending=False).dropna(subset=["industry_rvol"])

        st.markdown("#### 🏭 Industries with Largest Turnover Surge")
        if not latest_ind.empty:
            st.dataframe(
                latest_ind[["custom_industry", "turnover_cr", "ind_turnover_sma20", "industry_rvol"]]
                .rename(columns={
                    "turnover_cr": "Turnover Today (₹ Cr)", 
                    "ind_turnover_sma20": "20D Avg Turnover (₹ Cr)", 
                    "industry_rvol": "Volume Surge Multiple (RVOL)"
                })
                .head(10)
                .style.format({
                    "Turnover Today (₹ Cr)": "₹{:.1f} Cr", 
                    "20D Avg Turnover (₹ Cr)": "₹{:.1f} Cr", 
                    "Volume Surge Multiple (RVOL)": "{:.2f}x"
                }),
                use_container_width=True
            )
        else:
            st.info("Insufficient historical sessions to compute 20-day industry turnover baselines.")

    # 2. Stock Level Volume Spike Leaderboard
    st.markdown("#### 🚀 Stocks with Large Volume Spikes")
    spike_stocks = latest_stocks[(latest_stocks["rvol"] >= min_rvol) & (latest_stocks["turnover_cr"] >= min_turnover)].copy()

    if not spike_stocks.empty and "custom_industry" in spike_stocks.columns:
        spike_stocks["vol_rank"] = spike_stocks.groupby("custom_industry")["rvol"].rank(ascending=False, method="dense")
        spike_stocks = spike_stocks.sort_values(["rvol"], ascending=False)

        cols_stock_view = ["custom_industry", "symbol", "company_name", "close", "return_1d", "volume", "vol_sma_20", "rvol", "turnover_cr"]
        avail_stock_cols = [c for c in cols_stock_view if c in spike_stocks.columns]

        st.dataframe(
            spike_stocks[avail_stock_cols]
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
    else:
        st.info(f"No stocks found with RVOL ≥ {min_rvol}x and Turnover ≥ ₹{min_turnover} Cr today.")

# ----------------- TAB 3: SHAREHOLDING -----------------
with tab3:
    st.markdown("### Institutional Stake Changes (QoQ)")
    
    if shp_df.empty or "quarter_dt" not in shp_df.columns:
        st.info("Quarterly shareholding dataset is currently empty or updating.")
    else:
        latest_quarter = shp_df["quarter_dt"].max()
        quarter_str = latest_quarter.strftime("%b %Y") if pd.notna(latest_quarter) else "Latest"
        st.caption(f"Showing filings for quarter: **{quarter_str}**")

        quarter_shp = shp_df[shp_df["quarter_dt"] == latest_quarter].copy()

        # Build clean metadata directly from master CSV (priority) or day_df (fallback)
        if MASTER_PATH.exists():
            master_raw = pd.read_csv(MASTER_PATH, dtype=str)
            target_meta_cols = [c for c in ["isin", "custom_sector", "custom_industry", "symbol"] if c in master_raw.columns]
            master_meta = master_raw[target_meta_cols].drop_duplicates("isin")
        else:
            avail_meta_cols = [c for c in ["isin", "custom_sector", "custom_industry", "symbol"] if c in day_df.columns]
            master_meta = day_df[avail_meta_cols].drop_duplicates("isin")

        # Drop any overlapping non-key columns before merging to prevent _x/_y suffix collision
        for overlap_col in ["custom_sector", "custom_industry", "symbol"]:
            if overlap_col in quarter_shp.columns and overlap_col in master_meta.columns:
                quarter_shp = quarter_shp.drop(columns=[overlap_col])

        quarter_shp = quarter_shp.merge(master_meta, on="isin", how="inner")

        if selected_sector != "All" and "custom_sector" in quarter_shp.columns:
            quarter_shp = quarter_shp[quarter_shp["custom_sector"] == selected_sector]
        if selected_industry != "All" and "custom_industry" in quarter_shp.columns:
            quarter_shp = quarter_shp[quarter_shp["custom_industry"] == selected_industry]

        avail_metrics = [m for m in ["FIIs_delta", "DIIs_delta", "Promoters_delta"] if m in quarter_shp.columns]
        
        if avail_metrics and "custom_industry" in quarter_shp.columns:
            stake_metric = st.selectbox("Rank By Stake Change", avail_metrics)
            quarter_shp["stake_rank"] = quarter_shp.groupby("custom_industry")[stake_metric].rank(ascending=False, method="dense")
            top_stake = quarter_shp[quarter_shp["stake_rank"] <= 3].sort_values(["custom_industry", "stake_rank"]).copy()

            shp_cols = ["custom_industry", "stake_rank", "symbol", "FIIs", "FIIs_delta", "DIIs", "DIIs_delta", "Promoters", "Promoters_delta"]
            avail_shp_cols = [c for c in shp_cols if c in top_stake.columns]

            shp_fmt = {
                "FIIs": "{:.2f}%", "FIIs_delta": "{:+.2f}%",
                "DIIs": "{:.2f}%", "DIIs_delta": "{:+.2f}%",
                "Promoters": "{:.2f}%", "Promoters_delta": "{:+.2f}%"
            }
            active_shp_fmt = {k: v for k, v in shp_fmt.items() if k in avail_shp_cols}

            st.dataframe(top_stake[avail_shp_cols].style.format(active_shp_fmt), use_container_width=True)
        else:
            st.warning("Stake change delta columns (FIIs_delta, DIIs_delta, Promoters_delta) not found in shareholding dataset.")
