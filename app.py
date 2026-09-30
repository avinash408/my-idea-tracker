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

    # 1. Recover index if 'isin' was stored as the index
    if "isin" not in day_df.columns and ("isin" in str(day_df.index.name).lower() or day_df.index.name == "isin"):
        day_df = day_df.reset_index()
    if "isin" not in shp_df.columns and ("isin" in str(shp_df.index.name).lower() or shp_df.index.name == "isin"):
        shp_df = shp_df.reset_index()

    # 2. Normalize 'symbol' column
    if "symbol" not in day_df.columns:
        for alt in ["symbol_x", "symbol_master", "symbol_y", "TckrSymb"]:
            if alt in day_df.columns:
                day_df["symbol"] = day_df[alt]
                break

    # 3. Normalize 'isin' column
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

    # 5. Numeric conversions
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

    # 7. Multi-horizon return calculations (1D, 3D, 4D, 1W, 1M)
    day_df = day_df.sort_values(["isin", "trade_date"])
    if "return_1d" not in day_df.columns and "close" in day_df.columns:
        day_df["return_1d"] = day_df.groupby("isin")["close"].pct_change(1) * 100
    
    # 3-Day and 4-Day trailing returns
    if "return_3d" not in day_df.columns and "close" in day_df.columns:
        day_df["return_3d"] = day_df.groupby("isin")["close"].pct_change(3) * 100
    if "return_4d" not in day_df.columns and "close" in day_df.columns:
        day_df["return_4d"] = day_df.groupby("isin")["close"].pct_change(4) * 100

    if "return_1w" not in day_df.columns and "close" in day_df.columns:
        day_df["return_1w"] = day_df.groupby("isin")["close"].pct_change(5) * 100
    if "return_1m" not in day_df.columns and "close" in day_df.columns:
        day_df["return_1m"] = day_df.groupby("isin")["close"].pct_change(21) * 100

    for ret_col in ["return_1d", "return_3d", "return_4d", "return_1w", "return_1m"]:
        if ret_col not in day_df.columns:
            day_df[ret_col] = 0.0
        else:
            day_df[ret_col] = day_df[ret_col].fillna(0.0)

    # 8. Volume baseline & RVOL calculations (20D baseline)
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

st.title("📊 NSE Sector Tracker & Momentum Scanner")

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

# App Navigation Tabs
tab_surge, tab_returns, tab_vol, tab_shp = st.tabs([
    "🔥 Momentum & Surge Sectors (3-4D)",
    "🚀 Price Returns",
    "⚡ Daily Volume Spikes",
    "🏛️ Institutional Stakes (FII/DII)"
])

# ----------------- TAB 0: 3-4 DAY POSITIVE SECTORS + 2-MONTH VOLUME SURGE -----------------
with tab_surge:
    st.markdown(f"### 🔥 Positive Sectors with 2-Month Volume Surge ({date_str})")
    st.caption("Filters sectors/industries with positive 3-day or 4-day cumulative returns and surging volume relative to their 2-month (~42 trading days) average.")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        lookback_days = st.radio("Lookback Period for Positive Momentum", [3, 4], horizontal=True,
                                 format_func=lambda x: f"Last {x} Trading Days")
    with col_s2:
        min_surge_mult = st.slider("Minimum 2-Month Volume Surge", min_value=0.8, max_value=4.0, value=1.2, step=0.1,
                                   help="1.2x means trading at 120% of its trailing 2-month average turnover")
    with col_s3:
        group_level = st.radio("Classification Level", ["custom_sector", "custom_industry"], horizontal=True,
                               format_func=lambda x: "Coarse (Sector)" if x == "custom_sector" else "Granular (Industry)")

    ret_col_target = f"return_{lookback_days}d"

    if group_level in day_df.columns:
        # 1. Aggregate daily turnover and average return per sector/industry
        sec_agg = (
            day_df.groupby(["trade_date", group_level])
            .agg(
                turnover_cr=("turnover_cr", "sum"),
                avg_return=(ret_col_target, "mean"),
                ret_1d=("return_1d", "mean"),
                stock_count=("isin", "nunique")
            )
            .reset_index()
            .sort_values([group_level, "trade_date"])
        )

        # 2. Two-month (42 trading days) baseline calculation
        # Uses shift(1) to compare today's turnover strictly against prior baseline
        sec_agg["turnover_2m_sma"] = (
            sec_agg.groupby(group_level)["turnover_cr"]
            .transform(lambda s: s.shift(1).rolling(42, min_periods=5).mean())
        )
        sec_agg["surge_2m_mult"] = (sec_agg["turnover_cr"] / sec_agg["turnover_2m_sma"]).round(2)

        # 3. Filter for latest trading session
        latest_sec = sec_agg[sec_agg["trade_date"] == latest_date].copy()

        # 4. Conditions: Positive in last 3-4 days AND volume surge >= threshold
        qualified_sectors = latest_sec[
            (latest_sec["avg_return"] > 0) & 
            (latest_sec["surge_2m_mult"] >= min_surge_mult)
        ].sort_values("surge_2m_mult", ascending=False)

        if not qualified_sectors.empty:
            st.markdown(f"#### Identified **{len(qualified_sectors)}** Sectors with Positive {lookback_days}D Return & 2-Month Volume Accumulation")
            st.dataframe(
                qualified_sectors[[group_level, "avg_return", "ret_1d", "turnover_cr", "turnover_2m_sma", "surge_2m_mult", "stock_count"]]
                .rename(columns={
                    group_level: "Sector / Industry",
                    "avg_return": f"Avg {lookback_days}D Return",
                    "ret_1d": "Today's Return",
                    "turnover_cr": "Today's Turnover (₹ Cr)",
                    "turnover_2m_sma": "2-Month Avg Turnover (₹ Cr)",
                    "surge_2m_mult": "2M Volume Surge Ratio",
                    "stock_count": "Total Stocks"
                })
                .style.format({
                    f"Avg {lookback_days}D Return": "{:+.2f}%",
                    "Today's Return": "{:+.2f}%",
                    "Today's Turnover (₹ Cr)": "₹{:.1f} Cr",
                    "2-Month Avg Turnover (₹ Cr)": "₹{:.1f} Cr",
                    "2M Volume Surge Ratio": "{:.2f}x",
                    "Total Stocks": "{:.0f}"
                }),
                use_container_width=True
            )

            # Drill-down: Top constituent stocks in these surging sectors
            st.markdown("#### 🎯 Leading Stocks in Surging Positive Sectors")
            surging_names = qualified_sectors[group_level].tolist()
            drill_stocks = latest_stocks[latest_stocks[group_level].isin(surging_names)].copy()
            
            # Rank stocks by 3-4 day return inside the surging sector
            drill_stocks["rank_in_sector"] = drill_stocks.groupby(group_level)[ret_col_target].rank(ascending=False, method="dense")
            top_constituents = drill_stocks[drill_stocks["rank_in_sector"] <= 3].sort_values([group_level, "rank_in_sector"])

            st.dataframe(
                top_constituents[[group_level, "rank_in_sector", "symbol", "company_name", "close", ret_col_target, "return_1d", "turnover_cr", "rvol"]]
                .rename(columns={
                    group_level: "Sector / Industry",
                    "rank_in_sector": "Rank",
                    ret_col_target: f"{lookback_days}D Return",
                    "return_1d": "1D Return",
                    "turnover_cr": "Turnover (₹ Cr)",
                    "rvol": "Daily RVOL"
                })
                .style.format({
                    "close": "₹{:.2f}",
                    f"{lookback_days}D Return": "{:+.2f}%",
                    "1D Return": "{:+.2f}%",
                    "Turnover (₹ Cr)": "₹{:.2f} Cr",
                    "Daily RVOL": "{:.2f}x"
                }),
                use_container_width=True
            )
        else:
            st.info(f"No sectors met both criteria: Positive {lookback_days}-day return (> 0%) and 2-Month volume surge ≥ {min_surge_mult}x. Try adjusting the surge slider.")
    else:
        st.warning(f"Classification column '{group_level}' not found in dataset.")

# ----------------- TAB 1: RETURNS -----------------
with tab_returns:
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Daily Avg Return", f"{target_df['return_1d'].mean():.2f}%")
    col_m2.metric("Weekly Avg Return", f"{target_df['return_1w'].mean():.2f}%")
    col_m3.metric("Monthly Avg Return", f"{target_df['return_1m'].mean():.2f}%")

    st.markdown(f"#### Top 3 Gainers by Industry ({date_str})")
    timeframe = st.radio(
        "Select Return Horizon", 
        ["return_1d", "return_3d", "return_4d", "return_1w", "return_1m"], 
        horizontal=True,
        format_func=lambda x: {
            "return_1d": "1 Day", "return_3d": "3 Days", "return_4d": "4 Days", 
            "return_1w": "1 Week (5D)", "return_1m": "1 Month (21D)"
        }.get(x, x)
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
with tab_vol:
    st.markdown(f"### ⚡ Daily Volume Spikes & Unusual Activity ({date_str})")

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

        st.markdown("#### 🏭 Industries with Largest Daily Turnover Surge")
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
with tab_shp:
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
            st.warning("Stake change delta columns not found in shareholding dataset.")
