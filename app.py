import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="NSE Sector & Breakout Tracker", layout="wide")

MASTER_PATH = Path("data/master/stock_master_final.csv")
STOCK_PARQUET = Path("data/processed/stock_day.parquet")
SHP_PARQUET = Path("data/processed/shareholding.parquet")


def get_file_mtime(path: Path) -> float:
    return os.path.getmtime(path) if path.exists() else 0.0


@st.cache_data
def load_data(stock_mtime: float, shp_mtime: float):
    day_df = pd.read_parquet(STOCK_PARQUET)
    shp_df = pd.read_parquet(SHP_PARQUET) if SHP_PARQUET.exists() else pd.DataFrame()
    return day_df, shp_df


try:
    day_df, shp_df = load_data(get_file_mtime(STOCK_PARQUET), get_file_mtime(SHP_PARQUET))

    # 1. Recover index if 'isin' was stored as index
    if "isin" not in day_df.columns and ("isin" in str(day_df.index.name).lower() or day_df.index.name == "isin"):
        day_df = day_df.reset_index()
    if not shp_df.empty and "isin" not in shp_df.columns and ("isin" in str(shp_df.index.name).lower() or shp_df.index.name == "isin"):
        shp_df = shp_df.reset_index()

    # 2. Normalize 'symbol' and 'isin'
    if "symbol" not in day_df.columns:
        for alt in ["symbol_x", "symbol_master", "symbol_y", "TckrSymb"]:
            if alt in day_df.columns:
                day_df["symbol"] = day_df[alt]
                break

    if "isin" not in day_df.columns:
        for alt in ["isin_x", "isin_master", "isin_y", "ISIN"]:
            if alt in day_df.columns:
                day_df["isin"] = day_df[alt]
                break

    # 3. Convert category columns to string to prevent Cartesian groupby expansion bugs
    for cat_col in ["custom_sector", "custom_industry", "custom_subindustry"]:
        if cat_col in day_df.columns:
            day_df[cat_col] = day_df[cat_col].astype(str)

    # 4. Standardize dates & upcast numeric columns to float64 for safe math
    day_df["trade_date"] = pd.to_datetime(day_df["trade_date"], errors="coerce")
    if not shp_df.empty and "quarter_dt" in shp_df.columns:
        shp_df["quarter_dt"] = pd.to_datetime(shp_df["quarter_dt"], errors="coerce")

    for col in ["open", "high", "low", "close", "prev_close", "volume", "turnover", "turnover_cr"]:
        if col in day_df.columns:
            day_df[col] = pd.to_numeric(day_df[col], errors="coerce").astype("float64")

    # 5. Turnover in ₹ Crores
    if "turnover_cr" not in day_df.columns:
        if "turnover" in day_df.columns:
            day_df["turnover_cr"] = day_df["turnover"] / 1e7
        elif "close" in day_df.columns and "volume" in day_df.columns:
            day_df["turnover_cr"] = (day_df["close"] * day_df["volume"]) / 1e7
        else:
            day_df["turnover_cr"] = 0.0

    # 6. Sort & Compute Multi-Horizon Returns + Breakout Technicals
    day_df = day_df.sort_values(["isin", "trade_date"]).reset_index(drop=True)
    grp = day_df.groupby("isin", sort=False)

    if "return_1d" not in day_df.columns:
        day_df["return_1d"] = grp["close"].pct_change(1) * 100
    if "return_3d" not in day_df.columns:
        day_df["return_3d"] = grp["close"].pct_change(3) * 100
    if "return_4d" not in day_df.columns:
        day_df["return_4d"] = grp["close"].pct_change(4) * 100
    if "return_1w" not in day_df.columns:
        day_df["return_1w"] = grp["close"].pct_change(5) * 100
    if "return_1m" not in day_df.columns:
        day_df["return_1m"] = grp["close"].pct_change(21) * 100

    for ret_col in ["return_1d", "return_3d", "return_4d", "return_1w", "return_1m"]:
        day_df[ret_col] = day_df[ret_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 7. Volume Baselines (20D & 42D / 2-Month) + 3-Day Sustained Accumulation
    day_df["vol_sma_20"] = grp["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=3).mean())
    day_df["vol_sma20"] = day_df["vol_sma_20"]
    day_df["vol_sma_42"] = grp["volume"].transform(lambda s: s.shift(1).rolling(42, min_periods=5).mean())
    day_df["vol_3d_avg"] = grp["volume"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    day_df["rvol"] = (day_df["volume"] / day_df["vol_sma_20"].replace(0, np.nan)).fillna(1.0)
    day_df["sustained_vol_ratio"] = (day_df["vol_3d_avg"] / day_df["vol_sma_42"].replace(0, np.nan)).fillna(1.0)

    # 8. Early Breakout Indicators (Prior 20D & 50D Resistance Highs + Close Range Strength)
    if "high" in day_df.columns and "low" in day_df.columns:
        day_df["prior_high_20d"] = grp["high"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).max())
        day_df["prior_high_50d"] = grp["high"].transform(lambda s: s.shift(1).rolling(50, min_periods=10).max())
        day_df["is_20d_bo"] = day_df["close"] > day_df["prior_high_20d"]
        day_df["is_50d_bo"] = day_df["close"] > day_df["prior_high_50d"]

        candle_range = (day_df["high"] - day_df["low"]).replace(0, np.nan)
        day_df["close_strength"] = ((day_df["close"] - day_df["low"]) / candle_range).fillna(0.5)
    else:
        day_df["prior_high_20d"] = grp["close"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).max())
        day_df["prior_high_50d"] = grp["close"].transform(lambda s: s.shift(1).rolling(50, min_periods=10).max())
        day_df["is_20d_bo"] = day_df["close"] > day_df["prior_high_20d"]
        day_df["is_50d_bo"] = day_df["close"] > day_df["prior_high_50d"]
        day_df["close_strength"] = 1.0

except Exception as e:
    st.error(f"Error loading or initializing Parquet files: {e}")
    st.stop()

st.title("📊 NSE Sector & Day-1 Breakout Scanner")

# Sidebar Filters
sectors = sorted([s for s in day_df["custom_sector"].dropna().unique() if s not in ("", "nan", "None")]) if "custom_sector" in day_df.columns else []
selected_sector = st.sidebar.selectbox("Filter Sector", ["All"] + sectors)

sub_df = day_df if selected_sector == "All" else day_df[day_df["custom_sector"] == selected_sector]
industries = sorted([i for i in sub_df["custom_industry"].dropna().unique() if i not in ("", "nan", "None")]) if "custom_industry" in sub_df.columns else []
selected_industry = st.sidebar.selectbox("Filter Industry", ["All"] + industries)

target_df = sub_df if selected_industry == "All" else sub_df[sub_df["custom_industry"] == selected_industry]

latest_date = target_df["trade_date"].max()
latest_stocks = target_df[target_df["trade_date"] == latest_date].copy()
date_str = latest_date.strftime("%d-%b-%Y") if pd.notna(latest_date) else "Latest"

# Navigation Tabs
tab_bo, tab_surge, tab_backtest, tab_returns, tab_vol, tab_shp = st.tabs([
    "🎯 Day-1 Breakouts & Money Flow",
    "🔥 Momentum & Surge Sectors (3-4D)",
    "🧪 Backtest Engine",
    "🚀 Price Returns",
    "⚡ Daily Volume Spikes",
    "🏛️ Institutional Stakes (FII/DII)"
])

# ----------------- TAB 0: DAY-1 BREAKOUTS & EARLY MONEY FLOW -----------------
with tab_bo:
    st.markdown(f"### 🎯 Day-1 Institutional Breakouts & Early Sector Thrust ({date_str})")
    st.caption(
        "Catches money-moving breakouts on Day 1: stocks clearing 20D/50D resistance highs, closing near the top of their range, "
        "and backed by single-day RVOL or 3-day sustained volume accumulation—independent of lagging sector averages."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        bo_horizon = st.radio("Resistance Breakout", ["20D High", "50D High"], horizontal=True)
    with c2:
        bo_min_turnover = st.number_input(
            "Min Turnover (₹ Cr)", min_value=0.5, value=5.0, step=1.0, key="bo_turn",
            help="Filters out illiquid penny stocks (₹5 Cr+ recommended)"
        )
    with c3:
        bo_min_vol = st.slider(
            "Min Volume Expansion", min_value=1.0, max_value=4.0, value=1.4, step=0.1, key="bo_vol",
            help="Triggers if either Daily RVOL OR 3D Sustained Volume vs 2M Avg exceeds this threshold"
        )
    with c4:
        bo_min_strength = st.slider(
            "Min Close Strength", min_value=0.50, max_value=0.95, value=0.70, step=0.05,
            help="0.70 = Price closed in the top 30% of today's High-Low range"
        )

    bo_flag_col = "is_20d_bo" if bo_horizon == "20D High" else "is_50d_bo"
    ref_high_col = "prior_high_20d" if bo_horizon == "20D High" else "prior_high_50d"

    bo_candidates = latest_stocks[
        (latest_stocks[bo_flag_col].fillna(False)) &
        (latest_stocks["turnover_cr"] >= bo_min_turnover) &
        (latest_stocks["close_strength"] >= bo_min_strength) &
        ((latest_stocks["rvol"] >= bo_min_vol) | (latest_stocks["sustained_vol_ratio"] >= bo_min_vol))
    ].copy()

    if not bo_candidates.empty:
        bo_candidates["pct_above_res"] = ((bo_candidates["close"] - bo_candidates[ref_high_col]) / bo_candidates[ref_high_col]) * 100
        bo_candidates["flow_score"] = (
            (bo_candidates[["rvol", "sustained_vol_ratio"]].max(axis=1) * 40) +
            (bo_candidates["close_strength"] * 30) +
            (np.log1p(bo_candidates["turnover_cr"]) * 10)
        ).round(1)
        bo_candidates = bo_candidates.sort_values("flow_score", ascending=False)

    m_bo1, m_bo2, m_bo3 = st.columns(3)
    m_bo1.metric("Day-1 Breakout Stocks", f"{len(bo_candidates)}")
    m_bo2.metric("Total Breakout Turnover", f"₹{bo_candidates['turnover_cr'].sum():,.1f} Cr" if not bo_candidates.empty else "₹0.0 Cr")
    top_sec_name = bo_candidates["custom_industry"].value_counts().index[0] if not bo_candidates.empty and "custom_industry" in bo_candidates.columns else "—"
    m_bo3.metric("Most Active Industry Today", top_sec_name)

    st.markdown("#### 🚀 Day-1 Breakout Leaderboard (Includes New Listings & Standalone Leaders)")
    if not bo_candidates.empty:
        bo_cols = [
            "symbol", "company_name", "custom_industry", "close", "return_1d",
            "pct_above_res", "rvol", "sustained_vol_ratio", "close_strength", "turnover_cr", "flow_score"
        ]
        avail_bo_cols = [c for c in bo_cols if c in bo_candidates.columns]

        st.dataframe(
            bo_candidates[avail_bo_cols]
            .rename(columns={
                "symbol": "Symbol",
                "company_name": "Company",
                "custom_industry": "Industry",
                "close": "Close",
                "return_1d": "1D Return",
                "pct_above_res": f"% > {bo_horizon}",
                "rvol": "1D RVOL",
                "sustained_vol_ratio": "3D Accum vs 2M",
                "close_strength": "Range Close %",
                "turnover_cr": "Turnover (₹ Cr)",
                "flow_score": "Flow Score",
            })
            .style.format({
                "Close": "₹{:.2f}",
                "1D Return": "{:+.2f}%",
                f"% > {bo_horizon}": "{:+.2f}%",
                "1D RVOL": "{:.2f}x",
                "3D Accum vs 2M": "{:.2f}x",
                "Range Close %": "{:.0%}",
                "Turnover (₹ Cr)": "₹{:.2f} Cr",
                "Flow Score": "{:.1f}",
            }),
            use_container_width=True
        )
    else:
        st.info("No stocks matched the strict Day-1 breakout criteria today. Try lowering Min Turnover or Min Volume Expansion.")

    st.markdown("#### 🌊 Early Sector Breadth Thrust (Sectors Breaking Out Together Today)")
    st.caption("Detects sectors where multiple liquid constituents simultaneously break out or surge in volume today—before 3-4 day averages turn.")

    all_today = day_df[(day_df["trade_date"] == latest_date) & (day_df["turnover_cr"] >= 2.0)].copy()
    if not all_today.empty and "custom_industry" in all_today.columns:
        all_today["is_active_thrust"] = (
            (all_today["return_1d"] > 1.0) &
            ((all_today["rvol"] >= 1.3) | (all_today["sustained_vol_ratio"] >= 1.3) | (all_today["is_20d_bo"].fillna(False)))
        )
        breadth_df = (
            all_today.groupby("custom_industry", observed=True)
            .agg(
                liquid_stocks=("isin", "nunique"),
                thrust_stocks=("is_active_thrust", "sum"),
                breakout_20d=("is_20d_bo", "sum"),
                avg_1d_ret=("return_1d", "mean"),
                total_turnover=("turnover_cr", "sum"),
            )
            .reset_index()
        )
        breadth_df = breadth_df[breadth_df["liquid_stocks"] >= 3].copy()
        breadth_df["breadth_pct"] = (breadth_df["thrust_stocks"] / breadth_df["liquid_stocks"]) * 100
        breadth_df = breadth_df[breadth_df["thrust_stocks"] >= 2].sort_values(
            ["breadth_pct", "breakout_20d", "total_turnover"], ascending=False
        )

        if not breadth_df.empty:
            st.dataframe(
                breadth_df.head(15)
                .rename(columns={
                    "custom_industry": "Industry",
                    "liquid_stocks": "Liquid Stocks",
                    "thrust_stocks": "Stocks in Thrust",
                    "breadth_pct": "Breadth Thrust %",
                    "breakout_20d": "20D High Breakouts",
                    "avg_1d_ret": "Avg 1D Return",
                    "total_turnover": "Industry Turnover (₹ Cr)",
                })
                .style.format({
                    "Breadth Thrust %": "{:.1f}%",
                    "Avg 1D Return": "{:+.2f}%",
                    "Industry Turnover (₹ Cr)": "₹{:.1f} Cr",
                }),
                use_container_width=True
            )
        else:
            st.info("No broad multi-stock industry thrust detected today.")

# ----------------- TAB 1: 3-4 DAY POSITIVE SECTORS + 2-MONTH VOLUME SURGE -----------------
with tab_surge:
    st.markdown(f"### 🔥 Positive Sectors with 2-Month Volume Surge ({date_str})")
    st.caption("Filters sectors/industries with positive 3-day or 4-day cumulative returns and surging volume relative to their 2-month (~42 trading days) average.")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        lookback_days = st.radio(
            "Lookback Period for Positive Momentum", [3, 4], horizontal=True,
            format_func=lambda x: f"Last {x} Trading Days"
        )
    with col_s2:
        min_surge_mult = st.slider(
            "Minimum 2-Month Volume Surge", min_value=0.8, max_value=4.0, value=1.2, step=0.1,
            help="1.2x means trading at 120% of its trailing 2-month average turnover"
        )
    with col_s3:
        group_level = st.radio(
            "Classification Level", ["custom_sector", "custom_industry"], horizontal=True,
            format_func=lambda x: "Coarse (Sector)" if x == "custom_sector" else "Granular (Industry)"
        )

    ret_col_target = f"return_{lookback_days}d"

    if group_level in day_df.columns:
        sec_agg = (
            day_df.groupby(["trade_date", group_level], observed=True)
            .agg(
                turnover_cr=("turnover_cr", "sum"),
                avg_return=(ret_col_target, "mean"),
                ret_1d=("return_1d", "mean"),
                stock_count=("isin", "nunique")
            )
            .reset_index()
            .sort_values([group_level, "trade_date"])
        )

        sec_agg["turnover_2m_sma"] = (
            sec_agg.groupby(group_level, observed=True)["turnover_cr"]
            .transform(lambda s: s.shift(1).rolling(42, min_periods=5).mean())
        )
        sec_agg["surge_2m_mult"] = (sec_agg["turnover_cr"] / sec_agg["turnover_2m_sma"].replace(0, np.nan)).round(2)

        latest_sec = sec_agg[sec_agg["trade_date"] == latest_date].copy()
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

            st.markdown("#### 🎯 Leading Stocks in Surging Positive Sectors")
            surging_names = qualified_sectors[group_level].tolist()
            drill_stocks = latest_stocks[latest_stocks[group_level].isin(surging_names)].copy()
            if not drill_stocks.empty:
                drill_stocks["rank_in_sector"] = drill_stocks.groupby(group_level, observed=True)[ret_col_target].rank(ascending=False, method="dense")
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
            st.info(f"No sectors met both criteria: Positive {lookback_days}-day return (> 0%) and 2-Month volume surge ≥ {min_surge_mult}x.")

# ----------------- TAB 2: BACKTEST ENGINE -----------------
with tab_backtest:
    st.markdown("### 🧪 Strategy Backtest Engine")

    bt_strategy = st.radio(
        "Select Strategy to Backtest",
        ["Day-1 Stock Breakouts (Money Flow)", "Positive Sectors + 2M Volume Surge (3-4D)"],
        horizontal=True,
        key="bt_strategy_selector"
    )

    all_dates = sorted(day_df["trade_date"].dropna().unique())
    if not all_dates:
        st.error("No valid trading dates found in dataset.")
        st.stop()

    min_available_date = pd.Timestamp(all_dates[0]).date()
    max_available_date = pd.Timestamp(all_dates[-1]).date()

    # ================= STRATEGY A: DAY-1 STOCK BREAKOUTS =================
    if bt_strategy == "Day-1 Stock Breakouts (Money Flow)":
        st.caption("Simulates buying stocks on the close of Day-1 resistance breakouts backed by volume accumulation and strong range closes.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            bt_from = st.date_input("From Date", value=min_available_date, min_value=min_available_date, max_value=max_available_date, key="bo_bt_from")
        with c2:
            bt_till = st.date_input("Till Date", value=max_available_date, min_value=min_available_date, max_value=max_available_date, key="bo_bt_till")
        with c3:
            bt_bo_horizon = st.radio("Breakout Level", ["20D High", "50D High"], horizontal=True, key="bo_bt_hor")
        with c4:
            bt_hold_days = st.selectbox("Holding Period", [5, 10, 20], index=1, format_func=lambda x: f"{x} Trading Days", key="bo_bt_hold")

        c5, c6, c7 = st.columns(3)
        with c5:
            bt_bo_turn = st.number_input("Min Turnover (₹ Cr)", min_value=0.5, value=5.0, step=1.0, key="bo_bt_turn")
        with c6:
            bt_bo_vol = st.slider("Min Volume Expansion", min_value=1.0, max_value=4.0, value=1.4, step=0.1, key="bo_bt_vol")
        with c7:
            bt_bo_str = st.slider("Min Close Strength", min_value=0.50, max_value=0.95, value=0.70, step=0.05, key="bo_bt_str")

        if bt_from > bt_till:
            st.error("'From Date' must be earlier than 'Till Date'.")
        else:
            bt_df = target_df.sort_values(["isin", "trade_date"]).copy()
            grp_bt = bt_df.groupby("isin", sort=False)

            safe_close = bt_df["close"].replace(0, np.nan)
            bt_df["fwd_5d"] = (grp_bt["close"].shift(-5) / safe_close - 1.0) * 100.0
            bt_df["fwd_10d"] = (grp_bt["close"].shift(-10) / safe_close - 1.0) * 100.0
            bt_df["fwd_20d"] = (grp_bt["close"].shift(-20) / safe_close - 1.0) * 100.0

            high_col = "high" if "high" in bt_df.columns else "close"
            bt_df["max_20d_high"] = grp_bt[high_col].transform(lambda s: s.shift(-20).rolling(20, min_periods=1).max())
            bt_df["peak_gain_20d"] = ((bt_df["max_20d_high"] / safe_close) - 1.0) * 100.0

            bo_col = "is_20d_bo" if bt_bo_horizon == "20D High" else "is_50d_bo"
            trade_dates_only = bt_df["trade_date"].dt.date
            date_mask = (trade_dates_only >= bt_from) & (trade_dates_only <= bt_till)

            triggers = bt_df[
                date_mask &
                (bt_df[bo_col].fillna(False)) &
                (bt_df["turnover_cr"] >= bt_bo_turn) &
                (bt_df["close_strength"] >= bt_bo_str) &
                ((bt_df["rvol"] >= bt_bo_vol) | (bt_df["sustained_vol_ratio"] >= bt_bo_vol))
            ].copy().sort_values("trade_date", ascending=False)

            target_ret_col = f"fwd_{bt_hold_days}d"

            if triggers.empty:
                st.info("No historical breakout triggers matched these parameters in the selected window.")
            else:
                matured = triggers.dropna(subset=[target_ret_col]).copy()
                total_trades = len(triggers)
                matured_count = len(matured)

                if matured_count > 0:
                    wins = matured[matured[target_ret_col] > 0]
                    losses = matured[matured[target_ret_col] <= 0]
                    win_rate = (len(wins) / matured_count) * 100.0
                    avg_ret = float(matured[target_ret_col].mean())
                    avg_peak = float(matured["peak_gain_20d"].dropna().mean()) if matured["peak_gain_20d"].notna().any() else 0.0
                    gross_win = float(wins[target_ret_col].sum())
                    gross_loss = float(abs(losses[target_ret_col].sum()))
                    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else np.inf
                else:
                    win_rate, avg_ret, avg_peak, profit_factor = 0.0, 0.0, 0.0, 0.0

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Breakout Signals", f"{total_trades}")
                m2.metric(f"Win Rate (+{bt_hold_days}D)", f"{win_rate:.1f}%", f"{matured_count} closed trades")
                m3.metric(f"Avg Return (+{bt_hold_days}D)", f"{avg_ret:+.2f}%")
                m4.metric("Avg 20D Peak Gain", f"{avg_peak:+.2f}%")
                m5.metric("Profit Factor", f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "∞")

                if matured_count > 0:
                    daily_curve = (
                        matured.groupby("trade_date", as_index=False)[target_ret_col]
                        .mean()
                        .sort_values("trade_date")
                    )
                    daily_curve["Cumulative Return (%)"] = ((1.0 + daily_curve[target_ret_col] / 100.0).cumprod() - 1.0) * 100.0
                    st.markdown(f"#### 📈 Cumulative Strategy Curve (+{bt_hold_days}D Holding Cohorts)")
                    st.line_chart(daily_curve.set_index("trade_date")["Cumulative Return (%)"])

                st.markdown("#### 📋 Historical Breakout Trade Log")
                triggers["Signal Date"] = triggers["trade_date"].dt.strftime("%Y-%m-%d")
                log_cols = [
                    "Signal Date", "symbol", "custom_industry", "close", "turnover_cr",
                    "rvol", "sustained_vol_ratio", "fwd_5d", "fwd_10d", "fwd_20d", "peak_gain_20d"
                ]
                avail_log_cols = [c for c in log_cols if c in triggers.columns]

                st.dataframe(
                    triggers[avail_log_cols]
                    .rename(columns={
                        "symbol": "Symbol",
                        "custom_industry": "Industry",
                        "close": "Entry Price",
                        "turnover_cr": "Turnover (₹ Cr)",
                        "rvol": "1D RVOL",
                        "sustained_vol_ratio": "3D Accum",
                        "fwd_5d": "+5D Ret",
                        "fwd_10d": "+10D Ret",
                        "fwd_20d": "+20D Ret",
                        "peak_gain_20d": "Max 20D Peak",
                    })
                    .style.format({
                        "Entry Price": "₹{:.2f}",
                        "Turnover (₹ Cr)": "₹{:.1f} Cr",
                        "1D RVOL": "{:.2f}x",
                        "3D Accum": "{:.2f}x",
                        "+5D Ret": "{:+.2f}%",
                        "+10D Ret": "{:+.2f}%",
                        "+20D Ret": "{:+.2f}%",
                        "Max 20D Peak": "{:+.2f}%",
                    }, na_rep="Open"),
                    use_container_width=True
                )

    # ================= STRATEGY B: SECTOR MOMENTUM (3-4D) =================
    else:
        st.caption("Evaluate how sectors performed after meeting positive 3-4 day return and 2-month volume surge conditions.")

        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            bt_from = st.date_input("From Date", value=min_available_date, min_value=min_available_date, max_value=max_available_date, key="sec_bt_from")
        with col_b2:
            bt_till = st.date_input("Till Date", value=max_available_date, min_value=min_available_date, max_value=max_available_date, key="sec_bt_till")
        with col_b3:
            bt_lookback = st.radio("Lookback Days", [3, 4], horizontal=True, key="bt_lookback", format_func=lambda x: f"{x} Days")
        with col_b4:
            bt_surge_min = st.slider("Min 2M Surge Ratio", min_value=1.0, max_value=3.0, value=1.2, step=0.1, key="bt_surge")

        col_b5, col_b6 = st.columns(2)
        with col_b5:
            bt_group = st.radio(
                "Classification Level", ["custom_sector", "custom_industry"], horizontal=True, key="bt_group",
                format_func=lambda x: "Sector" if x == "custom_sector" else "Industry"
            )
        with col_b6:
            bt_fwd_horizon = st.selectbox(
                "Benchmark Forward Horizon", ["fwd_return_5d", "fwd_return_10d", "fwd_return_20d"],
                key="sec_bt_fwd",
                format_func=lambda x: {"fwd_return_5d": "+5 Days", "fwd_return_10d": "+10 Days", "fwd_return_20d": "+20 Days (1 Month)"}[x]
            )

        if bt_from > bt_till:
            st.error("Error: 'From Date' must be earlier than 'Till Date'.")
        elif bt_group not in day_df.columns:
            st.warning(f"Column '{bt_group}' not found in dataset.")
        else:
            ret_target_col = f"return_{bt_lookback}d"
            sec_hist = (
                day_df.groupby(["trade_date", bt_group], observed=True)
                .agg(
                    turnover_cr=("turnover_cr", "sum"),
                    sector_return_1d=("return_1d", "mean"),
                    lookback_ret=(ret_target_col, "mean")
                )
                .reset_index()
                .sort_values([bt_group, "trade_date"])
            )

            sec_hist["turnover_2m_sma"] = (
                sec_hist.groupby(bt_group, observed=True)["turnover_cr"]
                .transform(lambda s: s.shift(1).rolling(42, min_periods=5).mean())
            )
            sec_hist["surge_2m_mult"] = (sec_hist["turnover_cr"] / sec_hist["turnover_2m_sma"].replace(0, np.nan)).round(2)

            # Vectorized log-return compounding
            sec_hist["log_ret"] = np.log1p((sec_hist["sector_return_1d"].fillna(0.0) / 100.0).clip(lower=-0.99))
            grp_sec = sec_hist.groupby(bt_group, observed=True)["log_ret"]

            sec_hist["fwd_return_5d"] = np.expm1(grp_sec.transform(lambda s: s.shift(-5).rolling(5, min_periods=5).sum())) * 100.0
            sec_hist["fwd_return_10d"] = np.expm1(grp_sec.transform(lambda s: s.shift(-10).rolling(10, min_periods=10).sum())) * 100.0
            sec_hist["fwd_return_20d"] = np.expm1(grp_sec.transform(lambda s: s.shift(-20).rolling(20, min_periods=20).sum())) * 100.0

            trade_dates_sec = sec_hist["trade_date"].dt.date
            in_range_mask = (trade_dates_sec >= bt_from) & (trade_dates_sec <= bt_till)
            trigger_mask = in_range_mask & (sec_hist["lookback_ret"] > 0) & (sec_hist["surge_2m_mult"] >= bt_surge_min)
            signals_df = sec_hist[trigger_mask].copy().sort_values("trade_date", ascending=False)

            if signals_df.empty:
                st.info("No sectors met both positive return and 2-month volume surge conditions during this date window.")
            else:
                valid_fwd = signals_df.dropna(subset=[bt_fwd_horizon])
                total_signals = len(signals_df)
                matured_signals = len(valid_fwd)

                if matured_signals > 0:
                    win_count = int((valid_fwd[bt_fwd_horizon] > 0).sum())
                    win_rate = (win_count / matured_signals) * 100.0
                    avg_fwd_ret = float(valid_fwd[bt_fwd_horizon].mean())
                    best_ret = float(valid_fwd[bt_fwd_horizon].max())
                else:
                    win_count, win_rate, avg_fwd_ret, best_ret = 0, 0.0, 0.0, 0.0

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Signals Triggered", f"{total_signals}")
                m2.metric(f"Win Rate ({bt_fwd_horizon.replace('fwd_return_', '+').upper()})", f"{win_rate:.1f}%", f"{win_count}/{matured_signals} wins" if matured_signals > 0 else "N/A")
                m3.metric("Avg Forward Return", f"{avg_fwd_ret:+.2f}%")
                m4.metric("Best Forward Return", f"{best_ret:+.2f}%")

                signals_df["trigger_date"] = signals_df["trade_date"].dt.strftime("%Y-%m-%d")
                display_cols = [
                    "trigger_date", bt_group, "lookback_ret", "turnover_cr",
                    "surge_2m_mult", "fwd_return_5d", "fwd_return_10d", "fwd_return_20d"
                ]

                st.dataframe(
                    signals_df[display_cols]
                    .rename(columns={
                        "trigger_date": "Signal Date",
                        bt_group: "Sector / Industry",
                        "lookback_ret": f"{bt_lookback}D Trailing Ret",
                        "turnover_cr": "Turnover (₹ Cr)",
                        "surge_2m_mult": "2M Surge Multiple",
                        "fwd_return_5d": "Next +5D Ret",
                        "fwd_return_10d": "Next +10D Ret",
                        "fwd_return_20d": "Next +20D Ret"
                    })
                    .style.format({
                        f"{bt_lookback}D Trailing Ret": "{:+.2f}%",
                        "Turnover (₹ Cr)": "₹{:.1f} Cr",
                        "2M Surge Multiple": "{:.2f}x",
                        "Next +5D Ret": "{:+.2f}%",
                        "Next +10D Ret": "{:+.2f}%",
                        "Next +20D Ret": "{:+.2f}%"
                    }, na_rep="Ongoing"),
                    use_container_width=True
                )

# ----------------- TAB 3: RETURNS -----------------
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
        latest_stocks["rank"] = latest_stocks.groupby("custom_industry", observed=True)[timeframe].rank(ascending=False, method="dense")
        top_3 = latest_stocks[latest_stocks["rank"] <= 3].sort_values(["custom_industry", "rank"]).copy()

        desired_cols = ["custom_industry", "rank", "symbol", "company_name", "close", timeframe]
        avail_cols = [c for c in desired_cols if c in top_3.columns]

        fmt_dict = {}
        if "close" in avail_cols:
            fmt_dict["close"] = "₹{:.2f}"
        if timeframe in avail_cols:
            fmt_dict[timeframe] = "{:+.2f}%"

        st.dataframe(top_3[avail_cols].style.format(fmt_dict), use_container_width=True)

# ----------------- TAB 4: VOLUME SPIKES -----------------
with tab_vol:
    st.markdown(f"### ⚡ Daily Volume Spikes & Unusual Activity ({date_str})")

    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        min_rvol = st.slider("Minimum Relative Volume (RVOL)", min_value=1.5, max_value=10.0, value=2.0, step=0.5)
        min_turnover = st.number_input("Min Turnover (₹ Cr)", min_value=0.0, value=1.0, step=0.5)

    spike_stocks = latest_stocks[(latest_stocks["rvol"] >= min_rvol) & (latest_stocks["turnover_cr"] >= min_turnover)].copy()

    if not spike_stocks.empty:
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

# ----------------- TAB 5: SHAREHOLDING -----------------
with tab_shp:
    st.markdown("### Institutional Stake Changes (QoQ)")

    if shp_df.empty or "quarter_dt" not in shp_df.columns:
        st.info("Quarterly shareholding dataset is currently empty or updating.")
    else:
        latest_quarter = shp_df["quarter_dt"].max()
        quarter_str = latest_quarter.strftime("%b %Y") if pd.notna(latest_quarter) else "Latest"
        st.caption(f"Showing filings for quarter: **{quarter_str}**")

        quarter_shp = shp_df[shp_df["quarter_dt"] == latest_quarter].copy()

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
            quarter_shp["stake_rank"] = quarter_shp.groupby("custom_industry", observed=True)[stake_metric].rank(ascending=False, method="dense")
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
