#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
MASTER = BASE_DIR / "data" / "master" / "stock_master_final.csv"
RAW_DIR = BASE_DIR / "data" / "raw" / "market_data"
OUT_PARQUET = BASE_DIR / "data" / "processed" / "stock_day.parquet"

def build_daily_dataset():
    master = pd.read_csv(MASTER, dtype=str).fillna("")
    files = sorted(RAW_DIR.glob("*_NSE_CM.csv"))
    if not files:
        raise SystemExit("No daily bhavcopy files found.")

    market = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    market = market[market["isin"].ne("")].copy()
    
    # Numeric conversions
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce")
    for col in ["open", "high", "low", "close", "prev_close", "volume", "turnover"]:
        if col in market.columns:
            market[col] = pd.to_numeric(market[col], errors="coerce")

    # Drop intraday duplicates & sort chronologically
    market = market.drop_duplicates(["isin", "trade_date"], keep="last")
    market = market.sort_values(["isin", "trade_date"])

    # Rolling price returns (%)
    market["return_1d"] = market.groupby("isin")["close"].pct_change(1) * 100
    market["return_1w"] = market.groupby("isin")["close"].pct_change(5) * 100
    market["return_1m"] = market.groupby("isin")["close"].pct_change(21) * 100

    # Volume Spike / Relative Volume (RVOL)
    # 20-day rolling average volume per stock (min 5 days required for early listings)
    market["vol_sma_20"] = (
        market.groupby("isin")["volume"]
        .rolling(window=20, min_periods=5)
        .mean()
        .reset_index(drop=True)
    )
    # Calculate RVOL ratio
    market["rvol"] = np.where(
        market["vol_sma_20"] > 0,
        (market["volume"] / market["vol_sma_20"]).round(2),
        1.0
    )

    # Convert turnover to Crores (NSE raw turnover is in ₹)
    if "turnover" in market.columns:
        market["turnover_cr"] = (market["turnover"] / 1e7).round(2)

    # Join classification taxonomy
    taxonomy_cols = [
        "isin", "symbol", "company_name", "custom_sector", 
        "custom_industry", "custom_subindustry", "screener_url"
    ]
    joined = market.merge(master[taxonomy_cols], on="isin", how="inner")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(OUT_PARQUET, index=False, compression="snappy")
    print(f"Saved {len(joined):,} rows with RVOL metrics to {OUT_PARQUET}")

if __name__ == "__main__":
    build_daily_dataset()
