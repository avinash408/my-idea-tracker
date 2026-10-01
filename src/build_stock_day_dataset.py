#!/usr/bin/env python3
"""
Join normalized NSE daily market data to stock_master_final.csv
and calculate trailing returns and volume metrics.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
MASTER = BASE_DIR / "data" / "master" / "stock_master_final.csv"
RAW_DIR = BASE_DIR / "data" / "raw" / "market_data"
OUT_PARQUET = BASE_DIR / "data" / "processed" / "stock_day.parquet"

CLASSIFICATION_COLS = [
    "isin", "symbol", "company_name",
    "macro_sector", "sector", "industry", "basic_industry",
    "basic_industry_code",
    "custom_sector", "custom_industry", "custom_subindustry",
    "theme_1", "theme_2", "theme_3", "theme_4", "theme_5",
    "screener_url"
]

def main():
    master = pd.read_csv(MASTER, dtype=str).fillna("")
    master = master.drop_duplicates("isin", keep="last")

    files = sorted(RAW_DIR.glob("*_NSE_CM.csv"))
    if not files:
        raise SystemExit(f"No daily market files found in {RAW_DIR}.")

    market = pd.concat([pd.read_csv(f, dtype=str).fillna("") for f in files], ignore_index=True)
    market = market[market["isin"].ne("")].copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce")

    # Clean numeric columns
    for col in ["open", "high", "low", "close", "prev_close", "volume", "turnover"]:
        if col in market.columns:
            market[col] = pd.to_numeric(market[col], errors="coerce")

    market = market.drop_duplicates(["isin", "trade_date"], keep="last")
    market = market.sort_values(["isin", "trade_date"])

    # 1. Trailing Returns (% change)
    market["return_1d"] = market.groupby("isin")["close"].pct_change(1) * 100
    market["return_3d"] = market.groupby("isin")["close"].pct_change(3) * 100
    market["return_4d"] = market.groupby("isin")["close"].pct_change(4) * 100
    market["return_1w"] = market.groupby("isin")["close"].pct_change(5) * 100
    market["return_1m"] = market.groupby("isin")["close"].pct_change(21) * 100

    # 2. Volume & Turnover Baselines
    market["vol_sma_20"] = (
        market.groupby("isin")["volume"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=1).mean())
    )
    market["vol_sma20"] = market["vol_sma_20"]
    market["rvol"] = (market["volume"] / market["vol_sma_20"].replace(0, np.nan)).fillna(1.0)
    market["turnover_cr"] = (market["turnover"] if "turnover" in market.columns else market["close"] * market["volume"]) / 1e7

    # 3. Merge Taxonomy
    class_cols = [c for c in CLASSIFICATION_COLS if c in master.columns]
    joined = market.merge(
        master[class_cols],
        on="isin",
        how="inner",
        suffixes=("", "_master")
    )

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(OUT_PARQUET, index=False, compression="snappy")
    print(f"Successfully updated {OUT_PARQUET} ({len(joined):,} rows).")

if __name__ == "__main__":
    main()
