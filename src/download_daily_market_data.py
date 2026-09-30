#!/usr/bin/env python3
"""
Download and normalize daily NSE Capital Market (CM) Bhavcopy files for a date range.

Usage:
    python src/download_daily_market_data.py --start-date 2026-09-01 --end-date 2026-09-30
    python src/download_daily_market_data.py --days 30
"""

import argparse
import datetime as dt
import io
import time
import zipfile
from pathlib import Path
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw" / "market_data"

# Standard columns expected by build_stock_day_dataset.py
OUTPUT_COLUMNS = [
    "trade_date", "isin", "symbol", "series",
    "open", "high", "low", "close",
    "last_price", "prev_close",
    "volume", "turnover", "trades",
]

# Column mapping for modern UDiFF format (post-July 2024)
UDIFF_COL_MAP = {
    "TradDt": "trade_date",
    "ISIN": "isin",
    "TckrSymb": "symbol",
    "SctySrs": "series",
    "OpnPric": "open",
    "HghPric": "high",
    "LwPric": "low",
    "ClsPric": "close",
    "LastPric": "last_price",
    "PrvsClsgPric": "prev_close",
    "TtlTradgVol": "volume",
    "TtlValOfTxsExctd": "turnover",
    "TtlTrfVal": "turnover",
    "TtlNbOfTxsExctd": "trades",
}

# Column mapping for legacy format (pre-July 2024)
LEGACY_COL_MAP = {
    "TIMESTAMP": "trade_date",
    "ISIN": "isin",
    "SYMBOL": "symbol",
    "SERIES": "series",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "LAST": "last_price",
    "PREVCLOSE": "prev_close",
    "TOTTRDQTY": "volume",
    "TOTTRDVAL": "turnover",
    "TOTALTRADES": "trades",
}


def get_nse_session() -> requests.Session:
    """Initializes a requests session with realistic browser headers and cookies."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    })
    try:
        # Prime session cookies
        session.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass
    return session


def normalize_bhavcopy(df: pd.DataFrame, target_date: dt.date) -> pd.DataFrame:
    """Standardizes columns and formats across different NSE file generations."""
    df.columns = [str(c).strip() for c in df.columns]

    # Map column names based on available headers
    if "TckrSymb" in df.columns:
        df = df.rename(columns=UDIFF_COL_MAP)
    elif "SYMBOL" in df.columns:
        df = df.rename(columns=LEGACY_COL_MAP)

    # Standardize trade_date
    df["trade_date"] = target_date.strftime("%Y-%m-%d")

    # Clean text columns
    for c in ["isin", "symbol", "series"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Clean numeric columns
    num_cols = ["open", "high", "low", "close", "last_price", "prev_close", "volume", "turnover", "trades"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Keep only target columns that exist
    keep_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    return df[keep_cols]


def download_single_day(session: requests.Session, target_date: dt.date, out_dir: Path) -> bool:
    """Attempts to download and save Bhavcopy for a given date."""
    date_str_iso = target_date.strftime("%Y-%m-%d")
    out_file = out_dir / f"{date_str_iso}_NSE_CM.csv"

    if out_file.exists():
        print(f"[{date_str_iso}] Already exists. Skipping.")
        return True

    # 1. URL Pattern 1: New UDiFF format (July 2024 onwards)
    ymd = target_date.strftime("%Y%m%d")
    udiff_url = f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"

    # 2. URL Pattern 2: Legacy format (Before July 2024)
    ddmmmyyyy = target_date.strftime("%d%b%Y").upper()
    month_str = target_date.strftime("%b").upper()
    year_str = target_date.strftime("%Y")
    legacy_url = (
        f"https://archives.nseindia.com/content/historical/EQUITIES/"
        f"{year_str}/{month_str}/cm{ddmmmyyyy}bhav.csv.zip"
    )

    candidate_urls = [udiff_url, legacy_url]
    zip_content = None

    for url in candidate_urls:
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200 and resp.content[:2] == b"PK":  # Valid zip magic bytes
                zip_content = resp.content
                break
        except Exception:
            continue

    if not zip_content:
        print(f"[{date_str_iso}] No Bhavcopy found (Weekend / Holiday / Not yet published).")
        return False

    # Extract CSV from ZIP in-memory
    try:
        with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
            csv_names = [name for name in z.namelist() if name.endswith(".csv")]
            if not csv_names:
                return False
            with z.open(csv_names[0]) as f:
                df = pd.read_csv(f, dtype=str)

        normalized_df = normalize_bhavcopy(df, target_date)
        normalized_df.to_csv(out_file, index=False)
        print(f"[{date_str_iso}] Downloaded & saved ({len(normalized_df):,} rows)")
        return True

    except Exception as e:
        print(f"[{date_str_iso}] Error processing zip: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download daily NSE Bhavcopy for a date range.")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=30, help="Number of trailing days if start-date is omitted")
    parser.add_argument("--output", default=str(RAW_DIR), help="Output directory for raw CSVs")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    today = dt.date.today()
    if args.end_date:
        end_date = dt.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        end_date = today

    if args.start_date:
        start_date = dt.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    else:
        start_date = end_date - dt.timedelta(days=args.days)

    print(f"Fetching NSE Bhavcopy from {start_date} to {end_date}...")
    session = get_nse_session()

    current_date = start_date
    success_count = 0

    while current_date <= end_date:
        # Skip Saturdays (5) and Sundays (6)
        if current_date.weekday() < 5:
            ok = download_single_day(session, current_date, out_dir)
            if ok:
                success_count += 1
            time.sleep(0.4)  # Rate limiting courtesy delay
        current_date += dt.timedelta(days=1)

    print(f"\nDone! Successfully obtained data for {success_count} trading session(s).")


if __name__ == "__main__":
    main()
