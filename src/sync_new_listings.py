#!/usr/bin/env python3
"""
Sync newly listed NSE Mainboard & SME equities directly from NSE's official
securities master files (EQUITY_L.csv & SME_EQUITY_L.csv) + recent Bhavcopy files
into data/master/stock_master_final.csv.
"""

from io import StringIO
from pathlib import Path
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
MASTER_PATH = BASE_DIR / "data" / "master" / "stock_master_final.csv"
RAW_DIR = BASE_DIR / "data" / "raw" / "market_data"

NSE_MASTER_URLS = [
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv",
]


def get_nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/market-data/securities-available-for-trading",
    })
    try:
        session.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass
    return session


def fetch_official_nse_listings(session: requests.Session) -> pd.DataFrame:
    frames = []
    for url in NSE_MASTER_URLS:
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200 and "SYMBOL" in resp.text:
                df = pd.read_csv(StringIO(resp.text), dtype=str)
                df.columns = [c.strip() for c in df.columns]
                frames.append(df)
                print(f"Fetched {len(df):,} rows from {url.split('/')[-1]}")
        except Exception as e:
            print(f"Warning: Could not fetch {url}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # Standardize column names from NSE's EQUITY_L.csv format
    col_map = {
        "SYMBOL": "symbol",
        "NAME OF COMPANY": "company_name",
        "SERIES": "series",
        "DATE OF LISTING": "listing_date",
        "PAID UP VALUE": "paid_up_value",
        "MARKET LOT": "market_lot",
        "ISIN NUMBER": "isin",
        "FACE VALUE": "face_value",
    }
    combined = combined.rename(columns={k: v for k, v in col_map.items() if k in combined.columns})
    for col in combined.columns:
        combined[col] = combined[col].astype(str).str.strip()

    combined = combined[combined["isin"].ne("") & combined["isin"].ne("nan")]
    return combined.drop_duplicates("isin", keep="last")


def sync_new_stocks():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master file not found at {MASTER_PATH}")

    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")
    master["isin"] = master["isin"].str.strip()
    existing_isins = set(master["isin"])

    session = get_nse_session()
    official_df = fetch_official_nse_listings(session)

    # Also scan latest 5 Bhavcopy files as a secondary catch-all
    bhav_files = sorted(RAW_DIR.glob("*_NSE_CM.csv"))
    bhav_df = pd.DataFrame()
    if bhav_files:
        recent = [pd.read_csv(f, dtype=str).fillna("") for f in bhav_files[-5:]]
        bhav_df = pd.concat(recent, ignore_index=True)
        if "series" in bhav_df.columns:
            bhav_df = bhav_df[bhav_df["series"].isin(["EQ", "BE", "SM", "ST", "BZ"])]
        bhav_df["isin"] = bhav_df["isin"].str.strip()
        bhav_df = bhav_df.drop_duplicates("isin", keep="last")

    # Merge official listings + recent bhavcopy discoveries
    candidates = {}
    if not official_df.empty:
        for _, r in official_df.iterrows():
            isin = r["isin"]
            if isin not in existing_isins:
                candidates[isin] = r.to_dict()

    if not bhav_df.empty:
        for _, r in bhav_df.iterrows():
            isin = r["isin"]
            if isin and isin not in existing_isins and isin not in candidates:
                candidates[isin] = {
                    "isin": isin,
                    "symbol": r.get("symbol", "").strip(),
                    "company_name": r.get("company_name", r.get("symbol", "")).strip(),
                    "series": r.get("series", "EQ").strip(),
                }

    if not candidates:
        print("✓ Master catalog is up to date. No new exchange scrips found.")
        return

    print(f"✨ Discovered {len(candidates)} new exchange scrip(s)! Appending to master catalog...")

    new_rows = []
    for isin, info in candidates.items():
        symbol = info.get("symbol", "")
        row = {col: "" for col in master.columns}
        row.update({
            "isin": isin,
            "symbol": symbol,
            "company_name": info.get("company_name") or symbol,
            "series": info.get("series") or "EQ",
            "listing_date": info.get("listing_date", ""),
            "paid_up_value": info.get("paid_up_value", ""),
            "market_lot": info.get("market_lot", ""),
            "face_value": info.get("face_value", ""),
            "custom_sector": "Unclassified",
            "custom_industry": "Diversified / New Listing",
            "custom_subindustry": "New Listing",
            "needs_review": "True",
            "confidence": "LOW",
            "confidence_score": "0",
            "classification_source": "nse_official_equity_l",
            "mapping_method": "auto_exchange_sync",
            "screener_url": f"https://www.screener.in/company/{symbol}/consolidated/",
        })
        new_rows.append(row)

    updated_master = pd.concat([master, pd.DataFrame(new_rows)], ignore_index=True)
    updated_master = updated_master.drop_duplicates("isin", keep="first")
    updated_master.to_csv(MASTER_PATH, index=False)
    print(f"✓ Added {len(new_rows)} scrip(s) ({', '.join([r['symbol'] for r in new_rows[:10]])}...) to {MASTER_PATH}")


if __name__ == "__main__":
    sync_new_stocks()
