#!/usr/bin/env python3
import time
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parents[1]
MASTER = BASE_DIR / "data" / "master" / "stock_master_final.csv"
OUT_SHP = BASE_DIR / "data" / "processed" / "shareholding.parquet"

HEADERS = {"User-Agent": "Mozilla/5.0"}

def scrape_screener_shareholding(url: str, isin: str):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.content, "html.parser")
        section = soup.find("section", id="shareholding")
        if not section:
            return None
        
        table = section.find("table")
        df = pd.read_html(str(table))[0]
        df = df.rename(columns={df.columns[0]: "category"})
        
        # Melt dates: columns are typically ['Jun 2024', 'Sep 2024', ...]
        melted = df.melt(id_vars=["category"], var_name="quarter", value_name="pct")
        melted["pct"] = pd.to_numeric(melted["pct"].str.replace("%", ""), errors="coerce")
        melted["isin"] = isin
        return melted
    except Exception:
        return None

def main():
    master = pd.read_csv(MASTER)
    records = []
    
    # Run in batches / update quarterly
    for idx, row in master.iterrows():
        if pd.notna(row["screener_url"]) and row["screener_url"] != "":
            res = scrape_screener_shareholding(row["screener_url"], row["isin"])
            if res is not None:
                records.append(res)
            time.sleep(0.2)  # Respect rate limits

    if not records:
        return

    full_shp = pd.concat(records, ignore_index=True)
    
    # Pivot to columns: Promoter, FIIs, DIIs
    pivoted = full_shp.pivot_table(index=["isin", "quarter"], columns="category", values="pct").reset_index()
    
    # Sort and compute QoQ deltas per stock
    pivoted["quarter_dt"] = pd.to_datetime(pivoted["quarter"], format="%b %Y")
    pivoted = pivoted.sort_values(["isin", "quarter_dt"])
    
    for stakeholder in ["Promoters", "FIIs", "DIIs"]:
        if stakeholder in pivoted.columns:
            pivoted[f"{stakeholder}_delta"] = pivoted.groupby("isin")[stakeholder].diff()

    OUT_SHP.parent.mkdir(parents=True, exist_ok=True)
    pivoted.to_parquet(OUT_SHP, index=False, compression="snappy")
    print(f"Shareholding data saved to {OUT_SHP}")

if __name__ == "__main__":
    main()
