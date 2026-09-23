"""Build the tradable universe: top-N US-listed common equities by market cap.

The list comes from today's Yahoo screener, so companies delisted before today are
missing. That survivorship bias can affect each strategy differently. Subtracting a
shared benchmark doesn't remove it, and counting listings doesn't measure its effect
on returns.
"""
import json
import time
import pandas as pd
import yfinance as yf
import config
from sv import db


def fetch_screener(size_target: int = config.UNIVERSE_SIZE, page: int = 250) -> pd.DataFrame:
    q = yf.EquityQuery("and", [
        yf.EquityQuery("eq", ["region", "us"]),
        yf.EquityQuery("is-in", ["exchange", *config.EXCHANGES]),
    ])
    rows, offset = [], 0
    # over-fetch ~15% so that dedup of share classes / non-equities still leaves size_target
    while len(rows) < size_target * 1.15:
        r = yf.screen(q, sortField="intradaymarketcap", sortAsc=False, size=page, offset=offset)
        quotes = r.get("quotes", [])
        if not quotes:
            break
        rows.extend(quotes)
        offset += page
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    keep = ["symbol", "shortName", "exchange", "marketCap", "quoteType", "sector", "industry"]
    for c in keep:
        if c not in df.columns:
            df[c] = None
    df = df[keep].rename(columns={"symbol": "ticker", "shortName": "name", "marketCap": "market_cap"})
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["quoteType"] == "EQUITY"].copy()
    df = df[df["market_cap"].notna() & (df["market_cap"] > 0)]
    # drop preferreds / warrants / units (Yahoo uses '-' suffixes) and keep one class per company
    df = df[~df["ticker"].str.contains(r"[-^$]", regex=True)]
    df = df.sort_values("market_cap", ascending=False)
    df = df.drop_duplicates(subset=["name"], keep="first")   # GOOGL/GOOG, BRK-A/B -> largest class
    df = df.drop_duplicates(subset=["ticker"])
    df = df.head(config.UNIVERSE_SIZE).reset_index(drop=True)
    df["rank_by_mcap"] = df.index + 1
    df["asof_date"] = pd.Timestamp(config.UNIVERSE_ASOF)
    return df.drop(columns=["quoteType"])


def build(con) -> pd.DataFrame:
    raw = fetch_screener()
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    (config.RAW_DIR / "screener_raw.json").write_text(raw.to_json(orient="records"))
    uni = clean(raw)
    con.execute("DELETE FROM universe")
    db.write_df(con, uni, "universe")
    return uni


if __name__ == "__main__":
    con = db.connect()
    db.init_schema(con)
    uni = build(con)
    print(f"universe: {len(uni)} tickers, mcap range "
          f"${uni.market_cap.min()/1e6:,.0f}M to ${uni.market_cap.max()/1e9:,.0f}B")
    print(uni.head(10)[["ticker", "name", "market_cap"]])
    print(uni.tail(5)[["ticker", "name", "market_cap"]])
