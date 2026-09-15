"""Download daily OHLCV (+ adjusted close) from Yahoo Finance into the prices table.

Batches of tickers are downloaded together and stored long-format. Re-runnable:
existing (ticker, date) rows are replaced (INSERT OR REPLACE).
"""
import sys
import time
import numpy as np
import pandas as pd
import yfinance as yf
import config
from sv import db

BATCH = 100
FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def download_batch(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    for attempt in range(3):
        try:
            raw = yf.download(tickers, start=start, end=end, interval="1d",
                              auto_adjust=False, group_by="column", threads=True, progress=False)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt+1}: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    else:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):       # single ticker
        raw.columns = pd.MultiIndex.from_product([raw.columns, tickers])
    long = raw[FIELDS].stack(level=1, future_stack=True).reset_index()
    long.columns = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    long = long.dropna(subset=["close"])
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
    return long


def upsert(con, df: pd.DataFrame) -> None:
    con.executemany(
        "INSERT OR REPLACE INTO prices(date,ticker,open,high,low,close,adj_close,volume) VALUES (?,?,?,?,?,?,?,?)",
        df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
          .replace({np.nan: None}).itertuples(index=False, name=None))
    con.commit()


def ingest(con, tickers: list[str], start=config.PRICE_START, end=config.PRICE_END) -> None:
    n = 0
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        df = download_batch(batch, start, end)
        if not df.empty:
            upsert(con, df)
            n += len(df)
        print(f"[{i+len(batch):>5}/{len(tickers)}] rows so far: {n:,}", flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    con = db.connect()
    db.init_schema(con)
    tickers = db.read(con, "SELECT ticker FROM universe ORDER BY rank_by_mcap")["ticker"].tolist()
    extra = [config.BENCHMARK]
    tickers = extra + [t for t in tickers if t not in extra]
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        done = set(db.read(con, "SELECT DISTINCT ticker FROM prices")["ticker"])
        tickers = [t for t in tickers if t not in done]
        print(f"resume: {len(tickers)} tickers remaining")
    ingest(con, tickers)
    s = db.read(con, "SELECT COUNT(*) n, COUNT(DISTINCT ticker) k, MIN(date) d0, MAX(date) d1 FROM prices")
    print(s)
