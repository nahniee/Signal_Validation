"""Build the weekly rebalance panel (sql/queries/build_panel.sql) and expose
a wide price matrix for the candidate models."""
import pandas as pd
import config
from sv import db


def build_panel(con) -> None:
    con.execute("DELETE FROM panel")
    db.run_sql_file(con, "build_panel", {"min_price": config.MIN_PRICE, "min_adv": config.MIN_ADV_USD,
                                         "start": config.BACKTEST_START})


def rebalance_dates(con) -> list[pd.Timestamp]:
    return db.read(con, "SELECT DISTINCT date FROM panel ORDER BY date")["date"].tolist()


def load_wide(con, field: str = "adj_close", start: str | None = None) -> pd.DataFrame:
    """T x N matrix of a price field for the whole universe (daily)."""
    q = f"SELECT date, ticker, {field} AS v FROM prices WHERE {field} > 0"
    if start:
        q += " AND date >= $start"
    df = db.read(con, q, {"start": start} if start else None)
    return df.pivot(index="date", columns="ticker", values="v").sort_index()


if __name__ == "__main__":
    con = db.connect()
    build_panel(con)
    print(db.read(con, """
        SELECT COUNT(*) AS rows, COUNT(DISTINCT date) AS weeks, COUNT(DISTINCT ticker) AS tickers,
               MIN(date) AS d0, MAX(date) AS d1, AVG(tradable::INT) AS tradable_share
        FROM panel"""))
