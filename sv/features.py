"""Build the weekly rebalance panel (sql/queries/build_panel.sql) and expose
a wide price matrix for the candidate models."""
import pandas as pd
import config
from sv import db


def build_panel(con) -> None:
    con.execute("DELETE FROM panel")
    sql = (db.SQL_DIR / "queries" / "build_panel.sql").read_text()
    con.execute(sql, {"min_price": config.MIN_PRICE, "min_adv": config.MIN_ADV_USD,
                      "start": config.BACKTEST_START})
    con.commit()


def rebalance_dates(con) -> list[str]:
    return db.read(con, "SELECT DISTINCT date FROM panel ORDER BY date")["date"].tolist()


def load_wide(con, field: str = "adj_close", start: str | None = None) -> pd.DataFrame:
    """T x N matrix of a price field for the whole universe (daily)."""
    q = f"SELECT date, ticker, {field} AS v FROM prices"
    if start:
        q += f" WHERE date >= '{start}'"
    df = db.read(con, q)
    return df.pivot(index="date", columns="ticker", values="v").sort_index()


if __name__ == "__main__":
    con = db.connect()
    build_panel(con)
    print(db.read(con, """
        SELECT COUNT(*) rows, COUNT(DISTINCT date) weeks, COUNT(DISTINCT ticker) tickers,
               MIN(date) d0, MAX(date) d1, AVG(tradable) tradable_share
        FROM panel"""))
