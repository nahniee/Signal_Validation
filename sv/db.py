"""Thin SQLite helpers. All analytics tables are created from sql/schema.sql so the
schema is reviewable as plain SQL."""
import sqlite3
from pathlib import Path
import pandas as pd
import config

SQL_DIR = config.ROOT / "sql"


def connect(path: Path = config.DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def init_schema(con: sqlite3.Connection) -> None:
    con.executescript((SQL_DIR / "schema.sql").read_text())
    con.commit()


def run_sql_file(con: sqlite3.Connection, name: str, params: dict | None = None) -> pd.DataFrame:
    """Execute sql/queries/<name>.sql and return a DataFrame."""
    sql = (SQL_DIR / "queries" / f"{name}.sql").read_text()
    return pd.read_sql_query(sql, con, params=params or {})


def write_df(con: sqlite3.Connection, df: pd.DataFrame, table: str, if_exists: str = "append") -> int:
    df.to_sql(table, con, if_exists=if_exists, index=False, chunksize=5_000)
    con.commit()
    return len(df)


def read(con: sqlite3.Connection, sql: str, params: dict | tuple | None = None) -> pd.DataFrame:
    return pd.read_sql_query(sql, con, params=params or {})
