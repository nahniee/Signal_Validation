"""Thin DuckDB helpers. All tables are created from sql/schema.sql so the schema is
reviewable as plain SQL, and every query can be run unchanged in the DuckDB CLI.

Conventions
  * named parameters are written $name in SQL and passed as a dict
  * DATE columns come back as pandas datetime64[ns]; config dates are ISO strings
    and DuckDB casts them on comparison, so `WHERE date >= $start` just works
  * DuckDB allows one writing process per database file. Run scripts one at a
    time; close a separate-process writer before opening notebooks/readers.
"""
import os
from pathlib import Path
import duckdb
import pandas as pd
import config

SQL_DIR = config.ROOT / "sql"


def connect(path: Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = Path(path or os.environ.get("SV_DB_PATH") or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def init_schema(con) -> None:
    con.execute((SQL_DIR / "schema.sql").read_text())


def read(con, sql: str, params: dict | list | tuple | None = None) -> pd.DataFrame:
    df = con.execute(sql, params if params is not None else {}).df()
    for c in df.select_dtypes("datetime").columns:
        df[c] = df[c].astype("datetime64[ns]")
    return df


def run_sql_file(con, name: str, params: dict | None = None) -> pd.DataFrame:
    """Execute sql/queries/<name>.sql and return a DataFrame (empty for INSERT/UPDATE)."""
    return read(con, (SQL_DIR / "queries" / f"{name}.sql").read_text(), params)


def write_df(con, df: pd.DataFrame, table: str, replace: bool = False) -> int:
    """Append a DataFrame to `table`, matching columns by name.
    replace=True upserts on the table's primary key (INSERT OR REPLACE)."""
    if df.empty:
        return 0
    con.register("_df", df)
    verb = "INSERT OR REPLACE INTO" if replace else "INSERT INTO"
    con.execute(f"{verb} {table} BY NAME SELECT * FROM _df")
    con.unregister("_df")
    return len(df)
