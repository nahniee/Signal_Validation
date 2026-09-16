"""Compute momentum + GBM scores for every rebalance date and write to `signals`."""
import sys
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import config
from sv import db, features
from sv.candidates import momentum, gbm

con = db.connect()
dates = features.rebalance_dates(con)
adj = features.load_wide(con, "adj_close")
print(f"{len(dates)} rebalance dates, price matrix {adj.shape}")

for mod in (momentum, gbm):
    df = mod.scores(adj, dates)
    df = df[np.isfinite(df.score)]
    for name in df["model"].unique():
        con.execute("DELETE FROM signals WHERE model = ?", [name])
    db.write_df(con, df[["date", "ticker", "model", "score"]], "signals")
    print(mod.NAME, len(df), "rows")
print(db.read(con, "SELECT model, COUNT(*) AS n, COUNT(DISTINCT date) AS d FROM signals GROUP BY model ORDER BY model"))
