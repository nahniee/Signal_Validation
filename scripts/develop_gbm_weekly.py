"""Developer-side model selection for GBM weekly v2.

Scores all six variants, backtests them, and picks the best *development-sample*
active Sharpe (dates < config.OOT_START). Out-of-time performance is deliberately
not printed here - that is the validator's job. The winner is copied to model
name 'gbm_weekly'; the six variants stay in `signals` as DSR trials.
"""
import sys
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import config
from sv import db, features, backtest
from sv.candidates import gbm_weekly
from sv.validation.overfit import sharpe

con = db.connect()
dates = features.rebalance_dates(con)
adj = features.load_wide(con, "adj_close")
df = gbm_weekly.scores(adj, dates)
df = df[np.isfinite(df.score)]
for name in gbm_weekly.VARIANTS + ["gbm_weekly"]:
    con.execute("DELETE FROM signals WHERE model = ?", [name])
db.write_df(con, df[["date", "ticker", "model", "score"]], "signals")

ew = db.read(con, "SELECT date, ret_net FROM portfolio_returns WHERE strategy = 'universe_ew'").set_index("date")["ret_net"]
rows = {}
for name in gbm_weekly.VARIANTS:
    pr = backtest.run_model(con, name)
    backtest.save(con, pr)
    dev = pr[pr.date < pd.Timestamp(config.OOT_START)].set_index("date")
    act = (dev.ret_net - ew.reindex(dev.index)).dropna()
    rows[name] = {"dev_weeks": len(act), "dev_active_sharpe": sharpe(act), "dev_cagr": (1 + dev.ret_net).prod() ** (52 / len(dev)) - 1,
                  "turnover": dev.turnover.mean()}
tab = pd.DataFrame(rows).T.sort_values("dev_active_sharpe", ascending=False)
pd.set_option("display.width", 160); print(tab.round(3))

best = tab.index[0]
con.execute("INSERT INTO signals SELECT date, ticker, 'gbm_weekly' AS model, score FROM signals WHERE model = ?", [best])
backtest.save(con, backtest.run_model(con, "gbm_weekly"))
backtest.save(con, backtest.run_model(con, "gbm_weekly", strategy="gbm_weekly_lag1", ret_col="ret_fwd_1w_lag1"))
print(f"\nselected {best} -> 'gbm_weekly'  (development sample only; {len(gbm_weekly.VARIANTS)} trials)")
