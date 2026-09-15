"""Weekly long-only top-N backtest shared by every candidate (same universe,
same costs, same execution assumption) so that differences are due to the
signal alone.

Execution: signal on rebalance close -> trade at that close (base case) or at the
next day's close (lag-1 sensitivity). Cost = COST_BPS one-way on traded weight.
"""
import numpy as np
import pandas as pd
import config
from sv import db


def _panel(con, model: str | None = None) -> pd.DataFrame:
    q = """SELECT p.date, p.ticker, p.ret_fwd_1w, p.ret_fwd_1w_lag1, s.score
           FROM panel p LEFT JOIN signals s ON s.date = p.date AND s.ticker = p.ticker AND s.model = :m
           WHERE p.tradable = 1"""
    return db.read(con, q, {"m": model})


def top_n_weights(df: pd.DataFrame, n: int = config.HOLD_TOP_N, score_col: str = "score") -> pd.DataFrame:
    """Equal-weight top-n by score within each date. Returns (date, ticker, w)."""
    d = df.dropna(subset=[score_col])
    d = d.assign(rk=d.groupby("date")[score_col].rank(ascending=False, method="first"))
    d = d[d.rk <= n]
    d["w"] = 1.0 / d.groupby("date")["ticker"].transform("size")
    return d[["date", "ticker", "w", "ret_fwd_1w", "ret_fwd_1w_lag1"]]


def portfolio_returns(w: pd.DataFrame, ret_col: str = "ret_fwd_1w", cost_bps: float = config.COST_BPS) -> pd.DataFrame:
    """Gross/net weekly returns + one-way turnover from a (date, ticker, w) table."""
    W = w.pivot(index="date", columns="ticker", values="w").fillna(0.0).sort_index()
    R = w.pivot(index="date", columns="ticker", values=ret_col).reindex_like(W).fillna(0.0)
    gross = (W * R).sum(axis=1)
    # weights drift with returns before next rebalance; approximate turnover on target weights
    dW = W.diff().abs().sum(axis=1)
    dW.iloc[0] = W.iloc[0].abs().sum()
    net = gross - dW * cost_bps / 1e4
    out = pd.DataFrame({"ret_gross": gross, "ret_net": net, "turnover": dW / 2.0,
                        "n_held": (W > 0).sum(axis=1)})
    return out.reset_index()


def run_model(con, model: str, strategy: str | None = None, ret_col: str = "ret_fwd_1w") -> pd.DataFrame:
    strategy = strategy or model
    pr = portfolio_returns(top_n_weights(_panel(con, model)), ret_col=ret_col)
    pr["strategy"] = strategy
    return pr


def run_benchmarks(con) -> pd.DataFrame:
    spy = db.read(con, "SELECT date, ret_fwd_1w AS ret_gross FROM panel WHERE ticker = :b ORDER BY date",
                  {"b": config.BENCHMARK})
    spy = spy.assign(ret_net=spy.ret_gross, turnover=0.0, n_held=1, strategy="benchmark_spy")
    ew = db.read(con, """SELECT date, AVG(ret_fwd_1w) AS ret_gross, COUNT(*) AS n_held
                         FROM panel WHERE tradable = 1 GROUP BY date ORDER BY date""")
    ew = ew.assign(ret_net=ew.ret_gross, turnover=0.0, strategy="universe_ew")
    return pd.concat([spy, ew], ignore_index=True)


def save(con, pr: pd.DataFrame) -> None:
    pr = pr.dropna(subset=["ret_net"])
    for s in pr.strategy.unique():
        con.execute("DELETE FROM portfolio_returns WHERE strategy = ?", (s,))
    db.write_df(con, pr[["date", "strategy", "ret_gross", "ret_net", "turnover", "n_held"]], "portfolio_returns")


def summary(pr: pd.DataFrame, bench: pd.Series | None = None) -> dict:
    r = pr.set_index("date")["ret_net"].dropna()
    ann = 52
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    out = {"weeks": len(r), "cagr": eq.iloc[-1] ** (ann / len(r)) - 1,
           "vol": r.std() * np.sqrt(ann), "sharpe": r.mean() / r.std() * np.sqrt(ann),
           "max_dd": dd.min(), "turnover_avg": pr["turnover"].mean()}
    if bench is not None:
        a = (r - bench.reindex(r.index)).dropna()
        out["active_ret"] = a.mean() * ann
        out["info_ratio"] = a.mean() / a.std() * np.sqrt(ann)
    return out


if __name__ == "__main__":
    import sys
    con = db.connect()
    models = sys.argv[1:] or [m for (m,) in con.execute("SELECT DISTINCT model FROM signals")]
    bench = run_benchmarks(con)
    save(con, bench)
    ew = bench[bench.strategy == "universe_ew"].set_index("date")["ret_net"]
    rows = {}
    for m in models:
        pr = run_model(con, m)
        save(con, pr)
        save(con, run_model(con, m, strategy=f"{m}_lag1", ret_col="ret_fwd_1w_lag1"))
        rows[m] = summary(pr, ew)
    for s in ("benchmark_spy", "universe_ew"):
        rows[s] = summary(bench[bench.strategy == s], ew)
    pd.set_option("display.width", 160)
    print(pd.DataFrame(rows).T.round(3))
