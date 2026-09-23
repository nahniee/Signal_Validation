"""Weekly long-only top-N backtest shared by every candidate.

The universe, costs and execution assumption are identical for all models, so
differences in outcome come from the scores. A signal formed at the week-end close is
traded at the next session close and held to the following execution close; same-close
results are kept only as an optimistic diagnostic. Costs are COST_BPS one-way on traded weight.
"""
import numpy as np
import pandas as pd
import config
from sv import db


def _panel(con, model: str | None = None) -> pd.DataFrame:
    q = """SELECT p.date, p.ticker, p.ret_fwd_1w, p.ret_fwd_1w_lag1, s.score
           FROM panel p LEFT JOIN signals s ON s.date = p.date AND s.ticker = p.ticker AND s.model = $m
           WHERE p.tradable AND p.date <= $end AND p.date IN (SELECT date FROM panel WHERE ticker = 'SPY' AND ret_fwd_1w_lag1 IS NOT NULL)"""
    return db.read(con, q, {"m": model, "end": config.EVALUATION_END})


def top_n_weights(df: pd.DataFrame, n: int = config.HOLD_TOP_N, score_col: str = "score") -> pd.DataFrame:
    """Equal-weight top-n by score within each date. Returns (date, ticker, w)."""
    d = df.dropna(subset=[score_col])
    d = d.assign(rk=d.groupby("date")[score_col].rank(ascending=False, method="first"))
    d = d[d.rk <= n]
    d["w"] = 1.0 / d.groupby("date")["ticker"].transform("size")
    return d[["date", "ticker", "w", "ret_fwd_1w", "ret_fwd_1w_lag1"]]


def portfolio_returns(w: pd.DataFrame, ret_col: str = config.EXECUTION_RETURN, cost_bps: float = config.COST_BPS) -> pd.DataFrame:
    """Gross/net weekly returns + one-way turnover from a (date, ticker, w) table."""
    W = w.pivot(index="date", columns="ticker", values="w").fillna(0.0).sort_index()
    raw = w.pivot(index="date", columns="ticker", values=ret_col).reindex_like(W)
    if ((W > 0) & raw.isna()).any().any():
        raise ValueError("Missing held-name return: investigate prices; never silently replace with zero")
    R = raw.fillna(0.0)
    gross = (W * R).sum(axis=1)
    # Previous holdings drift before the next execution; include exits as well as entries.
    before = W.shift(1) * (1 + R.shift(1))
    before = before.div(1 + gross.shift(1), axis=0).fillna(0.0)
    traded = (W - before).abs().sum(axis=1)
    net = gross - traded * cost_bps / 1e4
    out = pd.DataFrame({"ret_gross": gross, "ret_net": net, "turnover": traded / 2,
                        "n_held": (W > 0).sum(axis=1)})
    return out.reset_index()


def run_model(con, model: str, strategy: str | None = None, ret_col: str = config.EXECUTION_RETURN) -> pd.DataFrame:
    strategy = strategy or model
    pr = portfolio_returns(top_n_weights(_panel(con, model)), ret_col=ret_col)
    pr["strategy"] = strategy
    return pr


def run_benchmarks(con) -> pd.DataFrame:
    panel = _panel(con)
    # Equal-weight comparator includes its own rebalancing costs.
    ew_w = panel.assign(score=1.0)
    ew = portfolio_returns(top_n_weights(ew_w, n=1_000_000)).assign(strategy="universe_ew")
    spy_w = panel[panel.ticker == config.BENCHMARK].assign(score=1.0)
    spy = portfolio_returns(top_n_weights(spy_w, n=1)).assign(strategy="benchmark_spy")
    return pd.concat([spy, ew], ignore_index=True)


def save(con, pr: pd.DataFrame) -> None:
    pr = pr.dropna(subset=["ret_net"])
    for s in pr.strategy.unique():
        con.execute("DELETE FROM portfolio_returns WHERE strategy = ?", [s])
    db.write_df(con, pr[["date", "strategy", "ret_gross", "ret_net", "turnover", "n_held"]], "portfolio_returns")


def summary(pr: pd.DataFrame, bench: pd.Series | None = None) -> dict:
    r = pr.set_index("date")["ret_net"].dropna()
    ann = 52
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax().clip(lower=1.0) - 1
    out = {"weeks": len(r), "cagr": eq.iloc[-1] ** (ann / len(r)) - 1,
           "vol": r.std() * np.sqrt(ann), "sharpe": r.mean() / r.std() * np.sqrt(ann) if r.std() > 0 else np.nan,
           "max_dd": dd.min(), "turnover_avg": pr["turnover"].mean()}
    if bench is not None:
        a = (r - bench.reindex(r.index)).dropna()
        out["active_ret"] = a.mean() * ann
        out["info_ratio"] = a.mean() / a.std() * np.sqrt(ann) if a.std() > 0 else np.nan
    return out


if __name__ == "__main__":
    import sys
    con = db.connect()
    models = sys.argv[1:] or [m for (m,) in con.execute("SELECT DISTINCT model FROM signals").fetchall()]
    bench = run_benchmarks(con)
    save(con, bench)
    ew = bench[bench.strategy == "universe_ew"].set_index("date")["ret_net"]
    rows = {}
    for m in models:
        pr = run_model(con, m)
        save(con, pr)
        save(con, run_model(con, m, strategy=f"{m}_same_close", ret_col="ret_fwd_1w"))
        rows[m] = summary(pr, ew)
    for s in ("benchmark_spy", "universe_ew"):
        rows[s] = summary(bench[bench.strategy == s], ew)
    pd.set_option("display.width", 160)
    print(pd.DataFrame(rows).T.round(3))
