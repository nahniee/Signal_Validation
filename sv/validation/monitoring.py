"""Rolling monitoring metrics in the vocabulary of bank model-risk reports.

Every metric for signal date t uses the trailing MONITOR_WINDOW weeks of pooled
cross-sections, and only targets that have matured by t.

  psi_score              Population Stability Index of the score distribution against the
                         development sample (first 104 weeks), binned on that sample.
                         Below 0.10 stable, 0.10-0.25 watch, above 0.25 shifted.
  psi_input              PSI of trailing 252-day realised volatility, a regime measure
                         shared by every candidate.
  auc_1w, ks_1w          Separation of next-week winners from losers, where a winner beats
                         the cross-sectional median. AUC 0.5 is no discriminatory power.
  auc_13w, ks_13w        The same at the 65-day horizon, using exact target-maturity dates.
  association_slope_13w  Descriptive regression of realised on predicted, not calibration:
                         targets and units differ across candidates.
  rolling_sharpe         52-week annualised Sharpe of active return against universe_ew.
  drawdown               Current drawdown of the net strategy equity curve.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import config
from sv import db

DEV_WEEKS = 104
W = config.MONITOR_WINDOW


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = config.PSI_BUCKETS, edges=None) -> float:
    if edges is None:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    # Infinite tails keep drifted observations inside the probability distribution.
    expected = np.asarray(expected); actual = np.asarray(actual)
    expected = expected[np.isfinite(expected)]; actual = actual[np.isfinite(actual)]
    if not len(expected) or not len(actual):
        return np.nan
    interior = np.unique(np.asarray(edges)[1:-1])
    edges = np.r_[-np.inf, interior, np.inf]
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_stat(score: np.ndarray, y: np.ndarray) -> float:
    s1, s0 = np.sort(score[y == 1]), np.sort(score[y == 0])
    if len(s1) == 0 or len(s0) == 0:
        return np.nan
    grid = np.sort(score)
    c1 = np.searchsorted(s1, grid, side="right") / len(s1)
    c0 = np.searchsorted(s0, grid, side="right") / len(s0)
    return float(np.max(np.abs(c1 - c0)))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if 0 < y.mean() < 1 else np.nan


def load_scored_panel(con, model: str) -> pd.DataFrame:
    return db.read(con, """
        WITH maturity AS (SELECT date, LEAD(date,65) OVER (ORDER BY date) AS target_date_13w FROM prices WHERE ticker='SPY' AND date <= $end)
        SELECT p.date, p.ticker, s.score, p.ret_fwd_1w_lag1 AS ret_fwd_1w, p.ret_fwd_13w, t.target_date_13w
        FROM panel p JOIN signals s ON s.date = p.date AND s.ticker = p.ticker AND s.model = $m
        JOIN maturity t ON t.date=p.date
        WHERE p.tradable AND p.date <= $end ORDER BY p.date""", {"m": model, "end": config.EVALUATION_END})


def realised_vol_panel(con) -> pd.DataFrame:
    """Trailing 252d realised vol per (rebalance date, ticker) via SQL window function."""
    return db.run_sql_file(con, "realised_vol")


def rolling_metrics(con, model: str) -> pd.DataFrame:
    df = load_scored_panel(con, model)
    df["y1"] = (df.ret_fwd_1w > df.groupby("date").ret_fwd_1w.transform("median")).astype(float)
    df["y13"] = (df.ret_fwd_13w > df.groupby("date").ret_fwd_13w.transform("median")).astype(float)
    dates = sorted(df.date.drop_duplicates().tolist())
    dev = df[df.date.isin(dates[:DEV_WEEKS])]
    edges = np.unique(np.quantile(dev.score, np.linspace(0, 1, config.PSI_BUCKETS + 1)))

    vol = realised_vol_panel(con)
    vol_dev = vol[vol.date.isin(dates[:DEV_WEEKS])].vol252.dropna().values
    vol_edges = np.unique(np.quantile(vol_dev, np.linspace(0, 1, config.PSI_BUCKETS + 1)))
    by_date = {d: g for d, g in df.groupby("date")}
    vol_by_date = {d: g.vol252.dropna().values for d, g in vol.groupby("date")}

    rows = []
    for i in range(max(W + 1, DEV_WEEKS), len(dates)):
        win = dates[i - W - 1:i - 1]                            # excludes the latest unfinished delayed holding period
        g = pd.concat([by_date[d] for d in win])
        cur = g.dropna(subset=["score"])
        m = {"psi_score": psi(dev.score.values, cur.score.values, edges=edges),
             "psi_input": psi(vol_dev, np.concatenate([vol_by_date.get(d, []) for d in win]), edges=vol_edges)}
        g1 = g.dropna(subset=["score", "ret_fwd_1w"])
        m["auc_1w"], m["ks_1w"] = auc(g1.score.values, g1.y1.values), ks_stat(g1.score.values, g1.y1.values)
        # Use each target's real maturity date, which accounts for holidays.
        known = df[(df.date < dates[i]) & (df.target_date_13w <= dates[i])]
        mature_dates = sorted(known.date.unique())[-W:]
        g13 = known[known.date.isin(mature_dates)].dropna(subset=["score", "ret_fwd_13w"])
        if len(g13):
            m["auc_13w"], m["ks_13w"] = auc(g13.score.values, g13.y13.values), ks_stat(g13.score.values, g13.y13.values)
            x, y = g13.score.values, g13.ret_fwd_13w.values
            # Descriptive only: these scores forecast a different target.
            # GBM path-average != terminal return; CLAM High != Close; weekly != 65d.
            m["association_slope_13w"] = float(np.polyfit(x, y, 1)[0]) if x.std() > 0 else np.nan
        rows.append({"date": dates[i], **m})
    out = pd.DataFrame(rows).melt(id_vars="date", var_name="metric", value_name="value")
    out["model"] = model
    return out


def portfolio_metrics(con, strategy: str, bench: str = "universe_ew") -> pd.DataFrame:
    pr = db.read(con, """SELECT s.date, s.ret_net, b.ret_net AS benchmark, s.ret_net - b.ret_net AS active
                         FROM portfolio_returns s JOIN portfolio_returns b ON b.date = s.date AND b.strategy = $b
                         WHERE s.strategy = $s ORDER BY s.date""", {"s": strategy, "b": bench}).set_index("date")
    rs = pr.active.rolling(W).mean() / pr.active.rolling(W).std() * np.sqrt(52)
    eq = (1 + pr.ret_net).cumprod()
    dd = eq / eq.cummax().clip(lower=1.0) - 1
    rel = (1 + pr.ret_net).cumprod() / (1 + pr.benchmark).cumprod()
    rel_dd = rel / rel.cummax().clip(lower=1.0) - 1
    out = pd.DataFrame({"rolling_sharpe": rs, "drawdown": dd, "active_drawdown": rel_dd}).reset_index()
    out = out.melt(id_vars="date", var_name="metric", value_name="value").dropna()
    out["model"] = strategy
    return out


def save(con, m: pd.DataFrame) -> None:
    for model in m.model.unique():
        con.execute("DELETE FROM monitoring WHERE model = ?", [model])
    db.write_df(con, m.dropna(subset=["value"])[["date", "model", "metric", "value"]], "monitoring")


def run(con, models: list[str]) -> None:
    for m in models:
        out = pd.concat([rolling_metrics(con, m), portfolio_metrics(con, m)], ignore_index=True)
        save(con, out)
        last = out[out.date == out.date.max()].set_index("metric")["value"].round(3)
        print(m, last.to_dict(), flush=True)


if __name__ == "__main__":
    import sys
    con = db.connect()
    run(con, sys.argv[1:] or ["momentum", "gbm", "gbm_expected"])
