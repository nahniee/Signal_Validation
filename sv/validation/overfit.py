"""Evidence that a candidate's out-of-time record is not explained by chance.

All three tests run on weekly active returns (strategy minus the equal-weighted
tradable universe), which controls some shared universe exposure but neither removes
survivorship bias nor measures it.

Stationary block bootstrap (Politis & Romano 1994) gives a confidence interval for the
annualised Sharpe ratio; quarter-length blocks preserve the autocorrelation and
volatility clustering an i.i.d. bootstrap would destroy.

Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) is the probability that the observed
Sharpe exceeds the best of N noise strategies, adjusted for skew, kurtosis and sample length.

The cross-sectional permutation null resamples holdings within each signal date, keeping
the dates, universe and top-N mechanics, and compares gross returns: a shuffled signal
turns the book over almost completely each week, so a net-return comparison would reward
low turnover rather than selection.
"""
import numpy as np
import pandas as pd
from scipy import stats
import config
from sv import db, backtest

ANN = 52
EULER_GAMMA = 0.5772156649


def sharpe(r: pd.Series | np.ndarray, ann: int = ANN) -> float:
    r = np.asarray(r, dtype=float)
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ann)) if r.std(ddof=1) > 0 else np.nan


def active_returns(con, strategy: str, bench: str = "universe_ew") -> pd.Series:
    df = db.read(con, """SELECT s.date, s.ret_net - b.ret_net AS active
                         FROM portfolio_returns s JOIN portfolio_returns b ON b.date = s.date AND b.strategy = $b
                         WHERE s.strategy = $s AND s.date >= $start ORDER BY s.date""", {"s": strategy, "b": bench, "start": config.OOT_START})
    return df.set_index("date")["active"]


def stationary_bootstrap_indices(n: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano: geometric block lengths with mean `block`, wrapping."""
    p = 1.0 / block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for i in range(1, n):
        idx[i] = rng.integers(n) if rng.random() < p else (idx[i - 1] + 1) % n
    return idx


def bootstrap_sharpe_ci(r: pd.Series, n_boot: int = config.BOOTSTRAP_N, block: float = config.BOOTSTRAP_BLOCK,
                        alpha: float = 0.05, seed: int = 0) -> dict:
    x = r.dropna().values
    rng = np.random.default_rng(seed)
    boots = np.array([sharpe(x[stationary_bootstrap_indices(len(x), block, rng)]) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"statistic": sharpe(x), "ci_low": lo, "ci_high": hi,
            "p_value": float((boots <= 0).mean()),          # one-sided P(SR <= 0)
            "verdict": "PASS" if lo > 0 else "FAIL",
            "detail": f"stationary block bootstrap, block={block}w, n={n_boot}, T={len(x)}w"}


def probabilistic_sharpe(sr: float, sr_bench: float, T: int, skew: float, kurt: float) -> float:
    """PSR: P(true SR > sr_bench). sr, sr_bench in per-period units. kurt = raw (3 for normal)."""
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf((sr - sr_bench) * np.sqrt(T - 1) / denom))


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """E[max SR] of n_trials i.i.d. noise strategies with Sharpe variance var_sr (per period)."""
    if n_trials <= 1:
        return 0.0
    return np.sqrt(var_sr) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
                              + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * np.e)))


def deflated_sharpe(r: pd.Series, trial_sharpes_ann: list[float], n_trials: int | None = None) -> dict:
    """DSR = PSR evaluated at the expected maximum Sharpe of the trials actually run."""
    x = r.dropna().values
    T = len(x)
    sr = x.mean() / x.std(ddof=1)                                   # per-week
    trials = np.array(trial_sharpes_ann) / np.sqrt(ANN)             # per-week
    n = n_trials or len(trials)
    var_sr = trials.var(ddof=1) if len(trials) > 1 else 1.0 / T
    sr0 = expected_max_sharpe(var_sr, n)
    dsr = probabilistic_sharpe(sr, sr0, T, stats.skew(x), stats.kurtosis(x, fisher=False))
    return {"statistic": dsr, "ci_low": None, "ci_high": None, "p_value": 1 - dsr,
            "verdict": "PASS" if dsr >= 0.95 else "FAIL",
            "detail": f"N_trials={n}, SR0(ann)={sr0*np.sqrt(ANN):.2f}, skew={stats.skew(x):.2f}, "
                      f"kurt={stats.kurtosis(x, fisher=False):.1f}, T={T}w"}


def permutation_null(con, model: str, n_perm: int = 500, seed: int = 0, bench: str = "universe_ew") -> dict:
    panel = backtest._panel(con, model).dropna(subset=["score"])
    panel = panel[panel.date >= pd.Timestamp(config.OOT_START)]
    ew = db.read(con, "SELECT date, ret_gross FROM portfolio_returns WHERE strategy = $b", {"b": bench}) \
           .set_index("date")["ret_gross"]
    obs_pr = backtest.portfolio_returns(backtest.top_n_weights(panel))
    obs = sharpe((obs_pr.set_index("date")["ret_gross"] - ew).dropna())
    rng = np.random.default_rng(seed)
    groups = list(panel.groupby("date", sort=True))
    null_returns = np.empty((n_perm, len(groups)))
    for j, (date, group) in enumerate(groups):
        returns = group[config.EXECUTION_RETURN].to_numpy()
        if not np.isfinite(returns).all():
            raise ValueError("Permutation universe contains missing realised returns")
        k = min(config.HOLD_TOP_N, len(group))
        # Exchangeability null conditional on this snapshot/universe; not a causal test.
        for i in range(n_perm):
            null_returns[i, j] = returns[rng.choice(len(returns), k, replace=False)].mean() - ew.loc[date]
    null = null_returns.mean(axis=1) / null_returns.std(axis=1, ddof=1) * np.sqrt(ANN)
    p = float((1 + (null >= obs).sum()) / (n_perm + 1))
    return {"statistic": obs, "ci_low": float(np.percentile(null, 2.5)), "ci_high": float(np.percentile(null, 97.5)),
            "p_value": p, "verdict": "PASS" if p < 0.05 else "FAIL",
            "detail": f"cross-sectional permutation on gross active return, n={n_perm}; "
                      f"null mean={null.mean():.2f} sd={null.std():.2f}",
            "null": null}


def save_result(con, model: str, test: str, res: dict) -> None:
    con.execute("INSERT OR REPLACE INTO validation_results(model,test,statistic,ci_low,ci_high,p_value,verdict,detail) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [model, test, res["statistic"], res["ci_low"], res["ci_high"], res["p_value"], res["verdict"], res["detail"]])


def run_all(con, models: list[str], n_trials: int | None = None, n_perm: int = 500) -> pd.DataFrame:
    act = {m: active_returns(con, m) for m in models}
    import json
    registry = json.loads((config.ROOT / "experiments.json").read_text())
    trial_names = [x["model"] for x in registry["trials"]]
    trial_series = {name: active_returns(con, name) for name in trial_names}
    available = {name: values for name, values in trial_series.items() if len(values)}
    missing = [name for name in trial_names if name not in available]
    trial_sr = [sharpe(a) for a in available.values()]
    n_trials = max(n_trials or 0, len(trial_names))

    rows = []
    for m in models:
        r = act[m]
        for test, res in (("bootstrap_sharpe_ci", bootstrap_sharpe_ci(r)),
                          ("deflated_sharpe", deflated_sharpe(r, trial_sr, n_trials)),
                          ("permutation_null", permutation_null(con, m, n_perm=n_perm))):
            res["detail"] += f"; OOT >= {config.OOT_START}; next-session-close execution"
            if test == "deflated_sharpe":
                res["verdict"] = "PROVISIONAL"
                res["detail"] += (f"; historical search incomplete; variance from {len(available)} available trials; "
                                  f"missing={','.join(missing)}; N={n_trials} is a lower bound, not an approval test")
            save_result(con, m, test, res)
            rows.append({"model": m, "test": test, **{k: v for k, v in res.items() if k != "null"}})
            print(f"{m:14s} {test:20s} stat={res['statistic']:.3f} p={res['p_value']:.3f} {res['verdict']}  {res['detail']}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    con = db.connect()
    models = sys.argv[1:] or ["momentum", "gbm", "gbm_expected"]
    run_all(con, models, n_trials=config.N_TRIALS_DSR)
