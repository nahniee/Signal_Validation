"""Candidate 1: GBM path-simulation score, ported verbatim from
Long_Term_Trading/stats_model_process.get_gbm_path_simulation (mode='quarterly').

As deployed: fit mu, sigma on ~2y of daily log returns, simulate ONE 65-step GBM
path, score = mean(path)/S0 - 1. The single Monte-Carlo draw is reproduced with a
fixed seed per (date) so the backtest is deterministic.

Validation note (see report): E[score] = mean_i exp(mu * t_i) - 1 depends on mu
only, so the ranking is in expectation a monotone transform of trailing mean log
return; the MC draw adds pure noise of order sigma*sqrt(T). We also emit the
deterministic expectation as model 'gbm_expected' to quantify that noise.
"""
import numpy as np
import pandas as pd
import config

NAME = "gbm"
NAME_EXPECTED = "gbm_expected"
T, N, SCALE = 0.25, config.GBM_HORIZON, 252


def _mu_sigma(adj_close: pd.DataFrame, dates: list[str]):
    lr = np.log(adj_close).diff()
    lb = config.GBM_LOOKBACK
    mu = lr.rolling(lb, min_periods=int(lb * 0.9)).mean() * SCALE
    sig = lr.rolling(lb, min_periods=int(lb * 0.9)).std() * np.sqrt(SCALE)
    return mu.reindex(dates), sig.reindex(dates)


def scores(adj_close: pd.DataFrame, dates: list[str], seed: int = 42) -> pd.DataFrame:
    mu, sig = _mu_sigma(adj_close, dates)
    dt = T / N
    grid = np.linspace(0, T, N + 1)
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        m, s = mu.loc[d].values, sig.loc[d].values
        ok = ~(np.isnan(m) | np.isnan(s))
        if not ok.any():
            continue
        # one Brownian path per ticker, exactly as deployed
        inc = rng.normal(0, np.sqrt(dt), size=(ok.sum(), N))
        W = np.concatenate([np.zeros((ok.sum(), 1)), np.cumsum(inc, axis=1)], axis=1)
        path = np.exp((m[ok, None] - 0.5 * s[ok, None] ** 2) * grid[None, :] + s[ok, None] * W)
        mc = path.mean(axis=1) - 1.0                                   # S0 cancels
        expected = np.exp(m[ok, None] * grid[None, :]).mean(axis=1) - 1.0
        tick = mu.columns[ok]
        rows.append(pd.DataFrame({"date": d, "ticker": tick, "score": mc, "model": NAME}))
        rows.append(pd.DataFrame({"date": d, "ticker": tick, "score": expected, "model": NAME_EXPECTED}))
    return pd.concat(rows, ignore_index=True)
