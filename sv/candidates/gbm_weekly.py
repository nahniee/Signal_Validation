"""Candidate 1b: GBM weekly v2, ported from Long_Term_Trading/gbm_weekly.py
(vectorised over the whole universe). Six development variants are emitted:
lookback {63, 126, 252} x score {expected, prob_up}; the one chosen on the
development sample (< config.OOT_START) is aliased as 'gbm_weekly'. All six stay
in `signals` so the Deflated Sharpe Ratio can count them as trials.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

HORIZON = 5
LOOKBACKS = (63, 126, 252)
MODES = ("expected", "prob_up")
VARIANTS = [f"gbm_w_{lb}_{m}" for lb in LOOKBACKS for m in MODES]


def scores(adj_close: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    lr = np.log(adj_close).diff()
    rows = []
    for lb in LOOKBACKS:
        mu = lr.rolling(lb, min_periods=int(lb * 0.9)).mean().reindex(dates)
        sd = lr.rolling(lb, min_periods=int(lb * 0.9)).std().reindex(dates)
        expected = np.exp(mu * HORIZON) - 1.0
        prob_up = pd.DataFrame(norm.cdf((mu - 0.5 * sd ** 2) * np.sqrt(HORIZON) / sd),
                               index=mu.index, columns=mu.columns).where(sd > 0)
        for m, s in (("expected", expected), ("prob_up", prob_up)):
            out = s.stack(future_stack=True).rename("score").reset_index()
            out.columns = ["date", "ticker", "score"]
            out["model"] = f"gbm_w_{lb}_{m}"
            rows.append(out.dropna(subset=["score"]))
    return pd.concat(rows, ignore_index=True)
