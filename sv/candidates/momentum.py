"""Untuned 12-1 cross-sectional momentum, used as the benchmark rule.

score = P[t-21] / P[t-252] - 1 on adjusted close."""
import numpy as np
import pandas as pd
import config

NAME = "momentum"


def scores(adj_close: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    lb, skip = config.MOM_LOOKBACK, config.MOM_SKIP
    s = adj_close.shift(skip) / adj_close.shift(lb) - 1.0
    out = s.reindex(dates).stack(future_stack=True).rename("score").reset_index()
    out.columns = ["date", "ticker", "score"]
    out["model"] = NAME
    return out.dropna(subset=["score"])
