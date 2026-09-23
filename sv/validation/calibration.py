"""Target-matched descriptive calibration, never a cross-horizon skill claim."""
import numpy as np
import pandas as pd
import config
from sv import db, features


def evaluate(con):
    calendar = pd.DatetimeIndex(db.read(con,"SELECT date FROM prices WHERE ticker='SPY' AND date <= $end ORDER BY date", {"end": config.EVALUATION_END}).date)
    adj = features.load_wide(con,'adj_close').reindex(calendar)
    high = features.load_wide(con,'high').reindex(calendar)
    rows=[]
    specs = {
        'gbm': (adj.rolling(64,min_periods=64).mean().shift(-63)/adj-1,63,'mean adjusted price over t..t+63 / price[t] - 1'),
        'gbm_expected': (adj.rolling(64,min_periods=64).mean().shift(-63)/adj-1,63,'mean adjusted price over t..t+63 / price[t] - 1'),
        'gbm_weekly': (adj.shift(-5)/adj-1,5,'adjusted close[t+5] / close[t] - 1'),
        'clam_2021': (high.shift(-65)/high-1,65,'raw High[t+65] / High[t] - 1'),
    }
    for model,(target,horizon,description) in specs.items():
        end_dates = pd.Series(calendar,index=calendar).shift(-horizon)
        scores=db.read(con,"SELECT s.date,s.ticker,s.score FROM signals s JOIN panel p USING(date,ticker) WHERE s.model=$m AND p.tradable AND s.date >= $start AND s.date <= $end",{'m':model,'start':config.OOT_START,'end':config.EVALUATION_END})
        target=target.reindex(pd.DatetimeIndex(scores.date.unique()).sort_values())
        actual=target.stack(future_stack=True).rename('actual').reset_index()
        actual.columns=['date','ticker','actual']
        pairs=scores.merge(actual,on=['date','ticker']).replace([np.inf,-np.inf],np.nan).dropna()
        # Exclude outcomes not mature by the stated evaluation cutoff.
        pairs=pairs[pairs.date.map(end_dates) <= pd.Timestamp(config.EVALUATION_END)]
        slope,intercept=np.polyfit(pairs.score,pairs.actual,1)
        rows.append({'model':model,'horizon_sessions':horizon,'target':description,'n':len(pairs),
                     'slope':float(slope),'intercept':float(intercept),
                     'mae':float(np.mean(np.abs(pairs.actual-pairs.score))),
                     'interpretation':'descriptive pooled OOT regression; overlapping/correlated observations; no significance claim'})
    return pd.DataFrame(rows)
