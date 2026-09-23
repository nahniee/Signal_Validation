"""Evaluate the predeclared training-size runs (94/500/3000), then refresh shared evidence."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import config
from sv import db,backtest
from sv.validation import overfit,monitoring,gate

SIZES=(94,500,3000)
CONTROL=500
NAMES=[f'clam_weekly_cs_rank_small_n{n}_seed20260922' for n in SIZES]

def main():
    con=db.connect()
    for name in NAMES:
        backtest.save(con,backtest.run_model(con,name))
        backtest.save(con,backtest.run_model(con,name,strategy=name+'_same_close',ret_col='ret_fwd_1w'))
    # Recompute DSR for all candidates because the number/dispersion of trials changed.
    overfit.run_all(con,config.VALIDATION_MODELS,n_trials=config.N_TRIALS_DSR)
    monitoring.run(con,NAMES);gate.run(con,NAMES)
    registry=json.loads((config.ROOT/'experiments.json').read_text())
    for t in registry['trials']:
        if t['model'] in NAMES:t['status']='trained, scored and evaluated'
    (config.ROOT/'experiments.json').write_text(json.dumps(registry,indent=2)+'\n')
    comparison=[]
    for name in NAMES:
        tag=name.removeprefix('clam_weekly_')
        meta=json.loads((config.QMR_DIR/'clam_weekly'/tag/'clam_weekly_meta.json').read_text())
        pr=db.read(con,'SELECT * FROM portfolio_returns WHERE strategy=$m AND date >= $start ORDER BY date',{'m':name,'start':config.OOT_START})
        act=overfit.active_returns(con,name);s=backtest.summary(pr)
        tests=db.read(con,'SELECT * FROM validation_results WHERE model=$m',{'m':name}).set_index('test')
        comparison.append({'model':name,'requested_tickers':meta['requested_n_tickers'],'usable_tickers':meta['n_tickers'],
           'training_windows':meta['train_windows'],'validation_windows':meta['val_windows'],'purged_windows':meta['purged_windows'],
           'epochs':meta['epochs_run'],'validation_rank_ic':meta['val_rank_ic'],'oot_weeks':len(act),
           'oot_cagr':s['cagr'],'oot_sharpe':s['sharpe'],'oot_active_sharpe':overfit.sharpe(act),'oot_max_drawdown':s['max_dd'],
           'bootstrap':tests.loc['bootstrap_sharpe_ci','verdict'],'permutation_p':tests.loc['permutation_null','p_value']})
    pd.DataFrame(comparison).to_csv(config.REPORTS_DIR/'clam_universe_comparison.csv',index=False)
    series={n:overfit.active_returns(con,f'clam_weekly_cs_rank_small_n{n}_seed20260922') for n in SIZES}
    rng=np.random.default_rng(20260922); contrasts=[]
    for n in SIZES:
        if n==CONTROL: continue
        diff=(series[n]-series[CONTROL]).dropna().to_numpy()
        sampled=np.array([52*diff[overfit.stationary_bootstrap_indices(len(diff),13,rng)].mean() for _ in range(5000)])
        contrasts.append({'contrast':f'{n} minus matched {CONTROL} control; net weekly portfolio returns',
                          'annualized_mean_return_difference':float(52*diff.mean()),
                          'ci_95':[float(x) for x in np.quantile(sampled,[.025,.975])],'weeks':len(diff)})
    paired={'control':CONTROL,'block_weeks':13,'draws':5000,'contrasts':contrasts,
            'limitation':'Single seed, reused OOT and later-universe snapshot. Not a general conclusion about training-set size.'}
    (config.REPORTS_DIR/'clam_universe_paired_test.json').write_text(json.dumps(paired,indent=2)+'\n')
    print(pd.DataFrame(comparison).to_string(index=False));print(json.dumps(paired,indent=2));con.close()

if __name__=='__main__':main()
