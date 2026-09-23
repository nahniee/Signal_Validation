"""Build the review report, figures, notebook inputs and PDF from the revised DB."""
import sys,json,hashlib,textwrap,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import config
from sv import db,backtest
from sv.validation.calibration import evaluate
from sv.validation.gate import metrics_wide


def table(frame):
    columns=list(frame.columns)
    def cell(v):
        if isinstance(v,float): return f'{v:.3f}' if np.isfinite(v) else 'n/a'
        return str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def main():
    con=db.connect(read_only=True)
    root=config.REPORTS_DIR; root.mkdir(exist_ok=True); config.FIG_DIR.mkdir(exist_ok=True)
    pr=db.read(con,'SELECT * FROM portfolio_returns ORDER BY date')
    pr=pr[(pr.date>=pd.Timestamp(config.OOT_START)) & (pr.date<=pd.Timestamp(config.EVALUATION_END))]
    ew=pr[pr.strategy=='universe_ew'].set_index('date').ret_net
    names=config.VALIDATION_MODELS+['universe_ew','benchmark_spy']
    performance=[]
    for name in names:
        part=pr[pr.strategy==name]
        if part.empty: continue
        summary=backtest.summary(part,ew)
        performance.append({'model':name,'weeks':summary['weeks'],'CAGR':f"{summary['cagr']:.1%}",
                            'vol':f"{summary['vol']:.1%}",'Sharpe':summary['sharpe'],
                            'max DD':f"{summary['max_dd']:.1%}",'active SR':summary['info_ratio']})
    perf=pd.DataFrame(performance)
    results=db.read(con,'SELECT * FROM validation_results ORDER BY model,test')
    qrows=[]
    for m in config.VALIDATION_MODELS:
        a=results[results.model==m].set_index('test')
        b=a.loc['bootstrap_sharpe_ci']; p=a.loc['permutation_null']; d=a.loc['deflated_sharpe']
        qrows.append({'model':m,'active SR [95% CI]':f"{b.statistic:.2f} [{b.ci_low:.2f}, {b.ci_high:.2f}]",
                      'bootstrap':b.verdict,'permutation p':p.p_value,'permutation':p.verdict,
                      'DSR (provisional)':d.statistic})
    q1=pd.DataFrame(qrows)
    gates=[]
    for m in config.VALIDATION_MODELS:
        original=backtest.summary(pr[pr.strategy==m]); gated=backtest.summary(pr[pr.strategy==m+'_gated'])
        off=con.execute('SELECT AVG((NOT g.gate_on)::INT) FROM gate_decisions g JOIN portfolio_returns p ON p.strategy=g.model AND p.date=g.date WHERE g.model=? AND g.date>=? AND g.date<=?',
                        [m,config.OOT_START,config.EVALUATION_END]).fetchone()[0]
        gates.append({'model':m,'CAGR ungated / gated':f"{original['cagr']:.1%} / {gated['cagr']:.1%}",
                      'max DD ungated / gated':f"{original['max_dd']:.1%} / {gated['max_dd']:.1%}",
                      'Sharpe ungated / gated':f"{original['sharpe']:.2f} / " + (f"{gated['sharpe']:.2f}" if np.isfinite(gated['sharpe']) else 'n/a'),'off':f'{off:.1%}'})
    gate_table=pd.DataFrame(gates)
    # Report only metrics available at the cutoff, using the same lags as decisions.
    latest=pd.DataFrame({m:metrics_wide(con,m).iloc[-1] for m in config.VALIDATION_MODELS}).T
    latest.index.name='model'
    latest=latest[['psi_score','psi_input','auc_1w','rolling_sharpe','active_drawdown']].reset_index()
    calibration=evaluate(con)
    calibration.to_csv(root/'matched_calibration.csv',index=False)
    audit=json.loads((root/'clam_metric_audit.json').read_text()) if (root/'clam_metric_audit.json').exists() else None
    if audit and audit.get('observation_cutoff') != config.EVALUATION_END:
        raise ValueError('CLAM metric audit cutoff differs; rerun scripts/audit_clam_metric.py')
    registry=json.loads((config.ROOT/'experiments.json').read_text())
    trial_count=len(registry['trials'])
    available_trials=sum(bool((pr.strategy==t['model']).any()) for t in registry['trials'])
    for name,frame in [('oot_performance',perf),('oot_tests',q1),('oot_gate',gate_table),('latest_monitoring',latest)]:
        frame.to_csv(root/(name+'.csv'),index=False)

    # Figures use revised execution, costs and the common evaluation window.
    coverage=db.read(con,"SELECT EXTRACT(year FROM date)::INT AS year,COUNT(DISTINCT ticker) AS names FROM panel GROUP BY 1 ORDER BY 1")
    fig,ax=plt.subplots(figsize=(9,4)); ax.plot(coverage.year,coverage.names,marker='o')
    ax.set(title='Stocks in the 2026 snapshot with price history, by year',ylabel='Names with history',xlabel='Year')
    fig.tight_layout();fig.savefig(config.FIG_DIR/'fig1_universe_coverage.png',dpi=140);plt.close(fig)
    mon=db.read(con,'SELECT * FROM monitoring ORDER BY date')
    fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for model in ['momentum','gbm_weekly','clam_2021']:
        for ax,metric in zip(axs,['psi_score','auc_1w']):
            p=mon[(mon.model==model)&(mon.metric==metric)&(mon.date>=pd.Timestamp(config.OOT_START))]
            ax.plot(p.date,p.value,label=model);ax.set_ylabel(metric)
    axs[0].axhline(.25,color='gray',ls='--');axs[1].axhline(.5,color='gray',ls='--');axs[0].legend()
    fig.suptitle('Weekly monitoring metrics');fig.tight_layout()
    fig.savefig(config.FIG_DIR/'fig2_monitoring.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4.5))
    for ax,model in zip(axs,['momentum','gbm_weekly']):
        for name in [model,model+'_gated']:
            part=pr[pr.strategy==name];ax.plot(part.date,(1+part.ret_net).cumprod(),label=name)
        ax.set_title(model);ax.set_ylabel('Growth of 1, net');ax.legend()
    fig.suptitle('With and without the gate, out-of-time');fig.tight_layout()
    fig.savefig(config.FIG_DIR/'fig3_champion_challenger.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    for ax,group,title in [(axs[0],['gbm','gbm_expected','gbm_weekly','momentum','universe_ew'],'GBM'),
                           (axs[1],['clam_2021','clam_weekly_cs_demeaned','clam_weekly_cs_rank_small','momentum','universe_ew'],'CLAM')]:
        for name in group:
            p=pr[pr.strategy==name];ax.plot(p.date,(1+p.ret_net).cumprod(),label=name)
        ax.set_title(title);ax.legend(fontsize=8);ax.set_ylabel('Growth of 1, net')
    fig.suptitle('Out-of-time results for the frozen candidates');fig.tight_layout()
    fig.savefig(config.FIG_DIR/'fig4_round2_oot.png',dpi=140);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,8))
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    steps=[('Original quarterly code','Code, inputs, what the score means, where the weights came from'),
           ('Findings','Defects in the original implementation'),
           ('Weekly redevelopment','New target and horizon, per-ticker input windows'),
           ('Revalidation','Frozen candidates, out-of-time tests, same execution rules'),
           ('Decision','Nothing approved; new data needed before resubmitting')]
    for i,(title,body) in enumerate(steps):
        y=.9-i*.19
        ax.text(.5,y,title+'\n'+body,ha='center',va='center',fontsize=11,
                bbox=dict(boxstyle='round,pad=.8',facecolor='#edf3fa',edgecolor='#34577c'))
        if i<4:
            ax.annotate('',xy=(.5,y-.135),xytext=(.5,y-.055),arrowprops=dict(arrowstyle='->',lw=1.8,color='#34577c'))
    fig.tight_layout();fig.savefig(config.FIG_DIR/'fig5_review_process.png',dpi=150);plt.close(fig)

    audit_text=('The 2021 twin was scored on '+str(audit['windows'])+' single-stock out-of-time windows ('+
                str(audit['daily_close_targets'])+' daily close targets). On the same predictions, the old scaled-sign accuracy was '+
                f"{audit['legacy_scaled_sign_accuracy']:.2%}"+' and the corrected return-direction accuracy was '+
                f"{audit['inverse_scaled_return_direction_accuracy']:.2%}." if audit else
                "The metric audit hasn't been run, so there is no corrected accuracy to report.")
    universe_section=""
    comparison_path=root/'clam_universe_comparison.csv'
    if comparison_path.exists():
        u=pd.read_csv(comparison_path)
        paired=json.loads((root/'clam_universe_paired_test.json').read_text())
        display=u[['requested_tickers','usable_tickers','training_windows','epochs','validation_rank_ic','oot_cagr','oot_active_sharpe','bootstrap','permutation_p']].copy()
        display['oot_cagr']=display.oot_cagr.map(lambda x:f'{x:.1%}')
        fig,ax=plt.subplots(figsize=(12,5))
        for name in ['clam_weekly_cs_rank_small']+u.model.tolist()+['momentum','universe_ew']:
            part=pr[pr.strategy==name]
            label=name.replace('clam_weekly_cs_rank_small','CLAM rank').replace('_seed20260922','')
            ax.plot(part.date,(1+part.ret_net).cumprod(),label=label)
        ax.set(title='Training-universe experiment: OOT net outcomes',ylabel='Growth of 1');ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(config.FIG_DIR/'fig6_training_universe.png',dpi=140);plt.close(fig)
        def verdict(c):
            lo, hi = c['ci_95']
            return 'worse' if hi < 0 else 'better' if lo > 0 else 'not distinguishable'
        by_size = {int(c['contrast'].split()[0]): c for c in paired['contrasts']}
        outcome = ' '.join(
            f"The {n}-stock run is {verdict(c)} than the 500-stock run on this sample."
            if verdict(c) != 'not distinguishable' else
            f"The {n}-stock and 500-stock runs can't be told apart on this sample."
            for n, c in sorted(by_size.items()))
        universe_section=("### 7.1 Training-universe size: 94, 500 and 3000 stocks\n\n"
            "The original CLAM was trained on 94 hand-picked stocks, mostly because training took a long time. "
            "To check whether that held it back, I trained the small weekly rank-target model three times and changed "
            "only the number of stocks used in training: 94, 500 and 3000, picked by market cap. All three runs use "
            "seed 20260922, the same callbacks and the same purged validation split, and none of them was chosen by "
            "looking at out-of-time results. Training targets end before 2019-12-31 and selection targets by 2021-12-31. "
            "Fewer stocks are usable than requested because newer listings have no training history. A smaller "
            "universe is also an easier ranking problem, so validation IC can't be compared directly across the three.\n\n"
            +table(display)+"\n\n![Training-universe comparison](figures/fig6_training_universe.png)\n\n"
            +"Mean annual net-return difference against the new 500-stock run, with 95% bootstrap intervals: "
            +"; ".join(f"{n} stocks {c['annualized_mean_return_difference']:.2%} "
                       f"[{c['ci_95'][0]:.2%}, {c['ci_95'][1]:.2%}]" for n, c in sorted(by_size.items()))
            +". These are differences in mean return, which isn't the same as a CAGR difference. "+outcome
            +" So there's no sign that the original 94-stock list was what held CLAM back. This is one seed on a period "
            +"I had already studied, and changing the universe also changes the peer group each stock is ranked "
            +"against, so it doesn't settle how training-set size matters in general. The older 500-stock model is "
            +"shown for reference; the new 500-stock run is the like-for-like comparison.\n")

    report=f'''# Model validation review: GBM and CLAM

Revised 2026-09-22. I built these models and reviewed them myself, using the structure a bank's second-line model validation team would follow. That makes this a self-review. It is not independent and has no regulatory standing.

## 1. Decision

**No candidate is approved for production.** Momentum stays in the review as a yardstick for the others.

The review ran in two stages. First I went through the original quarterly models and how they would be used in a weekly strategy. Then I rebuilt both models for a weekly horizon and reviewed the new versions on their own. Fixing a coding defect and changing the forecast horizon are different kinds of change, so the weekly results say nothing about whether a correctly built quarterly model would work.

The figures here replace the ones in earlier drafts. Those drafts ran the tests over the whole sample, used a shortened holding period in the lag check and scored CLAM with a faulty direction metric, so none of their numbers should be used. Testing now covers {config.OOT_START} to {config.EVALUATION_END}: {len(ew)} complete weekly holding periods, all after the training and selection cutoff. I had already looked at this period in earlier rounds, which makes it out-of-time but not a clean holdout.

## 2. How the review ran

![Review process](figures/fig5_review_process.png)

| stage | what it covered | outcome |
| --- | --- | --- |
| Original quarterly code | The code, how inputs are built, what the score means, where the weights came from | GBM ranks stocks on one random path; CLAM mixes tickers inside training windows; the CLAM direction metric was wrong; the original weights' training cutoff can't be verified |
| 2021 twin | The original CLAM method retrained on data up to 2021-12-31 | `clam_2021`, which keeps the original defects so it stays true to the original method |
| Weekly redevelopment | A weekly target and horizon, and per-ticker input windows | `gbm_weekly` and `clam_weekly_*`, reviewed as new models |
| Revalidation | The same execution rules, out-of-time tests, monitoring and gate for every candidate | Sections 5 to 7; nothing approved |
| Next review | A frozen specification and new evidence | Data collected after the freeze, a point-in-time universe, a full experiment log, real execution data |

The strategy under test holds the 50 highest-scored stocks each week, long only and equally weighted. It isn't a quarterly buy-and-hold backtest; section 7 checks the quarterly forecasts against quarterly targets separately. Risk limits, capacity and live trading were out of scope.

## 3. Candidates and how closely they match the originals

| candidate | what it is | notes |
| --- | --- | --- |
| `gbm` | The original score: the mean of one simulated 63-step price path (T = 0.25) | Two calendar years of history (at least 454 returns), adjusted closes up to and including the signal day, one fixed seed per run. Same formula as the original, though it can't replay the original's live downloads or random draws |
| `gbm_expected` | The expected value of that score, in closed form | Ranks stocks exactly as their mean log return does. Used as a diagnostic |
| `clam_orig` | The original quarterly weights | The .h5 and scaler files aren't in this checkout and their training cutoff is unknown. A file timestamp can't show which dates were used in training |
| `clam_2021` | The original CLAM method, retrained to 2021-12-31 | Weights unchanged since training. It keeps the original window and metric defects on purpose |
| `gbm_weekly` | `gbm_w_126_expected`, picked in the earlier development round | All six lookback and score variants are logged. I didn't re-pick it after changing the execution rules |
| `clam_weekly_*` | Weekly CLAM with per-ticker windows and a single-number target | The universe-size runs were trained separately. Demeaned and rank scores aren't return forecasts |
| `momentum` | The standard 12-1 momentum rule | Same data and execution rules as the others |

The original GBM uses 63 steps; earlier drafts said 65 by mistake. The two quarterly models also score different things: GBM averages a simulated price path, and CLAM adds up predicted daily log changes in the High price. Reproducing the GBM formula checks the code, but the model's economic assumptions still need their own justification. How accurate a forecast is and how well it picks stocks are separate questions.

## 4. Data and backtest rules

Prices run from 2013-01-02 to {config.EVALUATION_END}. Anything later in the raw files is ignored. The stock list is a snapshot of the 3,000 largest US companies on 2026-09-15, so it only holds firms that survived to that date and were picked using later market caps. Comparing against an equal-weighted portfolio of the same stocks doesn't remove that bias, and I can't tell how much it helps or hurts each strategy. Downloading the data again would give a different dataset.

Each signal uses the week's last close. The trade fills at the next session's close and is held until the next session's close after the following signal. All dates come from SPY's trading calendar, and a missing quote is never filled with a later price. Same-close results (`*_same_close`) are kept as an optimistic comparison only.

No price after {config.EVALUATION_END} is used anywhere, including signals, returns, calibration and the CLAM audit. Holding periods and targets that end after that date are dropped, and a missing return for a held stock stops the run with an error. Every candidate has the same cutoff.

A stock is tradable on a signal date if it closes at $5 or more and has traded at least $5M a day on average over the last 20 sessions. The top 50 tradable scores get equal weight. Trading costs 10 bps of the absolute weight traded, including drift in existing positions, and the equal-weight benchmark pays the same costs. Filling at the close with no market impact is a simplification. Cash earns nothing, and active Sharpe uses the strategy's return minus the equal-weight benchmark.

## 5. Q1: is the out-of-time record better than chance?

The stationary bootstrap uses 5,000 draws with a mean block of 13 weeks, and a model passes if the 95% interval sits above zero. The permutation test reshuffles which stocks are held within each week's scored set 500 times, compares gross active returns, and reports p = (1 + exceedances) / 501. It is conditional on this snapshot and doesn't keep sector or exposure structure. Passing either test on its own wouldn't show that the model can make money in practice.

{table(q1)}

The Deflated Sharpe Ratio is reported but doesn't decide anything yet. `experiments.json` lists at least {trial_count} trials, including the three training-universe runs, and counts the `gbm_weekly` alias once. {available_trials} of them have returns; the raw-target and weekly-bar CLAM runs left no usable scores. The DSR takes its variance from those {available_trials} Sharpe ratios and uses N = {trial_count}. Searches I didn't log and correlation between trials would both raise the hurdle, so the trial count can only go up from here.

## 6. Out-of-time portfolio results and the gate

{table(perf)}

![Revised OOT performance](figures/fig4_round2_oot.png)

These figures describe the tested versions on this dataset. Failing the tests doesn't prove a model has no skill, and four CLAM variants can't settle whether price-based sequence models work in general. For approval, a model would have to add value over the benchmark on new data, and a high CAGR alone doesn't show that.

{table(gate_table)}

![Gate comparison](figures/fig3_champion_challenger.png)

The gate moves a strategy to cash for the week if score PSI is above 0.25, trailing active Sharpe is below zero, AUC is below 0.50, or relative drawdown is worse than -15%. It also switches off if any of those metrics is missing. Portfolio metrics lag two signal weeks, since with next-day execution the previous holding period hasn't finished by the current signal. Costs come from the positions the gated strategy actually holds, and relative drawdown compares strategy wealth with benchmark wealth.

The thresholds are common conventions and weren't fitted to this data, and I can't show they were set before I looked at the out-of-time period. Changing them would need a fresh evaluation. The gate is an experiment and shouldn't be the only risk control.

## 7. Monitoring, targets and the CLAM metric audit

{table(latest)}

![Monitoring](figures/fig2_monitoring.png)

PSI buckets are open at both tails so drifted scores still get counted. The reference period is the first 104 signal weeks, and PSI starts after it. Weekly AUC uses only finished targets, and the 65-session diagnostics use each target's actual maturity date. AUC here is pooled across all stocks and dates, which isn't the same as how well the top 50 do.

The calibration table pairs each raw-return score with its own target and horizon, using out-of-time observations that matured by {config.EVALUATION_END}. The coefficients are descriptive and come without significance tests. Regressing a weekly score on a 13-week return only shows association.

{table(calibration[['model','horizon_sessions','n','slope','intercept','mae']])}

The targets are: for GBM, the average adjusted price from t to t+63 relative to t; for weekly GBM, adjusted Close[t+5] / Close[t] - 1; for the original CLAM, raw High[t+65] / High[t] - 1. The rank and demeaned weekly CLAM scores can't be mapped back to returns without the exact training cross-section, so they aren't calibrated. Overlapping targets and correlation between stocks make all of these hard to read.

{audit_text}

That fixed alphabetical sample tests the metric itself. It doesn't reproduce the 76% and 64% training and validation figures from the original run. MinMax scaling to (-1, 1) doesn't send a zero return to zero, so the corrected metric compares scaled values against the scaled value of zero. New training of the original model uses the corrected metric for early stopping. The existing weights weren't retrained, and the 2021 twin script asks for the old metric on purpose so it matches the original. The cross-ticker windows are a separate, confirmed defect. An earlier draft said the network had learned same-day market direction; I've dropped that claim.

{universe_section}
## 8. Findings

| ID | finding | what was done | status |
| --- | --- | --- | --- |
| F1 | Scoring on one simulated path adds noise to the ranking | Added the closed-form expectation for comparison and revised the tables | Original not approved. The fix on its own doesn't show an edge |
| F2 | The expected GBM score ranks stocks the same way as mean log return | Stated in the model description and compared with momentum | Needs an economic reason to be a separate model |
| F3 | The original CLAM training windows mix tickers | Found in code review; the weekly version uses per-ticker windows | Open in the original. The fixed version is a different model |
| F4 | No way to verify what data the original weights were trained on | Dropped the timestamp-based cutoff; the files aren't here; the twin's cutoff is recorded | Blocks any verdict on the original weights |
| F5 | Same-close fills, and the lag check used a shorter holding period | Next-close to next-close execution, drift-aware costs, calendar tests | Fixed in the backtest; no live execution data yet |
| F6 | Zero, negative and missing prices | Input filters; held-stock returns never filled with zero; one common end date | Problems in the source data remain |
| F7 | My permutation test was biased by trading costs | Now compares gross returns, with a finite-sample p-value | Fixed; the test's conditional null is noted |
| F8 | Earlier tests included development-period data | Q1 now runs on out-of-time data only | Fixed, though that period has been looked at before |
| F9 | Earlier drafts generalised from a few CLAM runs to the whole approach | Conclusions now cover only what was tested | Claim withdrawn |
| F10 | Not every experiment was counted for multiple testing | A log of at least {trial_count} trials, {available_trials} with returns; DSR marked provisional | Open until the search history is complete |
| F11 | The direction metric took signs after scaling | Corrected the threshold, re-scored the same predictions, fixed the training monitor | Metric fixed; old weights and early-stopping choices unchanged |
| F12 | Calibration compared forecasts and targets over different horizons | Matched targets, left rank scores out, renamed the cross-horizon slope | Fixed; nothing approved on calibration |
| F13 | Earlier drafts overstated independence and how exact the reproductions were | Stated the self-review role, listed the differences, recorded file hashes | Wording fixed |

## 9. Decision and what a resubmission needs

None of the GBM or CLAM versions is approved for production. `clam_orig` also lacks the files and training cutoff it would need to be reviewed at all. Momentum is only a comparison. None of this rules out quarterly models in general, or sequence models as a family.

A resubmission would need a fixed use and horizon, a documented history of the training data and model, a complete experiment log, a point-in-time stock universe or a bounded estimate of the survivorship bias, realistic execution, cost and capacity tests, matched-target diagnostics, and new data collected after the specification is frozen. Trials already run keep counting. A quarterly-only study would be its own project, with quarterly targets and quarterly holding rules.

Ongoing monitoring would check PSI, AUC, relative drawdown and active Sharpe every week. A breach or a missing metric pauses the strategy until someone investigates. Any material change to the model, the data or the execution means revalidating. This report doesn't clear any model for production.

## 10. Reproducing the results

`sh scripts/revalidate.sh` rebuilds everything from the saved database and scores: the panel, the GBM reproduction, portfolio returns, the out-of-time tests, monitoring, the gate, this report and its figures. It doesn't re-pick the weekly models. `scripts/audit_clam_metric.py` rebuilds the metric audit and needs TensorFlow. `tests/test_validation_contracts.py` covers timing, missing data, drift, PSI tails, out-of-time filtering and the metric threshold.

The notebooks read the same database, and the CSVs next to this report hold its tables. `review_manifest.json` records file hashes and the evaluation settings, and the PDF is generated from this Markdown. No new prices were downloaded and nothing was deployed. The historical CLAM weights haven't changed; the 94, 500 and 3000-stock runs are new models trained for section 7.1.

The structure follows the conceptual-soundness, monitoring and outcome-analysis parts of the Federal Reserve's [SR 11-7 guidance](https://www.federalreserve.gov/supervisionreg/srletters/sr1107a1.pdf). I used it as a template; the report makes no claim of regulatory compliance.
'''
    (root/'validation_report.md').write_text(report)
    manifest={'evaluation_start':config.OOT_START,'evaluation_end':config.EVALUATION_END,'cutoff_kind':'inclusive observation cutoff','last_included_signal':str(ew.index.max().date()),'execution':'next-session-close to next-session-close',
              'trial_count_lower_bound':trial_count,'available_trial_returns':available_trials,'oot_reused':True,'original_weight_cutoff':None,
              'historical_weights_unchanged':True,'new_training_runs':['n94_seed20260922','n500_seed20260922','n3000_seed20260922'],'gbm_weekly_alias':'gbm_w_126_expected','sha256':{}}
    for p in (list((config.ROOT/'sv').rglob('*.py')) + list((config.ROOT/'scripts').glob('*.py'))
              + list((config.ROOT/'sql').rglob('*.sql')) + [config.ROOT/'config.py',config.ROOT/'experiments.json',
              config.QMR_DIR/'clam_model.py',config.QMR_DIR/'clam_weekly.py']
              + list(config.MODELS_DIR.glob('*')) + list(config.QMR_DIR.glob('clam_weekly/*/*.keras'))):
        manifest['sha256'][str(p.relative_to(config.ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'review_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    render_pdf(report,root/'validation_report.pdf')
    print(perf.to_string(index=False)); print('Rebuilt report, PDF, figures, CSVs and manifest.')


def render_pdf(markdown,path):
    """Typeset headings, paragraphs, wrapped tables and inline figures from Markdown."""
    from matplotlib.patches import Rectangle
    width,height=11.7,8.3
    left,right,top,bottom=.55,11.15,7.7,.5
    available=right-left
    page=None; y=top; number=0
    def clean(value):
        return re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',value).replace('**','').replace('`','')
    with PdfPages(path) as pdf:
        def finish():
            if page is not None:
                page.text(right/width,.025,str(number),ha='right',fontsize=8,color='#667085')
                pdf.savefig(page);plt.close(page)
        def new_page():
            nonlocal page,y,number
            finish();page=plt.figure(figsize=(width,height));y=top;number+=1
        def ensure(size):
            if page is None or y-size<bottom: new_page()
        def text_block(value,size=10,bold=False,space=.1):
            nonlocal y
            lines=textwrap.wrap(clean(value),width=int(available*72/(size*.51)),break_long_words=False) or ['']
            for line in lines:
                ensure(size/72*1.35+space)
                page.text(left/width,y/height,line,va='top',fontsize=size,
                          weight='bold' if bold else 'normal',family='DejaVu Sans',color='#17212f')
                y-=size/72*1.35
            y-=space
        def table_block(raw):
            nonlocal y
            rows=[[clean(v.strip()) for v in line.strip().strip('|').split('|')] for line in raw]
            rows=[row for row in rows if not all(re.fullmatch(r'[:\- ]+',v) for v in row)]
            count=len(rows[0]); fontsize=8
            lengths=[min(65,max(len(row[j]) for row in rows)) for j in range(count)]
            weights=np.sqrt(np.maximum(lengths,8))
            widths=weights/weights.sum()*available
            def draw(row,header=False):
                nonlocal y
                wrapped=[textwrap.wrap(value,width=max(9,int(w*72/(fontsize*.54))),break_long_words=True) or [''] for value,w in zip(row,widths)]
                h=max(map(len,wrapped))*fontsize/72*1.3+.14
                ensure(h)
                x=left
                for values,w in zip(wrapped,widths):
                    page.patches.append(Rectangle((x/width,(y-h)/height),w/width,h/height,
                                        transform=page.transFigure,facecolor='#eaf0f7' if header else '#ffffff',
                                        edgecolor='#ccd5e0',linewidth=.5))
                    page.text((x+.04)/width,(y-.05)/height,'\n'.join(values),va='top',fontsize=fontsize,
                              linespacing=1.3,weight='bold' if header else 'normal',color='#17212f')
                    x+=w
                y-=h
            draw(rows[0],True)
            for row in rows[1:]:
                estimate=max(len(textwrap.wrap(v,width=max(9,int(w*72/(fontsize*.54))))) or 1 for v,w in zip(row,widths))*fontsize/72*1.3+.14
                if y-estimate<bottom:
                    new_page();draw(rows[0],True)
                draw(row)
            y-=.16
        lines=markdown.splitlines();i=0
        while i<len(lines):
            line=lines[i].strip()
            if not line: i+=1;continue
            if line.startswith('|'):
                chunk=[]
                while i<len(lines) and lines[i].strip().startswith('|'):
                    chunk.append(lines[i]);i+=1
                table_block(chunk);continue
            if line.startswith('!['):
                match=re.search(r'\]\(([^)]+)\)',line)
                if match:
                    picture=plt.imread(config.REPORTS_DIR/match.group(1));ratio=picture.shape[1]/picture.shape[0]
                    h=min(5.9,available/ratio);w=h*ratio
                    ensure(h+.15)
                    ax=page.add_axes(((left+(available-w)/2)/width,(y-h)/height,w/width,h/height))
                    ax.imshow(picture);ax.axis('off');y-=h+.2
            elif line.startswith('# '):
                text_block(line[2:],19,True,.2)
            elif line.startswith('## '):
                ensure(.9);text_block(line[3:],13,True,.14)
            else:
                text_block(line)
            i+=1
        finish()

if __name__=='__main__': main()
