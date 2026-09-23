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
    ax.set(title='Historical coverage of the 2026 snapshot (not a bias estimate)',ylabel='Names with history',xlabel='Year')
    fig.tight_layout();fig.savefig(config.FIG_DIR/'fig1_universe_coverage.png',dpi=140);plt.close(fig)
    mon=db.read(con,'SELECT * FROM monitoring ORDER BY date')
    fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for model in ['momentum','gbm_weekly','clam_2021']:
        for ax,metric in zip(axs,['psi_score','auc_1w']):
            p=mon[(mon.model==model)&(mon.metric==metric)&(mon.date>=pd.Timestamp(config.OOT_START))]
            ax.plot(p.date,p.value,label=model);ax.set_ylabel(metric)
    axs[0].axhline(.25,color='gray',ls='--');axs[1].axhline(.5,color='gray',ls='--');axs[0].legend()
    fig.suptitle('Monitoring with matured targets and complete PSI tails');fig.tight_layout()
    fig.savefig(config.FIG_DIR/'fig2_monitoring.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4.5))
    for ax,model in zip(axs,['momentum','gbm_weekly']):
        for name in [model,model+'_gated']:
            part=pr[pr.strategy==name];ax.plot(part.date,(1+part.ret_net).cumprod(),label=name)
        ax.set_title(model);ax.set_ylabel('Growth of 1, net');ax.legend()
    fig.suptitle('Ungated / gated: next-session-close execution, OOT');fig.tight_layout()
    fig.savefig(config.FIG_DIR/'fig3_champion_challenger.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    for ax,group,title in [(axs[0],['gbm','gbm_expected','gbm_weekly','momentum','universe_ew'],'GBM'),
                           (axs[1],['clam_2021','clam_weekly_cs_demeaned','clam_weekly_cs_rank_small','momentum','universe_ew'],'CLAM')]:
        for name in group:
            p=pr[pr.strategy==name];ax.plot(p.date,(1+p.ret_net).cumprod(),label=name)
        ax.set_title(title);ax.legend(fontsize=8);ax.set_ylabel('Growth of 1, net')
    fig.suptitle('Frozen candidates, revised OOT execution (not an untouched holdout)');fig.tight_layout()
    fig.savefig(config.FIG_DIR/'fig4_round2_oot.png',dpi=140);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,8))
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    steps=[('Original quarterly implementation','Review code, inputs, score meaning and provenance'),
           ('Findings','Identify defects; do not infer failure of quarterly models'),
           ('Separately versioned weekly redevelopment','Changed target / horizon and repaired sequences'),
           ('Revalidation','Frozen candidates, post-training tests, matched targets and execution'),
           ('Decision','No production approval; obtain fresh evidence before resubmission')]
    for i,(title,body) in enumerate(steps):
        y=.9-i*.19
        ax.text(.5,y,title+'\n'+body,ha='center',va='center',fontsize=11,
                bbox=dict(boxstyle='round,pad=.8',facecolor='#edf3fa',edgecolor='#34577c'))
        if i<4:
            ax.annotate('',xy=(.5,y-.135),xytext=(.5,y-.055),arrowprops=dict(arrowstyle='->',lw=1.8,color='#34577c'))
    fig.tight_layout();fig.savefig(config.FIG_DIR/'fig5_review_process.png',dpi=150);plt.close(fig)

    audit_text=('The frozen 2021 twin was evaluated on '+str(audit['windows'])+' single-stock OOT windows ('+
                str(audit['daily_close_targets'])+' daily Close targets). On the SAME predictions, legacy scaled-sign accuracy was '+
                f"{audit['legacy_scaled_sign_accuracy']:.2%}"+' and inverse-scaled return-direction accuracy was '+
                f"{audit['inverse_scaled_return_direction_accuracy']:.2%}." if audit else
                'Metric audit has not been run; no corrected accuracy is claimed.')
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
        universe_section=("### 7.1 Training-universe size: matched 94 / 500 / 3000 experiment\n\n"
            "The historical 94-name training list reflected compute constraints, so this experiment varies only "
            "training-universe size on the frozen small weekly rank-target architecture. All three runs share seed "
            "20260922, callbacks and a purged validation boundary, and select tickers by market-cap rank; none was "
            "selected using OOT outcomes. Training targets end before 2019-12-31, model-selection targets by "
            "2021-12-31. Requested size differs from usable tickers because later listings lack training history. "
            "A smaller universe also means a smaller ranking peer group, so validation rank IC is not comparable "
            "across rungs on equal terms.\n\n"
            +table(display)+"\n\n![Training-universe comparison](figures/fig6_training_universe.png)\n\n"
            +"Paired annualized mean net-return differences against the matched 500-name control: "
            +"; ".join(f"{c['contrast'].split(' minus')[0]} {c['annualized_mean_return_difference']:.2%} "
                       f"[{c['ci_95'][0]:.2%}, {c['ci_95'][1]:.2%}]" for c in paired['contrasts'])
            +". These are mean-return differences, not CAGR differences. Single-seed evidence on a reused historical "
            +"period cannot establish a general size effect. The original 500-name artifact is a historical comparator; "
            +"the newly trained 500-name run is the matched control. "
            +"A smaller training universe also changes the rank-target peer group, so size and task difficulty "
            +"are not separated here. Existing production non-approval remains: fresh holdout, universe-bias and "
            +"operational evidence are still required.\n")

    report=f'''# Model validation review - GBM and CLAM

**Revised:** 2026-09-22. **Role:** self-validation project adopting a second-line review structure; developer and reviewer are the same person. This is not organizationally independent validation or a regulatory approval.

## 1. Executive decision

**No candidate is approved for production.** Momentum is retained as a comparison rule, not an approved trading model. These are project decisions under the evidence and limitations below, not proof that the model families have no predictive value.

The story has two distinct stages: review the original quarterly forecasters and their proposed weekly use; then assess separately versioned weekly redevelopments. Repairing an implementation defect and changing the prediction horizon are different interventions. This project does **not** establish that a properly implemented quarterly strategy cannot work.

The revised OOT table below supersedes earlier report numbers. In particular, whole-sample significance, shortened lagged holding periods and the old CLAM direction metric must not be used as approval evidence. The evaluation ends on **{config.EVALUATION_END}**. Q1 uses {len(ew)} complete weekly holding periods after {config.OOT_START}, with entry and exit prices both observed by that cutoff. This is after the recorded training/selection cutoff, but the period has already been inspected in earlier rounds and is not an untouched holdout.

## 2. Review trail and scope

![Review process](figures/fig5_review_process.png)

| stage | purpose | evidence / consequence |
| --- | --- | --- |
| Original quarterly implementation | Examine original code, input construction, score meaning and provenance | GBM single-path noise; CLAM cross-ticker windows; incorrect direction metric; original-weight training cutoff unverified |
| Original-methodology twin | Freeze training at 2021-12-31 to examine later observations | `clam_2021`; retains original sequence and historical metric defects deliberately |
| Weekly redevelopment | Change intended prediction target/horizon and repair input construction | Separate `gbm_weekly` / `clam_weekly_*` model versions, not a successful quarterly validation |
| Revalidation | Apply shared execution, OOT tests, monitoring and controls | Results below; no production approval |
| Next approval review | Freeze specifications and seek new evidence | Untouched future data, historical universe data, complete experiments and operational execution evidence |

The portfolio application here is weekly top-50 equal-weight long-only selection. It is not a native quarterly-holding backtest. Original-horizon target-matched diagnostics are reported separately in section 7. Risk limits, capacity and live execution are not validated.

## 3. Model identity and reproduction differences

| candidate | identity | fidelity / limitation |
| --- | --- | --- |
| `gbm` | Original 63-step, T=0.25, single-path mean-price score | Two-calendar-year rolling window, at least 454 returns; adjusted panel close; includes current signal close; deterministic full-run seed. Controlled reproduction, not verbatim replay of live downloads/RNG |
| `gbm_expected` | Closed-form mean-path expectation of that implemented formula | Monotone in the fitted mean log return; analytical diagnostic, not a separately trained model |
| `clam_orig` | Historical quarterly weights | Original h5/scaler absent in this checkout; cutoff unverified. File mtime cannot establish training membership |
| `clam_2021` | Saved original-methodology twin, cutoff 2021-12-31 | Existing weights unchanged; original sequence and metric defects retained for historical comparison |
| `gbm_weekly` | Frozen alias `gbm_w_126_expected`, selected in the earlier development exercise | Six lookback/score variants logged. Alias was not reselected after this review changed execution |
| `clam_weekly_*` | Saved per-ticker weekly scalar-target redevelopments | Historical weights unchanged; the universe-size pair is separately trained. Demeaned-return and rank scores are not raw predicted returns |
| `momentum` | Fixed 12-1 ranking rule | Comparator under the same snapshot/execution assumptions; no production approval |

A 63-step original GBM must not be described as 65 steps. GBM and CLAM quarterly outputs also have different meanings: mean simulated path versus terminal cumulative High log changes. The GBM formula uses the source's mean-log-return drift convention; reproducing that formula is not validation of its economic assumptions. Original forecast error and portfolio selection performance answer different questions.

## 4. Data and executable backtest contract

The price observation window is **2013-01-02 through {config.EVALUATION_END}**. Later prices retained in the raw archive are excluded from this evaluation. The existing universe snapshot is as of 2026-09-15, so its later membership remains a disclosed source of bias. Re-running a live screener/download is a new dataset, not exact reproduction. The universe contains survivors and was selected using later market capitalizations. Subtracting an equal-weight comparator does **not** cancel this bias; strategy-specific effects remain unknown. Listing coverage describes the sample and does not measure the size of return bias.

Signals use week-end close information. Orders are assumed filled at the **next session close**, and held until the following rebalance's next-session close. A common SPY trading calendar determines all dates. Missing ticker quotes are not replaced by the next available future quote. Same-close trades are retained only as optimistic diagnostics (`*_same_close`), with no claim they were executable.

**{config.EVALUATION_END} is the final observation date.** No later price is used for signals, portfolio returns, calibration or the CLAM metric audit. Weekly holding periods and forecast targets that are not complete by the cutoff are excluded. Missing held-name returns within completed periods raise an error. The same fixed cutoff applies to every candidate.

Tradability at signal time: price >= $5, trailing 20-session average dollar volume >= $5M. Top 50 available scores are equally weighted. Costs are 10 bps per unit of absolute traded weight, including drift in previous holdings; the universe comparator also pays rebalancing costs. This remains a simplified close-fill assumption without capacity/market-impact evidence. Cash earns zero. Sharpe uses zero cash return; active SR is strategy minus equal-weight comparator.

## 5. Q1 - evidence after training/selection

Stationary bootstrap uses 5,000 draws and mean block 13 weeks; positive 95% CI is the bootstrap screen. The 500-draw permutation comparison uses gross active returns and p=(1+exceedances)/(501). It tests exchangeability of names within each scored cross-section, conditional on this biased snapshot; it does not preserve all sector, exposure or serial score structure. Neither test alone is proof of deployable skill.

{table(q1)}

DSR is **provisional**, not PASS/FAIL approval evidence. `experiments.json` records a minimum of {trial_count} trials: historical baseline/development runs plus the two predeclared training-universe experiments. The `gbm_weekly` alias is not counted twice. {available_trials} trial return series are available: raw-target and collapsed weekly-bar CLAM runs lack recoverable scores. The variance estimate uses the {available_trials} available trial Sharpes on the same OOT window and N={trial_count}; unknown prior searches and correlation between trials remain limitations. The historical search count must never be reduced to obtain approval.

## 6. OOT portfolio outcomes and controls

{table(perf)}

![Revised OOT performance](figures/fig4_round2_oot.png)

Point estimates describe the tested implementations and this snapshot. Failure to reject a null does not establish absence of skill. The four CLAM specifications do not establish that OHLCV or sequence models in general cannot work. Benchmark-relative approval requires fresh evidence of incremental value, not only a favorable absolute CAGR.

{table(gate_table)}

![Gate comparison](figures/fig3_champion_challenger.png)

The gate uses PSI >0.25, trailing active SR <0, AUC <0.50 or relative drawdown <-15%. Missing required metrics switch the strategy off. Portfolio metrics are delayed two signal weeks because the preceding delayed-execution holding period has not ended at the current signal close. Entry/exit/rebalance costs are recomputed from actual gated holdings, not added to hypothetical ungated turnover. Relative drawdown uses the ratio of strategy and benchmark wealth.

Thresholds are documented conventions, not empirically established universal cutoffs. The historical record does not establish prospective preregistration. Reuse of the OOT period and any further changes to thresholds require fresh evaluation. The gate remains an experimental control, not the sole approved risk control.

## 7. Monitoring, targets and CLAM metric audit

{table(latest)}

![Monitoring](figures/fig2_monitoring.png)

PSI bins include infinite tails, so drifted observations are counted. The reference is the first 104 signal weeks; PSI is not reported before that reference exists. Next-execution-period AUC uses only completed targets; 65-session diagnostics use actual target maturity dates. AUC is pooled across stocks/dates and is not a top-50 portfolio skill test.

Calibration below pairs each available raw-return score with its own target and horizon, uses OOT observations matured by {config.EVALUATION_END}, and reports descriptive pooled coefficients without significance claims. In particular a weekly score's slope against a 13-week return is only an association, never evidence of calibration.

{table(calibration[['model','horizon_sessions','n','slope','intercept','mae']])}

GBM targets the mean of adjusted prices from t through t+63 relative to t; weekly GBM targets adjusted Close[t+5]/Close[t]-1; original CLAM targets raw High[t+65]/High[t]-1. Rank/de-meaned weekly CLAM targets lack the exact training cross-section mapping required for comparable return calibration, so no calibration claim is made for them. Overlapping outcomes and stock dependence limit interpretation.

{audit_text}

This fixed alphabetical sample is a diagnostic of the metric, not a reconstruction of the historical 76%/64% training/validation result. MinMax scaling to (-1,1) does not generally map a zero return to zero; the corrected metric compares scaled values against the scaler's image of zero. New original-model training defaults to this corrected metric and early-stopping monitor. Existing weights were not retrained. The historical twin script explicitly requests the legacy metric for fidelity. Cross-ticker sequences remain a separate confirmed defect; the claim that the network specifically learned same-day market direction is withdrawn.

{universe_section}
## 8. Findings, remediation and closure evidence

| ID | finding / cause | remediation / evidence | status |
| --- | --- | --- | --- |
| F1 | Single MC path adds ranking noise | Analytical expectation comparator; revised OOT tables | Original implementation not approved; correction does not prove alpha |
| F2 | Expected GBM rank is monotone in mean log return | Explicit model identity and benchmark comparison | Documented; economic rationale still required |
| F3 | Original CLAM mixes tickers within windows | Source review; weekly redevelopment uses per-ticker sequences | Original defect open; repaired structure is a separate model |
| F4 | Original weights' training provenance unverified | mtime inference removed; original artifacts absent; twin cutoff separately recorded | Blocking for original-weight outcome validation |
| F5 | Same-close execution and shortened lag sensitivity | Next-close to next-close contract, drift-aware costs, calendar regression tests | Revised implementation; live execution evidence outstanding |
| F6 | Nonpositive/missing prices | Positive input filters; no zero-filling held-name returns; common sample end | Source data limitations remain |
| F7 | Validator's cost-biased permutation comparison | Gross-return comparison, finite-sample p correction | Corrected; conditional-null limitations disclosed |
| F8 | Full-sample tests included development observations | OOT-only Q1 in code and regenerated report | Corrected; reused OOT is not pristine |
| F9 | Limited CLAM experiments generalized to a whole approach | Conclusions limited to tested specifications/data/period | Overstatement withdrawn |
| F10 | Incomplete multiple-testing accounting | Minimum {trial_count}-trial registry, {available_trials} available returns, provisional DSR | Open until search history/evidence complete |
| F11 | Direction metric used signs after scaling | Zero-threshold correction, same-prediction audit, corrected new-training monitor | Metric fixed; legacy weights/early selection unchanged |
| F12 | Cross-horizon calibration interpretation | Target-matched diagnostics; ranks excluded; association relabeled | Corrected; no calibration-based approval |
| F13 | Independent-review / exact-reproduction overclaims | Self-review role, model differences and artifact hashes recorded | Disclosure corrected |

## 9. Decisions and resubmission conditions

All GBM/CLAM production uses remain unapproved in this project. `clam_orig` additionally lacks original artifact/cutoff evidence. Momentum remains only a comparator. No quarterly model family is rejected in general, and no recommendation to abandon all sequence models is supported.

Resubmission requires: frozen intended use and horizon; verified training/data/model provenance; complete experiment registry; point-in-time universe or explicitly bounded bias evidence; realistic execution/cost/capacity validation; matched-target diagnostics; and new untouched observations after specification freeze. Trial counts carry forward. A quarterly-only study would be a different scope requiring quarterly prediction targets and portfolio holding rules, not a rewrite of these weekly-use results.

Weekly monitoring reviews PSI, discrimination, relative drawdown and active SR; a breach or missing evidence suspends the experimental strategy and triggers investigation. Revalidation is required after material model/data/execution changes. No monitored model is currently authorized for production by this report.

## 10. Reproduction and evidence

Run `sh scripts/revalidate.sh` against the existing frozen DB and saved scores. It rebuilds the panel, controlled GBM reproduction, portfolio returns, OOT tests, monitoring, gates, report and figures without reselecting weekly models. Run `scripts/audit_clam_metric.py` with TensorFlow separately to rebuild the metric audit. `tests/test_validation_contracts.py` covers timing, missing data, drift, PSI tails, OOT filtering and metric threshold semantics.

The updated notebook workpapers query this same database. CSVs beside this report expose the tables; `review_manifest.json` records code/artifact hashes and the evaluation contract. The PDF is generated from this report. No fresh price download or production deployment was performed. Historical CLAM weights are preserved; the separately registered 500/3000-ticker runs are new trained artifacts.

The review structure is inspired by conceptual-soundness review, ongoing monitoring and outcome analysis in the historical [SR 11-7 guidance](https://www.federalreserve.gov/supervisionreg/srletters/sr1107a1.pdf). This is a methodological reference, not a claim of current regulatory compliance.
'''
    (root/'validation_report.md').write_text(report)
    manifest={'evaluation_start':config.OOT_START,'evaluation_end':config.EVALUATION_END,'cutoff_kind':'inclusive observation cutoff','last_included_signal':str(ew.index.max().date()),'execution':'next-session-close to next-session-close',
              'trial_count_lower_bound':trial_count,'available_trial_returns':available_trials,'oot_reused':True,'original_weight_cutoff':None,
              'historical_weights_unchanged':True,'new_training_runs':['n500_seed20260922','n3000_seed20260922'],'gbm_weekly_alias':'gbm_w_126_expected','sha256':{}}
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
