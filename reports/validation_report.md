# Model validation review - GBM and CLAM

**Revised:** 2026-09-22. **Role:** self-validation project adopting a second-line review structure; developer and reviewer are the same person. This is not organizationally independent validation or a regulatory approval.

## 1. Executive decision

**No candidate is approved for production.** Momentum is retained as a comparison rule, not an approved trading model. These are project decisions under the evidence and limitations below, not proof that the model families have no predictive value.

The story has two distinct stages: review the original quarterly forecasters and their proposed weekly use; then assess separately versioned weekly redevelopments. Repairing an implementation defect and changing the prediction horizon are different interventions. This project does **not** establish that a properly implemented quarterly strategy cannot work.

The revised OOT table below supersedes earlier report numbers. In particular, whole-sample significance, shortened lagged holding periods and the old CLAM direction metric must not be used as approval evidence. The evaluation ends on **2026-08-28**. Q1 uses 241 complete weekly holding periods after 2022-01-03, with entry and exit prices both observed by that cutoff. This is after the recorded training/selection cutoff, but the period has already been inspected in earlier rounds and is not an untouched holdout.

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

The price observation window is **2013-01-02 through 2026-08-28**. Later prices retained in the raw archive are excluded from this evaluation. The existing universe snapshot is as of 2026-09-15, so its later membership remains a disclosed source of bias. Re-running a live screener/download is a new dataset, not exact reproduction. The universe contains survivors and was selected using later market capitalizations. Subtracting an equal-weight comparator does **not** cancel this bias; strategy-specific effects remain unknown. Listing coverage describes the sample and does not measure the size of return bias.

Signals use week-end close information. Orders are assumed filled at the **next session close**, and held until the following rebalance's next-session close. A common SPY trading calendar determines all dates. Missing ticker quotes are not replaced by the next available future quote. Same-close trades are retained only as optimistic diagnostics (`*_same_close`), with no claim they were executable.

**2026-08-28 is the final observation date.** No later price is used for signals, portfolio returns, calibration or the CLAM metric audit. Weekly holding periods and forecast targets that are not complete by the cutoff are excluded. Missing held-name returns within completed periods raise an error. The same fixed cutoff applies to every candidate.

Tradability at signal time: price >= $5, trailing 20-session average dollar volume >= $5M. Top 50 available scores are equally weighted. Costs are 10 bps per unit of absolute traded weight, including drift in previous holdings; the universe comparator also pays rebalancing costs. This remains a simplified close-fill assumption without capacity/market-impact evidence. Cash earns zero. Sharpe uses zero cash return; active SR is strategy minus equal-weight comparator.

## 5. Q1 - evidence after training/selection

Stationary bootstrap uses 5,000 draws and mean block 13 weeks; positive 95% CI is the bootstrap screen. The 500-draw permutation comparison uses gross active returns and p=(1+exceedances)/(501). It tests exchangeability of names within each scored cross-section, conditional on this biased snapshot; it does not preserve all sector, exposure or serial score structure. Neither test alone is proof of deployable skill.

| model | active SR [95% CI] | bootstrap | permutation p | permutation | DSR (provisional) |
| --- | --- | --- | --- | --- | --- |
| momentum | 0.90 [0.26, 1.53] | PASS | 0.020 | PASS | 0.390 |
| gbm | 0.06 [-0.80, 0.86] | FAIL | 0.084 | FAIL | 0.019 |
| gbm_expected | 0.39 [-0.31, 1.12] | FAIL | 0.164 | FAIL | 0.086 |
| gbm_weekly | 0.50 [-0.12, 1.15] | FAIL | 0.106 | FAIL | 0.129 |
| clam_2021 | 0.36 [-0.37, 1.05] | FAIL | 0.136 | FAIL | 0.075 |
| clam_weekly_cs_demeaned | -0.89 [-1.56, -0.19] | FAIL | 0.838 | FAIL | 0.000 |
| clam_weekly_cs_rank_small | -0.09 [-0.82, 0.65] | FAIL | 0.323 | FAIL | 0.008 |
| clam_weekly_cs_rank_small_n94_seed20260922 | 0.09 [-0.83, 1.01] | FAIL | 0.307 | FAIL | 0.022 |
| clam_weekly_cs_rank_small_n500_seed20260922 | -0.04 [-0.80, 0.75] | FAIL | 0.267 | FAIL | 0.011 |
| clam_weekly_cs_rank_small_n3000_seed20260922 | -0.99 [-1.65, -0.35] | FAIL | 0.902 | FAIL | 0.000 |

DSR is **provisional**, not PASS/FAIL approval evidence. `experiments.json` records a minimum of 17 trials: historical baseline/development runs plus the two predeclared training-universe experiments. The `gbm_weekly` alias is not counted twice. 15 trial return series are available: raw-target and collapsed weekly-bar CLAM runs lack recoverable scores. The variance estimate uses the 15 available trial Sharpes on the same OOT window and N=17; unknown prior searches and correlation between trials remain limitations. The historical search count must never be reduced to obtain approval.

## 6. OOT portfolio outcomes and controls

| model | weeks | CAGR | vol | Sharpe | max DD | active SR |
| --- | --- | --- | --- | --- | --- | --- |
| momentum | 241 | 37.8% | 41.1% | 0.987 | -28.6% | 0.898 |
| gbm | 241 | 10.5% | 31.5% | 0.474 | -37.1% | 0.065 |
| gbm_expected | 241 | 18.4% | 41.3% | 0.614 | -34.5% | 0.391 |
| gbm_weekly | 241 | 22.9% | 41.2% | 0.706 | -39.6% | 0.501 |
| clam_2021 | 241 | 17.2% | 34.2% | 0.635 | -32.4% | 0.360 |
| clam_weekly_cs_demeaned | 241 | -4.1% | 29.7% | 0.010 | -50.6% | -0.895 |
| clam_weekly_cs_rank_small | 241 | 10.3% | 24.7% | 0.522 | -27.8% | -0.089 |
| clam_weekly_cs_rank_small_n94_seed20260922 | 241 | 9.2% | 37.4% | 0.423 | -49.8% | 0.087 |
| clam_weekly_cs_rank_small_n500_seed20260922 | 241 | 11.0% | 24.2% | 0.552 | -32.8% | -0.040 |
| clam_weekly_cs_rank_small_n3000_seed20260922 | 241 | -0.3% | 15.3% | 0.058 | -25.5% | -0.988 |
| universe_ew | 241 | 12.6% | 19.9% | 0.696 | -21.3% | n/a |
| benchmark_spy | 241 | 12.7% | 16.7% | 0.802 | -21.8% | -0.054 |

![Revised OOT performance](figures/fig4_round2_oot.png)

Point estimates describe the tested implementations and this snapshot. Failure to reject a null does not establish absence of skill. The four CLAM specifications do not establish that OHLCV or sequence models in general cannot work. Benchmark-relative approval requires fresh evidence of incremental value, not only a favorable absolute CAGR.

| model | CAGR ungated / gated | max DD ungated / gated | Sharpe ungated / gated | off |
| --- | --- | --- | --- | --- |
| momentum | 37.8% / 26.8% | -28.6% / -28.2% | 0.99 / 0.91 | 33.6% |
| gbm | 10.5% / 0.0% | -37.1% / 0.0% | 0.47 / n/a | 100.0% |
| gbm_expected | 18.4% / 3.5% | -34.5% / -21.5% | 0.61 / 0.30 | 85.9% |
| gbm_weekly | 22.9% / 1.9% | -39.6% / -36.3% | 0.71 / 0.21 | 61.0% |
| clam_2021 | 17.2% / 0.0% | -32.4% / 0.0% | 0.63 / n/a | 100.0% |
| clam_weekly_cs_demeaned | -4.1% / 0.0% | -50.6% / 0.0% | 0.01 / n/a | 100.0% |
| clam_weekly_cs_rank_small | 10.3% / 0.0% | -27.8% / 0.0% | 0.52 / n/a | 100.0% |
| clam_weekly_cs_rank_small_n94_seed20260922 | 9.2% / 0.0% | -49.8% / 0.0% | 0.42 / n/a | 100.0% |
| clam_weekly_cs_rank_small_n500_seed20260922 | 11.0% / 0.0% | -32.8% / 0.0% | 0.55 / n/a | 100.0% |
| clam_weekly_cs_rank_small_n3000_seed20260922 | -0.3% / 0.0% | -25.5% / 0.0% | 0.06 / n/a | 100.0% |

![Gate comparison](figures/fig3_champion_challenger.png)

The gate uses PSI >0.25, trailing active SR <0, AUC <0.50 or relative drawdown <-15%. Missing required metrics switch the strategy off. Portfolio metrics are delayed two signal weeks because the preceding delayed-execution holding period has not ended at the current signal close. Entry/exit/rebalance costs are recomputed from actual gated holdings, not added to hypothetical ungated turnover. Relative drawdown uses the ratio of strategy and benchmark wealth.

Thresholds are documented conventions, not empirically established universal cutoffs. The historical record does not establish prospective preregistration. Reuse of the OOT period and any further changes to thresholds require fresh evaluation. The gate remains an experimental control, not the sole approved risk control.

## 7. Monitoring, targets and CLAM metric audit

| model | psi_score | psi_input | auc_1w | rolling_sharpe | active_drawdown |
| --- | --- | --- | --- | --- | --- |
| momentum | 0.236 | 0.614 | 0.508 | 0.903 | -0.217 |
| gbm | 0.129 | 0.614 | 0.502 | 1.361 | -0.292 |
| gbm_expected | 0.183 | 0.614 | 0.505 | -0.004 | -0.316 |
| gbm_weekly | 0.179 | 0.614 | 0.507 | 0.826 | -0.257 |
| clam_2021 | 0.080 | 0.614 | 0.507 | 1.247 | -0.299 |
| clam_weekly_cs_demeaned | 0.148 | 0.614 | 0.498 | -0.911 | -0.702 |
| clam_weekly_cs_rank_small | 0.125 | 0.614 | 0.502 | 1.419 | -0.329 |
| clam_weekly_cs_rank_small_n94_seed20260922 | 0.407 | 0.614 | 0.505 | 0.964 | -0.268 |
| clam_weekly_cs_rank_small_n500_seed20260922 | 0.133 | 0.614 | 0.508 | 0.897 | -0.270 |
| clam_weekly_cs_rank_small_n3000_seed20260922 | 0.092 | 0.614 | 0.509 | -1.739 | -0.643 |

![Monitoring](figures/fig2_monitoring.png)

PSI bins include infinite tails, so drifted observations are counted. The reference is the first 104 signal weeks; PSI is not reported before that reference exists. Next-execution-period AUC uses only completed targets; 65-session diagnostics use actual target maturity dates. AUC is pooled across stocks/dates and is not a top-50 portfolio skill test.

Calibration below pairs each available raw-return score with its own target and horizon, uses OOT observations matured by 2026-08-28, and reports descriptive pooled coefficients without significance claims. In particular a weekly score's slope against a 13-week return is only an association, never evidence of calibration.

| model | horizon_sessions | n | slope | intercept | mae |
| --- | --- | --- | --- | --- | --- |
| gbm | 63 | 470585 | -0.004 | 0.018 | 0.137 |
| gbm_expected | 63 | 470585 | -0.074 | 0.019 | 0.093 |
| gbm_weekly | 5 | 518297 | -0.041 | 0.003 | 0.044 |
| clam_2021 | 65 | 477325 | 0.002 | 0.032 | 0.368 |

GBM targets the mean of adjusted prices from t through t+63 relative to t; weekly GBM targets adjusted Close[t+5]/Close[t]-1; original CLAM targets raw High[t+65]/High[t]-1. Rank/de-meaned weekly CLAM targets lack the exact training cross-section mapping required for comparable return calibration, so no calibration claim is made for them. Overlapping outcomes and stock dependence limit interpretation.

The frozen 2021 twin was evaluated on 1478 single-stock OOT windows (96070 daily Close targets). On the SAME predictions, legacy scaled-sign accuracy was 59.08% and inverse-scaled return-direction accuracy was 49.79%.

This fixed alphabetical sample is a diagnostic of the metric, not a reconstruction of the historical 76%/64% training/validation result. MinMax scaling to (-1,1) does not generally map a zero return to zero; the corrected metric compares scaled values against the scaler's image of zero. New original-model training defaults to this corrected metric and early-stopping monitor. Existing weights were not retrained. The historical twin script explicitly requests the legacy metric for fidelity. Cross-ticker sequences remain a separate confirmed defect; the claim that the network specifically learned same-day market direction is withdrawn.

### 7.1 Training-universe size: matched 94 / 500 / 3000 experiment

The historical 94-name training list reflected compute constraints, so this experiment varies only training-universe size on the frozen small weekly rank-target architecture. All three runs share seed 20260922, callbacks and a purged validation boundary, and select tickers by market-cap rank; none was selected using OOT outcomes. Training targets end before 2019-12-31, model-selection targets by 2021-12-31. Requested size differs from usable tickers because later listings lack training history. A smaller universe also means a smaller ranking peer group, so validation rank IC is not comparable across rungs on equal terms.

| requested_tickers | usable_tickers | training_windows | epochs | validation_rank_ic | oot_cagr | oot_active_sharpe | bootstrap | permutation_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 94 | 89 | 27869 | 8 | 0.067 | 9.2% | 0.087 | FAIL | 0.307 |
| 500 | 476 | 142185 | 7 | 0.024 | 11.0% | -0.040 | FAIL | 0.267 |
| 3000 | 2567 | 659736 | 9 | 0.016 | -0.3% | -0.988 | FAIL | 0.902 |

![Training-universe comparison](figures/fig6_training_universe.png)

Paired annualized mean net-return differences against the matched 500-name control: 94 2.48% [-14.56%, 21.78%]; 3000 -12.46% [-23.06%, -1.20%]. These are mean-return differences, not CAGR differences. Single-seed evidence on a reused historical period cannot establish a general size effect. The original 500-name artifact is a historical comparator; the newly trained 500-name run is the matched control. A smaller training universe also changes the rank-target peer group, so size and task difficulty are not separated here. Existing production non-approval remains: fresh holdout, universe-bias and operational evidence are still required.

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
| F10 | Incomplete multiple-testing accounting | Minimum 17-trial registry, 15 available returns, provisional DSR | Open until search history/evidence complete |
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
