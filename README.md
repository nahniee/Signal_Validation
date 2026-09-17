# Signal Validation

Independent, second-line validation of three deployed trading signals — the kind of review a bank's
Model Risk Management team performs before a front-office model is approved for use.

The models under review are my own earlier work
([Quant_Model_Research](https://github.com/nahniee/Quant_Model_Research),
[Long_Term_Trading](https://github.com/nahniee/Long_Term_Trading)). This repository does **not**
modify or tune them; it re-runs them as deployed on a 3,000-stock universe and asks the three
questions a risk team asks:

| # | question | module | tests |
|---|---|---|---|
| Q1 | Is the backtest real or overfit? | `sv/validation/overfit.py` | block-bootstrap Sharpe CI · Deflated Sharpe Ratio · cross-sectional permutation null |
| Q2 | Is the signal still valid today? | `sv/validation/monitoring.py` | rolling PSI · KS · AUC · calibration slope |
| Q3 | When must it be switched off? | `sv/validation/gate.py` | rule-based kill switch, champion vs challenger out-of-time |

Final deliverable: an SR 11-7-style validation report (`reports/validation_report.md`) with
findings, limitations, monitoring plan and prohibited-use conditions.

## Candidates

| model | origin | role |
|---|---|---|
| `gbm` | `Long_Term_Trading/stats_model_process.py` | deployed statistical model — single Monte-Carlo path score, reproduced verbatim |
| `gbm_expected` | derived here | same model's closed-form expectation, to isolate the Monte-Carlo noise |
| `clam_orig` | `Quant_Model_Research/quarterly_model.h5` | deployed CNN-LSTM-Attention model, weights as shipped |
| `clam_2021` | `scripts/retrain_clam.py` | same training code, 94-ticker universe and architecture, window cut at 2021-12-31 — the deployed methodology given an honest out-of-time period |
| `momentum` | textbook 12-1 momentum | untuned benchmark rule: a complex model that cannot beat it is not approved |
| `gbm_weekly` | `Long_Term_Trading/gbm_weekly.py` (round 2) | redeveloped GBM: 5-day horizon, closed-form score; best of 6 variants on the development sample |
| `clam_weekly_*` | `Quant_Model_Research/clam_weekly.py` (round 2) | redeveloped CLAM: per-ticker sequences, training cut 2021-12-31, weekly scalar target; two variants retained |

**Outcome so far:** two validation rounds, nine candidate variants, none approved. Round 2 showed the GBM
defect was an implementation error (the fixed model passes the noise tests but does not beat the benchmark)
and that the CLAM methodology has no skill either as deployed (retrained with an honest cut-off) or after its data-construction defect is repaired.
Details in `reports/validation_report.md` §12 and `notebooks/05_round2.ipynb`.

## Layout

```
config.py                 every modelling assumption in one place (cited by the report)
sql/schema.sql            DuckDB tables: universe -> prices -> panel -> signals -> portfolio_returns -> monitoring / gate_decisions / validation_results
sql/queries/*.sql         set-based work lives in SQL: rebalance panel, realised vol, PSI, AUC via ranks, drawdown-based performance summary
sv/                       Python library: ingest, features, backtest, candidates/, validation/
scripts/                  long-running batch jobs (candidate scoring, CLAM retrain, GPU env)
notebooks/00..05          validation workpapers: data quality, Q1, Q2, Q3, findings, round-2 revalidation
reports/                  validation report + figures
```

## Reproduce

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m sv.universe            # Yahoo screener snapshot (top-3000 by market cap)
.venv/bin/python -m sv.ingest              # daily OHLCV 2013 -> today into DuckDB (~3 min)
.venv/bin/python -m sv.features            # weekly rebalance panel (SQL window functions)
.venv/bin/python scripts/run_candidates.py # momentum + GBM scores
.venv/bin/python -m sv.backtest            # shared top-50 backtest, benchmarks
.venv/bin/python -m sv.validation.overfit
.venv/bin/python -m sv.validation.monitoring
.venv/bin/python -m sv.validation.gate
```

CLAM needs the original artefacts next to this repo (`../Quant_Model_Research/quarterly_model.h5`,
`quarterly_scaler.pkl`, `clam_model.py`, `clam_workflow.py`) and TensorFlow with a GPU:

```bash
. scripts/gpu_env.sh
.venv/bin/python -m sv.candidates.clam clam_orig
.venv/bin/python scripts/retrain_clam.py && .venv/bin/python -m sv.candidates.clam clam_2021
```

Then re-run backtest / overfit / monitoring / gate with the CLAM model names as arguments.

Round 2 (redeveloped weekly models):

```bash
.venv/bin/python scripts/develop_gbm_weekly.py                 # 6 GBM-weekly variants, picks one on the dev sample
(cd ../Quant_Model_Research && ../Signal_Validation/.venv/bin/python clam_weekly.py cs_demeaned)
(cd ../Quant_Model_Research && ../Signal_Validation/.venv/bin/python clam_weekly.py cs_rank small)
.venv/bin/python -m sv.candidates.clam_weekly cs_demeaned
.venv/bin/python -m sv.candidates.clam_weekly cs_rank_small
sh scripts/revalidate.sh
```

DuckDB allows one writing process at a time: run the CLAM trainer (which reads the database) before or
after, not during, a scoring job.

## Data caveats (carried into the report)

* **Survivorship bias** — the universe is today's listings; delisted names are absent. Quantified in
  `notebooks/00_data_quality.ipynb`; all pass/fail decisions use active return vs the same universe.
* **Liquidity** — price ≥ $5 and 20-day ADV ≥ $5M at every rebalance; the filtered set is the only
  approved scope of use.
* **Costs** — 10 bps one-way on traded weight; a one-day execution lag variant (`*_lag1`) is kept
  as a sensitivity.
