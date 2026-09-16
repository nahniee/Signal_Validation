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
| `clam_2021` | `scripts/retrain_clam.py` | same training code with the window cut at 2021-12-31, to obtain an honest out-of-time period |
| `momentum` | textbook 12-1 momentum | untuned benchmark rule: a complex model that cannot beat it is not approved |

## Layout

```
config.py                 every modelling assumption in one place (cited by the report)
sql/schema.sql            DuckDB tables: universe -> prices -> panel -> signals -> portfolio_returns -> monitoring / gate_decisions / validation_results
sql/queries/*.sql         set-based work lives in SQL: rebalance panel, realised vol, PSI, AUC via ranks, drawdown-based performance summary
sv/                       Python library: ingest, features, backtest, candidates/, validation/
scripts/                  long-running batch jobs (candidate scoring, CLAM retrain, GPU env)
notebooks/00..04          validation workpapers: data quality, Q1, Q2, Q3, findings
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

## Data caveats (carried into the report)

* **Survivorship bias** — the universe is today's listings; delisted names are absent. Quantified in
  `notebooks/00_data_quality.ipynb`; all pass/fail decisions use active return vs the same universe.
* **Liquidity** — price ≥ $5 and 20-day ADV ≥ $5M at every rebalance; the filtered set is the only
  approved scope of use.
* **Costs** — 10 bps one-way on traded weight; a one-day execution lag variant (`*_lag1`) is kept
  as a sensitivity.
