# Signal Validation

I built a GBM ranking model and a CNN-LSTM forecaster (CLAM) for a weekly trading bot, then went back and reviewed them the way a bank's model risk team would. The developer and the reviewer are the same person here, so this is a self-review that follows a second-line structure. It isn't independent validation.

Related repos: [Quant_Model_Research](https://github.com/nahniee/Quant_Model_Research) (CLAM) and [Long_Term_Trading](https://github.com/nahniee/Long_Term_Trading) (GBM and the bot).

## What happened

The review had two rounds. In the first I read through the original quarterly models and tested them inside a weekly strategy. That turned up coding defects in both. In the second I rebuilt them for a weekly horizon and tested the new versions against the same rules. Changing the horizon changes what the model is for, so the weekly results don't tell us whether a correctly built quarterly model would have worked.

**No model is approved for production.** Momentum stays in as a benchmark. The full write-up is in the [report](reports/validation_report.md) ([PDF](reports/validation_report.pdf)), and the working notebooks are in [notebooks](notebooks).

Testing uses the period from 2022-01-03 to 2026-08-28. Every model was trained and selected on data before 2022, but I had already looked at this period in earlier rounds, so it counts as out-of-time but isn't a clean holdout. Prices after 2026-08-28 aren't used, and holding periods that hadn't finished by then are dropped.

The review asks three questions of each model. The first is whether its out-of-time record beats chance, tested with a stationary block bootstrap and a permutation test. The Deflated Sharpe Ratio is reported too, but only as a provisional number, because two early CLAM runs left no scores; [experiments.json](experiments.json) logs at least 17 trials. The second is whether the signal is still working, tracked with weekly PSI, AUC and KS on targets that have finished, plus calibration against each model's own target (the rank and demeaned CLAM scores aren't return forecasts, so they're left out). The third is when to switch it off, handled by an experimental rule-based gate that uses lagged metrics, recalculates costs from the positions actually held and switches off when a metric is missing.

Signals are taken at the week's last close and traded at the next session's close, then held to the next execution close. Costs include drift in existing positions, and the benchmark pays them too.

The stock list is a snapshot of today's 3,000 largest US companies, so companies that failed along the way are missing. Comparing against an equal-weighted portfolio of the same stocks doesn't remove that bias.

## Rerunning the review

From this directory, with the existing DuckDB database, saved scores and virtual environment:

```bash
.venv/bin/python -m unittest discover -s tests -v
sh scripts/revalidate.sh
. scripts/gpu_env.sh
.venv/bin/python scripts/audit_clam_metric.py
.venv/bin/python scripts/build_report.py
```

`revalidate.sh` rebuilds the panel, the GBM reproduction, the backtests, the out-of-time tests, monitoring, the gate and the report. It keeps the weekly model choices and CLAM weights as they are. DuckDB allows one writer at a time, so don't open the database from another process while it runs. The original `clam_orig` weights aren't in this repo and their training cutoff is unknown; the `clam_2021` retrain is available locally.

Outputs: `reports/oot_*.csv`, `latest_monitoring.csv`, `matched_calibration.csv`, `clam_metric_audit.json`, `review_manifest.json`, the report and its figures. Notebooks 00 to 05 read these tables, and the tests check the timing, accounting and statistical rules.

## New data or new models

Install `requirements.txt` in Python 3.12. `sv.universe`, `sv.ingest` and `sv.features` download and build a new snapshot, which will differ from the one used here, so keep the original if you want to compare.

Training the original CLAM now uses the corrected direction metric by default. `scripts/retrain_clam.py` asks for the old metric on purpose, because it rebuilds the original method as it was, cross-ticker windows included. The weekly redevelopment lives in `../Quant_Model_Research/clam_weekly.py`. Any retrain, re-selection or new variant should get its own version, a new line in the experiment log and evaluation data it hasn't seen. `scripts/develop_gbm_weekly.py` is for development and isn't part of the frozen review.

## Limitations

The 3,000-stock snapshot leaves out companies that failed and uses later information to decide which stocks are in, including for CLAM's training set. Costs are a flat 10 bps and fills happen at the close; capacity and market impact weren't tested. The search history and the original weights' provenance are incomplete. A good Sharpe ratio or a failed null test doesn't fix any of these.

## Training-universe experiment

The original CLAM was trained on 94 hand-picked stocks, mostly to keep training time down. To see whether that held it back, I trained the small weekly rank-target model three times with 94, 500 and 3000 stocks picked by market cap. Everything else was kept the same: seed 20260922, the architecture and a purged validation split. Only data up to 2021-12-31 was used for training and selection, and the historical weights weren't touched.

```bash
. scripts/gpu_env.sh
.venv/bin/python scripts/train_clam_universe.py 94
.venv/bin/python scripts/train_clam_universe.py 500
.venv/bin/python scripts/train_clam_universe.py 3000
sh scripts/run_clam_universe_evaluation.sh
```

The trainer won't overwrite a finished run. Results are in `reports/clam_universe_comparison.csv` and `clam_universe_paired_test.json`, and all three runs count in the experiment log whatever the outcome.

Going to 3000 stocks made things worse than the 500-stock run. The 94- and 500-stock runs couldn't be told apart, so there's no evidence the original 94-stock list was the problem. With fewer stocks the ranking task is also easier, so validation IC can't be compared directly across the three. It's one seed on a period I'd already looked at, so it doesn't settle how training-set size matters in general.
