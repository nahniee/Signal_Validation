# Signal Validation

A self-validation case study of my earlier GBM and CLAM models, using the structure of a second-line model review. Developer and reviewer are the same person; this is not organizationally independent validation.

**Review story:** original quarterly implementations and their proposed weekly use → code/provenance findings → separately versioned weekly redevelopments → revalidation. The evidence does not establish that a correctly implemented quarterly strategy cannot work. A new horizon is a change of intended use, not merely a bug fix.

## Current evidence and decision

**Evaluation end: 2026-08-28.** This is the last allowed price observation. Unfinished holding periods are excluded; archived later prices are not used.

See [the revised report](reports/validation_report.md), [PDF](reports/validation_report.pdf), and [workpapers](notebooks). No model is approved for production. Momentum is a comparator only. Conclusions are limited to tested specifications, the survivor-selected snapshot and the reused evaluation period.

- Q1: OOT-only stationary bootstrap and gross-return permutation test, 2022-01-03 through 2026-08-28.
- DSR: provisional sensitivity with a minimum of 17 documented trials, including the 94/500/3000 training runs; two historical CLAM return series remain missing. See [experiment registry](experiments.json). Never reduce historical trial counts to gain approval.
- Execution: signal at week-end close, trade at next-session close and hold to the following execution close; drift-aware traded-weight costs for candidates and the equal-weight comparator.
- Q2: matured-target monitoring, PSI including tails, and target-matched descriptive calibration. Rank/de-meaned CLAM outputs are not calibrated raw returns.
- Q3: lagged controls and actual gated-holdings costs; missing required metrics switch off. The gate is experimental.
- CLAM: corrected original-return direction metric, same-prediction frozen-weight audit; historical weights preserved.

The OOT window has already been examined in earlier work. It is a post-training historical evaluation, not a pristine holdout. Same-universe active returns do not remove survivorship bias.

## Revalidate the existing snapshot

From this directory, with the existing DuckDB database, saved scores and virtual environment:

```bash
.venv/bin/python -m unittest discover -s tests -v
sh scripts/revalidate.sh
. scripts/gpu_env.sh
.venv/bin/python scripts/audit_clam_metric.py
.venv/bin/python scripts/build_report.py
```

The first script preserves weekly candidate choices and CLAM weights. It rebuilds the panel, controlled 63-step GBM reproduction, backtests, OOT tests, monitoring, gates and report. Do not run another writer or reader from a separate process against DuckDB during a writing job. The original `clam_orig` artifacts are absent; their training cutoff is unverified. `clam_2021` artifacts exist locally. Missing discarded CLAM runs are disclosed, not silently treated as recovered.

Generated evidence: `reports/oot_*.csv`, `latest_monitoring.csv`, `matched_calibration.csv`, `clam_metric_audit.json`, `review_manifest.json`, Markdown/PDF report and six figures. Workpapers 00–05 read these revised tables. Tests exercise the changed timing/accounting/statistical contracts.

## New data or model development

Install `requirements.txt` in a Python 3.12 environment if needed. `sv.universe`, `sv.ingest`, and `sv.features` build a **new** live-data snapshot; a new download is not exact reproduction of the existing frozen one. Preserve the original snapshot and its provenance for comparisons.

Original-model training defaults to the corrected direction metric; `scripts/retrain_clam.py` explicitly retains the legacy metric for an original-methodology twin. The historical cross-ticker sequence defect remains in that original training path. Weekly per-ticker redevelopment is in `../Quant_Model_Research/clam_weekly.py`. Any retraining, changed selection or new variant requires a new artifact/version, an appended experiment record and untouched evaluation data. `scripts/develop_gbm_weekly.py` is a development tool, not part of the frozen revalidation command.

## Remaining limitations

Today's 3,000-name market-cap snapshot omits historical failures and uses future membership information, including in CLAM's training-universe choice. Fixed 10 bps costs and next-close fills are still assumptions; capacity and market impact are unvalidated. The evaluation ends on 2026-08-28: later prices are excluded and only holding periods completed by that date are evaluated. Historical search records and original-weight provenance are incomplete. None of these limitations is resolved by a favorable Sharpe or a failed null test.

## Training-universe experiment

Three separately saved runs share the small weekly rank-target architecture, seed
20260922 and a purged validation split, and differ only in training-universe size:
94, 500 and 3000 names by market-cap rank. The 94 rung matches the size of the original
hand-picked training list, which was chosen under compute constraints. Only data through
2021-12-31 enter training and model selection, and the historical weights are preserved.

```bash
. scripts/gpu_env.sh
.venv/bin/python scripts/train_clam_universe.py 94
.venv/bin/python scripts/train_clam_universe.py 500
.venv/bin/python scripts/train_clam_universe.py 3000
sh scripts/run_clam_universe_evaluation.sh
```

The trainer refuses to overwrite a completed run. Results are in
`reports/clam_universe_comparison.csv` and `clam_universe_paired_test.json`, and every run
counts toward the experiment registry regardless of outcome. Expanding to 3000 names is
worse than the 500-name control, while 94 and 500 are not distinguishable on this sample,
so the original 94-name list is not evidence of a training-size constraint. A smaller
universe is also a smaller ranking peer group, so validation rank IC is not comparable
across rungs on equal terms. One seed and a reused evaluation period cannot establish a
general size effect.
