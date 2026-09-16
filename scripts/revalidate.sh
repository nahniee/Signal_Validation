#!/bin/sh
# Round-2 revalidation after the redevelopment of GBM (weekly) and CLAM (weekly).
# Run from the project root after scripts/develop_gbm_weekly.py and clam_weekly scoring.
set -e
M="momentum gbm gbm_expected gbm_weekly clam_weekly_cs_demeaned clam_weekly_cs_rank_small"
.venv/bin/python -m sv.backtest $M
.venv/bin/python -m sv.validation.overfit $M
.venv/bin/python -m sv.validation.monitoring $M
.venv/bin/python -m sv.validation.gate $M
