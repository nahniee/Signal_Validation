#!/bin/sh
# Revalidate existing frozen scores. No OOT-driven model reselection or retraining.
set -eu
.venv/bin/python -m sv.features
.venv/bin/python scripts/run_candidates.py
M="momentum gbm gbm_expected gbm_weekly clam_2021 clam_weekly_cs_demeaned clam_weekly_cs_rank_small clam_weekly_cs_rank_small_n500_seed20260922 clam_weekly_cs_rank_small_n3000_seed20260922"
TRIALS="gbm_w_63_expected gbm_w_63_prob_up gbm_w_126_expected gbm_w_126_prob_up gbm_w_252_expected gbm_w_252_prob_up"
.venv/bin/python -m sv.backtest $M $TRIALS
.venv/bin/python -m sv.validation.overfit $M
.venv/bin/python -m sv.validation.monitoring $M
.venv/bin/python -m sv.validation.gate $M
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/execute_workpapers.py
