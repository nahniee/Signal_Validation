#!/bin/sh
set -eu
. scripts/gpu_env.sh
.venv/bin/python -m sv.candidates.clam_weekly cs_rank_small_n94_seed20260922
.venv/bin/python -m sv.candidates.clam_weekly cs_rank_small_n500_seed20260922
.venv/bin/python -m sv.candidates.clam_weekly cs_rank_small_n3000_seed20260922
.venv/bin/python scripts/evaluate_clam_universe.py
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/execute_workpapers.py
