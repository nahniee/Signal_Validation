"""Retrain the CLAM 'quarterly' model with the ORIGINAL code path
(Quant_Model_Research/clam_model.py: same tickers, architecture, hyper-parameters,
loss, callbacks) but with the training window truncated at CLAM_RETRAIN_CUTOFF.

This is not a new model - it is the deployed methodology re-run on an earlier
information set so that an honest out-of-time window exists. Nothing is tuned.
Artifacts land in models/clam_2021_model.h5 / models/clam_2021_scaler.pkl.
"""
import os, sys, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
sys.path.insert(0, str(config.QMR_DIR))
import clam_model  # noqa: E402  (original training code, unmodified)

out = ROOT / "models"
out.mkdir(exist_ok=True)
os.chdir(out)                       # clam_model.main saves to cwd
clam_model.main("quarterly", training_end_date=config.CLAM_RETRAIN_CUTOFF)
shutil.move("quarterly_model.h5", "clam_2021_model.h5")
shutil.move("quarterly_scaler.pkl", "clam_2021_scaler.pkl")
print("saved models/clam_2021_model.h5")
