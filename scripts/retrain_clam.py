"""Retrains the quarterly CLAM with the original code in Quant_Model_Research/clam_model.py
(same tickers, architecture, hyperparameters, loss and callbacks), stopping the training
data at CLAM_RETRAIN_CUTOFF so there is a real out-of-time period to test on.

Nothing is tuned. legacy_metric=True keeps the old selection metric so the twin matches
the original method; new training uses the corrected metric by default. Output goes to
models/clam_2021_model.h5 and models/clam_2021_scaler.pkl.
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
clam_model.main("quarterly", training_end_date=config.CLAM_RETRAIN_CUTOFF, legacy_metric=True)
shutil.move("quarterly_model.h5", "clam_2021_model.h5")
shutil.move("quarterly_scaler.pkl", "clam_2021_scaler.pkl")
shutil.move("quarterly_metadata.json", "clam_2021_metadata.json")
print("saved models/clam_2021_model.h5")
