"""Audit frozen clam_2021 on sparse OOT single-stock windows; do not retrain.
Legacy and inverse-scaled direction use the identical predictions and actual 65-day
Close targets. This is an OOT diagnostic, not a reconstruction of training accuracy.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import config
from sv import db
from sv.candidates.clam import load, feature_matrix, windows_for_date

con = db.connect(read_only=True)
model, scaler = load('clam_2021')
X, dates, tickers = feature_matrix(con)
# Fixed alphabetical subset and 65-session spacing; no performance-based selection.
wanted = set(sorted(tickers)[:100])
positions = [i for i,d in enumerate(dates) if pd.Timestamp(config.OOT_START) <= d <= pd.Timestamp(config.EVALUATION_END)]
legacy = corrected = total = windows = 0
for t in positions[::65]:
    if t + 65 >= len(dates) or dates[t + 65] > pd.Timestamp(config.EVALUATION_END):
        continue
    W, names = windows_for_date(X, dates, tickers, dates[t], wanted)
    if W is None:
        continue
    inds = tickers.get_indexer(names)
    actual = np.transpose(X[t+1:t+66, inds, :], (1,0,2))
    valid = np.isfinite(actual).all(axis=(1,2))
    W, actual = W[valid], actual[valid]
    if not len(W):
        continue
    z = scaler.transform(W.reshape(-1,5)).reshape(W.shape)
    pred_scaled = model.predict(z, batch_size=64, verbose=0)
    true_scaled = scaler.transform(actual.reshape(-1,5)).reshape(actual.shape)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1,5)).reshape(actual.shape)
    legacy += np.equal(np.sign(pred_scaled[:,:,3]),np.sign(true_scaled[:,:,3])).sum()
    corrected += np.equal(np.sign(pred[:,:,3]),np.sign(actual[:,:,3])).sum()
    total += actual.shape[0]*65; windows += len(actual)
res = {'model':'clam_2021','sample':'first 100 alphabetical tickers, OOT starts every 65 sessions, complete targets only',
       'observation_cutoff':config.EVALUATION_END,'windows':windows,'daily_close_targets':total,'legacy_scaled_sign_accuracy':float(legacy/total),
       'inverse_scaled_return_direction_accuracy':float(corrected/total),'scaled_zero_close':float(scaler.min_[3]),
       'model_sha256':hashlib.sha256(config.MODELS_DIR.joinpath('clam_2021_model.h5').read_bytes()).hexdigest(),
       'limitation':'Frozen historical weights; not the old training/validation set; no causal attribution or skill test.'}
(config.REPORTS_DIR/'clam_metric_audit.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res,indent=2))
