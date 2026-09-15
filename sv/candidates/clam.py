"""Candidate 2: CLAM (CNN+LSTM+Attention) 65-day forecast, scored exactly as
deployed in Long_Term_Trading/clam_inference.py:

    features = [dlog Open, dlog High, dlog Low, dlog Close, dlog1p Volume]  (raw, unadjusted)
    input    = last 252 rows, MinMax-scaled with the training scaler
    output   = 65 x 5 scaled log-diffs -> inverse-scaled
    score    = exp( sum_{h=1..65} pred_dlog_High_h ) - 1     ("expected growth")

Two weight sets are evaluated:
    clam_orig : Quant_Model_Research/quarterly_model.h5  (trained through 2025-08 -> in-sample before that)
    clam_2021 : models/clam_2021_model.h5                 (same code, training cut at 2021-12-31)

Inference is batched per rebalance date over tradable tickers only, and written
to `signals` date-by-date so the job is resumable.
"""
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import config
from sv import db, features

sys.path.insert(0, str(config.CLAM_ORIGINAL_H5.parent))
from clam_workflow import Attention, directional_accuracy  # noqa: E402  (original custom layers)

MODELS = {
    "clam_orig": (config.CLAM_ORIGINAL_H5, config.CLAM_ORIGINAL_SCALER),
    "clam_2021": (config.MODELS_DIR / "clam_2021_model.h5", config.MODELS_DIR / "clam_2021_scaler.pkl"),
}
SEQ = config.CLAM_SEQ_LEN


def load(name: str):
    h5, pkl = MODELS[name]
    model = tf.keras.models.load_model(h5, custom_objects={"Attention": Attention,
                                                            "directional_accuracy": directional_accuracy},
                                       compile=False)
    return model, joblib.load(pkl)


def feature_matrix(con) -> tuple[np.ndarray, pd.Index, pd.Index]:
    """(T, N, 5) array of dlog features on the daily calendar, NaN where missing.
    Cached under data/cache/ (invalidate by deleting the files after re-ingest)."""
    cache = config.CACHE_DIR / "clam_features.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"], pd.Index(z["dates"]), pd.Index(z["tickers"])
    mats = []
    for f in ("open", "high", "low", "close", "volume"):
        w = features.load_wide(con, f)
        w = w.where(w > 0)
        mats.append(np.log1p(w).diff() if f == "volume" else np.log(w).diff())
    X = np.stack([m.values for m in mats], axis=-1).astype(np.float32)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, dates=mats[0].index.values, tickers=mats[0].columns.values)
    return X, mats[0].index, mats[0].columns


def windows_for_date(X, dates_idx, tickers_idx, date: str, wanted: set[str]):
    t = dates_idx.get_loc(date)
    if t + 1 < SEQ:
        return None, None
    block = X[t + 1 - SEQ: t + 1]                       # (SEQ, N, 5)
    ok = ~np.isnan(block).any(axis=(0, 2))              # ticker has a full clean window
    cols = np.array([c in wanted for c in tickers_idx])
    sel = np.where(ok & cols)[0]
    if len(sel) == 0:
        return None, None
    return np.transpose(block[:, sel, :], (1, 0, 2)), tickers_idx[sel]   # (n, SEQ, 5)


def score_batch(model, scaler, W: np.ndarray, batch: int = 512) -> np.ndarray:
    n = W.shape[0]
    Ws = scaler.transform(W.reshape(-1, 5)).reshape(n, SEQ, 5)
    pred = model.predict(Ws, batch_size=batch, verbose=0)               # (n, 65, 5) scaled
    inv = scaler.inverse_transform(pred.reshape(-1, 5)).reshape(n, -1, 5)
    return np.exp(inv[:, :, 1].sum(axis=1)) - 1.0                        # High column, as deployed


def run(con, name: str, start: str | None = None, end: str | None = None) -> None:
    model, scaler = load(name)
    X, didx, tidx = feature_matrix(con)
    dates = features.rebalance_dates(con)
    done = set(db.read(con, "SELECT DISTINCT date FROM signals WHERE model=?", (name,))["date"])
    todo = [d for d in dates if d not in done and (not start or d >= start) and (not end or d <= end)]
    print(f"{name}: {len(todo)} dates to score", flush=True)
    trad = db.read(con, "SELECT date, ticker FROM panel WHERE tradable = 1")
    trad_by_date = trad.groupby("date")["ticker"].apply(set).to_dict()
    for i, d in enumerate(todo):
        W, tick = windows_for_date(X, didx, tidx, d, trad_by_date.get(d, set()))
        if W is None:
            continue
        s = score_batch(model, scaler, W)
        df = pd.DataFrame({"date": d, "ticker": tick, "model": name, "score": s})
        db.write_df(con, df, "signals")
        if i % 10 == 0:
            print(f"  [{i+1}/{len(todo)}] {d}: {len(tick)} tickers", flush=True)


if __name__ == "__main__":
    name = sys.argv[1]
    con = db.connect()
    run(con, name, *(sys.argv[2:4]))
