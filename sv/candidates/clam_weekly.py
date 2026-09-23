"""Scores a redeveloped weekly CLAM model over the tradable universe.

One trained variant per signals model name clam_weekly_<variant>. Preprocessing is
imported from the training module so training and inference cannot drift apart, which
was the defect recorded as F3. Scores are written date by date, so a run is resumable.
"""
import sys
import numpy as np
import pandas as pd
import config
from sv import db, features

sys.path.insert(0, str(config.QMR_DIR))
import clam_weekly as cw  # noqa: E402



def feature_matrix(con, bars: str = "daily") -> tuple[np.ndarray, pd.Index, pd.Index]:
    """(T, N, 5) dlog features, split/dividend adjusted, NaN where missing.
    bars='daily'  -> rows are trading days.
    bars='weekly' -> rows are trading weeks (W-FRI periods), one OHLCV bar each, built with the
                     trainer's own weekly_bars() so train and inference agree."""
    cache = config.CACHE_DIR / f"clam_weekly_features_{bars}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"], pd.Index(z["dates"]), pd.Index(z["tickers"])
    if bars == "daily":
        wide = {f: features.load_wide(con, f) for f in ("open", "high", "low", "close", "adj_close", "volume")}
        factor = wide["adj_close"] / wide["close"]
        mats = [np.log(wide[f] * factor).diff() for f in ("open", "high", "low", "close")]
        mats.append(np.log1p(wide["volume"]).diff())
        idx = mats[0].index
    else:
        px = db.read(con, "SELECT date, ticker, open, high, low, close, adj_close, volume FROM prices "
                          "WHERE close > 0 AND adj_close > 0 ORDER BY ticker, date")
        f = px.adj_close / px.close
        for c in ("open", "high", "low", "close"):
            px[c] = px[c] * f
        feats = {t: cw.features_for_ticker(g, "weekly") for t, g in px.groupby("ticker", sort=False)}
        for t, fdf in feats.items():
            fdf.index = pd.DatetimeIndex(fdf.index).to_period("W-FRI")
        mats = [pd.DataFrame({t: fdf[c] for t, fdf in feats.items()}).sort_index() for c in cw.FEATURES]
        idx = mats[0].index.to_timestamp(how="start")                # Monday of each trading week
    X = np.stack([m.reindex(index=mats[0].index, columns=mats[0].columns).values for m in mats], axis=-1).astype(np.float32)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, dates=idx.values, tickers=mats[0].columns.values)
    return X, idx, mats[0].columns


def windows_for_date(X, dates_idx, tickers_idx, date, wanted: set[str], seq: int, bars: str = "daily"):
    key = date.to_period("W-FRI").start_time if bars == "weekly" else date
    if key not in dates_idx:
        return None, None
    t = dates_idx.get_loc(key)
    if t + 1 < seq:
        return None, None
    block = X[t + 1 - seq: t + 1]                                   # (seq, N, 5)
    ok = ~np.isnan(block).any(axis=(0, 2)) & np.array([c in wanted for c in tickers_idx])
    sel = np.where(ok)[0]
    if len(sel) == 0:
        return None, None
    return np.transpose(block[:, sel, :], (1, 0, 2)), tickers_idx[sel]     # (n, seq, 5)


def run(con, target: str = "raw", start: str | None = None, end: str | None = None) -> None:
    NAME = f"clam_weekly_{target}"
    model, scaler, meta = cw.load_artifacts(config.QMR_DIR / "clam_weekly" / target)
    seq, clip, bars = meta["seq_length"], meta["clip_sigma"], meta.get("bars", "daily")
    X, didx, tidx = feature_matrix(con, bars)
    dates = features.rebalance_dates(con)
    done = set(db.read(con, "SELECT DISTINCT date FROM signals WHERE model = ?", [NAME])["date"])
    start, end = (pd.Timestamp(x) if x else None for x in (start, end))
    todo = [d for d in dates if d not in done and (not start or d >= start) and (not end or d <= end)]
    trad = db.read(con, "SELECT date, ticker FROM panel WHERE tradable")
    trad_by_date = trad.groupby("date")["ticker"].apply(set).to_dict()
    print(f"{NAME}: {len(todo)} dates to score (train_end {meta['train_end']}, val IC {meta['val_rank_ic']:.3f})", flush=True)
    for i, d in enumerate(todo):
        W, tick = windows_for_date(X, didx, tidx, d, trad_by_date.get(d, set()), seq, bars)
        if W is None:
            continue
        Z = np.clip(scaler.transform(W.reshape(-1, 5)), -clip, clip).reshape(W.shape).astype(np.float32)
        s = model.predict(Z, batch_size=1024, verbose=0).ravel()
        db.write_df(con, pd.DataFrame({"date": d, "ticker": tick, "model": NAME, "score": s}), "signals")
        if i % 25 == 0:
            print(f"  [{i+1}/{len(todo)}] {d.date()}: {len(tick)} tickers", flush=True)


if __name__ == "__main__":
    con = db.connect()
    run(con, *sys.argv[1:4])
