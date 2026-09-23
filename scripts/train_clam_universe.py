"""Predeclared paired universe-size experiment; never select using OOT outcomes.
Frozen architecture: historical cs_rank_small. Same seed, purged validation split,
finite-window policy and callbacks for N=94, N=500 and N=3000, so only training-universe
size varies. N=94 matches the size of the original hand-picked training list, but selects
by market-cap rank like the other rungs. Original artifacts unchanged.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','1')
import sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); import config
sys.path.insert(0,str(config.QMR_DIR)); import clam_weekly as cw
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler

SEED=20260922

def configuration(n):
    cfg=dict(cw.CONFIG)
    cfg.update(n_tickers=n,target='cs_rank',lstm_layers=[{'units':128}]*2,
               cnn_layers=[{'filters':64,'kernel_size':5}]*2,dropout=.3,seed=SEED,
               batch_size=256,epochs=40,purge_validation_boundary=True)
    return cfg

def split_masks(ends,target_ends,split):
    return target_ends < np.datetime64(split), ends >= np.datetime64(split)

def build(cfg):
    # Explicitly constrain the DB query before any features or labels are built.
    import duckdb
    c=duckdb.connect(cfg['db_path'],read_only=True)
    prices=c.execute('''SELECT p.date,p.ticker,p.open,p.high,p.low,p.close,p.adj_close,p.volume
       FROM prices p JOIN universe u USING(ticker)
       WHERE u.rank_by_mcap <= $n AND p.date >= $start AND p.date <= $end
       AND p.close>0 AND p.adj_close>0 ORDER BY p.ticker,p.date''',
       {'n':cfg['n_tickers'],'start':cfg['train_start'],'end':cfg['train_end']}).df();c.close()
    factor=prices.adj_close/prices.close
    for col in ['open','high','low','close']: prices[col] *= factor
    split=pd.Timestamp(cfg['train_end'])-pd.DateOffset(years=cfg['val_years'])
    feats={t:cw.features_for_ticker(g).replace([np.inf,-np.inf],np.nan)
           for t,g in prices.groupby('ticker',sort=False)}
    fitting=pd.concat([f[f.index<split] for f in feats.values()]).dropna()
    scaler=StandardScaler().fit(fitting.values)
    del fitting,prices
    parts=[]; contributing=[]; removed=0
    for ticker,f in feats.items():
        ends=cw.window_ends(len(f),cfg['seq_length'],cfg['horizon'],cfg['stride'])
        if not len(ends): continue
        values=f.to_numpy()
        bad=(~np.isfinite(values).all(axis=1)).astype(int)
        sums=np.r_[0,np.cumsum(bad)]
        ok=(sums[ends+cfg['horizon']+1]-sums[ends-cfg['seq_length']+1])==0
        removed+=int((~ok).sum()); ends=ends[ok]
        if not len(ends): continue
        z=np.clip(scaler.transform(np.nan_to_num(values)), -cfg['clip_sigma'],cfg['clip_sigma'])
        parts.append((cw.make_windows(z,ends,cfg['seq_length']),
                      cw.forward_logret(np.nan_to_num(values[:,3]),ends,cfg['horizon']),
                      f.index[ends].values,f.index[ends+cfg['horizon']].values))
        contributing.append(ticker)
    X=np.concatenate([p[0] for p in parts]); y=np.concatenate([p[1] for p in parts])
    dates=np.concatenate([p[2] for p in parts]); maturity=np.concatenate([p[3] for p in parts]); del parts,feats
    # Preserve historical per-ticker stride and exact-date rank target for both sizes.
    y=(pd.Series(y).groupby(dates).rank(pct=True).values-.5).astype(np.float32)
    train,val=split_masks(dates,maturity,split)
    scale=float(y[train].std())
    if scale<=0 or not np.isfinite(X).all(): raise ValueError('Invalid dataset')
    metadata={'requested_n_tickers':cfg['n_tickers'],'n_tickers':len(contributing),
              'tickers':contributing,'train_windows':int(train.sum()),'val_windows':int(val.sum()),
              'purged_windows':int((~(train|val)).sum()),'invalid_windows_removed':removed,
              'split':str(split.date()),'y_scale':scale,
              'last_training_target':str(pd.Timestamp(maturity[train].max()).date()),
              'last_validation_target':str(pd.Timestamp(maturity[val].max()).date()),
              'limitation':'Later universe snapshot and historical per-ticker stride retained; OOT period already inspected.'}
    return X[train],(y[train]/scale)[:,None],X[val],(y[val]/scale)[:,None],scaler,metadata

class Progress(tf.keras.callbacks.Callback):
    def __init__(self,path): super().__init__();self.path=path;self.started=time.time()
    def on_epoch_end(self,epoch,logs=None):
        self.path.write_text(json.dumps({'epoch':epoch+1,'elapsed_seconds':time.time()-self.started,
                                        'metrics':{k:float(v) for k,v in (logs or {}).items()}},indent=2))

def run(n):
    cfg=configuration(n);tag=f'cs_rank_small_n{n}_seed{SEED}'
    out=config.QMR_DIR/'clam_weekly'/tag;out.mkdir(parents=True,exist_ok=True)
    if (out/'clam_weekly_meta.json').exists():
        raise FileExistsError(f'Completed experiment already exists: {out}')
    tf.keras.utils.set_random_seed(SEED)
    gpus=tf.config.list_physical_devices('GPU')
    if not gpus: raise RuntimeError('GPU unavailable; refusing an accidental CPU training run')
    for gpu in gpus:tf.config.experimental.set_memory_growth(gpu,True)
    (out/'experiment_config.json').write_text(json.dumps(cfg,indent=2))
    print(f'BUILD {tag}',flush=True)
    X,y,V,z,scaler,meta=build(cfg)
    print(json.dumps({k:v for k,v in meta.items() if k!='tickers'}),flush=True)
    (out/'dataset_metadata.json').write_text(json.dumps(meta,indent=2))
    model=cw.create_model(cfg)
    callbacks=[tf.keras.callbacks.ModelCheckpoint(str(out/'best.keras'),monitor='val_loss',save_best_only=True),
               tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=6,restore_best_weights=True),
               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.3,patience=3,min_lr=1e-6),
               tf.keras.callbacks.CSVLogger(str(out/'epochs.csv')),Progress(out/'progress.json')]
    history=model.fit(X,y,validation_data=(V,z),batch_size=cfg['batch_size'],epochs=cfg['epochs'],callbacks=callbacks,verbose=2)
    pred=model.predict(V,batch_size=1024,verbose=0).ravel()
    ic=float(pd.Series(pred).corr(pd.Series(z.ravel()),method='spearman'))
    da=float(np.mean(np.sign(pred)==np.sign(z.ravel())))
    model.save(out/'clam_weekly_model.keras');joblib.dump(scaler,out/'clam_weekly_scaler.pkl')
    result={**cfg,**meta,'val_rank_ic':ic,'val_dir_acc':da,'epochs_run':len(history.history['loss']),
            'history':{k:[float(v) for v in values] for k,values in history.history.items()},
            'model_sha256':hashlib.sha256((out/'clam_weekly_model.keras').read_bytes()).hexdigest()}
    (out/'clam_weekly_meta.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',tag,'val_rank_ic',ic,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('n',type=int,choices=[94,500,3000]);args=p.parse_args();run(args.n)
