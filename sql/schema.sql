-- Signal Validation Framework: core schema (SQLite)
-- Convention: dates stored as ISO text (YYYY-MM-DD); all money in USD.

CREATE TABLE IF NOT EXISTS universe (
    ticker        TEXT PRIMARY KEY,
    name          TEXT,
    exchange      TEXT,
    sector        TEXT,
    industry      TEXT,
    market_cap    REAL,
    rank_by_mcap  INTEGER,
    asof_date     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prices (
    date       TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    open       REAL, high REAL, low REAL, close REAL,
    adj_close  REAL,
    volume     REAL,
    PRIMARY KEY (ticker, date)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_prices_date ON prices(date);

-- One row per (rebalance date, ticker): model inputs + realised forward return.
CREATE TABLE IF NOT EXISTS panel (
    date         TEXT NOT NULL,      -- rebalance date (Friday close)
    ticker       TEXT NOT NULL,
    close        REAL,
    adj_close    REAL,
    adv20_usd    REAL,               -- 20d avg dollar volume (liquidity filter)
    tradable     INTEGER,            -- 1 if passes MIN_PRICE / MIN_ADV_USD
    ret_fwd_1w   REAL,               -- next-week total return, rebalance close -> next rebalance close
    ret_fwd_1w_lag1 REAL,            -- same but entered one trading day later (execution-lag sensitivity)
    ret_fwd_13w  REAL,               -- 65-trading-day forward return (candidate models' horizon)
    PRIMARY KEY (date, ticker)
) WITHOUT ROWID;

-- Raw candidate model outputs. One row per (date, ticker, model).
CREATE TABLE IF NOT EXISTS signals (
    date     TEXT NOT NULL,
    ticker   TEXT NOT NULL,
    model    TEXT NOT NULL,          -- 'momentum' | 'gbm' | 'clam_orig' | 'clam_2021'
    score    REAL,                   -- raw model score (higher = more bullish)
    PRIMARY KEY (model, date, ticker)
) WITHOUT ROWID;

-- Weekly portfolio returns per strategy (net of costs).
CREATE TABLE IF NOT EXISTS portfolio_returns (
    date       TEXT NOT NULL,
    strategy   TEXT NOT NULL,        -- model name, or '<model>_gated', or 'benchmark'
    ret_gross  REAL,
    ret_net    REAL,
    turnover   REAL,
    n_held     INTEGER,
    PRIMARY KEY (strategy, date)
) WITHOUT ROWID;

-- Rolling monitoring metrics per model per week.
CREATE TABLE IF NOT EXISTS monitoring (
    date      TEXT NOT NULL,
    model     TEXT NOT NULL,
    metric    TEXT NOT NULL,         -- 'psi' | 'ks' | 'auc' | 'rolling_sharpe' | 'drawdown' | 'calib_slope'
    value     REAL,
    PRIMARY KEY (model, metric, date)
) WITHOUT ROWID;

-- Kill-switch decisions.
CREATE TABLE IF NOT EXISTS gate_decisions (
    date      TEXT NOT NULL,
    model     TEXT NOT NULL,
    gate_on   INTEGER NOT NULL,      -- 1 = signal trusted this week
    reason    TEXT,
    PRIMARY KEY (model, date)
) WITHOUT ROWID;

-- Validation summary (one row per test per model) feeding the report.
CREATE TABLE IF NOT EXISTS validation_results (
    model      TEXT NOT NULL,
    test       TEXT NOT NULL,
    statistic  REAL,
    ci_low     REAL,
    ci_high    REAL,
    p_value    REAL,
    verdict    TEXT,
    detail     TEXT,
    run_at     TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model, test)
);
