-- Signal Validation Framework: core schema (DuckDB)
-- Conventions: real DATE columns (not text); all money in USD; one row = one observation.
-- Tables flow top-down: universe -> prices -> panel -> signals -> portfolio_returns
--                       -> monitoring / gate_decisions / validation_results (report inputs)

-- Snapshot of the top-N US equities by market cap on UNIVERSE_ASOF.
CREATE TABLE IF NOT EXISTS universe (
    ticker        VARCHAR PRIMARY KEY,
    name          VARCHAR,
    exchange      VARCHAR,
    sector        VARCHAR,
    industry      VARCHAR,
    market_cap    DOUBLE,
    rank_by_mcap  INTEGER,
    asof_date     DATE NOT NULL
);

-- Daily OHLCV from Yahoo Finance. adj_close is split/dividend adjusted; the others are raw.
CREATE TABLE IF NOT EXISTS prices (
    date       DATE NOT NULL,
    ticker     VARCHAR NOT NULL,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE,
    adj_close  DOUBLE,
    volume     DOUBLE,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS ix_prices_date ON prices(date);

-- One row per (rebalance date, ticker): tradability flag + realised forward returns.
-- Built by sql/queries/build_panel.sql.
CREATE TABLE IF NOT EXISTS panel (
    date             DATE NOT NULL,     -- rebalance date (last trading day of the week)
    ticker           VARCHAR NOT NULL,
    close            DOUBLE,
    adj_close        DOUBLE,
    adv20_usd        DOUBLE,            -- 20-day average dollar volume (liquidity filter)
    tradable         BOOLEAN,           -- passes MIN_PRICE and MIN_ADV_USD
    ret_fwd_1w       DOUBLE,            -- next-week total return, rebalance close -> next rebalance close
    ret_fwd_1w_lag1  DOUBLE,            -- next-session close to following next-session close (base execution)
    ret_fwd_13w      DOUBLE,            -- 65-trading-day forward return (candidate models' horizon)
    PRIMARY KEY (date, ticker)
);

-- Raw candidate model outputs. Higher score = more bullish.
CREATE TABLE IF NOT EXISTS signals (
    date     DATE NOT NULL,
    ticker   VARCHAR NOT NULL,
    model    VARCHAR NOT NULL,          -- 'momentum' | 'gbm' | 'gbm_expected' | 'clam_orig' | 'clam_2021'
    score    DOUBLE,
    PRIMARY KEY (model, date, ticker)
);

-- Weekly portfolio returns per strategy, net of transaction costs.
CREATE TABLE IF NOT EXISTS portfolio_returns (
    date       DATE NOT NULL,
    strategy   VARCHAR NOT NULL,        -- model name, '<model>_lag1', '<model>_gated', 'benchmark_spy', 'universe_ew'
    ret_gross  DOUBLE,
    ret_net    DOUBLE,
    turnover   DOUBLE,
    n_held     INTEGER,
    PRIMARY KEY (strategy, date)
);

-- Rolling model-monitoring metrics, long format (one row per model/metric/week).
CREATE TABLE IF NOT EXISTS monitoring (
    date      DATE NOT NULL,
    model     VARCHAR NOT NULL,
    metric    VARCHAR NOT NULL,         -- 'psi_score' | 'psi_input' | 'ks_1w' | 'auc_1w' | 'calib_slope' | 'rolling_sharpe' | 'drawdown' ...
    value     DOUBLE,
    PRIMARY KEY (model, metric, date)
);

-- Kill-switch decisions: whether the signal was used this week, and the reason if not.
CREATE TABLE IF NOT EXISTS gate_decisions (
    date      DATE NOT NULL,
    model     VARCHAR NOT NULL,
    gate_on   BOOLEAN NOT NULL,
    reason    VARCHAR,
    PRIMARY KEY (model, date)
);

-- Validation summary (one row per test per model) feeding the report.
CREATE TABLE IF NOT EXISTS validation_results (
    model      VARCHAR NOT NULL,
    test       VARCHAR NOT NULL,        -- 'bootstrap_sharpe_ci' | 'deflated_sharpe' | 'permutation_null'
    statistic  DOUBLE,
    ci_low     DOUBLE,
    ci_high    DOUBLE,
    p_value    DOUBLE,
    verdict    VARCHAR,                 -- 'PASS' | 'FAIL'
    detail     VARCHAR,
    run_at     TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (model, test)
);
