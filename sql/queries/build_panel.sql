-- Build the weekly rebalance panel from daily prices using window functions.
-- Rebalance date = last trading day of each ISO week (normally Friday).
-- Parameters: :min_price, :min_adv, :start
WITH daily AS (
    SELECT
        ticker, date, close, adj_close, volume,
        AVG(close * volume) OVER (
            PARTITION BY ticker ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)                     AS adv20_usd,
        COUNT(*) OVER (
            PARTITION BY ticker ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)                     AS n20,
        LEAD(adj_close, 1)  OVER (PARTITION BY ticker ORDER BY date)       AS adj_close_t1,
        LEAD(adj_close, 65) OVER (PARTITION BY ticker ORDER BY date)       AS adj_close_t65,
        strftime('%Y-%W', date)                                            AS wk
    FROM prices
),
week_end AS (
    SELECT strftime('%Y-%W', date) AS wk, MAX(date) AS date FROM prices
    -- use the market calendar (SPY) so every ticker shares the same rebalance dates
    WHERE ticker = 'SPY'
    GROUP BY strftime('%Y-%W', date)
),
weekly AS (
    SELECT d.*
    FROM daily d
    JOIN week_end w ON w.date = d.date
    WHERE d.date >= :start
),
weekly_lead AS (
    SELECT
        *,
        LEAD(adj_close, 1) OVER (PARTITION BY ticker ORDER BY date)         AS adj_close_nw
    FROM weekly
)
INSERT OR REPLACE INTO panel
    (date, ticker, close, adj_close, adv20_usd, tradable, ret_fwd_1w, ret_fwd_1w_lag1, ret_fwd_13w)
SELECT
    date, ticker, close, adj_close, adv20_usd,
    CASE WHEN close >= :min_price AND adv20_usd >= :min_adv AND n20 = 20 THEN 1 ELSE 0 END AS tradable,
    adj_close_nw  / adj_close     - 1.0                                     AS ret_fwd_1w,
    adj_close_nw  / adj_close_t1  - 1.0                                     AS ret_fwd_1w_lag1,
    adj_close_t65 / adj_close     - 1.0                                     AS ret_fwd_13w
FROM weekly_lead
WHERE adj_close > 0;
