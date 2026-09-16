-- Build the weekly rebalance panel from daily prices.
-- Parameters: $min_price, $min_adv, $start
--
-- Reading guide (each CTE feeds the next):
--   daily     : per ticker, per day -> liquidity stats + prices shifted forward with LEAD()
--   week_end  : the one calendar shared by every ticker: SPY's last trading day of each week
--   weekly    : keep only rebalance days
--   weekly_lead: price at the *next* rebalance date, to compute the 1-week forward return
--
-- Window functions in one line: "OVER (PARTITION BY ticker ORDER BY date ...)" means
-- "compute this per ticker, in date order, looking at a sliding range of rows".

WITH daily AS (
    SELECT
        ticker, date, close, adj_close, volume,
        -- trailing 20-day average dollar volume (this row + 19 before it)
        AVG(close * volume) OVER (
            PARTITION BY ticker ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)                       AS adv20_usd,
        -- how many rows the 20-day window actually contains (< 20 near a listing date)
        COUNT(*) OVER (
            PARTITION BY ticker ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)                       AS n20,
        -- LEAD(x, k) = value of x k rows later, within the same ticker
        LEAD(adj_close, 1)  OVER (PARTITION BY ticker ORDER BY date)         AS adj_close_t1,
        LEAD(adj_close, 65) OVER (PARTITION BY ticker ORDER BY date)         AS adj_close_t65
    FROM prices
    -- Yahoo occasionally returns zero or negative adjusted prices; treat them as missing
    WHERE close > 0 AND adj_close > 0
),
week_end AS (
    -- date_trunc('week', d) = the Monday of d's week, so grouping by it groups by ISO week
    SELECT MAX(date) AS date
    FROM prices
    WHERE ticker = 'SPY'
    GROUP BY date_trunc('week', date)
),
weekly AS (
    SELECT d.*
    FROM daily d
    JOIN week_end w USING (date)
    WHERE d.date >= $start
),
weekly_lead AS (
    SELECT
        *,
        LEAD(adj_close, 1) OVER (PARTITION BY ticker ORDER BY date)           AS adj_close_nw
    FROM weekly
)
INSERT INTO panel
    (date, ticker, close, adj_close, adv20_usd, tradable, ret_fwd_1w, ret_fwd_1w_lag1, ret_fwd_13w)
SELECT
    date, ticker, close, adj_close, adv20_usd,
    close >= $min_price AND adv20_usd >= $min_adv AND n20 = 20                AS tradable,
    adj_close_nw  / adj_close     - 1.0                                       AS ret_fwd_1w,
    adj_close_nw  / adj_close_t1  - 1.0                                       AS ret_fwd_1w_lag1,
    adj_close_t65 / adj_close     - 1.0                                       AS ret_fwd_13w
FROM weekly_lead
WHERE adj_close > 0;
