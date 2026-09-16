-- Trailing 252-trading-day realised volatility (annualised) on each rebalance date.
-- Used as the "model input" whose distribution drift we monitor with PSI.
--
--   lr : daily log return per ticker  (LAG = previous row's value)
--   rv : rolling variance over the last 252 rows via  E[r^2] - E[r]^2, then sqrt and annualise
-- The named WINDOW clause lets the same sliding window be reused by several columns.

WITH lr AS (
    SELECT ticker, date,
           LN(adj_close / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date)) AS r
    FROM prices
    WHERE adj_close > 0
),
rv AS (
    SELECT ticker, date,
           SQRT(252.0 * (AVG(r * r) OVER w - AVG(r) OVER w * AVG(r) OVER w)) AS vol252,
           COUNT(r) OVER w                                                   AS n
    FROM lr
    WINDOW w AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)
)
SELECT p.date, p.ticker, rv.vol252
FROM panel p
JOIN rv USING (ticker, date)
WHERE p.tradable AND rv.n >= 200;
