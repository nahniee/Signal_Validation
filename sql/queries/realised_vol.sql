-- Trailing 252-trading-day realised volatility (annualised) on each rebalance date.
-- Uses a window function over daily log returns, then joins to the weekly panel.
WITH lr AS (
    SELECT ticker, date,
           LN(adj_close / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date)) AS r
    FROM prices
    WHERE adj_close > 0
),
rv AS (
    SELECT ticker, date,
           -- population variance over the trailing 252 obs, annualised
           SQRT(252.0 * (AVG(r * r) OVER w - AVG(r) OVER w * AVG(r) OVER w)) AS vol252,
           COUNT(r) OVER w AS n
    FROM lr
    WINDOW w AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)
)
SELECT p.date, p.ticker, rv.vol252
FROM panel p
JOIN rv ON rv.ticker = p.ticker AND rv.date = p.date
WHERE p.tradable = 1 AND rv.n >= 200;
