-- Performance summary per strategy from $start onwards (out-of-time window).
-- Champion = '<model>', Challenger = '<model>_gated'; benchmarks alongside.
--
--   eq   : cumulative log-equity via a running SUM, and its running MAX (= the peak so far)
--   drawdown at each week = exp(log_equity - peak) - 1
-- Everything else is plain aggregation. 52 = weeks per year for annualisation.

WITH r AS (
    SELECT strategy, date, ret_net
    FROM portfolio_returns
    WHERE date >= $start AND ret_net IS NOT NULL
),
cum AS (
    SELECT
        strategy, date, ret_net,
        SUM(LN(1 + ret_net)) OVER (PARTITION BY strategy ORDER BY date)         AS log_eq
    FROM r
),
eq AS (
    -- a window function cannot be nested inside another, hence the second CTE
    SELECT
        *,
        MAX(log_eq) OVER (PARTITION BY strategy ORDER BY date)                  AS log_peak
    FROM cum
)
SELECT
    strategy,
    COUNT(*)                                            AS weeks,
    EXP(52.0 * AVG(LN(1 + ret_net))) - 1                AS cagr,
    STDDEV_SAMP(ret_net) * SQRT(52)                     AS vol,
    AVG(ret_net) / STDDEV_SAMP(ret_net) * SQRT(52)      AS sharpe,
    MIN(EXP(log_eq - log_peak) - 1)                     AS max_drawdown,
    EXP(SUM(LN(1 + ret_net))) - 1                       AS total_return
FROM eq
GROUP BY strategy
ORDER BY strategy;
