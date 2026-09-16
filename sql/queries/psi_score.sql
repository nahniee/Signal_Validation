-- Population Stability Index of a model's score distribution, per calendar year,
-- against the development sample (first $dev_weeks rebalance weeks).
-- Parameters: $model, $dev_weeks, $buckets
--
-- PSI = SUM over buckets of (actual% - expected%) * LN(actual% / expected%)
--   < 0.10 stable | 0.10 - 0.25 watch | > 0.25 shifted
--
--   dev      : the development sample
--   edges    : its decile boundaries (quantile_cont over an equally spaced list)
--   bucketed : every score assigned to a bucket = number of edges it exceeds
--   expected / actual : bucket shares for dev vs each year
-- 1e-4 floors a share so an empty bucket does not produce LN(0).

WITH scored AS (
    SELECT s.date, s.score, EXTRACT(year FROM s.date) AS yr,
           DENSE_RANK() OVER (ORDER BY s.date)         AS week_no
    FROM signals s JOIN panel p USING (date, ticker)
    WHERE s.model = $model AND p.tradable AND s.score IS NOT NULL
),
dev AS (
    SELECT score FROM scored WHERE week_no <= $dev_weeks
),
edges AS (
    SELECT UNNEST(quantile_cont(score, [x / $buckets::DOUBLE FOR x IN range(1, $buckets)])) AS edge
    FROM dev
),
bucketed AS (
    SELECT s.yr, s.week_no <= $dev_weeks AS is_dev,
           (SELECT COUNT(*) FROM edges e WHERE s.score > e.edge) AS bucket
    FROM scored s
),
expected AS (
    SELECT bucket, COUNT(*) / SUM(COUNT(*)) OVER () AS share
    FROM bucketed WHERE is_dev GROUP BY bucket
),
actual AS (
    SELECT yr, bucket, COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY yr) AS share
    FROM bucketed GROUP BY yr, bucket
),
grid AS (
    -- every (year, bucket) pair, so a bucket that is empty in some year still counts
    SELECT yr, bucket FROM (SELECT DISTINCT yr FROM scored) CROSS JOIN expected
)
SELECT
    g.yr                                                                        AS year,
    SUM((GREATEST(COALESCE(a.share, 0), 1e-4) - GREATEST(e.share, 1e-4))
        * LN(GREATEST(COALESCE(a.share, 0), 1e-4) / GREATEST(e.share, 1e-4)))   AS psi
FROM grid g
JOIN expected e USING (bucket)
LEFT JOIN actual a USING (yr, bucket)
GROUP BY g.yr
ORDER BY g.yr;
