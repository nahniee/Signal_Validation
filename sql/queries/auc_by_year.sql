-- AUC of each model's score for next-week direction, per calendar year, in pure SQL.
-- Parameter: $model
--
-- AUC = P(score of a random "up" stock > score of a random "down" stock).
-- Mann-Whitney identity: with RANK() over all scores in the group,
--     AUC = (sum of ranks of positives - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
-- so no ROC curve is needed - just ranks and two counts.
--
--   labelled : y = 1 when the stock beat the cross-sectional median that week
--   ranked   : rank every score within (year), ties get the average rank

WITH labelled AS (
    SELECT
        s.date, s.score,
        EXTRACT(year FROM s.date)                                               AS yr,
        p.ret_fwd_1w > MEDIAN(p.ret_fwd_1w) OVER (PARTITION BY p.date)          AS y
    FROM signals s
    JOIN panel p USING (date, ticker)
    WHERE s.model = $model AND p.tradable AND p.ret_fwd_1w IS NOT NULL AND s.score IS NOT NULL
),
ranked AS (
    SELECT
        yr, y,
        -- average rank for ties = (RANK + RANK counted from the other end) / 2
        (RANK() OVER (PARTITION BY yr ORDER BY score)
         + COUNT(*) OVER (PARTITION BY yr)
         - RANK() OVER (PARTITION BY yr ORDER BY score DESC) + 1) / 2.0         AS rk
    FROM labelled
)
SELECT
    yr                                                                          AS year,
    COUNT(*)                                                                    AS n,
    (SUM(CASE WHEN y THEN rk END) - SUM(y::INT) * (SUM(y::INT) + 1) / 2.0)
        / (SUM(y::INT) * SUM((NOT y)::INT))                                     AS auc
FROM ranked
GROUP BY yr
ORDER BY yr;
