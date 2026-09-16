-- How many of today's 3,000 tickers actually have prices, and pass the liquidity
-- filter, in each past year. The rising count is the footprint of survivorship
-- bias: companies that were delisted before the snapshot date never appear, and
-- companies listed after year X are absent from year X.

SELECT
    EXTRACT(year FROM date)                     AS year,
    COUNT(DISTINCT ticker)                      AS with_prices,
    COUNT(DISTINCT CASE WHEN tradable THEN ticker END) AS tradable_any_week,
    ROUND(AVG(tradable::INT), 3)                AS tradable_share,
    MEDIAN(adv20_usd) / 1e6                     AS median_adv_musd
FROM panel
GROUP BY 1
ORDER BY 1;
