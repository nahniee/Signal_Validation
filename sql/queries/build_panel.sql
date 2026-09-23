-- All dates come from SPY's trading calendar; a missing quote stays missing.
-- Signal at week-end close; base execution next session close to next execution close.
WITH available_prices AS (
 SELECT * FROM prices WHERE date <= $end
), calendar AS (
 SELECT date, LEAD(date) OVER (ORDER BY date) AS next_session,
        LEAD(date, 5) OVER (ORDER BY date) AS day5,
        LEAD(date, 65) OVER (ORDER BY date) AS day65
 FROM available_prices WHERE ticker = 'SPY'
), week_dates AS (
 SELECT MAX(date) AS date FROM calendar GROUP BY date_trunc('week', date)
), schedule AS (
 SELECT w.date, c.next_session AS entry_date, c.day5, c.day65,
        LEAD(w.date) OVER (ORDER BY w.date) AS next_date,
        LEAD(c.next_session) OVER (ORDER BY w.date) AS exit_date
 FROM week_dates w JOIN calendar c USING(date)
), daily AS (
 SELECT *, AVG(close * volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS adv,
 COUNT(*) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS n20
 FROM available_prices WHERE close > 0 AND adj_close > 0
)
INSERT INTO panel
SELECT d.date, d.ticker, d.close, d.adj_close, d.adv,
 d.close >= $min_price AND d.adv >= $min_adv AND d.n20 = 20,
 CASE WHEN n.adj_close > 0 THEN n.adj_close / d.adj_close - 1 END,
 CASE WHEN e.adj_close > 0 AND x.adj_close > 0 THEN x.adj_close / e.adj_close - 1 END,
 CASE WHEN h.adj_close > 0 THEN h.adj_close / d.adj_close - 1 END
FROM daily d JOIN schedule s USING(date)
LEFT JOIN available_prices n ON n.ticker=d.ticker AND n.date=s.next_date
LEFT JOIN available_prices e ON e.ticker=d.ticker AND e.date=s.entry_date
LEFT JOIN available_prices x ON x.ticker=d.ticker AND x.date=s.exit_date
LEFT JOIN available_prices h ON h.ticker=d.ticker AND h.date=s.day65
WHERE d.date >= $start;
