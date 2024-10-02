WITH user_dates AS (
    -- Step 1: Generate a continuous sequence of dates for each user
    SELECT
        u.user_id,
        d.date
    FROM (SELECT DISTINCT user_id FROM ds.equity_valueequity_value_data) u
    CROSS JOIN (
        SELECT generate_series(
            (SELECT MIN(timestamp)::date FROM ds.equity_valueequity_value_data),
            (SELECT MAX(timestamp)::date FROM ds.equity_valueequity_value_data),
            INTERVAL '1 day'
        )::date AS date
    ) d
),
equity_filled AS (
    -- Step 2: Fill missing equity values using last known value
    SELECT
        ud.user_id,
        ud.date,
        COALESCE(e.close_equity, LAG(e.close_equity) OVER (
            PARTITION BY ud.user_id ORDER BY ud.date
        )) AS close_equity
    FROM user_dates ud
    LEFT JOIN ds.equity_valueequity_value_data e
        ON ud.user_id = e.user_id AND ud.date = e.timestamp::date
),
below_10_flag AS (
    -- Step 3: Flag days where equity is below $10
    SELECT
        user_id,
        date,
        close_equity,
        CASE WHEN close_equity < 10 THEN 1 ELSE 0 END AS below_10
    FROM equity_filled
),
consecutive_days AS (
    -- Step 4: Identify consecutive periods where equity is below $10
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY date) -
        ROW_NUMBER() OVER (PARTITION BY user_id, (CASE WHEN below_10 = 1 THEN 1 ELSE 0 END) ORDER BY date) AS grp
    FROM below_10_flag
),
grouped_periods AS (
    -- Step 5: Group and count sequences
    SELECT
        user_id,
        MIN(date) AS start_date,
        MAX(date) AS end_date,
        COUNT(*) AS num_days
    FROM consecutive_days
    WHERE below_10 = 1
    GROUP BY user_id, grp
)
-- Final Result: Users who have churned
SELECT DISTINCT user_id
FROM grouped_periods
WHERE num_days >= 28;
