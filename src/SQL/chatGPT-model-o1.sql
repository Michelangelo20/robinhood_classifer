WITH all_dates AS (
    -- Step 1: Generate a continuous sequence of dates
    SELECT generate_series(
        (SELECT MIN(timestamp)::date FROM ds.equity_value_data),
        (SELECT MAX(timestamp)::date FROM ds.equity_value_data),
        INTERVAL '1 day'
    )::date AS date
),
users AS (
    -- Step 2: Get a list of all users
    SELECT DISTINCT user_id FROM ds.equity_value_data
),
user_dates AS (
    -- Step 2: Associate each user with every date
    SELECT u.user_id, d.date
    FROM users u
    CROSS JOIN all_dates d
),
left_joined AS (
    -- Step 3 & 4: Mark dates when equity ≥ $10, infer equity < $10 for missing dates
    SELECT
        ud.user_id,
        ud.date,
        CASE WHEN e.user_id IS NULL THEN 1 ELSE 0 END AS below_10
    FROM user_dates ud
    LEFT JOIN ds.equity_value_data e
        ON ud.user_id = e.user_id AND ud.date = e.timestamp::date
),
consecutive_periods AS (
    -- Step 5: Detect consecutive periods of equity < $10
    SELECT
        user_id,
        date,
        below_10,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY date) AS rn1,
        ROW_NUMBER() OVER (PARTITION BY user_id, below_10 ORDER BY date) AS rn2
    FROM left_joined
),
grouped_sequences AS (
    SELECT
        user_id,
        MIN(date) AS start_date,
        MAX(date) AS end_date,
        COUNT(*) AS num_days
    FROM consecutive_periods
    WHERE below_10 = 1
    GROUP BY user_id, (rn1 - rn2)
)
-- Step 6: Extract users who have churned
SELECT DISTINCT user_id
FROM grouped_sequences
WHERE num_days >= 28;
