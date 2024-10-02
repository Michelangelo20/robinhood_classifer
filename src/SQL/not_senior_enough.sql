WITH temp1 AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp) AS rn,
        timestamp::date - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp)::int AS streak_id
    FROM
        equity_value_data
),
temp2 AS (
    SELECT
        user_id,
        MIN(timestamp::date) AS min_date,
        MAX(timestamp::date) AS max_date,
        COUNT(*) AS duration_of_above10_streak
    FROM
        temp1
    GROUP BY
        user_id, streak_id
),
temp3 AS (
    SELECT
        *,
        LAG(max_date) OVER (PARTITION BY user_id ORDER BY min_date ASC) AS last_streak_date,
        min_date - LAG(max_date) OVER (PARTITION BY user_id ORDER BY min_date ASC) AS duration_between_above10_streaks
    FROM
        temp2
),
temp4 AS (
    SELECT *
    FROM temp3
    WHERE duration_between_above10_streaks >= 28
)
SELECT
    CONCAT(
        ROUND(
            (COUNT(DISTINCT user_id) * 1.0 / (SELECT COUNT(DISTINCT user_id) FROM equity_value_data)) * 100, 3
        ), '%'
    ) AS churn_rate
FROM
    temp4;
