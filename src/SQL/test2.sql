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
),
temp5 AS (
    SELECT DISTINCT user_id
    FROM temp4
),
temp6 AS (
    SELECT
        user_id,
        MIN(timestamp::date) AS user_behavior_first_login,
        MAX(timestamp::date) AS user_behavior_last_login
    FROM
        equity_value_data
    WHERE user_id IN (SELECT user_id FROM temp5)
    GROUP BY user_id
),
temp7 AS (
    SELECT
        t1.user_id,
        min_date,
        max_date,
        duration_of_above10_streak,
        last_streak_date,
        duration_between_above10_streaks,
        user_behavior_first_login,
        user_behavior_last_login
    FROM
        temp6 t1
    RIGHT JOIN temp4 t2 ON t1.user_id = t2.user_id
),
temp8 AS (
    SELECT
        user_id,
        min_date,
        max_date,
        duration_of_above10_streak,
        last_streak_date,
        duration_between_above10_streaks,
        user_behavior_first_login,
        user_behavior_last_login
    FROM (
        SELECT
            *,
            row_number() OVER (PARTITION BY user_id ORDER BY min_date DESC) AS ranking
        FROM temp7) t1
    WHERE ranking = 1
)
SELECT * FROM temp8 WHERE min_date=user_behavior_last_login

