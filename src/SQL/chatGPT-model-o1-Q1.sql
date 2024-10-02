WITH all_dates AS (
    -- Step 1: Generate a complete calendar of dates
    SELECT generate_series(
        (SELECT MIN(timestamp)::date FROM ds.equity_value_data),
        (SELECT MAX(timestamp)::date FROM ds.equity_value_data),
        INTERVAL '1 day'
    )::date AS date
),
market_status AS (
    -- Step 2: Identify market open and closed days
    SELECT
        ad.date,
        CASE WHEN EXISTS (
            SELECT 1 FROM ds.equity_value_data e WHERE e.timestamp::date = ad.date
        ) THEN 'open' ELSE 'closed' END AS market_status
    FROM all_dates ad
),
users AS (
    -- Get the list of all users
    SELECT DISTINCT user_id FROM ds.equity_value_data
),
user_dates AS (
    -- Step 3: Create user-date combinations
    SELECT u.user_id, ms.date, ms.market_status
    FROM users u
    CROSS JOIN market_status ms
),
user_equity AS (
    -- Left join to get close_equity where available
    SELECT
        ud.user_id,
        ud.date,
        ud.market_status,
        e.close_equity
    FROM user_dates ud
    LEFT JOIN ds.equity_value_data e
        ON ud.user_id = e.user_id AND ud.date = e.timestamp::date
),
user_state AS (
    -- Step 4: Determine equity state for each user-date
    SELECT
        ue.user_id,
        ue.date,
        ue.market_status,
        CASE
            WHEN ue.close_equity IS NOT NULL THEN 'equity_ge_10'
            WHEN ue.market_status = 'open' THEN 'equity_lt_10'
            ELSE NULL -- To be filled
        END AS state
    FROM user_equity ue
),
user_state_filled AS (
    -- Carry forward the last known state
    SELECT
        user_id,
        date,
        market_status,
        state,
        COALESCE(
            state,
            LAST_VALUE(state) OVER (
                PARTITION BY user_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            'equity_ge_10' -- Default initial state
        ) AS filled_state
    FROM user_state
    ORDER BY user_id, date
),
user_flag AS (
    -- Flag days where equity is less than $10
    SELECT
        user_id,
        date,
        market_status,
        filled_state,
        CASE WHEN filled_state = 'equity_lt_10' THEN 1 ELSE 0 END AS below_10
    FROM user_state_filled
),
user_sequences AS (
    -- Identify consecutive sequences
    SELECT
        user_id,
        date,
        below_10,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY date) AS rn1,
        ROW_NUMBER() OVER (PARTITION BY user_id, below_10 ORDER BY date) AS rn2
    FROM user_flag
),
user_sequences_grouped AS (
    -- Group and count sequences of equity < $10
    SELECT
        user_id,
        MIN(date) AS start_date,
        MAX(date) AS end_date,
        COUNT(*) AS num_days,
        (MAX(date) - MIN(date) + 1) AS num_calendar_days
    FROM user_sequences
    WHERE below_10 = 1
    GROUP BY user_id, (rn1 - rn2)
),
churned_users AS (
    -- Step 6: Determine churned users
    SELECT DISTINCT user_id
    FROM user_sequences_grouped
    WHERE num_calendar_days >= 28
)
-- Step 7: Calculate the percentage of users who have churned
SELECT
    (SELECT COUNT(DISTINCT user_id) FROM churned_users) * 100.0 /
    (SELECT COUNT(DISTINCT user_id) FROM users) AS percentage_churned;
