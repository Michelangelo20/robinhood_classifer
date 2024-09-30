-- Problem Clarification:

-- Available Data: We have data for days when the user's equity value is ≥ $10.
-- Missing Data Implication: For days not present in equity_value_data.csv, the user's equity value was below $10.
-- Objective: Identify users who have a period of 28 or more consecutive calendar days where their equity value was below $10, which corresponds to periods when they have no records in equity_value_data.csv.


-- Revised Approach:

-- We need to:

-- Generate a Continuous Date Range: Create a complete sequence of dates covering the entire period of interest (from the earliest to the latest date in the dataset).

-- Associate Users with All Dates: For each user, associate them with every date in the date range.

-- Mark Dates with Equity ≥ $10: Identify dates when users have records in equity_value_data.csv (equity ≥ $10).

-- Identify Gaps (Equity < $10): For dates where users have no records (missing dates), infer that their equity was below $10.

-- Detect Consecutive Periods of Equity < $10: Use window functions to find sequences of 28 or more consecutive days where the user had equity < $10 (i.e., missing records).

-- Extract Users Who Have Churned: Select users who have at least one such sequence.

-- Explanation:

-- all_dates CTE:
-- Generates a complete date range from the earliest to the latest date in the equity_value_data.
-- users CTE:
-- Retrieves all distinct user IDs from the dataset.
-- user_dates CTE:
-- Associates each user with every date in the date range.
-- left_joined CTE:
-- Left joins user_dates with equity_value_data to identify dates when users had equity ≥ $10.
-- The below_10 flag is set to 1 for dates where the user has no record (implying equity < $10).
-- consecutive_periods CTE:
-- Calculates row numbers to help identify consecutive sequences.
-- The difference (rn1 - rn2) is constant for consecutive dates with the same below_10 value.
-- grouped_sequences CTE:
-- Groups the data by user and sequence identifier to calculate the length of each sequence where below_10 = 1.
-- Final SELECT:
-- Retrieves user IDs of users who have at least one sequence where below_10 = 1 for 28 or more consecutive days.

-- Note:

-- This query assumes that any date not present in equity_value_data.csv for a user indicates that their equity was below $10 on that date.
-- The use of (rn1 - rn2) as a group identifier allows us to segment the data into consecutive sequences where below_10 is constant.

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
