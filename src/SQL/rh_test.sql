-- -- Creating a new column streak_id as a way to identify consecutive days. Note: This assumes no duplication in dates per user. Need to validate.
-- WITH temp1 AS (
--     SELECT
--         *,
--         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp) AS rn,
--         timestamp::date - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp)::int AS streak_id
--     FROM
--         equity_value_data
-- ),
-- -- Creating the start and end dates for each conseuctive streak of $10 for each user and how long each streak is.
-- temp2 AS (
--     SELECT
--         user_id,
--         MIN(timestamp::date) AS start_streak_date,
--         MAX(timestamp::date) AS end_streak_date,
--         COUNT(*) AS duration_of_above10_streak
--     FROM
--         temp1
--     GROUP BY
--         user_id, streak_id
-- ),
-- -- Creating the last end date range of consecutive streaks for the start of each consecutive streak.
-- -- Brainstorming: I'm wondering for users who stared with less than 10$, using that date and the first time they had a streak of $10+, which wuould be the minimum, and finding the day difference??????
-- temp3 AS (
--     SELECT
--         *,
--         LAG(end_streak_date) OVER (PARTITION BY user_id ORDER BY start_streak_date ASC) AS prev_above10_streak_date,
--         start_streak_date - LAG(end_streak_date) OVER (PARTITION BY user_id ORDER BY start_streak_date ASC) AS duration_between_above10_streaks
--     FROM
--         temp2
-- ),
-- -- Filtering for users who had 28+ gap days between the end of a user's $10 streak and the start of a user's $10 streak.
-- temp4 AS (
--     SELECT *
--     FROM temp3
--     WHERE duration_between_above10_streaks >= 28
-- ),
-- -- Getting all unique users who had 28+ gap days of less than $10.
-- temp5 AS (
--     SELECT DISTINCT user_id
--     FROM temp4
-- ),
-- -- Getting user's first and last 10+$ streak ranges
-- temp6 AS (
--     SELECT
--         user_id,
--         MIN(timestamp::date) AS first_above10_streak,
--         MAX(timestamp::date) AS last_above10_streak
--     FROM
--         equity_value_data
--     WHERE user_id IN (SELECT user_id FROM temp5)
--     GROUP BY user_id
-- ),
-- -- Joining user's first/last login dates to user's who had 28+gap days between 10+ streaks.
-- -- Note: What if user has 1 streak? Need to think about this more. (I think i need to make sure the start date equity balance for
-- -- each user was above $10 to begin with?)
-- temp7 AS (
--     SELECT
--         t1.user_id,
--         start_streak_date,
--         end_streak_date,
--         duration_of_above10_streak,
--         prev_above10_streak_date,
--         duration_between_above10_streaks,
--         first_above10_streak,
--         last_above10_streak
--     FROM
--         temp6 t1
--     RIGHT JOIN temp4 t2 ON t1.user_id = t2.user_id
-- ),
-- -- Filtering the last time range a user had 28+ gap days of less than $10. In other words, last consecutive 10+ streak the user had above $10.
-- -- Note/Question: What if I joined robinhood. I started off with 11 dollars. I have $11 dollars for 5 days straight. So streak is 10 days. Then I lose all my money. I never contribute back. Where/what am I?
-- -- ^ BrainStroming above: If I
-- temp8 AS (
--     SELECT
--         user_id,
--         start_streak_date,
--         end_streak_date,
--         duration_of_above10_streak,
--         prev_above10_streak_date,
--         duration_between_above10_streaks,
--         first_above10_streak,
--         last_above10_streak
--     FROM (
--         SELECT
--             *,
--             row_number() OVER (PARTITION BY user_id ORDER BY start_streak_date DESC) AS ranking
--         FROM temp7) t1
--     WHERE ranking = 1
-- )
-- SELECT * FROM TEMP8
-- -- -- Filtering for users  whose last time being aboved $10 happens to be your last consuective 10$streak??
-- -- SELECT * FROM temp8 WHERE start_streak_date=last_above10_streak
