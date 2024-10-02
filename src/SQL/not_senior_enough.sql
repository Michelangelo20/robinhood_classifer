with temp1 as (
select
*,
row_number() over(partition by user_id order by timestamp) as rn,
timestamp::date - row_number() over(partition by user_id order by timestamp)::int as streak_id
from
equity_value_data
),
temp2 as (
select
user_id,
min(timestamp::date) as min_date,
max(timestamp::date) as max_date,
count(*) as duration_of_above10_streak
from temp1
group by
user_id,
streak_id
),
temp3 as (
select
*,
lag(max_date) over (partition by user_id order by min_date asc) as last_streak_date,
min_date - lag(max_date) over (partition by user_id order by min_date asc) as duration_between_above10_streaks
from
temp2
),
temp4 as (
select
*
from
temp3
where duration_between_above10_streaks >= 28
)
select
concat(round((count(distinct user_id)*1.0/(select count (distinct user_id) from equity_value_data))*100,3),'%')
from temp4