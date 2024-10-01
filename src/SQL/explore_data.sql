SELECT
    user_id,
	close_equity,
    timestamp,
    TO_CHAR(timestamp::timestamp, 'day') AS day_of_week
FROM
    equity_value_data evd
order by
1,3