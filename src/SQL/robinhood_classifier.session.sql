-- No mismatches across user_ids for features and equity tables
SELECT
*
FROM features f
LEFT JOIN equity e on f.user_id=e.user_id
WHERE e.user_id is null
-- Find all matches across user_ids for features and equity tables
SELECT
e.user_id,
risk_tolerance,
investment_Experience,
time_horizon,
platform,
time_spent,
first_deposit_amount,
instrument_type_first_traded,
e.close_equity,
e.timestamp
FROM features f
LEFT JOIN equity e on f.user_id=e.user_id