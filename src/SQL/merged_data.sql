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
FROM ds.features_data f
LEFT JOIN ds.equity_value_data e on f.user_id=e.user_id
