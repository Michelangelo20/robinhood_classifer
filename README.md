# Robinhood-OA-Classifer
This code repository is for the Robinhood Data Science OA

### Problem:

In order to improve user retention and lower churn, the growth team at Robinhood is interested in understanding why and which users withdraw money from their Robinhood account.  A user is considered churn when their equity value (amount of money in Robinhood account) falls below $10 for a period of 28 consecutive calendar days or longer.

### Data:

features_data.csv - contains user level data such as:
- user_id - unique id for every user
- risk_tolerance - self-reported risk tolerance of the user
- investment_experience - self-reported liquidity needs of the user
- time_horizon - self-reported investment time horizon of the user
- platform - which platform (iOS or Android) the user is on
- time_spent - amount of time spent on the app
- first_deposity_amount - $ value of the amount first deposited
- instrument_type_first_traded - type of instrument first traded


equity_value_data.csv - contains user_id and equity_value for user along with timestamps for days when the user's equity value is greater than or equal to $10.

### Questions:

* Q1. What percentage of users have churned in the data provided?
* Q2. Build a classifier that given a user with their features assigns a churn probability for each user and predicts which users will churn. Based on the classifier output classify each user in the dataset as churned or not churned. What's the AUC score of your test set?
* Q3. What's the distribution of the given feature that has the highest correlation with churn (Right/left skewed, normal, multinomial distributions)
* Q4. What are the top 3 features? List the most important features that correlate to user churn

### Getting Started:
Check out this project repo, then run this command:
```
pc: python -m venv venv
macOS: python3 -m venv venv
```
In the IDE terminal run command:
```
pc: venv\scripts\activate
macOS: source venv/bin/activate
```
While the virutal environment is running, run the following command:
```
pip install -r requirements.txt
```
 Packages include:
```
polars - parallel distribution dataframe library
pandas - single distribution dataframe library
numpy - matrix library
pytorch - neural network library
scikit-learn - standard ml library
matplotlib - visualization library
seaborn - visualization library
imbalanced-learn - handling imabalnce dataset library
xgboost - gradient boosting library
lightgbm - gradient boosting library
mlflow - manage ml lifecycle platform
pyspark - distributed computing framework
shap - model interpretability library
optuna - hyperparameter optimization library
featuretools - automated feature engineering library
```
After the installs complete, run any notebook

### DB Setup:
* [sqlLite](https://www.sqlite.org/download.html) - links to SQLite database
* [DBeaver](https://dbeaver.io/) - links to SQL IDE