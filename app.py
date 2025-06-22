# Data Exploration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/masked_kiva_loans.csv')
df.head()

df.shape

df.info()

df.dtypes

df.describe()

# Data Preprocessing

#adjust date column type to date time
df['date'] = pd.to_datetime(df['date'])

## Handle Missing Values & duplicates

df.isna().sum()

df['partner_id'].fillna(-1, inplace=True)

df['borrower_genders'].value_counts(dropna=False)

def normalize_gender(genders):
    if isinstance(genders, str):
        unique_genders = set(g.strip() for g in genders.split(','))
        if unique_genders == {'female'}:
            return 'female'
        elif unique_genders == {'male'}:
            return 'male'
        else:
            return 'mixed'
    else:
        return 'unknown' # for nulls
df['gender_category'] = df['borrower_genders'].apply(normalize_gender)

df['gender_category'].value_counts()

df.drop('borrower_genders', axis=1, inplace=True)

df.duplicated().sum()

df.isna().sum()

df.info()

## Handle Outliers

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, lower, upper

numeric_cols = ['funded_amount', 'loan_amount', 'term_in_months', 'lender_count']
for col in numeric_cols:
    outliers, _, _ = detect_outliers_iqr(df, col)
    print(f"{col}: {len(outliers)} outliers")

plt.figure(figsize=(12, 8))

for i, column in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=df[column], color='lightgreen')
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)
plt.tight_layout()
plt.show()

def handling_outliers(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)

for col in numeric_cols:
    handling_outliers(col)

df.to_csv('Cleaned_masked_kiva_loans.csv', index=False)

# Data visualization

## Categorical Data

df['sector'].value_counts().plot(kind='bar')

df['country'].value_counts().head(5).plot(kind='bar', color='skyblue', figsize=(10, 6))
plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Top 5 Countries by Loan Count')
plt.xticks(rotation=45, ha='right')
plt.show()

df['repayment_interval'].value_counts().plot(kind='bar')

df['gender_category'].value_counts().plot(kind='bar')

## Continous Data

plt.figure(figsize=(12, 8))
for i, column in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=df[column], color='lightgreen')
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
for i, column in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[column], kde=True, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.title('Correlation Heatmap of Numeric Columns')
plt.show()

plt.figure(figsize=(12, 8))
for i, col1 in enumerate(numeric_cols):
    for j, col2 in enumerate(numeric_cols):
        if i < j:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=df[col1], y=df[col2], alpha=0.6, color='blue')
            plt.title(f'Scatter Plot: {col1} vs {col2}')
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.show()

# Machine learning

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

## Data Preparation

numerical_features = ['loan_amount', 'term_in_months', 'lender_count']
categorical_features = ['sector', 'country', 'repayment_interval', 'gender_category']

X_num = df[numerical_features]
X_cat = df[categorical_features]
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat_encoded = encoder.fit_transform(X_cat)
X_processed = np.hstack((X_num_scaled, X_cat_encoded))
y=df['funded_amount']

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

## Linear Regression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
print("Linear Regression Results:")
print(f"MSE: {mean_squared_error(y_test, y_pred_lin):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lin)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lin):.2f}")
print(f"R2: {r2_score(y_test, y_pred_lin):.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lin, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Funded Amount')
plt.ylabel('Predicted Funded Amount')
plt.title('Actual vs. Predicted (Linear Regression)')
plt.grid(True)
plt.show()

## Decision Tree Regressor


tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)
y_pred_tree = tree_reg.predict(X_test)
print("Decision Tree Results:")
print(f"MSE: {mean_squared_error(y_test, y_pred_tree):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_tree)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_tree):.2f}")
print(f"R2: {r2_score(y_test, y_pred_tree):.2f}")


from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(
    tree_reg,
    feature_names=numerical_features + list(encoder.get_feature_names_out()),
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=2
)
plt.title("Decision Tree")
plt.show()

importances = tree_reg.feature_importances_
feature_names = numerical_features + list(encoder.get_feature_names_out())
sorted_idx = np.argsort(importances)[::-1]
top_features = 5
plt.figure(figsize=(10, 6))
plt.barh(
    np.array(feature_names)[sorted_idx][:top_features],
    importances[sorted_idx][:top_features],
    color='skyblue'
)
plt.xlabel("Feature Importance (by Gini)")
plt.title("Top Features Influencing Funded Amount")
plt.gca().invert_yaxis()
plt.grid(True, axis='x')
plt.show()

## Random Forest Regressor


model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred_rand = model.predict(X_test)
print("Random Forest Results:")
print(f"MSE: {mean_squared_error(y_test, y_pred_rand):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rand)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rand):.2f}")
print(f"R2: {r2_score(y_test, y_pred_rand):.2f}")

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_rand, alpha=0.4)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', linewidth=2)
plt.xlabel("Actual Funded Amount")
plt.ylabel("Predicted Funded Amount")
plt.title("Random Forest: Actual vs Predicted")
plt.grid(True)
plt.show()

## Random Forest Regressor is the best one

# Analytical Questions

## Which sectors receive the highest amount of funding, and how does this change over time?

sector_funding = df.groupby('sector')['funded_amount'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sector_funding.plot(kind='bar', color='skyblue')
plt.title('Total Funded Amount by Sector')
plt.ylabel('Total Funding')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df['year_month'] = df['date'].dt.to_period('M')
sector_trends = df.groupby(['year_month', 'sector'])['funded_amount'].sum().unstack()
#top 5 sectors
top_sectors = sector_funding.head(5).index
sector_trends[top_sectors].plot(figsize=(14, 7))
plt.title('Monthly Funding Trends by Sector (Top 5)')
plt.ylabel('Funded Amount')
plt.xlabel('Date')
plt.grid(True)
plt.tight_layout()
plt.show()

## Is there a correlation between the number of lenders and the funded loan amount?

plt.figure(figsize=(10, 6))
sns.regplot(x='lender_count', y='funded_amount', data=df,
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Correlation Between Lender Count and Funded Amount')
plt.xlabel('Number of Lenders')
plt.ylabel('Funded Amount')
plt.show()

correlation = df[['lender_count', 'funded_amount']].corr().iloc[0,1]
print(f"Pearson Correlation Coefficient: {correlation:.2f}")

#  Time Series

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

arima_data =df[['funded_amount', 'date']].copy()
arima_data['date'] =pd.to_datetime(arima_data['date'])
arima_data.head()

Time_series = arima_data.groupby('date')['funded_amount'].sum().reset_index()

Time_series.set_index('date', inplace=True)

plt.figure(figsize=(20, 6))
plt.plot(Time_series.index, Time_series['funded_amount'], label='funded_amount', color='blue')
plt.title('Funded Amount Time Series')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.legend(loc='best')
plt.grid(True)
plt.show()

#Time series data by month
Time_series_monthly = Time_series.resample('M').sum()
plt.figure(figsize=(20, 6))
plt.plot(Time_series_monthly.index.strftime('%Y-%m'), Time_series_monthly['funded_amount'], marker='o', linestyle='-', color='blue')
plt.title('Funded Amount Time Series (Aggregated by Month)')
plt.xlabel('Month (Year)')
plt.ylabel('Total Amount')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


def stationarize_series(series):
    rolling_mean = series.rolling(window=12).mean()
    rolling_std = series.rolling(window=12).std()
    plt.figure(figsize=(20, 6))
    plt.plot(series, label='Original', color='blue')
    plt.plot(rolling_mean, label='Rolling Mean', color='red')
    plt.plot(rolling_std, label='Rolling Std', color='green')
    plt.title('Rolling Mean & Standard Deviation')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.ylabel('funded amount')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
stationarize_series( Time_series['funded_amount'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
plot_acf(Time_series.funded_amount.dropna(), ax=ax1, lags=20, title='Autocorrelation Function (ACF)')
plot_pacf(Time_series.funded_amount.dropna(), ax=ax2, lags=20, title='Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

train_size = int(len(Time_series) * 0.8)
train, test = Time_series[:train_size], Time_series[train_size:]

model = ARIMA(train['funded_amount'], order=(14,0,14))
arima_model = model.fit()

start = len(train)
end = len(train) + len(test) - 1
pred = arima_model.predict(start=start, end=end,typ = 'levels')
pred.index= Time_series.index[start:end+1]
print(pred)

pred.plot(figsize=(18,8),legend = True)
test['funded_amount'].plot(legend = True )

r2 = r2_score(test['funded_amount'], pred)
print(f"R-squared (R2): {r2}")

start_date = '2017-07-21'
end_date = '2018-08-21'

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

forecast = arima_model.forecast(steps=len(pd.date_range(start=start_date, end=end_date, freq=Time_series.index.freq)))
forecast_index = pd.date_range(start=start_date, end=end_date, freq=Time_series.index.freq)

plt.figure(figsize=(18, 8))
Time_series['funded_amount'].plot(label='Original Data')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.legend()
plt.show()
