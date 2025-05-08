# ==================== Section 1: Chart Inflation ====================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "bloom data transisition.xlsx"  
xls = pd.ExcelFile(file_path)

inflation_df = xls.parse('Inflation')

inflation_df.dropna(how='all', inplace=True)
inflation_df.reset_index(drop=True, inplace=True)

for i, row in inflation_df.iterrows():
    if 'Date' in row.values:
        inflation_df.columns = inflation_df.iloc[i] 
        inflation_df = inflation_df[i+1:] 
        break


inflation_df = inflation_df.rename(columns={"Date": "Date", "Inflation rate": "Inflation"})
inflation_df = inflation_df[["Date", "Inflation"]]


inflation_df["Date"] = pd.to_datetime(inflation_df["Date"], errors='coerce')
inflation_df["Inflation"] = pd.to_numeric(inflation_df["Inflation"], errors='coerce')


plt.figure(figsize=(12, 6))
sns.lineplot(x=inflation_df["Date"], y=inflation_df["Inflation"], marker='o', linestyle='-')
plt.title("Inflation Rate Trend")
plt.xlabel("Date")
plt.ylabel("Inflation Rate")
plt.grid(True)
plt.show()


# ==================== Section 2: Trend Seasonality Stationarity Analysis ====================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Load the Excel file and Inflation data
file_path = "bloom data transisition.xlsx"  # Ensure correct path
xls = pd.ExcelFile(file_path)
inflation_df = xls.parse('Inflation')

# Clean and format the data
inflation_df.dropna(how='all', inplace=True)
inflation_df.reset_index(drop=True, inplace=True)

# Identify the header row
for i, row in inflation_df.iterrows():
    if 'Date' in row.values:
        inflation_df.columns = inflation_df.iloc[i]  # Set this row as header
        inflation_df = inflation_df[i+1:]  # Remove previous rows
        break

# Rename columns
inflation_df = inflation_df.rename(columns={"Date": "Date", "Inflation rate": "Inflation"})
inflation_df = inflation_df[["Date", "Inflation"]]

# Convert Date column to datetime and Inflation to numeric
inflation_df["Date"] = pd.to_datetime(inflation_df["Date"], errors='coerce')
inflation_df["Inflation"] = pd.to_numeric(inflation_df["Inflation"], errors='coerce')

# Set Date as index
inflation_df.set_index("Date", inplace=True)

# ---- 1. Detecting Trends, Seasonality, and Cycles ----
decomposition = seasonal_decompose(inflation_df["Inflation"], model='additive', period=12)

# Plot the decomposed series
plt.figure(figsize=(12, 8))

plt.subplot(411)
plt.plot(inflation_df["Inflation"], label="Original", color='blue')
plt.legend()

plt.subplot(412)
plt.plot(decomposition.trend, label="Trend", color='red')
plt.legend()

plt.subplot(413)
plt.plot(decomposition.seasonal, label="Seasonality", color='green')
plt.legend()

plt.subplot(414)
plt.plot(decomposition.resid, label="Residuals", color='black')
plt.legend()

plt.tight_layout()
plt.show()

# ---- 2. Stationarity Test (ADF Test) ----
adf_test = adfuller(inflation_df["Inflation"].dropna())
print("Augmented Dickey-Fuller Test Results:")
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")
print("Critical Values:")
for key, value in adf_test[4].items():
    print(f"{key}: {value}")

# Interpretation:
if adf_test[1] < 0.05:
    print("The series is STATIONARY (Reject Null Hypothesis)")
else:
    print("The series is NOT STATIONARY (Fail to Reject Null Hypothesis)")

# Apply first-order differencing
inflation_df["Inflation_diff"] = inflation_df["Inflation"].diff().dropna()

# Re-run the ADF test on the differenced series
adf_test_diff = adfuller(inflation_df["Inflation_diff"].dropna())
print("ADF Test Results After Differencing:")
print(f"ADF Statistic: {adf_test_diff[0]}")
print(f"p-value: {adf_test_diff[1]}")
print("Critical Values:")
for key, value in adf_test_diff[4].items():
    print(f"{key}: {value}")

# Check stationarity after differencing
if adf_test_diff[1] < 0.05:
    print("The series is now STATIONARY (Null Hypothesis Rejected).")
else:
    print("The series is still NOT stationary. Additional transformation may be required.")


# ==================== Section 3: Acf Pacf Arima ====================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the Excel file and Inflation data
file_path = "bloom data transisition.xlsx"  # Ensure correct path
xls = pd.ExcelFile(file_path)
inflation_df = xls.parse('Inflation')

# Clean and format the data
inflation_df.dropna(how='all', inplace=True)
inflation_df.reset_index(drop=True, inplace=True)

# Identify the header row
for i, row in inflation_df.iterrows():
    if 'Date' in row.values:
        inflation_df.columns = inflation_df.iloc[i]  # Set this row as header
        inflation_df = inflation_df[i+1:]  # Remove previous rows
        break

# Rename columns
inflation_df = inflation_df.rename(columns={"Date": "Date", "Inflation rate": "Inflation"})
inflation_df = inflation_df[["Date", "Inflation"]]

# Convert Date column to datetime and Inflation to numeric
inflation_df["Date"] = pd.to_datetime(inflation_df["Date"], errors='coerce')
inflation_df["Inflation"] = pd.to_numeric(inflation_df["Inflation"], errors='coerce')

# Set Date as index
inflation_df.set_index("Date", inplace=True)

# Check if the series is stationary
inflation_df["Inflation_diff"] = inflation_df["Inflation"].diff().dropna()

# Plot ACF and PACF graphs
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Autocorrelation Function (ACF)
plot_acf(inflation_df["Inflation_diff"].dropna(), ax=axes[0])
axes[0].set_title("Autocorrelation Function (ACF)")

# Partial Autocorrelation Function (PACF)
plot_pacf(inflation_df["Inflation_diff"].dropna(), ax=axes[1])
axes[1].set_title("Partial Autocorrelation Function (PACF)")

plt.show()



#ARIMA comparison

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Train ARIMA(1,1,1)
model_111 = ARIMA(inflation_df["Inflation"], order=(1, 1, 1))
fit_111 = model_111.fit()
forecast_111 = fit_111.predict(start=1, end=len(inflation_df), dynamic=False)

# Train ARIMA(2,1,2)
model_212 = ARIMA(inflation_df["Inflation"], order=(2, 1, 2))
fit_212 = model_212.fit()
forecast_212 = fit_212.predict(start=1, end=len(inflation_df), dynamic=False)

# Evaluate models using Mean Absolute Error (MAE)
mae_111 = mean_absolute_error(inflation_df["Inflation"].iloc[1:], forecast_111.iloc[1:])
mae_212 = mean_absolute_error(inflation_df["Inflation"].iloc[1:], forecast_212.iloc[1:])

print(f"MAE for ARIMA(1,1,1): {mae_111}")
print(f"MAE for ARIMA(2,1,2): {mae_212}")

# Plot Actual vs Forecasted Inflation
plt.figure(figsize=(12, 6))
plt.plot(inflation_df.index, inflation_df["Inflation"], label="Actual Inflation", marker='o')
plt.plot(inflation_df.index, forecast_111, label="ARIMA(1,1,1) Forecast", linestyle='dashed', color='red')
plt.plot(inflation_df.index, forecast_212, label="ARIMA(2,1,2) Forecast", linestyle='dashed', color='green')
plt.title("Comparison of ARIMA Models")
plt.xlabel("Date")
plt.ylabel("Inflation Rate")
plt.legend()
plt.grid(True)
plt.show()



#ARIMA
from statsmodels.tsa.arima.model import ARIMA

# Train ARIMA(1,1,1)
model_111 = ARIMA(inflation_df["Inflation"], order=(1, 1, 1))
fit_111 = model_111.fit()

# Print the model summary (characteristics table)
print(fit_111.summary())

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Train ARIMA(1,1,1) model on training data
model_111 = ARIMA(train_data, order=(1, 1, 1))
fit_111 = model_111.fit()

# Generate forecasts for the test period using ARIMA(1,1,1)
forecast_111 = fit_111.forecast(steps=len(test_data))

# Align fitted values with actual training data
fitted_values = fit_111.fittedvalues
train_data_aligned = train_data.iloc[1:]  # Remove first value to match dimensions

# Calculate performance metrics
mse_train = mean_squared_error(train_data_aligned, fitted_values)
mae_train = mean_absolute_error(train_data_aligned, fitted_values)
r2_train = r2_score(train_data_aligned, fitted_values)

mse_test = mean_squared_error(test_data, forecast_111)
mae_test = mean_absolute_error(test_data, forecast_111)
r2_test = r2_score(test_data, forecast_111)

# Print performance metrics
print(f"Training Performance Metrics (ARIMA(1,1,1)):")
print(f"Mean Squared Error (MSE): {mse_train:.4f}")
print(f"Mean Absolute Error (MAE): {mae_train:.4f}")
print(f"R-squared (R²): {r2_train:.4f}\n")

print(f"Testing Performance Metrics (ARIMA(1,1,1)):")
print(f"Mean Squared Error (MSE): {mse_test:.4f}")
print(f"Mean Absolute Error (MAE): {mae_test:.4f}")
print(f"R-squared (R²): {r2_test:.4f}\n")

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))

# Plot training data
plt.plot(train_data.index, train_data, label="Training Data (Actual)", color="blue")

# Plot test data (Actual values)
plt.plot(test_data.index, test_data, label="Test Data (Actual)", color="green")

# Plot ARIMA(1,1,1) forecasted values
plt.plot(test_data.index, forecast_111, label="Test Data (Forecasted - ARIMA(1,1,1))", linestyle='dashed', color="red")

# Formatting
plt.title("ARIMA(1,1,1) - Training vs Test Forecast")
plt.xlabel("Date")
plt.ylabel("Inflation Rate")
plt.legend()
plt.grid(True)

# Show plot
plt.show()

# ==================== Section 4: Arima(1,1,1) ====================

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
file_path = "bloom data transisition.xlsx"  # Ensure correct file path
xls = pd.ExcelFile(file_path)
inflation_df = xls.parse('Inflation')

# Clean and format the data
inflation_df.dropna(how='all', inplace=True)
inflation_df.reset_index(drop=True, inplace=True)

# Identify the header row
for i, row in inflation_df.iterrows():
    if 'Date' in row.values:
        inflation_df.columns = inflation_df.iloc[i]  # Set this row as header
        inflation_df = inflation_df[i+1:]  # Remove previous rows
        break

# Rename columns
inflation_df = inflation_df.rename(columns={"Date": "Date", "Inflation rate": "Inflation"})
inflation_df = inflation_df[["Date", "Inflation"]]

# Convert Date column to datetime and Inflation to numeric
inflation_df["Date"] = pd.to_datetime(inflation_df["Date"], errors='coerce')
inflation_df["Inflation"] = pd.to_numeric(inflation_df["Inflation"], errors='coerce')

# Set Date as index and sort
inflation_df.set_index("Date", inplace=True)
inflation_df = inflation_df.sort_index()

# Define training and testing periods
train_end = "2024-01-31"
test_start = "2024-02-01"
test_end = "2025-02-28"

# Split dataset into training and testing sets
train_data = inflation_df.loc[:train_end, "Inflation"]
test_data = inflation_df.loc[test_start:test_end, "Inflation"]

# Train ARIMA(1,1,1) model
model_111 = ARIMA(train_data, order=(1, 1, 1))
fit_111 = model_111.fit()

# Generate forecasts for the test period
forecast_111 = fit_111.forecast(steps=len(test_data))

# Align fitted values with actual training data (Ensure equal length)
fitted_values = fit_111.fittedvalues
train_data_aligned = train_data.iloc[-len(fitted_values):]  # Match fitted values length

# Calculate performance metrics
mse_train = mean_squared_error(train_data_aligned, fitted_values)
mae_train = mean_absolute_error(train_data_aligned, fitted_values)
r2_train = r2_score(train_data_aligned, fitted_values)

mse_test = mean_squared_error(test_data, forecast_111)
mae_test = mean_absolute_error(test_data, forecast_111)
r2_test = r2_score(test_data, forecast_111)

# Print performance metrics
print(f"Training Performance Metrics (ARIMA(1,1,1)):\n"
      f"Mean Squared Error (MSE): {mse_train:.4f}\n"
      f"Mean Absolute Error (MAE): {mae_train:.4f}\n"
      f"R-squared (R²): {r2_train:.4f}\n")

print(f"Testing Performance Metrics (ARIMA(1,1,1)):\n"
      f"Mean Squared Error (MSE): {mse_test:.4f}\n"
      f"Mean Absolute Error (MAE): {mae_test:.4f}\n"
      f"R-squared (R²): {r2_test:.4f}\n")

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))

# Plot training data
plt.plot(train_data.index, train_data, label="Training Data (Actual)", color="blue")

# Plot test data (Actual values)
plt.plot(test_data.index, test_data, label="Test Data (Actual)", color="green")

# Plot ARIMA(1,1,1) forecasted values
plt.plot(test_data.index, forecast_111, label="Test Data (Forecasted - ARIMA(1,1,1))", linestyle='dashed', color="red")

# Formatting
plt.title("ARIMA(1,1,1) - Training vs Test Forecast")
plt.xlabel("Date")
plt.ylabel("Inflation Rate")
plt.legend()
plt.grid(True)

# Show plot
plt.show()


# ==================== Section 5: Ols ====================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
file_path = "bloom data transisition.xlsx"  
xls = pd.ExcelFile(file_path)
inflation_df = xls.parse('Inflation')

# Clean and format the data
inflation_df.dropna(how='all', inplace=True)
inflation_df.reset_index(drop=True, inplace=True)

# Identify the header row
for i, row in inflation_df.iterrows():
    if 'Date' in row.values:
        inflation_df.columns = inflation_df.iloc[i]  # Set this row as header
        inflation_df = inflation_df[i+1:]  # Remove previous rows
        break

# Rename columns
inflation_df = inflation_df.rename(columns={"Date": "Date", "Inflation rate": "Inflation"})
inflation_df = inflation_df[["Date", "Inflation"]]

# Convert Date column to datetime and Inflation to numeric
inflation_df["Date"] = pd.to_datetime(inflation_df["Date"], errors='coerce')
inflation_df["Inflation"] = pd.to_numeric(inflation_df["Inflation"], errors='coerce')

# Set Date as index and sort
inflation_df.set_index("Date", inplace=True)
inflation_df = inflation_df.sort_index()

# Create lag features (e.g., Inflation at t-1, t-2, t-3)
for lag in range(1, 4):  # Prendre 3 valeurs retardées
    inflation_df[f"Inflation_Lag_{lag}"] = inflation_df["Inflation"].shift(lag)

# Remove NaN values introduced by lagging
inflation_df.dropna(inplace=True)

# Define training and testing periods
train_end = "2024-01-31"
test_start = "2024-02-01"
test_end = "2025-02-28"

# Split dataset into training and testing sets
train_data = inflation_df.loc[:train_end]
test_data = inflation_df.loc[test_start:test_end]

# Define predictors (independent variables) and target variable (Inflation)
X_train = train_data.drop(columns=["Inflation"])  # Features
y_train = train_data["Inflation"]  # Target variable

X_test = test_data.drop(columns=["Inflation"])
y_test = test_data["Inflation"]

# Add constant term for OLS regression
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Train OLS model
model_ols = sm.OLS(y_train, X_train).fit()

# Generate forecasts
y_pred_train = model_ols.predict(X_train)
y_pred_test = model_ols.predict(X_test)

# Calculate performance metrics
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Print model summary
print(model_ols.summary())

# Print performance metrics
print(f"\nTraining Performance Metrics (OLS Regression):\n"
      f"Mean Squared Error (MSE): {mse_train:.4f}\n"
      f"Mean Absolute Error (MAE): {mae_train:.4f}\n"
      f"R-squared (R²): {r2_train:.4f}\n")

print(f"Testing Performance Metrics (OLS Regression):\n"
      f"Mean Squared Error (MSE): {mse_test:.4f}\n"
      f"Mean Absolute Error (MAE): {mae_test:.4f}\n"
      f"R-squared (R²): {r2_test:.4f}\n")

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))

# Plot training data
plt.plot(y_train.index, y_train, label="Training Data (Actual)", color="blue")
plt.plot(y_train.index, y_pred_train, label="Training Data (Predicted)", linestyle='dashed', color="orange")

# Plot test data
plt.plot(y_test.index, y_test, label="Test Data (Actual)", color="green")
plt.plot(y_test.index, y_pred_test, label="Test Data (Predicted)", linestyle='dashed', color="red")

# Formatting
plt.title("OLS Regression - Training vs Test Forecast")
plt.xlabel("Date")
plt.ylabel("Inflation Rate")
plt.legend()
plt.grid(True)

# Show plot
plt.show()


# ==================== Section 6: Correlation Matrix ====================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
file_path = "bloom data transisition.xlsx"
xls = pd.ExcelFile(file_path)

# Load data from the "ALL DATA" sheet
df = pd.read_excel(xls, sheet_name="ALL DATA", skiprows=1)  # Skip one row to avoid empty headers

# Properly rename columns by selecting the correct header row
df.columns = df.iloc[4]  # Use the 5th row as the actual header
df = df[5:].reset_index(drop=True)  # Remove unnecessary rows

# Convert columns to numeric format (except the date)
df = df.apply(pd.to_numeric, errors='ignore')

# Identify the date column (assuming it's the first column)
date_column = df.columns[0]  # Adjust if needed

# Drop the "Inflation" column and the "Date" column if they exist
df_filtered = df.drop(columns=["Inflation", date_column], errors="ignore")

# Compute the correlation matrix
corr_matrix = df_filtered.corr()

# Display the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix of Macroeconomic Variables")
plt.show()


# ==================== Section 7: Xgboost ====================

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Load the Excel file
file_path = "bloom data transisition - XGBoost.xlsx"
xls = pd.ExcelFile(file_path)

# Load data from the "ALL DATA" sheet
df = pd.read_excel(xls, sheet_name="ALL DATA", skiprows=1)  # Skip one row to avoid empty headers

# Rename columns correctly by selecting the appropriate header row
df.columns = df.iloc[4]  # Use the 5th row as the actual header
df = df[5:].reset_index(drop=True)  # Remove unnecessary rows

# Convert columns to numeric format (except the date)
df = df.apply(pd.to_numeric, errors='ignore')

# Identify the "Date" column (assumed to be the first column)
date_column = df.columns[0]
print(f"Detected date column: {date_column}")

# Convert Date column to datetime format
df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

# Sort data by date
df = df.sort_values(by=[date_column]).reset_index(drop=True)

# Define the target variable (inflation)
target_column = "Inflation"
if target_column not in df.columns:
    raise ValueError("The 'Inflation' column was not found in the dataset.")

# Exclude highly correlated variables (based on previous analysis)
variables_to_drop = ["M1 ECB", "M2 ECB", "DOW", "France CPI YoY Tobacco", date_column]  # Excluding date column
df_final = df.drop(columns=variables_to_drop, errors="ignore")

# Separate features (X) and target (y)
X = df_final.drop(columns=[target_column], errors="ignore")
y = df_final[target_column]

# Ensure all columns are numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Handle missing values
X = X.fillna(X.median())

# Reset index before splitting
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Split into train/test while preserving chronological order
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reset index for y_test to match indices
y_test = y_test.reset_index(drop=True)

# Extract actual dates corresponding to test values
dates_test = df[date_column].iloc[-len(y_test):].reset_index(drop=True)

# Train the XGBoost model
model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=500, 
    learning_rate=0.01, 
    max_depth=3, 
    subsample=0.6, 
    colsample_bytree=1,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print(f"\U0001F4CA XGBoost Model Performance:")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# Plot predictions vs actual values with real dates on the X-axis
plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test.values, label="Actual Values", linestyle='dashed')
plt.plot(dates_test, y_pred, label="XGBoost Predictions", color="orange")
plt.legend()
plt.title("Comparison of Predictions vs Actual Inflation - XGBoost")
plt.xlabel("Time")
plt.ylabel("Inflation")
plt.xticks(rotation=45)  # Rotate dates for better readability
plt.show()

# Grid Search for Hyperparameter Tuning (commented out for now)
# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'n_estimators': [500, 1000],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0]
# }

# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
# grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
# grid_search.fit(X_train, y_train)

# print("Best parameters :", grid_search.best_params_)

# ==================== Section 8: Random Forest ====================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ---- 1.Load data ----
file_path = "bloom data transisition.xlsx"
xls = pd.ExcelFile(file_path)

# Load inflation data
inflation_df = xls.parse('Inflation')

# Load macroeconomic variables
macro_df = xls.parse('ALL DATA')

# ---- 2. Data preprocessing ----
# Clean inflation and macro data
inflation_df.dropna(how='all', inplace=True)
inflation_df.reset_index(drop=True, inplace=True)

macro_df.dropna(how='all', inplace=True)
macro_df.reset_index(drop=True, inplace=True)

# Identify the header row and reassign
for i, row in inflation_df.iterrows():
    if 'Date' in row.values:
        inflation_df.columns = inflation_df.iloc[i]
        inflation_df = inflation_df[i+1:]
        break

for i, row in macro_df.iterrows():
    if 'Date' in row.values:
        macro_df.columns = macro_df.iloc[i]
        macro_df = macro_df[i+1:]
        break

# Convert dates and merge datasets
inflation_df["Date"] = pd.to_datetime(inflation_df["Date"], errors='coerce')
macro_df["Date"] = pd.to_datetime(macro_df["Date"], errors='coerce')
inflation_df["Inflation"] = pd.to_numeric(inflation_df["Inflation"], errors='coerce')

df = pd.merge(inflation_df, macro_df, on="Date", how="inner")

# ---- 3. Create lag variables ----
for lag in range(1, 4):
    df[f"Inflation_Lag_{lag}"] = df["Inflation"].shift(lag)

df.dropna(inplace=True)

# ---- 4. Train-test split ----
train_end = "2022-12-31"
test_start = "2023-01-01"

train_data = df[df["Date"] <= train_end]
test_data = df[df["Date"] > train_end]

X_train = train_data.drop(columns=["Inflation", "Date"])
y_train = train_data["Inflation"]

X_test = test_data.drop(columns=["Inflation", "Date"])
y_test = test_data["Inflation"]

# ---- 5. Train the Random Forest model ----
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Generate predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# ---- 6. Evaluate model performance ----
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Print performance metrics
print(f"\n Training Performance Metrics (Random Forest):\n"
      f"Mean Squared Error (MSE): {mse_train:.4f}\n"
      f"Mean Absolute Error (MAE): {mae_train:.4f}\n"
      f"R-squared (R²): {r2_train:.4f}\n")

print(f"\n Testing Performance Metrics (Random Forest):\n"
      f"Mean Squared Error (MSE): {mse_test:.4f}\n"
      f"Mean Absolute Error (MAE): {mae_test:.4f}\n"
      f"R-squared (R²): {r2_test:.4f}\n")

# ---- 7. Visualization ----
plt.figure(figsize=(12, 6))

plt.plot(y_train.index, y_train, label="Training Data (Actual)", color="blue")
plt.plot(y_train.index, y_pred_train, label="Training Data (Predicted)", linestyle='dashed', color="orange")

plt.plot(y_test.index, y_test, label="Test Data (Actual)", color="green")
plt.plot(y_test.index, y_pred_test, label="Test Data (Predicted)", linestyle='dashed', color="red")

plt.title("Random Forest - Training vs Test Forecast")
plt.xlabel("Date")
plt.ylabel("Inflation Rate")
plt.legend()
plt.grid(True)
plt.show()
