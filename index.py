import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import GridSearchCV

# Read the CSV file
nvda_data = pd.read_csv("nvda_stock_data.csv")

# Check the first few rows of data to confirm structure
print(nvda_data.head())

# Remove rows that contain 'Ticker' or other non-date values
nvda_data = nvda_data[~nvda_data['Date'].str.contains("Ticker", na=False)]
nvda_data = nvda_data[~nvda_data['Date'].str.contains("Date", na=False)]  # Remove header-like rows

# Convert the 'Date' column to datetime, specifying the format if needed
nvda_data['Date'] = pd.to_datetime(nvda_data['Date'], errors='coerce')  # 'coerce' will turn invalid dates into NaT

# Drop rows with invalid dates (NaT)
nvda_data.dropna(subset=['Date'], inplace=True)

# Ensure the numeric columns ('Open', 'Close') are actually numeric
nvda_data['Close'] = pd.to_numeric(nvda_data['Close'], errors='coerce')
nvda_data['Open'] = pd.to_numeric(nvda_data['Open'], errors='coerce')

# Drop any rows with NaN or Inf values after conversion
nvda_data = nvda_data.dropna(subset=['Open', 'Close'])
nvda_data = nvda_data[~np.isinf(nvda_data['Close'])]
nvda_data = nvda_data[~np.isinf(nvda_data['Open'])]

# Set 'Date' as the index
nvda_data.set_index('Date', inplace=True)

# Step 1: ADF Test for Stationarity
def adf_test(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    return result[1]

# Perform ADF test on 'Close' price data
p_value = adf_test(nvda_data['Close'])

# If p-value >= 0.05, data is non-stationary, so we apply differencing
if p_value >= 0.05:
    print("NVDA is NOT stationary (p-value >= 0.05). Differencing is required.")

    # Apply first-order differencing
    nvda_data['Close'] = nvda_data['Close'].diff().dropna()  # Apply differencing

    # Check for NaN or Inf values after differencing
    if nvda_data['Close'].isna().any() or np.isinf(nvda_data['Close']).any():
        print("Warning: NaN or Inf values detected after differencing. Dropping NaN and Inf values.")
        nvda_data = nvda_data.dropna(subset=['Close'])
        nvda_data = nvda_data[~np.isinf(nvda_data['Close'])]

    # Recheck the stationarity of the differenced data
    p_value = adf_test(nvda_data['Close'])

    if p_value >= 0.05:
        print("Differenced data is still non-stationary. Applying log transformation for stabilization.")
        # Apply log transformation if necessary (only if still non-stationary after differencing)
        nvda_data['Close'] = np.log(nvda_data['Close'])
        nvda_data = nvda_data.dropna(subset=['Close'])  # Drop NaN values after log transformation

        # Check for NaN or Inf values after log transformation
        if nvda_data['Close'].isna().any() or np.isinf(nvda_data['Close']).any():
            print("Warning: NaN or Inf values detected after log transformation. Dropping NaN and Inf values.")
            nvda_data = nvda_data.dropna(subset=['Close'])
            nvda_data = nvda_data[~np.isinf(nvda_data['Close'])]

        p_value = adf_test(nvda_data['Close'])

        if p_value >= 0.05:
            print("Data is still non-stationary after log transformation. Seasonal differencing might be required.")
            # If seasonal differencing is needed (e.g., monthly data with seasonal patterns)
            # We can apply seasonal differencing
            nvda_data['Close'] = nvda_data['Close'].diff(30).dropna()  # Seasonal differencing, adjust the period as needed
            p_value = adf_test(nvda_data['Close'])

# Ensure the date index has a frequency set
nvda_data = nvda_data.asfreq('D', method='ffill')  # 'D' for daily frequency, adjust if necessary for your data

# Step 2: ARIMA Grid Search Setup (ARIMA p, d, q)
# Set the range for grid search
p = d = q = range(0, 3)
param_grid = {'p': p, 'd': d, 'q': q}

# Step 3: Fit ARIMA Model Using GridSearchCV
def arima_grid_search(data, param_grid):
    best_score = float("inf")
    best_params = None
    best_model = None

    for p_val in param_grid['p']:
        for d_val in param_grid['d']:
            for q_val in param_grid['q']:
                try:
                    model = ARIMA(data, order=(p_val, d_val, q_val))
                    model_fit = model.fit()
                    score = model_fit.aic  # Use AIC for model selection
                    if score < best_score:
                        best_score = score
                        best_params = (p_val, d_val, q_val)
                        best_model = model_fit
                except Exception as e:
                    print(f"Error with parameters (p={p_val}, d={d_val}, q={q_val}): {e}")

    return best_model, best_params

# Perform Grid Search for ARIMA parameters
best_model, best_params = arima_grid_search(nvda_data['Close'], param_grid)

# Display best parameters and the AIC score
print(f"Best ARIMA model: ARIMA{best_params}")
print(f"Best AIC score: {best_model.aic}")

# Step 4: Forecasting with the Best Model
forecast_steps = 10  # Forecasting the next 10 days
forecast = best_model.forecast(steps=forecast_steps)

# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(nvda_data.index, nvda_data['Close'], label='Historical Data')
plt.plot(pd.date_range(start=nvda_data.index[-1], periods=forecast_steps + 1, freq='D')[1:], forecast, label='Forecast', color='red')
plt.legend()
plt.show()
