import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Read the CSV file
nvda_data = pd.read_csv("nvda_stock_data.csv")

# Clean the data
nvda_data['Date'] = pd.to_datetime(nvda_data['Date'])
nvda_data.set_index('Date', inplace=True)
nvda_data.dropna(inplace=True)

# Create lag features (e.g., previous day's close price)
nvda_data['Lag_1'] = nvda_data['Close'].shift(1)
nvda_data['Lag_2'] = nvda_data['Close'].shift(2)
nvda_data['Lag_3'] = nvda_data['Close'].shift(3)
nvda_data['Lag_4'] = nvda_data['Close'].shift(4)
nvda_data.dropna(inplace=True)

# Use 'Lag_1', 'Lag_2', 'Lag_3', and 'Lag_4' as features for the model
X = nvda_data[['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4']]
y = nvda_data['Close']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize the features (important for models like XGBoost)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning for XGBoost (you can also use GridSearchCV)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

# Best model and hyperparameters
best_model = grid_search.best_estimator_
print(f"Best hyperparameters: {grid_search.best_params_}")

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model using RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

# Plot the predictions
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='True Values')
plt.plot(y_test.index, y_pred, label='Predicted Values', color='red')
plt.legend()
plt.show()
