import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Read the dataset
df = pd.read_csv("nvda_stock_data.csv")

# Preprocess the data
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Use 'Close' column for prediction
data = df[['Close']].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create data for LSTM model
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Set the time step for LSTM
time_step = 60

# Create dataset for training and testing
X, y = create_dataset(scaled_data, time_step)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size

X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Reshape input data to be 3D [samples, time steps, features] for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Prepare for forecasting the next 10 days
last_data = scaled_data[-time_step:]  # Get the last 'time_step' values to start the prediction
predicted_values = []

# Loop to predict the next 10 days
for _ in range(10):
    last_data = last_data.reshape((1, time_step, 1))  # Reshape for LSTM input
    prediction = model.predict(last_data)  # Make the prediction
    predicted_values.append(prediction[0, 0])  # Append the predicted value

    # Update last_data with the new prediction
    last_data = np.append(last_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

# Invert scaling for the predictions
predicted_values = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))

# Create future dates for the next 10 dayspu
last_date = df.index[-1]
future_dates = pd.date_range(last_date, periods=11, freq='B')[1:]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], color='blue', label='True Stock Price')
plt.plot(future_dates, predicted_values, color='red', label='Predicted Stock Price for next 10 days')
plt.title('Stock Price Prediction for Next 10 Days using LSTM')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.xticks(rotation=45)  # Rotate dates for better readability
plt.show()
