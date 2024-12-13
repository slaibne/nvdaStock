**Stock Price Prediction using LSTM with Technical Indicators**

This project uses a Long Short-Term Memory (LSTM) neural network model to predict the stock prices of NVIDIA (NVDA) using historical stock data and technical indicators. It demonstrates how to preprocess stock data, calculate technical indicators like Moving Averages (MA), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD), and then use these features to predict future stock prices.

The goal of this project is to use historical stock data to predict future stock prices of NVIDIA (NVDA) over a 10-day period. The LSTM model uses multiple technical indicators as input features to learn from past stock price movements and predict the closing price for the next few days.

The model is trained on past stock data, and technical indicators like Moving Averages, Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD) are used as input features for the prediction.

Key Features:
Data Preprocessing: Clean and preprocess historical stock data by calculating key technical indicators.

Feature Engineering: Generate technical indicators like MA50, MA200, RSI, MACD, and MACD Signal.

LSTM Model: Implement an LSTM model to predict the future stock prices based on the calculated features.

Performance Evaluation: Evaluate the model using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

Prediction: Forecast the stock prices for the next 10 days and save the results to a CSV file.

Technologies Used
  
  Python: Programming language used for the project.
  
  TensorFlow/Keras: Deep learning framework used for building the LSTM model.
  
  Pandas: Library for data manipulation and analysis.
  
  NumPy: Library for numerical computations.
  
  Matplotlib: Library for plotting graphs and visualizations.
  
  Scikit-learn: Library for preprocessing and evaluation metrics.
  

Model Architecture
The model uses a 2-layer LSTM architecture with the following layers:


LSTM Layer 1: 100 units, return sequences to the next LSTM layer.

Dropout Layer 1: 20% dropout to prevent overfitting.

LSTM Layer 2: 100 units, does not return sequences as it is the final LSTM layer.

Dropout Layer 2: 20% dropout to prevent overfitting.

Dense Layer: Single unit to predict the closing price.
The model is compiled using the Adam optimizer and a mean squared error (MSE) loss function. It is trained for 100 epochs.

Results
After training, the model will output the following metrics:


Mean Squared Error (MSE): Measures the average squared difference between the predicted and actual stock prices.

Root Mean Squared Error (RMSE): The square root of MSE, providing an estimate of the error in the same units as the stock price.

Mean Absolute Error (MAE): The average of absolute errors between the predicted and actual stock prices.

The predicted stock prices for the next 10 days will be plotted alongside the actual stock prices.


Contributing
Contributions are welcome! If you'd like to improve this project, feel free to fork the repository, make changes, and submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.
