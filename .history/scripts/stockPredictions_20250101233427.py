from flask import Flask
from flask_cors import CORS  # Import CORS
import yfinance as yf
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from se

from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def ml_get_historical(ticker):
    try:
        # Create a Ticker object
        stock = yf.Ticker(ticker)

        # Define the date range
        end_date = datetime.today()
        start_date = end_date - timedelta(days=1800)

        # Fetch historical data
        historical_data = stock.history(start=start_date, end=end_date)
        # historical_data = stock.history(period="max")

        # Reset the index to bring Date as a column
        historical_data.reset_index(inplace=True)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.date

        # Keep only necessary columns
        columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
        
        if set(columns).issubset(historical_data.columns):
            historical_data = historical_data[columns]
        else:
            raise Exception("Missing expected columns in the fetched data.")

        historical_data["Percent Change"] = historical_data["Close"].pct_change() * 100

        historical_data = historical_data.iloc[1:].reset_index(drop=True)

        return historical_data

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None



def ml_to_predict(ticker_symbol):
    test = False
    # stock_data = ml_get_historical(ticker_symbol)

    stock_data = ml_get_historical(ticker_symbol)

    # series represents all the stock prices (each value is the price for that day)
    # plt.plot(stock_data.index, stock_data['Close'])
    series = stock_data['Close'].values.reshape(-1, 1) # percent change or close or sentiment
    # print(series)

    # Normalize the training data to make it easier for model to work with
    scaler = StandardScaler()
    scaler.fit(series[:])
    series = scaler.transform(series).flatten()

    T = 200  # Past values (input sequence length)
    N = 50   # Future values to predict (output sequence length)


    # Build the dataset
    X = []
    Y = []
    for t in range(0, len(series) - T - N + 1, 1):  # Sliding window with step 1
        x = series[t:t+T]  # Get previous 100 values
        X.append(x)
        
        y = series[t+T:t+T+N]  # Get next 10 values
        Y.append(y)

    X = np.array(X).reshape(-1, T, 1)  # Shape should be (Num of Dates, T, 1)
    Y = np.array(Y)  # Shape should be (Num of Dates, N)

    print(" INPUT X.shape", X.shape, "OUTPUT TARGETS Y.shape", Y.shape)


    # Build the LSTM model

    i = Input(shape=(T, 1))
    x = LSTM(100)(i)  # More units for better learning
    x = Dense(N)(x)  # Predict N values (future prices)
    model = Model(i, x)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))


    # Split data into training and validation
    train_size = int(len(X) * 0.6)  # Use ?% of data for training, rest % for validation
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_val, Y_val = X[train_size:], Y[train_size:]

    # Train the model
    r = model.fit(
        X_train, Y_train,  # Use training data
        epochs=10,
        validation_data=(X_val, Y_val)  # Use validation data
    )

    if test:
        # Plot loss per iteration
        plt.plot(r.history['loss'], label='loss')
        plt.plot(r.history['val_loss'], label='val_loss')
        plt.legend()

    # Predict the next 10 values for each sliding window on validation data
    # Make predictions for non-overlapping windows/points
    predictions = []
    time_indices = []

    # Loop with step T + N over validation set

    # Example:
    # T = 200  # Past values (input sequence length)
    # N = 50   # Future values to predict (output sequence length)
    # if len(X) = 1000 --> Number if samples/dates

    # create new input list that includes all possible windows of size T, including ones that dont have validation set

    P = []
    for t in range(0, len(series) - T + 1, 1):  # Sliding window with step 1
        nums = series[t:t+T]  # Get previous 100 values
        P.append(nums)

    P = np.array(P).reshape(-1, T, 1)  # Shape should be (Num of Dates, T, 1)

    for t in range(len(P) - 1, -1, -(T + N)):  # Start from the most recent data
        if t >= 0:  # Ensure there are enough points for a full input sequence
            x_current = P[t:t+1]  # Input sequence
            y_pred = model.predict(x_current)
            predictions.append(y_pred)
            time_indices.append(t + T)  # Record the start time of prediction

    # Convert predictions to numpy array
    predictions = np.array(predictions).reshape(-1, N)

    # Inverse transform for plotting
    predictions = scaler.inverse_transform(predictions)
    Y_val = scaler.inverse_transform(Y_val)

    if test:
        # Plot the predictions alongside actual values
        plt.figure(figsize=(10, 6))

        # Plot actual stock prices
        plt.plot(np.arange(len(series)), scaler.inverse_transform(series.reshape(-1, 1)), label='Actual Prices', color='blue')

        # Plot non-overlapping predictions
        for i, pred in enumerate(predictions):
            time_index_start = time_indices[i]
            time_index_end = time_index_start + N
            plt.plot(np.arange(time_index_start, time_index_end), pred, color='red', linewidth=2, label='Prediction' if i == 0 else "")

        plt.legend()
        plt.title('Stock Price Prediction (Non-Overlapping)')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.show()

    # predictions[0] # represents most current prediction (N values that go into future dates)
    
    # Print the results
    today = datetime.today().date()

    # Map actions to dates
    action_dates = []
    for i in range(len(predictions[0])):  # Loop based on the length of predictions[0]
        action_dates.append(str(today + timedelta(days=i)))

    return [predictions[0], action_dates]