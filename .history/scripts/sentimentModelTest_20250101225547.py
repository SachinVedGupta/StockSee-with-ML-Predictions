from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import yfinance as yf
from datetime import datetime, timedelta

from getStockPredictorInputData import df_with_sentiment

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D, Attention, Concatenate, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam




def with_sentiment_ml_to_predict(ticker_symbol):
    test = False
    # stock_data = ml_get_historical(ticker_symbol)

    stock_data = df_with_sentiment(ticker_symbol)
















    # series represents all the stock prices (each value is the price for that day)
    # plt.plot(stock_data.index, stock_data['Close'])
    close_series = stock_data['Close'].values.reshape(-1, 1) # percent change or close or sentiment
    sentiment_series = stock_data['Sentiment'].values.reshape(-1, 1) # percent change or close or sentiment
    # print(series)

    # Normalize the training data to make it easier for model to work with
    close_scaler = StandardScaler()
    close_scaler.fit(close_series[:])
    close_series = close_scaler.transform(close_series).flatten()

    sentiment_scaler = StandardScaler()
    sentiment_scaler.fit(sentiment_series[:])
    sentiment_series = sentiment_scaler.transform(sentiment_series).flatten()

    # Combine into a single dataset (series) with two features
    series = np.stack((close_series, sentiment_series), axis=-1)

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

    X = []
    Y = []
    for t in range(0, len(series) - T - N + 1, 1):  # Sliding window with step 1
        x = series[t:t+T]  # Input sequence to get previous 100 values
        y = series[t+T:t+T+N, 0]  # Outputs (target close prices only
        X.append(x)
        Y.append(y)


    X = np.array(X).reshape(-1, T, 2)  # Shape should be (Num of Dates, T, 2)
    Y = np.array(Y)  # Shape should be (Num of Dates, N)

    print(" INPUT X.shape", X.shape, "OUTPUT TARGETS Y.shape", Y.shape)


    # Build the LSTM model

    i = Input(shape=(T, 2))
    x = LSTM(100)(i)  # More units for better learning
    x = Dense(N)(x)  # Predict N values (future prices)
    model = Model(i, x)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))




    print(model.summary())















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

    # Inverse transform for plotting (use close_scaler as these predictions are close prices)
    predictions = close_scaler.inverse_transform(predictions)
    Y_val = close_scaler.inverse_transform(Y_val)

    if test:
        # Plot the predictions alongside actual values
        plt.figure(figsize=(10, 6))

        # Plot actual stock prices
        plt.plot(np.arange(len(series)), close_scaler.inverse_transform(series.reshape(-1, 1)), label='Actual Prices', color='blue')

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


print(with_sentiment_ml_to_predict("AMD"))