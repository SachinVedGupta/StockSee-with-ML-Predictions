from getPrices import ml_get_historical # pass in like ("AAPL")
from getNewsArticle import getArticle # pass in like ("AMD", "2024-12-24")
from sentimentAnalysisML import sentiment_from_sentence # pass in like ("sentence_string")

import yfinance as yf
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D, Attention, Concatenate, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam


def df_with_sentiment(ticker):

  stock_data = ml_get_historical(ticker)

  # print the dataframe
  if stock_data is not None:
    print("Stock Data for " + ticker + ":\n")
    #print(stock_data)
  else:
    print("Failed to fetch stock data.")
    return "ERROR"


  sentiment_values = []
  q = 0
  for date in stock_data['Date']:  # Start from index 1 to skip the headers
    if q % 150 == 0:
      string_date = date.strftime("%Y-%m-%d")
      article_string = getArticle(ticker, string_date)
      if article_string == "N/A":
        sentiment_values.append(float(0.5000))
      else:
        sentiment_score = sentiment_from_sentence(article_string)
        sentiment_values.append(round(float(sentiment_score), 4))
    else:
      sentiment_values.append(sentiment_values[q-1])
    q += 1

  #print(sentiment_values)
  stock_data['Sentiment'] = sentiment_values
  # print(stock_data)

  return stock_data






def with_sentiment_ml_to_predict(ticker_symbol):
    test = False
    # stock_data = ml_get_historical(ticker_symbol)

    stock_data = df_with_sentiment(ticker_symbol)
















    # series represents all the stock prices (each value is the price for that day)
    # plt.plot(stock_data.index, stock_data['Close'])
    series = stock_data['Close', 'Sentiment'].values.reshape(-1, 2) # percent change or close or sentiment
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

    i = Input(shape=(T, 2))
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

with