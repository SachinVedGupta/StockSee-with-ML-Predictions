from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import yfinance as yf
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, GlobalMaxPool1D, Attention, Concatenate, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def ml_get_historical(ticker):
    try:
        # Create a Ticker object
        stock = yf.Ticker(ticker)

        # Define the date range: last 3 months
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

    stock_data = ml_get_historical(ticker_symbol)


    # Print the DataFrame
    if stock_data is not None:
        print("Stock Data for " + ticker_symbol + ":\n")
    # print(stock_data)
    else:
        print("Failed to fetch stock data.")

    # stock_data.index = stock_data.pop('Date')
    # series represents all the stock prices (each value is the price for that day)

    # plt.plot(stock_data.index, stock_data['Close'])
    series = stock_data['Close'].values.reshape(-1, 1) # percent chang or close
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

    test = False

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






# Function to fetch historical prices from Yahoo Finance
def get_historical_price(ticker):
    try:
        # Create a Ticker object
        stock = yf.Ticker(ticker)

        # Define the date range: last 3 months
        end_date = datetime.today()
        start_date = end_date - timedelta(days=700)

        # Fetch historical data with weekly increments
        historical_data = stock.history(start=start_date, end=end_date)

        # Limit the data to only the "Date" and "Close" columns
        limited_data = historical_data[['Close']]

        # Convert the index (Date) to a column
        limited_data.reset_index(inplace=True)

        # Convert the DataFrame to a list of dictionaries
        data_list = [
            {'date': row['Date'].strftime('%Y-%m-%d'), 'close': row['Close']}
            for _, row in limited_data.iterrows()
        ]

        return data_list

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


@app.route('/historical_prices', methods=['GET'])
def historical_prices():
    # Get the ticker symbol from the query string
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker symbol is required"}), 400

    # Fetch historical data for the ticker
    data = get_historical_price(ticker)
    if data is None:
        return jsonify({"error": f"Failed to fetch data for {ticker}"}), 500

    # Separate the dates and prices for easier frontend handling
    dates = [item['date'] for item in data]
    prices = [item['close'] for item in data]

    # Add seperator/dummy values to distinguish between real and predicted values
    dates.append("Seperate-Dates")
    prices.append("Seperate-Prices")

    print("\n\n\n\n\nML STARTING")
    the_ml_predictions = ml_to_predict(ticker)
    print("\n\n\n\n\nML DONE")
    the_ml_predictions[0] = the_ml_predictions[0].tolist()
    print(the_ml_predictions)
    print("\n\n\nTHE ML PREDICTIONS ARE ABOVE\n\n\n")
    final_graph_values = []
    final_graph_values.append(dates + the_ml_predictions[1])
    final_graph_values.append(prices + the_ml_predictions[0])
    print(final_graph_values)
    print("\n\nDONE\n\n")


    return jsonify(final_graph_values) # dates = list where each item is a date    AND    # prices = list where each item is corresponding stock price


if __name__ == '__main__':
    app.run(debug=True)


'''
import yfinance as yf
import json

def getHistoricalPrice(ticker):

    # Create a Ticker object
    stock = yf.Ticker(ticker)

    # Fetch all available historical data
    historical_data = stock.history(period="max")

    # Limit the data to only the "Date" and "Close" columns
    limited_data = historical_data[['Close']]

    # Convert the index (Date) to a column
    limited_data.reset_index(inplace=True)

    # Convert the DataFrame to a dictionary
    data_dict = limited_data.set_index('Date')['Close'].to_dict()

    # Convert dictionary to a list of dictionaries, keeping only the date part
    data_list = [{'date': date.strftime('%Y-%m-%d'), 'close': value} for date, value in data_dict.items()]

    # Save the list of dictionaries to a JSON file
    with open('./stockData.json', 'w') as f:
        json.dump(data_list, f)

getHistoricalPrice('AAPL')
'''
