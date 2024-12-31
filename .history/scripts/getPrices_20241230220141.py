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


    return jsonify([dates, prices]) # dates = list where each item is a date    AND    # prices = list where each item is corresponding stock price


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
