from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import yfinance as yf
from datetime import datetime, timedelta

from stockPredictionsWithSentiment import with_sentiment_ml_to_predict
from stockPredictions import ml_to_predict

from tensorflow.keras.layers import Input, LSTM, Dense, GlobalMaxPool1D, Attention, Concatenate, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

# To run
    #
    # within main folder run "npm i" then "npm run dev" for front end

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to fetch historical prices from Yahoo Finance
def get_historical_price(ticker):
    try:
        # Create a Ticker object
        stock = yf.Ticker(ticker)

        # Define the date range: last 3 months
        end_date = datetime.today()
        start_date = end_date - timedelta(days=700)

        # Fetch historical data
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

    # the_ml_predictions = ml_to_predict(ticker) # for stock price 
    the_ml_predictions = with_sentiment_ml_to_predict(ticker)
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