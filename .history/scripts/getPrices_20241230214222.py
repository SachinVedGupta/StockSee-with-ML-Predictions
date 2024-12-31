from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


def ml_to_predict


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
