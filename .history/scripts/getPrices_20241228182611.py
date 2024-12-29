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

'''
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import random
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/historical_prices', methods=['GET'])
def historical_prices():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker symbol is required"}), 400
    
    # Generate dummy stock data for 100 days
    end_date = datetime.today()
    start_date = end_date - timedelta(days=100)
    dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(100)]
    prices = [round(random.uniform(100, 500), 2) for _ in range(100)]

    return jsonify([dates, prices])

if __name__ == '__main__':
    app.run(debug=True)
'''


from flask import Flask, request, jsonify
import yfinance as yf
import json
import os

app = Flask(__name__)
DATA_FILE = './stockData.json'

def get_historical_price(ticker):
    stock = yf.Ticker(ticker)
    historical_data = stock.history(period="max")
    limited_data = historical_data[['Close']]
    limited_data.reset_index(inplace=True)
    data_list = [{'date': date.strftime('%Y-%m-%d'), 'close': value} for date, value in limited_data.set_index('Date')['Close'].items()]

    with open(DATA_FILE, 'w') as f:
        json.dump(data_list, f)

@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    ticker = request.json.get('ticker')
    if not ticker:
        return jsonify({'error': 'Ticker symbol is required'}), 400

    try:
        get_historical_price(ticker)
        return jsonify({'message': 'Stock data fetched successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stock_data', methods=['GET'])
def stock_data():
    if not os.path.exists(DATA_FILE):
        return jsonify({'error': 'Data file not found'}), 404

    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
