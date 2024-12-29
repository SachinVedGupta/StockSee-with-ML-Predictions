

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
