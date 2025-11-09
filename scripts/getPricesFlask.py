from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from stock_prediction.stockPredictionWithSentimentModel import predict_stock_price

# To run locally
    # within this scripts folder run "python getPricesFlask.py"
    # within main folder run "npm i" then "npm run dev" for front end

app = Flask(__name__)
CORS(app)

@app.route('/predicted_prices', methods=['GET'])
def predicted_prices():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker symbol is required"}), 400

    try:
        print(f"\n\nFetching data and predictions for {ticker}...\n")
        result = predict_stock_price(ticker)
        
        print(f"Predictions complete for {ticker}\n\n")

        # Combine historical (recent 700 days for frontend display, though model uses all 1800 for training) and prediction data (future 50 days) with separator
        all_dates = result['historical_dates'][-700:] + ["Seperate-Dates"] + result['prediction_dates']
        all_prices = result['historical_prices'][-700:] + ["Seperate-Prices"] + result['predictions']
        
        return jsonify([all_dates, all_prices])
    
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return jsonify({"error": f"Failed to process data for {ticker}"}), 500

PUBLIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../public')
@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(PUBLIC_FOLDER, filename)

if __name__ == '__main__':
    app.run()