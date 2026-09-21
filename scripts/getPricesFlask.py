from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import re
import resource
from threading import Lock

# Bound TensorFlow/BLAS thread pools before the ML modules are imported.
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from sentiment.getNewsArticle import get_articles, NewsUnavailable
def predict_stock_price(ticker):
    # Let health checks and news respond while the ML runtime is loading.
    from stock_prediction.stockPredictionWithSentimentModel import predict_stock_price as infer
    return infer(ticker)

# To run locally
    # within this scripts folder run "python getPricesFlask.py"
    # within main folder run "npm i" then "npm run dev" for front end

app = Flask(__name__)
CORS(app)
prediction_lock = Lock()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/predicted_prices', methods=['GET'])
def predicted_prices():
    ticker = request.args.get('ticker', '').strip().upper()
    if not re.fullmatch(r'[A-Z0-9.^=-]{1,20}', ticker):
        return jsonify({"error": "Ticker symbol is required"}), 400

    # Do not occupy every HTTP thread waiting for the inference lock: Render
    # must still be able to health-check the process during a long prediction.
    if not prediction_lock.acquire(blocking=False):
        return jsonify({"error": "Another prediction is running", "code": "PREDICTION_BUSY"}), 429, {"Retry-After": "3"}
    try:
        print(f"\n\nFetching data and predictions for {ticker}...\n")
        # A single inference/plot at a time bounds peak memory and protects pyplot.
        result = predict_stock_price(ticker)
        
        print(f"Predictions complete for {ticker}; peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} (KiB on Linux)\n\n")

        # Combine historical (recent 700 days for frontend display, though model uses all 1800 for training) and prediction data (future 50 days) with separator
        all_dates = result['historical_dates'][-700:] + ["Seperate-Dates"] + result['prediction_dates']
        all_prices = result['historical_prices'][-700:] + ["Seperate-Prices"] + result['predictions']
        
        return jsonify([all_dates, all_prices, result.get("metadata", {})])
    
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return jsonify({"error": f"Failed to process data for {ticker}"}), 500
    finally:
        prediction_lock.release()

@app.route('/news', methods=['GET'])
def news():
    ticker = request.args.get('ticker', '').strip().upper()
    if not re.fullmatch(r'[A-Z0-9.^=-]{1,20}', ticker):
        return jsonify({"error": "A valid ticker symbol is required"}), 400
    try:
        return jsonify({"articles": get_articles(ticker)})
    except NewsUnavailable as error:
        return jsonify({"articles": [], "warning": str(error)}), 503


PUBLIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../public')
@app.route('/image/<filename>')
def serve_image(filename):
    return send_from_directory(PUBLIC_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
