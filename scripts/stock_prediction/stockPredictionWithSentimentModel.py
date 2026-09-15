from datetime import datetime, timedelta
from stock_prediction.getHistoricalDataFeatures import get_stock_features_with_sentiment

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gc
import numpy as np

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from functools import lru_cache

@lru_cache(maxsize=1)
def get_stock_model():
    return tf.keras.models.load_model(
        './sentiment_storage/DOW_Trained_stock_tf_model.keras', compile=False)


# ML (LSTM) model that predicts the Stock Price for 50 days into the future based on the previous 200 day Stock Prices and their News Sentiment Scores
def predict_stock_price(ticker_symbol):
    test = True
    stock_data = get_stock_features_with_sentiment(ticker_symbol)
    close_series = stock_data['Close'].values.reshape(-1, 1)
    sentiment_series = stock_data['Sentiment'].values.reshape(-1, 1)

    # Same population-variance standardization, without importing sklearn/scipy
    # into the memory-limited serving process.
    close_mean = close_series.mean()
    close_scale = close_series.std() or 1.0
    close_series = ((close_series - close_mean) / close_scale).flatten()
    sentiment_scale = sentiment_series.std() or 1.0
    sentiment_series = ((sentiment_series - sentiment_series.mean()) / sentiment_scale).flatten()

    # combine into a single dataset (series) with two features
    series = np.stack((close_series, sentiment_series), axis=-1)

    T = 200  # past values (input sequence length)
    N = 50   # future values to predict (output sequence length)

    # Inference needs only selected 200-day windows, not a training dataset.
    model = get_stock_model()

    ### PREDICT STOCK PRICE (INFERENCE) 

    # predict the next 50 values for each sliding window on validation data
    # make predictions for non-overlapping windows/points
    predictions = []
    time_indices = []

    # Preserve the original latest-first, non-overlapping chart windows.
    for t in range(len(series) - T, -1, -(T + N)):
        if t >= 0:
            x_current = series[t:t + T][None, ...]
            y_pred = model(x_current, training=False).numpy()

            align_gap = x_current[0][-1][0] - y_pred[0][0] # to align the gap between start price and prediction start

            for i in range(len(y_pred)):
                y_pred[i] += align_gap
            predictions.append(y_pred)

            time_indices.append(t + T)  # record the start time of prediction

    # convert predictions to numpy array
    predictions = np.array(predictions).reshape(-1, N)

    # inverse transform for plotting (go from normalized to actual dollar values) (use close_scaler as these predictions are close prices)
    predictions = predictions * close_scale + close_mean

    if test:
        # plot the predictions alongside actual values
        plt.figure(figsize=(10, 6))

        # plot actual stock prices
        plt.plot(np.arange(len(close_series)), close_series.reshape(-1, 1) * close_scale + close_mean, label='Actual Prices', color='blue')

        # plot non-overlapping predictions
        for i, pred in enumerate(predictions):
            time_index_start = time_indices[i]
            time_index_end = time_index_start + N
            plt.plot(np.arange(time_index_start, time_index_end), pred, color='red', linewidth=2, label='Prediction' if i == 0 else "")

        plt.legend()
        plt.title(f'Stock Price Predictions for {ticker_symbol}')
        plt.xlabel('Time (days 800-1500 are on the above, main StockSee graph)')
        plt.ylabel('Price')
        plt.savefig("../public/stock_predictions.png", dpi=120, bbox_inches='tight')  # Save the plot to a file
        print(f"Stock Predictions plot saved in /public/stock_predictions.png")
        plt.close()
        gc.collect()  # Release Matplotlib figure cycles before the next request.

    # predictions[0] = stock price predictions for 50 days into the future (N values that go into future dates)
    today = datetime.today().date()

    # action_dates = get the dates for the future 50 days (from current date)
    action_dates = []
    for i in range(len(predictions[0])):
        action_dates.append(str(today + timedelta(days=i)))

    # historical_dates and prices = prepare 700 day historical data for frontend graph
    historical_dates = [date.strftime('%Y-%m-%d') for date in stock_data['Date']]
    historical_prices = stock_data['Close'].tolist()

    return {
        'predictions': predictions[0].tolist(),
        'prediction_dates': action_dates,
        'historical_dates': historical_dates,
        'historical_prices': historical_prices,
        'metadata': {
            'inference': {
                'model': 'DOW_Trained_stock_tf_model.keras',
                'input_shape': [1, T, 2],
                'output_count': len(predictions[0]),
                'price_alignment': 'First model estimate shifted to the last observed closing price',
            },
            'sentiment': stock_data.attrs.get('sentiment', {}),
        }
    }