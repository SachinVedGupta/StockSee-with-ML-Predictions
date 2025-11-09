from datetime import datetime, timedelta
from stock_prediction.getHistoricalDataFeatures import get_stock_features_with_sentiment

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ML (LSTM) model that predicts the Stock Price for 50 days into the future based on the previous 200 day Stock Prices and their News Sentiment Scores
def predict_stock_price(ticker_symbol):
    test = True
    trainSetting = 0
        # DEFAULT: 0 (must not be 1/2) for using existing multi-trained model (trained on the 30 Dow Jones stocks)
        # 1 for training based on just the entered stocks historical data
        # 2 for adding to multi-trained model and enhancing it by training it with this stock as additional data

    stock_data = get_stock_features_with_sentiment(ticker_symbol)
    close_series = stock_data['Close'].values.reshape(-1, 1)
    sentiment_series = stock_data['Sentiment'].values.reshape(-1, 1)

    # normalize the training data to make it easier for model to work with
    close_scaler = StandardScaler()
    close_scaler.fit(close_series[:])
    close_series = close_scaler.transform(close_series).flatten()

    sentiment_scaler = StandardScaler()
    sentiment_scaler.fit(sentiment_series[:])
    sentiment_series = sentiment_scaler.transform(sentiment_series).flatten()

    # combine into a single dataset (series) with two features
    series = np.stack((close_series, sentiment_series), axis=-1)

    T = 200  # past values (input sequence length)
    N = 50   # future values to predict (output sequence length)

    # build the dataset
    X = []
    Y = []
    for t in range(0, len(series) - T - N + 1, 1):  # sliding window
        x = series[t:t+T]  # input sequence to get previous 200 values
        y = series[t+T:t+T+N, 0]  # allocating 50 prediction targets (supervised learning)
        X.append(x)
        Y.append(y)


    X = np.array(X).reshape(-1, T, 2)  # shape = (Num of Dates, T, 2)
    Y = np.array(Y)  # shape = (Num of Dates, N)

    print(" INPUT X.shape", X.shape, "OUTPUT TARGETS Y.shape", Y.shape)

    # split data into training and validation
    train_size = int(len(X) * 0.6)  # use 60% of data for training, rest 40% for validation
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_val, Y_val = X[train_size:], Y[train_size:]

    ### GET MODEL (TRAIN IF NEEDED)

    if trainSetting == 1: # train a model based on just the previous stock's data
        # build the LSTM model and train the model on it
        i = Input(shape=(T, 2))
        x = LSTM(100)(i)  # more units for better learning
        x = Dense(N)(x)  # predict N values (future prices)
        model = Model(i, x)
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

        print(model.summary())
        r = model.fit(
            X_train, Y_train,
            epochs=10,
            validation_data=(X_val, Y_val)
        )
        
        model.save('./sentiment_storage/individual_stock_tf_model.keras')

        # plot the loss graph
        plt.figure(figsize=(10, 6))
        the_title = f"Loss Function (MSE) for {ticker_symbol} prediction model"
        plt.plot(r.history["loss"])
        plt.plot(r.history["val_loss"])
        plt.title(the_title)
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.legend(["loss", "val_loss"])
        plt.savefig("../public/stock_loss.png", dpi=300, bbox_inches='tight')
        plt.close()
    elif trainSetting == 2: # fine-tune the general purpose model (trained on the 30 stocks in DOW JONES) further on this stock's data
        model = tf.keras.models.load_model('./sentiment_storage/DOW_Trained_stock_tf_model.keras')
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.0003)) # recompile with lower learning rate to prevent overfitting to this new dataset (weights can change to match new dataset, but low LR means they can't change too much)

        r = model.fit(
            X_train, Y_train,
            epochs=10,
            validation_data=(X_val, Y_val)
        )

        model.save('./sentiment_storage/DOW_Trained_stock_tf_model.keras')

        plt.figure(figsize=(10, 6))
        the_title = "Loss Function (MSE) for pretrained (on the 30 stocks in DOW JONES) prediction model"
        plt.title(the_title)
        plt.plot(r.history["loss"])
        plt.plot(r.history["val_loss"])
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.legend(["loss", "val_loss"])
        plt.savefig("../public/stock_loss.png", dpi=300, bbox_inches='tight')
        plt.close()
    else: # use pre-trained model (trained on the 30 stocks in the DOW JONES)
        model = tf.keras.models.load_model('./sentiment_storage/DOW_Trained_stock_tf_model.keras')

    ### PREDICT STOCK PRICE (INFERENCE) 

    # predict the next 50 values for each sliding window on validation data
    # make predictions for non-overlapping windows/points
    predictions = []
    time_indices = []

    # loop with step T + N over validation set

    # Example:
    # T = 200  # Past values (input sequence length)
    # N = 50   # Future values to predict (output sequence length)
    # if len(X) = 1000 --> Number of samples/dates

    # create new input list that includes all possible windows of size T, including ones that dont have validation set
    P = []
    for t in range(0, len(series) - T + 1, 1):  # Sliding window to get all possible 200 day segments
        nums = series[t:t+T]
        P.append(nums)

    P = np.array(P).reshape(-1, T, 2)  # Shape should be (Num of Dates - T, T, 2)

    # make predictions for non-overlapping windows/points
    for t in range(len(P) - 1, -1, -(T + N)):  # start from the most recent data
        if t >= 0:  # ensure there are enough points for a full input sequence
            x_current = P[t:t+1]  # input sequence
            y_pred = model.predict(x_current)

            align_gap = x_current[0][-1][0] - y_pred[0][0] # to align the gap between start price and prediction start

            for i in range(len(y_pred)):
                y_pred[i] += align_gap
            predictions.append(y_pred)

            time_indices.append(t + T)  # record the start time of prediction

    # convert predictions to numpy array
    predictions = np.array(predictions).reshape(-1, N)

    # inverse transform for plotting (go from normalized to actual dollar values) (use close_scaler as these predictions are close prices)
    predictions = close_scaler.inverse_transform(predictions)
    Y_val = close_scaler.inverse_transform(Y_val)

    if test:
        # plot the predictions alongside actual values
        plt.figure(figsize=(10, 6))

        # plot actual stock prices
        plt.plot(np.arange(len(close_series)), close_scaler.inverse_transform(close_series.reshape(-1, 1)), label='Actual Prices', color='blue')

        # plot non-overlapping predictions
        for i, pred in enumerate(predictions):
            time_index_start = time_indices[i]
            time_index_end = time_index_start + N
            plt.plot(np.arange(time_index_start, time_index_end), pred, color='red', linewidth=2, label='Prediction' if i == 0 else "")

        plt.legend()
        plt.title(f'Stock Price Predictions for {ticker_symbol}')
        plt.xlabel('Time (days 800-1500 are on the above, main StockSee graph)')
        plt.ylabel('Price')
        plt.savefig("../public/stock_predictions.png", dpi=300, bbox_inches='tight')  # Save the plot to a file
        print(f"Stock Predictions plot saved in /public/stock_predictions.png")
        plt.close()

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
        'historical_prices': historical_prices
    }