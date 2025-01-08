from datetime import datetime, timedelta
from getSentimentDataFrame import df_with_sentiment

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



def with_sentiment_ml_to_predict(ticker_symbol): # returns predictions made using both Close price and Sentiment Score as an input
    test = True
    train = False
    trainSetting = 0 
        # 0 (must not be 1/2) for using existing multi-trained model (trained on the 30 Dow Jones stocks)
        # 1 for training based on just the entered stocks historical data
        # 2 for adding to multi-trained model and enhancing it by training it with this stock as additional data
    # stock_data = ml_get_historical(ticker_symbol)

    stock_data = df_with_sentiment(ticker_symbol)

    # series represents all the stock prices (each value is the price for that day)
    # plt.plot(stock_data.index, stock_data['Close'])
    close_series = stock_data['Close'].values.reshape(-1, 1) # percent change or close or sentiment
    sentiment_series = stock_data['Sentiment'].values.reshape(-1, 1) # percent change or close or sentiment
    # print(series)

    # Normalize the training data to make it easier for model to work with
    close_scaler = StandardScaler()
    close_scaler.fit(close_series[:])
    close_series = close_scaler.transform(close_series).flatten()

    sentiment_scaler = StandardScaler()
    sentiment_scaler.fit(sentiment_series[:])
    sentiment_series = sentiment_scaler.transform(sentiment_series).flatten()

    # Combine into a single dataset (series) with two features
    series = np.stack((close_series, sentiment_series), axis=-1)

    T = 200  # Past values (input sequence length)
    N = 50   # Future values to predict (output sequence length)

    # Build the dataset
    X = []
    Y = []
    for t in range(0, len(series) - T - N + 1, 1):  # Sliding window with step 1
        x = series[t:t+T]  # Input sequence to get previous 100 values
        y = series[t+T:t+T+N, 0]  # Outputs (target close prices only for y)
        X.append(x)
        Y.append(y)


    X = np.array(X).reshape(-1, T, 2)  # Shape should be (Num of Dates, T, 2)
    Y = np.array(Y)  # Shape should be (Num of Dates, N)

    print(" INPUT X.shape", X.shape, "OUTPUT TARGETS Y.shape", Y.shape)



    # Split data into training and validation
    train_size = int(len(X) * 0.6)  # Use ?% of data for training, rest % for validation
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_val, Y_val = X[train_size:], Y[train_size:]



    # Train the model or load in existing one based on trainSetting value
    
    if trainSetting == 1: # train a model based on just the previous stock's data
        # Build the LSTM model

        i = Input(shape=(T, 2))
        x = LSTM(100)(i)  # More units for better learning
        x = Dense(N)(x)  # Predict N values (future prices)
        model = Model(i, x)
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

        print(model.summary())
        r = model.fit(
            X_train, Y_train,  # Use training data
            epochs=10,
            validation_data=(X_val, Y_val)  # Use validation data
        )
        model.save('./sentiment_storage/stock_tf_model.keras')

            # plot the loss graph

        print("\n\n\n\n\n\n")
        print(r.history)
        print(len(r.history["loss"]))  # Should be > 0 if the model is training correctly
        print("\n\n\n\n\n\n")


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
    
    elif trainSetting == 2:
        print(1)
    
    else: # use pre-trained model (trained on the DOW JONES stock)
        model = tf.keras.models.load_model('./sentiment_storage/stock_tf_model.keras')






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

    P = np.array(P).reshape(-1, T, 2)  # Shape should be (Num of Dates, T, 2)

    for t in range(len(P) - 1, -1, -(T + N)):  # Start from the most recent data
        if t >= 0:  # Ensure there are enough points for a full input sequence
            x_current = P[t:t+1]  # Input sequence
            y_pred = model.predict(x_current)
            predictions.append(y_pred)
            time_indices.append(t + T)  # Record the start time of prediction

    # Convert predictions to numpy array
    predictions = np.array(predictions).reshape(-1, N)

    # Inverse transform for plotting (use close_scaler as these predictions are close prices)
    predictions = close_scaler.inverse_transform(predictions)
    Y_val = close_scaler.inverse_transform(Y_val)

    if test:
        # Plot the predictions alongside actual values
        plt.figure(figsize=(10, 6))

        # Plot actual stock prices
        plt.plot(np.arange(len(close_series)), close_scaler.inverse_transform(close_series.reshape(-1, 1)), label='Actual Prices', color='blue')

        # Plot non-overlapping predictions
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
        # plt.show()

    # predictions[0] # represents most current prediction (N values that go into future dates)
    
    # Print the results
    today = datetime.today().date()

    # Map actions to dates
    action_dates = []
    for i in range(len(predictions[0])):  # Loop based on the length of predictions[0]
        action_dates.append(str(today + timedelta(days=i)))

    return [predictions[0], action_dates]