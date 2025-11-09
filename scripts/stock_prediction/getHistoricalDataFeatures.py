import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from sentiment.getNewsArticle import getArticle # pass in like ("AMD", "2024-12-24")
from sentiment.sentimentAnalysisModel import sentiment_from_sentence # pass in like ("sentence_string")


# feature engineering for the stock data
def get_stock_features(ticker):
    try:
        # get the stock data
        stock = yf.Ticker(ticker)

        end_date = datetime.today()
        start_date = end_date - timedelta(days=1800)

        historical_data = stock.history(start=start_date, end=end_date)
        historical_data.reset_index(inplace=True)

        # preparing and verifying the features
        historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.date
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


# feature engineering to produce the overall stock prediction features dataframe (with both stock data and sentiment scores)
def get_stock_features_with_sentiment(ticker):

  stock_data = get_stock_features(ticker)

  sentiment_values = []

  for day in range(0, len(stock_data['Date']), 150): # only get new sentiment in increments to avoid too much News API requests from being made
    string_date = stock_data['Date'][day].strftime("%Y-%m-%d")
    
    article_string = getArticle(ticker, string_date)
    if article_string == "N/A":
      sentiment_values.append(float(0.5000))
    else:
      sentiment_score = sentiment_from_sentence(article_string)
      sentiment_values.append(round(float(sentiment_score), 4))
    
    for i in range(day, day+150):
      sentiment_values.append(sentiment_values[i-1])

  stock_data['Sentiment'] = sentiment_values
  return stock_data