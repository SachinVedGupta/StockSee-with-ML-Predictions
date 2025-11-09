import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from sentiment.getNewsArticle import get_article # pass in like ("AMD", "2024-12-24")
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
  last_sentiment = 0.5  # default neutral sentiment

  # generate sentiment for every 150th day and fill in between
  for day in range(len(stock_data['Date'])):
    if day % 150 == 0:  # fetch new sentiment every 150 days
      string_date = stock_data['Date'][day].strftime("%Y-%m-%d")
      article_string = get_article(ticker, string_date)
      
      if article_string == "N/A":
        last_sentiment = float(0.5000)
      else:
        sentiment_score = sentiment_from_sentence(article_string)
        last_sentiment = round(float(sentiment_score), 4)
    
    sentiment_values.append(last_sentiment)

  stock_data['Sentiment'] = sentiment_values
  return stock_data