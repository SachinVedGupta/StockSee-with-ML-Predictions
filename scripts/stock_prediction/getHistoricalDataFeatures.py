import yfinance as yf
from time import sleep
import pandas as pd
from datetime import datetime, timedelta

from sentiment.getNewsArticle import get_article, NewsUnavailable # pass in like ("AMD", "2024-12-24")
from sentiment.sentimentAnalysisModel import sentiment_from_sentence # pass in like ("sentence_string")


# feature engineering for the stock data
def get_stock_features(ticker):
    try:
        # get the stock data
        stock = yf.Ticker(ticker)

        end_date = datetime.today()
        start_date = end_date - timedelta(days=1800)

        # Yahoo can fail its initial session request after a cold start.
        # Retry once with a fresh ticker; persistent failures still surface normally.
        for attempt in range(2):
            try:
                historical_data = stock.history(start=start_date, end=end_date)
                if not historical_data.empty:
                    break
            except Exception:
                if attempt:
                    raise
            if not attempt:
                sleep(0.5)
                stock = yf.Ticker(ticker)
        else:
            raise ValueError("No market data was returned for this ticker")
        if historical_data.empty:
            raise ValueError("No market data was returned for this ticker")
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
        raise ValueError("Market data is unavailable; verify the ticker or retry later") from e


# feature engineering to produce the overall stock prediction features dataframe (with both stock data and sentiment scores)
def get_stock_features_with_sentiment(ticker):

  stock_data = get_stock_features(ticker)

  if len(stock_data) < 251:
    raise ValueError("At least 251 trading observations are required for this model")
  sentiment_values = []
  news_samples = 0
  neutral_samples = 0
  unavailable_reason = None
  last_sentiment = 0.5  # default neutral sentiment

  # generate sentiment for every 150th day and fill in between
  for day in range(len(stock_data['Date'])):
    if day % 150 == 0:  # fetch new sentiment every 150 days
      string_date = stock_data['Date'][day].strftime("%Y-%m-%d")
      # Stop retrying a failed provider during this request (e.g. quota/auth errors).
      try:
        article_string = "N/A" if unavailable_reason else get_article(ticker, string_date)
      except NewsUnavailable as error:
        unavailable_reason = str(error)
        article_string = "N/A"
      if article_string == "N/A":
        neutral_samples += 1
        last_sentiment = 0.5
      else:
        sentiment_score = sentiment_from_sentence(article_string)
        last_sentiment = round(float(sentiment_score), 4)
        news_samples += 1
    
    sentiment_values.append(last_sentiment)

  stock_data['Sentiment'] = sentiment_values
  stock_data.attrs['sentiment'] = {
    'news_samples': news_samples,
    'neutral_samples': neutral_samples,
    'sampling_interval': 150,
    'warning': unavailable_reason,
  }
  return stock_data