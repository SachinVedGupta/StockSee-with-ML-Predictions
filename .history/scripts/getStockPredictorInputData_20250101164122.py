from getPrices import ml_get_historical # pass in like ("AAPL")
from getNewsArticle import getArticle # pass in like ("AMD", "2024-12-24")
from sentimentAnalysisML import sentiment_from_sentence # pass in like ("sentence_string")

def df_with_sentiment(ticker):

  stock_data = ml_get_historical(ticker)

  # print the dataframe
  if stock_data is not None:
    print("Stock Data for " + ticker + ":\n")
    #print(stock_data)
  else:
    print("Failed to fetch stock data.")
    return 


  sentiment_values = []
  q = 0
  for date in stock_data['Date']:  # Start from index 1 to skip the headers
    if q % 150 == 0:
      string_date = date.strftime("%Y-%m-%d")
      article_string = getArticle(ticker, string_date)
      if article_string == "N/A":
        sentiment_values.append(float(0.5000))
      else:
        sentiment_score = sentiment_from_sentence(article_string)
        sentiment_values.append(round(float(sentiment_score), 4))
    else:
      sentiment_values.append(sentiment_values[q-1])
    q += 1

  #print(sentiment_values)
  stock_data['Sentiment'] = sentiment_values
  # print(stock_data)

  return stock_data
