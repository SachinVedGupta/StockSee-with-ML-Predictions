from getPrices import ml_get_historical # pass in like ("AAPL")
from getNewsArticle import getArticle # pass in like ("AMD", "2024-12-24")
from sentimentAnalysisML import sentiment_from_sentence # pass in like ("sentence_string")

stock_data = ml_get_historical("AAPL")

# Print the DataFrame
if stock_data is not None:
  print("Stock Data for " + "AAPL" + ":\n")
  print(stock_data)
else:
  print("Failed to fetch stock data.")
  while True: pass

