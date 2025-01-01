from getPrices import ml_get_historical # pass in like ("AAPL")
from getNewsArticle import getArticle # pass in like ("AMD", "2024-12-24")
from sentimentAnalysisML import sentiment_from_sentence # pass in like ("sentence_string")

ticker = "AAPL"

stock_data = ml_get_historical(ticker)

# Print the DataFrame
if stock_data is not None:
  print("Stock Data for " + ticker + ":\n")
  print(stock_data)
else:
  print("Failed to fetch stock data.")
  while True: pass

# every 150 days get new sentiment score
sentiment_values = []
i = 0
for row in stock_data['Date']:  # Start from index 1 to skip the headers
    print(row)  # Prints each row of data
    sentiment_values.append(i)
    i += 1

print(sentiment_values)


