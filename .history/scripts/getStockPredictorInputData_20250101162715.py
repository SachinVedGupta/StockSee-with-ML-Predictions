from getPrices import ml_get_historical # pass in like ("AAPL")
from getNewsArticle import getArticle # pass in like ("AMD", "2024-12-24")
from sentimentAnalysisML import sentiment_from_sentence # pass in like ("sentence_string")

ticker = "AAPL"

stock_data = ml_get_historical(ticker)

# print the dataframe
if stock_data is not None:
  print("Stock Data for " + ticker + ":\n")
  print(stock_data)
else:
  print("Failed to fetch stock data.")
  while True: pass

# get certain dates for sentiment analysis to be performed on
chosen_dates = set()

for i in range(len(stock_data['Date'])):  # Start from index 1 to skip the headers
  if i % 150 == 0:
    chosen_dates.add(str(stock_data['Date'][i]))

print(chosen_dates)


sentiment_values = []
q = 0
for date in stock_data['Date']:  # Start from index 1 to skip the headers
  if q % 150 == 0:
    string_date = date.strftime("%Y-%m-%d")
    article_string = 
    sentiment_values.append(0)
  else:
    sentiment_values.append(sentiment_values[q-1])
  q += 1

print(sentiment_values)


