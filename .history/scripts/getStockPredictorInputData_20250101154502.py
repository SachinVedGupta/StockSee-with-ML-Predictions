from getPrices import ml_get_historical

stock_data = ml_get_historical("AAPL")

# Print the DataFrame
if stock_data is not None:
  print("Stock Data for " + "AAPL" + ":\n")
  print(stock_data)
else:
  print("Failed to fetch stock data.")