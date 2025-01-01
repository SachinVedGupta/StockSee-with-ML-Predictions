stock_data = ml_get_historical(ticker_symbol)

# Print the DataFrame
if stock_data is not None:
  print("Stock Data for " + ticker_symbol + ":\n")
    # print(stock_data)
  else:
    print("Failed to fetch stock data.")