import yfinance as yf
import json

def getHistoricalPrice(ticker):

    # Create a Ticker object
    stock = yf.Ticker(ticker)

    # Fetch all available historical data
    historical_data = stock.history(period="max")

    # Limit the data to only the "Date" and "Close" columns
    limited_data = historical_data[['Close']]

    # Convert the index (Date) to a column
    limited_data.reset_index(inplace=True)

    # Convert the DataFrame to a dictionary
    data_dict = limited_data.set_index('Date')['Close'].to_dict()

    # Convert dictionary to a list of dictionaries, keeping only the date part
    data_list = [{'date': date.strftime('%Y-%m-%d'), 'close': value} for date, value in data_dict.items()]

    # Save the list of dictionaries to a JSON file
    with open('./stockData.json', 'w') as f:
        json.dump(data_list, f)

getHistoricalPrice('AAPL')
