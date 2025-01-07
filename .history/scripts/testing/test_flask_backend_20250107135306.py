import requests

# For testing the getPrices file
# Define the Flask app's URL and the endpoint
BASE_URL = "https://stocksee-with-ml-predictions.onrender.com/"  # Adjust if the Flask app is hosted elsewhere
ENDPOINT = "/historical_prices"

def fetch_stock_data(ticker):
    try:
        # Make a GET request to the Flask app's endpoint
        response = requests.get(f"{BASE_URL}{ENDPOINT}", params={"ticker": ticker})

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            return data
        else:
            print(f"Error: {response.status_code}, {response.json()}")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Replace 'AAPL' with any desired stock ticker symbol
    ticker_symbol = "AAPL"

    # Fetch the stock data
    stock_data = fetch_stock_data(ticker_symbol)



    # Print the raw list
    if stock_data:
        print("Dates:", stock_data[0])  # List of dates
        print("Prices:", stock_data[1])  # List of closing prices
    else:
        print("Failed to fetch stock data.")