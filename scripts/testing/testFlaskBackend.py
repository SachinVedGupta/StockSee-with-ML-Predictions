import requests

# Local: BASE_URL = "http://127.0.0.1:5000"
BASE_URL = "https://stocksee-with-ml-predictions.onrender.com/"  # Adjust if the Flask app is hosted elsewhere
ENDPOINT = "/predicted_prices"

# For testing the getPricesFlask file (the backend and ml model for generating a prediction)
def fetch_stock_data(ticker):
    try:
        response = requests.get(f"{BASE_URL}{ENDPOINT}", params={"ticker": ticker})

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error: {response.status_code}, {response.json()}")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    ticker_symbol = "AAPL"
    stock_data = fetch_stock_data(ticker_symbol)

    if stock_data:
        print("Dates:", stock_data[0])
        print("Prices:", stock_data[1])
    else:
        print("Failed to fetch stock data.")