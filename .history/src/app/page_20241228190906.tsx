"use client";
import { useState } from "react";
import "chart.js/auto";
import { Line } from "react-chartjs-2";
import axios from "axios";

export default function Home() {
  const [stockSymbol, setStockSymbol] = useState("");
  const [chartData, setChartData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    setLoading(true);
    setError("");
    try {
      // Trigger the Python script to fetch stock data
      const triggerResponse = await axios.post("http://127.0.0.1:5000/get_stock_data", {
        ticker: stockSymbol,
      });
      if (triggerResponse.status !== 200) {
        throw new Error("Failed to fetch stock data.");
      }

      // Fetch the updated stock data from the JSON file
      const response = await axios.get("http://127.0.0.1:5000/stock_data");
      const data = response.data;

      // Parse the data into dates and prices
      const dates = data.map((item: any) => item.date);
      const prices = data.map((item: any) => item.close);

      // Update the chart data
      setChartData({
        labels: dates,
        datasets: [
          {
            label: `${stockSymbol.toUpperCase()} Stock Prices`,
            data: prices,
            borderColor: "rgba(59, 130, 246, 0.9)",
            backgroundColor: "rgba(59, 130, 246, 0.5)",
            tension: 0.2,
          },
        ],
      });
    } catch (err: any) {
      console.error("Error:", err);
      setError("Failed to fetch stock data. Please check the symbol and try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-6">
      <div className="w-full max-w-md">
        <h1 className="text-2xl font-bold text-center mb-4">Stock Viewer</h1>
        <input
          type="text"
          value={stockSymbol}
          onChange={(e) => setStockSymbol(e.target.value)}
          placeholder="Enter Stock Symbol (e.g., AAPL)"
          className="w-full p-2 border rounded mb-4"
        />
        <button
          onClick={handleSubmit}
          className="w-full bg-blue-500 text-white py-2 rounded mb-4"
          disabled={loading}
        >
          {loading ? "Loading..." : "Fetch Stock Data"}
        </button>
        {error && <p className="text-red-500 text-center mb-4">{error}</p>}
        {chartData && (
          <div>
            <h2 className="text-xl font-semibold text-center mb-4">
              {stockSymbol.toUpperCase()} Stock Prices
            </h2>
            <Line data={chartData} />
          </div>
        )}
      </div>
    </main>
  );
}
