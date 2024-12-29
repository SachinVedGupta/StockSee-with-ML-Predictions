"use client"; 
import { useState } from "react";
import "chart.js/auto";
import { Line } from "react-chartjs-2";
import axios from "axios";

export default function Home() {
  const [stockSymbol, setStockSymbol] = useState("");
  const [chartData, setChartData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    try {
      // Trigger the Python script
      await axios.post("http://127.0.0.1:5000/get_stock_data", { ticker: stockSymbol });

      // Fetch the updated JSON file
      const response = await axios.get("http://127.0.0.1:5000/stock_data");
      const data = response.data;

      const dates = data.map((item: any) => item.date);
      const prices = data.map((item: any) => item.close);

      setChartData({
        labels: dates,
        datasets: [
          {
            label: `${stockSymbol} Stock Prices`,
            data: prices,
            borderColor: "rgba(59, 130, 246, 0.9)",
            backgroundColor: "rgba(59, 130, 246, 0.5)",
          },
        ],
      });
    } catch (error) {
      console.error("Error fetching stock data:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Stock Viewer</h1>
      <input
        type="text"
        placeholder="Enter Stock Symbol"
        value={stockSymbol}
        onChange={(e) => setStockSymbol(e.target.value)}
      />
      <button onClick={handleSubmit} disabled={loading}>
        {loading ? "Loading..." : "Submit"}
      </button>
      {chartData && <Line data={chartData} />}
    </div>
  );
}
