"use client"; // Add this line at the top
import { useState } from "react";
import "chart.js/auto";
import { Line } from "react-chartjs-2";
import axios from "axios";
import Image from "next/image";
import theimg from "./logo.png";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  Title,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement
);

export default function Home() {
  const [stockSymbol, setStockSymbol] = useState("");
  const [chartDisplayData, setChartDisplayData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit() {
    setLoading(true); // Start loading
    setError("");

    try {
      // Trigger the Python script to fetch stock data
      const triggerResponse = await axios.post(
        "http://127.0.0.1:5000/get_stock_data",
        { ticker: stockSymbol }
      );
      if (triggerResponse.status !== 200) {
        throw new Error("Failed to trigger stock data fetch.");
      }

      // Fetch the updated JSON file
      const response = await axios.get("http://127.0.0.1:5000/stock_data");
      const stockData = response.data;

      const dates = stockData.map((item: any) => item.date);
      const prices = stockData.map((item: any) => item.close);

      setChartDisplayData({
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
      setLoading(false); // Stop loading
    }
  }

  async function getImageUrl(prompt: string) {
    const apiKey = "AIzaSyB41BZPIS7OSfBj81rbh1HjMdsiAYr_ATk";
    const searchEngineId = "41624844768c14c9a";
    const query = prompt; // This could be a static query or based on the prompt/content
    const url =
      "https://www.googleapis.com/customsearch/v1?key=" +
      apiKey +
      "&cx=" +
      searchEngineId +
      "&searchType=image&q=" +
      encodeURIComponent(query);

    const response = await axios.get(url);
    const results = response.data;

    if (results.items && results.items.length > 0) {
      console.log(results.items[0].link);
      return results.items[0].link; // Return the first image's URL
    } else {
      console.log("logo");
      return "https://cdn.discordapp.com/attachments/1208274732227764264/1208595162914103376/logo-no-background.png?ex=65e3daf5&is=65d165f5&hm=4d088d6dd1e4fb2cb3e9d75785f38efe79888a31a8d523724e00af838ff36143&"; // Return default image if no results found
    }
  }

  const [realImages, setRealImages] = useState(null);

  return (
    <>
      <main
        className="flex min-h-screen flex-col items-center justify-between p-24"
        id="large-it"
      >
        <div
          className="flex flex-col items-center justify-center w-full max-w-md mx-auto"
          id="cont-it"
        >
          <div id="the-div">
            <h4 id="title">StockSee</h4>
          </div>

          <input
            type="text"
            value={stockSymbol}
            onChange={(e) => setStockSymbol(e.target.value)}
            placeholder="Enter stock symbol"
            className="mb-4 p-2 border rounded"
          />
          <button
            onClick={handleSubmit}
            id="submit-btn"
            className="p-2 bg-blue-500 text-white rounded flex items-center justify-center"
            disabled={loading} // Disable button while loading
          >
            {loading ? (
              <div className="spinner-border text-light" role="status">
                <span className="sr-only">Loading...</span>
              </div>
            ) : (
              "Submit"
            )}
          </button>

          {error && <p className="text-red-500 text-center">{error}</p>}

          {realImages ? (
            <Image
              src={realImages}
              alt="Dynamic Image"
              width={500} // specify dimensions as per your needs
              height={500}
            />
          ) : null}
        </div>

        {chartDisplayData && (
          <Line data={chartDisplayData} options={chartDisplayData.options} />
        )}
      </main>
    </>
  );
}
