"use client";

// IMPORTS
import React, { useEffect, useState } from "react";
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

// CHART.JS CONFIGURATION
ChartJS.register(
  Title,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement
);

// MAIN COMPONENT
export default function Home() {
  // BACKEND CONFIGURATION
  const deployedBackendURL =
    "https://stocksee-with-ml-predictions.onrender.com";
  const localBackendURL = "http://127.0.0.1:5000";
  const backendURL = deployedBackendURL;

  // COMPONENT STATE
  const [stockSymbol, setStockSymbol] = useState("");
  const [chartDisplayData, setChartDisplayData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [showSummary, setShowSummary] = useState(false);
  const [showGraphs, setShowGraphs] = useState(false);
  const [realImages, setRealImages] = useState(null);
  const [imageUrls, setImageUrls] = useState<string[]>([]);

  // HELPER FUNCTIONS

  /**
   * Fetches company office image via Google Image Search API
   * @param prompt - Search query for the image
   * @returns URL of the first image result or a default logo
   */
  async function getImageUrl(prompt: string) {
    var apiKey = process.env.NEXT_PUBLIC_GOOGLE_API_KEY;
    var searchEngineId = process.env.NEXT_PUBLIC_SEARCH_ENGINE_ID;
    var query = prompt;
    var url =
      "https://www.googleapis.com/customsearch/v1?key=" +
      apiKey +
      "&cx=" +
      searchEngineId +
      "&searchType=image&q=" +
      encodeURIComponent(query);

    var response = await axios.get(url);
    var results = response.data;

    if (results.items && results.items.length > 0) {
      console.log(results.items[0].link);
      return results.items[0].link;
    } else {
      console.log("logo");
      return "https://cdn.discordapp.com/attachments/1208274732227764264/1208595162914103376/logo-no-background.png?ex=65e3daf5&is=65d165f5&hm=4d088d6dd1e4fb2cb3e9d75785f38efe79888a31a8d523724e00af838ff36143&";
    }
  }

  // DATA FETCHING AND CHART GENERATION

  /**
   * Main function to fetch stock data, predictions, and news
   * Processes the data and generates chart configuration
   */
  async function handleSubmit() {
    setLoading(true);
    try {
      // Fetch predicted prices from backend
      const chartResponse = await fetch(
        `${backendURL}/predicted_prices?ticker=${stockSymbol}`
      );
      const chartData = await chartResponse.json();

      const dates = chartData[0];
      const prices = chartData[1];

      // IDENTIFY SIGNIFICANT PRICE CHANGES
      // Calculate significant rise/drop points using a rolling window
      const significantPoints: any[] = [];
      const windowSize = 20; // Window size for delta calculation
      const threshold = 0.12; // Threshold for significant change (12%)
      const minDistance = 30; // Minimum distance between significant points (days)

      const changes: any[] = [];
      const date: any[] = [];
      for (let i = windowSize; i < prices.length; i++) {
        const pastPrice = prices[i - windowSize];
        const currentPrice = prices[i];
        const delta = (currentPrice - pastPrice) / pastPrice;

        if (Math.abs(delta) > threshold) {
          changes.push({
            index: i - windowSize,
            x: dates[i - windowSize],
            y: pastPrice,
            delta: delta,
          });
          date.push(dates[i - windowSize]);
          i = i + minDistance;
        }
      }

      // Fetch news analysis for significant dates
      const news = await axios.post("/api/gemini", { stockSymbol, date });

      // Fetch company office image
      const realImages = getImageUrl(stockSymbol + " ticker company office");
      realImages.then((url) => {
        setRealImages(url);
      });

      // SEPARATE HISTORICAL AND PREDICTED DATA
      const separateDateIndex = dates.indexOf("Seperate-Dates");

      // Bridge the gap between historical and predicted data
      const valueToLeft = dates[separateDateIndex - 1];
      dates.splice(separateDateIndex + 1, 0, valueToLeft);
      prices.splice(separateDateIndex + 1, 0, prices[separateDateIndex - 1]);

      // Split data into historical (first part) and predictions (second part)
      const firstPartDates = dates.slice(0, separateDateIndex);
      const firstPartPrices = prices.slice(0, separateDateIndex);

      const secondPartDates = dates.slice(separateDateIndex + 1);
      const uniqueSecondPartDates = new Set(secondPartDates);
      const secondPartPrices = prices.slice(separateDateIndex + 1);

      const combinedDates = [...firstPartDates, ...secondPartDates];
      const combinedPrices = [...firstPartPrices, ...secondPartPrices];

      // CONFIGURE CHART DATA
      setChartDisplayData({
        labels: combinedDates,
        datasets: [
          // Historical stock prices dataset
          {
            label: `${stockSymbol} Stock Price`,
            backgroundColor: dates.map((_: any, i: any) =>
              uniqueSecondPartDates.has(dates[i])
                ? "rgba(255, 165, 0, 0.5)"
                : "rgba(59, 130, 246, 0.5)"
            ),
            borderColor: dates.map((_: any, i: any) =>
              uniqueSecondPartDates.has(dates[i])
                ? "rgba(255, 165, 0, 0.9)"
                : "rgba(59, 130, 246, 0.9)"
            ),
            data: [
              ...firstPartPrices,
              ...Array(secondPartPrices.length).fill(null),
            ],
            pointBackgroundColor: dates.map((_: any, i: any) =>
              uniqueSecondPartDates.has(dates[i])
                ? "rgba(255, 165, 0, 1)"
                : changes.find((point) => point.index === i)
                ? changes.find((point) => point.index === i)!.delta > 0
                  ? "rgb(68, 246, 59)" // Green for rise
                  : "red" // Red for fall
                : "rgba(75, 192, 192, 0.6)"
            ),
            pointRadius: dates.map((_: any, i: any) =>
              changes.find((point) => point.index === i)
                ? 5
                : uniqueSecondPartDates.has(dates[i])
                ? 2
                : 0
            ),
            pointHoverRadius: 10,
          },
          // Future predictions dataset
          {
            label: "Future Predictions",
            data: [
              ...Array(firstPartPrices.length).fill(null),
              ...secondPartPrices,
            ],
            backgroundColor: "rgba(255, 165, 0, 0.5)",
            borderColor: "rgba(255, 165, 0, 0.9)",
            pointBackgroundColor: "rgba(255, 165, 0, 1)",
            pointRadius: 0.1,
            pointHoverRadius: 10,
          },
          // Legend entry for significant rise
          {
            label: "Significant Rise Coming",
            data: Array(dates.length).fill(null),
            backgroundColor: "rgba(68, 246, 59, 0.5)",
            borderColor: "rgba(68, 246, 59, 0.9)",
            pointBackgroundColor: "rgba(68, 246, 59, 1)",
            pointRadius: 5,
            pointHoverRadius: 10,
          },
          // Legend entry for significant fall
          {
            label: "Significant Fall Coming",
            data: Array(dates.length).fill(null),
            backgroundColor: "rgba(255, 0, 0, 0.5)",
            borderColor: "rgba(255, 0, 0, 0.9)",
            pointBackgroundColor: "rgba(255, 0, 0, 1)",
            pointRadius: 5,
            pointHoverRadius: 10,
          },
        ],
      });

      // CONFIGURE CHART OPTIONS (TOOLTIPS)
      const chartOptions = {
        plugins: {
          tooltip: {
            callbacks: {
              label: function (tooltipItem: any) {
                const point = changes.find(
                  (p: any) => p.index === tooltipItem.dataIndex
                );
                if (point) {
                  let theanswer = "N/A";
                  for (const thing in news.data.news) {
                    if (news.data.news[thing].includes(point.x)) {
                      theanswer = news.data.news[thing];
                    }
                  }
                  return `Price: ${tooltipItem.raw.toFixed(
                    2
                  )},  News: ${theanswer}`;
                }
                return `Price: ${tooltipItem.raw.toFixed(2)}`;
              },
            },
          },
        },
      };

      setChartDisplayData((prevState: any) => ({
        ...prevState,
        options: chartOptions,
      }));
    } catch (error) {
      console.error("Error fetching data:", error);
    } finally {
      setLoading(false);
    }
  }

  // LOAD ML MODEL GRAPH IMAGES
  useEffect(() => {
    const baseUrl = `${backendURL}/image`;

    // List of ML model graph image filenames
    const imageFilenames = [
      "sentiment_accuracy.png",
      "sentiment_loss.png",
      "stock_loss.png",
      "stock_predictions.png",
    ];

    const imageUrls = imageFilenames.map(
      (filename) => `${baseUrl}/${filename}`
    );

    setImageUrls(imageUrls);
  }, []);

  // RENDER UI
  return (
    <>
      <main
        className="flex min-h-screen flex-col items-center justify-between p-24"
        id="large-it"
      >
        {/* HEADER SECTION - Logo, Title, Input, Submit Button */}
        <div
          className="flex flex-col items-center justify-center w-full max-w-md mx-auto"
          id="cont-it"
        >
          {/* Logo and Title */}
          <div
            id="the-div"
            style={{ display: "flex", alignItems: "center", gap: "12px" }}
          >
            <Image
              src={theimg}
              id="the-img"
              alt="Logo"
              width={90}
              height={90}
              style={{ objectFit: "contain" }}
            />
            <h4 id="title">StockSee</h4>
          </div>

          {/* Stock Symbol Input */}
          <input
            type="text"
            value={stockSymbol}
            onChange={(e) => setStockSymbol(e.target.value)}
            placeholder="Enter stock symbol"
            className="mb-4 p-2 border rounded"
          />

          {/* Submit Button */}
          <button
            onClick={handleSubmit}
            id="submit-btn"
            className="p-2 bg-blue-500 text-white rounded flex items-center justify-center"
            disabled={loading}
          >
            {loading ? (
              <>
                <div className="spinner-border text-light" role="status">
                  <span className="sr-only">Loading...</span>
                </div>
                <p className="loading-note">
                  The first request may take longer as the backend powers on
                </p>
              </>
            ) : (
              "Submit"
            )}
          </button>

          {/* Company Office Image */}
          {realImages ? (
            <Image
              src={realImages}
              alt="Dynamic Image"
              width={500} // specify dimensions as per your needs
              height={500}
            />
          ) : null}
        </div>

        {/* MAIN CHART SECTION */}
        {chartDisplayData && (
          <>
            <Line
              data={chartDisplayData}
              options={chartDisplayData.options}
              style={{ marginBottom: "75px" }}
            />

            {/* ABOUT SECTION */}
            <h4
              id="title"
              style={{
                textAlign: "center",
                fontSize: "2.5rem",
                marginBottom: "10px",
                marginTop: "45px",
              }}
            >
              ML Stock Predictions - ABOUT
            </h4>

            {/* GitHub Link */}
            <div style={{ marginBottom: "20px", textAlign: "center" }}>
              <p
                style={{
                  width: "100%",
                  margin: "0 auto",
                  marginBottom: "30px",
                  fontSize: "16px",
                }}
              >
                <a
                  href="https://github.com/SachinVedGupta/StockSee-with-ML-Predictions"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ textDecoration: "none", color: "#007bff" }}
                >
                  View this project on GitHub
                </a>
              </p>
            </div>

            {/* Expandable Summary Section */}
            <div
              style={{
                marginBottom: "20px",
                textAlign: "center",
                width: "80%",
                margin: "20px auto",
              }}
            >
              <button
                onClick={() => setShowSummary(!showSummary)}
                style={{
                  width: "100%",
                  padding: "15px",
                  fontSize: "18px",
                  fontWeight: "bold",
                  backgroundColor: "#3b82f6",
                  color: "white",
                  border: "none",
                  borderRadius: "8px",
                  cursor: "pointer",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <span>SUMMARY</span>
                <span>{showSummary ? "▲" : "▼"}</span>
              </button>
              {showSummary && (
                <div
                  style={{
                    marginTop: "15px",
                    textAlign: "left",
                    padding: "20px",
                    backgroundColor: "#f3f4f6",
                    borderRadius: "8px",
                  }}
                >
                  <p style={{ marginBottom: "15px" }}>
                    The yellow dots on the graph above represent predictions for
                    the next 50 days of stock prices (into the future), made
                    using a TensorFlow LSTM machine learning model. Each
                    prediction is based on a batch of 200 previous daily stock
                    prices, and the model forecasts the prices for the upcoming
                    50 days. A total of 1500 days of historical data is utilized
                    in the training and validation process for each stock. By
                    repeatedly training the model on all 30 stocks in the DOW
                    JONES, a more comprehensive model has been created, of which
                    is used to predicted the entered stock.
                  </p>
                  <p style={{ marginBottom: "15px" }}>
                    The model takes as input both the stock's daily closing
                    prices and sentiment scores derived from public news
                    articles. These sentiment scores are generated through a
                    custom natural language processing (NLP) model, developed
                    using TensorFlow and trained on a Kaggle dataset. The NLP
                    model analyzes news articles related to the company,
                    gathered via a news API, to assign a sentiment score for
                    each day. By including not only the historical stock prices
                    but also external factors like public sentiment and company
                    news, the model is better equipped to predict future stock
                    prices. Simply relying on past prices is insufficient, as
                    factors such as company performance, innovation, and public
                    perception play a critical role, making sentiment analysis
                    an essential input for the prediction model. Furthermore, by
                    creating a comprehensive model trained on 30 stocks (the
                    ones in the DOW JONES), the prediction model used becomes
                    even more accurate.
                  </p>
                  <p>
                    Note: The loss/accuracy curves below will stay constant in
                    repeated entries since the pre-trained (on the 30 DOW JONES
                    stocks) model is being loaded in. By going in LOCAL, one can
                    then further train the prediction model and/or sentiment
                    analysis ML models. It can also be set so that every new
                    stock ticker entry further trains and improves the
                    prediction model, though this is not a feature in the
                    deployed version due to RAM constraints.
                  </p>
                </div>
              )}
            </div>

            {/* ML MODEL GRAPHS SECTION (Expandable) */}
            <div
              style={{
                marginBottom: "20px",
                textAlign: "center",
                width: "80%",
                margin: "20px auto",
              }}
            >
              <button
                onClick={() => setShowGraphs(!showGraphs)}
                style={{
                  width: "100%",
                  padding: "15px",
                  fontSize: "18px",
                  fontWeight: "bold",
                  backgroundColor: "#3b82f6",
                  color: "white",
                  border: "none",
                  borderRadius: "8px",
                  cursor: "pointer",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <span>ML MODEL GRAPHS</span>
                <span>{showGraphs ? "▲" : "▼"}</span>
              </button>
              {showGraphs && (
                <div style={{ marginTop: "15px" }}>
                  {/* 2x2 Grid for ML Model Images */}
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: "20px",
                      marginTop: "20px",
                    }}
                  >
                    {/* Stock Predictions Graph */}
                    <div
                      style={{
                        textAlign: "center",
                        padding: "20px",
                        paddingBottom: "10px",
                      }}
                    >
                      <div
                        style={{
                          display: "inline-block",
                          width: "500px", // Same width as the graph
                          border: "4px solid black",
                          borderRadius: "10px",
                          padding: "10px",
                        }}
                      >
                        <Image
                          src={`${imageUrls[3]}?${new Date().getTime()}`} // Adding timestamp
                          alt="Dynamic Image 1"
                          width={500}
                          height={500}
                          style={{ borderRadius: "10px" }}
                        />
                      </div>
                      <p
                        style={{
                          marginTop: "10px",
                          width: "500px",
                          marginLeft: "auto",
                          marginRight: "auto",
                        }}
                      >
                        Each stock prediction consists of 50 days and is based
                        on the previous 200-day time window as an input to the
                        model. The graph shows the past 1500 daily prices for
                        the entered stock. Note, the main StockSee graph above
                        features just the recent 700 days (Time = 800-1500). Can
                        compare the prediction line (in red) with the actual
                        price line (in blue) for a general idea of the model's
                        accuracy/performance.
                      </p>
                    </div>

                    {/* Stock Loss Graph */}
                    <div
                      style={{
                        textAlign: "center",
                        padding: "20px",
                        paddingBottom: "10px",
                      }}
                    >
                      <div
                        style={{
                          display: "inline-block",
                          width: "500px", // Same width as the graph
                          border: "4px solid black",
                          borderRadius: "10px",
                          padding: "10px",
                        }}
                      >
                        <Image
                          src={`${imageUrls[2]}?${new Date().getTime()}`} // Adding timestamp
                          alt="Stock Loss"
                          width={500}
                          height={500}
                          style={{ borderRadius: "10px" }}
                        />
                      </div>
                      <p
                        style={{
                          marginTop: "10px",
                          width: "500px",
                          marginLeft: "auto",
                          marginRight: "auto",
                        }}
                      >
                        Represents the loss graph for the ML model trained on
                        the stock data (the 30 DOW JONES stocks).
                      </p>
                    </div>

                    {/* Sentiment Accuracy Graph */}
                    <div
                      style={{
                        textAlign: "center",
                        padding: "20px",
                      }}
                    >
                      <div
                        style={{
                          display: "inline-block",
                          width: "500px", // Same width as the graph
                          border: "4px solid black",
                          borderRadius: "10px",
                          padding: "10px",
                        }}
                      >
                        <Image
                          src={`${imageUrls[0]}?${new Date().getTime()}`} // Adding timestamp
                          alt="Sentiment Accuracy"
                          width={500}
                          height={500}
                          style={{ borderRadius: "10px" }}
                        />
                      </div>
                      <p
                        style={{
                          marginTop: "10px",
                          width: "500px",
                          marginLeft: "auto",
                          marginRight: "auto",
                        }}
                      >
                        Accuracy graph for the sentiment analysis NLP ML model.
                        Since the model is not retrained/changing/updating
                        unless the specific retraining function is called, this
                        will generally remain the same as the saved model is
                        just loaded in/used.
                      </p>
                    </div>

                    {/* Sentiment Loss Graph */}
                    <div
                      style={{
                        textAlign: "center",
                        padding: "20px",
                      }}
                    >
                      <div
                        style={{
                          display: "inline-block",
                          width: "500px", // Same width as the graph
                          border: "4px solid black",
                          borderRadius: "10px",
                          padding: "10px",
                        }}
                      >
                        <Image
                          src={`${imageUrls[1]}?${new Date().getTime()}`} // Adding timestamp
                          alt="Sentiment Loss"
                          width={500}
                          height={500}
                          style={{ borderRadius: "10px" }}
                        />
                      </div>
                      <p
                        style={{
                          marginTop: "10px",
                          width: "500px",
                          marginLeft: "auto",
                          marginRight: "auto",
                        }}
                      >
                        Loss (MSE - Mean Squared Error) graph for the sentiment
                        analysis NLP ML model. Since the model is not
                        retrained/changing/updating unless the specific
                        retraining function is called, this will generally
                        remain the same as the saved model is just loaded
                        in/used.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </main>
    </>
  );
}
