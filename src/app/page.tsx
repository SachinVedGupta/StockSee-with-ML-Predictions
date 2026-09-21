"use client";

// IMPORTS
import React, { useEffect, useState } from "react";
import "chart.js/auto";
import { Line } from "react-chartjs-2";
import axios from "axios";
import { waitForBackend, fetchPrediction } from "../lib/waitForBackend";
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
  const backendURL =
    process.env.NEXT_PUBLIC_STOCKSEE_BACKEND_URL || deployedBackendURL;

  // COMPONENT STATE
  const [stockSymbol, setStockSymbol] = useState("");
  const [chartDisplayData, setChartDisplayData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [newsWarning, setNewsWarning] = useState("");
  const [showSummary, setShowSummary] = useState(false);
  const [showGraphs, setShowGraphs] = useState(false);
  const [realImages, setRealImages] = useState<string | null>(null);
  const [submittedTicker, setSubmittedTicker] = useState("");
  const [storiesWarning, setStoriesWarning] = useState("");
  const [stories, setStories] = useState<Array<{title: string; description: string; url: string; source: string; published_at: string; image_url?: string}>>([]);
  const [explanations, setExplanations] = useState<string[]>([]);
  const [explanationSources, setExplanationSources] = useState<Array<{date: string; title: string; url: string; publishedDate?: string}>>([]);
  const [imageUrls, setImageUrls] = useState<string[]>([]);

  // HELPER FUNCTIONS

  function getLogoUrl(ticker: string) {
    const token = process.env.NEXT_PUBLIC_LOGO_DEV_TOKEN;
    if (!token) return null;
    const query = new URLSearchParams({ token, size: "128", fallback: "404" });
    return `https://img.logo.dev/ticker/${encodeURIComponent(ticker)}?${query}`;
  }

  // DATA FETCHING AND CHART GENERATION

  /**
   * Main function to fetch stock data, predictions, and news
   * Processes the data and generates chart configuration
   */
  async function handleSubmit() {
    const ticker = stockSymbol.trim().toUpperCase();
    setErrorMessage("");
    setNewsWarning("");
    setStoriesWarning("");
    setStories([]);
    setExplanations([]);
    setExplanationSources([]);
    if (!/^[A-Z0-9.^=-]{1,20}$/.test(ticker)) {
      setErrorMessage("Enter a valid stock symbol, for example AAPL.");
      return;
    }
    setLoading(true);
    setChartDisplayData(null);
    setSubmittedTicker(ticker);
    setRealImages(getLogoUrl(ticker));
    try {
      setLoadingMessage("Connecting to the prediction server…");
      await waitForBackend(backendURL, () => setLoadingMessage("Waking the prediction server. This can take a few minutes…"));
      setLoadingMessage("Calculating predictions…");
      // Fetch predicted prices from backend
      const chartResponse = await fetchPrediction(backendURL, ticker,
        () => setLoadingMessage("Waiting for the prediction server, then calculating your results…"));
      if (!chartResponse.ok) {
        throw new Error(`Prediction service is unavailable (HTTP ${chartResponse.status}). Please try again shortly.`);
      }
      const chartData = await chartResponse.json();
      if (!Array.isArray(chartData) || !Array.isArray(chartData[0]) ||
          !Array.isArray(chartData[1]) || chartData[0].length !== chartData[1].length ||
          chartData[0].indexOf("Seperate-Dates") < 1) {
        throw new Error("The prediction service returned invalid data. Please try again.");
      }

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
      for (let i = windowSize; i < dates.indexOf("Seperate-Dates"); i++) {
        const pastPrice = prices[i - windowSize];
        const currentPrice = prices[i];
        const delta = (currentPrice - pastPrice) / pastPrice;

        if (Math.abs(delta) > threshold) {
          changes.push({
            index: i,
            x: dates[i],
            y: currentPrice,
            delta: delta,
          });
          date.push(dates[i]);
          i = i + minDistance;
        }
      }

      let newsItems: string[] = [];

      // Fetch independent enrichments together; each reports its own availability.
      const enrichmentPromise = fetch(`${backendURL}/news?ticker=${encodeURIComponent(ticker)}`, { signal: AbortSignal.timeout(25000) })
          .then(async response => {
            const result = await response.json();
            if (!response.ok) {
              setStoriesWarning(result.warning || "News stories are unavailable.");
              return;
            }
            const articles = Array.isArray(result.articles) ? result.articles.filter((article: any) =>
              typeof article.title === "string" && typeof article.url === "string" && article.url.startsWith("https://")) : [];
            setStories(articles);
            if (!articles.length) setStoriesWarning("No matching news stories were found.");
          })
          .catch(() => setStoriesWarning("News stories could not be loaded from the backend."));

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
            label: `${ticker} Stock Price`,
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

      // Render the chart before waiting for optional explanations.
      if (date.length > 0) {
        try {
          const news = await axios.post("/api/gemini", { stockSymbol: ticker, date, movements: changes.map(change => ({ date: change.x, changePercent: change.delta * 100 })) }, { timeout: 50000 });
          newsItems = Array.isArray(news.data?.news) ? news.data.news : [];
          setExplanations(newsItems);
          setExplanationSources(Array.isArray(news.data?.sources) ? news.data.sources.filter((source: any) => typeof source.url === "string" && source.url.startsWith("https://")) : []);
          setNewsWarning(news.data?.warning || "");
        } catch {
          setNewsWarning("AI explanations are temporarily unavailable. Predictions are still available.");
        }
      }

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
                  let theanswer = "";
                  for (const item of newsItems) {
                    if (item.includes(point.x)) {
                      theanswer = item;
                    }
                  }
                  return [`Price: ${tooltipItem.raw.toFixed(2)}`, ...(theanswer ? (theanswer.match(/.{1,90}(?:\s|$)/g) || [theanswer]) : [])];
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
      await enrichmentPromise;
    } catch (error) {
      setErrorMessage(error instanceof Error && error.name === "TimeoutError"
        ? "The prediction service took too long to respond. Please try again shortly."
        : error instanceof Error ? error.message : "Could not load predictions. Please try again.");
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
                  {loadingMessage}
                </p>
              </>
            ) : (
              "Submit"
            )}
          </button>

          {errorMessage && <p role="alert" className="mt-4 text-red-700">{errorMessage}</p>}

          {submittedTicker && (
            <div className="mt-5">
              {realImages ? <Image
                src={realImages}
                alt={`${submittedTicker} company logo`}
                onError={() => setRealImages(null)}
                width={128}
                height={128}
                className="h-32 w-32 object-contain"
              /> : <span aria-label="Company logo unavailable" className="flex h-16 w-16 items-center justify-center rounded bg-stone-100 text-xs font-bold">{submittedTicker}</span>}
            </div>
          )}
        </div>

        {/* MAIN CHART SECTION */}
        {chartDisplayData && (
          <>
            <p className="my-4 text-sm">Educational estimates, not financial advice.</p>
            <Line
              aria-label="Historical stock prices and future predictions"
              data={chartDisplayData}
              options={chartDisplayData.options}
              style={{ marginBottom: "75px" }}
            />

            <section className="w-full max-w-4xl mb-10" aria-label="News and AI context">
              <details className="mb-6" key={`stories-${submittedTicker}`}>
                <summary className="cursor-pointer text-xl font-semibold mb-3">Recent Stories relating to: {submittedTicker}</summary>
              {storiesWarning && <p role="status" className="mb-4">{storiesWarning}</p>}
              <div className="grid gap-4">
                {stories.map(story => (
                  <article key={story.url} className="rounded-lg border border-gray-200 bg-white p-5">
                    {story.image_url?.startsWith("https://") && <Image
                      src={story.image_url}
                      alt=""
                      width={96}
                      height={64}
                      className="mb-3 h-16 w-24 rounded object-contain"
                      onError={event => { event.currentTarget.style.display = "none"; }}
                    />}
                    <a href={story.url} target="_blank" rel="noopener noreferrer" className="text-blue-700 font-semibold underline">{story.title}</a>
                    <p className="text-sm text-gray-600 mt-1">{story.source} · {story.published_at.slice(0, 10)}</p>
                    <p className="mt-2">{story.description}</p>
                  </article>
                ))}
              </div>
              </details>
              <details key={`context-${submittedTicker}`}>
                <summary className="cursor-pointer text-xl font-semibold mb-3">AI News Context for Highlighted Dates</summary>
              <p className="text-sm mb-3">AI context considers news from the preceding two weeks; possible connections are not proven causes.</p>
              {newsWarning && <p role="status" className="mb-3">{newsWarning}</p>}
              {explanations.length > 0 ? <ul className="space-y-3">
                {explanations.map((explanation, index) => <li key={index} className="rounded-lg bg-white border border-gray-200 p-4">
                  <p>{explanation}</p>
                  {explanationSources.filter(source => explanation.includes(source.date)).map(source => <a key={source.url} href={source.url} target="_blank" rel="noopener noreferrer" className="block text-blue-700 underline text-sm mt-2">Source{source.publishedDate ? ` (${source.publishedDate})` : ""}: {source.title}</a>)}
                </li>)}
              </ul> : (!newsWarning && <p>{loading ? "Loading historical context…" : "No dated news sources were found for these chart highlights."}</p>)}
              </details>
            </section>

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
