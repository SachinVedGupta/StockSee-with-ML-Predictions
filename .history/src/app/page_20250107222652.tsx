

"use client"; // Add this line at the top

import React, { useEffect, useState } from 'react';
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
  async function getImageUrl(prompt: string) {
    var apiKey = "AIzaSyB41BZPIS7OSfBj81rbh1HjMdsiAYr_ATk";
    var searchEngineId = "41624844768c14c9a";
    var query = prompt; // This could be a static query or based on the prompt/content
    var url =
      "https://www.googleapis.com/customsearch/v1?key=" +
      apiKey +
      "&cx=" +
      searchEngineId +
      "&searchType=image&q=" +
      encodeURIComponent(query);
   // State to track loading

    var response = await axios.get(url);
    var results = response.data;

    if (results.items && results.items.length > 0) {
      console.log(results.items[0].link);
      return results.items[0].link; // Return the first image's URL
    } else {
      console.log("logo");
      return "https://cdn.discordapp.com/attachments/1208274732227764264/1208595162914103376/logo-no-background.png?ex=65e3daf5&is=65d165f5&hm=4d088d6dd1e4fb2cb3e9d75785f38efe79888a31a8d523724e00af838ff36143&"; // Return null if no images found
    }
  }

  const [realImages, setRealImages] = useState(null);

  async function handleSubmit() {
    setLoading(true); // Start loading
    try {
      // Backend Deployment: https://stocksee-with-ml-predictions.onrender.com/
      // LOCAL: const chartResponse = await fetch(`http://127.0.0.1:5000/historical_prices?ticker=${stockSymbol}`);
      const chartResponse = await fetch(`https://stocksee-with-ml-predictions.onrender.com/historical_prices?ticker=${stockSymbol}`);
      const chartData = await chartResponse.json();
  
      const dates = chartData[0];
      const prices = chartData[1];
  
      // Identify significant rise or drop points over a rolling window of 30 days
      const significantPoints: any[] = [];
      const windowSize = 20; // Window size for delta calculation
      const threshold = 0.12; // Example threshold for significant change (12%)
      const minDistance = 30; // Minimum distance between significant points (number of days)
  
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
  
      const news = await axios.post("/api/gemini", { stockSymbol, date });
      const images: { data: { news: string } } = await axios.post(
        "/api/images",
        { stockSymbol, date }
      );
  
      const realImages = getImageUrl(images.data.news);
      realImages.then((url) => {
        setRealImages(url);
      });
  
      
      const separateDateIndex = dates.indexOf('Seperate-Dates');

      /////////////

      // Get the value to the left of the index
      const valueToLeft = dates[separateDateIndex - 1];

      // Insert the value at the right of the separateDateIndex
      dates.splice(separateDateIndex + 1, 0, valueToLeft);

      // Similarly for prices array, assuming it follows the same structure
      prices.splice(separateDateIndex + 1, 0, prices[separateDateIndex - 1]);


      ////////////

      const firstPartDates = dates.slice(0, separateDateIndex);
      const firstPartPrices = prices.slice(0, separateDateIndex);
  
      const secondPartDates = dates.slice(separateDateIndex + 1);
      const uniqueSecondPartDates = new Set(secondPartDates);
      const secondPartPrices = prices.slice(separateDateIndex + 1);
  
      // Combine both datasets for use in the chart
      const combinedDates = [...firstPartDates, ...secondPartDates];
      const combinedPrices = [...firstPartPrices, ...secondPartPrices];

  
      setChartDisplayData({
        labels: combinedDates,
        datasets: [
          {
            label: `${stockSymbol} Stock Price`,
            backgroundColor: dates.map((_: any, i: any) =>
              // Check if the date contains "2025" (case sensitive)
              uniqueSecondPartDates.has(dates[i])
                ? "rgba(255, 165, 0, 0.5)"  // Orange for dates containing "2025"
                : "rgba(59, 130, 246, 0.5)"
            ),
            borderColor: dates.map((_: any, i: any) =>
              // Check if the date contains "2025" (case sensitive)
              uniqueSecondPartDates.has(dates[i])
                ? "rgba(255, 165, 0, 0.9)"  // Orange for dates containing "2025"
                : "rgba(59, 130, 246, 0.9)"
            ),
            data: [...firstPartPrices, ...Array(secondPartPrices.length).fill(null)],
            // data: combinedPrices,
            // data: [...firstPartPrices, ...Array(secondPartPrices.length).fill(null)];
            pointBackgroundColor: dates.map((_: any, i: any) =>
              // Check if the date contains "2025" (case sensitive)
              uniqueSecondPartDates.has(dates[i])
                ? "rgba(255, 165, 0, 1)"  // Orange for dates containing "2025"
                : changes.find((point) => point.index === i)
                ? changes.find((point) => point.index === i)!.delta > 0
                  ? "rgb(68, 246, 59)"  // Green for positive significant change
                  : "red"  // Red for negative significant change
                : "rgba(75, 192, 192, 0.6)"  // Default color for other points
            ),            
            pointRadius: dates.map((_: any, i: any) =>
              changes.find((point) => point.index === i)
                ? 5  // Show the point if it's significant or if the date contains "2025"
                : uniqueSecondPartDates.has(dates[i])
                  ? 2
                  : 0  // Hide the point if it's neither significant nor containing "2025"
            ),            
            pointHoverRadius: 10, // Increase hover size for better visibility
          },
          {
            label: "Future Predictions",
            // data: Array(dates.length).fill(null),  // Or adjust with prediction data
            data: [...Array(firstPartPrices.length).fill(null), ...secondPartPrices],
            backgroundColor: "rgba(255, 165, 0, 0.5)",  // Orange for predictions
            borderColor: "rgba(255, 165, 0, 0.9)",
            pointBackgroundColor: "rgba(255, 165, 0, 1)", // Orange
            pointRadius: 0.1,
            pointHoverRadius: 10,
          },
          {
            label: "Significant Rise Coming",
            data: Array(dates.length).fill(null),  // Modify with your data representing a large rise
            backgroundColor: "rgba(68, 246, 59, 0.5)",  // Green for large rise
            borderColor: "rgba(68, 246, 59, 0.9)",
            pointBackgroundColor: "rgba(68, 246, 59, 1)",  // Green
            pointRadius: 5,
            pointHoverRadius: 10,
          },
          {
            label: "Significant Fall Coming",
            data: Array(dates.length).fill(null),  // Modify with your data representing a large fall
            backgroundColor: "rgba(255, 0, 0, 0.5)",  // Red for large fall
            borderColor: "rgba(255, 0, 0, 0.9)",
            pointBackgroundColor: "rgba(255, 0, 0, 1)",  // Red
            pointRadius: 5,
            pointHoverRadius: 10,
          },
        ],
      });
  
      const chartOptions = {
        plugins: {
          tooltip: {
            callbacks: {
              label: function (tooltipItem: any) {
                const point = changes.find((p: any) => p.index === tooltipItem.dataIndex);
                if (point) {
                  let theanswer = "N/A";
                  for (const thing in news.data.news) {
                    if (news.data.news[thing].includes(point.x)) {
                      theanswer = news.data.news[thing];
                    }
                  }
                  return `Price: ${tooltipItem.raw.toFixed(2)},  News: ${theanswer}`;
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
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false); // Stop loading
    }
  }
  
  const [imageUrls, setImageUrls] = useState<string[]>([]);

  useEffect(() => {
    // Base URL of the Flask app
    const baseUrl = 'https://stocksee-with-ml-predictions.onrender.com/image';

    // List of image filenames (you can add or modify this list)
    const imageFilenames = ['sentiment_accuracy.png', 'sentiment_loss.png', 'stock_loss.png', 'stock_predictions.png'];

    // Construct full URLs for each image
    const imageUrls = imageFilenames.map((filename) => `${baseUrl}/${filename}`);
    
    // Set the image URLs in state
    setImageUrls(imageUrls);
  }, []);
  

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
            {/* <Image
              src={theimg}
              id="the-img"
              alt="Logo"
              width={100}
              height={100}
              max-height={100}
            /> */}
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
              'Submit'
            )}
          </button>

        {realImages ? (
          <Image
          src={realImages}
          alt="Dynamic Image"
          width={500} // specify dimensions as per your needs
          height={500}
        />
        ): null}
          

        </div>


        

        {chartDisplayData && (
  <>
    {/* Line Chart */}
    <Line data={chartDisplayData} options={chartDisplayData.options} style={{ marginBottom: "75px" }} />

    {/* Title for About Section */}
    <h4 id="title" style={{ textAlign: "center", fontSize: "2.5rem", marginBottom: "10px", marginTop: "45px" }}>
      ML Stock Predictions - ABOUT
    </h4>

    {/* GitHub Link */}
    <div style={{ marginBottom: "20px", textAlign: "center" }}>
      <p style={{ width: "100%", margin: "0 auto", marginBottom: "30px", fontSize: "16px" }}>
        <a href="https://github.com/SachinVedGupta/StockSee-with-ML-Predictions" target="_blank" rel="noopener noreferrer" style={{ textDecoration: "none", color: "#007bff" }}>
          View this project on GitHub
        </a>
      </p>
    </div>

    {/* Description Paragraphs */}
    <div style={{ marginBottom: "0px", textAlign: "center" }}>
      <p style={{ width: "80%", margin: "0 auto" }}>
        The yellow dots on the graph above represent predictions for the next 50 days of stock prices (into the future), made using a TensorFlow LSTM machine learning model. Each prediction is based on a batch of 200 previous daily stock prices, and the model forecasts the prices for the upcoming 50 days. A total of 1500 days of historical data is utilized in the training and validation process.
      </p>
      <p style={{ width: "80%", margin: "20px auto", marginBottom: "10px"}}>
        The model takes as input both the stock's daily closing prices and sentiment scores derived from public news articles. These sentiment scores are generated through a custom natural language processing (NLP) model, developed using TensorFlow and trained on a Kaggle dataset. The NLP model analyzes news articles related to the company, gathered via a news API, to assign a sentiment score for each day. By including not only the historical stock prices but also external factors like public sentiment and company news, the model is better equipped to predict future stock prices. Simply relying on past prices is insufficient, as factors such as company performance, innovation, and public perception play a critical role, making sentiment analysis an essential input for the prediction model.
      </p>
    </div>

    {/* 2x2 Grid for Images */}
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", marginTop: "20px" }}>
      
      {/* Module 1 */}
      <div style={{
        textAlign: "center",
        padding: "20px",
        paddingBottom: "10px"
      }}>
        <div style={{
          display: "inline-block",
          width: "500px", // Same width as the graph
          border: "4px solid black", 
          borderRadius: "10px",
          padding: "10px"
        }}>
          <Image
            src={`/stock_predictions.png?${new Date().getTime()}`}  // Adding timestamp
            alt="Dynamic Image 1"
            width={500}
            height={500}
            style={{ borderRadius: "10px" }}
          />
        </div>
        <p style={{ marginTop: "10px", width: "500px", marginLeft: "auto", marginRight: "auto" }}>
          Each stock prediction consists of 50 days and is based on the previous 200-day time window. The graphs shows the past 1500 days, which is the data used by the ML model. Note, the main StockSee graph above features just the recent 700 days (Time = 800-1500). Can compare the predictions with actual price graph for a general idea of the model's accuracy.
        </p>
      </div>

      {/* Module 2 */}
      <div style={{
        textAlign: "center",
        padding: "20px",
        paddingBottom: "10px"
      }}>
        <div style={{
          display: "inline-block",
          width: "500px", // Same width as the graph
          border: "4px solid black", 
          borderRadius: "10px",
          padding: "10px"
        }}>
          <Image
            src={`${imageUrls[2]}?${new Date().getTime()}`}  // Adding timestamp
            alt="Stock "
            width={500}
            height={500}
            style={{ borderRadius: "10px" }}
          />
        </div>
        <p style={{ marginTop: "10px", width: "500px", marginLeft: "auto", marginRight: "auto" }}>
          Represents the loss graph for the ML model trained on the stock data of the entered company ticker. It will be unique across various trials due to the randomness nature of an LSTM ML model.
        </p>
      </div>

      {/* Module 3 */}
      <div style={{
        textAlign: "center",
        padding: "20px"
      }}>
        <div style={{
          display: "inline-block",
          width: "500px", // Same width as the graph
          border: "4px solid black", 
          borderRadius: "10px",
          padding: "10px"
        }}>
          <Image
            src={`${imageUrls[0]}?${new Date().getTime()}`}  // Adding timestamp
            alt="Sentiment Accuracy"
            width={500}
            height={500}
            style={{ borderRadius: "10px" }}
          />
        </div>
        <p style={{ marginTop: "10px", width: "500px", marginLeft: "auto", marginRight: "auto" }}>
          Accuracy graph for the sentiment analysis NLP ML model. Since the model is not retrained/changing/updating unless the specific retraining function is called, this will generally remain the same as the saved model is just loaded in/used.
        </p>
      </div>

      {/* Module 4 */}
      <div style={{
        textAlign: "center",
        padding: "20px"
      }}>
        <div style={{
          display: "inline-block",
          width: "500px", // Same width as the graph
          border: "4px solid black", 
          borderRadius: "10px",
          padding: "10px"
        }}>
          <Image
            src={`${imageUrls[1]}?${new Date().getTime()}`}  // Adding timestamp
            alt="Sentiment Loss"
            width={500}
            height={500}
            style={{ borderRadius: "10px" }}
          />
        </div>
        <p style={{ marginTop: "10px", width: "500px", marginLeft: "auto", marginRight: "auto" }}>
          Loss (MSE - Mean Squared Error) graph for the sentiment analysis NLP ML model. Since the model is not retrained/changing/updating unless the specific retraining function is called, this will generally remain the same as the saved model is just loaded in/used.
        </p>
      </div>
      
    </div>
  </>
)}






      </main>
    </>
  );
}
