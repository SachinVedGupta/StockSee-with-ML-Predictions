

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
      const chartResponse = await fetch(`http://127.0.0.1:5000/historical_prices?ticker=${stockSymbol}`);
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
            data: combinedPrices,
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
          <Line data={chartDisplayData} options={chartDisplayData.options} />
          <Image
          src={"/sentiment_accuracy.png"}
          alt="Dynamic Image"
          width={500} // specify dimensions as per your needs
          height={500}
        />    
        )}

      </main>
    </>
  );
}
