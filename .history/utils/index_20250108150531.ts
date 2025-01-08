import { GoogleGenerativeAI } from "@google/generative-ai";
import dotenv from "dotenv";

dotenv.config();

const genAI = new GoogleGenerativeAI(process.env.API_KEY!);

async function retry(fn: () => Promise<any>, retries: number = 3, delay: number = 1000): Promise<any> {
  try {
    return await fn();
  } catch (err) {
    if (retries === 0) throw err;
    await new Promise(resolve => setTimeout(resolve, delay));
    return retry(fn, retries - 1, delay * 2);
  }
}

// Usage in your function
import axios from "axios";

const axiosInstance = axios.create({
  timeout: 10000, // Set a 10-second timeout
});

export async function fetchStockNews(ticker: string, date: string[]) {
  const model = genAI.getGenerativeModel({ model: "gemini-pro" });
  const prompt = `Tell me the reason why ${ticker} stocks shifted on ${date} in the past.`;

  try {
    const result = await axiosInstance.post("https://gemini-api-url", { prompt });
    return result.data.text.split('\n').filter((phrase: string) => phrase.trim() !== '');
  } catch (error: any) {
    console.error("Error fetching stock news:", error.message);
    throw new Error("Failed to fetch stock news.");
  }
}




export async function fetchStockNews2(ticker: string, date: []) {
  const model = genAI.getGenerativeModel({ model: "gemini-pro" });
  const prompt = `repeat ${ticker} company exactly back to me `;

  const result = await model.generateContent(prompt);
  
  const response = result.response;
  const text = response.text()
  return text
}