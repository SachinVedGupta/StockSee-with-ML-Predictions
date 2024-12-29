import { GoogleGenerativeAI } from "@google/generative-ai";
import dotenv from "dotenv";

dotenv.config();

const genAI = new GoogleGenerativeAI(process.env.API_KEY!);

export async function fetchStockNews(ticker: string, date: []) {
  const model = genAI.getGenerativeModel({ model: "gemini-pro" });
  const prompt = `Provide detailed information on the real-world event or company-related announcement that influenced the significant spike in the stock price of the company represented by the ticker symbol ${ticker} on the date ${date}. The event could include, but is not limited to, major company announcements (e.g., earnings reports, mergers/acquisitions, partnerships, new product launches), significant regulatory changes, industry-wide developments, or macroeconomic factors. \n Focus on events that plausibly correlate with or directly caused the observed stock price movement that begins on ${date}. If the specific event cannot be identified, include contextual data such as major news headlines, market sentiment, or trends for the company and its sector during that time. \n Formatting: \n Summarize the event(s) in a concise, single line. \n Ensure accuracy and relevance to ${ticker} on ${date}.`;

    //   const prompt = `Tell me the reason why ${ticker} stocks shifted on ${date} in the past. Give what event happened that day. Be in present tense and say the future impact the event will have on the companies stock. The date and the source url. Everything for a specific day must be on one continuous line.`;


  const result = await model.generateContent(prompt);
  const response = result.response;
  if
  const text = response.text()
  return text.split('\n').filter(phrase => phrase.trim() !== ''); // Split by newline and filter out empty strings
}

export async function fetchStockNews2(ticker: string, date: []) {
  const model = genAI.getGenerativeModel({ model: "gemini-pro" });
  const prompt = `repeat ${ticker} company exactly back to me `;

  const result = await model.generateContent(prompt);
  
  const response = result.response;
  const text = response.text()
  return text
}