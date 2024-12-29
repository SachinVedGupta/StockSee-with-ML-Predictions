import { GoogleGenerativeAI } from "@google/generative-ai";
import dotenv from "dotenv";

dotenv.config();

const genAI = new GoogleGenerativeAI(process.env.API_KEY!);

export async function fetchStockNews(ticker: string, date: []) {
  const model = genAI.getGenerativeModel({ model: "gemini-pro" });
  const prompt = `Tell me the reason why the ${ticker} stock price shifted on ${date}. Give what real world significant event happened that. If live data is NOT shown for a specific date, put Data Not Available for that dates line. Be in present tense and say the future impact the event will have on the companies stock. Everything for a specific day must be on one continuous line.`;

    //   const prompt = `Tell me the reason why ${ticker} stocks shifted on ${date} in the past. Give what event happened that day. Be in present tense and say the future impact the event will have on the companies stock. The date and the source url. Everything for a specific day must be on one continuous line.`;


  const result = await model.generateContent(prompt);
  const response = result.response;
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