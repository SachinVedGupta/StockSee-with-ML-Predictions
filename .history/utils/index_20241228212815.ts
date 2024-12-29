import { GoogleGenerativeAI } from "@google/generative-ai";
import dotenv from "dotenv";

dotenv.config();

const genAI = new GoogleGenerativeAI(process.env.API_KEY!);

export async function fetchStockNews(ticker: string, date: []) {
  const model = genAI.getGenerativeModel({ model: "gemini-pro" });
  const prompt = `Tell me an event (a real world event, company announcement/release, etc) that happened on ${date} or 6 days before that caused the large spike/shift in ${ticker} stock   why the ${ticker} stock price significantly shifted following ${date}. Formatting: Be in present tense and ensure the data is on one continuous line. Put Not Found if approriate data is not found or accessible.`;

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