import { GoogleGenerativeAI } from "@google/generative-ai";
import dotenv from "dotenv";

dotenv.config();

const genAI = new GoogleGenerativeAI(process.env.API_KEY!);

export async function fetchStockNews(ticker: string, date: []) {
  const model = genAI.getGenerativeModel({ model: "gemini-pro" });
  const prompt = `Tell me the reason (an event) why ${ticker} stock price shifted on ${date} in the past. Give event, how it will impact future stock price, date and the so. Everything for a specific day must be on one continuous line.`;

    //   const prompt = `Tell me the real world event or ${ticker} company announcement/release that happened on/near ${date}. Formatting: Ensure the data for a specific date is on one continuous line. Put no significant event occured if approriate data is not accessible for that date.`;


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