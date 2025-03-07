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
export async function fetchStockNews(ticker: string, date: string[]) {
  const model = genAI.getGenerativeModel({ model: "gemini-pro" });
  const prompt = `Tell me the reason why ${ticker} stocks shifted on ${date} in the past. Give what event happened that day. Be in present tense and say the future impact the event will have on the companies stock. The date and the source url. Everything for a specific day must be on one continuous line.`;

  try {
    const result = await model.generateContent(prompt);
    const response = result.response;
    const text = response.text();

    // Specify the type for 'phrase' explicitly
    return text.split('\n').filter((phrase: string) => phrase.trim() !== '');
  } catch (error: unknown) {
    // Type guard to check if 'error' has a 'message' property
    if (error instanceof Error) {
      console.error("Error fetching stock news:", error.message);
    } else {
      console.error("Unknown error occurred:", error);
    }
    throw new Error("Failed to fetch stock news. Please try again later.");
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