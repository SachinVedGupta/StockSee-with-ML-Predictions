import { GoogleGenerativeAI } from "@google/generative-ai";
import dotenv from "dotenv";

dotenv.config();

const genAI = new GoogleGenerativeAI("AIzaSyCqBxTRJPLjaTUXFrwMhOo5dpUx5fal2mE");

// Timeout wrapper for API calls
async function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  const timeout = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error('Request timeout')), timeoutMs)
  );
  return Promise.race([promise, timeout]);
}

export async function fetchStockNews(ticker: string, date: []) {
  try {
    const model = genAI.getGenerativeModel({ 
      model: "gemini-2.5-flash",
      generationConfig: {
        temperature: 0.5,
        maxOutputTokens: 200, // Keep response very short for speed
      }
    });
    
    // Original: no date limiting
    const prompt = `Tell me the reason (an event) why ${ticker} stock price shifted on ${date} in the past. Give event, how it will impact future stock price, all on one continuous line.`;

    console.log('Sending prompt to Gemini API:', prompt);
    
    // Add 8 second timeout for Gemini API call (Vercel free tier limit)
    const result = await withTimeout(
      model.generateContent(prompt),
      8000
    );
    
    const response = result.response;
    const text = response.text()
    console.log('Gemini response received');
    return text.split('\n').filter(phrase => phrase.trim() !== ''); // Split by newline and filter out empty strings
  } catch (error) {
    console.error('Error in fetchStockNews:', error);
    if (error instanceof Error && error.message === 'Request timeout') {
      throw new Error('Gemini API request timed out. Please try again.');
    }
    throw new Error(`Failed to fetch stock news: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export async function fetchStockNews2(ticker: string, date: []) {
  const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
  const prompt = `repeat ${ticker} company exactly back to me `;

  const result = await model.generateContent(prompt);
  
  const response = result.response;
  const text = response.text()
  return text
}