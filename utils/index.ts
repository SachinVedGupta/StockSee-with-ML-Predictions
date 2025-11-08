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
  // Limit to first 5 dates to stay under timeout (declare outside try for catch access)
  const limitedDates = Array.isArray(date) ? date.slice(0, 5) : [date];
  
  try {
    const model = genAI.getGenerativeModel({ 
      model: "gemini-2.5-flash",
      generationConfig: {
        temperature: 0.5,
        maxOutputTokens: 200, // Keep response very short for speed
      }
    });
    
    console.log(`Processing ${limitedDates.length} dates out of ${Array.isArray(date) ? date.length : 1} total`);
    
    const prompt = `Tell me the reason (an event) why ${ticker} stock price shifted on ${limitedDates} in the past. Give event, how it will impact future stock price, all on one continuous line.`;

    console.log('Sending prompt to Gemini API:', prompt);
    
    // Add 9 second timeout for Gemini API call (Vercel has 10s limit)
    const result = await withTimeout(
      model.generateContent(prompt),
      9000
    );
    
    const response = result.response;
    const text = response.text()
    console.log('Gemini response received');
    return text.split('\n').filter(phrase => phrase.trim() !== ''); // Split by newline and filter out empty strings
  } catch (error) {
    console.error('Error in fetchStockNews:', error);
    if (error instanceof Error && error.message === 'Request timeout') {
      // Return placeholder data instead of throwing
      console.log('Timeout - returning placeholder data');
      return limitedDates.map((d: string) => `${d}: Event data temporarily unavailable due to timeout`);
    }
    console.error('Gemini API error - returning placeholder');
    // Return placeholder instead of throwing to avoid breaking the UI
    return ['Unable to fetch news data. Please try again later.'];
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