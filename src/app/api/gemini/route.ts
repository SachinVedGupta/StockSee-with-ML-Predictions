import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextResponse } from "next/server"

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

async function fetchStockNews(ticker: string, date: []) {
  try {
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
    const prompt = `Tell me the reason (an event) why ${ticker} stock price shifted on ${date} in the past. Give event, how it will impact future stock price, all on one continuous line.`;

    console.log('Sending prompt to Gemini API:', prompt);
    const result = await model.generateContent(prompt);
    const response = result.response;
    const text = response.text()
    console.log('Gemini response received');
    return text.split('\n').filter(phrase => phrase.trim() !== '');
  } catch (error) {
    console.error('Error in fetchStockNews:', error);
    throw new Error(`Failed to fetch stock news: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

// Increase timeout for this API route
export const maxDuration = 60; // 60 seconds

export async function POST(req: Request) {
    try {
        const {stockSymbol, date} = await req.json()
        console.log('Fetching news for:', stockSymbol, date);
        
        if (!process.env.GEMINI_API_KEY) {
            console.error('GEMINI_API_KEY is not set in environment variables');
            return NextResponse.json(
                { error: 'GEMINI_API_KEY is not configured' },
                { status: 500 }
            )
        }
        
        const news = await fetchStockNews(stockSymbol, date)
        console.log('News fetched successfully');
        
        return NextResponse.json({ news })
    } catch (error) {
        console.error('Error in Gemini API route:', error);
        return NextResponse.json(
            { error: error instanceof Error ? error.message : 'An error occurred' },
            { status: 500 }
        )
    }
}