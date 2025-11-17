import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextResponse } from "next/server"

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

export const maxDuration = 60;

// get stock price shift explanations for a certain ticker on many dates
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
        
        const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-lite" });
        const prompt = `Tell me the reason (an event) why ${stockSymbol} stock price shifted on ${date} in the past. Give event, how it will impact future stock price, all on one continuous line.`;

        console.log('Sending prompt to Gemini API:', prompt);
        const result = await model.generateContent(prompt);
        const response = result.response.text();
        const news = response.split('\n').filter(phrase => phrase.trim() !== '');

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