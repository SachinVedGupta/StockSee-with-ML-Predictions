import { fetchStockNews } from "../../../../utils"
import { NextResponse } from "next/server"

export async function POST(req: Request) {
    try {
        const {stockSymbol, date} = await req.json()
        console.log('Fetching news for:', stockSymbol, date);
        
        if (!process.env.API_KEY) {
            console.error('API_KEY is not set in environment variables');
            return NextResponse.json(
                { error: 'API_KEY is not configured' },
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