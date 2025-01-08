import { fetchStockNews } from "../../../../utils"

export async function POST(req: Request) {
  try {
    const { stockSymbol, date } = await req.json();
    const news = await fetchStockNews(stockSymbol, date);
    return new Response(JSON.stringify({ news }), { status: 200 });
  } catch (error: any) {
    console.error("Error in /api/gemini route:", error.message);
    return new Response(
      JSON.stringify({ error: "Failed to fetch stock news. Please try again." }),
      { status: 500 }
    );
  }
}
