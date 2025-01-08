import { fetchStockNews } from "../../../../utils"

export async function POST(req: Request, res: Response) {
    const {stockSymbol, date} = await req.json()
    // console.log(stockSymbol, date);
    const news =  await fetchStockNews(stockSymbol, date)
    
    return Response.json({ news })
}