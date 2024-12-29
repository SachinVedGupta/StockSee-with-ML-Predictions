import { fetchStockNews2 } from "../../../../utils"

export async function POST(req: Request, res: Response) {
  const {stockSymbol, date} = await req.json()
  const news =  await fetchStockNews2(stockSymbol, date)
  return Response.json({ news })
}
