import { GoogleGenerativeAI, GoogleGenerativeAIFetchError } from "@google/generative-ai";
import { NextResponse } from "next/server";

export const maxDuration = 60;

export async function POST(req: Request) {
  let body;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON request" }, { status: 400 });
  }

  const stockSymbol = typeof body?.stockSymbol === "string"
    ? body.stockSymbol.trim().toUpperCase() : "";
  const dates = body?.date;
  if (!/^[A-Z0-9.^=-]{1,20}$/.test(stockSymbol) || !Array.isArray(dates) ||
      dates.length > 30 || dates.some((date: unknown) =>
        typeof date !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(date) ||
        !Number.isFinite(Date.parse(date)) || new Date(date).toISOString().slice(0, 10) !== date)) {
    return NextResponse.json({ error: "Provide a ticker and up to 30 valid YYYY-MM-DD dates" }, { status: 400 });
  }
  if (dates.length === 0) return NextResponse.json({ news: [] });

  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey) {
    return NextResponse.json({ news: [], code: "GEMINI_NOT_CONFIGURED",
      warning: "AI explanations are unavailable. Predictions are still available." });
  }

  const modelName = process.env.GEMINI_MODEL?.trim() || "gemini-3.5-flash-lite";
  try {
    // Initialize per request so configuration failures stay inside error handling.
    const model = new GoogleGenerativeAI(apiKey).getGenerativeModel(
      { model: modelName }, { timeout: 15000 }
    );
    const prompt = `Provide brief historical context for ${stockSymbol} on these dates: ${Array.from(new Set(dates)).join(", ")}. Write one line per date starting with its YYYY-MM-DD date. Mention a news event only if you know it; otherwise say reliable context is unavailable. Do not invent events, imply a proven cause, or explain future prices as historical news. These are unverified AI explanations for educational use, not financial advice.`;
    const result = await model.generateContent(prompt);
    const news = result.response.text().split("\n").map(line => line.trim()).filter(Boolean);
    return NextResponse.json({ news });
  } catch (error) {
    const upstreamStatus = error instanceof GoogleGenerativeAIFetchError ? error.status : undefined;
    // SDK messages can contain request URLs and provider details. Never expose them.
    console.error("Gemini explanation request failed", { model: modelName, upstreamStatus });
    return NextResponse.json({ news: [], code: "GEMINI_UNAVAILABLE",
      warning: "AI explanations are temporarily unavailable. Predictions are still available." },
      { status: upstreamStatus === 429 ? 503 : 502 });
  }
}
