// Wake the free backend without starting duplicate, expensive prediction jobs.
export async function waitForBackend(baseUrl: string, onWaiting: () => void) {
  const deadline = Date.now() + 180_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${baseUrl}/health`, {
        cache: "no-store",
        signal: AbortSignal.timeout(Math.min(15_000, deadline - Date.now())),
      });
      if (response.ok && (await response.json()).status === "ok") return;
      // Allow a frontend/backend rolling deployment with the previous API.
      if (response.status === 404) return;
      if (![502, 503, 504].includes(response.status)) {
        throw new Error(`Prediction service health check failed (HTTP ${response.status}).`);
      }
    } catch (error) {
      if (error instanceof Error && error.message.startsWith("Prediction service health")) throw error;
    }
    onWaiting();
    await new Promise(resolve => setTimeout(resolve, Math.min(2000, Math.max(0, deadline - Date.now()))));
  }
  throw new Error("The prediction server is still starting or temporarily unavailable. Please try again in a few minutes.");
}

export async function fetchPrediction(baseUrl: string, ticker: string, onWaiting: () => void) {
  const queueDeadline = Date.now() + 180_000;
  while (true) {
    const response = await fetch(`${baseUrl}/predicted_prices?ticker=${encodeURIComponent(ticker)}`, {
      signal: AbortSignal.timeout(120_000),
    });
    // Retry only requests explicitly rejected before inference started.
    // Network errors/timeouts must not submit a duplicate prediction job.
    if (response.status !== 429) return response;
    const body = await response.clone().json().catch(() => null);
    if (body?.code !== "PREDICTION_BUSY") return response;
    if (Date.now() >= queueDeadline) throw new Error("The prediction server is busy. Please try again shortly.");
    onWaiting();
    await new Promise(resolve => setTimeout(resolve, 3000));
  }
}
