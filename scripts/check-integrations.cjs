// Run from the repository root. Credentials stay in the local environment.
const path = require('node:path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env.local') });
require('dotenv').config({ path: path.join(__dirname, '.env') });

async function check(name, configured, request, validate) {
  if (!configured) {
    console.log(`${name}: NOT CONFIGURED`);
    return false;
  }
  try {
    const response = await request();
    const data = await response.json();
    if (!response.ok) {
      // Do not log provider messages or URLs: they may contain credentials.
      console.log(`${name}: FAILED (HTTP ${response.status})`);
      return false;
    }
    const passed = validate(data);
    console.log(`${name}: ${passed ? 'PASS' : 'NO RESULTS'}`);
    return passed;
  } catch {
    console.log(`${name}: FAILED (network, timeout, or invalid response)`);
    return false;
  }
}

async function main() {
  const env = process.env;
  const gemini = await check('Gemini generation', env.GEMINI_API_KEY, () => fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(env.GEMINI_MODEL || 'gemini-3.5-flash-lite')}:generateContent`, {
      method: 'POST', signal: AbortSignal.timeout(20000),
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': env.GEMINI_API_KEY },
      body: JSON.stringify({ contents: [{ parts: [{ text: 'Reply with OK.' }] }] })
    }), data => Boolean(data.candidates?.some(c => c.content?.parts?.some(p => p.text))));
  const images = await check('Google image search', env.NEXT_PUBLIC_GOOGLE_API_KEY && env.NEXT_PUBLIC_SEARCH_ENGINE_ID, () => {
    const query = new URLSearchParams({ key: env.NEXT_PUBLIC_GOOGLE_API_KEY, cx: env.NEXT_PUBLIC_SEARCH_ENGINE_ID, searchType: 'image', q: 'Apple company office', num: '1' });
    return fetch(`https://www.googleapis.com/customsearch/v1?${query}`, { signal: AbortSignal.timeout(20000) });
  }, data => Boolean(data.items?.[0]?.link));
  const news = await check('Historical news', env.NEWS_API_TOKEN, () => {
    const query = new URLSearchParams({ api_token: env.NEWS_API_TOKEN, search: 'AAPL', published_on: '2024-08-05', limit: '1' });
    return fetch(`https://api.thenewsapi.com/v1/news/all?${query}`, { signal: AbortSignal.timeout(20000) });
  }, data => Boolean(data.data?.length));
  process.exitCode = gemini && images && news ? 0 : 1;
}
main();
