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
  let images = false;
  if (env.NEXT_PUBLIC_LOGO_DEV_TOKEN) {
    try {
      const query = new URLSearchParams({ token: env.NEXT_PUBLIC_LOGO_DEV_TOKEN, fallback: '404' });
      const response = await fetch(`https://img.logo.dev/ticker/AAPL?${query}`, { signal: AbortSignal.timeout(20000) });
      images = response.ok && Boolean(response.headers.get('content-type')?.startsWith('image/')) && (await response.arrayBuffer()).byteLength > 0;
      console.log(`Company logo: ${images ? 'PASS' : `FAILED (HTTP ${response.status})`}`);
    } catch { console.log('Company logo: FAILED (network or timeout)'); }
  } else { console.log('Company logo: NOT CONFIGURED'); }
  let news = false;
  for (const [label, token] of [['Historical news (primary)', env.NEWS_API_TOKEN], ['Historical news (backup)', env.NEXT_NEWS_API_TOKEN]]) {
    if (!token) continue;
    news = await check(label, token, () => {
      const query = new URLSearchParams({ api_token: token, search: 'AAPL', published_on: '2024-08-05', limit: '1' });
      return fetch(`https://api.thenewsapi.com/v1/news/all?${query}`, { signal: AbortSignal.timeout(20000) });
    }, data => Array.isArray(data.data));
    if (news) break;
  }
  process.exitCode = gemini && images && news ? 0 : 1;
}
main();
