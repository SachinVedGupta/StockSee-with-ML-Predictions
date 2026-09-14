const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const ts = require('typescript');

const source = ts.transpileModule(fs.readFileSync('src/app/api/gemini/route.ts', 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 }
}).outputText;

const sourceArticle = {title: 'AAPL reported event', url: 'https://example.com/story', published_at: '2024-08-05T12:00:00Z'};

function route({ key, result = '2024-08-05: Context', status, fail = false, newsKey = "news-test", articles = [sourceArticle], newsStatus = 200, backupKey } = {}) {
  let calls = 0;
  let fetches = 0;
  const logs = [];
  const queries = [];
  class FetchError extends Error { constructor() { super('private-provider-detail'); this.status = status; } }
  const sandbox = {
    exports: {}, process: { env: { GEMINI_API_KEY: key, GEMINI_MODEL: 'test-model', NEWS_API_TOKEN: newsKey, NEXT_NEWS_API_TOKEN: backupKey } },
    URL, URLSearchParams, AbortSignal, fetch: async url => { queries.push(new URL(url).searchParams.get("search")); fetches++; return Response.json({ data: articles }, {status: Array.isArray(newsStatus) ? newsStatus[fetches - 1] : newsStatus}); },
    console: { error: (...args) => logs.push(args) },
    require(name) {
      if (name === 'next/server') return { NextResponse: Response };
      if (name === '@google/generative-ai') return {
        GoogleGenerativeAIFetchError: FetchError,
        GoogleGenerativeAI: class {
          getGenerativeModel(config, options) {
            assert.equal(config.model, 'test-model');
            assert.equal(options.timeout, 30000);
            return { generateContent: async () => {
              calls++;
              if (fail) throw new FetchError();
              return { response: { text: () => result } };
            } };
          }
        }
      };
      throw new Error(`Unexpected import ${name}`);
    }
  };
  vm.runInNewContext(source, sandbox);
  return { post: sandbox.exports.POST, calls: () => calls, fetches: () => fetches, queries, logs };
}
const request = body => new Request('http://localhost/api/gemini', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
});
const valid = { stockSymbol: 'AAPL', date: ['2024-08-05'] };

test('rejects malformed JSON and invalid inputs without contacting Google', async () => {
  const r = route({ key: 'test' });
  assert.equal((await r.post(new Request('http://localhost', { method: 'POST', body: '{' }))).status, 400);
  for (const body of [null, {}, { ...valid, stockSymbol: '' }, { ...valid, date: ['2024-02-30'] }, { ...valid, date: Array(31).fill('2024-08-05') }]) {
    assert.equal((await r.post(request(body))).status, 400);
  }
  assert.equal(r.calls(), 0);
});
test('empty dates skip Gemini', async () => {
  const r = route({ key: 'test' });
  assert.deepEqual(await (await r.post(request({ ...valid, date: [] }))).json(), { news: [] });
  assert.equal(r.calls(), 0);
});
test('missing key degrades gracefully', async () => {
  const response = await route().post(request(valid));
  assert.equal(response.status, 200);
  assert.equal((await response.json()).code, 'GEMINI_NOT_CONFIGURED');
});
test('returns parsed explanations', async () => {
  const response = await route({ key: 'test', result: ' 2024-08-05: Context\n\n-\n2024-08-06: Invented ' }).post(request(valid));
  assert.deepEqual((await response.json()).news, ['2024-08-05: Context']);
});
for (const [status, expected] of [[429, 503], [403, 502], [undefined, 502]]) {
  test(`provider failure ${status} is sanitized`, async () => {
    const r = route({ key: 'test', status, fail: true });
    const response = await r.post(request(valid));
    assert.equal(response.status, expected);
    assert.equal((await response.json()).code, 'GEMINI_UNAVAILABLE');
    assert.ok(!JSON.stringify(r.logs).includes('private-provider-detail'));
  });
}

 test('only cites articles in the lead-up window with HTTPS links', async () => {
  const r = route({key: 'test', newsKey: 'news-test', articles: [
    {title: 'Wrong date', url: 'https://example.com/wrong', published_at: '2025-08-05'},
    {title: 'Unsafe', url: 'javascript:alert(1)', published_at: '2024-08-05'},
    {title: 'AAPL matching source', url: 'https://example.com/story', published_at: '2024-08-05T12:00:00Z', description: 'Historical context'}
  ]});
  const result = await (await r.post(request(valid))).json();
  assert.deepEqual(result.sources, [{date:'2024-08-05',title:'AAPL matching source',url:'https://example.com/story',description:'Historical context',publishedDate:'2024-08-05'}]);
});

test('missing dated sources skip Gemini instead of generating placeholders', async () => {
  const r = route({key:'test', articles:[]});
  const body = await (await r.post(request(valid))).json();
  assert.equal(body.code, 'NEWS_SOURCES_UNAVAILABLE');
  assert.deepEqual(body.news, []);
  assert.equal(r.calls(), 0);
});
test('quota exhaustion gives one useful message and skips Gemini', async () => {
  const r = route({key:'test', newsStatus:402});
  const body = await (await r.post(request(valid))).json();
  assert.match(body.warning, /allowance/);
  assert.deepEqual(body.news, []);
  assert.equal(r.calls(), 0);
});
test('secondary token restores sources without a quota warning', async () => {
  const r = route({key:'test', backupKey:'backup-test', newsStatus:[402,200]});
  const body = await (await r.post(request(valid))).json();
  assert.equal(r.fetches(), 2);
  assert.equal(r.calls(), 1);
  assert.equal(body.warning, undefined);
  assert.equal(body.sources.length, 1);
});
test('accepts earlier events, excludes future or stale reports, and keeps highlight date', async () => {
  const r = route({key:'test', articles:[
    {...sourceArticle, published_at:'2024-08-01T12:00:00Z'},
    {...sourceArticle, published_at:'2024-08-06T00:00:00Z'},
    {...sourceArticle, published_at:'2024-07-21T23:59:59Z'}
  ]});
  const body = await (await r.post(request(valid))).json();
  assert.equal(body.sources.length, 1);
  assert.equal(body.sources[0].date, '2024-08-05');
  assert.equal(body.sources[0].publishedDate, '2024-08-01');
});
test('event-focused search requires the ticker or known company name in every alternative', async () => {
  const r = route({key:'test'});
  await r.post(request(valid));
  assert.ok(r.queries[0].startsWith('("AAPL" | "Apple") + ('));
  assert.ok(r.queries[0].endsWith(')'));
  assert.ok(r.queries[0].includes('launch*'));
  assert.ok(r.queries[0].includes('CEO'));
  assert.ok(r.queries[0].includes('earnings'));
});
test('rejects unrelated-company articles even when the provider returns them', async () => {
 const r = route({key:'test', articles:[{...sourceArticle,title:'Microsoft launches a product',description:'Compared with Apple products'}]});
 const body = await (await r.post(request(valid))).json();
 assert.equal(body.sources.length, 0);
 assert.equal(r.calls(), 0);
});
