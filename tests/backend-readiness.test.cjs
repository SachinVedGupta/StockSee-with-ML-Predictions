const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const ts=require('typescript');
function setup(responses) {
 let now=0, requests=[], waiting=0;
 const exports={};
 const context={exports, Error, Promise, Math, AbortSignal,
  Date:{now:()=>now}, setTimeout:fn=>{now+=2000;fn();},
  fetch:async url=>{requests.push(url);const r=responses.shift();if(r instanceof Error)throw r;return r || {ok:false,status:503};}};
 vm.runInNewContext(ts.transpileModule(fs.readFileSync('src/lib/waitForBackend.ts','utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS}}).outputText,context);
 return {run:()=>exports.waitForBackend('https://backend.test',()=>waiting++),requests,waiting:()=>waiting};
}
test('waits through wake-up errors using only health requests',async()=>{
 const x=setup([new Error('timeout'),{ok:false,status:503},{ok:true,json:async()=>({status:'ok'})}]);
 await x.run();assert.equal(x.waiting(),2);assert.deepEqual(x.requests,Array(3).fill('https://backend.test/health'));
});
test('stops when backend cannot wake',async()=>{
 const x=setup([]);await assert.rejects(x.run(),/still starting/);assert.equal(x.requests.length,90);
});
test('does not retry permission failures',async()=>{
 const x=setup([{ok:false,status:403}]);await assert.rejects(x.run(),/HTTP 403/);assert.equal(x.requests.length,1);
});
test('allows the previous backend during rolling deploy',async()=>{
 await setup([{ok:false,status:404}]).run();
});
