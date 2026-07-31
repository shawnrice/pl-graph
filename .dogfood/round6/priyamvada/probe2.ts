import { graphFromFormat } from '../../../packages/native/src/graph.js';
import { nativeGraph, tsGraph, tsQuery, classify, backend } from './harness.ts';

// Find exact nesting limits (binary-ish scan)
const nat = (q: string) => classify(() => nativeGraph.query(q));
const ts = (q: string) => classify(() => tsQuery(tsGraph, q));
let lastNatOk = 0,
  firstNatErr = 0;
for (let n = 1; n <= 200; n++) {
  const q = `RETURN ${'('.repeat(n)}1${')'.repeat(n)} AS x`;
  if (nat(q).kind === 'ok') lastNatOk = n;
  else {
    firstNatErr = n;
    break;
  }
}
let lastTsOk = 0,
  firstTsErr = 0;
for (let n = 1; n <= 400; n++) {
  const q = `RETURN ${'('.repeat(n)}1${')'.repeat(n)} AS x`;
  if (ts(q).kind === 'ok') lastTsOk = n;
  else {
    firstTsErr = n;
    break;
  }
}
console.log(
  `paren native: last-ok=${lastNatOk} first-err=${firstNatErr} | ts: last-ok=${lastTsOk} first-err=${firstTsErr}`,
);

// empty input on other native surfaces
console.log('\n--- empty on other native surfaces ---');
for (const fmt of ['ndjson', 'csv', 'pg-json', 'pg-text', 'graphson']) {
  const r = classify(() => graphFromFormat(backend, '', { format: fmt }));
  console.log(
    `deserialize '' ${fmt}: ${r.kind} ${(r as any).code || (r as any).name || ''} ${((r as any).msg || '').slice(0, 60)}`,
  );
}
// native prepare('')
console.log('\n--- native other query entrypoints with empty/edge ---');
for (const q of ['', ' ', '\x00', 'RETURN 1']) {
  const r = classify(() => nativeGraph.query(q));
  console.log(`query ${JSON.stringify(q)}: ${r.kind} ${(r as any).code || (r as any).name || ''}`);
}
