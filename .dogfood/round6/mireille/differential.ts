// TS-vs-Rust numeric differential, modeled on native/src/gql-functions-conformance.test.ts.
// Runs each RETURN expression on BOTH engines against identical data and reports
// byte-identity of JSON.stringify'd results. Also verifies against plain JS.
import { existsSync } from 'node:fs';

import { Graph } from '@lenke/core';
import { query as tsQuery } from '@lenke/gql';

import { createFfiBackend } from '../../../packages/native/src/backend-ffi.ts';
import { graphFromFormat } from '../../../packages/native/src/graph.ts';

const LIB = new URL('../../../crates/lenke-core/target/release/liblenke_core.so', import.meta.url)
  .pathname;
if (!existsSync(LIB)) throw new Error(`native lib missing: ${LIB}`);

const NDJSON = [
  '{"type":"node","id":"1","labels":["T"],"properties":{"num":-3.7,"lat":37.7749295,"lng":-122.4194155}}',
].join('\n');

const backend = createFfiBackend(LIB);
const nativeGraph = graphFromNdjson(backend, NDJSON);
const tsGraph = (await import('@lenke/serialization')).deserialize(NDJSON, 'ndjson', new Graph());

type Case = { expr: string; params?: Record<string, unknown>; js?: unknown };
const CASES: Case[] = [
  // The suspected divergence: mod-by-zero fn vs operator.
  { expr: 'mod(7, 0)', js: 'JS %: NaN' },
  { expr: '7 % 0', js: 'div-by-zero' },
  { expr: 'mod(-7, 0)' },
  { expr: 'mod(7.5, 0)' },
  // Non-finite coercion policy.
  { expr: 'power(-8, 0.5)', js: (-8) ** 0.5 },
  { expr: 'sqrt(-1)', js: Math.sqrt(-1) },
  { expr: 'ln(0)', js: Math.log(0) },
  { expr: 'ln(-1)', js: Math.log(-1) },
  { expr: 'log10(0)', js: Math.log10(0) },
  { expr: 'exp(1000)', js: Math.exp(1000) },
  { expr: 'power(10, 400)', js: 10 ** 400 },
  { expr: '$inf', params: { inf: Infinity }, js: Infinity },
  { expr: '$nan', params: { nan: NaN }, js: NaN },
  { expr: 'tan(radians(90))', js: Math.tan(Math.PI / 2) },
  { expr: 'cot(0)', js: 1 / Math.tan(0) },
  { expr: 'asin(2)', js: Math.asin(2) },
  { expr: 'acos(-2)', js: Math.acos(-2) },
  // Rounding half-away parity.
  { expr: 'round(-2.5)', js: 'half-away: -3' },
  { expr: 'round(2.675, 2)', js: 'fp: 2.67 or 2.68?' },
  { expr: 'round(0.5)', js: 1 },
  { expr: 'round(-0.5)', js: -1 },
  { expr: 'sign(-0.0)', js: 'signed zero' },
  // Haversine end-to-end on both engines.
  {
    expr: '2 * 6371.0088 * asin(sqrt(power(sin(radians(40.7127753 - n.lat)/2),2) + cos(radians(n.lat))*cos(radians(40.7127753))*power(sin(radians(-74.0059728 - n.lng)/2),2)))',
    js: (() => {
      const toRad = (d: number) => (d * Math.PI) / 180;
      const lat1 = 37.7749295,
        lng1 = -122.4194155,
        lat2 = 40.7127753,
        lng2 = -74.0059728;
      const a =
        Math.sin(toRad(lat2 - lat1) / 2) ** 2 +
        Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(toRad(lng2 - lng1) / 2) ** 2;
      return 2 * 6371.0088 * Math.asin(Math.sqrt(a));
    })(),
  },
];

const runTs = (c: Case) => {
  try {
    return JSON.stringify(tsQuery(tsGraph, `MATCH (n:T) RETURN ${c.expr} AS r`, c.params)[0]?.r);
  } catch (e: any) {
    return `ERR<${e?.code ?? e?.name}>`;
  }
};
const runNative = (c: Case) => {
  try {
    return JSON.stringify(
      nativeGraph.query(`MATCH (n:T) RETURN ${c.expr} AS r`, c.params ?? {})[0]?.r,
    );
  } catch (e: any) {
    return `ERR<${e?.code ?? e?.name ?? String(e?.message).slice(0, 40)}>`;
  }
};

console.log('=== TS vs RUST NUMERIC DIFFERENTIAL ===');
let diverged = 0;
for (const c of CASES) {
  const ts = runTs(c);
  const nv = runNative(c);
  const same = ts === nv;
  if (!same) diverged++;
  const tag = same ? 'agree ' : 'DIVERGE';
  console.log(
    `${tag}  ${c.expr.slice(0, 50).padEnd(52)} ts=${ts}  rust=${nv}${c.js !== undefined ? `  (js=${JSON.stringify(c.js)})` : ''}`,
  );
}
console.log(`\nTotal cases: ${CASES.length}, TS/Rust divergences: ${diverged}`);
