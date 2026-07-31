import { existsSync } from 'node:fs';

import { Graph } from '@lenke/core';
import { query as tsQuery } from '@lenke/gql';

import { createFfiBackend } from '../../../packages/native/src/backend-ffi.ts';
import { graphFromFormat } from '../../../packages/native/src/graph.ts';

// Three cities, population as bigint.
const NDJSON = [
  '{"type":"node","id":"1","labels":["City"],"properties":{"nm":"A","pop":100}}',
  '{"type":"node","id":"2","labels":["City"],"properties":{"nm":"B","pop":200}}',
  '{"type":"node","id":"3","labels":["City"],"properties":{"nm":"C","pop":300}}',
].join('\n');
// NOTE: NDJSON numbers deserialize as JS number, not bigint. To force bigint we
// build the TS graph by hand; native path can't easily hold bigint, so cross-check
// focuses on the TS-only bigint semantics + a number-typed native baseline.

const tsG = new Graph();
for (const [nm, pop] of [
  ['A', 100n],
  ['B', 200n],
  ['C', 300n],
] as const)
  tsG.addVertex({ labels: ['City'], properties: { nm, pop } });

// same data but number-typed, for contrast
const tsGnum = new Graph();
for (const [nm, pop] of [
  ['A', 100],
  ['B', 200],
  ['C', 300],
] as const)
  tsGnum.addVertex({ labels: ['City'], properties: { nm, pop } });

const names = (g: Graph, where: string, params?: any) => {
  try {
    return query_(g, where, params);
  } catch (e: any) {
    return `ERR<${e?.code ?? e?.name}>`;
  }
};
const query_ = (g: Graph, where: string, params?: any) =>
  tsQuery(g, `MATCH (c:City) WHERE ${where} RETURN c.nm AS n ORDER BY c.nm`, params).map(
    (r: any) => r.n,
  );

console.log('=== bigint-typed pop vs number-typed pop, same WHEREs ===');
const wheres: Array<[string, any?]> = [
  ['c.pop > 150'],
  ['c.pop >= 200'],
  ['c.pop = 200'],
  ['c.pop < 250'],
  ['c.pop > $t', { t: 150 }], // number param
  ['c.pop > $t', { t: 150n }], // bigint param
  ['c.pop = $t', { t: 200n }], // bigint param
];
for (const [w, p] of wheres) {
  console.log(
    `${w.padEnd(16)} ${p ? JSON.stringify(p, (_, v) => (typeof v === 'bigint' ? `${v}n` : v)) : ''}`.padEnd(
      34,
    ) +
      `bigint-store=${JSON.stringify(names(tsG, w, p))}   number-store=${JSON.stringify(names(tsGnum, w, p))}`,
  );
}

// Native (Rust) baseline with number-typed pop, to confirm number path parity.
const LIB = new URL('../../../crates/lenke-core/target/release/liblenke_core.so', import.meta.url)
  .pathname;
if (existsSync(LIB)) {
  const backend = createFfiBackend(LIB);
  const nvG = graphFromNdjson(backend, NDJSON);
  console.log('\n=== native (number-typed) baseline ===');
  for (const [w, p] of wheres) {
    try {
      const r = nvG
        .query(`MATCH (c:City) WHERE ${w} RETURN c.nm AS n ORDER BY c.nm`, p ?? {})
        .map((x: any) => x.n);
      console.log(`${w.padEnd(16)} => ${JSON.stringify(r)}`);
    } catch (e: any) {
      console.log(`${w.padEnd(16)} => ERR<${e?.code ?? String(e?.message).slice(0, 30)}>`);
    }
  }
}
