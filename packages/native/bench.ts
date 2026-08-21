/**
 * Cross-engine benchmark harness: pure-TS vs native (ffi) vs wasm.
 *
 * Two reasons this exists rather than another Rust example.
 *
 * The Rust benchmarks in `crates/lenke-core/examples` measure the NATIVE build
 * and only the native build. They compile for `wasm32-unknown-unknown` but
 * cannot run there — `Instant::now()` panics (that target has no clock), there
 * is no stdout, and there is no runner. A green `cargo build --target
 * wasm32-unknown-unknown --example …` proves nothing.
 *
 * And the pure-TS engine is a separate implementation that no Rust benchmark can
 * reach at all. It is what runs wherever a native artifact cannot ship, so a
 * regression there is invisible to every other measurement in this repo.
 *
 * Both are reachable from here, through one set of workloads:
 *
 *   bun run bench                       # every engine that is available
 *   BENCH_ENGINES=ts,wasm bun run bench
 *   BENCH_N=1000000 bun run bench       # bigger workload (default 200k)
 *   BENCH_REPS=7 bun run bench          # more samples
 *
 * Reports the MINIMUM of N runs, not the mean: it is the sample least polluted
 * by whatever else the machine was doing. See `crates/lenke-core/examples/
 * README.md` for the rest of what this suite has learned about trusting its own
 * numbers — in particular that a cache-resident workload answers a different
 * question, so vary BENCH_N before drawing a conclusion.
 *
 * READING THE RATIOS. They are against the first engine listed. The decode rows
 * are not purely codegen: decode defaults to the parallel path and only ffi has
 * threads, so wasm and TS both run it serially. (Not much of the gap — the
 * parallel decoder buys ~1.2x on ffi anyway.) Encode and query rows have no such
 * asymmetry.
 */
import { existsSync } from 'node:fs';

import { Graph as TsGraph } from '@lenke/core';
import { query as tsQuery } from '@lenke/gql';
import {
  deserialize as tsDeserialize,
  serialize as tsSerialize,
  type FormatName,
} from '@lenke/serialization';

import { createFfiEngineBackend } from './src/backend-ffi-engine.js';
import { createWasmEngineBackend } from './src/backend-wasm-engine.js';
import type { Backend } from './src/backend.js';
import { graphFromFormat, graphFromNdjson, type GraphFormat } from './src/graph.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB = new URL(
  `../../crates/lenke-engine/target/release/liblenke_engine.${LIB_EXTENSIONS[process.platform] ?? 'so'}`,
  import.meta.url,
).pathname;
const WASM = new URL(
  '../../crates/lenke-engine/target/wasm32-unknown-unknown/release/lenke_engine.wasm',
  import.meta.url,
).pathname;

const N = Number(process.env.BENCH_N ?? 200_000);
const REPS = Number(process.env.BENCH_REPS ?? 5);
const WANTED = new Set(
  (process.env.BENCH_ENGINES ?? 'ts,ffi,wasm').split(',').map((s) => s.trim()),
);

/**
 * What a workload needs from an engine.
 *
 * The two implementations have genuinely different shapes — the Rust core is a
 * handle behind a backend and must be freed, the TS core is an ordinary
 * GC-managed object — so workloads are written against this rather than against
 * either one.
 */
type Engine = {
  name: string;
  load: (doc: string) => unknown;
  loadFormat: (doc: string, format: string) => unknown;
  serialize: (g: unknown, format: string) => string;
  query: (g: unknown, text: string) => unknown;
  /** A no-op where the runtime collects for you. */
  free: (g: unknown) => void;
};

type NativeGraph = {
  serialize: (f: GraphFormat) => string;
  query: (t: string) => unknown;
  free: () => void;
};

const nativeEngine = (name: string, backend: Backend): Engine => ({
  name,
  load: (doc) => graphFromNdjson(backend, doc),
  loadFormat: (doc, format) => graphFromFormat(backend, doc, { format: format as GraphFormat }),
  serialize: (g, format) => (g as NativeGraph).serialize(format as GraphFormat),
  query: (g, text) => (g as NativeGraph).query(text),
  free: (g) => (g as NativeGraph).free(),
});

const tsEngine: Engine = {
  name: 'ts',
  load: (doc) => tsDeserialize(doc, 'ndjson', new TsGraph()),
  loadFormat: (doc, format) => tsDeserialize(doc, format as FormatName, new TsGraph()),
  serialize: (g, format) => tsSerialize(g as TsGraph, format as FormatName),
  query: (g, text) => tsQuery(g as TsGraph, text),
  free: () => {},
};

/** The workload documents, built once and shared by every engine. */
const nodesDoc = Array.from(
  { length: N },
  (_, i) =>
    `{"type":"node","id":"v${i}","labels":["Person"],"properties":{"name":"person${i}","city":"Springfield","age":${i % 90}}}`,
).join('\n');
const graphDoc = (() => {
  const lines = Array.from(
    { length: N },
    (_, i) => `{"type":"node","id":"v${i}","labels":["Person"],"properties":{"age":${i % 90}}}`,
  );

  // Five edges per node, endpoints scattered, every edge carrying an id — the
  // shape a reloaded snapshot actually has. A sparse or id-less fixture
  // understates everything on the edge path.
  for (let k = 0; k < N * 5; k++) {
    lines.push(
      `{"type":"edge","id":"e${k}","labels":["KNOWS"],"from":"v${(k * 7919) % N}","to":"v${(k * 104_729) % N}","properties":{"w":${k % 7}}}`,
    );
  }

  return lines.join('\n');
})();

const FORMATS = ['ndjson', 'pg-json', 'graphson', 'pg-text', 'csv'];
const DECODABLE = ['pg-json', 'graphson', 'pg-text'];

type Case = { name: string; run: (e: Engine) => void };

/** Run `fn` against a freshly loaded graph, releasing it afterwards. */
const withGraph = (e: Engine, doc: string, fn: (g: unknown) => void): void => {
  const g = e.load(doc);

  try {
    fn(g);
  } finally {
    e.free(g);
  }
};

const CASES: Case[] = [
  { name: 'decode ndjson (nodes)', run: (e) => e.free(e.load(nodesDoc)) },
  { name: 'decode ndjson (5 edges/node)', run: (e) => e.free(e.load(graphDoc)) },
  ...FORMATS.map((fmt) => ({
    name: `encode ${fmt}`,
    run: (e: Engine) => withGraph(e, nodesDoc, (g) => void e.serialize(g, fmt)),
  })),
  ...DECODABLE.map((fmt) => ({
    name: `decode ${fmt}`,
    run: (e: Engine) => {
      const src = e.load(nodesDoc);
      const text = e.serialize(src, fmt);

      e.free(src);
      e.free(e.loadFormat(text, fmt));
    },
  })),
  {
    name: 'query: count',
    run: (e) =>
      withGraph(e, nodesDoc, (g) => void e.query(g, 'MATCH (n:Person) RETURN count(*) AS c')),
  },
  {
    name: 'query: project 3 columns',
    run: (e) =>
      withGraph(
        e,
        nodesDoc,
        (g) => void e.query(g, 'MATCH (n:Person) RETURN n.name AS n, n.city AS c, n.age AS a'),
      ),
  },
  {
    name: 'query: group + aggregate',
    run: (e) =>
      withGraph(
        e,
        nodesDoc,
        (g) => void e.query(g, 'MATCH (n:Person) RETURN n.age AS a, count(*) AS c GROUP BY a'),
      ),
  },
  {
    name: 'query: 1-hop traversal',
    run: (e) =>
      withGraph(
        e,
        graphDoc,
        (g) => void e.query(g, 'MATCH (a:Person)-[:KNOWS]->(x) RETURN count(*) AS c'),
      ),
  },
];

const engines: Engine[] = [];

if (WANTED.has('ts')) {
  engines.push(tsEngine);
}

if (WANTED.has('ffi')) {
  if (existsSync(LIB)) {
    engines.push(nativeEngine('ffi', createFfiEngineBackend(LIB)));
  } else {
    console.warn(`skipping ffi: ${LIB} not found — run \`bun run build:rust\``);
  }
}

if (WANTED.has('wasm')) {
  if (existsSync(WASM)) {
    engines.push(nativeEngine('wasm', await createWasmEngineBackend(await Bun.file(WASM).arrayBuffer())));
  } else {
    console.warn(`skipping wasm: ${WASM} not found — run \`bun run build:wasm\``);
  }
}

if (engines.length === 0) {
  console.error('no engines available');
  process.exit(1);
}

/** Minimum of `REPS` runs, in milliseconds. */
const best = (f: () => void): number => {
  f(); // warm

  let ms = Infinity;

  for (let i = 0; i < REPS; i++) {
    const t = Bun.nanoseconds();

    f();
    ms = Math.min(ms, (Bun.nanoseconds() - t) / 1e6);
  }

  return ms;
};

const base = engines[0].name;

console.log(
  `\n${N} nodes, ${REPS} reps, best of each. Engines: ${engines.map((e) => e.name).join(', ')}`,
);
console.log(`Ratios are against ${base}.\n`);

const ratioHeads = engines
  .slice(1)
  .map((e) => `${e.name}/${base}`.padStart(12))
  .join('');
const header = `${['workload'.padEnd(30), ...engines.map((e) => e.name.padStart(11))].join('')}  ${ratioHeads}`;

console.log(header);
console.log('-'.repeat(header.length));

for (const c of CASES) {
  const times = engines.map((e) => {
    try {
      return best(() => c.run(e));
    } catch {
      return Number.NaN;
    }
  });
  const cells = times.map((t) => (Number.isNaN(t) ? 'n/a' : t.toFixed(1)).padStart(11));
  const ratios = times
    .slice(1)
    .map((t) =>
      Number.isNaN(t) || Number.isNaN(times[0])
        ? ''.padStart(12)
        : `${(t / times[0]).toFixed(2)}x`.padStart(12),
    );

  console.log(`${c.name.padEnd(30)}${cells.join('')}  ${ratios.join('')}`);
}
