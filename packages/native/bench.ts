/**
 * Cross-backend benchmark harness.
 *
 * The Rust benchmarks in `crates/lenke-core/examples` measure the NATIVE build
 * and only the native build. They compile for `wasm32-unknown-unknown` but
 * cannot run there — `Instant::now()` panics (that target has no clock), there
 * is no stdout, and there is no runner. So the wasm build can only be measured
 * the way it is actually used: from JS, through a backend.
 *
 * This runs the same workloads through whichever backends are asked for and
 * prints them side by side, so "how much does the browser path cost?" has an
 * answer.
 *
 *   bun run bench                      # every backend that is built
 *   BENCH_BACKENDS=wasm bun run bench  # just one
 *   BENCH_N=1000000 bun run bench      # bigger workload (default 200k)
 *   BENCH_REPS=7 bun run bench         # more samples
 *
 * Reports the MINIMUM of N runs, not the mean: it is the sample least polluted
 * by whatever else the machine was doing. See `examples/README.md` for the rest
 * of what this suite has learned about trusting its own numbers — in particular
 * that a cache-resident workload answers a different question, so vary BENCH_N
 * before drawing a conclusion.
 *
 * READING THE RATIO. It is not purely a measure of wasm codegen. Decode defaults
 * to the parallel path, and wasm has no threads, so it falls back to serial while
 * ffi uses rayon — part of the gap on decode rows is threads rather than
 * instructions. (Not much of it: the parallel decoder only buys ~1.2x on ffi
 * anyway, which is why the nodes-only row can come out FASTER on wasm.) The
 * encode and query rows have no such asymmetry.
 */
import { existsSync } from 'node:fs';

import { createFfiBackend } from './src/backend-ffi.js';
import { createWasmBackend } from './src/backend-wasm.js';
import { graphFromFormat, graphFromNdjson, type Backend, type GraphFormat } from './src/graph.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB = new URL(
  `../../crates/lenke-core/target/release/liblenke_core.${LIB_EXTENSIONS[process.platform] ?? 'so'}`,
  import.meta.url,
).pathname;
const WASM = new URL(
  '../../crates/lenke-core/target/wasm32-unknown-unknown/release/lenke_core.wasm',
  import.meta.url,
).pathname;

const N = Number(process.env.BENCH_N ?? 200_000);
const REPS = Number(process.env.BENCH_REPS ?? 5);
const WANTED = new Set((process.env.BENCH_BACKENDS ?? 'ffi,wasm').split(',').map((s) => s.trim()));

const enc = new TextEncoder();

/** The workload documents, built once and shared by every backend. */
const nodesDoc = enc.encode(
  Array.from(
    { length: N },
    (_, i) =>
      `{"type":"node","id":"v${i}","labels":["Person"],"properties":{"name":"person${i}","city":"Springfield","age":${i % 90}}}`,
  ).join('\n'),
);
const graphDoc = enc.encode(
  (() => {
    const lines = Array.from(
      { length: N },
      (_, i) => `{"type":"node","id":"v${i}","labels":["Person"],"properties":{"age":${i % 90}}}`,
    );

    // Five edges per node, endpoints scattered, every edge carrying an id —
    // the shape a reloaded snapshot actually has. A sparse or id-less fixture
    // understates everything on the edge path.
    for (let k = 0; k < N * 5; k++) {
      lines.push(
        `{"type":"edge","id":"e${k}","labels":["KNOWS"],"from":"v${(k * 7919) % N}","to":"v${(k * 104729) % N}","properties":{"w":${k % 7}}}`,
      );
    }

    return lines.join('\n');
  })(),
);

type Case = { name: string; run: (b: Backend) => void };

const CASES: Case[] = [
  {
    name: 'decode ndjson (nodes)',
    run: (b) => graphFromNdjson(b, nodesDoc).free(),
  },
  {
    name: 'decode ndjson (5 edges/node)',
    run: (b) => graphFromNdjson(b, graphDoc).free(),
  },
  ...(['ndjson', 'pg-json', 'graphson', 'pg-text', 'csv'] as GraphFormat[]).map((fmt) => ({
    name: `encode ${fmt}`,
    run: (b: Backend) => {
      const g = graphFromNdjson(b, nodesDoc);

      try {
        g.serialize(fmt);
      } finally {
        g.free();
      }
    },
  })),
  ...(['pg-json', 'graphson', 'pg-text'] as GraphFormat[]).map((fmt) => ({
    name: `decode ${fmt}`,
    run: (b: Backend) => {
      const src = graphFromNdjson(b, nodesDoc);
      const text = src.serialize(fmt);

      src.free();
      graphFromFormat(b, text, { format: fmt }).free();
    },
  })),
  {
    name: 'query: count',
    run: (b) => {
      const g = graphFromNdjson(b, nodesDoc);

      try {
        g.query('MATCH (n:Person) RETURN count(*) AS c');
      } finally {
        g.free();
      }
    },
  },
  {
    name: 'query: project 3 columns',
    run: (b) => {
      const g = graphFromNdjson(b, nodesDoc);

      try {
        g.query('MATCH (n:Person) RETURN n.name AS n, n.city AS c, n.age AS a');
      } finally {
        g.free();
      }
    },
  },
  {
    name: 'query: group + aggregate',
    run: (b) => {
      const g = graphFromNdjson(b, nodesDoc);

      try {
        g.query('MATCH (n:Person) RETURN n.age AS a, count(*) AS c GROUP BY a');
      } finally {
        g.free();
      }
    },
  },
  {
    name: 'query: 1-hop traversal',
    run: (b) => {
      const g = graphFromNdjson(b, graphDoc);

      try {
        g.query('MATCH (a:Person)-[:KNOWS]->(x) RETURN count(*) AS c');
      } finally {
        g.free();
      }
    },
  },
];

const backends: [string, Backend][] = [];

if (WANTED.has('ffi')) {
  if (existsSync(LIB)) {
    backends.push(['ffi', createFfiBackend(LIB)]);
  } else {
    console.warn(`skipping ffi: ${LIB} not found — run \`bun run build:rust\``);
  }
}

if (WANTED.has('wasm')) {
  if (existsSync(WASM)) {
    backends.push(['wasm', await createWasmBackend(await Bun.file(WASM).arrayBuffer())]);
  } else {
    console.warn(`skipping wasm: ${WASM} not found — run \`bun run build:wasm\``);
  }
}

if (backends.length === 0) {
  console.error('no backends available');
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

console.log(
  `\n${N} nodes, ${REPS} reps, best of each. Backends: ${backends.map(([n]) => n).join(', ')}\n`,
);

const header = ['workload'.padEnd(30), ...backends.map(([n]) => n.padStart(11))].join('');

console.log(header + (backends.length > 1 ? '        ratio' : ''));
console.log('-'.repeat(header.length + (backends.length > 1 ? 13 : 0)));

for (const c of CASES) {
  const times = backends.map(([, b]) => {
    try {
      return best(() => c.run(b));
    } catch {
      return Number.NaN;
    }
  });
  const cells = times.map((t) => (Number.isNaN(t) ? 'n/a' : t.toFixed(1)).padStart(11));
  const ratio =
    times.length > 1 && times.every((t) => !Number.isNaN(t))
      ? `        ${(times[1] / times[0]).toFixed(2)}x`
      : '';

  console.log(c.name.padEnd(30) + cells.join('') + ratio);
}
