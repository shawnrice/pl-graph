/**
 * Usage-shaped benchmarks: what applications actually do, on every engine.
 *
 * `bench.ts` measures bulk throughput — load a big document, run one query over
 * all of it. That is the ingest shape, not the serving shape. Real use is many
 * small operations against a warm graph, and often reads and writes INTERLEAVED,
 * which is the pattern that has historically broken things here: a write
 * invalidates the read-side CSR snapshot, so alternating them can repack the
 * adjacency on every cycle.
 *
 * The workloads below are drawn from the applications this engine has been
 * exercised against — an authoritative state server with high-rate writes and
 * per-viewport live queries, a relationship-based authorization check, a
 * recommendation walk, entity resolution over a keyed lookup, and an
 * append-only audit log.
 *
 *   bun run bench:usage
 *   BENCH_ENGINES=ts,ffi bun run bench:usage
 *   BENCH_OPS=20000 bun run bench:usage      # operations per workload
 *   BENCH_GRAPH=50000 bun run bench:usage    # graph size to serve from
 *
 * Reported as operations per second, best of three batches. A batch is a fixed
 * number of operations, so the number is a rate rather than a duration and is
 * comparable across engines with very different absolute speeds.
 */
import { existsSync } from 'node:fs';

import { Graph as TsGraph } from '@lenke/core';
import { query as tsQuery } from '@lenke/gql';
import { deserialize as tsDeserialize } from '@lenke/serialization';

import { createFfiBackend } from './src/backend-ffi.js';
import { createWasmBackend } from './src/backend-wasm.js';
import { graphFromNdjson, type Backend } from './src/graph.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB = new URL(
  `../../crates/lenke-core/target/release/liblenke_core.${LIB_EXTENSIONS[process.platform] ?? 'so'}`,
  import.meta.url,
).pathname;
const WASM = new URL(
  '../../crates/lenke-core/target/wasm32-unknown-unknown/release/lenke_core.wasm',
  import.meta.url,
).pathname;

const OPS = Number(process.env.BENCH_OPS ?? 5_000);
const GRAPH = Number(process.env.BENCH_GRAPH ?? 20_000);
const BATCHES = Number(process.env.BENCH_BATCHES ?? 3);
const WANTED = new Set(
  (process.env.BENCH_ENGINES ?? 'ts,ffi,wasm').split(',').map((s) => s.trim()),
);

/** A graph handle plus the one operation every workload needs. */
type Engine = {
  name: string;
  load: (doc: string) => unknown;
  query: (g: unknown, text: string, params?: Record<string, unknown>) => unknown;
  index: (g: unknown, on: 'vertex' | 'edge', key: string) => void;
  transaction: (g: unknown, body: () => void) => void;
  free: (g: unknown) => void;
};

type IndexSpec = { on: 'vertex' | 'edge'; kind: 'hash'; keys: string[] };

type NativeGraph = {
  query: (t: string, p?: Record<string, unknown>) => unknown;
  createIndex: (s: IndexSpec) => void;
  transaction: (f: () => void) => void;
  free: () => void;
};

const nativeEngine = (name: string, backend: Backend): Engine => ({
  name,
  load: (doc) => graphFromNdjson(backend, doc),
  query: (g, text, params) => (g as NativeGraph).query(text, params),
  index: (g, on, key) => (g as NativeGraph).createIndex({ on, kind: 'hash', keys: [key] }),
  transaction: (g, body) => (g as NativeGraph).transaction(body),
  free: (g) => (g as NativeGraph).free(),
});

const tsEngine: Engine = {
  name: 'ts',
  load: (doc) => tsDeserialize(doc, 'ndjson', new TsGraph()),
  query: (g, text, params) => tsQuery(g as TsGraph, text, params),
  index: (g, on, key) =>
    (g as unknown as { createIndex: (s: IndexSpec) => void }).createIndex({
      on,
      kind: 'hash',
      keys: [key],
    }),
  transaction: (g, body) =>
    (g as unknown as { transaction: (f: () => void) => void }).transaction(body),
  free: () => {},
};

/**
 * A serving-shaped fixture: users in groups, resources in a hierarchy, and a
 * social graph to walk. Small enough to stay warm, which is the point — this
 * measures per-operation cost, not cache behaviour.
 */
const fixture = ((): string => {
  const lines: string[] = [];
  const groups = Math.max(1, Math.floor(GRAPH / 100));
  const resources = Math.max(1, Math.floor(GRAPH / 10));

  for (let i = 0; i < GRAPH; i++) {
    lines.push(
      `{"type":"node","id":"u${i}","labels":["User"],"properties":{"name":"user${i}","score":${i % 100},"active":${i % 3 === 0}}}`,
    );
  }

  for (let i = 0; i < groups; i++) {
    lines.push(`{"type":"node","id":"g${i}","labels":["Team"],"properties":{"tier":${i % 5}}}`);
  }

  for (let i = 0; i < resources; i++) {
    lines.push(`{"type":"node","id":"r${i}","labels":["Resource"],"properties":{"kind":${i % 4}}}`);
  }

  for (let i = 0; i < GRAPH; i++) {
    lines.push(
      `{"type":"edge","id":"m${i}","labels":["MEMBER_OF"],"from":"u${i}","to":"g${i % groups}","properties":{}}`,
    );
    // A social edge for the recommendation walk.
    lines.push(
      `{"type":"edge","id":"f${i}","labels":["FOLLOWS"],"from":"u${i}","to":"u${(i * 7919) % GRAPH}","properties":{}}`,
    );
  }

  for (let i = 0; i < resources; i++) {
    lines.push(
      `{"type":"edge","id":"v${i}","labels":["VIEWER"],"from":"g${i % groups}","to":"r${i}","properties":{}}`,
    );
  }

  return lines.join('\n');
})();

type Workload = {
  name: string;
  /** One operation. `i` is the operation index, so each does different work. */
  op: (e: Engine, g: unknown, i: number) => void;
  /** Vertex property keys to index before the batch runs. */
  indexes?: string[];
  /** Fewer operations for workloads that are inherently expensive. */
  scale?: number;
};

const WORKLOADS: Workload[] = [
  {
    // Point lookup by id — the single most common serving query there is.
    name: 'read: point lookup',
    op: (e, g, i) =>
      void e.query(g, 'MATCH (u:User) WHERE u.name = $n RETURN u.score AS s', {
        n: `user${i % GRAPH}`,
      }),
  },
  {
    // Relationship-based authorization: does this user reach this resource
    // through a group? One query, two hops.
    name: 'read: permission check',
    op: (e, g, i) =>
      void e.query(
        g,
        'MATCH (u:User)-[:MEMBER_OF]->(gr:Team)-[:VIEWER]->(r:Resource) WHERE u.name = $n RETURN count(*) AS c',
        { n: `user${i % GRAPH}` },
      ),
  },
  {
    // Recommendation-shaped: who do the people I follow follow?
    name: 'read: 2-hop recommendation',
    op: (e, g, i) =>
      void e.query(
        g,
        'MATCH (u:User)-[:FOLLOWS]->()-[:FOLLOWS]->(x) WHERE u.name = $n RETURN count(*) AS c',
        { n: `user${i % GRAPH}` },
      ),
  },
  {
    // Entity resolution: find the candidate that matches on a keyed attribute.
    name: 'read: keyed dedup lookup',
    op: (e, g, i) =>
      void e.query(
        g,
        'MATCH (u:User) WHERE u.score = $s AND u.active = true RETURN count(*) AS c',
        {
          s: i % 100,
        },
      ),
  },
  {
    name: 'write: property update',
    op: (e, g, i) =>
      void e.query(g, 'MATCH (u:User) WHERE u.name = $n SET u.score = $v', {
        n: `user${i % GRAPH}`,
        v: i % 1000,
      }),
  },
  {
    name: 'write: append node',
    op: (e, g, i) => void e.query(g, `INSERT (:Event {id: 'ev${i}', seq: ${i}})`),
  },
  {
    // THE interleaved shape: one write, then one read, repeatedly. A write
    // invalidates the read-side snapshot, so this is where a layout that is
    // only good for reads shows its true cost.
    name: 'interleaved: write + point read',
    op: (e, g, i) => {
      e.query(g, 'MATCH (u:User) WHERE u.name = $n SET u.score = $v', {
        n: `user${i % GRAPH}`,
        v: i % 1000,
      });
      e.query(g, 'MATCH (u:User) WHERE u.name = $n RETURN u.score AS s', {
        n: `user${(i + 1) % GRAPH}`,
      });
    },
  },
  {
    // The same alternation, but the read TRAVERSES — the shape that repacks
    // adjacency if the read side is a snapshot rebuilt after every write.
    name: 'interleaved: write + traversal',
    op: (e, g, i) => {
      e.query(g, 'MATCH (u:User) WHERE u.name = $n SET u.score = $v', {
        n: `user${i % GRAPH}`,
        v: i % 1000,
      });
      e.query(g, 'MATCH (u:User)-[:FOLLOWS]->(x) WHERE u.name = $n RETURN count(*) AS c', {
        n: `user${(i + 1) % GRAPH}`,
      });
    },
  },
  {
    // The SAME point lookup with the property indexed. Without this the engine
    // scans every vertex, which is what the un-suffixed rows above measure —
    // easy to mistake for a lookup because the query reads like one.
    name: 'read: point lookup (indexed)',
    indexes: ['name'],
    op: (e, g, i) =>
      void e.query(g, 'MATCH (u:User) WHERE u.name = $n RETURN u.score AS s', {
        n: `user${i % GRAPH}`,
      }),
  },
  {
    // The SAME check with the anchor written inline rather than in a trailing
    // WHERE. On the Rust engine these two rows differ by ~60x: a WHERE-form
    // anchor followed by a traversal stops seeding from the index and falls back
    // to a scan, while the inline form keeps the seek. The pure-TS engine seeds
    // both. Two rows rather than one, so the cliff cannot regress unseen — and
    // so nobody reads the slow row as the cost of the check itself.
    name: 'read: permission check (indexed, inline anchor)',
    indexes: ['name'],
    op: (e, g, i) =>
      void e.query(
        g,
        'MATCH (u:User {name: $n})-[:MEMBER_OF]->(gr:Team)-[:VIEWER]->(r:Resource) RETURN count(*) AS c',
        { n: `user${i % GRAPH}` },
      ),
  },
  {
    // Authorization with the anchor indexed: the seek replaces the scan, and the
    // traversal from it is bounded by the user's own degree.
    name: 'read: permission check (indexed, WHERE)',
    indexes: ['name'],
    op: (e, g, i) =>
      void e.query(
        g,
        'MATCH (u:User)-[:MEMBER_OF]->(gr:Team)-[:VIEWER]->(r:Resource) WHERE u.name = $n RETURN count(*) AS c',
        { n: `user${i % GRAPH}` },
      ),
  },
  {
    name: 'read: 2-hop recommendation (indexed)',
    indexes: ['name'],
    op: (e, g, i) =>
      void e.query(
        g,
        'MATCH (u:User {name: $n})-[:FOLLOWS]->()-[:FOLLOWS]->(x) RETURN count(*) AS c',
        { n: `user${i % GRAPH}` },
      ),
  },
  {
    name: 'write: property update (indexed)',
    indexes: ['name'],
    op: (e, g, i) =>
      void e.query(g, 'MATCH (u:User) WHERE u.name = $n SET u.score = $v', {
        n: `user${i % GRAPH}`,
        v: i % 1000,
      }),
  },
  {
    // Writes batched into one transaction, against the same writes committing
    // individually above. A transaction records an undo entry per write and
    // defers its constraint checks to commit, so this is the cost of that
    // bookkeeping amortized over a batch.
    name: 'write: 100 updates in a transaction',
    indexes: ['name'],
    scale: 100,
    op: (e, g, i) =>
      e.transaction(g, () => {
        for (let k = 0; k < 100; k++) {
          e.query(g, 'MATCH (u:User) WHERE u.name = $n SET u.score = $v', {
            n: `user${(i * 100 + k) % GRAPH}`,
            v: k,
          });
        }
      }),
  },
  {
    // Money-flow shapes, on the follow graph as the transfer network: fan-out
    // over a bounded number of hops, the pattern a structuring check computes.
    name: 'analytic: fan-out spread 1-3 hops',
    indexes: ['name'],
    scale: 20,
    op: (e, g, i) =>
      void e.query(
        g,
        'MATCH (u:User) WHERE u.name = $n RETURN COUNT { MATCH (u)-[:FOLLOWS]->{1,3}(x:User) } AS spread',
        { n: `user${i % GRAPH}` },
      ),
  },
  {
    // Does value return to where it started? The cycle test a layering check runs.
    name: 'analytic: cycle detection 2-4 hops',
    indexes: ['name'],
    scale: 20,
    op: (e, g, i) =>
      void e.query(
        g,
        'MATCH (u:User) WHERE u.name = $n AND EXISTS { MATCH (u)-[:FOLLOWS]->{2,4}(u) } RETURN u.name AS n',
        { n: `user${i % GRAPH}` },
      ),
  },
  {
    // Append-only audit: a ledger entry written and attached in one statement.
    //
    // Deliberately ONE statement against a fixed set, not `MATCH (a:Entry),
    // (u:User) … INSERT`. That form scans the Entry set, which this workload is
    // itself growing, so its rate would fall through the batch and depend on
    // BENCH_OPS — a benchmark measuring its own accumulation rather than the
    // operation.
    name: 'interleaved: append + link',
    op: (e, g, i) =>
      void e.query(
        g,
        `MATCH (u:User) WHERE u.name = $u INSERT (:Entry {id: 'en${i}', amount: ${i % 500}})-[:AUTHORED_BY]->(u)`,
        { u: `user${i % GRAPH}` },
      ),
  },
];

const engines: Engine[] = [];

if (WANTED.has('ts')) {
  engines.push(tsEngine);
}

if (WANTED.has('ffi')) {
  if (existsSync(LIB)) {
    engines.push(nativeEngine('ffi', createFfiBackend(LIB)));
  } else {
    console.warn(`skipping ffi: ${LIB} not found — run \`bun run build:rust\``);
  }
}

if (WANTED.has('wasm')) {
  if (existsSync(WASM)) {
    engines.push(nativeEngine('wasm', await createWasmBackend(await Bun.file(WASM).arrayBuffer())));
  } else {
    console.warn(`skipping wasm: ${WASM} not found — run \`bun run build:wasm\``);
  }
}

if (engines.length === 0) {
  console.error('no engines available');
  process.exit(1);
}

/** Operations per second for one workload, best of `BATCHES` batches. */
const rate = (e: Engine, w: Workload): number => {
  let bestOps = 0;

  for (let b = 0; b < BATCHES; b++) {
    // A fresh graph per batch: the write workloads mutate, and a batch that
    // inherited the previous one's growth would not be comparable.
    const g = e.load(fixture);

    try {
      for (const key of w.indexes ?? []) {
        e.index(g, 'vertex', key);
      }

      const ops = Math.max(1, Math.floor(OPS / (w.scale ?? 1)));

      w.op(e, g, 0); // warm

      const t = Bun.nanoseconds();

      for (let i = 1; i <= ops; i++) {
        w.op(e, g, i);
      }

      const perSec = ops / ((Bun.nanoseconds() - t) / 1e9);

      bestOps = Math.max(bestOps, perSec);
    } finally {
      e.free(g);
    }
  }

  return bestOps;
};

const base = engines[0].name;
const fmt = (n: number): string => {
  if (n >= 1e6) {
    return `${(n / 1e6).toFixed(2)}M`;
  }

  if (n >= 1e3) {
    return `${(n / 1e3).toFixed(1)}k`;
  }

  return n.toFixed(0);
};

console.log(
  `\n${GRAPH} users, ${OPS} ops/batch, best of ${BATCHES}. Engines: ${engines.map((e) => e.name).join(', ')}`,
);
console.log(`Operations per second, higher is better. Ratios against ${base}.\n`);

const ratioHeads = engines
  .slice(1)
  .map((e) => `${e.name}/${base}`.padStart(12))
  .join('');
const header = `${['workload'.padEnd(32), ...engines.map((e) => e.name.padStart(10))].join('')}  ${ratioHeads}`;

console.log(header);
console.log('-'.repeat(header.length));

for (const w of WORKLOADS) {
  const rates = engines.map((e) => {
    try {
      return rate(e, w);
    } catch (err) {
      // Report rather than silently printing `n/a` — a workload that cannot run
      // is a bug in the workload or a gap in an engine, and either is worth
      // seeing.
      console.error(
        `  ! ${w.name} [${e.name}]: ${(err as { code?: string }).code ?? ''} ${(err as Error).message?.slice(0, 120)}`,
      );

      return Number.NaN;
    }
  });
  const cells = rates.map((r) => (Number.isNaN(r) ? 'n/a' : fmt(r)).padStart(10));
  const ratios = rates
    .slice(1)
    .map((r) =>
      Number.isNaN(r) || Number.isNaN(rates[0])
        ? ''.padStart(12)
        : `${(r / rates[0]).toFixed(1)}x`.padStart(12),
    );

  console.log(`${w.name.padEnd(32)}${cells.join('')}  ${ratios.join('')}`);
}
