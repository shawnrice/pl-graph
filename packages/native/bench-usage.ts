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
 *   BENCH_BUDGET_MS=500 bun run bench:usage  # longer batches, tighter numbers
 *   BENCH_GRAPH=50000 bun run bench:usage    # graph size to serve from
 *
 * Reported as operations per second, best of three batches. Each batch runs for
 * a fixed DURATION and counts how many operations fit, so the number is a rate
 * and is comparable across engines with very different absolute speeds — and
 * every cell costs the same wall clock regardless of how slow it is.
 */
import { existsSync } from 'node:fs';

import { Graph as TsGraph } from '@lenke/core';
import { query as tsQuery } from '@lenke/gql';
import { deserialize as tsDeserialize } from '@lenke/serialization';

import { createFfiBackend } from './src/backend-ffi.js';
import { createWasmBackend } from './src/backend-wasm.js';
import type { Backend } from './src/backend.js';
import { graphFromNdjson } from './src/graph.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB = new URL(
  `../../crates/lenke-core/target/release/liblenke_core.${LIB_EXTENSIONS[process.platform] ?? 'so'}`,
  import.meta.url,
).pathname;
const WASM = new URL(
  '../../crates/lenke-core/target/wasm32-unknown-unknown/release/lenke_core.wasm',
  import.meta.url,
).pathname;

/** Fixed op count — used only by the `grows` workloads now (see `rate`). */
const OPS = Number(process.env.BENCH_OPS ?? 5_000);
/** Milliseconds each timed batch runs for. */
const BUDGET_MS = Number(process.env.BENCH_BUDGET_MS ?? 250);
/**
 * Floor on ops per batch, so a very slow cell is still more than one sample.
 * Small on purpose: a cell running at 6 ops/s spends 167 ms PER OPERATION, and
 * an operation that large is already a stable measurement — the many-samples
 * argument is about microsecond operations, not this end of the range.
 */
const MIN_OPS = Number(process.env.BENCH_MIN_OPS ?? 4);
/** Ceiling, so a pilot that happens to land fast cannot run away. */
const MAX_OPS = Number(process.env.BENCH_MAX_OPS ?? 2_000_000);
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
  index: (g, on, key) => (g as TsGraph).createIndex({ on, kind: 'hash', keys: [key] }),
  transaction: (g, body) => (g as TsGraph).transaction(body),
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
  /** Fewer iterations for workloads that are inherently expensive. */
  scale?: number;
  /**
   * Elementary operations each iteration performs, so every row is reported in
   * the same units. A batch of 100 updates is 100, not 1 — without this the
   * batched rows read as catastrophically slow next to the single-op rows purely
   * because they are counted differently.
   */
  units?: number;
  /**
   * This workload ADDS elements, so its cost depends on how many iterations ran.
   * Those run a fixed count on every engine; everything else is time-boxed. A
   * time-boxed append would let a fast engine grow the graph 100x more than a
   * slow one and then charge it for the larger graph.
   */
  grows?: true;
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
    // Carries an INDEXED key deliberately. An append whose properties are all
    // unindexed pays no maintenance, so the indexed and unindexed columns would
    // come out the same and the row would look like indexes are free on writes.
    name: 'write: append node',
    grows: true,
    op: (e, g, i) =>
      void e.query(g, `INSERT (:Event {name: 'ev${i}', score: ${i % 100}, seq: ${i}})`),
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
    // The SAME check with the anchor written inline rather than in a trailing
    // WHERE. On the Rust engine these two rows differ by ~60x: a WHERE-form
    // anchor followed by a traversal stops seeding from the index and falls back
    // to a scan, while the inline form keeps the seek. The pure-TS engine seeds
    // both. Two rows rather than one, so the cliff cannot regress unseen — and
    // so nobody reads the slow row as the cost of the check itself.
    name: 'read: permission check (inline anchor)',
    op: (e, g, i) =>
      void e.query(
        g,
        'MATCH (u:User {name: $n})-[:MEMBER_OF]->(gr:Team)-[:VIEWER]->(r:Resource) RETURN count(*) AS c',
        { n: `user${i % GRAPH}` },
      ),
  },
  {
    // Writes batched into one transaction, against the same writes committing
    // individually above. A transaction records an undo entry per write and
    // defers its constraint checks to commit, so this is the cost of that
    // bookkeeping amortized over a batch.
    name: 'write: 100 updates in one transaction',
    scale: 100,
    units: 100,
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
    // The SAME 100 updates as one statement with an IN-list, rather than 100
    // statements. This is the algorithmic fix for the row above: the cost there
    // is not the transaction, it is that each of the 100 statements re-scans to
    // find its target. One statement scans once and updates all 100.
    //
    // Which means it should close most of the gap WITHOUT an index — and with an
    // index it should not help much, because a seek per statement is already
    // cheap. Both columns are the point.
    name: 'write: 100 updates in one statement',
    scale: 100,
    units: 100,
    op: (e, g, i) =>
      void e.query(g, 'MATCH (u:User) WHERE u.name IN $names SET u.score = $v', {
        names: Array.from({ length: 100 }, (_, k) => `user${(i * 100 + k) % GRAPH}`),
        v: i % 1000,
      }),
  },
  {
    // Money-flow shapes, on the follow graph as the transfer network: fan-out
    // over a bounded number of hops, the pattern a structuring check computes.
    name: 'analytic: fan-out spread 1-3 hops',
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
    grows: true,
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

/**
 * The property keys indexed in the "with indexes" pass.
 *
 * `name` anchors most reads; `score` is written by the update workloads and read
 * by the dedup lookup, so it shows index MAINTENANCE cost as well as seek gain.
 */
const INDEXED_KEYS = ['name', 'score'];

/**
 * Operations per second for one workload, best of `BATCHES` batches.
 *
 * Batches are TIME-BOXED, not fixed-count. Cells in this matrix differ by five
 * orders of magnitude — an unindexed 2-hop traversal runs at 6 ops/s and an
 * indexed batched update at 543k — so a single op count is wrong at both ends
 * at once: it made the slow cells take 5+ minutes each (the whole matrix ran
 * ~40 minutes, nearly all of it three TS cells) while giving the fast cells a
 * 2 ms sample, which is noise. A time box spends the same wall clock everywhere
 * and lets the fast cells take the many more samples they need.
 *
 * Ops/sec is a rate, so a cell that completes 40 ops in 250 ms reports just as
 * validly as one that completes a million.
 *
 * `grows` workloads are the exception and stay fixed-count — see the flag.
 */
const rate = (e: Engine, w: Workload, indexed: boolean): number => {
  let bestOps = 0;
  const budgetNs = BUDGET_MS * 1e6;
  const fixed = w.grows ? Math.max(1, Math.floor(OPS / (w.scale ?? 1))) : 0;

  for (let b = 0; b < BATCHES; b++) {
    // A fresh graph per batch: the write workloads mutate, and a batch that
    // inherited the previous one's growth would not be comparable.
    const g = e.load(fixture);

    try {
      if (indexed) {
        for (const key of INDEXED_KEYS) {
          e.index(g, 'vertex', key);
        }
      }

      // Warm, and use that one operation as the PILOT that sizes the batch.
      // Polling the clock inside the loop instead needs a chunk size, and any
      // fixed chunk is wrong at one end: 16 ops is nothing at 500k ops/s and is
      // 5.3 SECONDS at 6 ops/s, which is where most of this benchmark's runtime
      // went. Sizing up front costs one extra operation and nothing else.
      const pilot = Bun.nanoseconds();

      w.op(e, g, 0);

      const perOp = Math.max(1, Bun.nanoseconds() - pilot);
      const ops =
        fixed > 0 ? fixed : Math.max(MIN_OPS, Math.min(MAX_OPS, Math.ceil(budgetNs / perOp)));
      const t = Bun.nanoseconds();

      for (let i = 1; i <= ops; i++) {
        w.op(e, g, i);
      }

      const perSec = (ops * (w.units ?? 1)) / ((Bun.nanoseconds() - t) / 1e9);

      bestOps = Math.max(bestOps, perSec);
    } finally {
      e.free(g);
    }
  }

  return bestOps;
};

const fmt = (n: number): string => {
  if (Number.isNaN(n)) {
    return 'n/a';
  }

  if (n >= 1e6) {
    return `${(n / 1e6).toFixed(2)}M`;
  }

  if (n >= 1e3) {
    return `${(n / 1e3).toFixed(1)}k`;
  }

  return n.toFixed(0);
};

console.log(
  `\n${GRAPH} users, ${BUDGET_MS}ms batches, best of ${BATCHES}. Engines: ${engines.map((e) => e.name).join(', ')}`,
);
console.log(
  `Operations per second, higher is better. (-) no indexes, (+) indexed on ${INDEXED_KEYS.join(', ')}.\n`,
);

const cols = engines.flatMap((e) => [`${e.name}(-)`, `${e.name}(+)`]);
const header = ['workload'.padEnd(34), ...cols.map((c) => c.padStart(10))].join('');

console.log(header);
console.log('-'.repeat(header.length));

for (const w of WORKLOADS) {
  const cells = engines.flatMap((e) =>
    [false, true].map((indexed) => {
      try {
        return fmt(rate(e, w, indexed)).padStart(10);
      } catch (err) {
        // Reported rather than silently `n/a` — a workload that cannot run is a
        // bug in the workload or a gap in an engine, and both are worth seeing.
        console.error(
          `  ! ${w.name} [${e.name}${indexed ? '+' : '-'}]: ${(err as { code?: string }).code ?? ''} ${(err as Error).message?.slice(0, 100)}`,
        );

        return 'n/a'.padStart(10);
      }
    }),
  );

  console.log(`${w.name.padEnd(34)}${cells.join('')}`);
}
