// Interleaved write/read benchmark — the shape the existing benches structurally
// cannot see.
//
// WHY THIS EXISTS. `perf_bench.rs` bulk-loads a graph, times read shapes against
// a warm CSR, then times writes in a separate pass — explicitly "so it doesn't
// perturb the read timings above" (perf_bench.rs:98), with `add_edge` annotated
// as "grows adjacency + invalidates the CSR each call" (perf_bench.rs:105).
// `write-bench.ts` likewise builds N vertices and never reads. So writes are
// benched with no reads and reads are benched with no writes, and the cost of a
// read that follows a write is invisible to both.
//
// WHAT IT FOUND, AND THE FIX (2026-07-23). Native `read` was LINEAR in N even
// with zero writes in the phase — 41us at N=1k rising to 2.24ms at N=64k, while
// native `write` stayed flat (5.0us) and TS `read` stayed flat (~13us). At N=64k
// native traversal was ~187x slower than TS.
//
// Cache invalidation was NOT the cause: a cache that only dies on writes cannot
// explain linear cost in a phase performing none. The real cause was seed
// selection. `build_scan` fell through to `edge_index_seed`, whose `by_etype`
// fallback materializes EVERY edge of the traversed type, and
// `try_orient_node_seed` deliberately bails when an endpoint has a real index
// seek — so having a usable index actively DIVERTED the plan into that whole-type
// scan. Separately `scan_start_seed` walked the label bucket without consulting
// the vertex index at all. Both are fixed (eval.rs); native reads are now flat and
// faster than TS at every size (7.4us at N=64k, a ~303x improvement).
//
// Locked in by `traversal_from_indexed_anchor_is_independent_of_edge_type_size`
// in crates/lenke-core/src/gql/tests.rs.
//
// WHAT TO WATCH — the regression signal is `read ns/op`, which must stay FLAT as
// N grows, for BOTH engines. Do not use the interleaved/read ratio: it sat near
// 0.5x for native at every size even while absolute cost went linear, because both
// terms degrade together. `ns/op/N` is the honest normaliser — it should fall
// toward zero as N grows, and a roughly constant column means linear-in-N cost.
//
//   bun run benchmarks/interleave-bench.ts

import { Graph } from '@lenke/core';
import { query as tsQuery } from '@lenke/gql';
import { createEmptyGraph } from '@lenke/native';
import { createFfiBackend } from '@lenke/native/ffi';

const LIB = new URL('crates/lenke-core/target/release/liblenke_core.so', `file://${process.cwd()}/`)
  .pathname;

const SIZES = [1_000, 4_000, 16_000, 64_000];
const OPS = 200; // measured operations per phase, at each size

const fmt = (n: number): string => {
  if (n >= 100) {
    return n.toFixed(0);
  }

  return n >= 1 ? n.toFixed(2) : n.toFixed(3);
};
const pad = (s: string, n: number): string => s.padEnd(n);

type Engine = 'TS' | 'native';

type Phases = {
  read: number; // ns/op, pure reads against a warm cache
  write: number; // ns/op, pure writes, nothing reading
  interleaved: number; // ns/op, write then read, repeatedly
};

/** Build an N-vertex chain-ish graph, then time the three phases. */
const measure = (engine: Engine, n: number): Phases => {
  const g: any = engine === 'TS' ? new Graph() : createEmptyGraph(createFfiBackend(LIB));
  // The TS core is driven by `query(g, ...)` from @lenke/gql; the native handle
  // exposes `.query` directly. Normalise so both engines run identical text.
  const q = (s: string, p?: Record<string, unknown>) =>
    engine === 'TS' ? tsQuery(g, s, p as never) : g.query(s, p);
  g.disableEvents?.();
  g.createIndex({ on: 'vertex', kind: 'hash', keys: ['id'] });

  // Bulk load: N vertices, each with one outgoing edge to the next.
  for (let i = 0; i < n; i++) {
    q('INSERT (:P {id:$id, w:$w})', { id: `p${i}`, w: i });
  }

  for (let i = 0; i < n - 1; i++) {
    q('MATCH (a:P {id:$a}), (b:P {id:$b}) INSERT (a)-[:LINK {w:1}]->(b)', {
      a: `p${i}`,
      b: `p${i + 1}`,
    });
  }

  // A one-hop traversal off a specific, indexed anchor. Constant work per call:
  // the anchor has exactly one outgoing NEXT edge regardless of N.
  const read = (i: number) =>
    q('MATCH (a:P {id:$a})-[r:LINK]->(b:P) RETURN b.id AS id', { a: `p${i % (n - 1)}` });
  // A structural write — this is what invalidates the native CSR.
  const write = (i: number) => q('MATCH (a:P {id:$a}) SET a.w = $w', { a: `p${i % n}`, w: i });

  const time = (fn: (i: number) => unknown): number => {
    const s = performance.now();

    for (let i = 0; i < OPS; i++) {
      fn(i);
    }

    return ((performance.now() - s) / OPS) * 1e6; // ns/op
  };

  read(0); // warm the cache before the pure-read phase
  const readNs = time(read);
  const writeNs = time(write);
  const interleavedNs =
    time((i) => {
      write(i);
      read(i);
    }) / 2; // two ops per iteration

  g.free?.();

  return { read: readNs, write: writeNs, interleaved: interleavedNs };
};

console.log(
  '\nWrite/read cost by graph size. Watch `ns/op/N`: it must FALL toward zero.\n' +
    'A column that stays flat means per-op cost is linear in graph size.\n',
);
console.log(
  `${pad('engine', 8)} ${pad('N', 9)} ${pad('read ns/op', 13)} ${pad('write ns/op', 13)} ${pad('interleaved', 13)} ns/op/N`,
);
console.log('-'.repeat(72));

for (const engine of ['TS', 'native'] as Engine[]) {
  for (const n of SIZES) {
    const p = measure(engine, n);
    // ns/op/N: falls toward zero if per-op cost is independent of graph size.
    // Holding roughly CONSTANT down this column means cost is linear in N.
    const perN = p.read / n;
    console.log(
      `${pad(engine, 8)} ${pad(n.toLocaleString(), 9)} ${pad(fmt(p.read), 13)} ${pad(fmt(p.write), 13)} ${pad(fmt(p.interleaved), 13)} ${perN.toFixed(1)}`,
    );
  }

  console.log();
}
