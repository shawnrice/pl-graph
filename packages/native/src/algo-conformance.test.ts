// Differential conformance for the in-engine graph algorithms: the TS core
// (@lenke/core data-last free functions, in-process) vs the Rust core (this
// package, over bun:ffi), driven from ONE source of truth — the same NDJSON
// loaded into both — so an algorithm's result can't drift between the two forms.
//
//   load once:   identical NDJSON (same ids/labels/properties, same order)
//   TS core:     JSON.stringify(degree(config, tsGraph))
//   Rust core:   JSON.stringify(nativeGraph.degree(config))
//   assert:      the two serializations are byte-identical
//
// Both engines assign dense vertex ids / iterate vertices in NDJSON insertion
// order and count a vertex's edges in adjacency order with no sorting, so integer
// algorithms are exactly equal and their JSON serializations match byte-for-byte.
// A `writeProperty` config additionally writes the result back to a vertex
// property, which we read out through GQL on both engines to prove the two graphs
// mutated identically.
//
// Run: bun test packages/native/src/algo-conformance.test.ts
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import {
  type AlgorithmConfig,
  betweenness,
  closeness,
  connectedComponents,
  degree,
  Graph,
  labelPropagation,
  neighborAggregate,
  pagerank,
  peerPressure,
  shortestPath,
} from '@lenke/core';
import { query as tsQuery } from '@lenke/gql';
import { deserialize as tsDeserialize } from '@lenke/serialization';

import { createFfiBackend } from './backend-ffi.js';
import { graphFromNdjson } from './graph.js';

// --- native library bootstrap (mirrors gql-conformance.test.ts) -------------
const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-core/target/release/liblenke_core.${LIB_EXT}`,
  import.meta.url,
).pathname;
const hasLib = existsSync(LIB);

if (!hasLib) {
  console.warn(`[algo-conformance] skipping: ${LIB} not found — run \`bun run build:rust\`.`);
}

const suite = hasLib ? describe : describe.skip;

// The TinkerPop "modern" graph, nodes in NON-sorted insertion order (1,2,4,3,5,6)
// so the test proves both engines honour insertion order, not id order. KNOWS:
// marko→vadas, marko→josh. CREATED: marko→lop, josh→ripple, josh→lop, peter→lop.
const MODERN_NDJSON = [
  '{"type":"node","id":"1","labels":["Person"],"properties":{"name":"marko"}}',
  '{"type":"node","id":"2","labels":["Person"],"properties":{"name":"vadas"}}',
  '{"type":"node","id":"4","labels":["Person"],"properties":{"name":"josh"}}',
  '{"type":"node","id":"3","labels":["Software"],"properties":{"name":"lop"}}',
  '{"type":"node","id":"5","labels":["Software"],"properties":{"name":"ripple"}}',
  '{"type":"node","id":"6","labels":["Person"],"properties":{"name":"peter"}}',
  '{"type":"edge","id":"7","from":"1","to":"2","labels":["KNOWS"]}',
  '{"type":"edge","id":"8","from":"1","to":"4","labels":["KNOWS"]}',
  '{"type":"edge","id":"9","from":"1","to":"3","labels":["CREATED"]}',
  '{"type":"edge","id":"10","from":"4","to":"5","labels":["CREATED"]}',
  '{"type":"edge","id":"11","from":"4","to":"3","labels":["CREATED"]}',
  '{"type":"edge","id":"12","from":"6","to":"3","labels":["CREATED"]}',
].join('\n');

suite('graph-algorithm differential: degree (TS core vs native)', () => {
  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, MODERN_NDJSON);
  const tsGraph = tsDeserialize(MODERN_NDJSON, 'ndjson', new Graph());

  const both = async (config: AlgorithmConfig): Promise<[string, string]> => [
    JSON.stringify(await degree(config, tsGraph)),
    JSON.stringify(await nativeGraph.degree(config)),
  ];

  for (const config of [
    { direction: 'out' } as const,
    { direction: 'in' } as const,
    { direction: 'both' } as const,
    { direction: 'out', edgeLabel: 'KNOWS' } as const,
    { direction: 'in', edgeLabel: 'CREATED' } as const,
    { direction: 'both', edgeLabel: 'CREATED' } as const,
    { edgeLabel: 'NOPE' } as const, // unknown edge type → all zero
    {} as const, // defaults (out, all types)
  ]) {
    test(`degree ${JSON.stringify(config)} — byte-identical`, async () => {
      const [ts, native] = await both(config);
      expect(ts).toBe(native);
    });
  }

  test('known-answer: out-degree over all types', async () => {
    // marko(1)=3, vadas(2)=0, josh(4)=2, lop(3)=0, ripple(5)=0, peter(6)=1.
    expect(await nativeGraph.degree({ direction: 'out' })).toEqual([
      { node: '1', degree: 3 },
      { node: '2', degree: 0 },
      { node: '4', degree: 2 },
      { node: '3', degree: 0 },
      { node: '5', degree: 0 },
      { node: '6', degree: 1 },
    ]);
  });

  test('writeProperty round-trips identically through GQL on both engines', async () => {
    const config = { direction: 'both', writeProperty: 'deg' } as const;
    // Mutate both graphs.
    await degree(config, tsGraph);
    await nativeGraph.degree(config);

    // Read the written property back through BOTH GQL engines: identical output
    // proves the two graphs were mutated identically by their respective `degree`.
    const readBack = 'MATCH (n) RETURN n.name AS name, n.deg AS deg ORDER BY n.deg DESC, n.name';
    const tsRows = JSON.stringify(tsQuery(tsGraph, readBack));
    const nativeRows = JSON.stringify(nativeGraph.query(readBack));
    expect(tsRows).toBe(nativeRows);
    // lop(3) has in-degree 3 (marko, josh, peter), out 0 → both = 3.
    expect((await nativeGraph.degree({ direction: 'both' }))[3]).toEqual({ node: '3', degree: 3 });
    expect(nativeRows).toContain('"deg":3');
  });
});

// A graph with two disjoint components {a,b,c} and {e,d} plus an isolated vertex
// f — nodes in NON-sorted insertion order to prove both engines root each
// component at its first-inserted (lowest dense-id) member, not by id string.
const TWO_COMPONENT_NDJSON = [
  '{"type":"node","id":"a","labels":["N"]}',
  '{"type":"node","id":"b","labels":["N"]}',
  '{"type":"node","id":"c","labels":["N"]}',
  '{"type":"node","id":"e","labels":["N"]}',
  '{"type":"node","id":"d","labels":["N"]}',
  '{"type":"node","id":"f","labels":["N"]}',
  '{"type":"edge","id":"1","from":"a","to":"b","labels":["E"]}',
  '{"type":"edge","id":"2","from":"b","to":"c","labels":["E"]}',
  '{"type":"edge","id":"3","from":"e","to":"d","labels":["E"]}',
].join('\n');

suite('graph-algorithm differential: connectedComponents (TS core vs native)', () => {
  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, TWO_COMPONENT_NDJSON);
  const tsGraph = tsDeserialize(TWO_COMPONENT_NDJSON, 'ndjson', new Graph());

  for (const config of [{} as const, { edgeLabel: 'E' } as const, { edgeLabel: 'NOPE' } as const]) {
    test(`connectedComponents ${JSON.stringify(config)} — byte-identical`, async () => {
      expect(JSON.stringify(await connectedComponents(config, tsGraph))).toBe(
        JSON.stringify(await nativeGraph.connectedComponents(config)),
      );
    });
  }

  test('known-answer: roots are first-inserted member (a, e), f isolated', async () => {
    // Insertion order a,b,c,e,d,f → {a,b,c} root "a"; {e,d} root "e"; {f} root "f".
    expect(await nativeGraph.connectedComponents({})).toEqual([
      { node: 'a', componentId: 'a' },
      { node: 'b', componentId: 'a' },
      { node: 'c', componentId: 'a' },
      { node: 'e', componentId: 'e' },
      { node: 'd', componentId: 'e' },
      { node: 'f', componentId: 'f' },
    ]);
  });

  test('writeProperty round-trips identically through GQL on both engines', async () => {
    const config = { writeProperty: 'comp' } as const;
    await connectedComponents(config, tsGraph);
    await nativeGraph.connectedComponents(config);

    const readBack = 'MATCH (n) RETURN n.comp AS comp ORDER BY n.comp, n.comp';
    expect(JSON.stringify(tsQuery(tsGraph, readBack))).toBe(
      JSON.stringify(nativeGraph.query(readBack)),
    );
  });
});

// Two triangles {a,b,c} and {e,d,g} (non-sorted insertion order) plus a bridge
// edge c→e joining them, and an isolated vertex f. Exercises convergence, a
// bridged super-component, and a singleton in one graph.
const LABELPROP_NDJSON = [
  '{"type":"node","id":"a","labels":["N"]}',
  '{"type":"node","id":"b","labels":["N"]}',
  '{"type":"node","id":"c","labels":["N"]}',
  '{"type":"node","id":"e","labels":["N"]}',
  '{"type":"node","id":"d","labels":["N"]}',
  '{"type":"node","id":"g","labels":["N"]}',
  '{"type":"node","id":"f","labels":["N"]}',
  '{"type":"edge","id":"1","from":"a","to":"b","labels":["E"]}',
  '{"type":"edge","id":"2","from":"b","to":"c","labels":["E"]}',
  '{"type":"edge","id":"3","from":"a","to":"c","labels":["E"]}',
  '{"type":"edge","id":"4","from":"e","to":"d","labels":["E"]}',
  '{"type":"edge","id":"5","from":"d","to":"g","labels":["E"]}',
  '{"type":"edge","id":"6","from":"e","to":"g","labels":["E"]}',
  '{"type":"edge","id":"7","from":"c","to":"e","labels":["E"]}',
].join('\n');

suite('graph-algorithm differential: labelPropagation (TS core vs native)', () => {
  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, LABELPROP_NDJSON);
  const tsGraph = tsDeserialize(LABELPROP_NDJSON, 'ndjson', new Graph());

  for (const config of [
    {} as const, // default 10 iterations
    { iterations: 0 } as const, // no propagation
    { iterations: 1 } as const, // one round — catches any per-round drift
    { iterations: 3 } as const,
    { iterations: 25 } as const,
    { edgeLabel: 'E' } as const,
    { edgeLabel: 'NOPE' } as const, // unknown type → labels stay = own id
  ]) {
    test(`labelPropagation ${JSON.stringify(config)} — byte-identical`, async () => {
      expect(JSON.stringify(await labelPropagation(config, tsGraph))).toBe(
        JSON.stringify(await nativeGraph.labelPropagation(config)),
      );
    });
  }

  test('writeProperty round-trips identically through GQL on both engines', async () => {
    const config = { writeProperty: 'lbl' } as const;
    await labelPropagation(config, tsGraph);
    await nativeGraph.labelPropagation(config);

    const readBack = 'MATCH (n) RETURN n.lbl AS lbl ORDER BY n.lbl, n.lbl';
    expect(JSON.stringify(tsQuery(tsGraph, readBack))).toBe(
      JSON.stringify(nativeGraph.query(readBack)),
    );
  });
});

// Two communities {a,b,c} and {d,e} bridged by c→d, with INTERLEAVED edge types and
// VARIED out-degrees (a,b,c have 2 out-edges → vote 0.5; e has 1 → vote 1.0) so the
// 1/out-degree f64 vote energies exercise the canonical edge-order summation.
const PEERPRESSURE_NDJSON = [
  '{"type":"node","id":"a","labels":["N"]}',
  '{"type":"node","id":"b","labels":["N"]}',
  '{"type":"node","id":"c","labels":["N"]}',
  '{"type":"node","id":"d","labels":["N"]}',
  '{"type":"node","id":"e","labels":["N"]}',
  '{"type":"edge","id":"1","from":"a","to":"b","labels":["T1"]}',
  '{"type":"edge","id":"2","from":"a","to":"c","labels":["T2"]}',
  '{"type":"edge","id":"3","from":"b","to":"a","labels":["T1"]}',
  '{"type":"edge","id":"4","from":"b","to":"c","labels":["T2"]}',
  '{"type":"edge","id":"5","from":"c","to":"a","labels":["T1"]}',
  '{"type":"edge","id":"6","from":"c","to":"d","labels":["T2"]}',
  '{"type":"edge","id":"7","from":"d","to":"e","labels":["T1"]}',
  '{"type":"edge","id":"8","from":"e","to":"d","labels":["T2"]}',
].join('\n');

suite('graph-algorithm differential: peerPressure (TS core vs native, f64 votes)', () => {
  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, PEERPRESSURE_NDJSON);
  const tsGraph = tsDeserialize(PEERPRESSURE_NDJSON, 'ndjson', new Graph());

  for (const config of [
    {} as const, // default 30-iteration cap
    { iterations: 1 } as const, // one round — catches per-round drift
    { iterations: 3 } as const,
    { iterations: 50 } as const,
    { edgeLabel: 'T1' } as const, // typed filter changes out-degrees → different votes
    { edgeLabel: 'NOPE' } as const, // unknown type → every vertex its own cluster
  ]) {
    test(`peerPressure ${JSON.stringify(config)} — byte-identical`, async () => {
      expect(JSON.stringify(await peerPressure(config, tsGraph))).toBe(
        JSON.stringify(await nativeGraph.peerPressure(config)),
      );
    });
  }

  test('writeProperty round-trips identically through GQL on both engines', async () => {
    const config = { writeProperty: 'cl' } as const;
    await peerPressure(config, tsGraph);
    await nativeGraph.peerPressure(config);

    const readBack = 'MATCH (n) RETURN n.cl AS cl ORDER BY n.cl, n.cl';
    expect(JSON.stringify(tsQuery(tsGraph, readBack))).toBe(
      JSON.stringify(nativeGraph.query(readBack)),
    );
  });
});

// A weighted graph with INTERLEAVED edge types into a common hub (e's in-edges are
// T1,T2,T1,T2 in insertion order) — this is exactly the shape that diverges if an
// engine iterated adjacency grouped-by-type instead of edge-insertion order, so it
// pins the f64 summation order. Also a dangling structure (b, a sink) and weights.
const PAGERANK_NDJSON = [
  '{"type":"node","id":"a","labels":["N"]}',
  '{"type":"node","id":"b","labels":["N"]}',
  '{"type":"node","id":"c","labels":["N"]}',
  '{"type":"node","id":"d","labels":["N"]}',
  '{"type":"node","id":"e","labels":["N"]}',
  '{"type":"edge","id":"1","from":"a","to":"e","labels":["T1"],"properties":{"w":0.5}}',
  '{"type":"edge","id":"2","from":"b","to":"e","labels":["T2"],"properties":{"w":1.5}}',
  '{"type":"edge","id":"3","from":"c","to":"e","labels":["T1"],"properties":{"w":2.0}}',
  '{"type":"edge","id":"4","from":"d","to":"e","labels":["T2"],"properties":{"w":0.25}}',
  '{"type":"edge","id":"5","from":"e","to":"a","labels":["T1"],"properties":{"w":1.0}}',
  '{"type":"edge","id":"6","from":"a","to":"c","labels":["T2"],"properties":{"w":0.7}}',
  '{"type":"edge","id":"7","from":"c","to":"d","labels":["T1"],"properties":{"w":1.3}}',
].join('\n');

suite('graph-algorithm differential: pagerank (TS core vs native, f64)', () => {
  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, PAGERANK_NDJSON);
  const tsGraph = tsDeserialize(PAGERANK_NDJSON, 'ndjson', new Graph());

  for (const config of [
    {} as const, // default 20 iterations, d=0.85, unweighted
    { iterations: 1 } as const, // single round — catches first-step drift
    { iterations: 5 } as const,
    { iterations: 50 } as const, // near-converged: bit drift would compound
    { dampingFactor: 0.5 } as const,
    { dampingFactor: 0.99 } as const,
    { weightProperty: 'w' } as const, // weighted — stresses weight reads + order
    { weightProperty: 'w', iterations: 7 } as const,
    { weightProperty: 'w', edgeLabel: 'T1' } as const,
    { edgeLabel: 'T2' } as const,
    { edgeLabel: 'NOPE' } as const, // no edges → uniform 1/N
  ]) {
    test(`pagerank ${JSON.stringify(config)} — f64 byte-identical`, async () => {
      expect(JSON.stringify(await pagerank(config, tsGraph))).toBe(
        JSON.stringify(await nativeGraph.pagerank(config)),
      );
    });
  }

  test('writeProperty round-trips identically through GQL on both engines', async () => {
    const config = { writeProperty: 'pr' } as const;
    await pagerank(config, tsGraph);
    await nativeGraph.pagerank(config);

    const readBack = 'MATCH (n) RETURN n.pr AS pr ORDER BY n.pr DESC, n.pr';
    expect(JSON.stringify(tsQuery(tsGraph, readBack))).toBe(
      JSON.stringify(nativeGraph.query(readBack)),
    );
  });
});

// B1 regression: a node whose TOTAL out-weight is 0 (here `a`, whose only out-edge
// has w:0) must be treated as a DANGLING node, not divided by zero. Before the fix,
// `w / S[a] == 0/0 == NaN` poisoned every score to null on BOTH engines; the two
// stayed byte-identical only by being identically broken. This pins the repaired
// behavior: finite, mass-conserving scores, still byte-identical across engines.
const ZERO_WEIGHT_NDJSON = [
  '{"type":"node","id":"a","labels":["N"]}',
  '{"type":"node","id":"b","labels":["N"]}',
  '{"type":"node","id":"c","labels":["N"]}',
  '{"type":"edge","id":"1","from":"a","to":"b","labels":["R"],"properties":{"w":0}}',
  '{"type":"edge","id":"2","from":"b","to":"c","labels":["R"],"properties":{"w":0.5}}',
].join('\n');

suite('graph-algorithm differential: weighted pagerank with a zero-out-weight node', () => {
  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, ZERO_WEIGHT_NDJSON);
  const tsGraph = tsDeserialize(ZERO_WEIGHT_NDJSON, 'ndjson', new Graph());

  for (const config of [
    { edgeLabel: 'R', weightProperty: 'w' } as const, // the exact repro
    { edgeLabel: 'R', weightProperty: 'w', iterations: 30 } as const, // near-converged
  ]) {
    test(`zero-out-weight pagerank ${JSON.stringify(config)} — finite + byte-identical`, async () => {
      const ts = await pagerank(config, tsGraph);
      const nat = await nativeGraph.pagerank(config);
      expect(JSON.stringify(ts)).toBe(JSON.stringify(nat));

      // Repaired: every score is a finite number (no NaN/null poisoning).
      for (const row of ts as ReadonlyArray<{ score: number }>) {
        expect(Number.isFinite(row.score)).toBe(true);
      }
    });
  }
});

// A weighted diamond with fractional weights: a→b→d (0.1+0.2 = 0.30000000000000004)
// vs the direct a→d (0.3) — the classic f64 non-associativity trap, so this pins
// that both engines settle the same minimum float distance. Plus a longer branch
// and a sink (e).
// `he` is an admissible heuristic (≤ true distance to e) used by the A* cases.
const SHORTEST_NDJSON = [
  '{"type":"node","id":"a","labels":["N"],"properties":{"he":0.5}}',
  '{"type":"node","id":"b","labels":["N"],"properties":{"he":0.4}}',
  '{"type":"node","id":"c","labels":["N"],"properties":{"he":0.7}}',
  '{"type":"node","id":"d","labels":["N"],"properties":{"he":0.2}}',
  '{"type":"node","id":"e","labels":["N"],"properties":{"he":0.0}}',
  '{"type":"edge","id":"1","from":"a","to":"b","labels":["E"],"properties":{"w":0.1}}',
  '{"type":"edge","id":"2","from":"b","to":"d","labels":["E"],"properties":{"w":0.2}}',
  '{"type":"edge","id":"3","from":"a","to":"d","labels":["E"],"properties":{"w":0.3}}',
  '{"type":"edge","id":"4","from":"a","to":"c","labels":["E"],"properties":{"w":1.5}}',
  '{"type":"edge","id":"5","from":"c","to":"d","labels":["E"],"properties":{"w":0.5}}',
  '{"type":"edge","id":"6","from":"c","to":"e","labels":["E"],"properties":{"w":2.0}}',
  '{"type":"edge","id":"7","from":"d","to":"e","labels":["E"],"properties":{"w":0.25}}',
].join('\n');

suite('graph-algorithm differential: shortestPath (TS core vs native)', () => {
  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, SHORTEST_NDJSON);
  const tsGraph = tsDeserialize(SHORTEST_NDJSON, 'ndjson', new Graph());

  for (const config of [
    { source: 'a' } as const, // BFS (unweighted)
    { source: 'a', weightProperty: 'w' } as const, // Dijkstra, f64 diamond
    { source: 'c', weightProperty: 'w' } as const,
    { source: 'e', weightProperty: 'w' } as const, // sink → only e:0
    { source: 'a', weightProperty: 'w', edgeLabel: 'E' } as const,
    { source: 'a', edgeLabel: 'NOPE' } as const, // no edges → only source
    { source: 'zzz' } as const, // unknown source → empty
    // A* (goal-directed): h=0 (degrades to Dijkstra) and an admissible heuristic.
    { source: 'a', target: 'e', weightProperty: 'w', algorithm: 'astar' } as const,
    {
      source: 'a',
      target: 'e',
      weightProperty: 'w',
      algorithm: 'astar',
      heuristicProperty: 'he',
    } as const,
    { source: 'a', target: 'd', weightProperty: 'w', algorithm: 'astar' } as const,
    { source: 'e', target: 'a', weightProperty: 'w', algorithm: 'astar' } as const, // unreachable
  ]) {
    test(`shortestPath ${JSON.stringify(config)} — byte-identical`, async () => {
      expect(JSON.stringify(await shortestPath(config, tsGraph))).toBe(
        JSON.stringify(await nativeGraph.shortestPath(config)),
      );
    });
  }

  test('A* target distance equals Dijkstra (with and without a heuristic)', async () => {
    const dijkstra = await nativeGraph.shortestPath({ source: 'a', weightProperty: 'w' });

    for (const target of ['b', 'c', 'd', 'e']) {
      const dj = dijkstra.find((r) => r.node === target)?.distance;
      const plain = await nativeGraph.shortestPath({
        source: 'a',
        target,
        weightProperty: 'w',
        algorithm: 'astar',
      });
      const heur = await nativeGraph.shortestPath({
        source: 'a',
        target,
        weightProperty: 'w',
        algorithm: 'astar',
        heuristicProperty: 'he',
      });
      expect(plain).toEqual([{ node: target, distance: dj! }]);
      expect(heur).toEqual([{ node: target, distance: dj! }]);
    }
  });

  test('known-answer: weighted diamond settles the direct 0.3, not 0.1+0.2', async () => {
    const d = await nativeGraph.shortestPath({ source: 'a', weightProperty: 'w' });
    expect(d.find((r) => r.node === 'd')?.distance).toBe(0.3);
    expect(d.find((r) => r.node === 'e')?.distance).toBe(0.55);
  });

  test('writeProperty round-trips identically through GQL on both engines', async () => {
    const config = { source: 'a', weightProperty: 'w', writeProperty: 'sp' } as const;
    await shortestPath(config, tsGraph);
    await nativeGraph.shortestPath(config);

    const readBack = 'MATCH (n) RETURN n.sp AS sp ORDER BY n.sp, n.sp';
    expect(JSON.stringify(tsQuery(tsGraph, readBack))).toBe(
      JSON.stringify(nativeGraph.query(readBack)),
    );
  });
});

// A directed diamond with a tail: 1→2→4 and 1→3→4 are two equal shortest 1→4 paths
// (so σ(4)=2 exercises the fractional dependency split), then 4→5. Edge types are
// INTERLEAVED (node 1's out-edges are T1 then T2) so a grouped-by-label traversal
// would diverge — pinning the global edge-insertion order both engines' CSR uses.
// Weights are the 0.1/0.2 f64 trap but keep the two diamond paths equal (both
// 0.1+0.2 = 0.30000000000000004), so the weighted `σ += σ` and closeness distance
// sums exercise non-associative f64 in lockstep.
const CENTRALITY_NDJSON = [
  '{"type":"node","id":"1","labels":["N"]}',
  '{"type":"node","id":"2","labels":["N"]}',
  '{"type":"node","id":"3","labels":["N"]}',
  '{"type":"node","id":"4","labels":["N"]}',
  '{"type":"node","id":"5","labels":["N"]}',
  '{"type":"edge","id":"1","from":"1","to":"2","labels":["T1"],"properties":{"w":0.1}}',
  '{"type":"edge","id":"2","from":"1","to":"3","labels":["T2"],"properties":{"w":0.1}}',
  '{"type":"edge","id":"3","from":"2","to":"4","labels":["T1"],"properties":{"w":0.2}}',
  '{"type":"edge","id":"4","from":"3","to":"4","labels":["T2"],"properties":{"w":0.2}}',
  '{"type":"edge","id":"5","from":"4","to":"5","labels":["T1"],"properties":{"w":0.3}}',
].join('\n');

suite('graph-algorithm differential: betweenness (Brandes, TS core vs native)', () => {
  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, CENTRALITY_NDJSON);
  const tsGraph = tsDeserialize(CENTRALITY_NDJSON, 'ndjson', new Graph());

  for (const config of [
    {} as const, // unweighted BFS Brandes
    { weightProperty: 'w' } as const, // Dijkstra Brandes over the f64 trap
    { edgeLabel: 'T1' } as const, // typed: only 1→2→4→5 survives
    { edgeLabel: 'NOPE' } as const, // unknown type → all zero
  ]) {
    test(`betweenness ${JSON.stringify(config)} — f64 byte-identical`, async () => {
      expect(JSON.stringify(await betweenness(config, tsGraph))).toBe(
        JSON.stringify(await nativeGraph.betweenness(config)),
      );
    });
  }

  test('known-answer: diamond middles carry 0.5 each of (1,4); hub 4 carries 3', async () => {
    // CB[2]=CB[3]=1 (each half of pairs (1,4) and (1,5)); CB[4]=3 (on (1,5),(2,5),(3,5)).
    expect(await nativeGraph.betweenness({})).toEqual([
      { node: '1', centrality: 0 },
      { node: '2', centrality: 1 },
      { node: '3', centrality: 1 },
      { node: '4', centrality: 3 },
      { node: '5', centrality: 0 },
    ]);
  });

  test('writeProperty round-trips identically through GQL on both engines', async () => {
    const config = { writeProperty: 'bt' } as const;
    await betweenness(config, tsGraph);
    await nativeGraph.betweenness(config);

    const readBack = 'MATCH (n) RETURN n.bt AS bt ORDER BY n.bt DESC, n.bt';
    expect(JSON.stringify(tsQuery(tsGraph, readBack))).toBe(
      JSON.stringify(nativeGraph.query(readBack)),
    );
  });
});

suite('graph-algorithm differential: closeness (TS core vs native)', () => {
  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, CENTRALITY_NDJSON);
  const tsGraph = tsDeserialize(CENTRALITY_NDJSON, 'ndjson', new Graph());

  for (const config of [
    {} as const, // unweighted BFS distance sum
    { weightProperty: 'w' } as const, // Dijkstra f64 distance sum
    { edgeLabel: 'T1' } as const,
    { edgeLabel: 'NOPE' } as const, // no edges → every vertex 0
  ]) {
    test(`closeness ${JSON.stringify(config)} — f64 byte-identical`, async () => {
      expect(JSON.stringify(await closeness(config, tsGraph))).toBe(
        JSON.stringify(await nativeGraph.closeness(config)),
      );
    });
  }

  test('known-answer: unnormalized 1/Σd; sink 5 reaches nothing → 0', async () => {
    // 1 reaches {2:1,3:1,4:2,5:3} → 1/7; 2,3 → 1/3; 4 → 1/1; 5 → 0.
    expect(await nativeGraph.closeness({})).toEqual([
      { node: '1', centrality: 1 / 7 },
      { node: '2', centrality: 1 / 3 },
      { node: '3', centrality: 1 / 3 },
      { node: '4', centrality: 1 },
      { node: '5', centrality: 0 },
    ]);
  });

  test('writeProperty round-trips identically through GQL on both engines', async () => {
    const config = { writeProperty: 'cl' } as const;
    await closeness(config, tsGraph);
    await nativeGraph.closeness(config);

    const readBack = 'MATCH (n) RETURN n.cl AS cl ORDER BY n.cl DESC, n.cl';
    expect(JSON.stringify(tsQuery(tsGraph, readBack))).toBe(
      JSON.stringify(nativeGraph.query(readBack)),
    );
  });
});

// A small weighted feature graph with FRACTIONAL vectors, so a `mean`/`sum`
// aggregate is byte-identical only if both engines fold contributors in the same
// order. Directed edges, aggregated by direction. Nodes carry a 3-dim `h`.
const FEATURE_NDJSON = [
  '{"type":"node","id":"a","labels":["N"],"properties":{"h":[0.1,0.2,0.3]}}',
  '{"type":"node","id":"b","labels":["N"],"properties":{"h":[1.5,2.5,3.5]}}',
  '{"type":"node","id":"c","labels":["N"],"properties":{"h":[0.7,0.9,1.1]}}',
  '{"type":"node","id":"d","labels":["N"],"properties":{"h":[9.9,8.8,7.7]}}',
  // e has NO feature — a neighbour of it contributes, but e contributes nothing.
  '{"type":"node","id":"e","labels":["N"],"properties":{}}',
  '{"type":"edge","from":"a","to":"b","labels":["R"]}',
  '{"type":"edge","from":"a","to":"c","labels":["R"]}',
  '{"type":"edge","from":"a","to":"d","labels":["R"]}',
  '{"type":"edge","from":"b","to":"c","labels":["R"]}',
  '{"type":"edge","from":"c","to":"a","labels":["R"]}',
  '{"type":"edge","from":"e","to":"a","labels":["R"]}',
  '{"type":"edge","from":"a","to":"e","labels":["R"]}',
].join('\n');

suite('graph-algorithm differential: neighborAggregate (TS core vs native)', () => {
  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, FEATURE_NDJSON);
  const tsGraph = tsDeserialize(FEATURE_NDJSON, 'ndjson', new Graph());

  const both = async (config: AlgorithmConfig): Promise<[string, string]> => [
    JSON.stringify(await neighborAggregate(config, tsGraph)),
    JSON.stringify(await nativeGraph.neighborAggregate(config)),
  ];

  for (const op of ['mean', 'sum', 'max', 'min'] as const) {
    for (const direction of ['out', 'in', 'both'] as const) {
      test(`${op} / ${direction} is byte-identical`, async () => {
        const [ts, native] = await both({ feature: 'h', op, direction });
        expect(ts).toBe(native);
      });
    }
  }

  test('includeSelf is byte-identical (self folds first)', async () => {
    const [ts, native] = await both({
      feature: 'h',
      op: 'mean',
      direction: 'both',
      includeSelf: true,
    });
    expect(ts).toBe(native);
  });

  // GCN symmetric normalization (`1/sqrt(deg_i·deg_j)`) on the IRREGULAR feature graph —
  // varying degrees make the coefficients irrational, so byte-identity is a real test that
  // both engines compute the same f64 in the same order.
  for (const op of ['mean', 'sum'] as const) {
    for (const direction of ['out', 'both'] as const) {
      for (const includeSelf of [false, true]) {
        test(`gcn norm ${op} / ${direction} / self=${includeSelf} is byte-identical`, async () => {
          const [ts, native] = await both({
            feature: 'h',
            op,
            direction,
            includeSelf,
            norm: 'gcn',
          });
          expect(ts).toBe(native);
        });
      }
    }
  }

  // Edge weighting on a weighted feature graph — weighted sum / mean, and GCN composed with
  // weights (coefficient = weight × norm).
  const WEIGHTED = [
    '{"type":"node","id":"a","labels":["N"],"properties":{"h":[0.1,0.2]}}',
    '{"type":"node","id":"b","labels":["N"],"properties":{"h":[1.5,2.5]}}',
    '{"type":"node","id":"c","labels":["N"],"properties":{"h":[0.7,0.9]}}',
    '{"type":"edge","from":"a","to":"b","labels":["R"],"properties":{"w":2.5}}',
    '{"type":"edge","from":"a","to":"c","labels":["R"],"properties":{"w":0.4}}',
    '{"type":"edge","from":"b","to":"c","labels":["R"],"properties":{"w":3.1}}',
    '{"type":"edge","from":"c","to":"a","labels":["R"],"properties":{"w":1.2}}',
  ].join('\n');
  const natW = graphFromNdjson(backend, WEIGHTED);
  const tsW = tsDeserialize(WEIGHTED, 'ndjson', new Graph());

  for (const op of ['mean', 'sum'] as const) {
    for (const direction of ['out', 'both'] as const) {
      test(`weighted ${op} / ${direction} is byte-identical`, async () => {
        const cfg = { feature: 'h', op, direction, weightProperty: 'w' } as const;
        expect(JSON.stringify(await neighborAggregate(cfg, tsW))).toBe(
          JSON.stringify(await natW.neighborAggregate(cfg)),
        );
      });
    }
  }

  test('weighted + gcn compose byte-identically', async () => {
    const cfg = {
      feature: 'h',
      op: 'mean' as const,
      direction: 'both' as const,
      weightProperty: 'w',
      norm: 'gcn' as const,
      includeSelf: true,
    };
    expect(JSON.stringify(await neighborAggregate(cfg, tsW))).toBe(
      JSON.stringify(await natW.neighborAggregate(cfg)),
    );
  });

  test('writeProperty round-trips the aggregate list identically through GQL', async () => {
    const config = { feature: 'h', op: 'sum', direction: 'out', writeProperty: 'agg' } as const;
    await neighborAggregate(config, tsGraph);
    await nativeGraph.neighborAggregate(config);

    const readBack = 'MATCH (n) RETURN n.agg AS agg ORDER BY n.id';
    expect(JSON.stringify(tsQuery(tsGraph, readBack))).toBe(
      JSON.stringify(nativeGraph.query(readBack)),
    );
  });

  test('a missing feature contributes nothing, both engines', async () => {
    // `e` has no `h`, so when `a` aggregates its out-neighbours (b, c, d, e) the
    // featureless `e` is skipped: a.sum == b + c + d, byte-identical.
    const [ts, native] = await both({ feature: 'h', op: 'sum', direction: 'out' });
    expect(ts).toBe(native);
    const rows = JSON.parse(ts) as { node: string; vector: number[] }[];
    // a.sum == b + c + d (featureless e excluded); compare per element.
    const b = [1.5, 2.5, 3.5];
    const c = [0.7, 0.9, 1.1];
    const d = [9.9, 8.8, 7.7];
    const a = rows.find((r) => r.node === 'a')!.vector;
    b.forEach((_, i) => expect(a[i]).toBeCloseTo(b[i] + c[i] + d[i], 10));
  });

  test('CALL neighbor_aggregate is byte-identical to the method', async () => {
    // `vector` (not the reserved `aggregate`) is the result column.
    const call =
      "CALL neighbor_aggregate({feature:'h', op:'mean', direction:'both'}) YIELD node, vector RETURN node, vector ORDER BY node";
    expect(JSON.stringify(tsQuery(tsGraph, call))).toBe(JSON.stringify(nativeGraph.query(call)));
  });

  // The CALL config-key allowlist must accept `norm` (and `weightProperty`) on BOTH
  // engines — a native-only allowlist would reject the documented GCN recipe via GQL even
  // though the method form works. Guards the exact gap that shipped `norm` to native's
  // CONFIG_KEYS but not the TS mirror.
  test('CALL neighbor_aggregate accepts norm:gcn / weightProperty, byte-identical', async () => {
    for (const cfg of [
      "{feature:'h', op:'sum', direction:'both', includeSelf:true, norm:'gcn'}",
      "{feature:'h', op:'mean', direction:'out', weightProperty:'w'}",
    ]) {
      const call = `CALL neighbor_aggregate(${cfg}) YIELD node, vector RETURN node, vector ORDER BY node`;
      expect(JSON.stringify(tsQuery(tsGraph, call)), call).toBe(
        JSON.stringify(nativeGraph.query(call)),
      );
    }
  });
});
