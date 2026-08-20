// Randomized differential for the in-engine graph ALGORITHMS: the same random
// graph into both engines, the same config, byte-identical JSON out.
//
// `algo-conformance.test.ts` covers the algorithms against hand-picked fixtures.
// That leaves the shapes nobody thinks to write down — self-loops, parallel
// edges, isolated vertices, several components, and weights that are zero,
// negative, huge or absent — which is where traversal code actually breaks. This
// generates them.
//
// It earned its keep immediately: a negative `weightProperty` made `shortestPath`
// spin forever on BOTH engines (Dijkstra's non-negative precondition was
// documented and unenforced), and a negative self-loop — one node, one edge —
// was enough. Both engines now reject it; see the `shortestPath rejects negative
// weights` tests.
//
// Errors count as results: an engine that faults must fault on the other too,
// with the same code.
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import {
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
  type AlgorithmConfig,
} from '@lenke/core';
import { deserialize as tsDeserialize } from '@lenke/serialization';

import { createFfiEngineBackend } from './backend-ffi-engine.js';
import { graphFromNdjson } from './graph.js';

const LIB = new URL('../../../crates/lenke-engine/target/release/liblenke_engine.so', import.meta.url)
  .pathname;
const hasLib = existsSync(LIB);
const suite = hasLib ? describe : describe.skip;
const ffi = hasLib ? createFfiEngineBackend(LIB) : null;

const mulberry32 = (seed: number): (() => number) => {
  let a = seed >>> 0;

  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

// Distinct `FUZZ_SEED`s must explore DISJOINT cases. `SEED + i` did not: seeds 1
// and 2 differ in one case out of four hundred, so running eight seeds was ~1.02x
// the coverage of running one, not 8x. Multiplying by a large odd constant gives
// each base seed its own region while keeping a reported seed reproducible.
const caseSeed = (base: number, i: number): number => base * 1_000_003 + i;
const pick = <T>(r: () => number, xs: readonly T[]): T => xs[Math.floor(r() * xs.length)];
const int = (r: () => number, n: number): number => Math.floor(r() * n);

const ETYPES = ['KNOWS', 'CREATED', 'LIKES'];
const WEIGHTS = ['1', '0', '-1', '2.5', '0.1', '-0.5', '1e10', 'null'];

/**
 * A graph built to break traversal code: self-loops, parallel edges, isolated
 * vertices, several components, a hub, and a chain — plus weights that are zero,
 * negative, fractional, huge, null and absent.
 */
const randomGraph = (r: () => number): { ndjson: string; ids: string[] } => {
  const n = 3 + int(r, 14);
  const ids = Array.from({ length: n }, (_, i) => `v${i}`);
  const lines = ids.map((id, i) => {
    const props: string[] = [`idx: ${i}`];

    if (r() < 0.7) {
      props.push(`vec: [1, 2]`);
      props.push(`seed: ${pick(r, ['1', '2', 'null', '"a"'])}`);
    }

    return `{"type":"node","id":"${id}","labels":["${pick(r, ['P', 'Q'])}"],"properties":{${props
      .join(', ')
      .replace(/(\w+):/g, '"$1":')}}}`;
  });

  const edges: string[] = [];
  const addEdge = (a: string, b: string): void => {
    const w = pick(r, WEIGHTS);
    const props = r() < 0.85 ? `,"properties":{"w":${w}}` : '';

    edges.push(
      `{"type":"edge","id":"e${edges.length}","labels":["${pick(r, ETYPES)}"],"from":"${a}","to":"${b}"${props}}`,
    );
  };

  // A chain, so there are real paths.
  for (let i = 0; i + 1 < n && r() < 0.8; i++) {
    addEdge(ids[i], ids[i + 1]);
  }

  // A hub.
  if (r() < 0.6) {
    for (let i = 1; i < n; i++) {
      if (r() < 0.5) {
        addEdge(ids[0], ids[i]);
      }
    }
  }

  // Random extras, self-loops and parallel edges included.
  const extra = int(r, n * 2);

  for (let i = 0; i < extra; i++) {
    addEdge(ids[int(r, n)], ids[int(r, n)]);
  }

  return { ndjson: [...lines, ...edges].join('\n'), ids };
};

const configs = (r: () => number, ids: string[]): AlgorithmConfig[] => {
  const base: AlgorithmConfig[] = [
    {},
    { direction: 'out' },
    { direction: 'in' },
    { direction: 'both' },
    { edgeLabel: pick(r, ETYPES) },
    { weightProperty: 'w' },
    { weightProperty: 'missing' },
    { direction: 'both', weightProperty: 'w' },
    { iterations: int(r, 5) },
    { iterations: 30 },
    { dampingFactor: pick(r, [0, 0.5, 0.85, 1]) },
    { pivots: 1 + int(r, ids.length + 2) },
    { seedProperty: 'seed' },
    { source: pick(r, ids) },
    { source: pick(r, ids), weightProperty: 'w' },
    { sourceNodes: [pick(r, ids)] },
  ];

  return base;
};

type Algo = {
  name: string;
  ts: (c: AlgorithmConfig, g: Graph) => unknown;
  nat: (g: ReturnType<typeof graphFromNdjson>, c: AlgorithmConfig) => unknown;
  needsSource?: boolean;
};

const ALGOS: Algo[] = [
  { name: 'degree', ts: (c, g) => degree(c, g), nat: (g, c) => g.degree(c) },
  {
    name: 'connectedComponents',
    ts: (c, g) => connectedComponents(c, g),
    nat: (g, c) => g.connectedComponents(c),
  },
  {
    name: 'labelPropagation',
    ts: (c, g) => labelPropagation(c, g),
    nat: (g, c) => g.labelPropagation(c),
  },
  { name: 'peerPressure', ts: (c, g) => peerPressure(c, g), nat: (g, c) => g.peerPressure(c) },
  { name: 'pagerank', ts: (c, g) => pagerank(c, g), nat: (g, c) => g.pagerank(c) },
  { name: 'betweenness', ts: (c, g) => betweenness(c, g), nat: (g, c) => g.betweenness(c) },
  { name: 'closeness', ts: (c, g) => closeness(c, g), nat: (g, c) => g.closeness(c) },
  {
    name: 'shortestPath',
    ts: (c, g) => shortestPath(c, g),
    nat: (g, c) => g.shortestPath(c),
    needsSource: true,
  },
  {
    name: 'neighborAggregate',
    ts: (c, g) => neighborAggregate({ ...c, feature: 'vec' } as AlgorithmConfig, g),
    nat: (g, c) => g.neighborAggregate({ ...c, feature: 'vec' } as AlgorithmConfig),
  },
];

suite('algorithm differential: random graphs agree across engines', () => {
  const SEED_BASE =
    process.env.FUZZ_SEED === undefined
      ? Math.floor(Math.random() * 0x1_0000_0000)
      : Number(process.env.FUZZ_SEED) >>> 0;
  const ITERATIONS = 25;

  test(`${ITERATIONS} random graphs x every algorithm x every config`, async () => {
    const findings: string[] = [];
    let checks = 0;

    for (let i = 0; i < ITERATIONS && findings.length === 0; i++) {
      const r = mulberry32(caseSeed(SEED_BASE, i));
      const { ndjson, ids } = randomGraph(r);
      const tsG = tsDeserialize(ndjson, 'ndjson', new Graph());
      const natG = graphFromNdjson(ffi!, ndjson);

      try {
        for (const algo of ALGOS) {
          for (const c of configs(r, ids)) {
            if (algo.needsSource && c.source === undefined) {
              continue;
            }

            checks++;

            const run = async (f: () => unknown): Promise<string> => {
              try {
                return JSON.stringify(await f()) ?? 'undefined';
              } catch (e) {
                return `ERR ${(e as { code?: string }).code ?? 'UNCODED'}`;
              }
            };
            const a = await run(() => algo.ts(c, tsG));
            const b = await run(() => algo.nat(natG, c));

            if (a !== b && findings.length < 3) {
              findings.push(
                `${algo.name} ${JSON.stringify(c)}\n  ts:     ${a.slice(0, 200)}\n  native: ${b.slice(0, 200)}\n  graph:\n${ndjson}`,
              );
            }
          }
        }
      } finally {
        natG.free();
      }
    }

    expect(checks).toBeGreaterThan(0);

    const report = findings.length
      ? `FUZZ_SEED=${SEED_BASE} bun test <this file> to reproduce:\n\n${findings.join('\n\n')}`
      : 'no divergences';

    expect(report).toBe('no divergences');
  });
});
