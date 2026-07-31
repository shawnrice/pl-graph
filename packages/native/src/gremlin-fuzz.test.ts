// Differential fuzzer for GREMLIN. The GQL fuzzers cover reads and writes in the
// other language; Gremlin had a hand-written conformance corpus but no generator,
// and a corpus only covers what someone thought to write down.
//
// One source of truth per case, exactly like gremlin-conformance.test.ts: build a
// random `Plan`, run it on the TS engine directly, emit it to Groovy via
// `planToGremlin` and run THAT on the Rust core, then compare canonicalized JSON.
// Every iteration therefore also exercises the emitter and `parse.rs`.
//
// It immediately earned its keep: the conformance harness's `canonJson` claimed
// the engine emitted `{id, label}` for an element when it emits the rich
// `{id, labels, properties}` form, and no corpus case returned a bare element — so
// element results were never compared across engines at all. Fixed, with cases.
//
// Seed: random each run (FUZZ_SEED=<n> to replay); the failing seed is printed.
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { Edge, isElement } from '@lenke/core';
import {
  V,
  E,
  both,
  bothE,
  count,
  dedupe,
  eq,
  fold,
  gt,
  gte,
  has,
  hasLabel,
  id,
  inE,
  inV,
  label,
  limit,
  lt,
  lte,
  order,
  Order,
  out,
  outE,
  outV,
  otherV,
  path,
  planToGremlin,
  range,
  simplePath,
  skip,
  sum,
  toArray,
  traversal,
  values,
  createTestTinkerGraph,
  type Plan,
} from '@lenke/gremlin';

import { createFfiBackend } from './backend-ffi.js';

// Normalize a TS result to the Rust JSON-carrier shape (a copy of the
// conformance suite's `canonJson`; importing it from a *.test.ts pulls in
// `describe`, which only exists under the test runner).
const canonJson = (v: unknown): unknown => {
  if (v === null || typeof v === 'boolean' || typeof v === 'string') {
    return v;
  }

  if (typeof v === 'number') {
    return Number.isFinite(v) ? v : null;
  }

  if (typeof v === 'bigint') {
    return Number(v);
  }

  if (isElement(v)) {
    const props: Record<string, unknown> = {};

    for (const k of Object.keys(v.properties).sort()) {
      props[k] = canonJson(v.properties[k]);
    }

    const labels = [...v.labels].sort();

    return v instanceof Edge
      ? { id: v.id, from: v.from.id, to: v.to.id, labels, properties: props }
      : { id: v.id, labels, properties: props };
  }

  if (Array.isArray(v)) {
    return v.map(canonJson);
  }

  if (v instanceof Map) {
    const o: Record<string, unknown> = {};

    for (const [k, val] of v) {
      o[String(k)] = canonJson(val);
    }

    return o;
  }

  if (typeof v === 'object') {
    const o: Record<string, unknown> = {};

    for (const [k, val] of Object.entries(v)) {
      o[k] = canonJson(val);
    }

    return o;
  }

  return v;
};

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-core/target/release/liblenke_core.${LIB_EXT}`,
  import.meta.url,
).pathname;
const suite = existsSync(LIB) ? describe : describe.skip;
const backend = existsSync(LIB) ? createFfiBackend(LIB) : null;
const decoder = new TextDecoder();
const MODERN = [
  '{"type":"node","id":"1","labels":["PERSON"],"properties":{"name":"marko","age":29}}',
  '{"type":"node","id":"2","labels":["PERSON"],"properties":{"name":"vadas","age":27}}',
  '{"type":"node","id":"4","labels":["PERSON"],"properties":{"name":"josh","age":32}}',
  '{"type":"node","id":"6","labels":["PERSON"],"properties":{"name":"peter","age":35}}',
  '{"type":"node","id":"3","labels":["SOFTWARE"],"properties":{"name":"lop","lang":"java"}}',
  '{"type":"node","id":"5","labels":["SOFTWARE"],"properties":{"name":"ripple","lang":"java"}}',
  '{"type":"edge","id":"7","from":"1","to":"2","labels":["KNOWS"],"properties":{"weight":0.5}}',
  '{"type":"edge","id":"8","from":"1","to":"4","labels":["KNOWS"],"properties":{"weight":1.0}}',
  '{"type":"edge","id":"9","from":"1","to":"3","labels":["CREATED"],"properties":{"weight":0.4}}',
  '{"type":"edge","id":"10","from":"4","to":"5","labels":["CREATED"],"properties":{"weight":1.0}}',
  '{"type":"edge","id":"11","from":"4","to":"3","labels":["CREATED"],"properties":{"weight":0.4}}',
  '{"type":"edge","id":"12","from":"6","to":"3","labels":["CREATED"],"properties":{"weight":0.2}}',
].join('\n');

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
const pick = <T>(r: () => number, xs: readonly T[]): T => xs[Math.floor(r() * xs.length)];

const KEYS = ['name', 'age', 'lang', 'weight', 'missing'];
const VALUES: unknown[] = ['marko', 'lop', 'java', 29, 0.4, 0, -1, '', 'nope'];
const LABELS = ['PERSON', 'SOFTWARE', 'KNOWS', 'CREATED', 'NOPE'];
const preds = [eq, gt, gte, lt, lte];

// Steps that move the traverser, filter it, or reshape it — everything that can
// cross the Groovy text boundary (no JS closures, no non-finite literals).
const step = (r: () => number): unknown => {
  const p = r();

  if (p < 0.13) {
    return out(pick(r, ['KNOWS', 'CREATED']));
  }

  if (p < 0.22) {
    return outE(pick(r, ['KNOWS', 'CREATED']));
  }

  if (p < 0.28) {
    return inE(pick(r, ['KNOWS', 'CREATED']));
  }

  if (p < 0.33) {
    return both(pick(r, ['KNOWS', 'CREATED']));
  }

  if (p < 0.37) {
    return bothE(pick(r, ['KNOWS', 'CREATED']));
  }

  if (p < 0.41) {
    return inV();
  }

  if (p < 0.45) {
    return outV();
  }

  if (p < 0.48) {
    return otherV();
  }

  if (p < 0.58) {
    return has(pick(r, KEYS), pick(r, preds)(pick(r, VALUES) as never));
  }

  if (p < 0.63) {
    return hasLabel(pick(r, LABELS));
  }

  if (p < 0.68) {
    return values(pick(r, KEYS));
  }

  if (p < 0.71) {
    return label();
  }

  if (p < 0.74) {
    return id();
  }

  if (p < 0.78) {
    return dedupe();
  }

  if (p < 0.82) {
    return limit(Math.floor(r() * 4));
  }

  if (p < 0.85) {
    return skip(Math.floor(r() * 3));
  }

  if (p < 0.88) {
    return range(Math.floor(r() * 2), Math.floor(r() * 4));
  }

  if (p < 0.91) {
    return order(pick(r, [Order.asc, Order.desc]));
  }

  if (p < 0.94) {
    return simplePath();
  }

  if (p < 0.97) {
    return path();
  }

  return count();
};

const terminal = (r: () => number): unknown[] => {
  const p = r();

  if (p < 0.3) {
    return [count()];
  }

  if (p < 0.45) {
    return [fold()];
  }

  if (p < 0.6) {
    return [values(pick(r, KEYS))];
  }

  if (p < 0.7) {
    return [sum()];
  }

  if (p < 0.8) {
    return [dedupe(), count()];
  }

  return [];
};

const genPlan = (r: () => number): Plan => {
  const start = r() < 0.8 ? V() : E();
  const n = 1 + Math.floor(r() * 4);
  const steps = Array.from({ length: n }, () => step(r));

  return traversal(start, ...(steps as never[]), ...(terminal(r) as never[]));
};

const nativeRun = (text: string): unknown[] => {
  const handle = backend!.graphFromNdjson(new TextEncoder().encode(MODERN), false);

  try {
    return JSON.parse(decoder.decode(backend!.gremlinJson(handle, text))) as unknown[];
  } finally {
    backend!.graphFree(handle);
  }
};

suite('differential fuzz: gremlin (TS engine vs Rust core)', () => {
  const tsGraph = createTestTinkerGraph();
  const SEED =
    process.env.FUZZ_SEED === undefined
      ? Math.floor(Math.random() * 0x1_0000_0000)
      : Number(process.env.FUZZ_SEED) >>> 0;
  const ITERATIONS = 400;

  test(`${ITERATIONS} random traversals agree across the engines`, () => {
    const divergences: string[] = [];

    for (let i = 0; i < ITERATIONS && divergences.length < 5; i++) {
      const r = mulberry32(SEED + i);
      let plan: Plan;
      let text: string;

      try {
        plan = genPlan(r);
        text = planToGremlin(plan);
      } catch {
        continue; // a kind that cannot cross the text boundary — by design
      }

      const outcome = (run: () => unknown): string => {
        try {
          return JSON.stringify(run());
        } catch (e) {
          return `ERR ${(e as { code?: string }).code ?? 'throw'}`;
        }
      };
      const ts = outcome(() => toArray(plan, tsGraph).map(canonJson));
      const native = outcome(() => nativeRun(text));

      // Both failing is acceptable — each rejects the query. A divergence is one
      // side succeeding, or both succeeding with different results.
      if (ts !== native && !(ts.startsWith('ERR') && native.startsWith('ERR'))) {
        divergences.push(`[seed ${SEED + i}] ${text}\n    ts:     ${ts}\n    native: ${native}`);
      }
    }

    const report = divergences.length
      ? `FUZZ_SEED=${SEED} bun test <this file> to reproduce:\n\n${divergences.join('\n\n')}`
      : 'no divergences';

    expect(report).toBe('no divergences');
  });
});
