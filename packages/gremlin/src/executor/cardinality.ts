import { isSliceable, type Traverser } from './runtime.js';

export const takeTraversers = function* (
  stream: Iterable<Traverser<unknown>>,
  n: number,
): Iterable<Traverser<unknown>> {
  let i = 0;

  for (const t of stream) {
    if (i >= n) {
      return;
    }

    yield t;
    i++;
  }
};

export const skipTraversers = function* (
  stream: Iterable<Traverser<unknown>>,
  n: number,
): Iterable<Traverser<unknown>> {
  let i = 0;

  for (const t of stream) {
    if (i < n) {
      i++;
      continue;
    }

    yield t;
  }
};

export const tailTraversers = function* (
  stream: Iterable<Traverser<unknown>>,
  n: number,
): Iterable<Traverser<unknown>> {
  const buf: Traverser<unknown>[] = [];

  for (const t of stream) {
    buf.push(t);

    if (buf.length > n) {
      buf.shift();
    }
  }

  yield* buf;
};

// Local-scope slice helpers. Operate on a single traverser's iterable value
// (typically an array produced by `fold()` or a list-cardinality projection).
// `isSliceable` lives in `runtime.ts` (shared with the local-scope
// aggregations).
export const sliceLocal = (v: unknown, start: number, end: number): unknown => {
  if (!isSliceable(v)) {
    return v;
  }

  const arr = [...v];

  return arr.slice(start, end === Infinity ? undefined : end);
};

export const tailLocal = (v: unknown, n: number): unknown => {
  if (!isSliceable(v)) {
    return v;
  }

  const arr = [...v];

  return n >= arr.length ? arr : arr.slice(arr.length - n);
};

// Mulberry32 PRNG — tiny, fast, fully-specified. `sample(n)` uses it with a
// FIXED seed so the pseudo-random selection is reproducible AND byte-identical
// with the Rust engine, which runs the same algorithm. The seed constant and the
// draw/shuffle order must match `crates/lenke-engine/src/gremlin.rs`.
const SAMPLE_SEED = 0x9e3779b9;

const mulberry32 = (seed: number): (() => number) => {
  let s = seed >>> 0;

  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); // xor is commutative → matches Rust

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

// `sample(n)` — a pseudo-random partial Fisher-Yates pick-N over the stream,
// seeded (not `Math.random`) so it's reproducible and matches the Rust engine.
export const sampleStep = function* (
  stream: Iterable<Traverser<unknown>>,
  n: number,
): Iterable<Traverser<unknown>> {
  const buf = [...stream];
  const k = Math.min(n, buf.length);
  const rand = mulberry32(SAMPLE_SEED);

  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(rand() * (buf.length - i));
    const tmp = buf[i];
    buf[i] = buf[j]!;
    buf[j] = tmp;
  }

  for (let i = 0; i < k; i++) {
    yield buf[i];
  }
};
