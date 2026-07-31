// Differential fuzzer: generate random GQL queries from a seeded PRNG, run each
// through BOTH engines (the TS @lenke/gql engine and the Rust core over bun:ffi),
// and assert byte-identical behavior — the same JSON when both succeed, and
// both-error when either errors. Byte-identity is the hard invariant, so any value
// that renders/compares/coerces differently between the two engines is a bug. The
// generator deliberately favors the edge values that have bitten us before (extreme
// magnitudes, -0, NaN/Inf producers, astral/control chars, mixed types, deep
// nesting), and reaches across the whole scalar-function catalogue rather than a
// handful of names — the wider surface is where the coercion bugs live.
//
// Statement shapes are fuzzed too (aggregates, GROUP BY, ORDER BY, WHERE/FILTER,
// LET, FOR, SKIP/LIMIT, edge patterns), because several divergences were not in an
// expression at all but at a clause boundary.
//
// ORDER BY queries always carry `n.n` as a final sort key: row order is otherwise
// unspecified (see the ORDER-BY-less contract), and a tie would make the comparison
// flag a non-bug.
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { Graph } from '@lenke/core';
import { query as tsQuery } from '@lenke/gql';
import { deserialize as tsDeserialize } from '@lenke/serialization';

import { createFfiBackend } from './backend-ffi.js';
import { graphFromNdjson } from './graph.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-core/target/release/liblenke_core.${LIB_EXT}`,
  import.meta.url,
).pathname;
const suite = existsSync(LIB) ? describe : describe.skip;

// A tiny two-vertex, one-edge graph so property access, record fields, edge
// patterns, and aggregates over rows can all be fuzzed.
const NDJSON = [
  '{"type":"node","id":"1","labels":["T"],"properties":{"n":3,"s":"a","x":-1,"m":{"k":1,"j":"q"}}}',
  '{"type":"node","id":"2","labels":["T"],"properties":{"n":7,"s":"z","x":4,"m":{"k":2,"j":"r"}}}',
  '{"type":"edge","id":"e1","labels":["E"],"from":"1","to":"2","properties":{"w":2}}',
].join('\n');

// --- seeded PRNG (mulberry32) -----------------------------------------------
const mulberry32 = (seed: number): (() => number) => {
  let a = seed >>> 0;

  return () => {
    a |= 0;
    a = (a + 0x6d_2b_79_f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);

    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  };
};

const pick = <T>(r: () => number, xs: readonly T[]): T => xs[Math.floor(r() * xs.length)];

// Number literals: normal, signed-zero, extreme magnitudes (exponential threshold),
// safe-integer boundary, and values that stress toString/collation.
const NUMS = [
  '0',
  '1',
  '-1',
  '0.0',
  '-0.0',
  '3.14',
  '-2.5',
  '0.1',
  '0.5',
  '1e21',
  '1e-7',
  '1e100',
  '1e-300',
  '123456.789',
  '9007199254740992',
  '2',
  '3',
  '100',
] as const;

// String literals (GQL single-quoted), including astral, BMP-boundary, control,
// and escape chars — the collation / escaping edge space.
const STR_CHARS = ['a', 'Z', '0', ' ', '\\u0041', '\\uE000', '\\n', '\\t', 'ä', '中'] as const;

const genString = (r: () => number): string => {
  const len = Math.floor(r() * 4);
  let s = "'";

  for (let i = 0; i < len; i++) {
    s += pick(r, STR_CHARS);
  }

  // Occasionally splice an astral char (surrogate pair) — the code-point vs
  // code-unit collation trap.
  if (r() < 0.2) {
    s += '😀';
  }

  return `${s}'`;
};

const TEMPORALS = [
  "date('2020-01-01')",
  "date('0001-01-01')",
  "datetime('2020-06-15T08:30:00')",
  "duration('P1Y2M')",
  "duration('P3DT4H')",
  "duration('PT-2.5S')",
  "zoned_datetime('2020-01-01T00:00:00Z')",
  "local_time('08:30:00')",
  "zoned_time('08:30:00+05:30')",
] as const;

// Strings that stress numeric-string coercion: non-finite spellings, radix
// prefixes, whitespace, empty — the exact forms JS Number() and Rust
// str::parse::<f64> disagree on.
const NUM_STRINGS = ["'inf'", "'nan'", "'0x10'", "'  5  '", "''", "'Infinity'", "'1e3'"] as const;

// A leaf: a literal value, a record/field access, or a property reference.
const genLeaf = (r: () => number): string => {
  const p = r();

  if (p < 0.28) {
    return pick(r, NUMS);
  }

  if (p < 0.36) {
    return pick(r, NUM_STRINGS);
  }

  if (p < 0.5) {
    return genString(r);
  }

  if (p < 0.56) {
    return r() < 0.5 ? 'true' : 'false';
  }

  if (p < 0.62) {
    return 'null';
  }

  if (p < 0.74) {
    return pick(r, TEMPORALS);
  }

  if (p < 0.8) {
    // A stored record, and a dotted path into one (including a missing field).
    return r() < 0.5 ? 'n.m' : `n.m.${pick(r, ['k', 'j', 'zz'])}`;
  }

  // property access over the row (n: number, s: string, x: number)
  return `n.${pick(r, ['n', 's', 'x'])}`;
};

const ARITH = ['+', '-', '*', '/', '%'] as const;
const CMP = ['<', '>', '=', '<=', '>=', '<>'] as const;

// The engine's unary scalar catalogue: numeric, string, conversion, and list.
// `power` is deliberately ABSENT from the binary list below — see the note there.
const UNARY_FN = [
  'abs',
  'sqrt',
  'ceil',
  'ceiling',
  'floor',
  'sign',
  'round',
  'exp',
  'ln',
  'log10',
  'sin',
  'cos',
  'tan',
  'cot',
  'asin',
  'acos',
  'atan',
  'sinh',
  'cosh',
  'tanh',
  'degrees',
  'radians',
  'to_string',
  'to_integer',
  'to_float',
  'to_boolean',
  'to_list',
  'size',
  'cardinality',
  'upper',
  'lower',
  'trim',
  'btrim',
  'ltrim',
  'rtrim',
  'char_length',
  'character_length',
  'byte_length',
  'octet_length',
  'reverse',
  'head',
  'last',
  'tail',
  'list_sort',
] as const;

// `power` is excluded on purpose: JS `Math.pow` uses repeated multiplication for
// integer exponents while Rust's `f64::powf` is correctly rounded, so the two
// engines differ in the last ulp (`power(1e-7, 3)` → 9.999999999999997e-22 vs
// 1e-21). That is a known, reported deviation, not something this fuzzer should
// rediscover on every run. Every other math function agrees exactly.
const BINARY_FN = [
  'mod',
  'log',
  'atan2',
  'left',
  'right',
  'split',
  'contains',
  'starts_with',
  'ends_with',
  'nullif',
  'append',
  'list_union',
  'intersection',
  'difference',
  'list_contains',
  'coalesce',
  'duration_between',
] as const;

const TERNARY_FN = ['substring', 'replace'] as const;
const CAST_TYPES = ['INTEGER', 'FLOAT', 'STRING', 'BOOLEAN'] as const;
const IS_TESTS = [
  'IS NULL',
  'IS NOT NULL',
  'IS TRUE',
  'IS NOT TRUE',
  'IS FALSE',
  'IS UNKNOWN',
] as const;

const genExpr = (r: () => number, depth: number): string => {
  if (depth <= 0 || r() < 0.32) {
    return genLeaf(r);
  }

  const p = r();

  const e = (): string => genExpr(r, depth - 1);

  if (p < 0.14) {
    return `(${e()} ${pick(r, ARITH)} ${e()})`;
  }

  if (p < 0.24) {
    return `(${e()} ${pick(r, CMP)} ${e()})`;
  }

  if (p < 0.38) {
    return `${pick(r, UNARY_FN)}(${e()})`;
  }

  if (p < 0.5) {
    return `${pick(r, BINARY_FN)}(${e()}, ${e()})`;
  }

  // `range` takes bounded literal arguments: it materializes the whole list
  // eagerly, so a fuzzed `range(0, 1e21)` would hang both engines instead of
  // exploring anything.
  if (p < 0.52) {
    return `range(${pick(r, ['0', '1', '-3'])}, ${pick(r, ['0', '3', '-1', '10'])})`;
  }

  if (p < 0.58) {
    return `${pick(r, TERNARY_FN)}(${e()}, ${e()}, ${e()})`;
  }

  if (p < 0.64) {
    return `[${e()}, ${e()}]`;
  }

  // ISO list indexing — 0-based; a null/negative/non-integer/out-of-range index
  // is null-safe, not an error.
  if (p < 0.7) {
    return `[${e()}, ${e()}][${pick(r, ['0', '1', '2', '-1', 'null', "'a'", '0.5'])}]`;
  }

  if (p < 0.74) {
    return `(${e()} || ${e()})`;
  }

  if (p < 0.78) {
    return `CAST(${e()} AS ${pick(r, CAST_TYPES)})`;
  }

  if (p < 0.82) {
    return `(${e()} ${pick(r, IS_TESTS)})`;
  }

  if (p < 0.86) {
    return `(${e()} ${pick(r, ['AND', 'OR', 'XOR'])} ${e()})`;
  }

  if (p < 0.89) {
    return `(NOT ${e()})`;
  }

  if (p < 0.92) {
    return `(${e()} IN [${e()}, ${e()}])`;
  }

  // Record constructor, half the time with a field access (including a missing one).
  if (p < 0.95) {
    return `{a: ${e()}, b: ${e()}}${r() < 0.5 ? `.${pick(r, ['a', 'b', 'zz'])}` : ''}`;
  }

  if (p < 0.97) {
    return `coalesce(${e()}, ${e()}, ${e()})`;
  }

  return `CASE WHEN (${e()} ${pick(r, CMP)} ${e()}) THEN ${e()} WHEN (${e()} ${pick(r, CMP)} ${e()}) THEN ${e()} ELSE ${e()} END`;
};

const AGG = [
  'count',
  'sum',
  'avg',
  'min',
  'max',
  'collect_list',
  'stddev_pop',
  'stddev_samp',
] as const;

// A full query. Every ORDER BY ends with the distinct `n.n` so the row order is
// total — an unordered tie is unspecified, not a divergence.
const genQuery = (r: () => number): string => {
  const p = r();

  if (p < 0.2) {
    const distinct = r() < 0.5 ? 'DISTINCT ' : '';

    return `MATCH (n:T) RETURN ${pick(r, AGG)}(${distinct}${genExpr(r, 2)}) AS x`;
  }

  if (p < 0.26) {
    const kind = r() < 0.5 ? 'cont' : 'disc';

    return `MATCH (n:T) RETURN percentile_${kind}(${genExpr(r, 2)}, ${pick(r, ['0', '0.5', '1', '0.25'])}) AS x`;
  }

  // Grouped aggregate — exercises GROUP BY keying over a fuzzed key.
  if (p < 0.34) {
    return `MATCH (n:T) RETURN ${genExpr(r, 1)} AS k, count(*) AS c GROUP BY k ORDER BY k, c`;
  }

  if (p < 0.42) {
    const dir = pick(r, ['ASC', 'DESC']);
    const nulls = pick(r, ['', ' NULLS FIRST', ' NULLS LAST']);

    return `MATCH (n:T) RETURN ${genExpr(r, 2)} AS x, n.n AS t ORDER BY x ${dir}${nulls}, t`;
  }

  if (p < 0.48) {
    return `MATCH (n:T) WHERE ${genExpr(r, 2)} RETURN n.n AS x ORDER BY x`;
  }

  if (p < 0.54) {
    return `MATCH (n:T) LET v = ${genExpr(r, 2)} RETURN v AS x, n.n AS t ORDER BY t`;
  }

  if (p < 0.6) {
    return `FOR v IN ${genExpr(r, 2)} RETURN v AS x`;
  }

  if (p < 0.64) {
    return `MATCH (n:T) FILTER ${genExpr(r, 2)} RETURN n.n AS x ORDER BY x`;
  }

  if (p < 0.68) {
    return `MATCH (a:T)-[e:E]->(b:T) RETURN ${genExpr(r, 2)} AS x`;
  }

  if (p < 0.72) {
    const skip = pick(r, ['0', '1', '2']);
    const limit = pick(r, ['0', '1', '2']);

    return `MATCH (n:T) RETURN ${genExpr(r, 2)} AS x, n.n AS t ORDER BY t SKIP ${skip} LIMIT ${limit}`;
  }

  // ISO `<order by and page statement>` in STATEMENT position — paging as a
  // pipeline step BEFORE the RETURN, which sorts/slices the binding table rather
  // than the projected rows. `n.n` is the final sort key so the order is total.
  if (p < 0.78) {
    const dir = pick(r, ['', ' DESC']);
    const page = pick(r, ['', ' OFFSET 1', ' LIMIT 2', ' OFFSET 1 LIMIT 1', ' LIMIT 0']);

    return `MATCH (n:T) ORDER BY ${genExpr(r, 2)}${dir}, n.n${page} RETURN ${genExpr(r, 2)} AS x, n.n AS t`;
  }

  return `MATCH (n:T) RETURN ${genExpr(r, 3)} AS x, n.n AS t ORDER BY t`;
};

const codeOf = (e: unknown): string =>
  (e as { code?: string })?.code ?? (e instanceof Error ? e.name : 'unknown');

type Outcome = { ok: true; json: string } | { ok: false; code: string };

suite('differential fuzz: TS gql engine vs Rust core', () => {
  const backend = createFfiBackend(LIB);
  const nativeGraph = graphFromNdjson(backend, NDJSON);
  const tsGraph = tsDeserialize(NDJSON, 'ndjson', new Graph());

  const run = (engine: 'ts' | 'native', q: string): Outcome => {
    try {
      const rows = engine === 'ts' ? tsQuery(tsGraph, q) : nativeGraph.query(q);

      return { ok: true, json: JSON.stringify(rows) };
    } catch (e) {
      return { ok: false, code: codeOf(e) };
    }
  };

  // Seed: random each run, so every run explores fresh expressions — fuzzing is
  // discovery, not a fixed corpus (the specific bugs found here are pinned as their
  // own deterministic unit tests, which are the permanent regression guards). Set
  // FUZZ_SEED=<n> to replay a run exactly; the seed is printed on failure so any
  // divergence is reproducible. Property-based-testing convention: random by
  // default, seed on failure (proptest, QuickCheck, fast-check all do this).
  const SEED =
    process.env.FUZZ_SEED !== undefined
      ? Number(process.env.FUZZ_SEED) >>> 0
      : Math.floor(Math.random() * 0x1_0000_0000);
  const ITERATIONS = 20_000;

  test(`${ITERATIONS} random queries render byte-identically across engines`, () => {
    const divergences: string[] = [];

    for (let i = 0; i < ITERATIONS; i++) {
      const q = genQuery(mulberry32(SEED + i));
      const ts = run('ts', q);
      const nat = run('native', q);

      // Both errored → acceptable (both reject the input); a shape divergence is
      // when exactly one succeeds, or both succeed with different JSON.
      if (ts.ok && nat.ok) {
        if (ts.json !== nat.json) {
          divergences.push(
            `[seed ${SEED + i}] ${q}\n    ts:     ${ts.json}\n    native: ${nat.json}`,
          );
        }
      } else if (ts.ok !== nat.ok) {
        const tsSide = ts.ok ? `ok ${ts.json}` : `err ${(ts as { code: string }).code}`;
        const natSide = nat.ok ? `ok ${nat.json}` : `err ${(nat as { code: string }).code}`;
        divergences.push(`[seed ${SEED + i}] ${q}\n    ts:     ${tsSide}\n    native: ${natSide}`);
      }

      if (divergences.length >= 10) {
        break; // cap the report
      }
    }

    const report = divergences.length
      ? `FUZZ_SEED=${SEED} bun test <this file> to reproduce:\n\n${divergences.join('\n\n')}`
      : 'no divergences';
    expect(report).toBe('no divergences');
  });
});
