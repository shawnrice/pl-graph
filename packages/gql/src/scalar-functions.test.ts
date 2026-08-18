import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';
import { ErrorCode, hasErrorCode } from '@lenke/errors';

import type { Query } from './ast.js';
import { createTestSocialGraph } from './fixtures/createTestSocialGraph.js';
import { compile, parse, query } from './index.js';

// Mirrors the Rust engine's `scalar_functions_graph_string_list_conversion` and
// `unknown_function_errors_instead_of_silent_null` (gql/tests.rs) so the two
// engines stay byte-identical on the common ISO scalar/value functions.
const g = createTestSocialGraph();
const one = (q: string): unknown => {
  const rows = query(g, q);

  expect(rows).toHaveLength(1);

  return rows[0].x;
};

describe('GQL: ISO graph / conversion / string-list scalar functions', () => {
  test('graph functions (labels / type / keys are sorted)', () => {
    expect(query(g, `MATCH (n:Person {name:'marko'}) RETURN labels(n) AS x`)[0].x).toEqual([
      'Person',
    ]);
    expect(query(g, `MATCH ()-[r:KNOWS]->() RETURN type(r) AS x LIMIT 1`)[0].x).toBe('KNOWS');
    expect(query(g, `MATCH (n:Person {name:'marko'}) RETURN keys(n) AS x`)[0].x).toEqual([
      'age',
      'name',
    ]);
  });

  test('conversion (null in → null out; whole-string numeric parse)', () => {
    expect(one(`RETURN to_integer('42') AS x`)).toBe(42);
    expect(one(`RETURN to_float('3.5') AS x`)).toBe(3.5);
    expect(one(`RETURN to_string(42) AS x`)).toBe('42');
    // Strict parse — a trailing non-numeric tail is NULL on BOTH engines
    // (Rust `str::parse::<f64>()` rejects it; JS `parseFloat` would wrongly
    // read `12`, so the TS engine gates on the numeric grammar first).
    expect(one(`RETURN to_integer('12abc') AS x`)).toBeNull();
    expect(one(`RETURN to_float('nope') AS x`)).toBeNull();
    expect(one(`RETURN to_string(null) AS x`)).toBeNull();
  });

  test('string / list functions', () => {
    // 1-based start (SQL / ISO GQL convention): positions 1..3 of 'hello'.
    expect(one(`RETURN substring('hello', 1, 3) AS x`)).toBe('hel');
    // start past the end → empty; a start <= 0 shrinks the window from the front.
    expect(one(`RETURN substring('hello', 4) AS x`)).toBe('lo');
    expect(one(`RETURN substring('hello', 0, 3) AS x`)).toBe('he');
    expect(one(`RETURN split('a,b,c', ',') AS x`)).toEqual(['a', 'b', 'c']);
    expect(one(`RETURN replace('a.b.c', '.', '-') AS x`)).toBe('a-b-c');
    expect(one(`RETURN head([1, 2, 3]) AS x`)).toBe(1);
    expect(one(`RETURN last([1, 2, 3]) AS x`)).toBe(3);
    expect(one(`RETURN reverse('abc') AS x`)).toBe('cba');
  });

  test('math: round (half away from zero, optional digits), sign, pi, e', () => {
    expect(one(`RETURN round(2.5) AS x`)).toBe(3);
    expect(one(`RETURN round(-2.5) AS x`)).toBe(-3);
    expect(one(`RETURN round(3.14159, 2) AS x`)).toBe(3.14);
    expect(one(`RETURN round(1234.5678, -2) AS x`)).toBe(1200);
    expect(one(`RETURN sign(-3.7) AS x`)).toBe(-1);
    expect(one(`RETURN sign(0) AS x`)).toBe(0);
    expect(one(`RETURN sign(5) AS x`)).toBe(1);
    expect(one(`RETURN pi() AS x`)).toBe(Math.PI);
    expect(one(`RETURN e() AS x`)).toBe(Math.E);
    expect(one(`RETURN round(null) AS x`)).toBeNull();
  });

  test('infix CONTAINS / STARTS WITH / ENDS WITH predicates', () => {
    expect(one(`RETURN 'Hello World' CONTAINS 'World' AS x`)).toBe(true);
    expect(one(`RETURN 'Hello World' STARTS WITH 'Hello' AS x`)).toBe(true);
    expect(one(`RETURN 'Hello World' ENDS WITH 'World' AS x`)).toBe(true);
    expect(one(`RETURN 'Hello World' CONTAINS 'xyz' AS x`)).toBe(false);
    // as a WHERE filter over the social graph
    const names = query(g, `MATCH (p:Person) WHERE p.name STARTS WITH 'ma' RETURN p.name AS n`).map(
      (r) => r.n,
    );
    expect(names).toEqual(['marko']);
  });

  test('CAST(value AS type) desugars to the conversion functions', () => {
    expect(one(`RETURN CAST('42' AS INTEGER) AS x`)).toBe(42);
    expect(one(`RETURN CAST(3.7 AS INT) AS x`)).toBe(3);
    expect(one(`RETURN CAST('3.5' AS FLOAT) AS x`)).toBe(3.5);
    expect(one(`RETURN CAST(42 AS STRING) AS x`)).toBe('42');
    expect(one(`RETURN CAST('yes' AS BOOL) AS x`)).toBe(true);
    expect(one(`RETURN CAST('ab' AS LIST) AS x`)).toEqual(['a', 'b']);
    expect(one(`RETURN CAST('nope' AS INT) AS x`)).toBeNull();
  });

  test('temporal CAST targets desugar to the temporal constructor functions', () => {
    // `one` returns the raw temporal instance; compare its tagged JSON form.
    const tag = (q: string): unknown => (one(q) as { toJSON(): unknown }).toJSON();

    expect(tag(`RETURN CAST('2020-01-01' AS DATE) AS x`)).toEqual({ '@date': '2020-01-01' });
    expect(tag(`RETURN CAST('2020-01-01T08:30:00' AS DATETIME) AS x`)).toEqual({
      '@datetime': '2020-01-01T08:30:00',
    });
    // A bare date-only string coerces to midnight for a DATETIME target.
    expect(tag(`RETURN CAST('2020-01-01' AS TIMESTAMP) AS x`)).toEqual({
      '@datetime': '2020-01-01T00:00:00',
    });
    expect(tag(`RETURN CAST('2020-01-01T08:30:00' AS LOCAL DATETIME) AS x`)).toEqual({
      '@datetime': '2020-01-01T08:30:00',
    });
    expect(tag(`RETURN CAST('08:30:00' AS LOCAL TIME) AS x`)).toEqual({ '@localtime': '08:30:00' });
    expect(tag(`RETURN CAST('08:30:00+02:00' AS ZONED TIME) AS x`)).toEqual({
      '@zoned_time': '08:30:00+02:00',
    });
    expect(tag(`RETURN CAST('2020-01-01T08:30:00+02:00' AS ZONED DATETIME) AS x`)).toEqual({
      '@zoned_datetime': '2020-01-01T08:30:00+02:00',
    });
    expect(tag(`RETURN CAST('P1Y2M3DT4H' AS DURATION) AS x`)).toEqual({
      '@duration': 'P14M3DT14400S',
    });
    // A non-string operand desugars to the constructor and returns null (lenient).
    expect(one(`RETURN CAST(1 AS DATE) AS x`)).toBeNull();
  });

  test('CAST to an unrepresentable type is a loud syntax error', () => {
    expect(() => query(g, `RETURN CAST(1 AS BYTES) AS x`)).toThrow(/unsupported type/i);
    expect(() => query(g, `RETURN CAST(1 AS RECORD) AS x`)).toThrow(/unsupported type/i);
  });

  test('set-style list functions (dedup first-occurrence; sort reuses ORDER BY)', () => {
    expect(one(`RETURN list_union([1,2,2,3], [3,4,5]) AS x`)).toEqual([1, 2, 3, 4, 5]);
    expect(one(`RETURN intersection([1,2,3,3], [3,3,4,5]) AS x`)).toEqual([3]);
    expect(one(`RETURN difference([1,2,2,3], [3,4,5]) AS x`)).toEqual([1, 2]);
    // ISO GQL: list_contains returns numeric 1 / 0 (not a boolean).
    expect(one(`RETURN list_contains([1,2,3], 2) AS x`)).toBe(1);
    expect(one(`RETURN list_contains([1,2,3], 9) AS x`)).toBe(0);
    expect(one(`RETURN list_sort([3,1,4,1,5]) AS x`)).toEqual([1, 1, 3, 4, 5]);
    expect(one(`RETURN list_sort([3,1,2], 'desc') AS x`)).toEqual([3, 2, 1]);
    // null placement follows ORDER BY (default: nulls last on asc).
    expect(one(`RETURN list_sort([3,1,null,2]) AS x`)).toEqual([1, 2, 3, null]);
    expect(one(`RETURN list_sort([3,1,null,2], 'asc', 'first') AS x`)).toEqual([null, 1, 2, 3]);
  });

  test('an unknown function is an error, never a silent null', () => {
    // A typo'd/unknown function name is `UnknownFunction`, distinct from a
    // recognized-but-unimplemented feature — so a caller can tell them apart.
    expect(() => query(g, `RETURN nope_fn(1) AS x`)).toThrow(/unknown or unimplemented function/);

    try {
      query(g, `RETURN nope_fn(1) AS x`);

      throw new Error('expected a throw');
    } catch (e) {
      expect(hasErrorCode(e, ErrorCode.UnknownFunction)).toBe(true);
    }
  });

  test('an unknown function faults EAGERLY — even over empty input, a dead branch, or at compile', () => {
    // The name is resolved at COMPILE time (before any row runs), so an unknown
    // function faults identically whether the result set is empty or not, and even
    // when the call sits in a never-taken branch. A lazy per-row fault would
    // silently return `[]` over zero rows. Matches the Rust engine's plan-time
    // `unknown_fns` check.
    const codeOf = (fn: () => unknown): unknown => {
      try {
        fn();
      } catch (e) {
        return (e as { code?: unknown }).code;
      }

      throw new Error('expected a throw, got a normal return');
    };

    // Zero-row result still faults (the bug: this used to return []).
    expect(codeOf(() => query(g, `MATCH (n) WHERE false RETURN nope_fn(n) AS x`))).toBe(
      ErrorCode.UnknownFunction,
    );

    // A never-taken CASE branch: name resolution is reachability-independent.
    expect(codeOf(() => query(g, `RETURN CASE WHEN false THEN bogus_fn(1) ELSE 1 END AS x`))).toBe(
      ErrorCode.UnknownFunction,
    );

    // `compile(parse(...))` throws immediately — before the plan is ever run.
    expect(codeOf(() => compile(parse(`RETURN nope_fn(1) AS x`) as Query))).toBe(
      ErrorCode.UnknownFunction,
    );
  });
});

describe('typed results (opt-in row-shape generic)', () => {
  test('query<R> returns R[] — no per-field cast, and the values are correct', () => {
    // Compile-time: `name` is `string` (not `unknown`), so this assigns without a
    // cast. Runtime: the value is right. A regression in the generic breaks tsc.
    const rows = query<{ name: string }>(g, `MATCH (p:Person) RETURN p.name AS name`);
    const names: string[] = rows.map((r) => r.name);
    expect(names.length).toBeGreaterThan(0);
    expect(names.every((n) => typeof n === 'string')).toBe(true);
  });
});

// Regressions found by the cross-engine differential fuzzer
// (packages/native/src/differential-fuzz.test.ts). Each pins a case where this
// engine disagreed with the Rust core; the fuzzer is randomized, so these
// deterministic tests are the permanent guards.
describe('GQL: byte-identity regressions from the differential fuzzer', () => {
  const g2 = createTestSocialGraph();
  const val = (q: string): unknown => query(g2, q)[0].x;

  test('a value stringifies like the native engine, not like String(v)', () => {
    // A record is a `Map` subclass, so `String(v)` gave "[object Map]"; an element
    // and a path have debug `toString`s. All of them render through `str` now.
    expect(val(`RETURN to_string({b: 2, a: 1}) AS x`)).toBe('{"a":1,"b":2}');
    expect(val(`RETURN CAST({a: 1} AS STRING) AS x`)).toBe('{"a":1}');
    // upper()/char_length()/|| of a non-string (here a record) NO LONGER coerce — they
    // are type errors now (strict typing), matching the native engine — the explicit
    // to_string()/CAST above are the only stringification path.
    expect(() => val(`RETURN upper({a: 'q'}) AS x`)).toThrow();
    expect(() => val(`RETURN char_length({a: 1}) AS x`)).toThrow();
    expect(() => val(`RETURN ({a: 1} || 'x') AS x`)).toThrow();
    // A record nested in a list goes through `str` too — `Array.join` would have
    // re-entered `String` and reproduced "[object Map]".
    expect(val(`RETURN to_string([{a: 1}]) AS x`)).toBe('{"a":1}');
    // A temporal inside a record keeps its tagged serialized form.
    expect(val(`RETURN to_string({a: date('2020-01-01')}) AS x`)).toBe(
      '{"a":{"@date":"2020-01-01"}}',
    );
    // An element stringifies to its id (what `element_id` returns).
    expect(val(`MATCH (n:Person {name:'marko'}) RETURN to_string(n) AS x`)).toBe(
      val(`MATCH (n:Person {name:'marko'}) RETURN element_id(n) AS x`),
    );
    // A list joins with ',', rendering a null element as the empty string.
    expect(val(`RETURN to_string([1, null, 3]) AS x`)).toBe('1,,3');
  });

  test('right() truncates its length and rejects NaN', () => {
    // `slice(len - n)` with a fractional n moved the START index, taking one char
    // too many; with NaN it became `slice(0)` — the whole string.
    expect(val(`RETURN right('abcdef', 2.9) AS x`)).toBe('ef');
    // A NON-numeric length is now a type error (strict typing), not a coerced-to-NaN
    // empty string — the number position of right() is not JS-coerced.
    expect(() => val(`RETURN right('abcdef', 'nan') AS x`)).toThrow();
    expect(() => val(`RETURN right('abcdef', 'inf') AS x`)).toThrow();
    // The ordinary cases are unchanged.
    expect(val(`RETURN right('abcdef', 3) AS x`)).toBe('def');
    expect(val(`RETURN right('abcdef', 0) AS x`)).toBe('');
    expect(val(`RETURN right('abcdef', 99) AS x`)).toBe('abcdef');
    expect(val(`RETURN right('abcdef', null) AS x`)).toBe(null);
  });

  test('nullif compares by value, not by reference', () => {
    // `a === b` is false for two equal temporals/lists/records (distinct objects),
    // so nullif returned the value where the native `val_eq` returned null.
    expect(val(`RETURN nullif(date('2020-01-01'), date('2020-01-01')) AS x`)).toBe(null);
    expect(val(`RETURN nullif(duration('P1Y2M'), duration('P1Y2M')) AS x`)).toBe(null);
    expect(val(`RETURN nullif([1, 2], [1, 2]) AS x`)).toBe(null);
    expect(val(`RETURN nullif({a: 1}, {a: 1}) AS x`)).toBe(null);
    // Unequal operands still yield the first value.
    expect(val(`RETURN nullif([1, 2], [1, 3]) AS x`)).toEqual([1, 2]);
    expect(val(`RETURN nullif(1, 2) AS x`)).toBe(1);
  });

  test('the conversion functions accept only numbers and strings', () => {
    // They used to convert by stringifying, so a one-element list ("[0]" → "0")
    // or an element (whose `str` is its id) came back as a number.
    expect(val(`RETURN to_integer([0]) AS x`)).toBe(null);
    expect(val(`RETURN to_float([1.5]) AS x`)).toBe(null);
    expect(val(`RETURN to_boolean([true]) AS x`)).toBe(null);
    expect(val(`MATCH (n:Person {name:'marko'}) RETURN to_integer(n) AS x`)).toBe(null);
    expect(val(`MATCH (n:Person {name:'marko'}) RETURN to_boolean(n) AS x`)).toBe(null);
    // The supported conversions are unchanged.
    expect(val(`RETURN to_integer('42') AS x`)).toBe(42);
    expect(val(`RETURN to_integer(3.9) AS x`)).toBe(3);
    expect(val(`RETURN to_float('1.5') AS x`)).toBe(1.5);
    expect(val(`RETURN to_boolean('yes') AS x`)).toBe(true);
  });

  test('a numeric string that overflows to Infinity is not a number', () => {
    // '1e1000' passes the finite-numeric grammar but parses to Infinity. JSON
    // renders both Infinity and null as `null`, so only a null test sees it.
    expect(val(`RETURN (to_float('1e1000') IS NULL) AS x`)).toBe(true);
    expect(val(`RETURN (to_integer('1e1000') IS NULL) AS x`)).toBe(true);
    expect(val(`RETURN (to_float('-1e1000') IS NULL) AS x`)).toBe(true);
    expect(val(`RETURN (to_float('1e300') IS NULL) AS x`)).toBe(false);
  });

  test('percentile requires numeric values, not coerced strings', () => {
    // A non-numeric value is a data exception now (strict typing) — no string is coerced,
    // not even a numeric-looking one, matching the native percentile / sum / avg.
    expect(() => val(`RETURN percentile_cont('0x10', 0.5) AS x`)).toThrow();
    expect(() => val(`RETURN percentile_cont('5', 0.5) AS x`)).toThrow();
    // Numbers still work.
    expect(val(`RETURN percentile_cont(3, 0.5) AS x`)).toBe(3);
    expect(val(`RETURN percentile_disc(3, 0.5) AS x`)).toBe(3);
  });

  test('range is bounded, and terminates past the float-step stall', () => {
    // A GQL list is materialized, so an unbounded range is an OOM kill rather than a
    // query error — and `i += 1` is a NO-OP once i reaches 2^53, so the old
    // comparison-driven loop never terminated for a large enough end.
    expect(val(`RETURN size(range(0, 999999)) AS x`)).toBe(1_000_000);
    expect(() => query(g2, `RETURN size(range(0, 1000000)) AS x`)).toThrow();
    expect(() => query(g2, `RETURN size(range(0, 1e21)) AS x`)).toThrow();
    expect(
      hasErrorCode(
        (() => {
          try {
            query(g2, `RETURN size(range(0, 1e21)) AS x`);
          } catch (e) {
            return e;
          }
        })(),
        ErrorCode.ResourceExhausted,
      ),
    ).toBe(true);
    // A wide step brings a wide span back under the budget.
    expect(val(`RETURN size(range(0, 1000000, 2)) AS x`)).toBe(500_001);
    // The 2^53 stall: 3 elements, and it terminates.
    expect(
      val(`RETURN size(range(to_float('9007199254740992'), to_float('9007199254740994'))) AS x`),
    ).toBe(3);
    // Ordinary semantics are unchanged.
    expect(val(`RETURN range(0, 5) AS x`)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(val(`RETURN range(5, 0, -1) AS x`)).toEqual([5, 4, 3, 2, 1, 0]);
    expect(val(`RETURN range(0, 10, 3) AS x`)).toEqual([0, 3, 6, 9]);
    expect(val(`RETURN range(0, 0) AS x`)).toEqual([0]);
    expect(val(`RETURN range(5, 0) AS x`)).toEqual([]);
    expect(val(`RETURN range(0, 10, 0) AS x`)).toBe(null);
  });

  test('the total order ties every pair in the catch-all rank', () => {
    // Rank 4 holds graph elements, lists, and records. Same-kind pairs compare
    // structurally; a MIXED pair (list vs record) is a tie, so a stable sort keeps
    // input order. The old fallback compared their JS string coercions ("1,2" vs
    // "[object Map]") and invented an order the native engine does not have.
    expect(val(`RETURN list_sort([{a: 1}, [1, 2]]) AS x`)).toEqual(
      val(`RETURN [{a: 1}, [1, 2]] AS x`),
    );
    expect(val(`RETURN list_sort([[1, 2], {a: 1}]) AS x`)).toEqual(
      val(`RETURN [[1, 2], {a: 1}] AS x`),
    );
    // The groups and the within-group orders are unaffected.
    expect(
      val(`RETURN list_sort([[1, 2], date('2020-01-01'), {a: 1}, true, 'z', 3]) AS x`),
    ).toEqual(val(`RETURN [3, 'z', true, date('2020-01-01'), [1, 2], {a: 1}] AS x`));
    expect(val(`RETURN list_sort([[3], [1], [2]]) AS x`)).toEqual(
      val(`RETURN [[1], [2], [3]] AS x`),
    );
    expect(val(`RETURN list_sort([{b: 1}, {a: 1}]) AS x`)).toEqual(
      val(`RETURN [{a: 1}, {b: 1}] AS x`),
    );
  });

  test('stddev over non-numeric values is a data exception, like sum/avg', () => {
    // A non-numeric value is NOT coerced — stddev faults (strict typing), matching the
    // native engine's stddev / sum / avg.
    expect(() => val(`MATCH (p:Person) RETURN stddev_pop(p.name) AS x`)).toThrow();
    expect(() => val(`MATCH (p:Person) RETURN stddev_samp(p.name) AS x`)).toThrow();
  });
});

describe('GQL: clause-level conditions use three-valued truth', () => {
  const g3 = createTestSocialGraph();

  test('a non-boolean WHERE / FILTER coerces the same way IS TRUE does', () => {
    // The clause boundary compared the raw value to `true`, so `WHERE 1` dropped
    // every row — contradicting this same engine's `1 IS TRUE` and `CASE WHEN 1`,
    // and disagreeing with the native engine.
    const all = query(g3, `MATCH (p:Person) RETURN p.name AS x`).length;

    expect(all).toBeGreaterThan(0);
    expect(query(g3, `MATCH (p:Person) WHERE 1 RETURN p.name AS x`)).toHaveLength(all);
    expect(query(g3, `MATCH (p:Person) FILTER 1 RETURN p.name AS x`)).toHaveLength(all);
    expect(query(g3, `MATCH (p:Person) WHERE 'abc' RETURN p.name AS x`)).toHaveLength(all);
    // Falsy and UNKNOWN conditions still drop every row.
    expect(query(g3, `MATCH (p:Person) WHERE 0 RETURN p.name AS x`)).toHaveLength(0);
    expect(query(g3, `MATCH (p:Person) WHERE '' RETURN p.name AS x`)).toHaveLength(0);
    expect(query(g3, `MATCH (p:Person) WHERE null RETURN p.name AS x`)).toHaveLength(0);
    // And an ordinary predicate is unaffected.
    expect(query(g3, `MATCH (p:Person) WHERE p.name = 'marko' RETURN p.name AS x`)).toHaveLength(1);
  });
});

describe('GQL: ORDER BY + LIMIT projects only the emitted rows', () => {
  const g5 = createTestSocialGraph();

  test('a sort key that does not read the output enables the top-k', () => {
    // Only the top-k INPUT bindings are kept, and just those are projected — so a
    // projection that would fault on a row outside the top-k never runs. Mirrors
    // the native `ProjAccum` top-k mode (same five conditions).
    const ages = query(g5, `MATCH (p:Person) RETURN p.age AS a ORDER BY p.age`).map((r) => r.a);

    expect(ages.length).toBeGreaterThan(1);
    expect(query(g5, `MATCH (p:Person) RETURN p.age AS a ORDER BY p.age LIMIT 2`)).toEqual([
      { a: ages[0] },
      { a: ages[1] },
    ]);
    // Descending, SKIP, and a limit past the end all still work.
    expect(query(g5, `MATCH (p:Person) RETURN p.age AS a ORDER BY p.age DESC LIMIT 1`)).toEqual([
      { a: ages.at(-1) },
    ]);
    expect(query(g5, `MATCH (p:Person) RETURN p.age AS a ORDER BY p.age SKIP 1 LIMIT 1`)).toEqual([
      { a: ages[1] },
    ]);
    expect(query(g5, `MATCH (p:Person) RETURN p.age AS a ORDER BY p.age LIMIT 99`)).toHaveLength(
      ages.length,
    );
  });

  test('a sort key that READS the output disables it', () => {
    // The sort key is the projected column, so every row must be projected to
    // sort at all. Same for an alias of an input column, and for DISTINCT (whose
    // dedup key is built from the projected row).
    const byAlias = query(g5, `MATCH (p:Person) RETURN p.age AS a ORDER BY a LIMIT 2`);
    const byInput = query(g5, `MATCH (p:Person) RETURN p.age AS a ORDER BY p.age LIMIT 2`);

    expect(byAlias).toEqual(byInput);
    expect(
      query(g5, `MATCH (p:Person) RETURN DISTINCT p.age AS a ORDER BY p.age LIMIT 2`).length,
    ).toBeLessThanOrEqual(2);
  });
});

describe('GQL: a zero LIMIT returns no rows without projecting', () => {
  const g4 = createTestSocialGraph();

  test('LIMIT 0 never evaluates a faulting projection', () => {
    // The same rule the engine already follows for a non-zero LIMIT ("project
    // exactly the rows you emit"). Whether this faulted used to depend on the
    // presence of ORDER BY.
    expect(query(g4, `MATCH (p:Person) RETURN 1/0 AS x LIMIT 0`)).toHaveLength(0);
    expect(
      query(g4, `MATCH (p:Person) RETURN 1/0 AS x, p.name AS t ORDER BY t LIMIT 0`),
    ).toHaveLength(0);
    expect(query(g4, `MATCH (p:Person) RETURN DISTINCT 1/0 AS x LIMIT 0`)).toHaveLength(0);
    expect(query(g4, `MATCH (p:Person) RETURN sum(1/0) AS x LIMIT 0`)).toHaveLength(0);
    expect(query(g4, `MATCH (p:Person) WITH 1/0 AS x LIMIT 0 RETURN x`)).toHaveLength(0);
  });

  test('a non-zero LIMIT still projects (and still faults)', () => {
    expect(query(g4, `MATCH (p:Person) RETURN p.name AS x ORDER BY x LIMIT 1`)).toHaveLength(1);
    expect(() => query(g4, `MATCH (p:Person) RETURN 1/0 AS x LIMIT 1`)).toThrow();
    // Only a zero LIMIT short-circuits — SKIP past the end still evaluates.
    expect(() => query(g4, `MATCH (p:Person) RETURN 1/0 AS x SKIP 99`)).toThrow();
  });
});

describe('GQL: graph settings are constructor-only', () => {
  test('the range ceiling is set at construction', () => {
    // Settings are host policy, fixed for the graph's life — there is no runtime
    // mutator, so a query can never move its own ceiling.
    const dflt = new Graph();
    const raised = new Graph({ limits: { range: 5_000_000 } });
    const lowered = new Graph({ limits: { range: 10 } });

    expect(dflt.limits.range).toBe(1_000_000);
    expect(() => query(dflt, `RETURN size(range(0, 1000000)) AS x`)).toThrow();
    expect(query(raised, `RETURN size(range(0, 1000000)) AS x`)[0].x).toBe(1_000_001);
    expect(() => query(lowered, `RETURN size(range(0, 20)) AS x`)).toThrow();
    expect(query(lowered, `RETURN size(range(0, 5)) AS x`)[0].x).toBe(6);
  });

  test('the operator-chain ceiling is the same setting under two names', () => {
    const chain = `RETURN ${Array.from({ length: 30 }, (_, i) => i).join(' + ')} AS x`;

    expect(query(new Graph(), chain)[0].x).toBe(435);
    expect(new Graph({ limits: { operatorChain: 5 } }).maxOperatorChain).toBe(5);
    expect(new Graph({ maxOperatorChain: 5 }).limits.operatorChain).toBe(5);
    expect(() => query(new Graph({ limits: { operatorChain: 5 } }), chain)).toThrow();
    expect(() => query(new Graph({ maxOperatorChain: 5 }), chain)).toThrow();
  });

  test('unnamed settings keep their defaults, and a bad ceiling is rejected', () => {
    expect(new Graph({ limits: { trail: 50 } }).config).toEqual({
      limits: { range: 1_000_000, trail: 50, intermediate: 50_000_000, operatorChain: 10_000 },
      clock: null,
    });

    for (const bad of [0, -1, 1.5, Number.NaN]) {
      expect(() => new Graph({ limits: { range: bad } })).toThrow();
    }
  });

  test('the clock is the one runtime-settable member', () => {
    // A host dependency rather than a resource bound: settable at construction OR
    // later, and clearing it with null is part of the contract.
    const clock = (): never => null as never;

    expect(new Graph({ clock }).config.clock).toBe(clock);
    expect(new Graph().setClock(clock).clock).toBe(clock);
    expect(new Graph({ clock }).setClock(null).config.clock).toBe(null);
    // No general runtime mutator.
    expect('configure' in new Graph()).toBe(false);
  });
});
