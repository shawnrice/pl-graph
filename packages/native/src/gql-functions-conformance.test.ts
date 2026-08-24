// Table-driven GQL scalar/function differential: every query string here is run
// on BOTH the TS engine (@lenke/gql) and the Rust core (over bun:ffi) against
// identical data, and their `JSON.stringify`d results are asserted
// byte-identical. This is the guardrail for cross-engine function parity — a
// divergence in any scalar function, operator, or predicate shows up as a red
// diff. Add a case here whenever you touch a function's semantics.
//
// Run: bun test packages/native/src/gql-functions-conformance.test.ts
import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';
import { query as tsQuery } from '@lenke/gql';
import { deserialize as tsDeserialize } from '@lenke/serialization';

import { nativeBackend, NATIVE_LIB, nativeReady } from './conformance-harness.js';
import { graphFromNdjson } from './graph.js';

const hasLib = nativeReady;

if (!hasLib) {
  console.warn(`[gql-functions] skipping: ${NATIVE_LIB} not found — run \`bun run build:rust\`.`);
}

const suite = hasLib ? describe : describe.skip;

// A single-node graph is enough for scalar-function evaluation: every case
// projects a computed value, not graph structure. `n.s`/`n.num`/`n.xs` give a
// string, a number, and a list to feed the functions.
const NDJSON = [
  '{"type":"node","id":"1","labels":["T"],"properties":{"s":"Hello World","num":-3.7,"xs":[3,1,2]}}',
].join('\n');

suite('GQL function differential (TS vs native)', () => {
  const backend = nativeBackend();
  const nativeGraph = graphFromNdjson(backend, NDJSON);
  const tsGraph = tsDeserialize(NDJSON, 'ndjson', new Graph());

  // Each case is a RETURN expression; both engines evaluate it over the single
  // node `n`. The test asserts the two serializations are byte-identical.
  const CASES: string[] = [
    // Baseline (pre-existing functions) — proves the harness itself is sound.
    `upper(n.s)`,
    `lower(n.s)`,
    `abs(n.num)`,
    `size(n.xs)`,
    `char_length(n.s)`,
    // Slice 1 — substring is 1-based (SQL / ISO GQL).
    `substring(n.s, 1, 5)`,
    `substring(n.s, 7)`,
    `substring(n.s, 0, 4)`,
    `substring(n.s, 100)`,
    `substring(n.s, -3, 6)`,
    // Slice 2 — split/reverse operate on UTF-16 code units; astral input is
    // lossy (U+FFFD) but byte-identical across engines.
    `split(n.s, '')`,
    `split(n.s, 'o')`,
    `reverse(n.s)`,
    `split('a😀b', '')`,
    `reverse('a😀b')`,
    `reverse('café')`,
    `split('', '')`,
    `reverse(n.xs)`,
    // Slice 3 — round/sign/pi/e.
    `round(n.num)`,
    `round(n.num, 1)`,
    `round(3.14159, 2)`,
    `round(2.5)`,
    `round(-2.5)`,
    `round(2.675, 2)`,
    `round(1234.5678, -2)`,
    `sign(n.num)`,
    `sign(0)`,
    `sign(5)`,
    `sign(-0.0)`,
    `pi()`,
    `e()`,
    `round(pi(), 4)`,
    // Slice 4 — string bool predicates + conversions.
    `contains(n.s, 'World')`,
    `contains(n.s, 'xyz')`,
    `contains(n.s, '')`,
    `starts_with(n.s, 'Hello')`,
    `starts_with(n.s, 'World')`,
    `ends_with(n.s, 'World')`,
    `contains(null, 'a')`,
    `to_boolean('yes')`,
    `to_boolean('FALSE')`,
    `to_boolean('maybe')`,
    `to_boolean(1)`,
    `to_boolean(0)`,
    `to_boolean(n.num)`,
    `to_boolean(null)`,
    `to_list(n.s)`,
    `to_list(42)`,
    `to_list(n.xs)`,
    `to_list('a😀b')`,
    `byte_length(n.s)`,
    `byte_length('café')`,
    `octet_length('😀')`,
    `byte_length(null)`,
    // Slice 5 — tail / append / range.
    `tail(n.xs)`,
    `tail([1])`,
    `tail([])`,
    `append(n.xs, 9)`,
    `append(n.xs, null)`,
    `append([], 'x')`,
    `range(1, 5)`,
    `range(1, 10, 2)`,
    `range(10, 1, -3)`,
    `range(5, 5)`,
    `range(5, 1)`,
    `range(1, 5, 0)`,
    `size(range(0, 100))`,
    // Slice 6 — || concatenation: lists concat, strings unchanged.
    `[1, 2] || [3, 4]`,
    `n.xs || [9, 8]`,
    `[1] || [] || [2]`,
    `'a' || 'b'`,
    `n.s || '!'`,
    `[1, 2] || null`,
    `null || [1]`,
    `range(1, 3) || range(4, 6)`,
    // Slice 7 — CAST(value AS type) → conversion functions.
    `CAST('42' AS INTEGER)`,
    `CAST(3.7 AS INT)`,
    `CAST(n.num AS INTEGER)`,
    `CAST('3.5' AS FLOAT)`,
    `CAST(42 AS STRING)`,
    `CAST(n.num AS STRING)`,
    `CAST('yes' AS BOOL)`,
    `CAST(1 AS BOOLEAN)`,
    `CAST('ab' AS LIST)`,
    `CAST(n.s AS TEXT)`,
    `CAST(CAST('42' AS INT) AS STRING)`,
    `CAST('nope' AS INT)`,
    // Slice 7b — temporal CAST targets desugar to the temporal constructor
    // functions (date/datetime/local_time/zoned_*/duration). Single-word and the
    // two-word `LOCAL DATETIME` / `LOCAL TIME` / `ZONED TIME` / `ZONED DATETIME`
    // spellings; `TIMESTAMP` is a DATETIME alias.
    `CAST('2020-01-01' AS DATE)`,
    `CAST('2020-06-15T08:30:00' AS DATETIME)`,
    `CAST('2020-06-15T08:30:00' AS TIMESTAMP)`,
    `CAST('2020-06-15T08:30:00' AS LOCAL DATETIME)`,
    `CAST('08:30:00' AS LOCAL TIME)`,
    `CAST('08:30:00+02:00' AS ZONED TIME)`,
    `CAST('2020-06-15T08:30:00+02:00' AS ZONED DATETIME)`,
    `CAST('P1Y2M3DT4H' AS DURATION)`,
    // A bare date-only string coerces to midnight for a datetime CAST (bug 3).
    `CAST('2020-06-15' AS DATETIME)`,
    `CAST('not-a-date' AS DATE)`,
    // Slice 7c — datetime()/local_datetime() coerce a bare date-only string to
    // midnight, consistent with date() (bug 3). date() unchanged; a datetime with
    // a real time part is untouched.
    `datetime('2020-06-15')`,
    `local_datetime('2020-06-15')`,
    `datetime('2020-06-15T08:30:00')`,
    `date('2020-06-15')`,
    `datetime('nope')`,
    // Slice 8 — infix CONTAINS / STARTS WITH / ENDS WITH predicates.
    `n.s CONTAINS 'World'`,
    `n.s CONTAINS 'xyz'`,
    `n.s STARTS WITH 'Hello'`,
    `n.s STARTS WITH 'World'`,
    `n.s ENDS WITH 'World'`,
    `n.s ENDS WITH 'Hello'`,
    `NOT (n.s CONTAINS 'z')`,
    `n.s CONTAINS 'o' AND n.s STARTS WITH 'H'`,
    `n.s CONTAINS 'l' OR n.s STARTS WITH 'z'`,
    // Slice 10 — set-style list functions (dedup first-occurrence; sort reuses
    // the ORDER BY total order). list_contains returns numeric 1/0 per ISO.
    `list_union([1, 2, 2, 3], [3, 4, 5])`,
    `intersection([1, 2, 3, 3], [3, 3, 4, 5])`,
    `difference([1, 2, 2, 3], [3, 4, 5])`,
    `list_union(n.xs, [1, 9])`,
    `intersection(n.xs, [2, 9])`,
    `difference(n.xs, [1])`,
    `list_contains([1, 2, 3], 2)`,
    `list_contains([1, 2, 3], 9)`,
    `list_contains(n.xs, 3)`,
    `list_contains(['a', 'b'], 'b')`,
    `list_sort([3, 1, 4, 1, 5])`,
    `list_sort(n.xs)`,
    `list_sort([3, 1, 2], 'desc')`,
    `list_sort(['b', 'a', 'c'])`,
    `list_sort([3, 1, null, 2])`,
    `list_sort([3, 1, null, 2], 'asc', 'first')`,
    `list_sort([3, 1, null, 2], 'desc', 'last')`,
    // Mixed-type sorts: both engines now share a total order across type groups
    // (number < string < boolean < other; nulls last) — see cmp_total / typeRank.
    `list_sort([2, 'a', 1, 'b'])`,
    `list_sort([true, 1, 'x', false])`,
    `list_sort([2, 'a', 1, null])`,
    `list_sort([2, 'a', 1], 'desc')`,
    `list_sort(['banana', 'apple', 'cherry'])`,
    `list_union([1], 2)`,
    // Slice 3 — regressions the differential FUZZER found. The fuzzer is
    // randomized, so these pin the specific cases as permanent table rows.
    // Value stringification: `String(v)` disagreed for every non-primitive.
    `to_string({b: 2, a: 1})`,
    `CAST({a: 1} AS STRING)`,
    `upper({a: 'q'})`,
    `char_length({a: 1})`,
    `({a: 1} || 'x')`,
    `to_string([{a: 1}])`,
    `to_string({a: date('2020-01-01')})`,
    `to_string([1, null, 3])`,
    // right(): a fractional length truncates, a NaN length is empty.
    `right(n.s, 2.9)`,
    `right(n.s, 'nan')`,
    `right(n.s, 'inf')`,
    `right(n.s, 3)`,
    `right(n.s, 0)`,
    // nullif() compares by VALUE — equal temporals/lists/records are distinct JS
    // objects, so a `===` test disagreed with the native `val_eq`.
    `nullif(date('2020-01-01'), date('2020-01-01'))`,
    `nullif(duration('P1Y2M'), duration('P1Y2M'))`,
    `nullif([1, 2], [1, 2])`,
    `nullif({a: 1}, {a: 1})`,
    `nullif([1, 2], [1, 3])`,
    // The conversion functions take numbers and strings only — they must not
    // convert by stringifying (a one-element list, or an element's id).
    `to_integer([0])`,
    `to_float([1.5])`,
    `to_boolean([true])`,
    // A numeric string that overflows to Infinity is not a number. JSON renders
    // Infinity and null identically, so the IS NULL form is what shows it.
    `(to_float('1e1000') IS NULL)`,
    `(to_integer('1e1000') IS NULL)`,
    `(sqrt('1e1000') IS NULL)`,
    `(to_float('1e300') IS NULL)`,
    // percentile coerces with the engine rule, not raw Number() (hex).
    `percentile_cont('0x10', 0.5)`,
    `percentile_disc('0x10', 0.5)`,
    `percentile_cont('5', 0.5)`,
    // degrees/radians: multiply-then-divide, not the pre-rounded constant that
    // Rust's `to_degrees`/`to_radians` use.
    `degrees(1e100)`,
    `degrees(123456.789)`,
    `degrees(3.14)`,
    `radians(3)`,
    `radians('1e3')`,
    // stddev over non-numeric values is NaN (→ null), like avg — not a real 0.
    `stddev_pop(n.s)`,
    `stddev_samp(n.s)`,
    // The total order ties a list against a record (rank 4's catch-all) instead
    // of ordering them by a JS string coercion.
    `list_sort([{a: 1}, [1, 2]])`,
    `list_sort([[1, 2], {a: 1}])`,
    `list_sort([[1, 2], date('2020-01-01'), {a: 1}, true, 'z', 3])`,
    `list_sort([[3], [1], [2]])`,
    `list_sort([{b: 1}, {a: 1}])`,
    // range() is bounded (a materialized list) and its loop is count-driven, so
    // the float-step stall at 2^53 terminates instead of spinning.
    `size(range(0, 999999))`,
    `size(range(0, 1000000, 2))`,
    `size(range(to_float('9007199254740992'), to_float('9007199254740994')))`,
    `range(0, 5)`,
    `range(5, 0, -1)`,
    `range(0, 10, 3)`,
    `range(0, 0)`,
    `range(5, 0)`,
    `range(0, 10, 0)`,
  ];

  // A function either RETURNS a value (compare the rendered JSON byte-for-byte) or FAULTS
  // (compare the error `code`). Many strict-typing cases fault in both
  // engines — `upper({map})`, `radians('1e3')`, `stddev(string)` — so the outcome must
  // capture the throw, not let it bubble and fail the test structurally.
  const evalOutcome = (run: () => unknown): { json: string } | { code: unknown } => {
    try {
      return { json: JSON.stringify(run()) };
    } catch (e) {
      return { code: (e as { code?: unknown }).code };
    }
  };

  for (const expr of CASES) {
    test(`RETURN ${expr}`, () => {
      const q = `MATCH (n:T) RETURN ${expr} AS v`;
      expect(evalOutcome(() => nativeGraph.query(q))).toEqual(
        evalOutcome(() => tsQuery(tsGraph, q)),
      );
    });
  }

  // Both engines resolve function names EAGERLY (TS at compile, native off the
  // plan's `unknown_fns` before the first row), so an unknown function faults
  // identically whether the result set is non-empty, EMPTY, or the call sits in a
  // dead branch. A lazy per-row fault would silently return `[]` over zero rows.
  const outcome = (run: () => unknown): { ok: true } | { code: unknown } => {
    try {
      run();

      return { ok: true };
    } catch (e) {
      return { code: (e as { code?: unknown }).code };
    }
  };

  const UNKNOWN_FN_CASES: string[] = [
    `MATCH (n:T) RETURN nope_fn(n) AS v`, // one row
    `MATCH (n:Missing) RETURN nope_fn(n) AS v`, // ZERO rows — the empty-input bug
    `RETURN CASE WHEN false THEN bogus_fn(1) ELSE 1 END AS v`, // dead branch
  ];

  for (const q of UNKNOWN_FN_CASES) {
    test(`unknown fn faults identically: ${q}`, () => {
      const ts = outcome(() => tsQuery(tsGraph, q));
      const native = outcome(() => nativeGraph.query(q));
      expect(native).toEqual(ts);
      expect(ts).toEqual({ code: 'E_UNKNOWN_FUNCTION' });
    });
  }
});
