import { describe, expect, test } from 'bun:test';

import { Graph, parseDate } from '@lenke/core';
import { ErrorCode, hasErrorCode } from '@lenke/errors';

import { deserialize, serialize } from './index.js';
import { normalizeValue } from './value.js';

describe('serialization error codes', () => {
  test('an unknown format carries ErrorCode.UnknownFormat', () => {
    let caught: unknown;

    try {
      deserialize('', 'nope' as never, new Graph());
    } catch (e) {
      caught = e;
    }

    expect(hasErrorCode(caught, ErrorCode.UnknownFormat)).toBe(true);
  });

  test('invalid pg-json JSON carries ErrorCode.InvalidJson', () => {
    let caught: unknown;

    try {
      deserialize('{not json', 'pg-json', new Graph());
    } catch (e) {
      caught = e;
    }

    expect(hasErrorCode(caught, ErrorCode.InvalidJson)).toBe(true);
  });

  test('an out-of-model property value carries ErrorCode.InvalidValue', () => {
    let caught: unknown;

    try {
      normalizeValue(new Date());
    } catch (e) {
      caught = e;
    }

    expect(hasErrorCode(caught, ErrorCode.InvalidValue)).toBe(true);
  });

  test('a bigint is rejected (not silently coerced to a lossy number)', () => {
    // The numeric model is float64; a bigint would lose precision above 2^53, so
    // every value boundary rejects it rather than downgrade it. NaN/±Infinity are
    // still coerced to null (JS non-values), unchanged.
    const caught = (fn: () => void): unknown => {
      try {
        fn();
      } catch (e) {
        return e;
      }

      return undefined;
    };

    expect(
      hasErrorCode(
        caught(() => normalizeValue(9007199254740993n)),
        ErrorCode.InvalidValue,
      ),
    ).toBe(true);
    expect(
      hasErrorCode(
        caught(() => normalizeValue([1, [2, 3n]])),
        ErrorCode.InvalidValue,
      ),
    ).toBe(true);
    expect(normalizeValue(Number.NaN)).toBeNull();
    expect(normalizeValue(Infinity)).toBeNull();
  });

  test('a lone UTF-16 surrogate is rejected (match native serde)', () => {
    // A `\uD800` escape survives JSON.parse in JS but the native UTF-8 store
    // can't hold a lone surrogate — its serde JSON decoder rejects it at ingest.
    // The value boundary rejects it too so both engines accept/reject the same
    // documents. Valid pairs (astral chars) still pass.
    const caught = (fn: () => void): unknown => {
      try {
        fn();
      } catch (e) {
        return e;
      }

      return undefined;
    };

    for (const bad of ['\uD800', '\uDC00', 'a\uD800b', 'x\uDFFF']) {
      expect(
        hasErrorCode(
          caught(() => normalizeValue(bad)),
          ErrorCode.InvalidJson,
        ),
      ).toBe(true);
    }

    // a well-formed surrogate pair (😀) and plain text are unaffected
    expect(normalizeValue('😀')).toBe('😀');
    expect(normalizeValue('hello')).toBe('hello');

    // and end-to-end through the ndjson decoder
    const nd = String.raw`{"type":"node","id":"a","labels":["U"],"properties":{"bad":"\ud800"}}`;
    expect(
      hasErrorCode(
        caught(() => deserialize(nd, 'ndjson', new Graph())),
        ErrorCode.InvalidJson,
      ),
    ).toBe(true);
  });

  test('a temporal instance and a TC39 Temporal.Plain* both normalize to a temporal', () => {
    const d = parseDate('2020-01-01');
    expect(normalizeValue(d)).toBe(d);

    // Duck-typed TC39 Temporal.PlainDate (brand + ISO toString), no hard dep.
    const fake = { [Symbol.toStringTag]: 'Temporal.PlainDate', toString: () => '2021-02-03' };
    const normalized = normalizeValue(fake) as { toString(): string };
    expect(normalized.toString()).toBe('2021-02-03');
  });

  test('a native Date reject points at the explicit converter', () => {
    let message = '';

    try {
      normalizeValue(new Date());
    } catch (e) {
      ({ message } = e as Error);
    }

    expect(message).toContain('fromJSDate');
  });

  // Capture whatever a thunk throws (or undefined if it doesn't).
  const caughtFrom = (fn: () => void): unknown => {
    try {
      fn();

      return undefined;
    } catch (e) {
      return e;
    }
  };

  test('invalid ndjson JSON carries ErrorCode.InvalidJson', () => {
    const caught = caughtFrom(() => deserialize('{not json', 'ndjson', new Graph()));
    expect(hasErrorCode(caught, ErrorCode.InvalidJson)).toBe(true);
  });

  test('an ndjson line that is not a node/edge record carries ErrorCode.InvalidShape', () => {
    const caught = caughtFrom(() => deserialize('{"type":"banana"}', 'ndjson', new Graph()));
    expect(hasErrorCode(caught, ErrorCode.InvalidShape)).toBe(true);
  });

  test('a csv edge to a non-existent vertex carries ErrorCode.MissingVertex', () => {
    const csv = 'id,:LABEL\n=== EDGES ===\nid,:START_ID,:END_ID,:TYPE\ne1,x,y,KNOWS';
    const caught = caughtFrom(() => deserialize(csv, 'csv', new Graph()));
    expect(hasErrorCode(caught, ErrorCode.MissingVertex)).toBe(true);
  });

  test('a graphson edge to a missing vertex carries ErrorCode.MissingVertex', () => {
    // Encode a real 2-node/1-edge graph, then drop its vertices so the edge dangles.
    const g = new Graph();
    const a = g.addVertex({ id: 'a', labels: ['T'], properties: {} });
    const b = g.addVertex({ id: 'b', labels: ['T'], properties: {} });
    g.addEdge({ id: 'e0', from: a, to: b, labels: ['KNOWS'], properties: {} });
    const doc = JSON.parse(serialize(g, 'graphson'));
    doc.vertices = [];
    const caught = caughtFrom(() => deserialize(JSON.stringify(doc), 'graphson', new Graph()));
    expect(hasErrorCode(caught, ErrorCode.MissingVertex)).toBe(true);
  });

  test('a deeply-nested array value is a clean error, not a stack overflow', () => {
    let deep: unknown = 1;

    for (let i = 0; i < 2000; i += 1) {
      deep = [deep];
    }

    const caught = caughtFrom(() => normalizeValue(deep));
    expect(hasErrorCode(caught, ErrorCode.InvalidShape)).toBe(true);
  });
});
