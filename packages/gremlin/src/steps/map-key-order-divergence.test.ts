// Item 3 (TS side): the @lenke/gremlin executor emits map-producing steps
// (project/valueMap) as plain JS objects in INSERTION order, so JSON.stringify
// preserves the order the keys were written. The native Rust engine
// (crates/lenke-engine/src/exec/json_out.rs) sorts map keys LEXICOGRAPHICALLY. The two
// engines therefore serialize the same logical map to different bytes — a real
// TS<->Rust divergence that matters because the sync/live-query layer diffs
// cells by JSON.stringify byte-equality.
//
// Companion native coverage lives in the engine's gremlin tests
// (crates/lenke-engine/tests/gremlin_ported.rs): the native serializer sorts map
// keys lexicographically — {"a":...,"b":...}, and {"10":2,"9":1} for numeric keys.
//
// This file pins the TS behavior that contradicts them. CONFIRMED divergence.
import { describe, expect, test } from 'bun:test';

import { run } from '../executor.js';
import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph.js';
import { eq } from '../predicates.js';
import { V, has, project } from '../steps.js';
import { traversal } from '../traversal.js';

const arr = (r: Iterable<unknown>): unknown[] => [...r];

describe('map-key order: TS emits insertion order (diverges from native sort)', () => {
  const g = createTestTinkerGraph();

  // project('b','a') — keys written b-then-a. TS keeps insertion order; the
  // native engine sorts to a-then-b. Demonstrates the plain-string divergence.
  test('project keys keep insertion order (b before a) under JSON.stringify', () => {
    const result = arr(
      run(traversal(V(), has('name', eq('marko')), project(['b', 'a'], ['name', 'name'])), g),
    );
    const row = result[0] as Record<string, unknown>;

    // TS insertion order: b, a.
    expect(Object.keys(row)).toEqual(['b', 'a']);
    expect(JSON.stringify(row)).toBe('{"b":"marko","a":"marko"}');

    // Native results_to_json would emit the lexicographically-sorted form
    // '{"a":"marko","b":"marko"}' — the exact opposite key order. Locking the
    // contradiction here so a regression on either side is caught.
    expect(JSON.stringify(row)).not.toBe('{"a":"marko","b":"marko"}');
  });

  // Numeric-like keys make the divergence WORSE and non-obvious: JS objects
  // iterate integer-index keys in NUMERIC ascending order (9 before 10),
  // regardless of insertion order; the native engine sorts the stringified
  // keys LEXICOGRAPHICALLY ("10" before "9"). So the two engines land on
  // opposite orders for the very same map.
  test('numeric-like keys: JS orders 9 before 10; native sorts "10" before "9"', () => {
    const result = arr(
      run(traversal(V(), has('name', eq('marko')), project(['10', '9'], ['name', 'name'])), g),
    );
    const row = result[0] as Record<string, unknown>;

    // Even though '10' was inserted first, JS reorders integer-like keys
    // numerically: 9 then 10.
    expect(Object.keys(row)).toEqual(['9', '10']);
    expect(JSON.stringify(row)).toBe('{"9":"marko","10":"marko"}');

    // Native (see item3_native_numeric_keys_lexicographic_reversal) yields the
    // reverse: '{"10":...,"9":...}'.
    expect(JSON.stringify(row)).not.toBe('{"10":"marko","9":"marko"}');
  });
});
