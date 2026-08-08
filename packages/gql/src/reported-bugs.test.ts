import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';

import { query } from './index.js';

/**
 * Reproduction tests for four reported bugs in the GQL executor.
 * These tests DOCUMENT current behavior: a failing expectation means the bug
 * is CONFIRMED (the code does the wrong thing), a passing one means DISPROVED.
 * Do not "fix" these tests — they encode the ISO-correct expectation and are
 * expected to fail while the bug stands.
 */

const empty = new Graph();

// ---------------------------------------------------------------------------
// Bug 1: set operations keyed by name=value, not positionally.
// ISO UNION/EXCEPT/INTERSECT are positional; the result adopts the LEFT part's
// column names. Rows are compared by column *position*, not by (name,value).
// ---------------------------------------------------------------------------
describe('Bug 1: set ops should be positional, not keyed by column name', () => {
  test('EXCEPT with differently-aliased single column should be empty', () => {
    // Positionally, {a:1} and {b:1} are the same one-column row, so EXCEPT
    // must remove it → []. The keyed impl keys them "a=1" vs "b=1" (unequal)
    // and wrongly keeps the left row.
    const rows = query(empty, `RETURN 1 AS a EXCEPT RETURN 1 AS b`);
    expect(rows).toEqual([]);
  });

  test('UNION with differently-aliased single column should dedup to one row', () => {
    // Positionally the same row; ISO union adopts left names → [{a:1}].
    const rows = query(empty, `RETURN 1 AS a UNION RETURN 1 AS b`);
    expect(rows).toEqual([{ a: 1 }]);
  });
});

// ---------------------------------------------------------------------------
// Bug 2: `=`/`<>` (and property-map equality) use JS reference equality for
// lists/objects. `[1,2] = [1,2]` should be TRUE; a `{tags:[1,2]}` constraint
// should match a stored [1,2] list property.
// ---------------------------------------------------------------------------
describe('Bug 2: list/value equality should be by value, not reference', () => {
  test('[1,2] = [1,2] should be true', () => {
    const rows = query(empty, `RETURN ([1, 2] = [1, 2]) AS eq`);
    expect(rows).toEqual([{ eq: true }]);
  });

  test('property-map constraint should match a stored list property', () => {
    const g = new Graph();
    g.disableEvents();
    g.addVertex({ id: 'n1', labels: ['Item'], properties: { name: 'x', tags: [1, 2] } });
    g.enableEvents();

    const rows = query(g, `MATCH (n:Item {tags: [1, 2]}) RETURN n.name AS name`);
    expect(rows).toEqual([{ name: 'x' }]);
  });
});

// Bugs 3 (fixed-length trail uniqueness) and 4 (strict sum/avg) from the
// original bridge report are intentionally NOT here: the Rust engine — the
// byte-identity oracle — counts WALKS for separate segments (a shared edge is
// reused, matching the quantified-vs-separate distinction) and returns
// null-for-non-numeric / 0-for-empty sum. The TS engine already agrees with it
// on both. The bridge's ISO expectations predate that convergence, so asserting
// them now would REINTRODUCE a divergence rather than fix a bug.
