// by('key') on an ABSENT property FILTERS the traverser (order/group/groupCount/dedup/
// select) or OMITS the key (project), matching TinkerPop — while a stored PRESENT null is
// a value and is kept. This locks that behavior byte-identically across the TS engine and
// the native (FFI) engine, both built from the SAME NDJSON. The distinction (absent vs
// present-null) is exactly what the fuzzers do not generate, so it is checked directly.
import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';
import {
  V,
  as_,
  dedupe,
  group,
  groupCount,
  order,
  planToGremlin,
  project,
  select,
  toArray,
  traversal,
  values,
  type Plan,
} from '@lenke/gremlin';
import { deserialize } from '@lenke/serialization';

import { nativeBackend } from './conformance-harness.js';
import { graphFromNdjson } from './graph.js';

// marko/vadas/josh/peter have `age`; lop/ripple (software) do NOT (absent); `nuller` has a
// stored PRESENT null age.
const NDJSON = [
  '{"type":"node","id":"1","labels":["PERSON"],"properties":{"name":"marko","age":29}}',
  '{"type":"node","id":"2","labels":["PERSON"],"properties":{"name":"vadas","age":27}}',
  '{"type":"node","id":"3","labels":["SOFTWARE"],"properties":{"name":"lop","lang":"java"}}',
  '{"type":"node","id":"4","labels":["PERSON"],"properties":{"name":"josh","age":32}}',
  '{"type":"node","id":"5","labels":["SOFTWARE"],"properties":{"name":"ripple","lang":"java"}}',
  '{"type":"node","id":"6","labels":["PERSON"],"properties":{"name":"peter","age":35}}',
  '{"type":"node","id":"7","labels":["PERSON"],"properties":{"name":"nuller","age":null}}',
].join('\n');

const norm = (v: unknown): unknown => {
  if (v instanceof Map) {
    return Object.fromEntries([...v].map(([k, x]) => [String(k), norm(x)]));
  }

  if (Array.isArray(v)) {
    return v.map(norm);
  }

  return v;
};

const deepSort = (v: unknown): unknown => {
  const n = norm(v);

  if (Array.isArray(n)) {
    return [...n].map(deepSort).sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
  }

  if (n && typeof n === 'object') {
    return Object.fromEntries(
      Object.entries(n as Record<string, unknown>)
        .map(([k, val]) => [k, deepSort(val)] as const)
        .sort((a, b) => a[0].localeCompare(b[0])),
    );
  }

  return n;
};

describe('by(absent) filter semantics — TS ≡ native', () => {
  const cases: Record<string, Plan> = {
    // Absent `age` (software) is filtered; present-null (nuller) is kept.
    order: traversal(V(), order().by('age'), values('name')),
    group: traversal(V(), group().by('age').by('name')),
    groupCount: traversal(V(), groupCount().by('age')),
    dedup: traversal(V(), dedupe().by('age'), values('name')),
    select: traversal(V(), as_('a'), select('a').by('age')),
    project: traversal(V(), project('a').by('age')),
    // Control: a by() that is always present must be unchanged (all 7 kept).
    orderName: traversal(V(), order().by('name'), values('name')),
  };

  test('every keying step agrees byte-for-byte across engines', () => {
    const nat = graphFromNdjson(nativeBackend(), NDJSON);
    const ts = deserialize(NDJSON, 'ndjson', new Graph());

    for (const [name, plan] of Object.entries(cases)) {
      const native = deepSort(nat.gremlin(planToGremlin(plan)));
      const tsOut = deepSort([...toArray(plan, ts)]);
      expect([name, native]).toEqual([name, tsOut]);
    }
  });

  test('present-null is KEPT (a value), absent is FILTERED — not conflated', () => {
    const nat = graphFromNdjson(nativeBackend(), NDJSON);
    const ts = deserialize(NDJSON, 'ndjson', new Graph());

    // groupCount().by('age'): 4 present ages (count 1 each) + ONE null bucket (nuller);
    // the two software vertices (absent age) are filtered out entirely.
    const plan = traversal(V(), groupCount().by('age'));
    const native = norm(nat.gremlin(planToGremlin(plan))) as Record<string, unknown>[];
    const tsOut = norm([...toArray(plan, ts)]) as Record<string, unknown>[];
    expect(native).toEqual(tsOut);

    const [map] = native;
    expect(map.null).toBe(1); // present-null kept as its own bucket
    expect(Object.keys(map).sort()).toEqual(['27', '29', '32', '35', 'null']);
  });
});
