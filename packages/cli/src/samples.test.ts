import { describe, expect, test } from 'bun:test';

import { openBackend } from './engine.js';
import { findSample, loadSample, SAMPLES } from './samples.js';

const backend = await openBackend();

describe('samples', () => {
  test('the registry lists modern and employment', () => {
    expect(SAMPLES.map((s) => s.name)).toEqual(['modern', 'employment']);
  });

  test('findSample resolves by name, else undefined', () => {
    expect(findSample('employment')?.file).toBe('employment.ndjson');
    expect(findSample('nope')).toBeUndefined();
  });

  test('modern loads with the expected shape', () => {
    const g = loadSample(backend, findSample('modern')!);

    expect(g.vertexCount).toBe(6);
    g.free();
  });

  test('employment is bitemporal and answers as-of queries', () => {
    const g = loadSample(backend, findSample('employment')!);

    expect(g.vertexCount).toBe(5);
    expect(g.edgeCount).toBe(7);

    // Bob's role as *recorded* on two dates differs (a system-time correction).
    const roleAsOf = (asof: string) =>
      (
        g.query(
          `MATCH (p:Person {name:'Bob'})-[e:WORKS_AT]->(c)
           WHERE e.tf <= date('${asof}') AND e.tt > date('${asof}')
           RETURN e.role AS role`,
        ) as { role: string }[]
      ).map((r) => r.role);

    expect(roleAsOf('2019-12-01')).toEqual(['Engineer']);
    expect(roleAsOf('2021-01-01')).toEqual(['Manager']);
    g.free();
  });
});
