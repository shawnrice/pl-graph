import { describe, expect, test } from 'bun:test';

import { openBackend } from './engine.js';
import { findSample, loadSample, SAMPLES } from './samples.js';

const backend = await openBackend();

describe('samples', () => {
  test('the registry lists modern and dunder', () => {
    expect(SAMPLES.map((s) => s.name)).toEqual(['modern', 'dunder']);
  });

  test('findSample resolves by name, else undefined', () => {
    expect(findSample('dunder')?.file).toBe('dunder.ndjson');
    expect(findSample('nope')).toBeUndefined();
  });

  test('modern loads with the expected shape', () => {
    const g = loadSample(backend, findSample('modern')!);

    expect(g.vertexCount).toBe(6);
    g.free();
  });

  test('dunder is bitemporal and answers as-of queries', () => {
    const g = loadSample(backend, findSample('dunder')!);

    expect(g.vertexCount).toBe(26);
    expect(g.edgeCount).toBe(44);

    // The regional manager's chair turns over — queryable by date (valid time).
    const managerOn = (asof: string) =>
      (
        g.query(
          `MATCH (p:Person)-[m:MANAGES]->(:Company)
           WHERE m.vf <= date('${asof}') AND m.vt > date('${asof}')
           RETURN p.name AS name`,
        ) as { name: string }[]
      ).map((r) => r.name);

    expect(managerOn('2007-06-01')).toEqual(['Michael Scott']);
    expect(managerOn('2013-04-01')).toEqual(['Dwight Schrute']);

    // Ryan's VP tenure: as recorded in 2008 it ran open-ended; a 2008-11 correction
    // (his firing) closed it — same valid window, different system-time answer.
    const vpEndAsRecorded = (asof: string) =>
      (
        g.query(
          `MATCH (p:Person {name:'Ryan Howard'})-[e:WORKS_AT]->(:Company)
           WHERE e.tf <= date('${asof}') AND e.tt > date('${asof}')
           RETURN e.vt AS vt`,
        ) as { vt: { '@date': string } }[]
      ).map((r) => r.vt['@date']);

    expect(vpEndAsRecorded('2008-06-01')).toEqual(['2099-12-31']);
    expect(vpEndAsRecorded('2009-06-01')).toEqual(['2008-11-01']);
    g.free();
  });
});
