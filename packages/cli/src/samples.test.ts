import { describe, expect, test } from 'bun:test';

import { openBackend } from './engine.js';
import { findSample, loadSample, SAMPLES } from './samples.js';

const backend = await openBackend();

describe('samples', () => {
  test('the registry lists the bundled samples', () => {
    expect(SAMPLES.map((s) => s.name)).toEqual(['modern', 'dunder', 'hillvalley']);
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

  test('hillvalley: one 1985 fact, four rewritten records (bitemporal)', () => {
    const g = loadSample(backend, findSample('hillvalley')!);

    // Biff's status in 1985 (fixed valid time) reads differently depending on WHICH
    // version of history — transaction time — you ask as of.
    const biffIn1985AsOf = (rec: string) =>
      (
        g.query(
          `MATCH (b:Person {name:'Biff Tannen'})-[s:STATUS]->()
           WHERE s.vf <= date('1985-10-26') AND s.vt > date('1985-10-26')
             AND s.tf <= date('${rec}') AND s.tt > date('${rec}')
           RETURN s.timeline AS timeline`,
        ) as { timeline: string }[]
      ).map((r) => r.timeline);

    expect(biffIn1985AsOf('1985-10-25')).toEqual(['original']); // before any time travel
    expect(biffIn1985AsOf('1990-01-01')).toEqual(['restored']); // after BTTF
    expect(biffIn1985AsOf('2015-11-01')).toEqual(['biff-hell']); // after the almanac (BTTF II)
    expect(biffIn1985AsOf('2020-01-01')).toEqual(['restored-again']); // after it's burned

    // A record that only ever existed in the Biff-hell version: George murdered.
    const georgeFateAsOf = (rec: string) =>
      (
        g.query(
          `MATCH (george:Person {name:'George McFly'})-[f:FATE]->()
           WHERE f.tf <= date('${rec}') AND f.tt > date('${rec}')
           RETURN f.value AS fate`,
        ) as { fate: string }[]
      ).map((r) => r.fate);

    expect(georgeFateAsOf('2015-11-01')).toEqual(['Murdered in 1973']);
    expect(georgeFateAsOf('2020-01-01')).toEqual([]); // erased when history was restored
    g.free();
  });
});
