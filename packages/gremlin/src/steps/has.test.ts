import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';
import { query as gqlQuery } from '@lenke/gql';

import { run } from '../executor.js';
import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph.js';
import {
  between,
  eq,
  gt,
  gte,
  inside,
  lt,
  lte,
  neq,
  outside,
  regex,
  startsWith,
  within,
  without,
} from '../predicates.js';
import { V, elementMap, has, hasLabel, not, out, outE, values } from '../steps.js';
import { traversal } from '../traversal.js';

const arr = (r: Iterable<unknown>): unknown[] => [...r];

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('has tests', () => {
    test('we can filter using has with a predicate', () => {
      const result = arr(
        run(
          traversal(V(), hasLabel('PERSON'), out(), has('name', within('vadas', 'josh'))),
          tinkerGraph,
        ),
      ) as Array<{ id: string }>;
      expect(result.map((x) => x.id)).toEqual(['2', '4']);
    });

    test('we can chain filters using with predicates', () => {
      const result = arr(
        run(
          traversal(
            V(),
            hasLabel('PERSON'),
            out(),
            has('name', within('vadas', 'josh')),
            outE(),
            hasLabel('CREATED'),
          ),
          tinkerGraph,
        ),
      ) as Array<{ id: string }>;
      expect(result.map((x) => x.id)).toEqual(['10', '11']);
    });

    // Ages: marko=29, vadas=27, josh=32, peter=35.
    test('has(key, inside) — strict open interval', () => {
      // age in (28, 33) → marko (29), josh (32).
      const result = arr(
        run(
          traversal(V(), hasLabel('PERSON'), has('age', inside(28, 33)), values('name')),
          tinkerGraph,
        ),
      );
      expect((result as string[]).sort()).toEqual(['josh', 'marko']);
    });

    test('has(key, outside) — strict complement', () => {
      // age < 29 || > 32 → vadas (27), peter (35).
      const result = arr(
        run(
          traversal(V(), hasLabel('PERSON'), has('age', outside(29, 32)), values('name')),
          tinkerGraph,
        ),
      );
      expect((result as string[]).sort()).toEqual(['peter', 'vadas']);
    });

    test('we can do startsWith filtering', () => {
      // v1 used has('PERSON', 'name', P.startingWith('m')) — three-arg form
      // (label, key, pred). v2 has() is two-arg, so combine hasLabel + has.
      const result = arr(
        run(traversal(V(), hasLabel('PERSON'), has('name', startsWith('m'))), tinkerGraph),
      ) as Array<{ id: string }>;
      expect(result.map((x) => x.id)).toEqual(['1']);
    });

    // Key-only `has(key)`: keep elements that have the property key (any value).
    test('we can filter just by key (key existence)', () => {
      // PERSON vertices carry `age`; SOFTWARE carry `lang` — so has('age') keeps
      // the four people.
      const result = arr(run(traversal(V(), has('age'), values('name')), tinkerGraph));
      expect((result as string[]).sort()).toEqual(['josh', 'marko', 'peter', 'vadas']);
    });

    // (v1's has(null, …) edge case is gone: v2's has() requires a string key, so
    // the degenerate null-key form isn't representable — nothing to test.)

    // doc: g.V().has('age',inside(20,30)).values('age') — 29; 27
    test('has(age, inside) on all vertices', () => {
      const result = arr(
        run(traversal(V(), has('age', inside(20, 30)), values('age')), tinkerGraph),
      );
      expect(result).toEqual([29, 27]);
    });

    // doc: g.V().has('age',outside(20,30)).values('age') — 32; 35
    test('has(age, outside) on all vertices', () => {
      const result = arr(
        run(traversal(V(), has('age', outside(20, 30)), values('age')), tinkerGraph),
      );
      expect(result).toEqual([32, 35]);
    });

    // doc: g.V().has('name',within('josh','marko')).elementMap()
    test('has(name, within) projected via elementMap', () => {
      const result = arr(
        run(traversal(V(), has('name', within('josh', 'marko')), elementMap()), tinkerGraph),
      );
      expect(result).toEqual([
        { id: '1', label: 'PERSON', name: 'marko', age: 29 },
        { id: '4', label: 'PERSON', name: 'josh', age: 32 },
      ]);
    });

    // doc: g.V().has('name',without('josh','marko')).elementMap()
    test('has(name, without) projected via elementMap', () => {
      const result = arr(
        run(traversal(V(), has('name', without('josh', 'marko')), elementMap()), tinkerGraph),
      );
      expect(result).toEqual([
        { id: '2', label: 'PERSON', name: 'vadas', age: 27 },
        { id: '6', label: 'PERSON', name: 'peter', age: 35 },
        { id: '3', label: 'SOFTWARE', name: 'lop', lang: 'java' },
        { id: '5', label: 'SOFTWARE', name: 'ripple', lang: 'java' },
      ]);
    });

    // doc: g.V().has('name', not(within('josh','marko'))).elementMap()
    // Our `not(...)` step takes a sub-traversal, not a predicate, so we
    // express the equivalent via not(has(...)) wrapping.
    test('not(has(name, within)) is equivalent to has(name, without)', () => {
      const result = arr(
        run(traversal(V(), not(has('name', within('josh', 'marko'))), elementMap()), tinkerGraph),
      );
      expect(result).toEqual([
        { id: '2', label: 'PERSON', name: 'vadas', age: 27 },
        { id: '6', label: 'PERSON', name: 'peter', age: 35 },
        { id: '3', label: 'SOFTWARE', name: 'lop', lang: 'java' },
        { id: '5', label: 'SOFTWARE', name: 'ripple', lang: 'java' },
      ]);
    });

    // doc: g.V().has('person', 'name', regex('r')).values('name') — marko; peter
    // Our has() is two-arg; combine hasLabel + has(name, regex).
    test('hasLabel + has(name, regex) finds names containing the pattern', () => {
      const result = arr(
        run(
          traversal(V(), hasLabel('PERSON'), has('name', regex('r')), values('name')),
          tinkerGraph,
        ),
      );
      expect((result as string[]).sort()).toEqual(['marko', 'peter']);
    });

    // doc: g.V().hasLabel('person').out().has('name',within('vadas','josh')).outE().hasLabel('created')
    test('chained hasLabel + has + outE + hasLabel selects edges 10 and 11', () => {
      const result = arr(
        run(
          traversal(
            V(),
            hasLabel('PERSON'),
            out(),
            has('name', within('vadas', 'josh')),
            outE(),
            hasLabel('CREATED'),
          ),
          tinkerGraph,
        ),
      ) as Array<{ id: string }>;
      expect(result.map((e) => e.id)).toEqual(['10', '11']);
    });

    // doc: g.V().has('name', 'marko') — shorthand for has('name', eq('marko'))
    test('has(key, value) is shorthand for has(key, eq(value))', () => {
      const result = arr(run(traversal(V(), has('name', 'marko'), values('name')), tinkerGraph));
      expect(result).toEqual(['marko']);
    });

    // doc: g.V().has('person', 'name', 'marko') — 3-arg label+key+value
    test('has(label, key, value) filters by label AND property', () => {
      const result = arr(
        run(traversal(V(), has('PERSON', 'name', 'marko'), values('name')), tinkerGraph),
      );
      expect(result).toEqual(['marko']);
    });

    // doc: g.V().has('person', 'age', gt(30)) — 3-arg with predicate
    test('has(label, key, predicate) filters by label AND property predicate', () => {
      const result = arr(
        run(traversal(V(), has('PERSON', 'age', gt(30)), values('name')), tinkerGraph),
      );
      // marko=29, vadas=27, josh=32, peter=35 → josh, peter
      expect(result).toEqual(['josh', 'peter']);

      // Suppress unused-import noise — eq/startsWith are exercised above.
      void eq;
      void startsWith;
    });

    // Regression: temporal predicates. The graph stores temporals as class
    // instances, but callers pass the tagged wire form (`{"@date": "…"}`) — the
    // same shape `toJSON` emits and GQL params take. `eq` compared the two with
    // `===` and SILENTLY matched nothing, while the ordering predicates never
    // reached the temporal branch of `compareValues` (`isTemporal` recognizes only
    // instances) and threw E_INVALID_VALUE. A silent empty result is the worse of
    // the two: it ships.
    describe('temporal predicates', () => {
      const D = (iso: string) => ({ '@date': iso });
      // Built through GQL so `vf` is stored as a real temporal instance, which is
      // what a DATE literal produces.
      const dated = (): Graph => {
        const g = new Graph();

        gqlQuery(
          g,
          "INSERT (:V {vf: DATE '2020-01-01', n: 1}), (:V {vf: DATE '2022-06-15', n: 2})",
        );

        return g;
      };
      const ns = (pred: unknown): unknown[] =>
        arr(run(traversal(V(), has('vf', pred as never), values('n')), dated()));

      test('equality matches by value, not reference', () => {
        expect(ns(D('2020-01-01'))).toEqual([1]); // bare value
        expect(ns(eq(D('2020-01-01')))).toEqual([1]);
        expect(ns(neq(D('2020-01-01')))).toEqual([2]);
        expect(ns(within(D('2022-06-15')))).toEqual([2]);
        expect(ns(without(D('2022-06-15')))).toEqual([1]);
      });

      test('ordering predicates compare chronologically', () => {
        expect(ns(lt(D('2021-01-01')))).toEqual([1]);
        expect(ns(lte(D('2020-01-01')))).toEqual([1]);
        expect(ns(gt(D('2021-01-01')))).toEqual([2]);
        expect(ns(gte(D('2022-06-15')))).toEqual([2]);
        expect(ns(between(D('2019-01-01'), D('2021-01-01')))).toEqual([1]);
        expect(ns(inside(D('2019-01-01'), D('2021-01-01')))).toEqual([1]);
      });
    });
  });
});
