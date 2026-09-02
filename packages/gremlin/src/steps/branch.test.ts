import { describe, expect, test } from 'bun:test';

import { run } from '../executor.js';
import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph.js';
import { V, branch, constant, hasLabel, label, values } from '../steps.js';
import { traversal } from '../traversal.js';

const arr = (r: Iterable<unknown>): unknown[] => [...r];

describe('branch tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  // Route by label: persons -> name, software -> 'a software'.
  test('branch routes per traverser by test result', () => {
    const r = arr(
      run(
        traversal(
          V(),
          branch(label())
            .option('PERSON', values('name'))
            .option('SOFTWARE', constant('a software')),
        ),
        tinkerGraph,
      ),
    );
    expect(r.sort()).toEqual(['a software', 'a software', 'josh', 'marko', 'peter', 'vadas']);
  });

  // .none(plan) provides a default branch.
  test('branch falls back to .none when no option matches', () => {
    const r = arr(
      run(
        traversal(
          V(),
          hasLabel('PERSON'),
          branch(values('age')).option(29, constant('marko')).none(constant('other')),
        ),
        tinkerGraph,
      ),
    );
    // marko=29 -> 'marko'; vadas/josh/peter -> 'other'.
    expect(r.sort()).toEqual(['marko', 'other', 'other', 'other']);
  });

  // doc: g.V().branch(values('name')).option('marko', values('age')).option(none, values('name'))
  //      — 29; vadas; lop; josh; ripple; peter
  test('branch by values("name") with .none default', () => {
    const r = arr(
      run(
        traversal(V(), branch(values('name')).option('marko', values('age')).none(values('name'))),
        tinkerGraph,
      ),
    );
    // marko -> 29; everyone else -> own name
    expect(r.sort()).toEqual([29, 'josh', 'lop', 'peter', 'ripple', 'vadas']);
  });

  // A value-typed pick token (a list) matches its option key by VALUE, not reference —
  // TinkerPop keys `.option()` through a Map (Java `.equals`), verified against a real
  // gremlin-console. A structurally-equal-but-distinct list still matches.
  test('branch matches a value-typed (list) option key by value', () => {
    const matched = arr(
      run(
        traversal(
          V(),
          branch(constant([1, 2]))
            .option([1, 2], constant('by-value'))
            .none(constant('no')),
        ),
        tinkerGraph,
      ),
    );
    expect(new Set(matched)).toEqual(new Set(['by-value'])); // every vertex routed to the list option
    const missed = arr(
      run(
        traversal(
          V(),
          branch(constant([1, 2]))
            .option([9, 9], constant('by-value'))
            .none(constant('no')),
        ),
        tinkerGraph,
      ),
    );
    expect(new Set(missed)).toEqual(new Set(['no'])); // a non-equal list falls through to .none
  });

  // No matching option and no default => traverser dropped.
  test('branch drops traverser with no match and no default', () => {
    const r = arr(
      run(
        traversal(V(), hasLabel('PERSON'), branch(values('age')).option(29, constant('marko'))),
        tinkerGraph,
      ),
    );
    expect(r).toEqual(['marko']);
  });
});
