import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';
import { ErrorCode, hasErrorCode } from '@lenke/errors';

import type { Query, ReturnClause } from './ast.js';
import { createTestSocialGraph } from './fixtures/createTestSocialGraph.js';
import { parseQuery, query } from './index.js';
import { GqlSyntaxError } from './lexer.js';
import { parse } from './parser.js';

/** Capture a thrown error for code-level assertions. */
const thrown = (fn: () => unknown): unknown => {
  try {
    fn();
  } catch (e) {
    return e;
  }

  return undefined;
};

/** The value of the first RETURN item's literal, for accept-case assertions. */
const litValue = (text: string): unknown => {
  const q = parseQuery(`RETURN ${text} AS r`) as Query;
  const [item] = (q.parts[0].clauses[0] as ReturnClause).projection.items;

  return (item.expr as { value: unknown }).value;
};

// --- #1: SET / REMOVE keep the property index consistent ---------------------

describe('hardening: a prototype-key param reference throws, never reads Object.prototype', () => {
  test('$__proto__ / $constructor / $nope unbound throw MissingParameter, not a prototype object', () => {
    const g = new Graph();
    g.addVertex({ id: 'a', labels: ['N'], properties: { v: 1 } });

    // An unbound param reference is a loud error — and a prototype-key name is
    // no exception: it throws `MissingParameter` before it could ever resolve to
    // `Object.prototype`. (Previously these read as a silent unbound NULL.)
    for (const name of ['nope', '__proto__', 'constructor']) {
      const err = thrown(() => query(g, `MATCH (n:N) RETURN $${name} AS x`));
      expect(hasErrorCode(err, ErrorCode.MissingParameter)).toBe(true);
    }

    // A param the caller genuinely binds under that name (an OWN property —
    // computed key, so it doesn't set the prototype) still resolves as data.
    expect(query(g, 'MATCH (n:N) RETURN $__proto__ AS x', { ['__proto__']: 7 })).toEqual([
      { x: 7 },
    ]);
  });
});

describe('hardening: a bigint param is rejected, not silently mishandled', () => {
  test('a bigint $param throws InvalidValue; a safe-range number still works', () => {
    const g = new Graph();
    g.addVertex({ id: 'a', labels: ['N'], properties: { amt: 100 } });

    // The numeric model is float64 — a bigint can't bind without precision loss
    // above 2^53, so it is rejected the same way the native FFI boundary rejects
    // it. Previously a bigint param slipped into the comparator and `> $n`
    // silently dropped every row (round-6 R-BIGINT-MODEL).
    const err = thrown(() => query(g, 'MATCH (n:N) WHERE n.amt > $n RETURN n.amt', { n: 50n }));
    expect(hasErrorCode(err, ErrorCode.InvalidValue)).toBe(true);

    // the same query with a plain number binds and filters correctly
    expect(query(g, 'MATCH (n:N) WHERE n.amt > $n RETURN n.amt AS a', { n: 50 })).toEqual([
      { a: 100 },
    ]);
  });
});

describe('hardening: a lone-surrogate string param is rejected (round-12 F1)', () => {
  test('an unpaired UTF-16 surrogate $param throws InvalidJson; a valid pair binds', () => {
    const g = new Graph();

    // A JS string can carry a lone surrogate; the native store (UTF-8) cannot, and
    // rejects it as the param JSON-crosses the FFI boundary. The TS param path must
    // reject it too so both engines accept/reject the same inputs.
    for (const bad of ['\uD800', '\uDC00', 'a\uD800b']) {
      const err = thrown(() => query(g, 'INSERT (:X {v: $s})', { s: bad }));
      expect(hasErrorCode(err, ErrorCode.InvalidJson)).toBe(true);
    }

    // a well-formed astral character binds fine
    expect(() => query(g, 'INSERT (:X {v: $s})', { s: '😀' })).not.toThrow();
  });
});

describe('hardening: SET/REMOVE maintain the property index', () => {
  test('SET reindexes, so an indexed seek finds the new value', () => {
    const plain = createTestSocialGraph();
    const indexed = createTestSocialGraph();
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });

    for (const g of [plain, indexed]) {
      query(g, `MATCH (n:Person {name: 'marko'}) SET n.age = 31`);
    }

    // A WHERE equality seeds from `vertexPropertyIndex` when the key is indexed.
    // Before the fix the bare `el.properties =` write left the index mapping
    // marko under his old age, so the indexed seek missed him.
    const q = `MATCH (n:Person) WHERE n.age = 31 RETURN n.name`;
    expect(query(indexed, q)).toEqual([{ 'n.name': 'marko' }]);
    expect(query(indexed, q)).toEqual(query(plain, q));
  });

  test('SET reindexes, so the old value no longer seeks the node', () => {
    const indexed = createTestSocialGraph();
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    query(indexed, `MATCH (n:Person {name: 'marko'}) SET n.age = 31`);

    expect(query(indexed, `MATCH (n:Person) WHERE n.age = 29 RETURN n.name`)).toEqual([]);
  });

  test('REMOVE reindexes (indexed and unindexed agree)', () => {
    const plain = createTestSocialGraph();
    const indexed = createTestSocialGraph();
    indexed.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });

    for (const g of [plain, indexed]) {
      query(g, `MATCH (n:Person {name: 'marko'}) SET n.age = 100`);
      query(g, `MATCH (n:Person {name: 'marko'}) REMOVE n.age`);
    }

    const q = `MATCH (n:Person) WHERE n.age = 100 RETURN n.name`;
    expect(query(indexed, q)).toEqual([]);
    expect(query(indexed, q)).toEqual(query(plain, q));
  });
});

// --- #2: parser recursion-depth guard ---------------------------------------

describe('hardening: deep nesting is a syntax error, not a stack overflow', () => {
  test('nested parentheses', () => {
    const deep = `RETURN ${'('.repeat(5000)}1${')'.repeat(5000)} AS r`;
    const e = thrown(() => parse(deep));
    expect(e).toBeInstanceOf(GqlSyntaxError);
    expect(hasErrorCode(e, ErrorCode.Syntax)).toBe(true);
  });

  test('nested NOT', () => {
    const e = thrown(() => parse(`MATCH (n) WHERE ${'NOT '.repeat(5000)}n.x RETURN n`));
    expect(e).toBeInstanceOf(GqlSyntaxError);
  });

  test('nested label negation', () => {
    const e = thrown(() => parse(`MATCH (n:${'!'.repeat(5000)}A) RETURN n`));
    expect(e).toBeInstanceOf(GqlSyntaxError);
  });

  test('nested lists', () => {
    const deep = `RETURN ${'['.repeat(5000)}1${']'.repeat(5000)} AS r`;
    expect(() => parse(deep)).toThrow(GqlSyntaxError);
  });

  test('a normally-nested query still parses', () => {
    expect(() => parse(`RETURN (((1 + 2)) * 3) AS r`)).not.toThrow();
  });

  // Round-12 C1: the associative operator AST is n-ary (a flat array), so a long
  // left-associative chain far past the old crash point evaluates without a stack
  // overflow (native SIGSEGV'd; TS threw an uncatchable RangeError) instead of
  // aborting — given a ceiling that admits it. Only the anti-resource-abuse
  // ceiling (default 10k, configurable per graph) rejects.
  test('a long AND / arithmetic chain evaluates without a stack overflow', () => {
    const g = new Graph({ maxOperatorChain: 200_000 });
    expect(query(g, `RETURN ${Array(50_000).fill('true').join(' AND ')} AS r`)).toEqual([
      { r: true },
    ]);
    expect(query(g, `RETURN ${Array(50_000).fill('1').join(' + ')} AS s`)).toEqual([{ s: 50_000 }]);
  });

  test('an over-cap operator chain is a clean E_SYNTAX', () => {
    // 10_001 operators > default ceiling (10k)
    const e = thrown(() => parse(`RETURN ${Array(10_002).fill('true').join(' AND ')} AS r`));
    expect(e).toBeInstanceOf(GqlSyntaxError);
    expect(hasErrorCode(e, ErrorCode.Syntax)).toBe(true);
    expect(() => parse(`RETURN ${Array(10_002).fill('1').join(' + ')} AS r`)).toThrow(
      GqlSyntaxError,
    );
  });

  test('the operator-chain ceiling is configurable per graph', () => {
    // default: 10_000 ops ok, 10_001 rejected
    expect(() => parse(`RETURN ${Array(10_001).fill('true').join(' AND ')} AS r`)).not.toThrow();
    expect(() => parse(`RETURN ${Array(10_002).fill('true').join(' AND ')} AS r`)).toThrow(
      GqlSyntaxError,
    );
    // a lower configured ceiling rejects sooner; a higher one admits more
    const tight = new Graph({ maxOperatorChain: 5 });
    expect(() => query(tight, `RETURN ${Array(7).fill('true').join(' AND ')} AS r`)).toThrow(); // 6 ops
    const loose = new Graph({ maxOperatorChain: 100 });
    expect(query(loose, `RETURN ${Array(51).fill('true').join(' AND ')} AS r`)).toEqual([
      { r: true },
    ]);
  });
});

// --- #3: lexer numeric-literal validation -----------------------------------

describe('hardening: malformed numeric literals are rejected', () => {
  for (const bad of ['0x', '0b', '0o', '0b2', '0o8', '0o9', '1e', '1e+', '0xG']) {
    test(`rejects '${bad}'`, () => {
      expect(() => parse(`RETURN ${bad} AS r`)).toThrow(GqlSyntaxError);
    });
  }

  test('rejects an overflowing exponent (Infinity)', () => {
    expect(() => parse(`RETURN 1e999 AS r`)).toThrow(GqlSyntaxError);
  });

  test('rejects an integer beyond the safe range', () => {
    expect(() => parse(`RETURN 99999999999999999999 AS r`)).toThrow(GqlSyntaxError);
  });

  test('still accepts valid integers, bases, and floats', () => {
    expect(litValue('0')).toBe(0);
    expect(litValue('255')).toBe(255);
    expect(litValue('0xFF')).toBe(255);
    expect(litValue('0o17')).toBe(15);
    expect(litValue('0b101')).toBe(5);
    expect(litValue('1_000')).toBe(1000);
    expect(litValue('1.5')).toBe(1.5);
    expect(litValue('1.5e2')).toBe(150);
    expect(litValue('.5')).toBe(0.5);
  });
});

// --- #4: SKIP / LIMIT / quantifier integer validation -----------------------

describe('hardening: SKIP/LIMIT/quantifier require non-negative integers', () => {
  for (const clause of ['LIMIT 2.5', 'SKIP 1.5', 'LIMIT 0.5']) {
    test(`rejects '${clause}'`, () => {
      expect(() => parse(`MATCH (n) RETURN n ${clause}`)).toThrow(GqlSyntaxError);
    });
  }

  test('rejects a fractional quantifier bound', () => {
    expect(() => parse(`MATCH (a)-[:R]->{1.5}(b) RETURN b`)).toThrow(GqlSyntaxError);
  });

  test('rejects a quantifier whose upper bound is below its lower bound', () => {
    expect(() => parse(`MATCH (a)-[:R]->{3,2}(b) RETURN b`)).toThrow(GqlSyntaxError);
  });

  test('still accepts valid integer bounds', () => {
    expect(() => parse(`MATCH (n) RETURN n SKIP 1 LIMIT 2`)).not.toThrow();
    expect(() => parse(`MATCH (a)-[:R]->{1,3}(b) RETURN b`)).not.toThrow();
    expect(() => parse(`MATCH (a)-[:R]->{2}(b) RETURN b`)).not.toThrow();
  });

  // ISO `nonNegativeIntegerSpecification` (opengql:2268) = literal | dynamic param.
  test('LIMIT / OFFSET accept a dynamic $param', () => {
    expect(() => parse(`MATCH (n) RETURN n LIMIT $lim`)).not.toThrow();
    expect(() => parse(`MATCH (n) RETURN n OFFSET $off LIMIT $lim`)).not.toThrow();
  });

  test('SKIP stays literal-only (Cypher synonym) — SKIP $n is rejected', () => {
    expect(() => parse(`MATCH (n) RETURN n SKIP $n`)).toThrow(GqlSyntaxError);
  });

  test('a $param LIMIT bound must resolve to a non-negative integer', () => {
    const g = createTestSocialGraph();

    for (const bad of [2.5, -1, 'x', null]) {
      const e = thrown(() => query(g, `MATCH (n:Person) RETURN n LIMIT $lim`, { lim: bad }));
      expect(hasErrorCode(e, ErrorCode.InvalidValue)).toBe(true);
    }

    // A missing bound param is the usual MissingParameter error.
    const miss = thrown(() => query(g, `MATCH (n:Person) RETURN n LIMIT $lim`));
    expect(hasErrorCode(miss, ErrorCode.MissingParameter)).toBe(true);
    // A valid bound runs.
    expect(() => query(g, `MATCH (n:Person) RETURN n LIMIT $lim`, { lim: 2 })).not.toThrow();
  });
});

// --- #5: plain DELETE must not orphan relationships -------------------------

describe('hardening: DELETE vs DETACH DELETE', () => {
  test('plain DELETE of a connected node throws and leaves the graph intact', () => {
    const g = createTestSocialGraph();
    const e = thrown(() => query(g, `MATCH (n:Person {name: 'marko'}) DELETE n`));

    expect(hasErrorCode(e, ErrorCode.InvalidGraphOp)).toBe(true);
    expect(query(g, `MATCH (n:Person) RETURN count(*) AS c`)).toEqual([{ c: 4 }]);
  });

  test('plain DELETE of an isolated node succeeds', () => {
    const g = createTestSocialGraph();
    query(g, `INSERT (x:Loner {name: 'solo'})`);
    query(g, `MATCH (n:Loner) DELETE n`);

    expect(query(g, `MATCH (n:Loner) RETURN count(*) AS c`)).toEqual([{ c: 0 }]);
  });

  test('DETACH DELETE still cascades incident edges', () => {
    const g = createTestSocialGraph();
    query(g, `MATCH (n:Person {name: 'marko'}) DETACH DELETE n`);

    expect(query(g, `MATCH (n:Person) RETURN count(*) AS c`)).toEqual([{ c: 3 }]);
  });
});

// --- #6: variable-length segments can't bind an edge or carry a predicate ----

describe('hardening: variable-length segment restrictions', () => {
  const g = createTestSocialGraph();

  test('accepts a per-hop edge variable on a quantified segment', () => {
    // The edge variable names each hop's edge in turn for the per-hop predicate;
    // it is not yet a group/list variable exposed to the outer query.
    expect(() => query(g, `MATCH (a)-[r:KNOWS]->*(b) RETURN b`)).not.toThrow();
  });

  test('accepts a per-hop property predicate on a quantified segment', () => {
    expect(() => query(g, `MATCH (a)-[:KNOWS {weight: 1}]->+(b) RETURN b`)).not.toThrow();
  });

  test('accepts a per-hop predicate together with a shortest selector', () => {
    // The shortest BFS now expands only over predicate-passing edges (shortest path
    // in the filtered subgraph); both engines accept it.
    expect(() =>
      query(g, `MATCH ANY SHORTEST (a)-[r:KNOWS WHERE r.weight > 0]->*(b) RETURN b`),
    ).not.toThrow();
  });

  test('a plain quantified segment (label only) still works', () => {
    expect(() =>
      query(g, `MATCH (a:Person {name: 'marko'})-[:KNOWS]->+(b) RETURN b.name`),
    ).not.toThrow();
  });
});

// --- #7: undirected traversal counts a self-loop once -----------------------

describe('hardening: self-loop adjacency', () => {
  const loopGraph = (): Graph => {
    const g = new Graph();
    const n = g.addVertex({ labels: ['N'], properties: { name: 'n' } });
    g.addEdge({ from: n, to: n, labels: ['LOOP'], properties: {} });

    return g;
  };

  test('an undirected (~) walk yields a self-loop once, not twice', () => {
    expect(query(loopGraph(), `MATCH (a)~[r]~(b) RETURN count(*) AS c`)).toEqual([{ c: 1 }]);
  });

  test('directed walks each yield the self-loop once', () => {
    expect(query(loopGraph(), `MATCH (a)-[r]->(b) RETURN count(*) AS c`)).toEqual([{ c: 1 }]);
    expect(query(loopGraph(), `MATCH (a)<-[r]-(b) RETURN count(*) AS c`)).toEqual([{ c: 1 }]);
  });
});

// --- ISO medium-conformance batch -------------------------------------------

const g0 = createTestSocialGraph();

describe('hardening: ISO data exceptions in arithmetic', () => {
  for (const q of ['RETURN 1 / 0 AS r', 'RETURN 5 % 0 AS r', 'RETURN 1.0 / 0 AS r']) {
    test(`division by zero raises: ${q}`, () => {
      const e = thrown(() => query(g0, q));
      expect(hasErrorCode(e, ErrorCode.DataException)).toBe(true);
    });
  }

  for (const q of ["RETURN 'abc' + 1 AS r", 'RETURN true * 2 AS r']) {
    test(`non-numeric operand raises: ${q}`, () => {
      const e = thrown(() => query(g0, q));
      expect(hasErrorCode(e, ErrorCode.DataException)).toBe(true);
    });
  }

  test('a NULL operand still propagates to NULL (not an error)', () => {
    expect(query(g0, 'RETURN null + 1 AS r')).toEqual([{ r: null }]);
    expect(query(g0, 'RETURN 1 / null AS r')).toEqual([{ r: null }]);
  });
});

describe('hardening: ISO three-valued comparison of mixed types', () => {
  test('ordering across incomparable types is UNKNOWN (null)', () => {
    expect(query(g0, "RETURN 1 < 'a' AS r")).toEqual([{ r: null }]);
    expect(query(g0, "RETURN 'a' >= 1 AS r")).toEqual([{ r: null }]);
  });

  test('equality across types is simply false/true, not null', () => {
    expect(query(g0, "RETURN 5 = '5' AS r")).toEqual([{ r: false }]);
    expect(query(g0, "RETURN 5 <> '5' AS r")).toEqual([{ r: true }]);
  });

  test('same-type ordering (incl. booleans) still works', () => {
    expect(query(g0, 'RETURN 1 < 2 AS r')).toEqual([{ r: true }]);
    expect(query(g0, "RETURN 'a' < 'b' AS r")).toEqual([{ r: true }]);
    expect(query(g0, 'RETURN false >= false AS r')).toEqual([{ r: true }]);
  });
});

describe('hardening: aggregate validation', () => {
  test('nested aggregates are rejected', () => {
    const e = thrown(() => query(g0, 'MATCH (n:Person) RETURN sum(avg(n.age))'));
    expect(hasErrorCode(e, ErrorCode.Unsupported)).toBe(true);
  });

  test('an argless aggregate (other than count(*)) is rejected', () => {
    const e = thrown(() => query(g0, 'MATCH (n:Person) RETURN sum()'));
    expect(hasErrorCode(e, ErrorCode.Unsupported)).toBe(true);
  });

  test('count(*) and normal aggregates still work', () => {
    expect(query(g0, 'MATCH (n:Person) RETURN count(*) AS c')).toEqual([{ c: 4 }]);
  });
});

describe('hardening: group keys do not collide on non-finite numbers', () => {
  test('NaN and null form distinct groups', () => {
    const g = new Graph();
    g.addVertex({ labels: ['T'], properties: { v: -1 } }); // sqrt(-1) → NaN
    g.addVertex({ labels: ['T'], properties: {} }); //          sqrt(null) → null
    const rows = query(g, 'MATCH (n:T) RETURN sqrt(n.v) AS k, count(*) AS c');
    expect(rows).toHaveLength(2);
    expect(rows.every((r) => r.c === 1)).toBe(true);
  });
});

// --- variable-length trail semantics ----------------------------------------

describe('hardening: variable-length trail semantics', () => {
  const ring = (): Graph => {
    // a → b → c → a
    const g = new Graph();
    const a = g.addVertex({ labels: ['N'], properties: { name: 'a' } });
    const b = g.addVertex({ labels: ['N'], properties: { name: 'b' } });
    const c = g.addVertex({ labels: ['N'], properties: { name: 'c' } });
    g.addEdge({ from: a, to: b, labels: ['R'], properties: {} });
    g.addEdge({ from: b, to: c, labels: ['R'], properties: {} });
    g.addEdge({ from: c, to: a, labels: ['R'], properties: {} });

    return g;
  };

  test('a cycle terminates and yields one row per trail', () => {
    const g = ring();
    // From a, the trails of ≥1 hop are a→b, a→b→c, a→b→c→a (each edge once); the
    // next step would reuse a→b, so it stops. Three trails.
    expect(query(g, `MATCH (a:N {name:'a'})-[:R]->+(x) RETURN count(*) AS c`)).toEqual([{ c: 3 }]);
  });

  test('an endpoint reached by multiple trails appears once per trail', () => {
    const g = new Graph();
    const a = g.addVertex({ labels: ['N'], properties: { name: 'a' } });
    const b = g.addVertex({ labels: ['N'], properties: { name: 'b' } });
    const c = g.addVertex({ labels: ['N'], properties: { name: 'c' } });
    const d = g.addVertex({ labels: ['N'], properties: { name: 'd' } });
    g.addEdge({ from: a, to: b, labels: ['R'], properties: {} });
    g.addEdge({ from: a, to: c, labels: ['R'], properties: {} });
    g.addEdge({ from: b, to: d, labels: ['R'], properties: {} });
    g.addEdge({ from: c, to: d, labels: ['R'], properties: {} });
    // d is reached by two distinct 2-hop trails: a→b→d and a→c→d.
    expect(query(g, `MATCH (a:N {name:'a'})-[:R]->{2,2}(d) RETURN count(*) AS c`)).toEqual([
      { c: 2 },
    ]);
  });

  test('an unbounded * on a dense graph hits the trail budget instead of hanging', () => {
    const g = new Graph();
    const vs = Array.from({ length: 8 }, (_, i) =>
      g.addVertex({ labels: ['N'], properties: { i } }),
    );

    for (const f of vs) {
      for (const t of vs) {
        if (f !== t) {
          g.addEdge({ from: f, to: t, labels: ['R'], properties: {} });
        }
      }
    }

    const e = thrown(() => query(g, `MATCH (a)-[:R]->*(b) RETURN count(*) AS c`));
    expect(hasErrorCode(e, ErrorCode.ResourceExhausted)).toBe(true);
    // Explicit generous timeout: this asserts a *correctness* property (the trail
    // budget throws), not a wall-clock bound — the enumeration runs ~5 s and would
    // otherwise flake against bun's 5 s default on a loaded CI runner.
  }, 30_000);
});
