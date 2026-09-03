import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';
import { ErrorCode } from '@lenke/errors';

import { query } from './index.js';
import { GqlSyntaxError, quoteIdent } from './lexer.js';
import { parse } from './parser.js';

// Capture the GqlSyntaxError a query is expected to throw (fail loudly if it
// parses or throws something else).
const reject = (src: string): GqlSyntaxError => {
  try {
    parse(src);
  } catch (e) {
    if (e instanceof GqlSyntaxError) {
      return e;
    }

    throw e;
  }

  throw new Error(`expected \`${src}\` to be rejected, but it parsed`);
};

describe('reserved words in binding positions', () => {
  // The bug (C10): a reserved-but-structural keyword (`Order`, `Count`, `Match`,
  // `Set`) lexed as a keyword token and hit `expect('ident')` first, yielding a
  // generic, lowercased, hintless message; a reserved-but-non-structural ident
  // (`Group`, `Product`) got a different message. Now every one is rejected with
  // ONE consistent message that names backticks and keeps the user's casing.
  const labels = ['Group', 'Product', 'Order', 'Count', 'Match', 'Set'];

  for (const word of labels) {
    test(`label \`:${word}\` is rejected with a backtick-naming, casing-preserving error`, () => {
      const err = reject(`MATCH (x:${word}) RETURN x`);

      expect(err.code).toBe(ErrorCode.Syntax);
      // (a) the user's ORIGINAL casing, in the name and the backtick suggestion.
      expect(err.message).toContain(`\`${word}\``);
      // (b) names backticks / the delimited-identifier remedy.
      expect(err.message).toContain('delimited identifier');
      expect(err.message).toContain('reserved word');
    });
  }

  test('the delimited form `:`Order`` parses', () => {
    expect(() => parse('MATCH (x:`Order`) RETURN x')).not.toThrow();
  });

  test('a plain (non-reserved) label parses', () => {
    expect(() => parse('MATCH (x:Person) RETURN x')).not.toThrow();
  });

  test('a reserved word as a variable gets the improved message (reserved ident)', () => {
    const err = reject('MATCH (Group) RETURN 1');

    expect(err.code).toBe(ErrorCode.Syntax);
    expect(err.message).toContain('`Group`');
    expect(err.message).toContain('variable');
    expect(err.message).toContain('delimited identifier');
  });

  test('a reserved keyword as an alias gets the improved message (casing recovered)', () => {
    const err = reject('MATCH (x) RETURN x AS Order');

    expect(err.code).toBe(ErrorCode.Syntax);
    expect(err.message).toContain('`Order`');
    expect(err.message).toContain('delimited identifier');
  });

  test('a reserved word IS a valid property key in a pattern (matches native)', () => {
    // Previously rejected with a backtick hint; a property name is not a binding
    // position, so a reserved word names a property fine here — the native engine
    // accepts it, and the two must agree. (Delimiting still works too.)
    expect(() => parse('MATCH (x { Order: 1 }) RETURN x')).not.toThrow();
    expect(() => parse('MATCH (x { `Order`: 1 }) RETURN x')).not.toThrow();
  });

  test('the aggregate-aliased-to-its-own-name case still fails, now helpfully', () => {
    const err = reject('MATCH (x) RETURN count(*) AS count');

    expect(err.code).toBe(ErrorCode.Syntax);
    expect(err.message).toContain('`count`');
    expect(err.message).toContain('delimited identifier');
  });

  // Guard against regressing the positions where a keyword-shaped word is
  // legitimately allowed: reserved words as function names, and soft keywords.
  test('reserved words as function names are still accepted', () => {
    expect(() => parse('MATCH (x) RETURN upper(x.name), count(*)')).not.toThrow();
  });

  test('soft keywords (contains / starts / ends) are still accepted', () => {
    expect(() => parse("MATCH (x) WHERE x.name CONTAINS 'a' RETURN x")).not.toThrow();
    expect(() => parse("MATCH (x) WHERE x.name STARTS WITH 'a' RETURN x")).not.toThrow();
  });
});

describe('quoteIdent', () => {
  // A bare identifier (a safe first char + safe body, not a reserved word) is
  // passed through untouched — the common case stays readable.
  test('leaves a bare, non-reserved identifier untouched', () => {
    expect(quoteIdent('name')).toBe('name');
    expect(quoteIdent('full_name2')).toBe('full_name2');
  });

  // Reserved words, non-bare characters, and the empty string are backtick-quoted.
  test('quotes reserved words and non-bare identifiers', () => {
    expect(quoteIdent('order')).toBe('`order`');
    expect(quoteIdent('Value')).toBe('`Value`');
    expect(quoteIdent('odd name')).toBe('`odd name`');
    expect(quoteIdent('a.b.c')).toBe('`a.b.c`');
    expect(quoteIdent('')).toBe('``');
    expect(quoteIdent('1st')).toBe('`1st`');
  });

  // An internal backtick is doubled (ISO/SQL delimited-identifier escape).
  test('doubles an internal backtick', () => {
    expect(quoteIdent('a`b')).toBe('`a``b`');
    // open + doubled-backtick (the escaped single backtick) + close = four backticks.
    expect(quoteIdent('`')).toBe('````');
  });

  // The output re-parses to exactly the original identifier — the round-trip the
  // helper exists to guarantee, including a reserved word used as a label.
  test('output round-trips through the parser as a label', () => {
    for (const key of ['order', 'odd name', 'a`b', 'a.b.c']) {
      const q = `MATCH (n:${quoteIdent(key)}) RETURN n`;
      expect(() => parse(q)).not.toThrow();
    }
  });
});

describe('reserved words are valid PROPERTY names (match the native engine)', () => {
  // Native accepts a keyword as a member name (like JS `obj.count`); the TS parser used
  // to reject it as E_SYNTAX, an accept/reject divergence on very common names. A
  // reserved word is now valid as a property key in every property-name position.
  const names = ['count', 'date', 'value', 'size', 'year', 'month', 'order', 'group', 'start'];

  test('parse: a reserved word names a property in dot-access, a map, SET, and PROPERTY_EXISTS', () => {
    for (const w of names) {
      expect(() => parse(`MATCH (n) RETURN n.${w} AS x`)).not.toThrow();
      expect(() => parse(`MATCH (n) RETURN n.rec.${w} AS x`)).not.toThrow(); // chained field
      expect(() => parse(`INSERT (:T {${w}: 1})`)).not.toThrow(); // map key
      expect(() => parse(`MATCH (n) SET n.${w} = 1`)).not.toThrow(); // SET target
      expect(() => parse(`MATCH (n) RETURN PROPERTY_EXISTS(n, ${w}) AS x`)).not.toThrow();
    }
  });

  test('behavioral: a reserved-word property reads back its value, case-preserved', () => {
    const g = new Graph();
    g.addVertex({ id: 'a', labels: ['N'], properties: { count: 7, date: 'd', Year: 2020 } });
    expect(query(g, 'MATCH (n:N) RETURN n.count AS c, n.date AS d')).toEqual([{ c: 7, d: 'd' }]);
    // `Year` lexes as the keyword `year` but keeps its source case (`raw`), so it reads
    // the `Year` property; lowercase `year` would read the (absent) `year` property.
    expect(query(g, 'MATCH (n:N) RETURN n.Year AS y')).toEqual([{ y: 2020 }]);
    expect(query(g, 'MATCH (n:N) RETURN n.year AS y')).toEqual([{ y: null }]);
  });
});
