import { describe, expect, test } from 'bun:test';

import { complete, lastToken, META_COMMANDS } from './completion.js';

describe('lastToken', () => {
  test('the trailing identifier / backslash word', () => {
    expect(lastToken('MATCH (p:Per')).toBe('Per');
    expect(lastToken('\\c')).toBe('\\c');
    expect(lastToken('RETURN ')).toBe('');
    expect(lastToken('g.V().hasL')).toBe('hasL');
  });
});

describe('complete', () => {
  test('a leading backslash completes meta-commands', () => {
    const [hits, token] = complete('\\', 'gql');

    expect(token).toBe('\\');
    expect(hits).toEqual([...META_COMMANDS]);
  });

  test('\\c also offers \\clock', () => {
    const [hits] = complete('\\c', 'gql');

    expect(hits).toContain('\\c');
    expect(hits).toContain('\\clock');
  });

  test('GQL mode completes the live labels alongside keywords/functions', () => {
    // Context-free prefix match: `Comp` hits only the label.
    expect(complete('MATCH (p:Comp', 'gql', ['Person', 'Company'])[0]).toEqual(['Company']);
    // `Per` also matches a function — labels and the language share one pool.
    const [hits] = complete('MATCH (p:Per', 'gql', ['Person', 'Company']);

    expect(hits).toContain('Person');
    expect(hits).toContain('percentile_cont');
  });

  test('GQL keyword completion is case-insensitive', () => {
    const [hits] = complete('match (n) ret', 'gql');

    expect(hits).toContain('RETURN');
  });

  test('Gremlin mode completes steps', () => {
    const [hits] = complete('g.V().hasL', 'gremlin');

    expect(hits).toEqual(['hasLabel']);
  });

  test('JS mode offers no query completions', () => {
    expect(complete('_.fil', 'js')).toEqual([[], 'fil']);
  });

  test('an empty token yields no hits', () => {
    expect(complete('RETURN ', 'gql')).toEqual([[], '']);
  });
});
