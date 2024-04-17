/* eslint-disable max-lines-per-function */
import { Trie } from './Trie';

describe('Trie tests', () => {
  test('we can add two words', () => {
    const trie = new Trie<number>();
    trie.add('word', 1);
    trie.add('words', 2);

    expect(Array.from(trie.entries())).toEqual([
      ['word', 1],
      ['words', 2],
    ]);
  });

  test('Trie.keys works', () => {
    const trie = new Trie<number>();
    trie.add('word', 1);
    trie.add('words', 2);
    expect(Array.from(trie.keys())).toEqual(['word', 'words']);
  });

  test('Trie.values works', () => {
    const trie = new Trie<number>();
    trie.add('word', 1);
    trie.add('words', 2);
    expect(Array.from(trie.values())).toEqual([1, 2]);
  });

  test('we can remove a word', () => {
    const trie = new Trie<number>();
    trie.add('word', 1);
    trie.add('words', 2);
    trie.remove('words');

    expect(Array.from(trie.entries())).toEqual([['word', 1]]);
  });

  test('we can successfully check if we have a word', () => {
    const trie = new Trie<number>();
    trie.add('word', 1);
    trie.add('testing', 2);

    expect(trie.has('word')).toBe(true);
  });

  test('we do not get a false positive for a partial match', () => {
    const trie = new Trie<number>();
    trie.add('words', 2);
    expect(trie.has('words')).toBe(true);
    expect(trie.has('word')).toBe(false);
  });

  test('we can get a partial word', () => {
    const trie = new Trie<number>();
    trie.add('supercalifragilisticexpialidocious', Infinity);

    const node = trie.get('super');

    expect(!!node).toBe(true);
    expect(node?.char).toBe('r');
    expect(node?.isEndOfWord).toBe(false);
    expect(node?.value).toBeNull();
    expect(node?.word).toBeNull();
  });

  test('we can keep track of a word count and a node count when all unique', () => {
    const trie = new Trie<number>();
    'abcdefghijklmnopqrstuvwxyz'.split('').forEach((char, index) => {
      trie.add(char, index);
    });

    expect(trie.words).toBe(26);
    expect(trie.nodes).toBe(27); // the root node adds 1
  });

  test('we can track nodes and words', () => {
    const trie = new Trie<number>();
    trie.add('word', 1);
    trie.add('words', 2);

    expect(trie.words).toBe(2);
    expect(trie.nodes).toBe(6); // the root node adds 1

    trie.remove('word');
    expect(trie.words).toBe(1);
    expect(trie.nodes).toBe(6); // the root node adds 1

    trie.add('word', 1);
    trie.remove('words');
    expect(trie.words).toBe(1);
    expect(trie.nodes).toBe(5); // the root node adds 1
  });

  test('the root node exists in a new Trie', () => {
    const trie = new Trie<number>();
    expect(trie.words).toBe(0);
    expect(trie.nodes).toBe(1);
  });

  test('hasPartial works on a partial word', () => {
    const trie = new Trie<number>();
    trie.add('words', 2);
    expect(trie.hasPartial('word')).toBe(true);
  });

  test('hasPartial also detects a full word', () => {
    const trie = new Trie<number>();
    trie.add('words', 2);
    expect(trie.hasPartial('words')).toBe(true);
  });

  test('keyHasValue works with primitives', () => {
    const trie = new Trie<string>();
    trie.add('words', 'happy');
    expect(trie.keyHasValue('words', 'happy')).toBe(true);
  });

  test('keyHasValue incidentally detects null on a partial word', () => {
    const trie = new Trie<string>();
    trie.add('words', 'happy');
    // @ts-expect-error: this a defined edge case
    expect(trie.keyHasValue('word', null)).toBe(true);
  });

  test('keyHasValue does not detect has === equality', () => {
    const trie = new Trie<Record<string, number>>();
    trie.add('word', { a: 1 });
    expect(trie.keyHasValue('word', { a: 1 })).toBe(false);
  });

  test('we can get all the descendant words of a TrieNode', () => {
    const trie = new Trie<number>();
    trie.add('happy', 1);
    trie.add('happen', 2);
    trie.add('happenstance', 3);
    trie.add('whodunnit', 404);

    const node = trie.get('happ')!;

    expect(Array.from(node.descendants(), t => t.word)).toEqual([
      'happy',
      'happen',
      'happenstance',
    ]);
  });

  test('TrieNode.descendants does not include the current word', () => {
    const trie = new Trie<number>();
    trie.add('happi', 1);
    trie.add('happy', 1);
    trie.add('happier', 2);
    trie.add('happenstance', 3);
    trie.add('happiest', 4);

    const node = trie.get('happi')!;

    expect(Array.from(node.descendants(), t => t.word)).toEqual(['happier', 'happiest']);
  });

  test('descendantsOf works', () => {
    const trie = new Trie<number>();
    trie.add('happi', 1);
    trie.add('happy', 1);
    trie.add('happier', 2);
    trie.add('happenstance', 3);
    trie.add('happiest', 4);

    const nodes = trie.descendantsOf('happi')!;

    expect(Array.from(nodes, t => t.word)).toEqual(['happier', 'happiest']);
  });

  test('toMap works', () => {
    const trie = new Trie<number>();

    trie.add('abc', 1);
    trie.add('def', 2);
    trie.add('abba', 3);

    const map = trie.toMap();

    expect(Array.from(map.keys())).toEqual(['abc', 'abba', 'def']);
    expect(Array.from(map.values())).toEqual([1, 3, 2]);
  });

  test('toArray works', () => {
    const trie = new Trie<number>();
    trie.add('ok', 200);
    trie.add('forbidden', 400);
    trie.add("I'm a teapot", 418);
    expect(trie.toArray()).toEqual([
      ['ok', 200],
      ['forbidden', 400],
      ["I'm a teapot", 418],
    ]);
  });

  test('Trie.from works with an array of tuples', () => {
    const trie = Trie.from<number>([
      ['word', 1],
      ['words', 2],
    ]);
    expect(trie.has('word')).toBe(true);
    expect(trie.has('words')).toBe(true);
  });

  test('Trie.from works with a map', () => {
    const map = new Map<string, number>();
    map.set('word', 1);
    map.set('words', 2);

    const trie = Trie.from<number>(map);
    expect(trie.has('word')).toBe(true);
    expect(trie.has('words')).toBe(true);
  });
});
