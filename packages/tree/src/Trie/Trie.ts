import { root } from './root';
import { TrieNode } from './TrieNode';

/**
 * A regular Trie, except that it inserts a different `value` for each `word` or `key`.
 *
 * This is designed to make it super fast to do a partial search for a key/value pair
 */
export class Trie<T> {
  private root: TrieNode<T>;

  public words: number = 0;

  public nodes: number = 1; // root

  public static from<T>(iterable: Iterable<[string, T]>): Trie<T> {
    const trie = new Trie<T>();

    for (const [key, value] of iterable) {
      trie.add(key, value);
    }

    return trie;
  }

  public static fromArray<T>(arr: [key: string, value: T][]): Trie<T> {
    return Trie.from(arr);
  }

  public static fromMap<T>(map: Map<string, T>): Trie<T> {
    return Trie.from(map);
  }

  constructor() {
    // @ts-expect-error: we're doing something special for the root
    this.root = new TrieNode<T>(root);
  }

  /**
   * Adds a word with a value to the collection
   */
  public add(key: string, value: T): Trie<T> {
    let node = this.root;

    for (let i = 0; i < key.length; i++) {
      const char = key[i];
      const isEndOfWord = i === key.length - 1;

      if (!node.children.has(char)) {
        // we will need to add a new node
        node.addChild(new TrieNode<T>(char));
        this.nodes++;
      }

      const next = node.children.get(char)!;
      node.addChild(next);

      if (isEndOfWord) {
        next.word = key;
        next.isEndOfWord = true;
        next.value = value;
      }

      next.count++;

      node = next;
    }

    this.words++;

    return this;
  }

  /**
   * Removes a word from the collection
   */
  public remove(key: string): Trie<T> {
    let node = this.root;
    const queue: TrieNode<T>[] = [this.root];

    for (let i = 0; i < key.length; i++) {
      const char = key[i];
      const isLast = i === key.length;

      if (!node.children.has(char)) {
        // We didn't actually store this, so lets do nothing.
        return this;
      }

      const next = node.children.get(char)!;

      queue.unshift(next);

      if (isLast && !next.isEndOfWord) {
        // The word exists in the tree only as a part of another word, so this is a noop
        return this;
      }

      node = next;
    }

    // So, now we have a queue that we can sort through. The queue is held in reverse order
    for (let i = 0; i < queue.length; i++) {
      const current = queue[i];

      if (current === this.root) {
        // We don't need to alter the root
        break;
      }

      if (i === 0) {
        // This is the last node, so, let's do something special to it
        current.value = null; // remove the node's value
        current.isEndOfWord = false;
      }

      // Reduce the count
      current.count--;

      if (current.count === 0) {
        // The node is no longer needed, so we can get the next node, which will be its parent, and
        // we can remove it
        const parent = queue[i + 1]!;

        if (parent) {
          parent.children.delete(current.char);
          this.nodes--;
        }
      }
    }

    this.words--;
    return this;
  }

  /**
   * Gets a trie node from a word. This does match partial words.
   *
   * So, if you add `words` and search for `wo`, you'll get the `o` back.
   */
  public get(key: string): TrieNode<T> | null {
    let node: TrieNode<T> | null = this.root;

    for (const char of key) {
      node = node?.children.get(char) ?? null;
    }

    return node;
  }

  /**
   * Checks if a word if fully in there. If a partial word is there (e.g. it is a subset of a word
   * that is already in the tree), then this will be false
   */
  public has(key: string): boolean {
    const node = this.get(key);

    return !!node?.isEndOfWord && node?.word === key;
  }

  /**
   * Checks if a word is partially there. E.g. a subset of a word is fine
   */
  public hasPartial(key: string): boolean {
    const node = this.get(key);

    return !!node;
  }

  /**
   * Checks if a key has a certain value
   *
   * This uses `===` equality
   */
  public keyHasValue(key: string, value: T): boolean {
    const node = this.get(key);

    return node?.value === value;
  }

  /**
   * Gets all full words that are descendants of the search string
   *
   * Proxies to TrieNode.descendants()
   */
  public *descendantsOf(key: string): Generator<TrieNode<T>> {
    const node = this.get(key);

    if (node) {
      yield* node.descendants();
    }
  }

  /**
   * Like Map.entries returns an iterator of tuples of `[key, value]`
   *
   * The order might be a bit wonky in that we follow child insertion order. Since children
   * are reused in the trie, then a word added later might appear before, e.g., adding:
   * (1) abc, (2) def, (3) aba, (4) aa, would give you back the order ['abc', 'aba', 'aa', 'def']
   *
   * It's impossible to return insertion order. If you want a particular order, then you should
   * turn the iterator into an array and sort it
   */
  public *entries(): Generator<[string, T]> {
    const queue: TrieNode<T>[] = [this.root];

    while (queue.length) {
      const node = queue.shift();

      if (!node) {
        break;
      }

      if (node.isEndOfWord && node.word && node.value !== null) {
        yield [node.word, node.value];
      }

      const children = Array.from(node.children.values());
      // Follow insertion order
      for (let i = children.length - 1; i >= 0; i--) {
        queue.unshift(children[i]);
      }
    }
  }

  /**
   * Like Map.values, it returns the values of all the words
   */
  public *values(): Generator<T> {
    for (const [, value] of this.entries()) {
      yield value;
    }
  }

  /**
   * Like Map.keys, it returns the keys in the Trie
   */
  public *keys(): Generator<string> {
    for (const [key] of this.entries()) {
      yield key;
    }
  }

  public toArray(): [string, T][] {
    return Array.from(this.entries());
  }

  public toMap(): Map<string, T> {
    return new Map(this.entries());
  }
}
