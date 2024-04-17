import { root } from './root';

/**
 * A regular TrieNode, except that we can store values in it.
 *
 * The primary use case is to store key,value pairs and make them easily searchable for things like
 * autocomplete.
 */
export class TrieNode<T> {
  /**
   * A map of the children. This is `Map<char, TrieNode<T>>` and each "key" will be only a single
   * character.
   */
  children: Map<string, TrieNode<T>>;

  /**
   * This is the character value of the TrieNode. So if we were to take a word like `word` and turn
   * it into a Trie, we'd get four TrieNodes with chars `w` -> `o` -> `r` -> `d`. The different thing
   * about this implementation is that you can store an arbitrary value with a word. So on `d`, we'd
   * know that it is terminating for `word`, and we could store a value there that would be separate
   * than what we stored for `words`.
   */
  char: string;

  /**
   * The count is how many "words" pass through this node. We keep track of this because when
   * we remove a word, we can know if we need to delete the TrieNode or if we just need to decrement
   * the counter
   */
  count: number;

  /**
   * Whether or not this TrieNode represents the end of a word. This means that the TrieNode could
   * be both a the end of a word and not the end of a word at the same time. For instance, if we
   * have a Trie with the words `word` and `words`, then the TrieNodes representing `d` and `s`
   * would be the end of a word, but `d` would also be the middle of a word.
   */
  isEndOfWord: boolean;

  /**
   * If this is a "word" then it's the full text of it. This helps us not have to backtrack through
   * parents to find the full word from a single TrieNode
   */
  word: string | null;

  /**
   * An arbitrary value that the user is storing for each word
   */
  value: T | null;

  constructor(char: string) {
    // @ts-expect-error: we're doing something special for the root
    if (char !== root) {
      if (typeof char !== 'string') {
        throw new TypeError(
          // eslint-disable-next-line
          `A TrieNode's key must be a string. Received a ${typeof char} with a value of "${char}".`,
        );
      }

      if (char.length !== 1) {
        throw new RangeError(`A TrieNode's key must be a single character. Received "${char}"`);
      }
    }

    this.children = new Map();

    this.char = char;
    this.count = 0;
    this.isEndOfWord = false;
    this.value = null;
    this.word = null;
    // Typescript doesn't like what is below. BOO.
    // Object.assign(this, { char, count: 0, isEndOfWord: false, value: null, word: null });
  }

  addChild(child: TrieNode<T>) {
    this.children.set(child.char, child);
  }

  removeChild(child: TrieNode<T>) {
    this.children.delete(child.char);
  }

  /**
   * Gets all descendant words from this TrieNode
   *
   * This is useful for autocomplete:
   *
   * ```
   * const items = Array.from(
   *  trie.get(inputString)?.descendants() ?? [],
   *  node => ({ label: node.word, value: node.value })
   * )
   * ```
   */
  *descendants(): Generator<TrieNode<T>> {
    const queue: TrieNode<T>[] = Array.from(this.children.values());

    while (queue.length > 0) {
      const current = queue.shift();

      if (typeof current === 'undefined') {
        break;
      }

      if (current.isEndOfWord) {
        yield current;
      }

      const children = Array.from(current.children.values());

      // node.children is a map that respects insertion order, so we want to push things to the
      // front of the stack backwards
      for (let i = children.length - 1; i >= 0; i--) {
        queue.unshift(children[i]);
      }
    }
  }
}
