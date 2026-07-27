/**
 * An ISO GQL record / Gremlin map value: a `Map` with keys kept **sorted**
 * (canonical) and JSON serialization as an object. A `Map` subclass so it is
 * distinct from a graph element (a plain object with an `id`) and from a plain
 * object, while `JSON.stringify` still emits a sorted `{…}` (via `toJSON`) —
 * byte-identical to the native `Value::Map`. Shared by the GQL and Gremlin
 * engines and the serialization codecs (which is why it lives in `@lenke/core`,
 * the substrate both frontends read/write).
 */
export class LenkeRecord extends Map<string, unknown> {
  /**
   * Build a canonical record from entries: a duplicate key takes the **last**
   * value, then keys are sorted. Values are stored as-is — the caller normalizes
   * them first (each normalizer recurses in its own way).
   */
  static from(entries: Iterable<readonly [string, unknown]>): LenkeRecord {
    const dedup = new Map<string, unknown>(entries); // later entry wins
    const sorted = [...dedup].sort(([a], [b]) => {
      if (a < b) {
        return -1;
      }

      return a > b ? 1 : 0;
    });

    return new LenkeRecord(sorted);
  }

  toJSON(): Record<string, unknown> {
    return Object.fromEntries(this);
  }
}

/** Is `v` a record/map value (vs a graph element, list, or scalar)? */
export const isRecord = (v: unknown): v is LenkeRecord => v instanceof LenkeRecord;
