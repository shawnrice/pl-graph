import { TEMPORAL_TAG_KEYS } from '../temporal.js';

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
    const out: Record<string, unknown> = {};

    for (const [k, v] of this) {
      out[escapeRecordKey(k)] = v;
    }

    return out;
  }
}

/** Is `v` a record/map value (vs a graph element, list, or scalar)? */
export const isRecord = (v: unknown): v is LenkeRecord => v instanceof LenkeRecord;

// A temporal is carried on the JSON wire as a single-key tagged object
// (`{"@date": "…"}`); a record whose own key IS a temporal tag would be
// indistinguishable from one on decode (`{"@date": "…"}` → a LocalDate, not a record).
// Only such "temporal-shaped" keys (one or more leading `@` then a recognized tag —
// `@date`, `@@date`, …) are escaped with one extra `@` on the wire and stripped back on
// decode; every other key (including `@user`, `@id`) is left untouched. The temporal
// check runs on the raw key (a single recognized tag), so `@date` stays a temporal
// while a record's `@date` travels as `@@date`. The native codecs do the same, keeping
// the wire byte-identical.

/** A key that looks like a tagged temporal (`@`-run + a recognized tag). */
const isTemporalShaped = (key: string): boolean => {
  const bare = key.replace(/^@+/, '');
  return bare.length < key.length && TEMPORAL_TAG_KEYS.has(`@${bare}`);
};

/** Escape a temporal-shaped record key for the JSON wire (see above); else unchanged. */
export const escapeRecordKey = (key: string): string => (isTemporalShaped(key) ? `@${key}` : key);

/** Invert {@link escapeRecordKey} when decoding a record from the JSON wire. */
export const unescapeRecordKey = (key: string): string =>
  isTemporalShaped(key) ? key.slice(1) : key;
