/**
 * The core LPG property-value model: the vendor-neutral scalar set that the GQL
 * and Gremlin interfaces both share, plus lists of those scalars. Every codec
 * (PG-JSON, GraphSON, CSV, …) encodes and decodes exactly this model, so the
 * single source of truth for "what a property value may be" — and where richer
 * JS values lose information — lives here, not in each format.
 */
import { coerceTemporal, fromTaggedJson, LenkeRecord, type Temporal } from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';

export type PropertyValue =
  | string
  | boolean
  | number
  | null
  | Temporal
  | LenkeRecord
  | readonly PropertyValue[];

/** A property bag on a vertex or edge in the LPG model. */
export type PropertyBag = Readonly<Record<string, PropertyValue>>;

/**
 * Coerce an arbitrary JS value into the LPG `PropertyValue` model. This is the
 * one place lossiness is defined:
 *   - `undefined`            → `null`
 *   - `NaN` / `±Infinity`    → `null` (not representable across formats)
 *   - `bigint`               → throw `E_INVALID_VALUE` (the numeric model is
 *                              float64; a bigint would lose precision above 2^53,
 *                              so it is rejected everywhere rather than silently
 *                              downgraded — pass `Number(x)` or a string). Matches
 *                              the in-process store + FFI-param boundary.
 *   - arrays                 → each element normalized recursively
 *   - objects / Date / Map / Set / functions / symbols → throw (not LPG scalars)
 *
 * Codecs call this at the boundary so out-of-model values fail loudly rather
 * than silently producing a non-round-trippable document.
 */
/**
 * Maximum list-nesting depth. Bounds recursion so an adversarial deeply-nested
 * array cannot exhaust the stack; mirrors the Rust JSON decoders, which serde
 * caps at 128 levels during parsing.
 */
const MAX_NESTING = 128;

/**
 * Does `s` contain a lone (unpaired) UTF-16 surrogate? A high surrogate must be
 * immediately followed by a low surrogate and vice-versa; anything else is not a
 * valid Unicode scalar. The native engine stores UTF-8, which cannot represent a
 * lone surrogate, and its JSON decoders reject one at ingest — so the shared LPG
 * string model excludes them, and this is the boundary that enforces it (in TS a
 * `\uD800` escape survives `JSON.parse`, unlike Rust's serde). Exported so the
 * GQL param boundary can apply the identical check.
 */
export const hasLoneSurrogate = (s: string): boolean => {
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);

    if (c >= 0xd800 && c <= 0xdbff) {
      // high surrogate: must be followed by a low. `charCodeAt` past the end is
      // NaN, and `NaN >= …` is false, so the positive-form test catches an
      // end-of-string high surrogate too.
      const next = s.charCodeAt(i + 1);

      if (!(next >= 0xdc00 && next <= 0xdfff)) {
        return true;
      }

      i++; // valid pair — skip the low half
    } else if (c >= 0xdc00 && c <= 0xdfff) {
      return true; // low with no preceding high
    }
  }

  return false;
};

const normalizeAt = (value: unknown, depth: number): PropertyValue => {
  if (value === null || typeof value === 'boolean') {
    return value;
  }

  if (typeof value === 'string') {
    if (hasLoneSurrogate(value)) {
      throw new LenkeError(
        'a string value contains a lone (unpaired) UTF-16 surrogate, which is not a valid ' +
          'Unicode scalar in the LPG string model',
        { code: ErrorCode.InvalidJson },
      );
    }

    return value;
  }

  if (typeof value === 'number') {
    // Non-finite (±Infinity, NaN) is a DISTINCT present value (IEEE 754 / Postgres /
    // Neo4j model), NOT null — it is ordered (−∞ < finite < +∞), comparable, and
    // survives IS NULL / aggregates as a real value. It converts to null only when it
    // leaves through JSON (JSON has no NaN/Infinity), which is expected and lossy. Use
    // `_is_nan` / `_is_infinite` / `_is_finite` to test for it in a query.
    return value;
  }

  if (typeof value === 'undefined') {
    return null;
  }

  if (typeof value === 'bigint') {
    throw new LenkeError(
      `a bigint value is not supported: the numeric model is float64 — ` +
        `pass Number(${value}n) for a safe-range value, or a string`,
      { code: ErrorCode.InvalidValue },
    );
  }

  // A lenke temporal instance passes through; a TC39 `Temporal.PlainDate`/
  // `PlainDateTime`/`Duration` is coerced via its ISO string (no hard dep on the
  // Temporal proposal — it's duck-typed).
  const temporal = coerceTemporal(value);

  if (temporal) {
    return temporal;
  }

  if (Array.isArray(value)) {
    if (depth >= MAX_NESTING) {
      throw new LenkeError('Property value nesting exceeds the maximum depth', {
        code: ErrorCode.InvalidShape,
      });
    }

    return value.map((v) => normalizeAt(v, depth + 1));
  }

  // A tagged temporal object `{"@date":"…"}` (from a decoded JSON codec) revives
  // to its instance.
  const revived = fromTaggedJson(value);

  if (revived) {
    return revived;
  }

  // A native `Date` is a zoned instant; lenke's temporal types are zone-less, so
  // silently coercing would have to guess a timezone (a data-corruption footgun).
  // Require the user to name the interpretation.
  if (value instanceof Date) {
    throw new LenkeError(
      'A native `Date` is a zoned instant, but lenke temporal types are zone-less. ' +
        'Convert explicitly with `LocalDateTime.fromJSDate(date, { zone })` (or `LocalDate.fromJSDate`), ' +
        'or pass an ISO string / a TC39 `Temporal.PlainDateTime`.',
      { code: ErrorCode.InvalidValue },
    );
  }

  // A record/map value: an existing `LenkeRecord` re-normalizes its fields; a
  // bare plain object becomes a canonical record (sorted keys, dup last-wins).
  if (depth >= MAX_NESTING) {
    throw new LenkeError('Property value nesting exceeds the maximum depth', {
      code: ErrorCode.InvalidShape,
    });
  }

  if (value instanceof LenkeRecord) {
    return LenkeRecord.from([...value].map(([k, v]) => [k, normalizeAt(v, depth + 1)]));
  }

  if (Object.getPrototypeOf(value) === Object.prototype) {
    return LenkeRecord.from(Object.entries(value).map(([k, v]) => [k, normalizeAt(v, depth + 1)]));
  }

  throw new LenkeError(
    `Property value is outside the LPG model: ${Object.prototype.toString.call(value)}`,
    { code: ErrorCode.InvalidValue },
  );
};

export const normalizeValue = (value: unknown): PropertyValue => normalizeAt(value, 0);

/** Normalize every value in a property bag. */
export const normalizeBag = (bag: Record<string, unknown>): Record<string, PropertyValue> => {
  const out: Record<string, PropertyValue> = {};

  for (const key of Object.keys(bag)) {
    out[key] = normalizeValue(bag[key]);
  }

  return out;
};
