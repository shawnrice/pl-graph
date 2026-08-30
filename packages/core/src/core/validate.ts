import { ErrorCode, LenkeError } from '@lenke/errors';

import { fromTaggedJson } from '../temporal.js';
import { LenkeRecord } from './LenkeRecord.js';

/**
 * A well-formed **label** (node label / edge type): non-empty and free of the
 * `::` sequence. GraphSON joins a node's labels with `::`, so a `::` inside one
 * label is ambiguous there (and bare GQL can't name it); an empty label
 * collapses to "no labels" in GraphSON/CSV. Enforced at the graph's mutation
 * boundary so a name that won't round-trip can't enter the model. Mirrors the
 * Rust engine's `validate_label`.
 */
export const validateLabel = (label: string): void => {
  if (label === '') {
    throw new LenkeError('a label / edge type must be non-empty', { code: ErrorCode.InvalidValue });
  }

  if (label.includes('::')) {
    throw new LenkeError(
      `a label / edge type cannot contain '::' (the GraphSON multi-label separator): ${JSON.stringify(label)}`,
      { code: ErrorCode.InvalidValue },
    );
  }
};

/**
 * A well-formed **property key**: non-empty (an empty key has no CSV column
 * header / pg-text `key:value` form). Mirrors the Rust engine's `validate_prop_key`.
 */
export const validatePropertyKey = (key: string): void => {
  if (key === '') {
    throw new LenkeError('a property key must be non-empty', { code: ErrorCode.InvalidValue });
  }
};

/**
 * A well-formed **property value**. The LPG numeric type is float64 (the Rust
 * the Rust engine has no bigint; every codec + the FFI param boundary would coerce a bigint
 * to a number, losing precision above 2^53). So a JS `bigint` is rejected at the
 * mutation boundary rather than silently downgraded — pass `Number(x)` for a
 * safe-range value, or a string. Recurses into list elements so a bigint can't
 * hide inside an array. (`NaN`/`Infinity`/`undefined` are *coerced* to null by
 * the codec layer, not rejected here — those are JS non-values with no exact
 * representation; a bigint is a deliberate, present value whose exactness matters.)
 *
 * The param + FFI boundaries already reject bigint with the same code; this
 * closes the pure-JS in-process store, the one path that stored it raw. No Rust
 * mirror is needed — `bigint` is a JS-only type that cannot reach the engine.
 */
export const validatePropertyValue = (value: unknown): void => {
  if (typeof value === 'bigint') {
    throw new LenkeError(
      `a bigint property value is not supported: the numeric model is float64 — ` +
        `pass Number(${value}n) for a safe-range value, or a string`,
      { code: ErrorCode.InvalidValue },
    );
  }

  if (Array.isArray(value)) {
    for (const element of value) {
      validatePropertyValue(element);
    }
  }

  // A record/map is a valid property value; recurse its fields (a bigint inside
  // a map is still rejected). Both a `LenkeRecord` and a raw plain object (which
  // the write path canonicalizes into one) are checked.
  if (value instanceof LenkeRecord) {
    for (const v of value.values()) {
      validatePropertyValue(v);
    }
  } else if (
    value !== null &&
    typeof value === 'object' &&
    Object.getPrototypeOf(value) === Object.prototype
  ) {
    for (const v of Object.values(value)) {
      validatePropertyValue(v);
    }
  }
};

/** Validate every label, property key, and property value about to enter the graph. */
export const validateElementNames = (
  labels: Iterable<string>,
  properties: Readonly<Record<string, unknown>> | undefined,
): void => {
  for (const label of labels) {
    validateLabel(label);
  }

  if (properties) {
    for (const key of Object.keys(properties)) {
      validatePropertyKey(key);
      validatePropertyValue(properties[key]);
    }
  }
};

/**
 * Lift a tagged temporal literal (`{"@date": "2020-01-01"}`) back to its instance.
 *
 * Temporals are *boxed* on the way out — `toJSON` emits this tagged form, so it is
 * what comes back through NDJSON, snapshots, `elementMap`, and any caller handing
 * back a value it previously read. Storing it raw made it silently uncomparable:
 * `WHERE v.vf = DATE '2020-01-01'` and every Gremlin predicate returned nothing,
 * because comparison needs an instance. Boxing on output obliges unboxing on
 * input — a store has to be able to ingest its own output format.
 *
 * Lists lift element-wise. Everything else passes through by identity, and the
 * common case allocates nothing.
 */
export const normalizePropertyValue = (value: unknown): unknown => {
  const lifted = fromTaggedJson(value);

  if (lifted) {
    return lifted;
  }

  if (Array.isArray(value)) {
    let changed = false;
    const out = value.map((v) => {
      const n = normalizePropertyValue(v);

      changed ||= n !== v;

      return n;
    });

    return changed ? out : value;
  }

  // An existing record re-normalizes its values (they may be tagged temporals
  // from a decoded document); a plain object becomes a canonical record. A class
  // instance (Temporal, Vertex/Edge/Path) is NOT a plain object, so it passes
  // through — only a bare `{…}` (or a record) is a map value.
  if (value instanceof LenkeRecord) {
    return LenkeRecord.from([...value].map(([k, v]) => [k, normalizePropertyValue(v)]));
  }

  if (
    value !== null &&
    typeof value === 'object' &&
    Object.getPrototypeOf(value) === Object.prototype
  ) {
    return LenkeRecord.from(Object.entries(value).map(([k, v]) => [k, normalizePropertyValue(v)]));
  }

  return value;
};

/** [`normalizePropertyValue`] across a property bag, reusing it when nothing moved. */
export const normalizeProperties = (bag: Record<string, unknown>): Record<string, unknown> => {
  let changed = false;
  const out: Record<string, unknown> = {};

  for (const key of Object.keys(bag)) {
    const v = normalizePropertyValue(bag[key]);

    changed ||= v !== bag[key];
    out[key] = v;
  }

  return changed ? out : bag;
};
