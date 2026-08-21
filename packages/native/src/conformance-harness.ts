// The switch that runs the TS differential/conformance suites against the native
// backend — lenke-engine, the primary (and now only) Rust driver. Result comparison
// is STRUCTURAL (order-independent), because the engine legitimately differs from the
// pure-TS engine in unspecified output order (row order, property-key order, label-set
// order), never in the answer. A structural divergence is therefore a real engine bug.
// (The retiring `lenke-core` byte-identical path is gone — the crate was deleted.)
import { existsSync } from 'node:fs';

import { createFfiEngineBackend } from './backend-ffi-engine.js';
import type { Backend } from './backend.js';

const LIB_EXT: string = { darwin: 'dylib', win32: 'dll' }[process.platform as string] ?? 'so';

export const ENGINE_LIB = new URL(
  `../../../crates/lenke-engine/target/release/liblenke_engine.${LIB_EXT}`,
  import.meta.url,
).pathname;

/** Retained (always true) so existing suites that branch on it keep compiling — the
 *  native backend under test is always the engine now. */
export const USE_ENGINE = true;

/** The native library under test, and whether it's present (skip the suite if not). */
export const NATIVE_LIB = ENGINE_LIB;
export const nativeReady = existsSync(NATIVE_LIB);

/** Build the native backend under test — the engine. */
export const nativeBackend = (): Backend => createFfiEngineBackend(ENGINE_LIB);

// --- structural canonicalization (engine mode only) -------------------------

/** A stable key for sorting rows/values. */
const keyOf = (v: unknown): string => JSON.stringify(v);

/** Recursively canonicalize a value for order-independent comparison: object keys
 * sorted; a `labels` array treated as a SET (sorted); every OTHER array left in
 * place (a list property value is order-significant). */
const canonValue = (v: unknown): unknown => {
  if (Array.isArray(v)) {
    return v.map(canonValue);
  }

  if (v !== null && typeof v === 'object') {
    const out: Record<string, unknown> = {};

    for (const k of Object.keys(v as Record<string, unknown>).sort()) {
      const val = (v as Record<string, unknown>)[k];
      out[k] =
        k === 'labels' && Array.isArray(val)
          ? val.map(canonValue).sort((a, b) => keyOf(a).localeCompare(keyOf(b)))
          : canonValue(val);
    }

    return out;
  }

  return v;
};

/** Canonicalize a query-result JSON string: the outer ROWS array is unordered
 * (sorted); everything nested is canonicalized per {@link canonValue}. */
export const canonResult = (json: string): string => {
  const parsed: unknown = JSON.parse(json);
  const canon = Array.isArray(parsed)
    ? parsed.map(canonValue).sort((a, b) => keyOf(a).localeCompare(keyOf(b)))
    : canonValue(parsed);

  return JSON.stringify(canon);
};

/** Whether two result JSON strings are equivalent — structural (order-independent),
 * because the engine and pure-TS engine agree on the answer but not on unspecified
 * output order. */
export const resultsEqual = (a: string, b: string): boolean => canonResult(a) === canonResult(b);
