// The switch that turns the TS differential/conformance suites into a drop-in
// readiness check for the standalone engine. By default the suites run against
// lenke-core (byte-identical to the TS engine — the shipped hard invariant). With
// `LENKE_ENGINE=1` they run against lenke-engine instead, and result comparison
// relaxes from byte-identical to STRUCTURAL (order-independent) — because the
// engine legitimately differs in unspecified output order (row order, property-key
// order, label-set order), never in the answer. A structural divergence under
// `LENKE_ENGINE=1` is therefore a real engine bug.
import { existsSync } from 'node:fs';

import { createFfiEngineBackend } from './backend-ffi-engine.js';
import { createFfiBackend } from './backend-ffi.js';
import type { Backend } from './backend.js';

const LIB_EXT: string = { darwin: 'dylib', win32: 'dll' }[process.platform as string] ?? 'so';
const lib = (crate: string): string =>
  new URL(
    `../../../crates/${crate}/target/release/lib${crate.replace('-', '_')}.${LIB_EXT}`,
    import.meta.url,
  ).pathname;

export const CORE_LIB = lib('lenke-core');
export const ENGINE_LIB = lib('lenke-engine');

/** True when the suite is running against the standalone engine (`LENKE_ENGINE=1`). */
export const USE_ENGINE = process.env.LENKE_ENGINE === '1';

/** The native library under test, and whether it's present (skip the suite if not). */
export const NATIVE_LIB = USE_ENGINE ? ENGINE_LIB : CORE_LIB;
export const nativeReady = existsSync(NATIVE_LIB);

/** Build the native backend under test (core by default; engine with LENKE_ENGINE=1). */
export const nativeBackend = (): Backend =>
  USE_ENGINE ? createFfiEngineBackend(ENGINE_LIB) : createFfiBackend(CORE_LIB);

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

/** Whether two result JSON strings are equivalent: byte-identical against core (the
 * shipped guarantee); structural (order-independent) against the engine. */
export const resultsEqual = (a: string, b: string): boolean =>
  USE_ENGINE ? canonResult(a) === canonResult(b) : a === b;
