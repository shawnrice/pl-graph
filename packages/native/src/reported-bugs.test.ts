// Reproduction for the reported large-integer precision bug: does the Arrow
// path (queryArrow → decodeArrow, ARW1 has only FLOAT64) DIVERGE from the JSON
// query() path on an integer property > 2^53? Run:
//   bun test packages/native/src/reported-bugs.test.ts
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { createFfiBackend } from './backend-ffi.js';
import { decodeArrow, graphFromNdjson } from './graph.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-core/target/release/liblenke_core.${LIB_EXT}`,
  import.meta.url,
).pathname;
const hasLib = existsSync(LIB);

if (!hasLib) {
  console.warn(`[reported-bugs.test] skipping: ${LIB} not found — run \`bun run build:rust\`.`);
}

const suite = hasLib ? describe : describe.skip;

// Two integers that exceed JS's exact-integer range (2^53 = 9007199254740992):
//   2^53 + 1     — the classic "first unrepresentable odd integer"
//   a big 18-digit value
const OVER_53 = '9007199254740993';
const BIG = '123456789012345678';

const newGraph = () => {
  const backend = createFfiBackend(LIB);
  const ndjson = `{"type":"node","id":"a","labels":["N"],"properties":{"amount":${OVER_53},"total":${BIG}}}`;

  return graphFromNdjson(backend, new TextEncoder().encode(ndjson));
};

// =====================================================================
// Bug 4: decodeArrow loses precision on large integers — does it DIVERGE
// from query()? Both native paths ultimately hand back a JS `number`, so
// the question is whether one truncates and the other doesn't.
// =====================================================================
suite('reported-bug #4 · large-int precision: query() vs queryArrow()', () => {
  test('both native paths truncate large integers identically (consistent, not divergent)', () => {
    const g = newGraph();

    const jsonRows = g.query('MATCH (n:N) RETURN n.amount, n.total');
    const arrowRows = decodeArrow(g.queryArrow('MATCH (n:N) RETURN n.amount, n.total'));

    // Divergence check: the two native paths return the SAME (truncated) values.
    expect(arrowRows).toEqual(jsonRows);

    // And to be explicit about WHAT they return: the exact-integer input is
    // already lost by the time either path answers — both give the float64
    // value, not the original 2^53+1 / 18-digit integer.
    expect(jsonRows[0]['n.amount']).toBe(9007199254740992); // 2^53, not 2^53+1
    expect(arrowRows[0]['n.amount']).toBe(9007199254740992);
    expect(jsonRows[0]['n.total']).toBe(arrowRows[0]['n.total']);

    // Neither path preserves the original decimal string, so neither is "more
    // correct" than the other — they lose precision the same way.
    expect(String(jsonRows[0]['n.total'])).not.toBe(BIG);
    expect(String(arrowRows[0]['n.total'])).not.toBe(BIG);

    g.free();
  });
});
