// Shared harness for round6 adversarial fuzzing.
import { existsSync } from 'node:fs';

import { Graph, parseDate, parseDateTime } from '@lenke/core';
import { LenkeError, hasErrorCode } from '@lenke/errors';
import { query as tsQuery } from '@lenke/gql';
import { deserialize as tsDeserialize } from '@lenke/serialization';

import { createFfiBackend } from '../../../packages/native/src/backend-ffi.js';
import { graphFromFormat } from '../../../packages/native/src/graph.js';

const LIB_EXT =
  process.platform === 'darwin' ? 'dylib' : process.platform === 'win32' ? 'dll' : 'so';
const LIB = new URL(
  `../../../crates/lenke-core/target/release/liblenke_core.${LIB_EXT}`,
  import.meta.url,
).pathname;

export const MODERN_NDJSON = [
  '{"type":"node","id":"1","labels":["Person"],"properties":{"name":"marko","age":29}}',
  '{"type":"node","id":"2","labels":["Person"],"properties":{"name":"vadas","age":27}}',
  '{"type":"node","id":"4","labels":["Person"],"properties":{"name":"josh","age":32}}',
  '{"type":"node","id":"3","labels":["Software"],"properties":{"name":"lop","lang":"java"}}',
  '{"type":"node","id":"5","labels":["Software"],"properties":{"name":"ripple","lang":"java"}}',
  '{"type":"node","id":"6","labels":["Person"],"properties":{"name":"peter","age":35}}',
  '{"type":"edge","id":"7","from":"1","to":"2","labels":["KNOWS"],"properties":{"weight":0.5,"since":2018}}',
  '{"type":"edge","id":"8","from":"1","to":"4","labels":["KNOWS"],"properties":{"weight":1.0,"since":2020}}',
  '{"type":"edge","id":"9","from":"1","to":"3","labels":["CREATED"],"properties":{"weight":0.4,"since":2009}}',
  '{"type":"edge","id":"10","from":"4","to":"5","labels":["CREATED"],"properties":{"weight":1.0,"since":2011}}',
  '{"type":"edge","id":"11","from":"4","to":"3","labels":["CREATED"],"properties":{"weight":0.4,"since":2012}}',
  '{"type":"edge","id":"12","from":"6","to":"3","labels":["CREATED"],"properties":{"weight":0.2,"since":2017}}',
].join('\n');

export const backend = createFfiBackend(LIB);
export const nativeGraph = graphFromNdjson(backend, MODERN_NDJSON);
export const tsGraph = tsDeserialize(MODERN_NDJSON, 'ndjson', new Graph());

export { Graph, parseDate, parseDateTime, tsQuery, tsDeserialize, LenkeError, hasErrorCode };

// Classify an outcome from a thunk.
export type Outcome =
  | { kind: 'ok'; value: unknown }
  | { kind: 'coded'; code: string; msg: string }
  | { kind: 'uncoded'; name: string; msg: string };

export const classify = (fn: () => unknown): Outcome => {
  try {
    return { kind: 'ok', value: fn() };
  } catch (e) {
    if (e instanceof LenkeError) {
      return {
        kind: 'coded',
        code: (e as any).code,
        msg: String((e as any).message).slice(0, 200),
      };
    }
    // Some LenkeErrors may cross realms; check code property duck-typed.
    if (
      e &&
      typeof e === 'object' &&
      'code' in (e as any) &&
      String((e as any).code).startsWith('E_')
    ) {
      return {
        kind: 'coded',
        code: (e as any).code,
        msg: String((e as any).message).slice(0, 200),
      };
    }
    return {
      kind: 'uncoded',
      name: e && (e as any).constructor ? (e as any).constructor.name : typeof e,
      msg: String((e as any)?.message ?? e).slice(0, 200),
    };
  }
};

// Run one GQL query on both engines and diff.
export type Diff =
  | { status: 'match'; out: string }
  | { status: 'both-error'; tsCode: string; nativeCode: string; sameCode: boolean }
  | { status: 'divergence'; ts: Outcome; native: Outcome };

export const runBoth = (q: string, params?: Record<string, unknown>): Diff => {
  const ts = classify(() => tsQuery(tsGraph, q, params));
  const native = classify(() => nativeGraph.query(q, params));
  if (ts.kind === 'ok' && native.kind === 'ok') {
    const a = JSON.stringify((ts as any).value);
    const b = JSON.stringify((native as any).value);
    if (a === b) return { status: 'match', out: a };
    return { status: 'divergence', ts, native };
  }
  if (ts.kind !== 'ok' && native.kind !== 'ok') {
    const tsCode = ts.kind === 'coded' ? ts.code : `UNCODED:${(ts as any).name}`;
    const nativeCode = native.kind === 'coded' ? native.code : `UNCODED:${(native as any).name}`;
    return { status: 'both-error', tsCode, nativeCode, sameCode: tsCode === nativeCode };
  }
  return { status: 'divergence', ts, native };
};
