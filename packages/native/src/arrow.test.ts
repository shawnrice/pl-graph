// Proof that `@lenke/native/arrow` emits real, conformant Apache Arrow IPC: build
// an ARW1 blob from a live query, then round-trip BOTH IPC layouts (stream +
// file/Feather) through `apache-arrow`'s reference decoder — the same decoder
// pandas / Polars / DuckDB use. `apache-arrow` is a dev-only verifier here; the
// module itself hand-writes the IPC framing with zero runtime deps. Loads the real
// FFI cdylib by path.
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { type Table, tableFromIPC } from 'apache-arrow';

import { toArrowIPC } from './arrow.js';
import { createFfiBackend } from './backend-ffi.js';
import { graphFromFormat } from './graph.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-core/target/release/liblenke_core.${LIB_EXT}`,
  import.meta.url,
).pathname;

const hasLib = existsSync(LIB);

if (!hasLib) {
  console.warn(`[arrow.test] skipping: ${LIB} not found — run \`bun run build:rust\` first.`);
}

const suite = hasLib ? describe : describe.skip;

const NDJSON = [
  '{"type":"node","id":"a","labels":["P"],"properties":{"name":"marko","age":29,"active":true}}',
  '{"type":"node","id":"b","labels":["P"],"properties":{"name":"vadas","age":27}}',
  '{"type":"node","id":"c","labels":["P"],"properties":{"name":"josh","age":32,"active":false}}',
].join('\n');

suite('@lenke/native/arrow — real Arrow IPC egress', () => {
  const backend = createFfiBackend(LIB);
  const blobOf = (q: string): Uint8Array => {
    const g = graphFromFormat(backend, NDJSON, 'ndjson');

    try {
      return g.queryArrow(q);
    } finally {
      g.free();
    }
  };

  const QUERY =
    'MATCH (n:P) RETURN n.name AS name, n.age AS age, n.active AS active ORDER BY n.age';
  const EXPECTED = [
    { name: 'vadas', age: 27, active: null },
    { name: 'marko', age: 29, active: true },
    { name: 'josh', age: 32, active: false },
  ];

  test('the reconstructed schema is typed Utf8/Float64/Bool', () => {
    const t = tableFromIPC(toArrowIPC(blobOf(QUERY), 'stream'));

    expect(t.numRows).toBe(3);
    expect(t.schema.fields.map((f) => `${f.name}:${f.type}`)).toEqual([
      'name:Utf8',
      'age:Float64',
      'active:Bool',
    ]);
  });

  test('IPC stream round-trips through the reference decoder', () => {
    const back = tableFromIPC(toArrowIPC(blobOf(QUERY), 'stream'));

    expect([...back].map((r) => ({ name: r.name, age: r.age, active: r.active }))).toEqual(
      EXPECTED,
    );
  });

  test('IPC file / Feather layout round-trips and carries the ARROW1 magic', () => {
    const ipc = toArrowIPC(blobOf(QUERY), 'file');

    expect(new TextDecoder().decode(ipc.subarray(0, 6))).toBe('ARROW1');

    const back = tableFromIPC(ipc);
    expect([...back].map((r) => ({ name: r.name, age: r.age, active: r.active }))).toEqual(
      EXPECTED,
    );
  });

  test('default format is the IPC stream (no ARROW1 file magic)', () => {
    const ipc = toArrowIPC(blobOf(QUERY));

    expect(new TextDecoder().decode(ipc.subarray(0, 6))).not.toBe('ARROW1');
    expect(tableFromIPC(ipc).numRows).toBe(3);
  });

  test('an all-null column survives the round-trip', () => {
    for (const fmt of ['stream', 'file'] as const) {
      const back = tableFromIPC(toArrowIPC(blobOf('MATCH (n:P) RETURN n.dept AS dept'), fmt));
      expect([...back].map((r) => r.dept)).toEqual([null, null, null]);
    }
  });

  test('a zero-row result round-trips (empty buffers, valid framing)', () => {
    for (const fmt of ['stream', 'file'] as const) {
      const back = tableFromIPC(
        toArrowIPC(
          blobOf('MATCH (n:P) WHERE n.age > 999 RETURN n.name AS name, n.age AS age'),
          fmt,
        ),
      );
      expect(back.numRows).toBe(0);
      expect(back.schema.fields.map((f) => f.name)).toEqual(['name', 'age']);
    }
  });

  test('a single numeric column round-trips', () => {
    const back = tableFromIPC(toArrowIPC(blobOf('MATCH (n:P) RETURN n.age AS age ORDER BY n.age')));
    expect([...back].map((r) => r.age)).toEqual([27, 29, 32]);
  });

  test('unicode strings survive (multi-byte offsets)', () => {
    const g = graphFromFormat(
      backend,
      [
        '{"type":"node","id":"a","labels":["P"],"properties":{"name":"café ☕"}}',
        '{"type":"node","id":"b","labels":["P"],"properties":{"name":"日本語"}}',
        '{"type":"node","id":"c","labels":["P"],"properties":{"name":"marko`s"}}',
      ].join('\n'),
      'ndjson',
    );

    try {
      const back = tableFromIPC(toArrowIPC(g.queryArrow('MATCH (n:P) RETURN n.name AS name')));
      expect([...back].map((r) => r.name).sort()).toEqual(['café ☕', 'marko`s', '日本語'].sort());
    } finally {
      g.free();
    }
  });

  test('a wide result past the builder growth threshold round-trips', () => {
    // >1k rows and many string columns forces the flatbuffer builder to grow its
    // backing buffer and re-pad — exercises the grow path, not just the small case.
    const rows = Array.from(
      { length: 1500 },
      (_, i) =>
        `{"type":"node","id":"n${i}","labels":["P"],"properties":{"name":"user_${i}","age":${i % 90},"flag":${i % 2 === 0}}}`,
    ).join('\n');
    const g = graphFromFormat(backend, rows, 'ndjson');

    try {
      for (const fmt of ['stream', 'file'] as const) {
        const back = tableFromIPC(
          toArrowIPC(
            g.queryArrow(
              'MATCH (n:P) RETURN n.name AS name, n.age AS age, n.flag AS flag ORDER BY n.age, n.name',
            ),
            fmt,
          ),
        );
        expect(back.numRows).toBe(1500);
        // Spot-check a row deep in the batch (exercises a mid-buffer read).
        const r = [...back].at(777)!;
        expect(typeof r.name).toBe('string');
        expect(typeof r.age).toBe('number');
        expect(typeof r.flag).toBe('boolean');
      }
    } finally {
      g.free();
    }
  });

  test('rejects a non-ARW1 blob', () => {
    expect(() => toArrowIPC(new Uint8Array([1, 2, 3, 4]))).toThrow();
  });

  // The native one-shot: RustGraph.queryArrowIpc runs query→IPC entirely in Rust
  // (no JS transcode). It must produce byte-identical IPC to the JS encoder AND
  // read back through the reference decoder.
  test('native queryArrowIpc matches the JS encoder byte-for-byte and decodes', () => {
    const g = graphFromFormat(backend, NDJSON, 'ndjson');

    try {
      for (const format of ['stream', 'file'] as const) {
        const native = g.queryArrowIpc(QUERY, { format });
        const js = toArrowIPC(g.queryArrow(QUERY), format);

        // Copy into plain ArrayBuffer-backed views: `Buffer`/native blobs are typed
        // `Uint8Array<ArrayBufferLike>`, which `Buffer.compare` (wanting
        // `Uint8Array<ArrayBuffer>`) rejects under TS's generic-typed-array lib.
        expect(Buffer.compare(new Uint8Array(native), new Uint8Array(js))).toBe(0);
        expect(
          [...tableFromIPC(native)].map((r) => ({ name: r.name, age: r.age, active: r.active })),
        ).toEqual(EXPECTED);
      }
    } finally {
      g.free();
    }
  });

  test('native queryArrowIpc binds params (tagged-template form is stream)', () => {
    const g = graphFromFormat(backend, NDJSON, 'ndjson');

    try {
      const ipc = g.queryArrowIpc`MATCH (n:P) WHERE n.age > ${28} RETURN n.name AS name ORDER BY n.name`;
      const back = tableFromIPC(ipc);

      expect(new TextDecoder().decode(ipc.subarray(0, 6))).not.toBe('ARROW1'); // stream
      expect([...back].map((r) => r.name)).toEqual(['josh', 'marko']);
    } finally {
      g.free();
    }
  });
});

// A fixed-dim numeric-list column egresses as a real Arrow FixedSizeList<Float64>
// (not Utf8 JSON) — the feature-matrix egress the graph-ML use case wants.
const VEC_NDJSON = [
  '{"type":"node","id":"a","labels":["V"],"properties":{"name":"a","h":[1.5,2.5,3.5]}}',
  '{"type":"node","id":"b","labels":["V"],"properties":{"name":"b","h":[4,5,6]}}',
  '{"type":"node","id":"c","labels":["V"],"properties":{"name":"c"}}', // no h → null list
].join('\n');

suite('@lenke/native/arrow — FixedSizeList<Float64> egress', () => {
  const backend = createFfiBackend(LIB);
  const Q = 'MATCH (n:V) RETURN n.h AS h ORDER BY n.name';

  const table = (native: boolean): Table => {
    const g = graphFromFormat(backend, VEC_NDJSON, 'ndjson');

    try {
      return tableFromIPC(
        native ? g.queryArrowIpc(Q, { format: 'stream' }) : toArrowIPC(g.queryArrow(Q)),
      );
    } finally {
      g.free();
    }
  };

  const readRows = (t: Table) =>
    [...t].map((r) => {
      const v = r.h as { toArray(): Float64Array } | null;

      return v === null ? null : [...v.toArray()];
    });

  const EXPECTED = [
    [1.5, 2.5, 3.5],
    [4, 5, 6],
    null, // c had no `h` → a null list, not a zero vector
  ];

  for (const native of [true, false]) {
    test(`${native ? 'native' : 'JS'} IPC decodes as FixedSizeList<Float64>[3]`, () => {
      const t = table(native);

      expect(String(t.schema.fields[0].type)).toContain('FixedSizeList');
      expect(String(t.schema.fields[0].type.children[0].type)).toBe('Float64');
      expect(readRows(t)).toEqual(EXPECTED);
    });
  }

  test('native and JS encoders are byte-for-byte identical', () => {
    const g = graphFromFormat(backend, VEC_NDJSON, 'ndjson');

    try {
      for (const format of ['stream', 'file'] as const) {
        const nat = g.queryArrowIpc(Q, { format });
        const js = toArrowIPC(g.queryArrow(Q), format);
        expect(Buffer.compare(new Uint8Array(nat), new Uint8Array(js))).toBe(0);
      }
    } finally {
      g.free();
    }
  });
});
