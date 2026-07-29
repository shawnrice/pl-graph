// Proves the native crate is callable from Bun over FFI (the bun:ffi half of
// "shared crate, both bindings"): loads a tiny graph through the C ABI and reads
// its counts back. Run: bun test crates/lenke-core/ffi-smoke.test.ts
import { dlopen, FFIType, ptr } from 'bun:ffi';
import { describe, expect, test } from 'bun:test';

const ext = process.platform === 'darwin' ? 'dylib' : 'so';
const libPath = new URL(`./target/release/liblenke_core.${ext}`, import.meta.url).pathname;

const lib = dlopen(libPath, {
  lnk_abi_version: { args: [], returns: FFIType.u32 },
  lnk_graph_from_ndjson: {
    args: [FFIType.ptr, FFIType.u64_fast, FFIType.u32],
    returns: FFIType.ptr,
  },
  lnk_graph_vertex_count: { args: [FFIType.ptr], returns: FFIType.u64_fast },
  lnk_graph_edge_count: { args: [FFIType.ptr], returns: FFIType.u64_fast },
  lnk_graph_free: { args: [FFIType.ptr], returns: FFIType.void },
});

describe('lenke-core over bun:ffi', () => {
  test('abi version probe', () => {
    expect(lib.symbols.lnk_abi_version()).toBe(17);
  });

  test('graph round-trips through the C ABI: load ndjson, read counts back', () => {
    // Two vertices, one edge, in the crate's ndjson load format (see datagen.ts).
    const ndjson = new TextEncoder().encode(
      [
        JSON.stringify({ type: 'node', id: 'n0', labels: ['Person'], properties: {} }),
        JSON.stringify({ type: 'node', id: 'n1', labels: ['Person'], properties: {} }),
        JSON.stringify({
          type: 'edge',
          id: 'e0',
          from: 'n0',
          to: 'n1',
          labels: ['KNOWS'],
          properties: {},
        }),
      ].join('\n'),
    );

    const handle = lib.symbols.lnk_graph_from_ndjson(ptr(ndjson), BigInt(ndjson.length), 0);
    expect(handle).not.toBe(0);
    expect(Number(lib.symbols.lnk_graph_vertex_count(handle))).toBe(2);
    expect(Number(lib.symbols.lnk_graph_edge_count(handle))).toBe(1);
    lib.symbols.lnk_graph_free(handle);
  });
});
