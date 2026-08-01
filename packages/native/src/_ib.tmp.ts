import { createFfiBackend } from './backend-ffi.js';

const LIB = new URL('../../../crates/lenke-core/target/release/liblenke_core.so', import.meta.url)
  .pathname;
const b = createFfiBackend(LIB);
const enc = new TextEncoder();

const time = (label: string, bytes: number, elems: number, reps: number, f: () => void): void => {
  f();

  const runs: number[] = [];

  for (let i = 0; i < reps; i++) {
    const t0 = Bun.nanoseconds();

    f();
    runs.push(Bun.nanoseconds() - t0);
  }

  runs.sort((x, y) => x - y);

  const ns = runs[Math.floor(runs.length / 2)];

  console.log(
    `${label.padEnd(30)} ${(ns / 1e6).toFixed(0).padStart(5)} ms  ` +
      `${(bytes / (ns / 1e9) / 1024 ** 3).toFixed(3).padStart(6)} GiB/s  ` +
      `${(ns / elems).toFixed(0).padStart(5)} ns/elem`,
  );
};

const nodesOnly = (n: number, props: number): Uint8Array =>
  enc.encode(
    Array.from({ length: n }, (_, i) => {
      const p = Array.from({ length: props }, (_, k) => `"k${k}":"v${i}_${k}"`).join(',');

      return `{"type":"node","id":"v${i}","labels":["Person"],"properties":{${p}}}`;
    }).join('\n'),
  );

const withEdges = (n: number): Uint8Array => {
  const lines = Array.from(
    { length: n },
    (_, i) =>
      `{"type":"node","id":"v${i}","labels":["Person"],"properties":{"name":"n${i}","age":${i % 90}}}`,
  );

  for (let i = 0; i + 1 < n; i++) {
    lines.push(
      `{"type":"edge","id":"e${i}","labels":["KNOWS"],"from":"v${i}","to":"v${i + 1}","properties":{"since":${2000 + (i % 25)}}}`,
    );
  }

  return enc.encode(lines.join('\n'));
};

/** A hub-heavy graph — where growing adjacency lists hurt most. */
const hubGraph = (n: number, hubs: number): Uint8Array => {
  const lines = Array.from(
    { length: n },
    (_, i) => `{"type":"node","id":"v${i}","labels":["Person"],"properties":{"name":"n${i}"}}`,
  );

  for (let i = hubs; i < n; i++) {
    lines.push(
      `{"type":"edge","labels":["KNOWS"],"from":"v${i % hubs}","to":"v${i}","properties":{}}`,
    );
  }

  return enc.encode(lines.join('\n'));
};

console.log('=== decode (serial) ===');
for (const [n, p] of [
  [200_000, 1],
  [200_000, 4],
  [100_000, 16],
] as const) {
  const bytes = nodesOnly(n, p);

  time(`${n} nodes x ${p} props`, bytes.byteLength, n, 5, () => {
    b.graphFree(b.graphFromNdjson(bytes, false));
  });
}

const mixed = withEdges(200_000);

time('200k nodes + 200k edges', mixed.byteLength, 400_000, 5, () => {
  b.graphFree(b.graphFromNdjson(mixed, false));
});

const hubs = hubGraph(200_000, 50);

time('200k nodes, 50 hubs', hubs.byteLength, 400_000, 5, () => {
  b.graphFree(b.graphFromNdjson(hubs, false));
});

console.log('=== decode (parallel) ===');
time('200k nodes + 200k edges', mixed.byteLength, 400_000, 5, () => {
  b.graphFree(b.graphFromNdjson(mixed, true));
});
