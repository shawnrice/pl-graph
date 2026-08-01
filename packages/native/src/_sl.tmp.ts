import { createFfiBackend } from './backend-ffi.js';
const LIB = new URL('../../../crates/lenke-core/target/release/liblenke_core.so', import.meta.url).pathname;
const b = createFfiBackend(LIB);
const enc = new TextEncoder();
const time = (label: string, reps: number, f: () => void): void => {
  f();
  const runs: number[] = [];
  for (let i = 0; i < reps; i++) { const t0 = Bun.nanoseconds(); f(); runs.push(Bun.nanoseconds() - t0); }
  runs.sort((x, y) => x - y);
  console.log(`${label.padEnd(34)} ${(runs[Math.floor(runs.length / 2)] / 1e6).toFixed(0).padStart(5)} ms`);
};
for (const len of [8, 40, 200, 1000]) {
  const n = Math.max(2000, Math.floor(4_000_000 / len));
  const doc = enc.encode(Array.from({ length: n }, (_, i) =>
    `{"type":"node","id":"v${i}","labels":["P"],"properties":{"t":"${String(i).padStart(len, 'x')}"}}`).join('\n'));
  time(`${String(n).padStart(6)} nodes x ${String(len).padStart(4)}B value`, 5, () => {
    b.graphFree(b.graphFromNdjson(doc, false));
  });
}
