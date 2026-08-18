// Price of the present-null (nulls side-map) feature. Times the COMMON cases (null-free
// reads/writes — must not regress) and the WIN cases (a numeric column that carries a
// null: before this change one null de-opted the WHOLE column to Gen forever). Run the
// engine cdylib as-is, then `git stash` the engine + rebuild for the BEFORE numbers.
//   bun run present-null-bench.ts
import { createFfiEngineBackend } from './src/backend-ffi-engine.js';

const LIB_EXT = { darwin: 'dylib', win32: 'dll' }[process.platform as string] ?? 'so';
const ENGINE_LIB = new URL(
  `../../crates/lenke-engine/target/release/liblenke_engine.${LIB_EXT}`,
  import.meta.url,
).pathname;
const backend = createFfiEngineBackend(ENGINE_LIB);
const enc = new TextEncoder();

const N = 100_000;
const ndjson = (): Uint8Array => {
  const lines: string[] = [];

  for (let i = 0; i < N; i++) {
    lines.push(
      `{"type":"node","id":"n${i}","labels":["P"],"properties":{"id":"n${i}","age":${i % 1000},"city":"c${i % 50}"}}`,
    );
  }

  return enc.encode(lines.join('\n'));
};
const DOC = ndjson();

const REPS = 9;
// Rebuild the graph fresh for each timed op so writes don't accumulate across reps.
const time = (label: string, setup: string[], query: string): void => {
  let best = Infinity;

  for (let r = 0; r < REPS; r++) {
    const h = backend.graphFromNdjson(DOC, false);

    try {
      backend.createIndex(h, 'vertex', 'hash', ['age']);

      for (const s of setup) {
        backend.queryRows(h, s, '{}');
      }

      const t = performance.now();
      backend.queryRows(h, query, '{}');
      best = Math.min(best, performance.now() - t);
    } finally {
      backend.graphFree(h);
    }
  }

  console.log(`  ${label.padEnd(46)} ${best.toFixed(2)} ms`);
};

// A batch-write throughput: time applying `writes` (already the timed op).
const timeWrite = (label: string, write: string): void => {
  let best = Infinity;

  for (let r = 0; r < REPS; r++) {
    const h = backend.graphFromNdjson(DOC, false);

    try {
      const t = performance.now();
      backend.queryRows(h, write, '{}');
      best = Math.min(best, performance.now() - t);
    } finally {
      backend.graphFree(h);
    }
  }

  console.log(`  ${label.padEnd(46)} ${best.toFixed(2)} ms`);
};

console.log(`\npresent-null feature — ${N} nodes, min of ${REPS}\n`);

console.log('COST — the common (null-free) case must not regress:');
time('null-free  min(age) full scan', [], 'MATCH (n:P) RETURN min(n.age) AS m');
time('null-free  age = 500 (index seek)', [], 'MATCH (n:P) WHERE n.age = 500 RETURN n.id AS x');
time(
  'null-free  PROPERTY_EXISTS(age) count',
  [],
  'MATCH (n:P) WHERE PROPERTY_EXISTS(n, age) RETURN count(*) AS c',
);
timeWrite('write     SET age = age+1 (all rows)', 'MATCH (n:P) SET n.age = n.age + 1');
timeWrite('write     SET flag = null (all rows)', 'MATCH (n:P) SET n.flag = null');

console.log('\nWIN — a numeric column that carries ONE present null:');
// One null in `age`. BEFORE: `age` becomes Gen forever → every query below is boxed.
// AFTER: `age` stays Num → selective/self-healed queries read typed.
const oneNull = ["MATCH (n:P {id:'n7'}) SET n.age = null"];
time(
  'null-bearing  age = 500 (selective, avoids null)',
  oneNull,
  'MATCH (n:P) WHERE n.age = 500 RETURN n.id AS x',
);
time('null-bearing  min(age) full scan', oneNull, 'MATCH (n:P) RETURN min(n.age) AS m');
// self-heal: set the null, then a real value back — column should be typed-fast again.
const healed = ["MATCH (n:P {id:'n7'}) SET n.age = null", "MATCH (n:P {id:'n7'}) SET n.age = 7"];
time(
  'self-healed   age = 500 after null removed',
  healed,
  'MATCH (n:P) WHERE n.age = 500 RETURN n.id AS x',
);
console.log('');
