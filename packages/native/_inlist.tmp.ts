import { createFfiBackend } from './src/backend-ffi.js';
import { graphFromNdjson } from './src/graph.js';
const LIB = new URL('../../crates/lenke-core/target/release/liblenke_core.so', import.meta.url)
  .pathname;
const b = createFfiBackend(LIB);
const N = 20000;
const g = graphFromNdjson(
  b,
  Array.from(
    { length: N },
    (_, i) =>
      `{"type":"node","id":"u${i}","labels":["User"],"properties":{"name":"user${i}","score":0}}`,
  ).join('\n'),
);
g.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
const names = Array.from({ length: 20 }, (_, k) => `user${k}`);
const literal = names.map((n) => `'${n}'`).join(', ');
const time = (label: string, f: () => void) => {
  for (let i = 0; i < 10; i++) f();
  const t = Bun.nanoseconds();
  for (let i = 0; i < 200; i++) f();
  console.log(
    `${label.padEnd(44)} ${(200 / ((Bun.nanoseconds() - t) / 1e9)).toFixed(0).padStart(7)} stmt/s`,
  );
};
time('IN $param list        (indexed)', () =>
  g.query('MATCH (u:User) WHERE u.name IN $ns RETURN count(*) AS c', { ns: names }),
);
time('IN [literal, ...]     (indexed)', () =>
  g.query(`MATCH (u:User) WHERE u.name IN [${literal}] RETURN count(*) AS c`),
);
time('single = $param       (indexed)', () =>
  g.query('MATCH (u:User) WHERE u.name = $n RETURN count(*) AS c', { n: 'user1' }),
);
g.free();
