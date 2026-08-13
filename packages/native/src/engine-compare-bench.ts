import { fixtureNdjson } from './engine-compare-fixture.js';
// Cross-engine PERF: shipped row-based core vs the standalone columnar engine, same
// graph, same queries, one artifact, through the FFI. The apples-to-apples answer to
// "what did the engine work buy us." Correctness is gated separately by
// `engine-compare.test.ts`; this only times queries whose results already match.
//
// Build the feature'd artifact first, then run:
//   cargo build --release --features engine-compare --manifest-path ../../crates/lenke-core/Cargo.toml
//   bun run src/engine-compare-bench.ts
// Env: BENCH_N (vertices, default 100000), BENCH_DEG (out-degree, default 4),
//      BENCH_REPS (timed reps, min taken, default 7).
import { type Engine, loadCompare } from './engine-compare.js';

const N = Number(process.env.BENCH_N ?? 100_000);
const DEG = Number(process.env.BENCH_DEG ?? 4);
const REPS = Number(process.env.BENCH_REPS ?? 7);

// (feature tag, query). Tags group the summary so a regression is attributable.
const QUERIES: Array<[string, string, 'gremlin' | 'gql']> = [
  ['src-scan', 'g.V().count()', 'gremlin'],
  ['src-label', 'g.V().hasLabel("VIP").count()', 'gremlin'],
  ['has-num', 'g.V().has("age", gt(40)).count()', 'gremlin'],
  ['values', 'g.V().values("age")', 'gremlin'],
  ['id-label', 'g.V().label()', 'gremlin'],
  ['hop-out', 'g.V().out().count()', 'gremlin'],
  ['hop-in', 'g.V().in().count()', 'gremlin'],
  ['hop-both', 'g.V().both().count()', 'gremlin'],
  ['hop-edge', 'g.V().outE().inV().count()', 'gremlin'],
  ['hop2', 'g.V().out().out().count()', 'gremlin'],
  ['dedup', 'g.V().both().dedup().count()', 'gremlin'],
  ['values-hop', 'g.V().out().values("name")', 'gremlin'],
  ['groupcount', 'g.V().groupCount().by("city")', 'gremlin'],
  ['agg', 'g.V().values("score").sum()', 'gremlin'],
  ['order-limit', 'g.V().order().by("age").limit(10).values("name")', 'gremlin'],
  ['repeat', 'g.V().repeat(__.out()).times(2).count()', 'gremlin'],
  ['repeat-dedup', 'g.V().repeat(__.both()).times(2).dedup().count()', 'gremlin'],
  ['where-hop', 'g.V().out().where(__.both()).count()', 'gremlin'],
  ['gql-count', 'MATCH (n:Person) RETURN count(*) AS c', 'gql'],
  ['gql-where', 'MATCH (n:Person) WHERE n.age > 40 RETURN count(*) AS c', 'gql'],
  ['gql-agg', 'MATCH (n:Person) RETURN avg(n.age) AS a', 'gql'],
  ['gql-group', 'MATCH (n:Person) RETURN n.city AS c, count(*) AS n ORDER BY c', 'gql'],
  ['gql-hop', 'MATCH (n:Person)-[e:R]->(m) RETURN count(*) AS c', 'gql'],
  ['gql-order', 'MATCH (n:Person) RETURN n.name AS name ORDER BY n.age DESC LIMIT 10', 'gql'],
];

const pad = (s: string, n: number): string => s.padEnd(n);
const num = (x: number): string => x.toFixed(3).padStart(9);

const time = (fn: () => unknown): number => {
  const t0 = Bun.nanoseconds();

  fn();

  return (Bun.nanoseconds() - t0) / 1e6; // ms
};
const bestOf = (reps: number, fn: () => unknown): number => {
  let best = Infinity;

  for (let i = 0; i < reps; i++) {
    best = Math.min(best, time(fn));
  }

  return best;
};

const main = (): void => {
  const lib = loadCompare();
  process.stdout.write(`building fixture: ${N} vertices, out-degree ${DEG}…\n`);
  const h = lib.fromCoreNdjson(fixtureNdjson(N, DEG));
  const V = h.vertexCount('core');
  const E = h.edgeCount('core');

  if (V !== h.vertexCount('engine') || E !== h.edgeCount('engine')) {
    throw new Error('graphs differ — not comparable');
  }

  process.stdout.write(`graph: ${V} vertices, ${E} edges (both engines identical)\n\n`);

  const run = (e: Engine, q: string, kind: 'gremlin' | 'gql'): unknown =>
    kind === 'gremlin' ? h.gremlin(e, q) : h.gql(e, q);

  const rows: Array<{ tag: string; core: number; engine: number; ratio: number; q: string }> = [];

  for (const [tag, q, kind] of QUERIES) {
    // Warm once per engine (parse cache, first-touch), then time min-of-REPS.
    run('core', q, kind);
    run('engine', q, kind);

    const core = bestOf(REPS, () => run('core', q, kind));
    const engine = bestOf(REPS, () => run('engine', q, kind));

    rows.push({ tag, core, engine, ratio: core / engine, q });
  }

  h.free();
  rows.sort((a, b) => a.ratio - b.ratio);
  process.stdout.write(
    `${pad('ratio', 8)}${pad('core ms', 11)}${pad('engine ms', 11)}${pad('feature', 14)}query\n`,
  );
  process.stdout.write(`${'-'.repeat(90)}\n`);

  for (const r of rows) {
    const flag = r.ratio >= 1 ? ' ' : '!';

    process.stdout.write(
      `${flag}${r.ratio.toFixed(2).padStart(6)}x ${num(r.core)} ${num(r.engine)}  ${pad(r.tag, 14)}${r.q}\n`,
    );
  }

  const wins = rows.filter((r) => r.ratio >= 1).length;
  const gmean = Math.exp(rows.reduce((s, r) => s + Math.log(r.ratio), 0) / rows.length);
  process.stdout.write(`\n${'-'.repeat(90)}\n`);
  process.stdout.write(
    `${wins}/${rows.length} shapes: engine ≥ core. geomean speedup ${gmean.toFixed(2)}x  ` +
      `(>1 = engine faster). worst ${rows[0].tag} ${rows[0].ratio.toFixed(2)}x, ` +
      `best ${rows[rows.length - 1].tag} ${rows[rows.length - 1].ratio.toFixed(2)}x\n`,
  );
};

main();
