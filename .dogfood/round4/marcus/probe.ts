import {
  graphFromNdjson,
} from '@lenke/native';
import { createNodeBackend } from '@lenke/node/backend';

// small directed graph: a->b, a->c, b->c, c->a  (a cycle for pagerank)
const lines = [
  '{"type":"node","id":"a","labels":["N"],"properties":{"name":"a"}}',
  '{"type":"node","id":"b","labels":["N"],"properties":{"name":"b"}}',
  '{"type":"node","id":"c","labels":["N"],"properties":{"name":"c"}}',
  '{"type":"edge","id":"e1","from":"a","to":"b","labels":["L"],"properties":{}}',
  '{"type":"edge","id":"e2","from":"a","to":"c","labels":["L"],"properties":{}}',
  '{"type":"edge","id":"e3","from":"b","to":"c","labels":["L"],"properties":{}}',
  '{"type":"edge","id":"e4","from":"c","to":"a","labels":["L"],"properties":{}}',
];
const backend = createNodeBackend();
const g = graphFromNdjson(backend, new TextEncoder().encode(lines.join('\n')));

function attempt(label: string, fn: () => unknown) {
  try {
    const r = fn();
    console.log(`OK  ${label}:`, JSON.stringify(r)?.slice(0, 300));
  } catch (e) {
    console.log(`ERR ${label}:`, (e as Error).message.split('\n')[0]);
  }
}

// 1. Init pr + outdeg via SET with subquery/aggregation
attempt('SET literal', () => g.query`MATCH (n:N) SET n.pr = 1.0`);
attempt(
  'SET outdeg via COUNT subquery',
  () => g.query`MATCH (n:N) SET n.outdeg = COUNT { (n)-[:L]->() }`,
);
attempt(
  'read after set',
  () => g.query`MATCH (n:N) RETURN n.name AS name, n.pr AS pr, n.outdeg AS outdeg`,
);

// 2. One PageRank iteration: SET n.pr from neighbor aggregation
attempt(
  'SET from WITH-aggregation over neighbors',
  () =>
    g.query`MATCH (m:N)-[:L]->(n:N) WITH n, sum(m.pr / m.outdeg) AS inc SET n.pr = 0.15 + 0.85 * inc`,
);
attempt(
  'SET via correlated subquery scalar',
  () => g.query`MATCH (n:N) SET n.pr = 0.15 + 0.85 * (COUNT { (n)<-[:L]-() })`,
);

// 3. WITH aggregation then RETURN (whole graph sum)
attempt(
  'whole-graph WITH aggregate',
  () => g.query`MATCH (n:N) WITH sum(n.pr) AS total, count(*) AS c RETURN total, c`,
);

// 4. Gremlin repeat propagation attempt (min-label components)
attempt('gremlin repeat count', () => g.gremlin`g.V().repeat(out()).times(2).count()`);
attempt('gremlin groupCount by outdeg', () => g.gremlin`g.V().group().by(label()).by(count())`);
attempt('gremlin sack', () => g.gremlin`g.withSack(1.0).V().sack().sum()`);
attempt('gremlin math', () => g.gremlin`g.V().values('pr').sum()`);

g.free();
