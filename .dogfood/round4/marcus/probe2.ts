import {
  graphFromNdjson,
} from '@lenke/native';
import { createNodeBackend } from '@lenke/node/backend';

// a->b, a->c, b->c, c->a
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

const N = 3;
const d = 0.85;

// init
g.query`MATCH (n:N) SET n.pr = ${1.0 / N}`;
g.query`MATCH (n:N) SET n.outdeg = COUNT { (n)-[:L]->() }`;

for (let it = 0; it < 30; it++) {
  // double-buffer: compute pr_new from pr (snapshot), reset base then add contributions
  g.query`MATCH (n:N) SET n.pr_new = ${(1 - d) / N}`;
  // add teleport from dangling? no dangling here.
  g.query`MATCH (m:N)-[:L]->(n:N) WITH n, sum(m.pr / m.outdeg) AS inc SET n.pr_new = n.pr_new + ${d} * inc`;
  g.query`MATCH (n:N) SET n.pr = n.pr_new`;
}
const gqlPr = g.query`MATCH (n:N) RETURN n.name AS name, n.pr AS pr ORDER BY name`;
console.log('GQL PageRank:', gqlPr);

// JS reference
const adj: Record<string, string[]> = { a: ['b', 'c'], b: ['c'], c: ['a'] };
const nodes = ['a', 'b', 'c'];
let pr: Record<string, number> = { a: 1 / N, b: 1 / N, c: 1 / N };
for (let it = 0; it < 30; it++) {
  const next: Record<string, number> = {};
  for (const n of nodes) next[n] = (1 - d) / N;
  for (const m of nodes) for (const t of adj[m]) next[t] += (d * pr[m]) / adj[m].length;
  pr = next;
}
console.log(
  'JS  PageRank:',
  nodes.map((n) => ({ name: n, pr: pr[n] })),
);

g.free();
