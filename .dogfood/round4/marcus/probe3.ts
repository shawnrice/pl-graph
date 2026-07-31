import {
  graphFromNdjson,
} from '@lenke/native';
import { createNodeBackend } from '@lenke/node/backend';

// two components: {a,b,c} chain, {d,e}
const lines = [
  '{"type":"node","id":"a","labels":["N"],"properties":{"cid":0}}',
  '{"type":"node","id":"b","labels":["N"],"properties":{"cid":1}}',
  '{"type":"node","id":"c","labels":["N"],"properties":{"cid":2}}',
  '{"type":"node","id":"d","labels":["N"],"properties":{"cid":3}}',
  '{"type":"node","id":"e","labels":["N"],"properties":{"cid":4}}',
  '{"type":"edge","id":"e1","from":"a","to":"b","labels":["L"],"properties":{}}',
  '{"type":"edge","id":"e2","from":"b","to":"c","labels":["L"],"properties":{}}',
  '{"type":"edge","id":"e3","from":"d","to":"e","labels":["L"],"properties":{}}',
];
const backend = createNodeBackend();
const g = graphFromNdjson(backend, new TextEncoder().encode(lines.join('\n')));

function attempt(label: string, fn: () => unknown) {
  try {
    const r = fn();
    console.log(`OK  ${label}:`, JSON.stringify(r)?.slice(0, 200));
  } catch (e) {
    console.log(`ERR ${label}:`, (e as Error).message.split('\n')[0]);
  }
}

// Connected components via min-label propagation (undirected => use both directions).
// init cid already set to unique ints. Propagate min across undirected neighbors.
for (let it = 0; it < 10; it++) {
  // neighbor min in each direction, plus self
  g.query`MATCH (n:N)-[:L]->(m:N) WITH m, min(n.cid) AS nc SET m.cid = CASE WHEN nc < m.cid THEN nc ELSE m.cid END`;
  g.query`MATCH (n:N)-[:L]->(m:N) WITH n, min(m.cid) AS nc SET n.cid = CASE WHEN nc < n.cid THEN nc ELSE n.cid END`;
}
attempt(
  'components',
  () => g.query`MATCH (n:N) RETURN element_id(n) AS id, n.cid AS cid ORDER BY id`,
);

// Label propagation majority vote — is there a mode/argmax aggregate?
attempt(
  'mode aggregate?',
  () => g.query`MATCH (n:N)-[:L]->(m:N) WITH m, mode(n.cid) AS lbl RETURN m, lbl`,
);
attempt(
  'collect_list of neighbor labels',
  () =>
    g.query`MATCH (n:N)-[:L]->(m:N) WITH m, collect_list(n.cid) AS labels RETURN element_id(m) AS id, labels ORDER BY id`,
);

g.free();
