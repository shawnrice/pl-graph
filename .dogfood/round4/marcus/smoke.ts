import {
  decodeArrow,
  graphFromNdjson,
} from '@lenke/native';
import { createNodeBackend } from '@lenke/node/backend';

const ndjson = [
  '{"type":"node","id":"a","labels":["N"],"properties":{"name":"a","w":1.0}}',
  '{"type":"node","id":"b","labels":["N"],"properties":{"name":"b","w":2.0}}',
  '{"type":"node","id":"c","labels":["N"],"properties":{"name":"c","w":3.0}}',
  '{"type":"edge","id":"e1","from":"a","to":"b","labels":["L"],"properties":{"weight":0.5}}',
  '{"type":"edge","id":"e2","from":"a","to":"c","labels":["L"],"properties":{"weight":1.5}}',
  '{"type":"edge","id":"e3","from":"b","to":"c","labels":["L"],"properties":{"weight":2.5}}',
].join('\n');

const backend = createNodeBackend();
const g = graphFromNdjson(backend, new TextEncoder().encode(ndjson));
console.log('vertexCount', g.vertexCount, 'edgeCount', g.edgeCount);

// GQL rows
const rows = g.query`MATCH (n:N) RETURN n.name AS name, n.w AS w`;
console.log('gql rows', rows);

// degree via GQL aggregation
const deg = g.query`MATCH (n:N)-[e:L]->(m) RETURN element_id(n) AS id, count(*) AS outdeg`;
console.log('outdeg', deg);

// Arrow scalar columns
const blob = g.queryArrow`MATCH (n:N) RETURN n.name AS name, n.w AS w`;
console.log('arrow blob bytes', blob.length);
console.log('decoded', decodeArrow(blob));

// Gremlin
try {
  const gr = g.gremlin`g.V().hasLabel('N').count()`;
  console.log('gremlin count', gr);
} catch (e) {
  console.log('gremlin count ERR', (e as Error).message);
}

g.free();
