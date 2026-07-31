import {
  graphFromNdjson,
} from '@lenke/native';
import { createNodeBackend } from '@lenke/node/backend';

const nd = [
  '{"type":"node","id":"1","labels":["Person"],"properties":{"name":"marko","age":29}}',
  '{"type":"node","id":"2","labels":["Person"],"properties":{"name":"josh","age":32}}',
  '{"type":"edge","id":"7","from":"1","to":"2","labels":["KNOWS"],"properties":{"weight":1.0}}',
].join('\n');
const backend = createNodeBackend();

function check(label: string, fn: () => unknown) {
  try {
    console.log(`OK  ${label}:`, JSON.stringify(fn())?.slice(0, 120));
  } catch (e) {
    console.log(`ERR ${label}:`, (e as Error).message.split('\n')[0]);
  }
}

// native.md: `using g = graphFromNdjson(...)` disposal
{
  using g = graphFromNdjson(backend, new TextEncoder().encode(nd));
  check('native.md tagged query', () => g.query`MATCH (p:Person) RETURN p.name AS name`);
  check('native.md param query', () =>
    g.query('MATCH (p:Person) WHERE p.age > $min RETURN p.name', { min: 30 }),
  );
  check('native.md gremlin tagged', () => g.gremlin`g.V().has('name', ${'marko'}).values('age')`);
  // gql README claim: `RETURN count(*) AS count` should FAIL (reserved word)
  check(
    'gql README: AS count should FAIL',
    () => g.query`MATCH (p:Person) RETURN count(*) AS count`,
  );
  check('gql README: AS `count` works', () =>
    g.query('MATCH (p:Person) RETURN count(*) AS `count`'),
  );
  // gql README divergence: substring 1-based
  check('gql README substring 1-based', () => g.query`RETURN substring('crystal', 1, 3) AS s`);
}
console.log('(using-block disposed without error)');
