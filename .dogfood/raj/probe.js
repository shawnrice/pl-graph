import {
  graphFromNdjson,
} from '@lenke/native';
import { createNodeBackend } from '@lenke/node/backend';
const backend = createNodeBackend();
const nd = Buffer.from(
  '{"type":"node","id":"p1","labels":["Person"],"properties":{"uid":1}}\n',
  'utf8',
);
const g = graphFromNdjson(backend, nd);
const tryIt = (label, fn) => {
  try {
    const r = fn();
    console.log(label, '=> OK', JSON.stringify(r));
  } catch (e) {
    console.log(
      label,
      '=> THROW:',
      e.constructor.name,
      '| code=',
      e.code,
      '|',
      String(e.message).slice(0, 160),
    );
  }
};
tryIt('syntax error', () => g.query('MATCH (p:Person RETURN p'));
tryIt('unknown fn', () => g.query('MATCH (p:Person) RETURN frobnicate(p.uid)'));
tryIt('missing param', () => g.query('MATCH (p:Person {uid:$uid}) RETURN p.uid'));
tryIt('use-after-free', () => {
  const p = g.prepare('MATCH (p:Person) RETURN count(*) AS n');
  g.free();
  return p.query();
});
