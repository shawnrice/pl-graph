import { readFile } from 'node:fs/promises';

import {
  decodeArrow,
  graphFromNdjson,
} from '@lenke/native';
import { createNodeBackend } from '@lenke/node/backend';
const bytes = await readFile(new URL('./org-graph.ndjson', import.meta.url).pathname);
using g = graphFromNdjson(createNodeBackend(), bytes);
const q = `MATCH (p:Person) WHERE p.team='Brand' RETURN p.dept AS dept, collect_list(p.name) AS names`;
const j = g.query(q);
const a = decodeArrow(g.queryArrow(q));
console.log('list-col JSON===Arrow parity?', JSON.stringify(j) === JSON.stringify(a));
console.log(
  'JSON names isArray:',
  Array.isArray((j[0] as any).names),
  '| Arrow names typeof:',
  typeof (a[0] as any).names,
);
// no error was thrown -> silent lossy coercion
