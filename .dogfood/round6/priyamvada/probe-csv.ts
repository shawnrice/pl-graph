import { serialize as tsSerialize } from '@lenke/serialization';

import { graphFromFormat } from '../../../packages/native/src/graph.js';
import { tsDeserialize, Graph, classify, backend } from './harness.ts';
const both = (input: string) => {
  const t = classify(() => {
    const g = tsDeserialize(input, 'csv', new Graph());
    return tsSerialize(g, 'ndjson').trim();
  });
  const n = classify(() => graphFromFormat(backend, input, { format: 'csv' }).serialize('ndjson').trim());
  const tv = t.kind === 'ok' ? (t as any).value : `${t.kind}:${(t as any).code || (t as any).name}`;
  const nv = n.kind === 'ok' ? (n as any).value : `${n.kind}:${(n as any).code || (n as any).name}`;
  const mark = tv === nv ? 'MATCH' : '*** DIVERGE ***';
  console.log(`${mark}\n  IN : ${JSON.stringify(input)}\n  TS : ${tv}\n  NAT: ${nv}`);
};
// quoted value in last column
both(':ID,:LABEL,name\n1,A,"a,b"');
// quoted value NOT in last column
both(':ID,:LABEL,name,city\n1,A,"a,b",NYC');
// no quotes at all
both(':ID,:LABEL,name\n1,A,hello');
// quoted value with no comma inside
both(':ID,:LABEL,name\n1,A,"hello"');
// quoted middle column, single col name
both(':ID,:LABEL,x,name\n1,A,"q,q",bob');
// two rows, second quoted
both(':ID,:LABEL,name\n1,A,plain\n2,A,"q,q"');
