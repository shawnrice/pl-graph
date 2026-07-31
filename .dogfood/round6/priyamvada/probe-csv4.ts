import { graphFromFormat } from '../../../packages/native/src/graph.js';
import { tsDeserialize, Graph, tsQuery, classify, backend } from './harness.ts';
const show = (label: string, inp: string) => {
  const t = classify(() => tsQuery(tsDeserialize(inp, 'csv', new Graph()), 'MATCH (n) RETURN n'));
  const n = classify(() => graphFromFormat(backend, inp, { format: 'csv' }).query('MATCH (n) RETURN n'));
  const tv =
    t.kind === 'ok'
      ? JSON.stringify((t as any).value)
      : `${t.kind}:${(t as any).code || (t as any).name}`;
  const nv =
    n.kind === 'ok'
      ? JSON.stringify((n as any).value)
      : `${n.kind}:${(n as any).code || (n as any).name}`;
  console.log(`${label}\n  TS : ${tv}\n  NAT: ${nv}`);
};
show('name:string', ':ID,:LABEL,name:string\n1,A,hello');
show('name (bare)', ':ID,:LABEL,name\n1,A,hello');
show('age:int    ', ':ID,:LABEL,age:int\n1,A,5');
show('age:long   ', ':ID,:LABEL,age:long\n1,A,5');
show('w:double   ', ':ID,:LABEL,w:double\n1,A,0.5');
show('single a   ', ':ID,:LABEL,a\n1,A,x');
show('ab bare    ', ':ID,:LABEL,ab\n1,A,x');
