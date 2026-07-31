import { graphFromFormat } from '../../../packages/native/src/graph.js';
import { tsDeserialize, Graph, classify, backend } from './harness.ts';
// Isolate: pure deserialize, no method calls
console.log('=== TS empty deserialize (pure) ===');
for (const fmt of ['ndjson', 'csv', 'pg-json', 'pg-text', 'graphson'] as const) {
  const r = classify(() => {
    tsDeserialize('', fmt, new Graph());
    return 'ok';
  });
  console.log(
    `ts '' ${fmt}: ${r.kind} ${(r as any).code || (r as any).name || ''} ${((r as any).msg || '').slice(0, 70)}`,
  );
}
console.log('\n=== CSV blank-lines divergence ===');
for (const inp of ['\n\n', '\n', ' ', '\r\n']) {
  const tr = classify(() => {
    tsDeserialize(inp, 'csv', new Graph());
    return 'ok';
  });
  const nr = classify(() => graphFromFormat(backend, inp, { format: 'csv' }));
  console.log(
    `csv ${JSON.stringify(inp)}: ts=${tr.kind}/${(tr as any).code || (tr as any).name || ''} native=${nr.kind}/${(nr as any).code || (nr as any).name || ''}`,
  );
}
console.log('\n=== ndjson blank divergence ===');
for (const inp of ['\n\n', '\n', ' ', '\r\n', '  \n  ']) {
  const tr = classify(() => {
    tsDeserialize(inp, 'ndjson', new Graph());
    return 'ok';
  });
  const nr = classify(() => graphFromNdjson(backend, inp));
  console.log(
    `ndjson ${JSON.stringify(inp)}: ts=${tr.kind}/${(tr as any).code || (tr as any).name || ''} native=${nr.kind}/${(nr as any).code || (nr as any).name || ''}`,
  );
}
