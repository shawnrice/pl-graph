import { graphFromFormat } from '../../../packages/native/src/graph.js';
import { backend, classify } from './harness.ts';
for (const [inp, fmt] of [
  ['\n', 'ndjson'],
  [' ', 'ndjson'],
  ['\n\n', 'csv'],
  ['\n', 'graphson'],
] as const) {
  const r = classify(() => graphFromFormat(backend, inp, { format: fmt }));
  console.log(
    `native deser ${JSON.stringify(inp)} ${fmt}: ${r.kind} ${(r as any).code || (r as any).name || ''} ${((r as any).msg || '').slice(0, 50)}`,
  );
}
