/**
 * End-to-end: paired CSV -> lenke Graph -> validate -> bulk-load into the native
 * Rust engine (mergeNdjson) -> re-export to NDJSON -> verify the round trip with
 * graphContentEqual.
 *
 * Run:  bun main.ts   (after `bun gen-fixtures.ts`)
 */
import { readFileSync } from 'node:fs';

import { Graph } from '@lenke/core';
import {
  createEmptyGraph,
  graphFromNdjson,
} from '@lenke/native';
import { createFfiBackend } from '@lenke/native/ffi';
import { serialize, deserialize, graphContentEqual } from '@lenke/serialization';

import { importPairedCsv, type Issue } from './importer.ts';

const DIR = new URL('./data/', import.meta.url).pathname;
const FFI = '/home/shawn/projects/pl-graph/crates/lenke-core/target/release/liblenke_core.so';
const read = (f: string) => readFileSync(DIR + f, 'utf8');

const rule = (s: string) => console.log('\n' + '─'.repeat(64) + '\n' + s + '\n' + '─'.repeat(64));
const report = (issues: Issue[]) => {
  if (issues.length === 0) return console.log('  ✓ no issues');
  for (const i of issues) console.log(`  ✗ [${i.severity}] ${i.kind}: ${i.message}`);
};

// ---------------------------------------------------------------------------
// 1. Import + validate the CLEAN paired CSVs.
// ---------------------------------------------------------------------------
rule('1. Import clean paired CSV  (nodes.csv + edges.csv)');
const clean = importPairedCsv(read('nodes.csv'), read('edges.csv'), {
  Person: ['age'], // Person must have an age (all clean nodes do)
});
console.log('  stats:', clean.stats);
report(clean.issues);

// Show the parser handled the hard cells: quoted comma/newline, typed props,
// multi-label, \N null, present empty-string.
const u1 = clean.graph.getVertexById('u1')!;
const u3 = clean.graph.getVertexById('u3')!;
const u4 = clean.graph.getVertexById('u4')!;
console.log('  u1.name           =', JSON.stringify(u1.getProperty('name')));
console.log('  u1.tags (list)    =', JSON.stringify(u1.getProperty('tags')));
console.log(
  '  u1.age typeof     =',
  typeof u1.getProperty('age'),
  '| score typeof:',
  typeof u1.getProperty('score'),
  '| active typeof:',
  typeof u1.getProperty('active'),
);
console.log('  u2.labels (multi) =', JSON.stringify([...clean.graph.getVertexById('u2')!.labels]));
console.log('  u3.name (newline) =', JSON.stringify(u3.getProperty('name')));
console.log(
  '  u4.score (\\N null)=',
  JSON.stringify(u4.getProperty('score')),
  '(present:',
  'score' in u4.properties,
  ')',
);
console.log('  u4.name (present"")=', JSON.stringify(u4.getProperty('name')));

// ---------------------------------------------------------------------------
// 2. Import + validate the MALFORMED CSVs (duplicate id, missing prop, dangling).
// ---------------------------------------------------------------------------
rule('2. Import malformed paired CSV  (nodes.bad.csv + edges.bad.csv)');
const bad = importPairedCsv(read('nodes.bad.csv'), read('edges.bad.csv'), {
  Person: ['name'],
});
console.log('  stats:', bad.stats);
report(bad.issues);

// Also show the library's OWN strict error when a dangling edge is NOT
// pre-filtered: decodeEdges throws MissingVertex.
rule("2b. Library's own strict decodeEdges on a dangling edge (unfiltered)");
try {
  const { decodeEdges } = await import('./paired-csv.ts');
  const g = new Graph();
  const { decodeNodes } = await import('./paired-csv.ts');
  decodeNodes(read('nodes.bad.csv'), g);
  decodeEdges(read('edges.bad.csv'), g); // e9 -> ghost99
  console.log('  (unexpected: no throw)');
} catch (e) {
  const err = e as Error & { code?: string };
  console.log('  ✓ threw:', err.constructor.name, '| code:', err.code, '|', err.message);
}

// ---------------------------------------------------------------------------
// 3. Bulk-load the clean graph into the native Rust engine.
// ---------------------------------------------------------------------------
rule('3. Bulk-load into native engine  (mergeNdjson COPY FROM)');
const ndjson = serialize(clean.graph, 'ndjson');
const ndjsonBytes = new TextEncoder().encode(ndjson);

const backend = createFfiBackend(FFI);

// 3a. mergeNdjson into a cold-booted graph (the auditable bulk-append path).
const merged = createEmptyGraph(backend);
const mergeReport = merged.mergeNdjson(ndjsonBytes);
console.log('  mergeReport:', mergeReport);
console.log('  native counts:', { v: merged.vertexCount, e: merged.edgeCount });

// 3b. Sanity query against the native graph via GQL.
const rows = merged.query`MATCH (p:Person) RETURN p.age AS age ORDER BY age`;
console.log('  GQL MATCH (p:Person) RETURN p.age =>', JSON.stringify(rows));

// 3c. Also the one-shot fastest cold-load path for comparison.
const cold = graphFromNdjson(backend, ndjsonBytes);
console.log('  graphFromNdjson counts:', { v: cold.vertexCount, e: cold.edgeCount });

// ---------------------------------------------------------------------------
// 4. Re-export to NDJSON from the native graph and verify the ROUND TRIP.
// ---------------------------------------------------------------------------
rule('4. Round-trip verify  (native.toNdjson -> Graph, graphContentEqual)');
const exported = merged.toNdjson(); // Uint8Array
const roundtripped = deserialize(new TextDecoder().decode(exported), 'ndjson', new Graph());

const eq = graphContentEqual(roundtripped, clean.graph);
console.log('  graphContentEqual(reimported, original) =', eq);

// Also verify the paired-CSV re-export round-trips (encode halves -> decode halves).
const { encodeNodes, encodeEdges } = await import('./paired-csv.ts');
const g2 = new Graph();
const { decodeNodes, decodeEdges } = await import('./paired-csv.ts');
decodeNodes(encodeNodes(clean.graph), g2);
decodeEdges(encodeEdges(clean.graph), g2);
console.log('  graphContentEqual(csv-roundtrip, original) =', graphContentEqual(g2, clean.graph));

merged.free();
cold.free();

rule(eq ? '✓ ROUND TRIP VERIFIED' : '✗ ROUND TRIP FAILED');
process.exit(eq ? 0 : 1);
