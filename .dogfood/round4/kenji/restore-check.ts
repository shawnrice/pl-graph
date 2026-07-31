/**
 * Warm-boot verification: reload the checkpoint.ndjson the server wrote and
 * confirm it round-trips into a fresh graph with the same counts.
 * Run:  bun restore-check.ts
 */
import {
  graphFromNdjson,
} from '@lenke/native';
import { createFfiBackend } from '@lenke/native/ffi';

const LIB = '/home/shawn/projects/pl-graph/crates/lenke-core/target/release/liblenke_core.so';
const bytes = await Bun.file(`${import.meta.dir}/checkpoint.ndjson`).bytes();
const backend = createFfiBackend(LIB);
const g = graphFromNdjson(backend, bytes);

const shapes = g.query('MATCH (s:Shape) RETURN count(*) AS `c`')[0].c;
const cursors = g.query('MATCH (c:Cursor) RETURN count(*) AS `c`')[0].c;
const counter = g.query('MATCH (c:Counter) RETURN c.n AS `n`')[0]?.n;

console.log(`restored ${bytes.byteLength} bytes: vertices=${g.vertexCount} edges=${g.edgeCount}`);
console.log(`  shapes=${shapes} cursors=${cursors} counter.n=${counter}`);
g.free();
