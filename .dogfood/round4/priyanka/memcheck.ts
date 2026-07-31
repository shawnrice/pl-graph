import {
  graphFromNdjson,
} from '@lenke/native';
import { createNodeBackend } from '@lenke/node/backend';
const before = process.memoryUsage().rss;
const bytes = await Bun.file(`${import.meta.dir}/graph.ndjson`).bytes();
const g = graphFromNdjson(createNodeBackend(), bytes);
g.createVertexIndex('uid');
g.createVertexIndex('rid');
g.createVertexIndex('gid');
// touch it
g.query('MATCH (r:Resource {rid:$r}) RETURN r.rid AS x', { r: 'r1' });
const after = process.memoryUsage().rss;
console.log(`RSS before load: ${(before / 1e6).toFixed(0)} MB`);
console.log(`RSS after load+index (174k v / 403k e): ${(after / 1e6).toFixed(0)} MB`);
console.log(`delta: ${((after - before) / 1e6).toFixed(0)} MB  (ndjson source on disk = 53 MB)`);
g.free();
