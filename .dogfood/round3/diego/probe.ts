// Scratch: probe analytical depth + Arrow edge cases to find friction.
import { readFile } from 'node:fs/promises';

import {
  decodeArrow,
  graphFromNdjson,
  type Row,
} from '@lenke/native';
import { createNodeBackend } from '@lenke/node/backend';

const here = (f: string) => new URL(f, import.meta.url).pathname;
const bytes = await readFile(here('./org-graph.ndjson'));
using g = graphFromNdjson(createNodeBackend(), bytes);

const tryQ = (label: string, q: string, params?: Record<string, unknown>) => {
  try {
    const rows = g.query(q, params as any);
    console.log(`OK   ${label}: ${rows.length} rows | sample:`, rows[0]);
  } catch (e: any) {
    console.log(`FAIL ${label}: ${e?.code ?? ''} ${e?.message ?? e}`);
  }
};

const tryArrow = (label: string, q: string) => {
  try {
    const rows = decodeArrow(g.queryArrow(q));
    console.log(`OK-ARROW   ${label}: ${rows.length} rows | sample:`, rows[0]);
  } catch (e: any) {
    console.log(`FAIL-ARROW ${label}: ${e?.code ?? ''} ${e?.message ?? e}`);
  }
};

// --- multi-part WITH pipeline: degree, then filter, then aggregate ---
tryQ(
  'WITH pipeline (degree -> filter -> group)',
  `MATCH (p:Person)
   WITH p, COUNT { (p)-[:KNOWS]->() } AS deg
   WHERE deg >= 8
   RETURN p.dept AS dept, count(*) AS hubs, avg(deg) AS avgDeg
   ORDER BY hubs DESC`,
);

// --- WITH with ORDER BY/LIMIT in the middle (top-k then join back) ---
tryQ(
  'WITH ... ORDER BY ... LIMIT then downstream',
  `MATCH (p:Person)
   WITH p, COUNT { (p)-[:KNOWS]->() } AS deg
   ORDER BY deg DESC LIMIT 5
   RETURN p.name AS name, deg`,
);

// --- UNION ALL vs UNION ---
tryQ(
  'UNION (distinct depts two ways)',
  `MATCH (p:Person) WHERE p.dept = 'Sales' RETURN p.team AS team
   UNION
   MATCH (p:Person) WHERE p.dept = 'Support' RETURN p.team AS team`,
);

// --- correlated COUNT{} with a WHERE inside the subquery ---
tryQ(
  'COUNT{} with inner WHERE (recent ties)',
  `MATCH (p:Person)
   RETURN p.name AS name,
          COUNT { (p)-[k:KNOWS]->() WHERE k.since >= 2022 } AS recent
   ORDER BY recent DESC LIMIT 5`,
);

// --- EXISTS{} subquery in WHERE ---
tryQ(
  'EXISTS{} filter',
  `MATCH (p:Person)
   WHERE EXISTS { (p)-[:KNOWS]->(q:Person) WHERE q.dept <> p.dept }
   RETURN count(*) AS crossDeptConnectors`,
);

// --- collect_list aggregate ---
tryQ(
  'collect_list',
  `MATCH (p:Person) WHERE p.team = 'Brand'
   RETURN p.dept AS dept, collect_list(p.name) AS names`,
);

// --- Arrow with a LIST column (collect_list) — unsupported type? ---
tryArrow(
  'arrow: list column (collect_list)',
  `MATCH (p:Person) WHERE p.team = 'Brand'
   RETURN p.dept AS dept, collect_list(p.name) AS names`,
);

// --- Arrow with a boolean column ---
tryArrow(
  'arrow: boolean column',
  `MATCH (p:Person) RETURN p.name AS name, (p.salary > 150000) AS wellPaid ORDER BY p.name LIMIT 3`,
);

// --- Arrow returning a whole node element ---
tryArrow('arrow: node element column', `MATCH (p:Person) RETURN p LIMIT 3`);

// --- JSON returning a whole node element (for comparison) ---
tryQ('json: node element column', `MATCH (p:Person) RETURN p LIMIT 1`);

// --- Arrow with element_id (string) ---
tryArrow(
  'arrow: element_id column',
  `MATCH (p:Person) RETURN element_id(p) AS id, p.name AS name ORDER BY p.name LIMIT 3`,
);

// --- integer-vs-float check: does count(*) come back integral over Arrow? ---
const q = `MATCH (p:Person) RETURN p.dept AS dept, count(*) AS n ORDER BY dept`;
const j = g.query<Row>(q);
const a = decodeArrow(g.queryArrow(q));
console.log('int check json n typeof:', typeof j[0].n, j[0].n, '| arrow n:', typeof a[0].n, a[0].n);
