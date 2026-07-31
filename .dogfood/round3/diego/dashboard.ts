// Graph-analytics dashboard backend over an org/social graph.
// - Loads NDJSON into the native Rust core via the N-API backend.
// - Computes metrics with GQL analytics (aggregates, COUNT{}, CASE, set ops).
// - Ships a large result set over the columnar Arrow path and proves it decodes
//   to the SAME rows as the JSON path, at a smaller wire size.

import { readFile } from 'node:fs/promises';

import {
  decodeArrow,
  graphFromNdjson,
  type Row,
} from '@lenke/native';
import { createNodeBackend } from '@lenke/node/backend';

const here = (f: string) => new URL(f, import.meta.url).pathname;
const bytes = await readFile(here('./org-graph.ndjson'));

const backend = createNodeBackend();
using g = graphFromNdjson(backend, bytes);

const rule = (s: string) => console.log(`\n=== ${s} ===`);
const show = (rows: Row[], n = rows.length) => console.table(rows.slice(0, n));

console.log(`loaded graph: ${g.vertexCount} vertices, ${g.edgeCount} edges`);

// ---------------------------------------------------------------------------
// 1. Top-N by out-degree — COUNT { } correlated subquery + ORDER BY DESC LIMIT
// ---------------------------------------------------------------------------
rule('1. Top 10 people by KNOWS out-degree');
type DegRow = { name: string; dept: string; deg: number };
const topDegree = g.query<DegRow>(`
  MATCH (p:Person)
  RETURN p.name AS name, p.dept AS dept, COUNT { (p)-[:KNOWS]->() } AS deg
  ORDER BY deg DESC
  LIMIT 10
`);
show(topDegree);

// ---------------------------------------------------------------------------
// 2. Per-department head-counts + averages — implicit grouping aggregates
// ---------------------------------------------------------------------------
rule('2. Per-department head-count, avg age, avg salary, min/max salary');
type DeptRow = {
  dept: string;
  headcount: number;
  avgAge: number;
  avgSalary: number;
  minSalary: number;
  maxSalary: number;
};
const deptStats = g.query<DeptRow>(`
  MATCH (p:Person)
  RETURN p.dept        AS dept,
         count(*)      AS headcount,
         avg(p.age)    AS avgAge,
         avg(p.salary) AS avgSalary,
         min(p.salary) AS minSalary,
         max(p.salary) AS maxSalary
  ORDER BY headcount DESC
`);
show(
  deptStats.map((r) => ({
    ...r,
    avgAge: Math.round(r.avgAge * 10) / 10,
    avgSalary: Math.round(r.avgSalary),
  })),
);

// ---------------------------------------------------------------------------
// 3. Shared-connection overlap between two teams — set INTERSECT / EXCEPT
//    "Who is known by BOTH the Platform team AND the Product team?"
// ---------------------------------------------------------------------------
rule('3. People known by BOTH Platform and Product teams (INTERSECT)');
const shared = g.query<{ name: string }>(`
  MATCH (a:Person)-[:KNOWS]->(x:Person) WHERE a.team = 'Platform'
  RETURN DISTINCT x.name AS name
  INTERSECT
  MATCH (b:Person)-[:KNOWS]->(y:Person) WHERE b.team = 'Product'
  RETURN DISTINCT y.name AS name
`);
console.log(`overlap size: ${shared.length}`);
show(shared, 8);

rule('3b. Known by Platform but NOT by Product (EXCEPT)');
const onlyPlatform = g.query<{ name: string }>(`
  MATCH (a:Person)-[:KNOWS]->(x:Person) WHERE a.team = 'Platform'
  RETURN DISTINCT x.name AS name
  EXCEPT
  MATCH (b:Person)-[:KNOWS]->(y:Person) WHERE b.team = 'Product'
  RETURN DISTINCT y.name AS name
`);
console.log(`platform-only size: ${onlyPlatform.length}`);

// ---------------------------------------------------------------------------
// 4. Cohort comparison — CASE age buckets, aggregate per cohort
// ---------------------------------------------------------------------------
rule('4. Cohort comparison by age band (CASE) — count + avg salary/level');
type CohortRow = {
  cohort: string;
  people: number;
  avgSalary: number;
  avgLevel: number;
};
const cohorts = g.query<CohortRow>(`
  MATCH (p:Person)
  RETURN CASE
           WHEN p.age < 30 THEN 'A: <30'
           WHEN p.age < 45 THEN 'B: 30-44'
           WHEN p.age < 60 THEN 'C: 45-59'
           ELSE 'D: 60+'
         END           AS cohort,
         count(*)      AS people,
         avg(p.salary) AS avgSalary,
         avg(p.level)  AS avgLevel
  ORDER BY cohort
`);
show(
  cohorts.map((r) => ({
    ...r,
    avgSalary: Math.round(r.avgSalary),
    avgLevel: Math.round(r.avgLevel * 100) / 100,
  })),
);

// ---------------------------------------------------------------------------
// 5. The Arrow columnar path — ship a LARGE result set as ARW1, prove parity.
//    Full per-person export (600 rows) as JSON vs. columnar Arrow.
// ---------------------------------------------------------------------------
rule('5. Arrow columnar path vs JSON path (full 600-person export)');
const exportQ = `
  MATCH (p:Person)
  RETURN p.name AS name, p.dept AS dept, p.team AS team,
         p.age AS age, p.level AS level, p.salary AS salary
  ORDER BY p.name
`;

// JSON path (what a naive REST endpoint would send)
const jsonRows = g.query<Row>(exportQ);
const jsonWire = Buffer.from(JSON.stringify(jsonRows));

// Arrow columnar path
const arrowBlob = g.queryArrow(exportQ);
const arrowRows = decodeArrow(arrowBlob);

// Parity: identical decoded rows?
const identical = JSON.stringify(jsonRows) === JSON.stringify(arrowRows);
console.log(`rows: json=${jsonRows.length}  arrow=${arrowRows.length}`);
console.log(`ARW1 magic: ${new TextDecoder().decode(arrowBlob.subarray(0, 4))}`);
console.log(`identical rows (JSON === decodeArrow)? ${identical}`);
console.log(
  `wire size — json: ${jsonWire.byteLength} B   arrow: ${arrowBlob.byteLength} B   ` +
    `(arrow is ${((arrowBlob.byteLength / jsonWire.byteLength) * 100).toFixed(1)}% of JSON)`,
);
console.log('sample decoded arrow row:', arrowRows[0]);

if (!identical) {
  // Show the first divergence to make a failure debuggable.
  for (let i = 0; i < jsonRows.length; i++) {
    if (JSON.stringify(jsonRows[i]) !== JSON.stringify(arrowRows[i])) {
      console.log('FIRST DIVERGENCE at row', i, {
        json: jsonRows[i],
        arrow: arrowRows[i],
      });
      break;
    }
  }
  process.exitCode = 1;
}

// ---------------------------------------------------------------------------
// 6. Arrow on an aggregate result set (dept stats) — confirm numeric columns
// ---------------------------------------------------------------------------
rule('6. Arrow parity on aggregate query (dept stats)');
const aggQ = `
  MATCH (p:Person)
  RETURN p.dept AS dept, count(*) AS headcount, avg(p.salary) AS avgSalary
  ORDER BY dept
`;
const aggJson = g.query<Row>(aggQ);
const aggArrow = decodeArrow(g.queryArrow(aggQ));
console.log(`aggregate arrow parity? ${JSON.stringify(aggJson) === JSON.stringify(aggArrow)}`);
console.log('json  :', aggJson[0]);
console.log('arrow :', aggArrow[0]);

console.log('\nAll metrics computed. Done.');
