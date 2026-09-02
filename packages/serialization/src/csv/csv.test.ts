import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';
import { ErrorCode, hasErrorCode } from '@lenke/errors';

import { chunked, collect } from '../streaming.js';
import type { ChunkSource } from '../streaming.js';
import { graphContentEqual, randomLpgGraph } from '../testkit.js';
import {
  csvCodec,
  decode,
  decodeEdges,
  decodeNodes,
  encode,
  encodeEdges,
  encodeNodes,
  encodeStream,
} from './index.js';

const roundTripNodes = (graph: Graph): Graph => decodeNodes(encodeNodes(graph), new Graph());
const roundTrip = (graph: Graph): Graph => decode(encode(graph), new Graph());

test('foreign numeric cells outside the strict grammar decode to null (matches Rust)', () => {
  // JS Number() accepts " 5 " / "0x1F" as 5 / 31; Rust's str::parse rejects them.
  // Both decoders now gate on a finite-decimal grammar → NaN → null. (Our encoder
  // only ever writes canonical decimals, which still round-trip.)
  const header = 'id,:LABEL,n:integer\n';
  const footer = '\n=== EDGES ===\nid,:START_ID,:END_ID,:TYPE';

  for (const cell of [' 5 ', '0x1F']) {
    const back = decode(`${header}1,T,${cell}${footer}`, new Graph());
    // Foreign cell rejected → NaN (a query then renders it null, matching Rust),
    // NOT coerced to 5 / 31 as JS Number() would.
    expect(back.getVertexById('1')?.properties.n).toBeNaN();
  }

  const ok = decode(`${header}1,T,5${footer}`, new Graph());
  expect(ok.getVertexById('1')?.properties.n).toBe(5);
});

const FORMULA = /^[=+\-@\t\r]/;
const FORMULA_CHARS = ['=', '+', '-', '@', '\t', '\r'] as const;

/** Quote-aware split into each cell's spreadsheet-visible (RFC-4180 unquoted) content. */
const cellsOf = (csv: string): string[] => {
  const out: string[] = [];
  let f = '';
  let inQ = false;

  for (let i = 0; i < csv.length; i += 1) {
    const c = csv[i];

    if (inQ) {
      if (c === '"') {
        if (csv[i + 1] === '"') {
          f += '"';
          i += 1;
        } else {
          inQ = false;
        }
      } else {
        f += c;
      }
    } else if (c === '"') {
      inQ = true;
    } else if (c === ',' || c === '\n') {
      out.push(f);
      f = '';
    } else if (c !== '\r') {
      f += c;
    }
  }

  out.push(f);

  return out;
};

// No cell a spreadsheet would evaluate as a formula, EXCEPT the fixed
// `=== EDGES ===` section marker of the combined format — a constant,
// library-controlled structural line, never attacker data, and absent from the
// per-file `encodeNodes`/`encodeEdges` artifacts that are the spreadsheet path.
// (Use only on graphs with no bare-number cell, since `-5` legitimately leads with `-`.)
const assertNoFormulaCells = (csv: string): void => {
  for (const cell of cellsOf(csv)) {
    if (cell === '=== EDGES ===') {
      continue;
    }

    expect({ cell, formula: FORMULA.test(cell) }).toEqual({ cell, formula: false });
  }
};

describe('CSV formula neutralization: complete coverage (every string-cell surface)', () => {
  test('formula-leading node ids, edge ids, and endpoints are neutralized and round-trip', () => {
    for (const lead of FORMULA_CHARS) {
      const g = new Graph();
      const a = g.addVertex({ id: `${lead}from`, labels: ['N'], properties: {} });
      const b = g.addVertex({ id: `${lead}to`, labels: ['N'], properties: {} });
      g.addEdge({ id: `${lead}edge`, from: a, to: b, labels: ['R'], properties: {} });

      const csv = encode(g);
      assertNoFormulaCells(csv);
      expect(graphContentEqual(roundTrip(g), g)).toBe(true);
    }
  });

  test('formula-leading labels and edge types are neutralized and round-trip (any position)', () => {
    for (const lead of FORMULA_CHARS) {
      const g = new Graph();
      const a = g.addVertex({ id: 'a', labels: [`${lead}First`, 'Plain'], properties: {} });
      const b = g.addVertex({ id: 'b', labels: ['Plain', `${lead}Second`], properties: {} });
      g.addEdge({ from: a, to: b, labels: [`${lead}TYPE`], properties: {} });

      assertNoFormulaCells(encode(g));
      expect(graphContentEqual(roundTrip(g), g)).toBe(true);
    }
  });

  test('formula-leading property keys are neutralized in the header and round-trip', () => {
    for (const lead of FORMULA_CHARS) {
      const g = new Graph();
      g.addVertex({ id: 'a', labels: ['N'], properties: { [`${lead}key`]: 'v' } });

      assertNoFormulaCells(encode(g));
      expect(graphContentEqual(roundTrip(g), g)).toBe(true);
    }
  });

  test('formula-leading STRING values and string list elements are neutralized and round-trip', () => {
    for (const lead of FORMULA_CHARS) {
      const g = new Graph();
      g.addVertex({
        id: 'a',
        labels: ['N'],
        properties: {
          scalar: `${lead}danger`,
          list: [`${lead}first`, 'ok', `${lead}later`], // first AND non-first
        },
      });

      assertNoFormulaCells(encode(g));
      expect(graphContentEqual(roundTrip(g), g)).toBe(true);
    }
  });

  test('all surfaces at once, every formula char, still round-trips exactly', () => {
    const g = new Graph();

    for (const lead of FORMULA_CHARS) {
      const a = g.addVertex({
        id: `${lead}id-${lead.charCodeAt(0)}`,
        labels: [`${lead}Label`],
        properties: { [`${lead}k`]: `${lead}v`, [`${lead}list`]: [`${lead}e`] },
      });
      const b = g.addVertex({ id: `n-${lead.charCodeAt(0)}`, labels: [], properties: {} });
      g.addEdge({
        from: a,
        to: b,
        labels: [`${lead}T`],
        properties: { [`${lead}ek`]: `${lead}ev` },
      });
    }

    assertNoFormulaCells(encode(g));
    expect(graphContentEqual(roundTrip(g), g)).toBe(true);
  });

  test('numbers are NOT neutralized — a spreadsheet reads them as numbers', () => {
    const g = new Graph();
    g.addVertex({
      id: 'a',
      labels: ['N'],
      properties: { balance: -5, delta: -2.5, nums: [-5, -6, -7] },
    });

    const csv = encode(g);
    // The number cells legitimately begin with `-`; they must not be backslash-guarded.
    expect(csv).not.toContain('\\-5');
    expect(csv).not.toContain('\\-2.5');
    expect(graphContentEqual(roundTrip(g), g)).toBe(true);
  });

  test('genuine backslash-leading content in every surface round-trips (no corruption)', () => {
    const g = new Graph();
    const a = g.addVertex({
      id: '\\node',
      labels: ['\\Label', '\\=trap'], // plain-backslash and backslash-then-formula
      properties: { '\\key': '\\value', '\\list': ['\\a', '\\=b'] },
    });
    const b = g.addVertex({ id: '\\=weird', labels: [], properties: {} });
    g.addEdge({ id: '\\edge', from: a, to: b, labels: ['\\R'], properties: {} });

    // Backslash-leading cells are already spreadsheet-safe; the point is fidelity.
    assertNoFormulaCells(encode(g));
    expect(graphContentEqual(roundTrip(g), g)).toBe(true);
  });

  test('a null LIST element round-trips exactly (the \\Tn: sigil, not the string "null")', () => {
    const g = new Graph();
    g.addVertex({
      labels: ['T'],
      properties: { dims: [1, null, 2], oneNull: [null] }, // null is a first-class value
    });

    // On the wire the null element uses the type-override sigil, not "null".
    expect(encode(g)).toContain('Tn:');

    const back = roundTrip(g);
    const [v] = [...back.vertices];
    expect(v.getProperty<unknown[]>('dims')).toEqual([1, null, 2]); // was [1, "null", 2] before the fix
    expect(v.getProperty<unknown[]>('oneNull')).toEqual([null]);
  });

  test('the streaming encoder neutralizes the same way (separate build path)', async () => {
    const g = new Graph();
    const a = g.addVertex({ id: '=sid', labels: ['=SLabel'], properties: { '=sk': '=sv' } });
    const b = g.addVertex({ id: 'plain', labels: [], properties: {} });
    g.addEdge({ id: '=se', from: a, to: b, labels: ['=ST'], properties: {} });

    const streamed = await collect(encodeStream(g));
    assertNoFormulaCells(streamed);
    expect(graphContentEqual(decode(streamed, new Graph()), g)).toBe(true);
  });
});

describe('CSV hardening: header quoting + formula neutralization', () => {
  test('a property key containing a comma/quote/newline round-trips (header is quoted)', () => {
    const g = new Graph();
    g.addVertex({ id: 'n1', labels: ['N'], properties: { 'a,b': 1, 'c"d': 2, 'e\nf': 3 } });

    const csv = encodeNodes(g);
    expect(csv.split('\n')[0]).toContain('"a,b:integer"'); // header cell quoted

    const back = roundTripNodes(g);
    expect(graphContentEqual(back, g)).toBe(true); // keys survive intact
  });

  test('a string value starting with a formula char is neutralized and round-trips', () => {
    const g = new Graph();
    g.addVertex({
      id: 'n1',
      labels: ['N'],
      properties: { name: '=1+2', cmd: '@SUM(A1)', dash: '-danger', plus: '+x' },
    });

    const csv = encodeNodes(g);
    // On the wire each dangerous string begins with a backslash — inert to a
    // spreadsheet (no leading `=`/`@`/`-`/`+`).
    expect(csv).toContain('"\\=1+2"');
    expect(csv).toContain('"\\@SUM(A1)"');
    expect(csv).toContain('"\\-danger"');
    expect(csv).toContain('"\\+x"');

    const back = roundTripNodes(g);
    expect(graphContentEqual(back, g)).toBe(true); // decode strips the guard back off
  });

  test('a negative NUMBER is left alone (a spreadsheet reads it as a number, not a formula)', () => {
    const g = new Graph();
    g.addVertex({ id: 'n1', labels: ['N'], properties: { balance: -5, delta: -2.5 } });

    const csv = encodeNodes(g);
    expect(csv).not.toContain('\\-5'); // no guard — numbers are safe as-is

    const back = roundTripNodes(g);
    expect(graphContentEqual(back, g)).toBe(true);
  });
});

describe('CSV escaping / quoting (RFC 4180)', () => {
  test('quotes fields containing commas, quotes, and newlines', () => {
    const g = new Graph();
    g.addVertex({
      id: 'n1',
      labels: ['Person'],
      properties: { note: 'a, b "c"\nd', plain: 'x' },
    });
    const csv = encodeNodes(g);
    expect(csv).toContain('"a, b ""c""\nd"');

    const back = roundTripNodes(g);
    expect(graphContentEqual(back, g)).toBe(true);
  });

  test('quotes list-context strings containing the ; separator', () => {
    const g = new Graph();
    g.addVertex({ id: 'n1', labels: [], properties: { tags: ['a;b', 'c'] } });
    const back = roundTripNodes(g);
    expect(back.getVertexById('n1')!.properties.tags as string[]).toEqual(['a;b', 'c']);
  });
});

describe('typed headers', () => {
  test('emits key:type columns spanning the union of all node keys', () => {
    const g = new Graph();
    g.addVertex({ id: 'n1', labels: ['A'], properties: { age: 30, active: true } });
    g.addVertex({ id: 'n2', labels: ['B'], properties: { ratio: 1.5, name: 'x' } });
    const [header] = encodeNodes(g).split('\n');
    expect(header).toBe('id,:LABEL,age:integer,active:boolean,ratio:float,name:string');
  });

  test('list columns carry an element type and []', () => {
    const g = new Graph();
    g.addVertex({ id: 'n1', labels: [], properties: { nums: [1, 2, 3], words: ['a'] } });
    const [header] = encodeNodes(g).split('\n');
    expect(header).toBe('id,:LABEL,nums:integer[],words:string[]');
    const back = roundTripNodes(g);
    expect(graphContentEqual(back, g)).toBe(true);
  });

  test('integer vs float vs boolean vs string survive', () => {
    const g = new Graph();
    g.addVertex({
      id: 'n1',
      labels: [],
      properties: { i: 5, f: 5.5, b: false, s: '5' },
    });
    const back = roundTripNodes(g);
    const p = back.getVertexById('n1')!.properties;
    expect(p.i).toBe(5);
    expect(p.f).toBe(5.5);
    expect(p.b).toBe(false);
    expect(p.s).toBe('5');
  });

  // A bare (untyped) header column keeps its FULL name as the key. Regression:
  // the no-colon path did `slice(0, -1)`, silently truncating every plain header
  // (`name` → `nam`), diverging from the native codec which keeps the full name.
  test('a bare/untyped header keeps the full column name as the key', () => {
    const g = decodeNodes('id,:LABEL,name\n1,A,hello', new Graph());
    const v = g.getVertexById('1')!;
    expect(Object.keys(v.properties)).toEqual(['name']);
    expect(v.properties.name).toBe('hello');
  });
});

describe('null vs empty-string vs absent distinction', () => {
  test('null, empty string, and missing key are all distinct after round-trip', () => {
    const g = new Graph();
    g.addVertex({ id: 'hasNull', labels: [], properties: { x: null, anchor: 1 } });
    g.addVertex({ id: 'hasEmpty', labels: [], properties: { x: '', anchor: 1 } });
    g.addVertex({ id: 'hasMissing', labels: [], properties: { anchor: 1 } });

    const back = roundTripNodes(g);

    const nullNode = back.getVertexById('hasNull')!.properties;
    expect('x' in nullNode).toBe(true);
    expect(nullNode.x).toBe(null);

    const emptyNode = back.getVertexById('hasEmpty')!.properties;
    expect('x' in emptyNode).toBe(true);
    expect(emptyNode.x).toBe('');

    const missingNode = back.getVertexById('hasMissing')!.properties;
    expect('x' in missingNode).toBe(false);
  });

  test('a node lacking a key does not gain a spurious null', () => {
    const g = new Graph();
    g.addVertex({ id: 'a', labels: [], properties: { only: 1 } });
    g.addVertex({ id: 'b', labels: [], properties: { other: 2 } });
    const back = roundTripNodes(g);
    expect('other' in back.getVertexById('a')!.properties).toBe(false);
    expect('only' in back.getVertexById('b')!.properties).toBe(false);
  });
});

describe('sentinel-collision safety', () => {
  test('a literal string equal to the null token round-trips as a string', () => {
    const g = new Graph();
    g.addVertex({ id: 'n1', labels: [], properties: { x: '\\N', y: '\\Ti:5' } });
    const back = roundTripNodes(g);
    const p = back.getVertexById('n1')!.properties;
    expect(p.x).toBe('\\N');
    expect(p.y).toBe('\\Ti:5');
  });

  test('heterogeneous list elements (mixed int/string) round-trip', () => {
    const g = new Graph();
    g.addVertex({ id: 'n1', labels: [], properties: { mixed: [1, 'a', 2] } });
    const back = roundTripNodes(g);
    expect(back.getVertexById('n1')!.properties.mixed).toEqual([1, 'a', 2]);
  });

  test('a string LIST element equal to a type-override sigil round-trips as a string', () => {
    // The wire encodes a mixed-type element as `\T<code>:<raw>` (e.g. `\Ti:5` = int 5
    // inside a string list). A genuine string element whose *value* begins with `\T`
    // would, once escaped, be indistinguishable from that override — so it must be
    // emitted as an explicit `\Ts:` string-code override. Covers the int/null/string
    // sigils plus a real override in the same list, so the disambiguation is exercised
    // both ways. (Rust's csv codec has the matching branch — the wire is byte-identical.)
    const g = new Graph();
    g.addVertex({
      id: 'n1',
      labels: [],
      properties: {
        sigils: ['\\Ti:5', '\\Tn:', '\\Ts:x', 'ok'], // strings that LOOK like overrides
        real: ['\\Ti:5', 5, null], // a genuine string, a real int override, a real null
      },
    });
    const back = roundTripNodes(g);
    const p = back.getVertexById('n1')!.properties;
    expect(p.sigils).toEqual(['\\Ti:5', '\\Tn:', '\\Ts:x', 'ok']);
    expect(p.real).toEqual(['\\Ti:5', 5, null]);
  });

  test('a key that is a list on one node and a scalar on another round-trips', () => {
    const g = new Graph();
    g.addVertex({ id: 'n1', labels: [], properties: { v: [1, 2] } });
    g.addVertex({ id: 'n2', labels: [], properties: { v: 'hello' } });
    g.addVertex({ id: 'n3', labels: [], properties: { v: 9 } });
    const back = roundTripNodes(g);
    expect(back.getVertexById('n1')!.properties.v).toEqual([1, 2]);
    expect(back.getVertexById('n2')!.properties.v).toBe('hello');
    expect(back.getVertexById('n3')!.properties.v).toBe(9);
  });

  test('empty list round-trips distinctly from absent', () => {
    const g = new Graph();
    g.addVertex({ id: 'n1', labels: [], properties: { xs: [], anchorListType: [1] } });
    g.addVertex({ id: 'n2', labels: [], properties: { anchorListType: [2] } });
    const back = roundTripNodes(g);
    expect(back.getVertexById('n1')!.properties.xs).toEqual([]);
    expect('xs' in back.getVertexById('n2')!.properties).toBe(false);
  });

  test('a label containing the `;` separator round-trips (escaped, not split)', () => {
    const g = new Graph();
    const a = g.addVertex({ id: 'a', labels: ['has;semi', 'Plain'], properties: {} });
    const b = g.addVertex({ id: 'b', labels: [], properties: {} });
    g.addEdge({ id: 'e', from: a, to: b, labels: ['REL;X'], properties: {} });
    const back = decode(encode(g), new Graph());
    expect([...back.getVertexById('a')!.labels].sort()).toEqual(['Plain', 'has;semi']);
    expect([...[...back.edges][0].labels]).toEqual(['REL;X']);
  });

  test('a value containing the `=== EDGES ===` marker does not split the document', () => {
    const g = new Graph();
    const a = g.addVertex({ id: 'a', labels: ['N'], properties: { note: 'x\n=== EDGES ===\ny' } });
    const b = g.addVertex({ id: 'b', labels: ['N'], properties: {} });
    g.addEdge({ id: 'e', from: a, to: b, labels: ['R'], properties: {} });
    const back = decode(encode(g), new Graph());
    expect(back.vertexCount).toBe(2);
    expect(back.edgeCount).toBe(1);
    expect(back.getVertexById('a')!.properties.note).toBe('x\n=== EDGES ===\ny');
  });
});

describe('labels', () => {
  test('multi-label nodes ;-join in :LABEL', () => {
    const g = new Graph();
    g.addVertex({ id: 'n1', labels: ['Person', 'Admin'], properties: {} });
    g.addVertex({ id: 'n2', labels: [], properties: {} });
    const csv = encodeNodes(g);
    // The ; element separator forces RFC-4180 quoting of the label cell.
    expect(csv).toContain('n1,"Person;Admin"');
    expect(csv).toContain('n2,');
    const back = roundTripNodes(g);
    expect([...back.getVertexById('n1')!.labels].sort()).toEqual(['Admin', 'Person']);
    expect([...back.getVertexById('n2')!.labels]).toEqual([]);
  });
});

describe('edges', () => {
  test('round-trips endpoints, :TYPE label, and typed props', () => {
    const g = new Graph();
    const a = g.addVertex({ id: 'a', labels: [], properties: {} });
    const b = g.addVertex({ id: 'b', labels: [], properties: {} });
    g.addEdge({ id: 'e1', from: a, to: b, labels: ['KNOWS'], properties: { since: 2020 } });
    g.addEdge({ id: 'e2', from: a, to: b, labels: ['KNOWS'], properties: {} }); // parallel

    const target = new Graph();
    decodeNodes(encodeNodes(g), target);
    decodeEdges(encodeEdges(g), target);
    expect(graphContentEqual(target, g)).toBe(true);
  });

  test('decodeEdges throws MissingVertex on a dangling endpoint', () => {
    const g = new Graph();
    let caught: unknown;

    try {
      decodeEdges('id,:START_ID,:END_ID,:TYPE\ne1,x,y,KNOWS', g);
    } catch (e) {
      caught = e;
    }

    // The code is the contract; the message is just a human hint.
    expect(hasErrorCode(caught, ErrorCode.MissingVertex)).toBe(true);
  });
});

describe('codec single-string adapter', () => {
  test('encode/decode round-trips through the sentinel', () => {
    const g = randomLpgGraph(7);
    const back = decode(encode(g), new Graph());
    expect(graphContentEqual(back, g)).toBe(true);
    expect(csvCodec.name).toBe('csv');
    expect(encode(g)).toContain('=== EDGES ===');
  });
});

describe('round-trip property test', () => {
  test('graphContentEqual over 500 seeds', () => {
    for (let seed = 0; seed < 500; seed += 1) {
      const original = randomLpgGraph(seed);
      const back = csvCodec.decode(csvCodec.encode(original), new Graph());

      if (!graphContentEqual(back, original)) {
        throw new Error(`round-trip failed for seed ${seed}`);
      }
    }

    expect(true).toBe(true);
  });
});

describe('throughput smoke test', () => {
  test('encodes and decodes ~10k elements quickly', () => {
    const g = new Graph();
    g.disableEvents();
    const nodeCount = 5000;
    const verts = [];

    for (let i = 0; i < nodeCount; i += 1) {
      verts.push(
        g.addVertex({
          id: `n${i}`,
          labels: i % 2 === 0 ? ['Even', 'Num'] : ['Odd'],
          properties: { idx: i, ratio: i / 3, label: `node-${i}`, flag: i % 3 === 0 },
        }),
      );
    }

    for (let i = 0; i < 5000; i += 1) {
      g.addEdge({
        id: `e${i}`,
        from: verts[i % nodeCount],
        to: verts[(i * 7 + 1) % nodeCount],
        labels: ['LINK'],
        properties: { w: i * 0.5 },
      });
    }

    g.enableEvents();

    const start = performance.now();
    const str = encode(g);
    const back = decode(str, new Graph());
    const elapsed = performance.now() - start;

    expect(graphContentEqual(back, g)).toBe(true);
    expect(back.vertexCount).toBe(nodeCount);
    expect(back.edgeCount).toBe(5000);
    // eslint-disable-next-line no-console
    console.log(`csv throughput: 10k elements encode+decode in ${elapsed.toFixed(1)}ms`);
    // NOTE: deliberately no wall-clock assertion. This ran under
    // `expect(elapsed).toBeLessThan(2000)`, a benchmark wearing a test's clothes:
    // it asserts nothing about correctness, and the suite runs in parallel with
    // every other nx target, so a loaded machine blew the budget and failed the
    // build. The throughput is still logged above; the assertions that carry
    // signal — round-trip equality and the counts — are timing-independent. A hard
    // perf budget belongs in `benchmarks/`, run alone, not the parallel suite.
  });
});

describe('streaming', () => {
  test('decodeStream(encodeStream(g)) round-trips over 200 seeds', async () => {
    for (let seed = 0; seed < 200; seed += 1) {
      const original = randomLpgGraph(seed);
      // A generator IS a ChunkSource — pipe it straight into decode.
      const back = await csvCodec.decodeStream!(encodeStream(original), new Graph());

      if (!graphContentEqual(back, original)) {
        throw new Error(`stream round-trip failed for seed ${seed}`);
      }
    }

    expect(true).toBe(true);
  });

  test('decode from adversarial tiny chunks equals non-streaming decode', async () => {
    const g = randomLpgGraph(42);
    const text = await collect(encodeStream(g));
    const streamed = await csvCodec.decodeStream!(chunked(text, 3), new Graph());
    const batched = decode(encode(g), new Graph());
    expect(graphContentEqual(streamed, g)).toBe(true);
    expect(graphContentEqual(streamed, batched)).toBe(true);
  });

  test('a multi-byte (emoji) value fed byte-by-byte as Uint8Array round-trips', async () => {
    const g = new Graph();
    g.addVertex({ id: 'n1', labels: ['Emoji'], properties: { face: '🦊 fox 日本語', tag: '✓' } });

    const text = await collect(encodeStream(g));
    const bytes = new TextEncoder().encode(text);
    const byteByByte: ChunkSource = {
      async *[Symbol.asyncIterator]() {
        for (const b of bytes) {
          yield new Uint8Array([b]);
        }
      },
    };

    const back = await csvCodec.decodeStream!(byteByByte, new Graph());
    expect(graphContentEqual(back, g)).toBe(true);
    expect(back.getVertexById('n1')!.properties.face).toBe('🦊 fox 日本語');
    expect(back.getVertexById('n1')!.properties.tag).toBe('✓');
  });

  test('large-graph pipe (25k nodes + 25k edges) through encodeStream → decodeStream', async () => {
    const g = new Graph();
    g.disableEvents();
    const nodeCount = 25000;
    const verts = [];

    for (let i = 0; i < nodeCount; i += 1) {
      verts.push(
        g.addVertex({
          id: `n${i}`,
          labels: i % 2 === 0 ? ['Even', 'Num'] : ['Odd'],
          properties: { idx: i, ratio: i / 3, label: `node-${i}`, tags: ['a', 'b'] },
        }),
      );
    }

    for (let i = 0; i < 25000; i += 1) {
      g.addEdge({
        id: `e${i}`,
        from: verts[i % nodeCount],
        to: verts[(i * 7 + 1) % nodeCount],
        labels: ['LINK'],
        properties: { w: i * 0.5 },
      });
    }

    g.enableEvents();

    const start = performance.now();
    const back = await csvCodec.decodeStream!(encodeStream(g), new Graph());
    const elapsed = performance.now() - start;

    expect(back.vertexCount).toBe(nodeCount);
    expect(back.edgeCount).toBe(25000);
    // eslint-disable-next-line no-console
    console.log(`csv stream throughput: 50k elements encode+decode in ${elapsed.toFixed(0)}ms`);
  });
});
