// Shared fixture for the cross-engine comparison: a deterministic core-dialect NDJSON
// graph of `n` vertices at out-degree `deg`, with a mix of labels, property types
// (number / string / negative / map), multi-label vertices and edges (the cases that
// hide label-bucket bugs), and edge ids (so the external-id path is exercised).

const CITIES = ['oslo', 'bergen', 'trondheim', 'tromso', 'stavanger'];

// xorshift32 — deterministic, seedable, no deps.
const rng = (seed: number): (() => number) => {
  let x = seed >>> 0 || 1;

  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;

    return (x >>> 0) / 4_294_967_296;
  };
};

// ~1 in 5 edges carries a second type (multi-label edge), the case that hides
// label-bucket bugs; the rest split single-typed R / F.
const edgeLabels = (roll: number): string[] => {
  if (roll < 0.2) {
    return ['R', 'F'];
  }

  return roll < 0.5 ? ['F'] : ['R'];
};

export const fixtureNdjson = (n: number, deg: number, seed = 0x1234_5678): string => {
  const r = rng(seed);
  const lines: string[] = [];

  for (let i = 0; i < n; i++) {
    // ~1 in 4 vertices is also a VIP (multi-label), so `(:Person)` and `(:VIP)` differ.
    const labels = r() < 0.25 ? ['Person', 'VIP'] : ['Person'];
    const props = {
      name: `p${i}`,
      age: 18 + Math.floor(r() * 60),
      score: Math.round(r() * 100) - 20, // some negatives
      city: CITIES[Math.floor(r() * CITIES.length)],
      m: { k: i % 7, tag: r() < 0.5 ? 'x' : 'y' },
    };

    lines.push(JSON.stringify({ type: 'node', id: String(i), labels, properties: props }));
  }

  let eid = 0;

  for (let i = 0; i < n; i++) {
    for (let d = 0; d < deg; d++) {
      const to = Math.floor(r() * n);

      lines.push(
        JSON.stringify({
          type: 'edge',
          id: `e${eid++}`,
          labels: edgeLabels(r()),
          from: String(i),
          to: String(to),
          properties: { w: Math.floor(r() * 10), since: 2000 + Math.floor(r() * 25) },
        }),
      );
    }
  }

  return lines.join('\n');
};

// The differential-fuzz edge-case graph (tiny, but carries the multi-label vertex/edge
// and nested-map cases that a random fixture may under-sample).
export const EDGE_CASE_NDJSON = [
  '{"type":"node","id":"1","labels":["T"],"properties":{"n":3,"s":"a","x":-1,"m":{"k":1,"j":"q"}}}',
  '{"type":"node","id":"2","labels":["T"],"properties":{"n":7,"s":"z","x":4,"m":{"k":2,"j":"r"}}}',
  '{"type":"node","id":"3","labels":["T","U"],"properties":{"n":5,"s":"m","x":2,"m":{"k":3,"j":"s"}}}',
  '{"type":"edge","id":"e1","labels":["E"],"from":"1","to":"2","properties":{"w":2}}',
  '{"type":"edge","id":"e2","labels":["E","F"],"from":"2","to":"3","properties":{"w":5}}',
].join('\n');
