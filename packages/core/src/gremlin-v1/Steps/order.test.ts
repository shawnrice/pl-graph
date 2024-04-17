import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  test('simple ordering works', () => {
    const result = tinkerGraph.traverse(g => g.V().values('name').order()).toArray();

    expect(result).toEqual(['josh', 'lop', 'marko', 'peter', 'ripple', 'vadas']);
  });

  test.skip('we can order by desc', () => {
    // what is DESC?
    expect(tinkerGraph.query(x => x.V().values('name').order().by(DESC).toArray())).toEqual([
      'vadas',
      'ripple',
      'peter',
      'marko',
      'lop',
      'josh',
    ]);
  });

  test.skip('we can order by as property', () => {
    expect(
      tinkerGraph
        .query(x => x.V().hasLabel('person').order().by('age', ASC).values('name'))
        .toArray(),
    ).toEqual(['vadas', 'marko', 'josh', 'peter']);
  });
});
