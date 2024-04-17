import { List } from '../../../../list/src';
import { createTestGraph } from '../../fixtures/createTestGraph';
import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  describe('Traversal tests', () => {
    const tinkerGraph = createTestTinkerGraph();
    const movieGraph = createTestGraph();

    test('a traversal can be a list', () => {
      const result = tinkerGraph
        .traverse(query => query.V('1').out('KNOWS'))
        .toList()
        .map(x => x.properties.name);

      expect(result.equals(List.from(['vadas', 'josh']))).toBeTruthy();
    });

    test('it finds the actors in twister', () => {
      const result = movieGraph
        .traverse(g =>
          g.V().hasLabel('Movie').has('title', 'Twister').in('ACTED_IN').values('name'),
        )
        .toArray();

      expect(result).toEqual([
        'Bill Paxton',
        'Helen Hunt',
        'Zach Grenier',
        'Philip Seymour Hoffman',
      ]);
    });

    test('it finds the directors of movies that had the actors from Twister in them', () => {
      const result = movieGraph
        .traverse(g =>
          g
            .V()
            .hasLabel('Movie')
            .has('title', 'Twister')
            .in('ACTED_IN')
            .out()
            .hasLabel('Movie')
            .in('DIRECTED')
            .dedup()
            .values('name'),
        )
        .toArray();

      expect(result).toEqual([
        'Ron Howard',
        'Jan de Bont',
        'Penny Marshall',
        'James L. Brooks',
        'Robert Zemeckis',
        'Werner Herzog',
        'Mike Nichols',
      ]);
    });

    test('it finds Kevin Bacon', () => {
      const result = movieGraph
        .traverse(g => g.V().has('name', 'Kevin Bacon').values('name'))
        .toArray();

      expect(result).toEqual(['Kevin Bacon']);
    });

    test('it finds one degree of Kevin Bacon', () => {
      const result = movieGraph
        .traverse(g =>
          g
            .V()
            .has('name', 'Kevin Bacon')
            .out('ACTED_IN')
            .dedup()
            .as('movie')
            .in('ACTED_IN', 'DIRECTED', 'WROTE')
            .as('person')
            .select('movie', 'person'),
        )
        .toArray();

      expect(
        result.map(
          ({ movie, person }) =>
            `(Kevin Bacon) -[:Acted In]-> ${movie.properties.title} with ${person.properties.name} in ${movie.properties.released}`,
        ),
      ).toEqual([
        '(Kevin Bacon) -[:Acted In]-> A Few Good Men with Tom Cruise in 1992',
        '(Kevin Bacon) -[:Acted In]-> A Few Good Men with Jack Nicholson in 1992',
        '(Kevin Bacon) -[:Acted In]-> A Few Good Men with Demi Moore in 1992',
        '(Kevin Bacon) -[:Acted In]-> A Few Good Men with Kevin Bacon in 1992',
        '(Kevin Bacon) -[:Acted In]-> A Few Good Men with Kiefer Sutherland in 1992',
        '(Kevin Bacon) -[:Acted In]-> A Few Good Men with Noah Wyle in 1992',
        '(Kevin Bacon) -[:Acted In]-> A Few Good Men with Cuba Gooding Jr. in 1992',
        '(Kevin Bacon) -[:Acted In]-> A Few Good Men with Kevin Pollak in 1992',
        '(Kevin Bacon) -[:Acted In]-> A Few Good Men with J.T. Walsh in 1992',
        '(Kevin Bacon) -[:Acted In]-> A Few Good Men with James Marshall in 1992',
        '(Kevin Bacon) -[:Acted In]-> A Few Good Men with Christopher Guest in 1992',
        '(Kevin Bacon) -[:Acted In]-> A Few Good Men with Aaron Sorkin in 1992',
        '(Kevin Bacon) -[:Acted In]-> A Few Good Men with Rob Reiner in 1992',
        '(Kevin Bacon) -[:Acted In]-> A Few Good Men with Aaron Sorkin in 1992',
        '(Kevin Bacon) -[:Acted In]-> Frost/Nixon with Frank Langella in 2008',
        '(Kevin Bacon) -[:Acted In]-> Frost/Nixon with Michael Sheen in 2008',
        '(Kevin Bacon) -[:Acted In]-> Frost/Nixon with Kevin Bacon in 2008',
        '(Kevin Bacon) -[:Acted In]-> Frost/Nixon with Oliver Platt in 2008',
        '(Kevin Bacon) -[:Acted In]-> Frost/Nixon with Sam Rockwell in 2008',
        '(Kevin Bacon) -[:Acted In]-> Frost/Nixon with Ron Howard in 2008',
        '(Kevin Bacon) -[:Acted In]-> Apollo 13 with Tom Hanks in 1995',
        '(Kevin Bacon) -[:Acted In]-> Apollo 13 with Kevin Bacon in 1995',
        '(Kevin Bacon) -[:Acted In]-> Apollo 13 with Ed Harris in 1995',
        '(Kevin Bacon) -[:Acted In]-> Apollo 13 with Bill Paxton in 1995',
        '(Kevin Bacon) -[:Acted In]-> Apollo 13 with Gary Sinise in 1995',
        '(Kevin Bacon) -[:Acted In]-> Apollo 13 with Ron Howard in 1995',
      ]);
    });
  });
});
