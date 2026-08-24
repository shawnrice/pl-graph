// Tab-completion candidates. Kept declarative + the completer pure, so both are
// unit-testable without a live readline.

export const GQL_KEYWORDS: readonly string[] = [
  'MATCH',
  'OPTIONAL',
  'WHERE',
  'RETURN',
  'WITH',
  'LET',
  'ORDER BY',
  'SKIP',
  'LIMIT',
  'DISTINCT',
  'AS',
  'AND',
  'OR',
  'NOT',
  'XOR',
  'IS',
  'NULL',
  'TRUE',
  'FALSE',
  'IN',
  'CONTAINS',
  'STARTS WITH',
  'ENDS WITH',
  'ASC',
  'DESC',
  'NULLS',
  'FIRST',
  'LAST',
  'INSERT',
  'SET',
  'REMOVE',
  'DELETE',
  'DETACH',
  'MERGE',
  'CALL',
  'YIELD',
  'UNION',
  'UNION ALL',
  'EXCEPT',
  'INTERSECT',
  'EXISTS',
  'COUNT',
  'ANY SHORTEST',
  'GROUP BY',
  'HAVING',
  'FOR',
  'CASE',
  'WHEN',
  'THEN',
  'ELSE',
  'END',
];

// Scalar functions worth completing (not exhaustive — the common ones).
export const GQL_FUNCTIONS: readonly string[] = [
  'count',
  'sum',
  'avg',
  'min',
  'max',
  'collect',
  'stddev',
  'percentile_cont',
  'labels',
  'type',
  'id',
  'keys',
  'properties',
  'size',
  'upper',
  'lower',
  'trim',
  'substring',
  'char_length',
  'date',
  'datetime',
  'localdatetime',
  'duration',
  'current_date',
  'current_timestamp',
  'abs',
  'ceil',
  'floor',
  'round',
  'sqrt',
  'power',
  'coalesce',
  'nullif',
];

export const GREMLIN_STEPS: readonly string[] = [
  'V',
  'E',
  'out',
  'in',
  'both',
  'outE',
  'inE',
  'bothE',
  'outV',
  'inV',
  'otherV',
  'has',
  'hasLabel',
  'hasId',
  'hasKey',
  'hasValue',
  'not',
  'where',
  'and',
  'or',
  'values',
  'valueMap',
  'elementMap',
  'label',
  'id',
  'key',
  'value',
  'count',
  'sum',
  'mean',
  'max',
  'min',
  'fold',
  'unfold',
  'group',
  'groupCount',
  'order',
  'by',
  'limit',
  'range',
  'tail',
  'skip',
  'dedup',
  'path',
  'simplePath',
  'cyclicPath',
  'repeat',
  'until',
  'emit',
  'times',
  'select',
  'as',
  'project',
  'coalesce',
  'choose',
  'optional',
  'union',
  'local',
  'cap',
  'sack',
  'match',
  'aggregate',
  'addV',
  'addE',
  'property',
  'drop',
  'constant',
  'inject',
];

export const META_COMMANDS: readonly string[] = [
  '\\?',
  '\\q',
  '\\r',
  '\\l',
  '\\c',
  '\\d',
  '\\dv',
  '\\de',
  '\\i',
  '\\o',
  '\\format',
  '\\timing',
  '\\gql',
  '\\gremlin',
  '\\js',
  '\\clock',
  '\\save',
];

export type Mode = 'gql' | 'gremlin' | 'js';

/** The last whitespace-or-paren-delimited token of `line` (what a tab completes). */
export const lastToken = (line: string): string => {
  const m = line.match(/[A-Za-z_\\][A-Za-z0-9_\\]*$/);

  return m ? m[0] : '';
};

/**
 * Completions for the current input. Returns `[matches, token]` in the shape
 * `readline`'s completer expects. `labels` are the live graph's labels (mode-gql).
 */
export const complete = (
  line: string,
  mode: Mode,
  labels: readonly string[] = [],
): [string[], string] => {
  const token = lastToken(line);

  if (line.trimStart().startsWith('\\')) {
    const hits = META_COMMANDS.filter((c) => c.startsWith(token));

    return [hits, token];
  }

  const poolFor = (): readonly string[] => {
    if (mode === 'gremlin') {
      return GREMLIN_STEPS;
    }

    if (mode === 'js') {
      return [];
    }

    return [...GQL_KEYWORDS, ...GQL_FUNCTIONS, ...labels];
  };
  const pool = poolFor();

  if (token === '') {
    return [[], token];
  }

  const lower = token.toLowerCase();
  const hits = pool.filter((c) => c.toLowerCase().startsWith(lower));

  return [hits, token];
};
