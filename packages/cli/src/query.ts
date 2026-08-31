import { formatRows, type Row } from '@lenke/inspect';
import type { RustGraph } from '@lenke/native';

export type Lang = 'gql' | 'gremlin';

// A Gremlin traversal starts from the `g` source (`g.V()…`); everything else is
// treated as GQL. `.gql` / `.gremlin` in the REPL force the choice explicitly.
export const classify = (text: string): Lang =>
  /^g\s*\./.test(text.trimStart()) ? 'gremlin' : 'gql';

// Gremlin returns a heterogeneous stream (elements, maps, scalars); wrap bare
// scalars in a `value` column so anything renders as a table.
export const asRows = (items: readonly unknown[]): Row[] =>
  items.map((item) =>
    item !== null && typeof item === 'object' && !Array.isArray(item)
      ? (item as Row)
      : { value: item },
  );

export type QueryResult = { lang: Lang; output: string };

export const runQuery = (
  graph: RustGraph,
  text: string,
  lang: Lang = classify(text),
  color?: boolean,
): QueryResult => {
  const output =
    lang === 'gremlin'
      ? formatRows(asRows(graph.gremlin(text)), { color })
      : formatRows(graph.query(text), { color });

  return { lang, output };
};
