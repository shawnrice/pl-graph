// Statement execution: the write clauses (INSERT/SET/REMOVE/DELETE, _MERGE),
// per-clause processing that threads bindings through a linear pipeline, and the
// set operators (UNION/EXCEPT/INTERSECT). Extracted from the executor.

import { runAlgorithmSync } from '@lenke/core';
import type { AlgorithmConfig, AlgorithmName, Edge, Graph, Vertex } from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';
import { filter, flatMap, map, toArray } from '@lenke/fp';

import type { Query, SetOp, TxControl } from '../ast.js';
// Back-edges into the executor trunk (compiled types + helpers), used lazily.
import type {
  Binding,
  CCallInline,
  CCallNamed,
  CDelete,
  CFor,
  CInsert,
  CInsertNode,
  CLinear,
  CMatch,
  CMerge,
  CPath,
  CProp,
  CRemove,
  CSet,
  CSetItem,
  EvalEnv,
  Params,
  Row,
} from '../executor.js';
import {
  applyProjection,
  compareSort,
  isEdge,
  isElement,
  isVertex,
  resolveCount,
  valueKey,
} from '../executor.js';
import { hasIncidentEdges } from '../graph-queries.js';
import { matchPattern } from './matching.js';
import { asTruth } from './scalars.js';

export const evalProps = (
  props: readonly CProp[],
  b: Binding,
  params: Params,
  graph: Graph,
): Record<string, unknown> => {
  const env: EvalEnv = { binding: b, params, graph };
  const out: Record<string, unknown> = {};

  for (const { key, value } of props) {
    out[key] = value(env);
  }

  return out;
};

/**
 * Insert a vertex, using a string `id` property as the element's identity — so
 * `element_id(n)` equals it and serialization round-trips by domain identity
 * instead of a random synthetic id, while `id` is still stored as an ordinary
 * property. A non-string or absent id mints a synthetic one; a duplicate string id
 * throws (ids are unique). Mirrors the native `insert_vertex_with_id`.
 */
export const insertVertexWithId = (
  graph: Graph,
  labels: readonly string[],
  properties: Record<string, unknown>,
): Vertex => {
  const { id } = properties;

  if (typeof id === 'string') {
    if (graph.getVertexById(id) !== null) {
      throw new LenkeError(
        `an element with id '${id}' already exists — a string \`id\` property is the ` +
          `element's unique identity; use _MERGE to upsert, or a fresh id`,
        { code: ErrorCode.ConstraintViolation },
      );
    }

    return graph.addVertex({ id, labels: [...labels], properties });
  }

  return graph.addVertex({ labels: [...labels], properties });
};

/**
 * Insert an edge, using a string `id` property as its external identity — the edge
 * analogue of {@link insertVertexWithId}. Edge ids are unique among edges; a
 * duplicate string id throws. Mirrors native `insert_edge_with_id`.
 */
export const insertEdgeWithId = (
  graph: Graph,
  from: Vertex,
  to: Vertex,
  labels: readonly string[],
  properties: Record<string, unknown>,
): Edge => {
  const { id } = properties;

  if (typeof id === 'string') {
    if (graph.getEdgeById(id) !== null) {
      throw new LenkeError(
        `an element with id '${id}' already exists — a string \`id\` property is the ` +
          `element's unique identity; use _MERGE to upsert, or a fresh id`,
        { code: ErrorCode.ConstraintViolation },
      );
    }

    return graph.addEdge({ id, from, to, labels: [...labels], properties });
  }

  return graph.addEdge({ from, to, labels: [...labels], properties });
};

/** Create a node from a pattern, reusing an already-bound variable. */
export const ensureNode = (
  graph: Graph,
  binding: Map<string, unknown>,
  node: CInsertNode,
  params: Params,
): Vertex => {
  if (node.variable && binding.has(node.variable)) {
    return binding.get(node.variable) as Vertex;
  }

  const properties = evalProps(node.props, binding, params, graph);

  // A plain INSERT that breaks a unique constraint is rejected — but the check is
  // deferred to commit (via `addVertex`'s constraint chokepoint + `runDeferredChecks`),
  // not eager, so a transient duplicate resolved before commit is allowed. This
  // matches the native engine's deferred-check (transaction) semantics; an eager check here
  // wrongly rejected `INSERT` + later `DELETE` of the dup within one transaction
  // `_MERGE` still reconciles instead (docs/design/gql-extensions.md §3).
  const vertex = insertVertexWithId(graph, node.labels, properties);

  if (node.variable) {
    binding.set(node.variable, vertex);
  }

  return vertex;
};

export const runInsert = (
  graph: Graph,
  clause: CInsert,
  binding: Binding,
  params: Params,
): Binding => {
  const out = new Map(binding);

  for (const pattern of clause.patterns) {
    let prev = ensureNode(graph, out, pattern.start, params);

    for (const { rel, node } of pattern.segments) {
      const next = ensureNode(graph, out, node, params);
      const [from, to] = rel.direction === 'in' ? [next, prev] : [prev, next];
      const edge = insertEdgeWithId(
        graph,
        from,
        to,
        rel.labels,
        evalProps(rel.props, out, params, graph),
      );

      if (rel.variable) {
        out.set(rel.variable, edge);
      }

      prev = next;
    }
  }

  return out;
};

/**
 * Infer the conflict key for `_MERGE`: the single unique-constrained key present
 * in the pattern's properties. No applicable constraint → error (can't define
 * "the key"); more than one → ambiguous. See docs/design/gql-extensions.md §2.2.
 */
export const inferMergeKey = (
  graph: Graph,
  labels: readonly string[],
  properties: Record<string, unknown>,
): { label: string; key: string; value: unknown } => {
  const candidates: { label: string; key: string; value: unknown }[] = [];

  for (const label of labels) {
    for (const key of graph.uniqueKeys(label)) {
      if (key in properties) {
        candidates.push({ label, key, value: properties[key] });
      }
    }
  }

  if (candidates.length === 0) {
    throw new LenkeError(
      `_MERGE needs a unique constraint on the pattern's label(s) [${labels.join(', ')}] to define the key — declare one with createUniqueConstraint`,
      { code: ErrorCode.InvalidGraphOp },
    );
  }

  if (candidates.length > 1) {
    throw new LenkeError(
      `_MERGE key is ambiguous: the pattern touches multiple unique constraints (${candidates
        .map((c) => `${c.label}.${c.key}`)
        .join(', ')}) — narrow it to one`,
      { code: ErrorCode.InvalidGraphOp },
    );
  }

  return candidates[0];
};

/** Apply `_ON_CREATE` / `_ON_UPDATE` SET items to the merged `vertex`. */
// Apply `_ON_CREATE` / `_ON_UPDATE` SET items to the node or edge each item's
// variable resolves to in `binding` (mirrors `runSet`).
export const applyMergeSets = (
  graph: Graph,
  items: readonly CSetItem[],
  binding: Binding,
  params: Params,
): void => {
  for (const item of items) {
    const el = binding.get(item.variable);

    if (!isElement(el)) {
      continue;
    }

    if ('label' in item) {
      if (isEdge(el)) {
        graph.addLabelToEdge(item.label, el);
      } else {
        graph.addLabelToVertex(item.label, el);
      }
    } else {
      el.setProperty(item.key, item.value({ binding, params, graph }));
    }
  }
};

// Resolve a `_MERGE` edge endpoint: the vertex matched by its unique-constraint
// key. Throws (InvalidGraphOp) if no key can be inferred or no vertex matches.
export const resolveMergeEndpoint = (
  graph: Graph,
  node: CInsertNode,
  binding: Binding,
  params: Params,
): Vertex => {
  // An endpoint bound by a preceding clause — `MATCH (a), (b) _MERGE (a)-[:R]->(b)`,
  // the natural way to merge an edge between two known vertices — is already a
  // resolved vertex. Use it directly rather than re-inferring a unique key from the
  // (empty) node pattern, which would throw and made the bound-variable form of
  // edge `_MERGE` unusable. Mirrors the native engine's `resolve_merge_endpoint`.
  if (node.variable !== undefined) {
    const bound = binding.get(node.variable);

    if (isVertex(bound)) {
      return bound;
    }
  }

  const properties = evalProps(node.props, binding, params, graph);
  const { label, key, value } = inferMergeKey(graph, node.labels, properties);
  const found = graph.uniqueLookup(label, key, value);

  if (found === undefined) {
    throw new LenkeError(
      `_MERGE: endpoint (:${node.labels.join('&')} {${key}: …}) not found — its key must match an existing vertex`,
      { code: ErrorCode.InvalidGraphOp },
    );
  }

  return found;
};

// `_MERGE` edge form (v1): match both endpoints by key, then upsert the single
// edge between them keyed structurally by (from, to, type). Dispositions apply to
// the edge (which has no key prop, so the default clobbers all its props).
export const runMergeEdge = (
  graph: Graph,
  clause: CMerge,
  binding: Binding,
  params: Params,
): Binding => {
  const out = new Map(binding);
  const [seg] = clause.pattern.segments;
  const startV = resolveMergeEndpoint(graph, clause.pattern.start, out, params);
  const endV = resolveMergeEndpoint(graph, seg.node, out, params);
  const [from, to] = seg.rel.direction === 'in' ? [endV, startV] : [startV, endV];
  const [relType] = seg.rel.labels;

  if (relType === undefined) {
    throw new LenkeError('_MERGE: an edge must carry exactly one type', {
      code: ErrorCode.InvalidGraphOp,
    });
  }

  const edgeProps = evalProps(seg.rel.props, out, params, graph);

  // Bind the resolved endpoints so the dispositions can read them.
  if (clause.pattern.start.variable) {
    out.set(clause.pattern.start.variable, startV);
  }

  if (seg.node.variable) {
    out.set(seg.node.variable, endV);
  }

  const existing = graph.findEdge(from, to, relType);
  let edge: Edge;

  if (existing === undefined) {
    edge = graph.addEdge({ from, to, labels: [relType], properties: edgeProps });

    if (seg.rel.variable) {
      out.set(seg.rel.variable, edge);
    }

    if (clause.onCreate) {
      applyMergeSets(graph, clause.onCreate, out, params);
    }
  } else {
    edge = existing;

    if (seg.rel.variable) {
      out.set(seg.rel.variable, edge);
    }

    const disp = clause.onUpdate;

    if (disp === undefined) {
      // An edge has no key prop → the default clobbers all its props.
      for (const [k, v] of Object.entries(edgeProps)) {
        edge.setProperty(k, v);
      }
    } else if (disp.kind === 'set') {
      const passes =
        disp.where === undefined || asTruth(disp.where({ binding: out, params, graph })) === true;

      if (passes) {
        applyMergeSets(graph, disp.items, out, params);
      }
    }
    // disp.kind === 'nothing' → leave the edge untouched.
  }

  if (seg.rel.variable) {
    out.set(seg.rel.variable, edge);
  }

  return out;
};

// `_MERGE` keyed upsert. Node form: match by the constraint key; on miss insert
// the pattern (key + payload) then `_ON_CREATE`; on hit apply the update
// disposition — default clobbers the non-key payload, `_ON_UPDATE SET … [WHERE]`
// replaces it, `_ON_UPDATE_NOTHING` leaves it. One segment → the edge form above;
// multi-hop compound patterns are deferred (v2).
export const runMerge = (
  graph: Graph,
  clause: CMerge,
  binding: Binding,
  params: Params,
): Binding => {
  if (clause.pattern.segments.length === 1) {
    return runMergeEdge(graph, clause, binding, params);
  }

  if (clause.pattern.segments.length > 1) {
    throw new LenkeError('_MERGE multi-hop compound patterns are not yet supported (v2)', {
      code: ErrorCode.NotImplemented,
    });
  }

  const out = new Map(binding);
  const node = clause.pattern.start;
  const properties = evalProps(node.props, out, params, graph);
  const { label, key, value } = inferMergeKey(graph, node.labels, properties);
  const existing = graph.uniqueLookup(label, key, value);

  let vertex: Vertex;

  if (existing === undefined) {
    vertex = insertVertexWithId(graph, node.labels, properties);

    if (node.variable) {
      out.set(node.variable, vertex);
    }

    if (clause.onCreate) {
      applyMergeSets(graph, clause.onCreate, out, params);
    }
  } else {
    vertex = existing;

    if (node.variable) {
      out.set(node.variable, vertex);
    }

    const disp = clause.onUpdate;

    if (disp === undefined) {
      // Default clobber: write every non-key payload prop to the pattern's value.
      for (const [k, v] of Object.entries(properties)) {
        if (k !== key) {
          vertex.setProperty(k, v);
        }
      }
    } else if (disp.kind === 'set') {
      // An explicit update replaces the default, gated by WHERE if present.
      const passes =
        disp.where === undefined || asTruth(disp.where({ binding: out, params, graph })) === true;

      if (passes) {
        applyMergeSets(graph, disp.items, out, params);
      }
    }
    // disp.kind === 'nothing' → leave the existing element untouched.
  }

  if (node.variable) {
    out.set(node.variable, vertex);
  }

  return out;
};

// Both labels and properties go through the element's index-maintaining
// mutators (`addLabelTo*` / `setProperty`) so the graph's label and property
// value indexes stay consistent — a later MATCH seeds from `vertexPropertyIndex`,
// so a direct `el.properties =` write would leave that index stale (and skip
// mutation events).
export const runSet = (graph: Graph, clause: CSet, binding: Binding, params: Params): void => {
  for (const item of clause.items) {
    const el = binding.get(item.variable);

    if (!isElement(el)) {
      continue;
    }

    if ('label' in item) {
      if (isEdge(el)) {
        graph.addLabelToEdge(item.label, el);
      } else {
        graph.addLabelToVertex(item.label, el);
      }
    } else if (item.key === 'id' && el.id === el.properties.id) {
      // An element keyed by a string `id` has that id as its identity (external id
      // === the `id` property), fixed at creation — re-keying it would break
      // `element_id` / round-trip stability, so reject the SET. A numeric/absent
      // `id` is an ordinary (possibly unique-constrained) property and stays
      // SET-able. Mirrors native `vertex_id_is_identity` / `edge_id_is_identity`.
      throw new LenkeError(
        "cannot SET `id`: a string `id` is the element's identity and is fixed at " +
          'creation — insert a new element with the new id instead',
        { code: ErrorCode.InvalidGraphOp },
      );
    } else {
      const value = item.value({ binding, params, graph });

      // A SET that collides under a unique constraint is rejected via the core
      // property-write chokepoint (`assertUniqueOnSet`), which defers the check to
      // commit inside a transaction. Deferring (rather than an eager check here)
      // lets a transient collision that is reverted before commit succeed, matching
      // the native engine's transaction semantics. Constraints are vertex-only.
      el.setProperty(item.key, value);
    }
  }
};

export const runRemove = (graph: Graph, clause: CRemove, binding: Binding): void => {
  for (const item of clause.items) {
    const el = binding.get(item.variable);

    if (!isElement(el)) {
      continue;
    }

    if ('label' in item) {
      if (isEdge(el)) {
        graph.removeLabelFromEdge(item.label, el);
      } else {
        graph.removeLabelFromVertex(item.label, el);
      }
    } else {
      el.removeProperty(item.key);
    }
  }
};

export const runDelete = (
  graph: Graph,
  clause: CDelete,
  binding: Binding,
  params: Params,
): void => {
  for (const target of clause.targets) {
    const el = target({ binding, params, graph });

    if (isEdge(el)) {
      graph.removeEdge(el);
    } else if (isElement(el)) {
      const vertex = el as Vertex;

      // Plain DELETE must not orphan relationships: deleting a still-connected
      // node is a graph violation unless the user opted into DETACH (which
      // cascades the incident edges).
      if (!clause.detach && hasIncidentEdges(graph, vertex)) {
        throw new LenkeError(
          'Cannot delete a node that still has relationships; use DETACH DELETE',
          { code: ErrorCode.InvalidGraphOp },
        );
      }

      graph.removeVertex(vertex);
    }
  }
};

// --- clause processing -------------------------------------------------------

/**
 * How much it costs to START this pattern, given what is already bound. Lower is
 * better. Mirrors native `pattern_rank` in `gql/eval/pathfind.rs` exactly — the
 * two must agree or the engines emit rows in different orders.
 */
const patternRank = (p: CPath, b: Binding): number => {
  const bound = (v?: string) => v !== undefined && b.has(v);

  if (
    bound(p.start.variable) ||
    p.segments.some((s) => bound(s.rel.variable) || bound(s.node.variable))
  ) {
    return 0; // continues an existing binding — no fresh scan at all
  }

  return p.start.label !== undefined || p.start.pred.props.length > 0 ? 1 : 2;
};

/** Position within `remaining` of the next pattern to extend into. */
const pickPattern = (patterns: readonly CPath[], remaining: readonly number[], b: Binding) => {
  let best = 0;
  let bestRank = 3;

  for (let i = 0; i < remaining.length; i++) {
    const rank = patternRank(patterns[remaining[i]], b);

    if (rank < bestRank) {
      bestRank = rank;
      best = i;
    }

    if (rank === 0) {
      break; // can't do better than continuing an existing binding
    }
  }

  return best;
};

/**
 * Extend a binding through the remaining patterns, choosing at each step rather
 * than following the order they were written.
 *
 * `remaining` is copied rather than mutated-and-restored: these are generators,
 * so a shared mutable done-mask would be corrupted the moment a consumer
 * interleaved two of them.
 */
const visitRemaining = function* (
  graph: Graph,
  patterns: readonly CPath[],
  remaining: readonly number[],
  binding: Binding,
  params: Params,
): Iterable<Binding> {
  if (remaining.length === 0) {
    yield binding;

    return;
  }

  const pick = pickPattern(patterns, remaining, binding);
  const rest = remaining.filter((_, i) => i !== pick);

  for (const b of matchPattern(graph, patterns[remaining[pick]], binding, params)) {
    yield* visitRemaining(graph, patterns, rest, b, params);
  }
};

/**
 * Extend a binding through every pattern of a MATCH clause, then filter WHERE.
 * Fully lazy: bindings flow one at a time to whatever consumes the result, so a
 * clause that expands to millions of them never materializes them.
 *
 * Patterns are visited in the order [`pickPattern`] chooses. Running a pattern
 * whose start is already bound before one that needs a fresh scan is what keeps
 * two spellings of the same query from costing wildly different amounts — see
 * `docs/design/query-ir.md`; the anchored and unanchored spellings measured
 * 121,336x apart at 300k vertices before this existed. Ties keep the written
 * order, so a query whose patterns are equally cheap to start emits rows exactly
 * as it did before.
 */
export const matchClauseBindings = (
  graph: Graph,
  clause: CMatch,
  binding: Binding,
  params: Params,
): Iterable<Binding> => {
  // One pattern is the overwhelmingly common case and has nothing to choose
  // between — keep it on the flat `flatMap` with no generator frame per row.
  const stream: Iterable<Binding> =
    clause.patterns.length === 1
      ? flatMap((b: Binding) => matchPattern(graph, clause.patterns[0], b, params), [binding])
      : visitRemaining(
          graph,
          clause.patterns,
          clause.patterns.map((_, i) => i),
          binding,
          params,
        );

  return clause.where === undefined
    ? stream
    : filter(
        (b: Binding) => asTruth(clause.where!({ binding: b, params, graph })) === true,
        stream,
      );
};

/** Per-incoming-binding: stream its matches, or (for OPTIONAL) one null-filled row. */
export const matchOrOptional = function* (
  graph: Graph,
  clause: CMatch,
  binding: Binding,
  params: Params,
): Iterable<Binding> {
  let matched = false;

  for (const m of matchClauseBindings(graph, clause, binding, params)) {
    matched = true;

    yield m;
  }

  if (!matched && clause.optional) {
    // No match: keep the row with the pattern's new variables set to null.
    const filled = new Map(binding);

    for (const v of clause.nullVars) {
      if (!filled.has(v)) {
        filled.set(v, null);
      }
    }

    yield filled;
  }
};

/** Lazily expand a binding stream through a MATCH — no intermediate array. */
export const runMatch = (
  graph: Graph,
  clause: CMatch,
  bindings: Iterable<Binding>,
  params: Params,
): Iterable<Binding> =>
  flatMap((binding: Binding) => matchOrOptional(graph, clause, binding, params), bindings);

/**
 * Lazily unwind a list per incoming binding — one row per element (ISO GQL's
 * FOR / UNWIND). A list unwinds its elements; null/undefined yields zero rows;
 * any other scalar unwinds as a one-element list. Matches the Rust engine
 * byte-for-byte. ORDINALITY counts from 1, OFFSET from 0.
 */
/**
 * The built-in procedure catalog: procedure name → its algorithm and non-`node`
 * result column. Output columns are always `[node, <result>]`. Mirrors native
 * `procedure_spec` in plan.rs.
 */
export const PROCEDURES: Record<string, { algo: AlgorithmName; resultColumn: string }> = {
  pagerank: { algo: 'pagerank', resultColumn: 'score' },
  personalized_pagerank: { algo: 'personalizedPagerank', resultColumn: 'score' },
  connected_components: { algo: 'connectedComponents', resultColumn: 'componentId' },
  strongly_connected_components: {
    algo: 'stronglyConnectedComponents',
    resultColumn: 'componentId',
  },
  on_cycle: { algo: 'onCycle', resultColumn: 'onCycle' },
  label_propagation: { algo: 'labelPropagation', resultColumn: 'label' },
  peer_pressure: { algo: 'peerPressure', resultColumn: 'cluster' },
  degree: { algo: 'degree', resultColumn: 'degree' },
  betweenness: { algo: 'betweenness', resultColumn: 'centrality' },
  closeness: { algo: 'closeness', resultColumn: 'centrality' },
  shortest_path: { algo: 'shortestPath', resultColumn: 'distance' },
  neighbor_aggregate: { algo: 'neighborAggregate', resultColumn: 'vector' },
};

export const procedureSpec = (name: string): { algo: AlgorithmName; resultColumn: string } | null =>
  PROCEDURES[name] ?? null;

/**
 * For an unknown procedure name, the canonical snake_case name it most likely
 * meant — matched by ignoring case and `_` separators, so a camelCase spelling
 * (`pageRank`, `connectedComponents`) resolves to its surface name. `null` when
 * nothing plausibly matches. Mirrors the native `suggest_procedure` so both
 * engines' "did you mean" faults read identically.
 */
export const normProcName = (s: string): string => s.replaceAll('_', '').toLowerCase();

export const suggestProcedure = (name: string): string | null => {
  const target = normProcName(name);

  return Object.keys(PROCEDURES).find((n) => normProcName(n) === target) ?? null;
};

/** The accepted `CALL <algo>({...})` config keys and the value type each expects.
 * Order is fixed so the "did you mean" tie-break matches the native engine. */
export const ALGO_CONFIG_TYPES: ReadonlyArray<
  readonly [string, 'string' | 'number' | 'stringList' | 'boolean']
> = [
  ['edgeLabel', 'string'],
  ['direction', 'string'],
  ['weightProperty', 'string'],
  ['dampingFactor', 'number'],
  ['iterations', 'number'],
  ['pivots', 'number'],
  ['seedProperty', 'string'],
  ['source', 'string'],
  ['sourceNodes', 'stringList'],
  ['target', 'string'],
  ['writeProperty', 'string'],
  ['algorithm', 'string'],
  ['heuristicProperty', 'string'],
  ['feature', 'string'],
  ['op', 'string'],
  ['includeSelf', 'boolean'],
  ['norm', 'string'],
];

/** Case-insensitive Levenshtein edit distance — a plain DP over code points,
 * matching the native `edit_distance` so both engines suggest identically. */
export const editDistance = (a: string, b: string): number => {
  const x = Array.from(a.toLowerCase());
  const y = Array.from(b.toLowerCase());
  let prev = Array.from({ length: y.length + 1 }, (_, i) => i);
  const cur = new Array<number>(y.length + 1);

  for (let i = 0; i < x.length; i++) {
    cur[0] = i + 1;

    for (let j = 0; j < y.length; j++) {
      const cost = x[i] === y[j] ? 0 : 1;
      cur[j + 1] = Math.min(prev[j + 1] + 1, cur[j] + 1, prev[j] + cost);
    }

    prev = cur.slice();
  }

  return prev[y.length];
};

/** For an unknown config key, the closest known key within edit distance 2 (else
 * null). Scans in fixed order so ties resolve to the earliest — identical to native. */
export const suggestConfigKey = (name: string): string | null => {
  let best: string | null = null;
  let bestDist = 3;

  for (const [key] of ALGO_CONFIG_TYPES) {
    const d = editDistance(name, key);

    if (d <= 2 && d < bestDist) {
      best = key;
      bestDist = d;
    }
  }

  return best;
};

/** Set one algorithm-config field from a CALL config-map entry. An unknown key
 * (with a "did you mean" hint) or a wrong-typed value is an error — a silently
 * dropped key once hid the `pivots` bug. Mirrors native `apply_algo_config`. */
export const applyAlgoConfig = (cfg: AlgorithmConfig, key: string, v: unknown): void => {
  const spec = ALGO_CONFIG_TYPES.find(([k]) => k === key);

  if (spec === undefined) {
    const s = suggestConfigKey(key);

    throw new LenkeError(
      s ? `unknown config key '${key}' (did you mean '${s}'?)` : `unknown config key '${key}'`,
      { code: ErrorCode.InvalidValue },
    );
  }

  const [, expected] = spec;
  const okByType: Record<typeof expected, boolean> = {
    string: typeof v === 'string',
    number: typeof v === 'number',
    stringList: Array.isArray(v),
    boolean: typeof v === 'boolean',
  };

  if (!okByType[expected]) {
    const label = expected === 'stringList' ? 'a list' : `a ${expected}`;

    throw new LenkeError(`config key '${key}' expects ${label}`, { code: ErrorCode.InvalidValue });
  }

  (cfg as Record<string, unknown>)[key] =
    expected === 'stringList' ? (v as unknown[]).filter((x) => typeof x === 'string') : v;
};

/**
 * `[OPTIONAL] CALL name(config) YIELD …`: run the algorithm once (uncorrelated),
 * then cross-join its rows into the binding stream, binding each yielded column.
 * OPTIONAL keeps the outer row (null-filled) when the procedure yields nothing.
 */
export const runCall = (
  graph: Graph,
  clause: CCallNamed,
  bindings: Iterable<Binding>,
  params: Params,
): Iterable<Binding> => {
  if (!clause.algo) {
    const suggestion = suggestProcedure(clause.procName);
    const msg = suggestion
      ? `unknown procedure: ${clause.procName} (did you mean '${suggestion}'?)`
      : `unknown procedure: ${clause.procName}`;

    throw new LenkeError(msg, { code: ErrorCode.Unsupported });
  }

  const config: AlgorithmConfig = {};
  const scratch: Binding = new Map();

  for (const c of clause.config) {
    applyAlgoConfig(config, c.key, c.value({ binding: scratch, params, graph }));
  }

  const rows = runAlgorithmSync(clause.algo, config, graph) as Array<Record<string, unknown>>;

  // Validate the YIELD columns against what the procedure actually exposes: a
  // vertex handle as `node`, plus its single result column. Anything else is an
  // error — previously an undeclared column (`YIELD nodeId`, or any typo) bound
  // silently to `undefined`, so the query returned rows with that column simply
  // missing. A silent wrong answer, and a divergence: native raises
  // E_INVALID_VALUE here. Checked after the algorithm runs, mirroring native, so a
  // `writeProperty` side effect lands identically on both engines.
  const resultColumn = procedureSpec(clause.procName)?.resultColumn;

  for (const bind of clause.binds) {
    if (bind.column !== 'node' && bind.column !== resultColumn) {
      throw new LenkeError(
        `procedure \`${clause.procName}\` has no output column \`${bind.column}\``,
        { code: ErrorCode.InvalidValue },
      );
    }
  }

  // Materialize the outer bindings first: the call may write a property, and the
  // outer stream must be read against the pre-write graph.
  const outer = toArray(bindings);

  return flatMap((binding: Binding) => {
    if (rows.length === 0 && clause.optional) {
      const b = new Map(binding);

      for (const bind of clause.binds) {
        b.set(bind.var, null);
      }

      return [b];
    }

    return rows.map((r) => {
      const b = new Map(binding);

      for (const bind of clause.binds) {
        // `node` binds as the live Vertex handle (so it hydrates only when
        // actually returned whole, and `node.name` / further MATCH work);
        // mirrors native's `Val::Node`. Other columns are the raw value.
        const value =
          bind.column === 'node' ? graph.verticesById.get(r.node as string) : r[bind.column];
        b.set(bind.var, value);
      }

      return b;
    });
  }, outer);
};

/**
 * `[OPTIONAL] CALL (scope) { … }`: run the nested query once per outer row
 * (correlated / lateral), seeding it with only the imported scope variables, and
 * merge the nested RETURN columns back — duplicating the outer row per nested row.
 * OPTIONAL keeps the outer row (nested columns null-filled) when the subquery is
 * empty; a non-OPTIONAL empty subquery drops the outer row.
 */
export const runCallInline = (
  graph: Graph,
  clause: CCallInline,
  bindings: Iterable<Binding>,
  params: Params,
): Iterable<Binding> =>
  flatMap((outer: Binding) => {
    // Import only the scoped variables into the subquery's initial binding.
    const seed = new Map<string, unknown>();

    for (const v of clause.scope) {
      if (outer.has(v)) {
        seed.set(v, outer.get(v));
      }
    }

    let nested = runLinearClauses(clause.body, graph, params, new Map(seed));

    // Fold in any set-op parts (`… UNION/EXCEPT/INTERSECT …`), each run against
    // the same seed, matching the top-level set-op semantics.
    for (const { op, part } of clause.bodyMore) {
      nested = combineRows(op, nested, runLinearClauses(part, graph, params, new Map(seed)));
    }

    if (nested.length === 0 && clause.optional) {
      const b = new Map(outer);

      for (const col of clause.returnColumns) {
        b.set(col, null);
      }

      return [b];
    }

    return nested.map((row) => {
      const b = new Map(outer);

      for (const [k, val] of Object.entries(row)) {
        b.set(k, val);
      }

      return b;
    });
  }, bindings);

export const runFor = (
  graph: Graph,
  clause: CFor,
  bindings: Iterable<Binding>,
  params: Params,
): Iterable<Binding> =>
  flatMap((binding: Binding) => {
    const listv = clause.list({ binding, params, graph });
    let elems: unknown[];

    if (listv === null || listv === undefined) {
      elems = [];
    } else if (Array.isArray(listv)) {
      elems = listv;
    } else {
      elems = [listv];
    }

    return elems.map((elem, i) => {
      const b = new Map(binding);
      b.set(clause.alias, elem);

      if (clause.ordinality) {
        b.set(clause.ordinality.var, clause.ordinality.kind === 'ordinality' ? i + 1 : i);
      }

      return b;
    });
  }, bindings);

export const mapToRow = (b: Binding): Row => {
  const row: Row = {};

  for (const [k, v] of b) {
    row[k] = v;
  }

  return row;
};

export const WRITE_CLAUSES = new Set(['insert', 'merge', 'set', 'remove', 'delete']);

/** Does a query mutate the graph (contain any INSERT/MERGE/SET/REMOVE/DELETE)? */
export const queryHasWrite = (query: Query): boolean =>
  query.parts.some((part) => part.clauses.some((clause) => WRITE_CLAUSES.has(clause.kind)));

/**
 * Execute an ISO GQL transaction-control command (`START TRANSACTION`/`COMMIT`/
 * `ROLLBACK`) by driving the session's transaction frame on the graph. Returns
 * nothing (a write-only shape — no rows/columns). ISO semantics are enforced
 * here, not in the core primitives:
 *  - `START TRANSACTION` while one is already active → `E_INVALID_GRAPH_OP`
 *    (ISO forbids nesting). The graph's tx depth reflects only explicit
 *    transactions here, since a TxControl is not a write and so is never wrapped
 *    in a per-statement auto-commit frame.
 *  - `COMMIT`/`ROLLBACK` with no active transaction → `E_INVALID_GRAPH_OP`. The
 *    depth is checked in the executor so ROLLBACK is symmetric with COMMIT
 *    *without* changing the core `rollbackTransaction`'s idempotent contract.
 *  - The READ ONLY access mode is recorded on the graph (and cleared on
 *    commit/rollback); a subsequent write statement consults it (see `execute`).
 */
export const runTxControl = (tx: TxControl, graph: Graph): void => {
  switch (tx.kind) {
    case 'start':
      if (graph.isTransacting()) {
        throw new LenkeError('START TRANSACTION: a transaction is already active', {
          code: ErrorCode.InvalidGraphOp,
        });
      }

      graph.beginTransaction();
      graph.setTransactionReadOnly(tx.accessMode === 'read only');
      break;
    case 'commit':
      if (!graph.isTransacting()) {
        throw new LenkeError('COMMIT: no active transaction', { code: ErrorCode.InvalidGraphOp });
      }

      try {
        graph.commitTransaction(); // may throw (deferred checks) after rolling back
      } finally {
        graph.setTransactionReadOnly(false);
      }

      break;
    case 'rollback':
      if (!graph.isTransacting()) {
        throw new LenkeError('ROLLBACK: no active transaction', { code: ErrorCode.InvalidGraphOp });
      }

      graph.rollbackTransaction();
      graph.setTransactionReadOnly(false);
      break;
  }
};

/**
 * Run one compiled linear query (clause sequence) to result rows. A statement
 * that writes runs inside one transaction, so a mid-statement fault (e.g. a
 * later row of a multi-row INSERT violating a constraint) rolls the earlier rows
 * back instead of leaving the write half-applied — per-statement atomicity,
 * byte-identical to the native engine's auto-commit frame. Read-only statements
 * skip the frame (no undo/commit overhead).
 */
export const runLinear = (linear: CLinear, graph: Graph, params: Params): Row[] => {
  const writes = linear.clauses.some((clause) => WRITE_CLAUSES.has(clause.kind));

  if (!writes) {
    return runLinearClauses(linear, graph, params);
  }

  return graph.transaction(() => runLinearClauses(linear, graph, params));
};

export const runLinearClauses = (
  linear: CLinear,
  graph: Graph,
  params: Params,
  initial?: Binding,
): Row[] => {
  // The fast paths assume an empty start; a seeded (inline-subquery) run skips
  // them and takes the general clause loop.
  if (initial === undefined) {
    // Direct `count(*)` shortcut (edge-bucket size / degree product) — skips
    // enumerating every match. Only fires for the exact `MATCH … RETURN count(*)`
    // shapes `detectCountShortcut` accepts.
    if (linear.countShortcut) {
      return [linear.countShortcut(graph, params)];
    }

    // Unbounded var-length + DISTINCT → BFS the reachable set instead of enumerating
    // trails (exponential, hits the trail budget). See `detectReachableShortcut`.
    if (linear.reachShortcut) {
      return linear.reachShortcut(graph, params);
    }
  }

  // Bindings flow as a lazy stream; only barriers (mutations, aggregation,
  // ORDER BY) force materialization — so a streaming read never holds the whole
  // result set in memory.
  let bindings: Iterable<Binding> = [initial ?? new Map()];

  for (const clause of linear.clauses) {
    switch (clause.kind) {
      case 'match':
        bindings = runMatch(graph, clause, bindings, params);
        break;
      case 'for':
        bindings = runFor(graph, clause, bindings, params);
        break;
      case 'callNamed':
        bindings = runCall(graph, clause, bindings, params);
        break;
      case 'callInline':
        bindings = runCallInline(graph, clause, bindings, params);
        break;
      case 'with': {
        const projected = applyProjection(clause.projection, bindings, params, graph);
        bindings =
          clause.where === undefined
            ? projected
            : filter(
                (b: Binding) => asTruth(clause.where!({ binding: b, params, graph })) === true,
                projected,
              );
        break;
      }
      case 'filter':
        // ISO §14.6: drop rows where the condition is not TRUE (three-valued).
        bindings = filter(
          (b: Binding) => asTruth(clause.where({ binding: b, params, graph })) === true,
          bindings,
        );
        break;
      case 'page': {
        // ISO `<order by and page statement>` in statement position: sort and/or
        // slice the working BINDING table. Because this runs before any
        // projection, a later RETURN only ever projects the surviving rows.
        let rows = toArray(bindings);

        if (clause.orderBy.length > 0) {
          // Key each row once, then a STABLE sort — the same comparator (so the
          // same total order and NULLS FIRST/LAST) as a projection's ORDER BY.
          const keyed = rows.map((b: Binding) => ({
            b,
            keys: clause.orderBy.map((s) => s.fn({ binding: b, params, graph })),
          }));

          keyed.sort((x, y) => {
            for (const [i, sortItem] of clause.orderBy.entries()) {
              const c = compareSort(x.keys[i], y.keys[i], sortItem.descending, sortItem.nullsFirst);

              if (c !== 0) {
                return c;
              }
            }

            return 0;
          });
          rows = keyed.map((r) => r.b);
        }

        const start = resolveCount(clause.skip, params) ?? 0;
        const take = resolveCount(clause.limit, params);

        rows = rows.slice(start, take === undefined ? undefined : start + take);
        bindings = rows;
        break;
      }
      case 'let':
        // ISO §14.7: bind new vars additively, left-to-right (a later item sees
        // an earlier one via the in-progress binding copy).
        bindings = map((b: Binding) => {
          const nb = new Map(b);

          for (const it of clause.items) {
            nb.set(it.var, it.expr({ binding: nb, params, graph }));
          }

          return nb;
        }, bindings);
        break;
      case 'insert':
        // Mutations must run eagerly and exactly once — force evaluation.
        bindings = toArray(map((b: Binding) => runInsert(graph, clause, b, params), bindings));
        break;
      case 'merge':
        bindings = toArray(map((b: Binding) => runMerge(graph, clause, b, params), bindings));
        break;
      case 'set': {
        const arr = toArray(bindings);

        for (const b of arr) {
          runSet(graph, clause, b, params);
        }

        bindings = arr;
        break;
      }
      case 'remove': {
        const arr = toArray(bindings);

        for (const b of arr) {
          runRemove(graph, clause, b);
        }

        bindings = arr;
        break;
      }
      case 'delete': {
        const arr = toArray(bindings);

        for (const b of arr) {
          runDelete(graph, clause, b, params);
        }

        bindings = arr;
        break;
      }
      case 'finish':
        return [];
      case 'return':
        return toArray(map(mapToRow, applyProjection(clause.projection, bindings, params, graph)));
    }
  }

  return []; // a write-only query produces no rows
};

// --- set operations ----------------------------------------------------------

/**
 * Stable key for a result row; graph-element columns key by id. Keyed by column
 * POSITION, not name: ISO set operations (`UNION`/`EXCEPT`/`INTERSECT`) compare
 * rows positionally, so differently-aliased parts (`… AS a` vs `… AS b`) are the
 * same row when their values line up — the result adopts the left part's names.
 * The `\x01` separator avoids adjacent-value collisions (`1,2` vs `12`). This
 * matches the Rust engine, which is already positional here. DISTINCT is
 * unaffected: a single query's rows all carry the same columns.
 */
export const rowKeyOf = (row: Row): string => Object.values(row).map(valueKey).join('\x01');

export const distinctRows = (rows: readonly Row[]): Row[] => {
  const seen = new Set<string>();

  return rows.filter((r) => {
    const k = rowKeyOf(r);

    return seen.has(k) ? false : (seen.add(k), true);
  });
};

/** Combine two row sets per a set operator. */
export const combineRows = (op: SetOp, left: readonly Row[], right: readonly Row[]): Row[] => {
  const rightKeys = new Set(right.map(rowKeyOf));

  switch (op.op) {
    case 'union':
      return op.all ? [...left, ...right] : distinctRows([...left, ...right]);
    case 'except': {
      const kept = left.filter((r) => !rightKeys.has(rowKeyOf(r)));

      return op.all ? kept : distinctRows(kept);
    }
    case 'intersect': {
      const kept = left.filter((r) => rightKeys.has(rowKeyOf(r)));

      return op.all ? kept : distinctRows(kept);
    }
  }
};
