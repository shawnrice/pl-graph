// The executable tools the server exposes. They run against the portable pure-TS
// engine (@lenke/core + @lenke/gql), so the server needs no native binary: an
// assistant can build a scratch graph from NDJSON, run a GQL query or a graph
// algorithm, and validate query text — then iterate on the real results.

import {
  betweenness,
  closeness,
  connectedComponents,
  degree,
  Graph,
  type GraphAlgorithm,
  labelPropagation,
  neighborAggregate,
  onCycle,
  pagerank,
  peerPressure,
  personalizedPagerank,
  shortestPath,
  stronglyConnectedComponents,
} from '@lenke/core';
import { parse, query } from '@lenke/gql';
import { deserialize, type FormatName, FORMATS } from '@lenke/serialization';

import type { ToolDef } from './protocol.js';

/** Cap on rows echoed back to the model, so a big result doesn't flood context. */
const MAX_ROWS = 200;

const asString = (v: unknown, field: string): string => {
  if (typeof v !== 'string') {
    throw new Error(`\`${field}\` must be a string`);
  }

  return v;
};

/** Build a scratch graph from optional serialized `data` (default NDJSON). */
const buildGraph = (args: Record<string, unknown>): Graph => {
  const { data } = args;

  if (data === undefined || data === null || data === '') {
    return new Graph();
  }

  const format = (args.format as FormatName | undefined) ?? 'ndjson';

  if (!FORMATS.includes(format)) {
    throw new Error(`\`format\` must be one of ${FORMATS.join(' | ')}`);
  }

  return deserialize(asString(data, 'data'), format);
};

/** Serialize rows for the model, truncating past {@link MAX_ROWS}. */
const renderRows = (rows: readonly unknown[]): string => {
  const shown = rows.slice(0, MAX_ROWS);
  const head = `${rows.length} row${rows.length === 1 ? '' : 's'}`;
  const note = rows.length > MAX_ROWS ? ` (showing first ${MAX_ROWS})` : '';

  return `${head}${note}:\n${JSON.stringify(shown, null, 2)}`;
};

/** A best-effort nudge when a query looks like it was written for another graph
 * language. lenke's GQL is ISO/IEC 39075, which differs from Cypher in a few
 * common spots; point at the GQL form rather than leave a bare syntax error. */
const gqlOnboardingHint = (text: string): string | null => {
  const hints: string[] = [];

  if (/\[[^\]]*\*\s*\d/.test(text) || /\]\s*\*\s*\d/.test(text)) {
    hints.push(
      'Variable-length paths use the ISO quantifier after the relationship: `-[:R]->{1,5}` (or `->*` / `->+`), not `-[:R*1..5]`.',
    );
  }

  if (/duration\s*\(\s*\{/.test(text)) {
    hints.push("Durations are ISO-8601 strings: `duration('PT24H')`, `duration('P1D')`.");
  }

  if (/\bdatetime\s*\(\s*['"][^'"]*(?:Z|[+-]\d\d:?\d\d)['"]/.test(text)) {
    hints.push(
      "`datetime()` is zoneless; for a timestamp carrying an offset/`Z` use `zoned_datetime('…Z')`.",
    );
  }

  return hints.length > 0 ? `\nGQL notes:\n- ${hints.join('\n- ')}` : null;
};

const gqlRun: ToolDef = {
  name: 'gql_run',
  description:
    'Run an ISO-GQL query against a scratch graph and return the result rows. ' +
    'Optionally seed the graph from serialized `data` (NDJSON by default). Use ' +
    'this to iterate on a query against real data.',
  inputSchema: {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'The GQL query text.' },
      data: {
        type: 'string',
        description: 'Optional graph data to load first (NDJSON unless `format` says otherwise).',
      },
      format: {
        type: 'string',
        enum: [...FORMATS],
        description: 'Serialization format of `data` (default: ndjson).',
      },
      params: {
        type: 'object',
        description: 'Optional `$param` bindings for the query.',
        additionalProperties: true,
      },
    },
    required: ['query'],
  },
  handle: (args) => {
    const text = asString(args.query, 'query');
    const graph = buildGraph(args);
    const params = (args.params as Record<string, unknown> | undefined) ?? undefined;
    const rows = query(graph, text, params);

    return renderRows(rows);
  },
};

const gqlCheck: ToolDef = {
  name: 'gql_check',
  description:
    'Parse an ISO-GQL query without running it and report whether it is valid, ' +
    "with the parse error and a hint about lenke's GQL syntax if it doesn't parse.",
  inputSchema: {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'The GQL query text to validate.' },
    },
    required: ['query'],
  },
  handle: (args) => {
    const text = asString(args.query, 'query');

    try {
      parse(text);

      return 'Valid GQL — parses cleanly.';
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      const hint = gqlOnboardingHint(text);

      return `Invalid GQL: ${message}${hint ?? ''}`;
    }
  },
};

/** The graph algorithms exposed by `algorithm_run`, by name. */
const ALGORITHMS: Record<string, GraphAlgorithm<Record<string, unknown>>> = {
  degree,
  pagerank,
  personalizedPagerank,
  connectedComponents,
  stronglyConnectedComponents,
  labelPropagation,
  peerPressure,
  betweenness,
  closeness,
  shortestPath,
  onCycle,
  neighborAggregate,
};

const algorithmRun: ToolDef = {
  name: 'algorithm_run',
  description:
    'Run an in-engine graph algorithm against a scratch graph and return the ' +
    `per-node rows. Algorithms: ${Object.keys(ALGORITHMS).join(', ')}. Seed the ` +
    'graph from `data` (NDJSON by default); pass algorithm `config` as needed.',
  inputSchema: {
    type: 'object',
    properties: {
      name: { type: 'string', enum: Object.keys(ALGORITHMS), description: 'Algorithm to run.' },
      config: {
        type: 'object',
        description:
          'Algorithm config (e.g. { direction, iterations, dampingFactor, writeProperty, source, ' +
          'weightProperty, feature, op }). All fields optional.',
        additionalProperties: true,
      },
      data: { type: 'string', description: 'Graph data to load first (NDJSON unless `format`).' },
      format: {
        type: 'string',
        enum: [...FORMATS],
        description: 'Format of `data` (default: ndjson).',
      },
    },
    required: ['name'],
  },
  handle: async (args) => {
    const name = asString(args.name, 'name');
    const algo = ALGORITHMS[name];

    if (!algo) {
      throw new Error(
        `unknown algorithm '${name}'. Available: ${Object.keys(ALGORITHMS).join(', ')}`,
      );
    }

    const graph = buildGraph(args);
    const config = (args.config as Record<string, unknown> | undefined) ?? {};
    const rows = await algo(config, graph);

    return renderRows(rows);
  },
};

export const TOOLS: readonly ToolDef[] = [gqlRun, gqlCheck, algorithmRun];
