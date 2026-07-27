import type { Graph } from '@lenke/core';
import { isTemporal, LenkeRecord, temporalFormat, temporalParse } from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';

import type { Codec } from '../codec.js';
import { type ChunkSource, linesFromChunks } from '../streaming.js';
import { normalizeBag, type PropertyValue } from '../value.js';

/**
 * The **PG textual format** (`.pg`) — the line-based companion to PG-JSON
 * (https://pg-format.readthedocs.io). One element per line:
 *
 *   <id> :Label* key:value*          ← a node
 *   <from> <to> :Label* key:value*   ← an edge
 *
 * A node line has one leading id; an edge line has two. They're told apart by
 * the second token: a bare id (no `:`) means an edge, while a `:Label` or
 * `key:value` means more of a node. `#` starts a comment line.
 *
 * ## Value mapping
 *   - string  → always double-quoted (`name:"Alice"`), so strings never collide
 *     with numbers / booleans / null / ids. Escapes `"` and `\`.
 *   - number  → bare (`age:15`, `weight:3.5`).
 *   - boolean → bare `true` / `false`.
 *   - null    → bare `null`.
 *   - list    → **repeated keys** (`tags:1 tags:2`) — the PG format's native
 *     multi-value mechanism. On decode, a key seen once is a scalar; a key seen
 *     more than once is a list.
 *
 * ## Lossiness (use PG-JSON or GraphSON for exact round-trips)
 * Because lists ride on repeated keys, two cases can't be represented distinctly
 * in the textual form:
 *   - an **empty list** `[]` emits no key, so it decodes as *absent*;
 *   - a **single-element list** `[x]` is indistinguishable from the scalar `x`.
 * Scalars and multi-element lists round-trip exactly. **Node** ids are preserved
 * (a node line leads with its id); **edge** ids are NOT — the textual format has
 * no edge-id slot, so a decoded edge gets a fresh id (its endpoints, labels, and
 * properties are preserved). Property keys and node ids must be bare (no
 * whitespace/colon); quote string values instead.
 */

// Escape the quote/backslash AND the line/whitespace control chars: pg-text is
// line-oriented, so an unescaped newline in a value would split the token across
// physical lines and corrupt the round-trip. Must match the Rust codec exactly.
const STR_ESCAPE = /[\\"\n\r\t]/g;
const STR_ESCAPE_MAP: Record<string, string> = {
  '\\': '\\\\',
  '"': '\\"',
  '\n': '\\n',
  '\r': '\\r',
  '\t': '\\t',
};
const STR_UNESCAPE_MAP: Record<string, string> = { n: '\n', r: '\r', t: '\t' };

// An id (or property KEY) must be quoted when it contains a `:` (else it reads
// as a `:label` / `key:value` separator), whitespace (which would split the
// token), or a quote/backslash. Keys share this rule so an arbitrary property
// key round-trips instead of tearing a token in two (or, with a newline,
// forging a whole element on decode).
const ID_NEEDS_QUOTE = /[\s:"\\]/;

// A LABEL is emitted as `:label`, so a `:` inside it is unambiguous (the whole
// remainder is the label) and needs no quoting — but whitespace / quote /
// backslash still do (whitespace splits the token; a leading quote would be
// misread as a quoted span).
const LABEL_NEEDS_QUOTE = /[\s"\\]/;

/** Render an id / property-key token, quoting + escaping it when it needs it. */
const idToken = (s: string): string =>
  s === '' || ID_NEEDS_QUOTE.test(s) ? `"${s.replace(STR_ESCAPE, (c) => STR_ESCAPE_MAP[c])}"` : s;

/** Render a label token (bare, or quoted when it carries whitespace/quote/backslash). */
const labelToken = (s: string): string =>
  LABEL_NEEDS_QUOTE.test(s) ? `"${s.replace(STR_ESCAPE, (c) => STR_ESCAPE_MAP[c])}"` : s;

/** Read a quoted-or-bare token, unquoting + unescaping it if it was quoted. */
const parseId = (raw: string): string => {
  if (!raw.startsWith('"')) {
    return raw;
  }

  const body = raw.endsWith('"') && raw.length >= 2 ? raw.slice(1, -1) : raw.slice(1);

  return body.replace(/\\(.)/g, (_, c: string) => STR_UNESCAPE_MAP[c] ?? c);
};

/**
 * Index just past the closing `"` of a quoted span at `start` (`token[start]`
 * must be `"`), respecting `\`-escapes; `token.length` if unterminated.
 */
const quotedSpanEnd = (token: string, start: number): number => {
  let i = start + 1;

  while (i < token.length) {
    if (token[i] === '\\') {
      i += 2;
      continue;
    }

    if (token[i] === '"') {
      return i + 1;
    }

    i += 1;
  }

  return token.length;
};

// A leading token is an id iff it's a *whole* quoted span (`"…"` with nothing
// after — an id) or has no `:` (a bare id). A quoted-then-`:value` token is a
// quoted property key, NOT an id — this is what keeps node/edge detection
// correct now that keys can be quoted.
const isIdToken = (t: string): boolean =>
  t.startsWith('"') ? quotedSpanEnd(t, 0) === t.length : !t.includes(':');

/** Encode one scalar `PropertyValue` (never a list) as a PG-text token value. */
const encodeScalar = (value: Exclude<PropertyValue, readonly PropertyValue[]>): string => {
  if (value === null) {
    return 'null';
  }

  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }

  if (typeof value === 'number') {
    return String(value);
  }

  // A temporal rides as an unquoted `@<kind>:<iso>` token — the ISO form has no
  // whitespace/newline (stays on one line), and the `@` sigil distinguishes it
  // from a quoted string on decode.
  if (isTemporal(value)) {
    return `@${value.kind}:${temporalFormat(value)}`;
  }

  // A map/record has no faithful flat-token form — reject loudly (mirrors the
  // native codec) rather than mangle it. Use a structured format instead.
  if (value instanceof LenkeRecord) {
    throw new LenkeError(
      'a map/record property cannot be serialized to pg-text (a flat format); ' +
        'use ndjson, graphson, or pg-json',
      { code: ErrorCode.Unsupported },
    );
  }

  return `"${value.replace(STR_ESCAPE, (c) => STR_ESCAPE_MAP[c])}"`;
};

/** Append `key:value` tokens for a property; a list expands to one token per element. */
const pushProperty = (out: string[], key: string, value: PropertyValue): void => {
  const k = idToken(key); // arbitrary keys are quoted like ids

  if (Array.isArray(value)) {
    for (const element of value) {
      out.push(`${k}:${encodeScalar(element as Exclude<PropertyValue, readonly PropertyValue[]>)}`);
    }

    return;
  }

  out.push(`${k}:${encodeScalar(value as Exclude<PropertyValue, readonly PropertyValue[]>)}`);
};

const elementTokens = (
  leading: string[],
  labels: Iterable<string>,
  properties: Record<string, unknown>,
): string => {
  const tokens = leading.map(idToken);

  for (const label of labels) {
    tokens.push(`:${labelToken(label)}`);
  }

  const bag = normalizeBag(properties);

  for (const key of Object.keys(bag)) {
    pushProperty(tokens, key, bag[key]);
  }

  return tokens.join(' ');
};

/** Serialize a graph to the PG textual format: node lines, then edge lines. */
export const encode = (graph: Graph): string => {
  const lines: string[] = [];

  for (const vertex of graph.vertices) {
    lines.push(elementTokens([vertex.id], vertex.labels, vertex.properties));
  }

  for (const edge of graph.edges) {
    lines.push(elementTokens([edge.from.id, edge.to.id], edge.labels, edge.properties));
  }

  return lines.join('\n');
};

/** Split a line into tokens, keeping double-quoted spans (with `\` escapes) whole. */
const tokenizeLine = (line: string): string[] => {
  const tokens: string[] = [];
  let current = '';
  let started = false;
  let inQuote = false;

  for (let i = 0; i < line.length; i += 1) {
    const c = line[i];

    if (inQuote) {
      current += c;

      if (c === '\\' && i + 1 < line.length) {
        current += line[i + 1];
        i += 1;
      } else if (c === '"') {
        inQuote = false;
      }

      continue;
    }

    if (c === '"') {
      inQuote = true;
      started = true;
      current += c;
      continue;
    }

    if (c === ' ' || c === '\t') {
      if (started) {
        tokens.push(current);
        current = '';
        started = false;
      }

      continue;
    }

    current += c;
    started = true;
  }

  if (started) {
    tokens.push(current);
  }

  return tokens;
};

const NUMBER = /^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

/** Parse the value half of a `key:value` token into a scalar `PropertyValue`. */
const parseScalar = (raw: string): PropertyValue => {
  if (raw.startsWith('"')) {
    const body = raw.endsWith('"') && raw.length >= 2 ? raw.slice(1, -1) : raw.slice(1);

    return body.replace(/\\(.)/g, (_, c: string) => STR_UNESCAPE_MAP[c] ?? c);
  }

  if (raw === 'true') {
    return true;
  }

  if (raw === 'false') {
    return false;
  }

  if (raw === 'null') {
    return null;
  }

  // A tagged temporal `@<kind>:<iso>` (unquoted). A malformed tag/ISO falls
  // through to string handling, matching pg-text's lenient decode policy.
  if (raw.startsWith('@')) {
    const colon = raw.indexOf(':');

    if (colon !== -1) {
      try {
        return temporalParse(raw.slice(1, colon), raw.slice(colon + 1));
      } catch {
        // fall through to the remaining scalar checks
      }
    }
  }

  if (NUMBER.test(raw)) {
    return Number(raw);
  }

  return raw; // bare, unquoted string (lenient: for foreign `.pg` input)
};

/** Pull labels and properties (collapsing repeated keys into lists) from tokens. */
const parseLabelsAndProps = (
  tokens: string[],
): { labels: string[]; properties: Record<string, PropertyValue> } => {
  const labels: string[] = [];
  const collected = new Map<string, PropertyValue[]>();

  for (const token of tokens) {
    if (token.startsWith(':')) {
      // `:label` or `:"quoted label"` — unquote the latter.
      const rest = token.slice(1);
      labels.push(rest.startsWith('"') ? parseId(rest) : rest);
      continue;
    }

    // Find the key:value separator, skipping a quoted key's own colons.
    let sep: number;

    if (token.startsWith('"')) {
      const end = quotedSpanEnd(token, 0);

      // A whole quoted span with no trailing `:value` isn't a property (a stray
      // id-shaped token); ignore it rather than misread it.
      if (end >= token.length || token[end] !== ':') {
        continue;
      }

      sep = end;
    } else {
      sep = token.indexOf(':');

      if (sep < 0) {
        continue; // not a label or a property; ignore
      }
    }

    const key = parseId(token.slice(0, sep)); // unquotes a quoted key, else verbatim
    const value = parseScalar(token.slice(sep + 1));
    const list = collected.get(key);

    if (list) {
      list.push(value);
    } else {
      collected.set(key, [value]);
    }
  }

  const properties: Record<string, PropertyValue> = {};

  for (const [key, values] of collected) {
    properties[key] = values.length === 1 ? values[0]! : values;
  }

  return { labels, properties };
};

// A second token that is an id (quoted, or `:`-free — not a `:label`/`key:value`)
// marks an edge line.
const isEdgeLine = (tokens: string[]): boolean => tokens.length >= 2 && isIdToken(tokens[1]);

const addNodeLine = (graph: Graph, tokens: string[]): void => {
  const [rawId, ...rest] = tokens;
  const { labels, properties } = parseLabelsAndProps(rest);
  graph.addVertex({ id: parseId(rawId), labels, properties });
};

const addEdgeLine = (graph: Graph, tokens: string[]): void => {
  const [rawFrom, rawTo, ...rest] = tokens;
  const from = parseId(rawFrom);
  const to = parseId(rawTo);
  const fromVertex =
    graph.getVertexById(from) ?? graph.addVertex({ id: from, labels: [], properties: {} });
  const toVertex =
    graph.getVertexById(to) ?? graph.addVertex({ id: to, labels: [], properties: {} });
  const { labels, properties } = parseLabelsAndProps(rest);
  graph.addEdge({ from: fromVertex, to: toVertex, labels, properties });
};

/** Tokenize a line, or `null` for a blank or comment (`#`) line. */
const lineTokens = (raw: string): string[] | null => {
  const line = raw.trim();

  if (line === '' || line.startsWith('#')) {
    return null;
  }

  const tokens = tokenizeLine(line);

  return tokens.length === 0 ? null : tokens;
};

/**
 * Deserialize a PG-text string into `graph`. Two passes — nodes first, then
 * edges — so an edge may reference a node declared anywhere; an endpoint not
 * declared as its own node line is created as a bare node.
 */
export const decode = (input: string, graph: Graph): Graph => {
  const wasEnabled = graph.eventsEnabled();

  if (wasEnabled) {
    graph.disableEvents();
  }

  const nodeLines: string[][] = [];
  const edgeLines: string[][] = [];

  for (const raw of input.split('\n')) {
    const tokens = lineTokens(raw);

    if (tokens) {
      (isEdgeLine(tokens) ? edgeLines : nodeLines).push(tokens);
    }
  }

  for (const tokens of nodeLines) {
    addNodeLine(graph, tokens);
  }

  for (const tokens of edgeLines) {
    addEdgeLine(graph, tokens);
  }

  if (wasEnabled) {
    graph.enableEvents();
    graph.snapshot();
  }

  return graph;
};

/**
 * Stream the document in batched chunks (each a run of complete lines), so a
 * large graph is written without materializing the whole string.
 */
export async function* encodeStream(graph: Graph): AsyncGenerator<string> {
  const batchSize = 1024;
  let batch: string[] = [];

  for (const vertex of graph.vertices) {
    batch.push(elementTokens([vertex.id], vertex.labels, vertex.properties));

    if (batch.length >= batchSize) {
      yield `${batch.join('\n')}\n`;
      batch = [];
    }
  }

  for (const edge of graph.edges) {
    batch.push(elementTokens([edge.from.id, edge.to.id], edge.labels, edge.properties));

    if (batch.length >= batchSize) {
      yield `${batch.join('\n')}\n`;
      batch = [];
    }
  }

  if (batch.length > 0) {
    yield `${batch.join('\n')}\n`;
  }
}

/**
 * Slurp a large graph from a chunk source one line at a time (memory bounded by
 * the longest line, not the document). Single pass, so — unlike `decode` — node
 * lines must precede the edges that reference them, which `encodeStream`
 * guarantees; an as-yet-unseen edge endpoint is created as a bare node.
 */
export const decodeStream = async (source: ChunkSource, graph: Graph): Promise<Graph> => {
  const wasEnabled = graph.eventsEnabled();

  if (wasEnabled) {
    graph.disableEvents();
  }

  for await (const raw of linesFromChunks(source)) {
    const tokens = lineTokens(raw);

    if (!tokens) {
      continue;
    }

    if (isEdgeLine(tokens)) {
      addEdgeLine(graph, tokens);
    } else {
      addNodeLine(graph, tokens);
    }
  }

  if (wasEnabled) {
    graph.enableEvents();
    graph.snapshot();
  }

  return graph;
};

export const pgTextCodec: Codec = { name: 'pg-text', encode, decode, encodeStream, decodeStream };
