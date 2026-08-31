import { appendFileSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import path from 'node:path';
import { stderr, stdout } from 'node:process';
import { createInterface, type Interface } from 'node:readline';

import { LocalDateTime } from '@lenke/core';
import { describe as inspectDescribe, formatGraph, formatRows } from '@lenke/inspect';
import type { RustGraph } from '@lenke/native';

import { complete, type Mode } from './completion.js';
import { type Backend, formatFor, loadGraph, saveGraph } from './io.js';
import { asRows, classify } from './query.js';
import { findSample, loadSample, SAMPLES } from './samples.js';

export type ShellContext = { graph: RustGraph; backend: Backend; color: boolean };

type OutFormat = 'table' | 'json' | 'ndjson';

// The clock feeds `current_date` / `current_timestamp`. `live` follows the system
// clock (read per query), which is the shell default; `fixed` pins a date for
// as-of exploration; `off` leaves the now-functions unset.
type ClockDesc = { kind: 'live' } | { kind: 'off' } | { kind: 'fixed'; date: Date };

const applyClock = (graph: RustGraph, c: ClockDesc): void => {
  if (c.kind === 'off') {
    graph.setClock(null);
  } else if (c.kind === 'live') {
    graph.setClock(() => LocalDateTime.fromJSDate(new Date(), { zone: 'utc' }));
  } else {
    graph.setClock(() => LocalDateTime.fromJSDate(c.date, { zone: 'utc' }));
  }
};

const describeClock = (c: ClockDesc): string => {
  if (c.kind === 'live') {
    return 'live (follows the system clock)';
  }

  if (c.kind === 'off') {
    return 'off (current_date is unset)';
  }

  return `${c.date.toISOString().slice(0, 10)} (fixed)`;
};

export type State = {
  graph: RustGraph;
  backend: Backend;
  color: boolean;
  mode: Mode;
  format: OutFormat;
  timing: boolean;
  labels: string[];
  last: readonly unknown[];
  clock: ClockDesc;
  /** `\o` output tee: a file every line is appended to, or undefined for stdout only. */
  outFile?: string;
};

export const makeState = (ctx: ShellContext): State => {
  // Wire the wall clock by default, so current_date works without a `\clock` first.
  applyClock(ctx.graph, { kind: 'live' });

  return {
    graph: ctx.graph,
    backend: ctx.backend,
    color: ctx.color,
    mode: 'gql',
    format: 'table',
    timing: false,
    labels: labelsOf(ctx.graph),
    last: [],
    clock: { kind: 'live' },
  };
};

const freeGraph = (graph: RustGraph): void => {
  try {
    graph.free();
  } catch {
    // already freed / backend gone
  }
};

const labelsOf = (graph: RustGraph): string[] => {
  try {
    const s = inspectDescribe(graph);

    return [...s.vertexLabels, ...s.edgeLabels].map((l) => l.label);
  } catch {
    return [];
  }
};

export const render = (rows: readonly unknown[], state: State): string => {
  if (state.format === 'json') {
    return JSON.stringify(rows, null, 2);
  }

  if (state.format === 'ndjson') {
    return rows.map((r) => JSON.stringify(r)).join('\n');
  }

  return formatRows(asRows(rows), { color: state.color });
};

// Run one statement in the current language and return its rows (also stored as `_`).
export const runStatement = (state: State, text: string): readonly unknown[] => {
  if (state.mode === 'gremlin') {
    return state.graph.gremlin(text);
  }

  if (state.mode === 'js') {
    const scope = {
      g: state.graph,
      _: state.last,
      query: (q: string, p?: Record<string, unknown>) => state.graph.query(q, p),
      gremlin: (q: string) => state.graph.gremlin(q),
    };
    const keys = Object.keys(scope);
    // The `\js` mode's whole purpose: evaluate the user's own REPL line as a JS
    // expression (so `_.filter(...).map(...)` works). The input is the operator's,
    // not untrusted data.
    // oxlint-disable-next-line no-implied-eval -- this IS the shell's JS-eval mode
    const fn = new Function(...keys, `return (${text});`);
    const value = fn(...keys.map((k) => (scope as Record<string, unknown>)[k]));

    return Array.isArray(value) ? value : [value];
  }

  return state.graph.query(text);
};

// A statement is "complete" when brackets/quotes balance — lets a pasted multi-line
// query run when it closes, without requiring a `;` terminator.
export const isBalanced = (text: string): boolean => {
  let depth = 0;
  let quote = '';

  for (let i = 0; i < text.length; i++) {
    const c = text[i];

    if (quote) {
      if (c === '\\') {
        i++;
      } else if (c === quote) {
        quote = '';
      }

      continue;
    }

    if (c === '"' || c === "'" || c === '`') {
      quote = c;
    } else if (c === '(' || c === '[' || c === '{') {
      depth++;
    } else if (c === ')' || c === ']' || c === '}') {
      depth--;
    }
  }

  return depth <= 0 && quote === '';
};

// Blank out quoted spans so keyword sniffing never fires on a string literal.
const withoutStrings = (s: string): string =>
  s.replace(/'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`/g, "''");

const OPENS_QUERY = /^\s*(match|optional|with|let|unwind)\b/i;
const CLOSES_QUERY = /\b(return|finish|insert|set|remove|merge|delete|detach|call|create|drop)\b/i;

// A GQL read-query opens with MATCH/WITH/… and only completes at a clause that can end
// it (RETURN, or a write verb). Brackets balance the moment `MATCH (a)-[:R]->(b)` closes
// — long before RETURN — so balance alone would run a half-written query. In GQL mode,
// treat "opened but no ending clause yet" as still-being-typed, so a query authored
// across lines keeps reading (like psql) instead of executing prematurely.
export const awaitingClause = (buffer: string, mode: Mode): boolean => {
  if (mode !== 'gql') {
    return false;
  }

  const s = withoutStrings(buffer);

  return OPENS_QUERY.test(s) && !CLOSES_QUERY.test(s);
};

// A trailing `;` is an explicit "run it now" (psql muscle memory), stripped before the
// statement reaches the engine — which has no use for the terminator.
export const endsWithTerminator = (line: string): boolean => /;\s*$/.test(line);
export const stripTerminator = (stmt: string): string => stmt.replace(/;\s*$/, '').trimEnd();

const HISTORY_FILE = path.join(homedir(), '.lenke_history');

const loadHistory = (): string[] => {
  try {
    return readFileSync(HISTORY_FILE, 'utf8').split('\n').filter(Boolean).reverse();
  } catch {
    return [];
  }
};

const appendHistory = (line: string): void => {
  try {
    appendFileSync(HISTORY_FILE, `${line}\n`);
  } catch {
    // history is best-effort
  }
};

const BANNER = (color: boolean): string => {
  const b = color ? (s: string) => `\x1b[1m${s}\x1b[0m` : (s: string) => s;

  return `${b('lenke')} — a graph shell. Type GQL and press Enter; \\? for help, \\q to quit.
Load a sample to explore: ${b('\\l')} lists them, e.g. ${b('\\c hillvalley')}.`;
};

const HELP = `Queries
  <GQL>                 run a GQL query (the default mode)
  \\gremlin / \\gql       switch the query language (prompt shows the mode)
  \\js                   JavaScript on results: _ is the last rows, g is the graph

Meta-commands
  \\l                    list bundled sample graphs
  \\c <name|file> [fmt]  load a sample by name, or a file (codec from extension or fmt)
  \\d                    describe the graph (labels + counts)
  \\dv / \\de            list vertex / edge labels
  \\d <Label>            property keys + count for a label
  \\clock [date|now|off] as-of date for current_date (default: the system clock)
  \\format table|json|ndjson    how results render
  \\timing on|off        show query time
  \\i <file>             run queries from a file (one per line)
  \\o <file>|off         also write query output to a file
  \\save <file> [fmt]    serialize the graph to a file
  \\r                    reset the current input buffer
  \\? / \\q              this help / quit`;

const promptFor = (state: State, cont: boolean): string => {
  const mode = state.mode === 'gql' ? '' : `(${state.mode})`;
  const tail = cont ? '-# ' : '=# ';

  return `lenke${mode}${tail}`;
};

const IDENT = /^[A-Za-z_][A-Za-z0-9_]*$/;

const metaLoad = (state: State, rest: string[], out: (s: string) => void): void => {
  const sample = findSample(rest[0]);
  const next = sample
    ? loadSample(state.backend, sample)
    : loadGraph(state.backend, new Uint8Array(readFileSync(rest[0])), formatFor(rest[0], rest[1]));

  freeGraph(state.graph);
  state.graph = next;
  state.labels = labelsOf(next);
  applyClock(next, state.clock); // a fresh graph starts clockless — carry the session's setting
  out(
    `loaded ${sample ? sample.name : rest[0]} — ${next.vertexCount} vertices, ${next.edgeCount} edges`,
  );
};

const metaDescribeLabel = (state: State, label: string, out: (s: string) => void): void => {
  if (!IDENT.test(label)) {
    out(`\\d: '${label}' is not a label name`);

    return;
  }

  try {
    // A label is an identifier, not a value — it can't be a `$param`. Validated as an
    // identifier above, so this interpolation is not an injection vector.
    // oxlint-disable-next-line lenke/no-raw-interpolation -- label validated as an identifier; labels aren't parameterizable
    const rows = state.graph.query(`MATCH (n:${label}) RETURN keys(n) AS k`) as { k: string[] }[];
    const keys = [...new Set(rows.flatMap((r) => r.k ?? []))].sort();

    out(`${label}: ${rows.length} elements\n  keys: ${keys.join(', ') || '(none)'}`);
  } catch (e) {
    out(`\\d ${label}: ${(e as Error).message}`);
  }
};

const metaInclude = (state: State, file: string, out: (s: string) => void): boolean => {
  for (const fileLine of readFileSync(file, 'utf8').split('\n')) {
    const line = fileLine.trim();

    if (line === '' || line.startsWith('--')) {
      continue;
    }

    if (line.startsWith('\\')) {
      if (runMeta(state, line, out)) {
        return true;
      }

      continue;
    }

    try {
      const rows = runStatement(state, line);

      state.last = rows;
      out(render(rows, state));
    } catch (e) {
      out((e as Error).message);
    }
  }

  return false;
};

/** Point (or clear) the `\o` output tee, returning the status line to echo. */
const setOutputTee = (state: State, arg: string): string => {
  state.outFile = !arg || arg === 'off' ? undefined : arg;

  return `output: ${state.outFile ?? 'stdout only'}`;
};

/** Handlers for `\` meta-commands. Returns true when the session should quit. */
export const runMeta = (state: State, raw: string, out: (s: string) => void): boolean => {
  const [cmd, ...rest] = raw.trim().split(/\s+/);
  const arg = rest.join(' ');

  switch (cmd) {
    case '\\?':
    case '\\h':
      out(HELP);

      return false;
    case '\\q':
      return true;
    case '\\gql':
    case '\\gremlin':
    case '\\js':
      state.mode = cmd.slice(1) as Mode;
      out(`mode: ${state.mode}`);

      return false;
    case '\\l':
      out(SAMPLES.map((s) => `  ${s.name.padEnd(12)} ${s.description}`).join('\n'));

      return false;
    case '\\c':
      if (!arg) {
        out('usage: \\c <sample-name | file> [format]');
      } else {
        metaLoad(state, rest, out);
      }

      return false;
    case '\\d':
      if (arg) {
        metaDescribeLabel(state, arg, out);
      } else {
        out(formatGraph(state.graph, { color: state.color }));
      }

      return false;
    case '\\dv':
    case '\\de': {
      const s = inspectDescribe(state.graph);
      const list = cmd === '\\dv' ? s.vertexLabels : s.edgeLabels;

      out(
        formatRows(
          list.map((l) => ({ label: l.label, count: l.count })),
          { color: state.color },
        ),
      );

      return false;
    }
    case '\\format':
      if (arg !== 'table' && arg !== 'json' && arg !== 'ndjson') {
        out('usage: \\format table|json|ndjson');
      } else {
        state.format = arg;
        out(`format: ${arg}`);
      }

      return false;
    case '\\timing':
      state.timing = arg !== 'off';
      out(`timing: ${state.timing ? 'on' : 'off'}`);

      return false;
    case '\\clock': {
      const set = (c: ClockDesc): void => {
        state.clock = c;
        applyClock(state.graph, c);
        out(`clock: ${describeClock(c)}`);
      };

      if (!arg) {
        out(`clock: ${describeClock(state.clock)}`); // no arg → report, don't change
      } else if (arg === 'now' || arg === 'live') {
        set({ kind: 'live' });
      } else if (arg === 'off') {
        set({ kind: 'off' });
      } else {
        const d = new Date(`${arg}T00:00:00Z`);

        if (Number.isNaN(d.getTime())) {
          out(`\\clock: '${arg}' is not a date (try YYYY-MM-DD, or 'now' / 'off')`);
        } else {
          set({ kind: 'fixed', date: d });
        }
      }

      return false;
    }
    case '\\i':
      if (!rest[0]) {
        out('usage: \\i <file>');

        return false;
      }

      return metaInclude(state, rest[0], out);
    case '\\save': {
      if (!rest[0]) {
        out('usage: \\save <file> [format]');

        return false;
      }

      const fmt = formatFor(rest[0], rest[1]);

      writeFileSync(rest[0], saveGraph(state.graph, fmt));
      out(`saved ${rest[0]} (${fmt})`);

      return false;
    }
    case '\\o':
      // The output tee. Interactively this is intercepted before `runMeta` (to
      // refresh the prompt), but routing it through here too means it also works
      // inside an `\i` script instead of erroring as an unknown command.
      out(setOutputTee(state, arg));

      return false;
    case '\\r':
      // Reset the pending multi-line buffer — an interactive-only concept (a script
      // runs one complete line at a time), so it's a no-op here rather than an error.
      return false;
    default:
      out(`unknown command '${cmd}' — try \\?`);

      return false;
  }
};

/**
 * Start the interactive graph shell: a readline loop (Node + Bun) where GQL is the
 * default, `\`-commands drive the session, and results render as tables. Replaces
 * the old Node-only `node:repl` wrapper.
 */
export const runShell = async (ctx: ShellContext): Promise<void> => {
  const state = makeState(ctx);

  const emit = (s: string): void => {
    stdout.write(`${s}\n`);

    if (state.outFile) {
      try {
        appendFileSync(state.outFile, `${s}\n`);
      } catch {
        // tee is best-effort
      }
    }
  };

  const rl: Interface = createInterface({
    input: process.stdin,
    output: stdout,
    terminal: Boolean(stdout.isTTY),
    completer: (line: string) => complete(line, state.mode, state.labels),
  });

  // Best-effort persistent history (readline exposes `history` as most-recent-first).
  (rl as unknown as { history: string[] }).history = loadHistory();

  stdout.write(`${BANNER(ctx.color)}\n`);

  let buffer = '';
  const setPrompt = (): void => rl.setPrompt(promptFor(state, buffer !== ''));

  setPrompt();
  rl.prompt();

  const done = new Promise<void>((resolve) => {
    rl.on('close', () => {
      freeGraph(state.graph);
      resolve();
    });
  });

  let quit = false;

  rl.on('line', (line) => {
    const trimmed = line.trim();

    // `\o` sets the output tee (on `state`, so `runMeta` owns the logic and `\i`
    // scripts can use it too); the interactive path just adds the prompt refresh.
    if (buffer === '' && /^\\o(\s|$)/.test(trimmed)) {
      runMeta(state, trimmed, emit);
      setPrompt();
      rl.prompt();

      return;
    }

    // Meta-commands are single-line and only at a fresh prompt.
    if (buffer === '' && trimmed.startsWith('\\')) {
      if (trimmed === '\\r') {
        buffer = '';
      } else {
        try {
          quit = runMeta(state, trimmed, emit);
        } catch (e) {
          emit((e as Error).message);
        }
      }

      if (quit) {
        rl.close();

        return;
      }

      setPrompt();
      rl.prompt();

      return;
    }

    // Accumulate; continue on a trailing backslash or unbalanced brackets.
    let piece = line;
    let cont = false;

    if (piece.endsWith('\\')) {
      piece = piece.slice(0, -1);
      cont = true;
    }

    buffer = buffer ? `${buffer}\n${piece}` : piece;

    // A trailing `;` forces execution now; otherwise keep reading while the line
    // continues (trailing `\`), brackets are open, or a GQL query has opened but not
    // yet reached a clause that ends it.
    const forced = endsWithTerminator(piece);

    if (
      !forced &&
      (cont || (trimmed !== '' && (!isBalanced(buffer) || awaitingClause(buffer, state.mode))))
    ) {
      setPrompt();
      rl.prompt();

      return;
    }

    const stmt = stripTerminator(buffer.trim());

    buffer = '';

    if (stmt !== '') {
      appendHistory(stmt);

      try {
        const t0 = state.timing ? performance.now() : 0;
        const rows = runStatement(state, stmt);

        state.last = rows;
        emit(render(rows, state));

        if (state.timing) {
          emit(`(${(performance.now() - t0).toFixed(1)} ms)`);
        }
      } catch (e) {
        stderr.write(`${(e as Error).message}\n`);

        // A common wrong-mode slip: a Gremlin traversal typed at the GQL prompt.
        if (state.mode === 'gql' && classify(stmt) === 'gremlin') {
          stderr.write('(that looks like Gremlin — switch with \\gremlin)\n');
        }
      }
    }

    setPrompt();
    rl.prompt();
  });

  return done;
};
