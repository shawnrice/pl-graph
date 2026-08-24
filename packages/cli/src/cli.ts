import { readFile, writeFile } from 'node:fs/promises';
import { env, stderr, stdout } from 'node:process';

import { openBackend } from './engine.js';
import { emptyGraph, formatFor, loadGraph, saveGraph } from './io.js';
import { type Lang, runQuery } from './query.js';
import { runShell } from './shell.js';

const USAGE = `lenke — a REPL and CLI for the lenke graph engine

Usage:
  lenke [file] [options]

With no file it starts an empty graph; with a file it loads it (codec from the
extension, or --format). With neither -q nor -o it opens an interactive REPL.

Options:
  -q, --query <text>        run one query (GQL, or Gremlin if it starts with "g.") and exit
  -l, --lang <gql|gremlin>  force the language for -q
  -f, --format <fmt>        input codec: ndjson | csv | graphson | pg-json | pg-text
  -o, --out <file>          serialize the graph to a file, then exit (a codec converter)
      --out-format <fmt>    output codec (default: from the --out extension)
      --wasm <path>         path to lenke_engine.wasm ($LENKE_WASM, else the build output)
      --no-color            disable colored output
  -h, --help                show this

Examples:
  lenke graph.ndjson
  lenke graph.csv -q "MATCH (p:Person) RETURN p.name, p.age"
  lenke graph.ndjson -q "g.V().hasLabel('Person').count()"
  lenke graph.graphson -o graph.ndjson`;

type Args = {
  input?: string;
  format?: string;
  query?: string;
  lang?: Lang;
  out?: string;
  outFormat?: string;
  wasm?: string;
  color?: boolean;
  help: boolean;
};

const NEEDS_VALUE = new Set([
  '-q',
  '--query',
  '-l',
  '--lang',
  '-f',
  '--format',
  '-o',
  '--out',
  '--out-format',
  '--wasm',
]);

const parseArgs = (argv: readonly string[]): Args => {
  const args: Args = { help: false };

  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    const takesValue = NEEDS_VALUE.has(flag);
    const value = takesValue ? argv[++i] : undefined;

    if (takesValue && value === undefined) {
      throw new Error(`option '${flag}' needs a value`);
    }

    switch (flag) {
      case '-h':
      case '--help':
        args.help = true;
        break;
      case '-q':
      case '--query':
        args.query = value;
        break;
      case '-l':
      case '--lang':
        if (value !== 'gql' && value !== 'gremlin') {
          throw new Error("--lang must be 'gql' or 'gremlin'");
        }

        args.lang = value;
        break;
      case '-f':
      case '--format':
        args.format = value;
        break;
      case '-o':
      case '--out':
        args.out = value;
        break;
      case '--out-format':
        args.outFormat = value;
        break;
      case '--wasm':
        args.wasm = value;
        break;
      case '--color':
        args.color = true;
        break;
      case '--no-color':
        args.color = false;
        break;
      default:
        if (flag.startsWith('-')) {
          throw new Error(`unknown option '${flag}' (try --help)`);
        }

        if (args.input !== undefined) {
          throw new Error(`unexpected argument '${flag}'`);
        }

        args.input = flag;
    }
  }

  return args;
};

/** Entry point. `argv` is the args after the node/bin prefix (i.e. `process.argv.slice(2)`). */
export const main = async (argv: readonly string[]): Promise<void> => {
  const args = parseArgs(argv);

  if (args.help) {
    stdout.write(`${USAGE}\n`);

    return;
  }

  const color = args.color ?? (Boolean(stdout.isTTY) && !env.NO_COLOR);
  const backend = await openBackend(args.wasm);
  const graph = args.input
    ? loadGraph(
        backend,
        new Uint8Array(await readFile(args.input)),
        formatFor(args.input, args.format),
      )
    : emptyGraph(backend);

  const oneShot = args.query !== undefined;
  const convert = args.out !== undefined;

  if (!oneShot && !convert) {
    await runShell({ graph, backend, color });

    return;
  }

  try {
    if (oneShot) {
      stdout.write(`${runQuery(graph, args.query as string, args.lang, color).output}\n`);
    }

    if (convert) {
      const format = formatFor(args.out as string, args.outFormat);
      await writeFile(args.out as string, saveGraph(graph, format));
      stderr.write(`saved ${args.out} (${format})\n`);
    }
  } finally {
    try {
      graph.free();
    } catch {
      // process is exiting anyway
    }
  }
};
