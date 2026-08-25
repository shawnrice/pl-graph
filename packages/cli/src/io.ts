import { extname } from 'node:path';

import { graphFromFormat, graphFromNdjson, type RustGraph } from '@lenke/native';

// The backend type, taken from the loader so we needn't import it by name.
export type Backend = Parameters<typeof graphFromNdjson>[0];

// The CLI runs the wasm engine, whose 32-bit linear memory (~2 GB) cannot hold the
// multi-hundred-million-row intermediate a wide multi-segment MATCH can materialize
// before its WHERE/LIMIT prunes it — an allocation the native engine (64-bit, no such
// ceiling) survives. Left unbounded, that allocation aborts the whole wasm module,
// killing the REPL. Cap the intermediate frontier low enough to fit, so the engine
// trips a catchable `E_RESOURCE_EXHAUSTED` the shell can report instead. (The default
// is 50M; this is not a semantics change — `intermediate` is an anti-runaway bound,
// so a query under it behaves identically. Raise it with a bigger native build.)
const WASM_INTERMEDIATE_CAP = 10_000_000;
const CLI_LIMITS = { limits: { intermediate: WASM_INTERMEDIATE_CAP } } as const;

export const FORMATS = ['ndjson', 'csv', 'graphson', 'pg-json', 'pg-text'] as const;
export type Format = (typeof FORMATS)[number];

export const isFormat = (s: string): s is Format => (FORMATS as readonly string[]).includes(s);

// File extension → codec. `.json` is deliberately absent: it's ambiguous between
// pg-json and graphson, so those need an explicit format.
const BY_EXT: Record<string, Format> = {
  '.ndjson': 'ndjson',
  '.jsonl': 'ndjson',
  '.csv': 'csv',
  '.graphson': 'graphson',
  '.pgjson': 'pg-json',
  '.pgtext': 'pg-text',
};

export const detectFormat = (file: string): Format | undefined =>
  BY_EXT[extname(file).toLowerCase()];

// Resolve the format for a file: the explicit override, else the extension, else
// an error naming the choices — never a silent guess.
export const formatFor = (file: string, override?: string): Format => {
  if (override !== undefined) {
    if (!isFormat(override)) {
      throw new Error(`Unknown format '${override}'. Choose one of: ${FORMATS.join(', ')}.`);
    }

    return override;
  }

  const detected = detectFormat(file);

  if (!detected) {
    throw new Error(`Can't infer a format from '${file}'. Pass --format <${FORMATS.join(' | ')}>.`);
  }

  return detected;
};

export const emptyGraph = (backend: Backend): RustGraph =>
  graphFromNdjson(backend, new Uint8Array(), CLI_LIMITS);

export const loadGraph = (backend: Backend, bytes: Uint8Array, format: Format): RustGraph =>
  format === 'ndjson'
    ? graphFromNdjson(backend, bytes, CLI_LIMITS)
    : graphFromFormat(backend, bytes, { format, ...CLI_LIMITS });

export const saveGraph = (graph: RustGraph, format: Format): Uint8Array =>
  format === 'ndjson' ? graph.toNdjson() : new TextEncoder().encode(graph.serialize(format));
