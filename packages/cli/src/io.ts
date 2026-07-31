import { extname } from 'node:path';

import { graphFromFormat, graphFromNdjson, type RustGraph } from '@lenke/native';

// The backend type, taken from the loader so we needn't import it by name.
export type Backend = Parameters<typeof graphFromNdjson>[0];

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
  graphFromNdjson(backend, new Uint8Array());

export const loadGraph = (backend: Backend, bytes: Uint8Array, format: Format): RustGraph =>
  format === 'ndjson'
    ? graphFromNdjson(backend, bytes)
    : graphFromFormat(backend, bytes, { format });

export const saveGraph = (graph: RustGraph, format: Format): Uint8Array =>
  format === 'ndjson' ? graph.toNdjson() : new TextEncoder().encode(graph.serialize(format));
