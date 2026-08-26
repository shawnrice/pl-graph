import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import type { RustGraph } from '@lenke/native';

import { type Backend, loadGraph } from './io.js';

/** A graph that ships with the CLI so you can explore lenke without a data file. */
export type Sample = { name: string; description: string; file: string };

export const SAMPLES: readonly Sample[] = [
  {
    name: 'modern',
    description:
      'Apache TinkerPop "modern": 6 person/software vertices, weighted knows/created edges.',
    file: 'modern.ndjson',
  },
  {
    name: 'dunder',
    description:
      'Dunder Mifflin (The Office) — 24 employees over 9 seasons: role changes, the manager succession, a Sabre acquisition, and a bitemporal correction. Try as-of queries.',
    file: 'dunder.ndjson',
  },
  {
    name: 'ledger',
    description:
      "A small business's Q1 general ledger — an append-only book of record, on a graph that forgets everything when you close it. Valid time = a posting's effective date, transaction time = when it was booked; corrections are new postings, never edits. Ask for Q1 revenue as reported on Apr 1 ($20k) vs as it stands now ($25k, after a late invoice).",
    file: 'ledger.ndjson',
  },
  {
    name: 'hillvalley',
    description:
      'Back to the Future — Hill Valley across rewritten timelines. Facts carry BOTH valid time (vf/vt, when true in-story) and transaction time (tf/tt, which version of history recorded it), so a time-travel edit is a bitemporal correction. Ask what was true in 1985 "as of" the original vs the restored timeline.',
    file: 'hillvalley.ndjson',
  },
  {
    name: 'primer',
    description:
      'Primer — one day, lived over and over. The densest bitemporal graph: a single valid-time evening re-recorded each loop (transaction time), so the same instant reads differently "as of" take 0, 1, 2 — and the count of Aaron\'s doubles grows 1 → 2 → 3 with the recursion.',
    file: 'primer.ndjson',
  },
];

// samples/ ships beside the built code. From dist/esm/*.mjs it is ../../samples;
// running the sources directly it is ../samples. Try both, plus the package root.
const resolveSamplesDir = (): string => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const candidates = [
    path.join(here, '..', '..', 'samples'),
    path.join(here, '..', 'samples'),
    path.join(here, 'samples'),
  ];

  return candidates.find((d) => existsSync(d)) ?? candidates[0];
};

export const findSample = (name: string): Sample | undefined =>
  SAMPLES.find((s) => s.name === name);

export const samplePath = (sample: Sample): string => path.join(resolveSamplesDir(), sample.file);

export const loadSample = (backend: Backend, sample: Sample): RustGraph =>
  loadGraph(backend, new Uint8Array(readFileSync(samplePath(sample))), 'ndjson');
