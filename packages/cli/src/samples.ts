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
    name: 'employment',
    description:
      'Bitemporal org — people, companies, WORKS_AT with valid-time (vf/vt) and system-time (tf/tt). Try as-of queries.',
    file: 'employment.ndjson',
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
