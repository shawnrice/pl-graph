import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildPackage } from '@lenke/dev';

const __dirname = dirname(fileURLToPath(import.meta.url));

await buildPackage({
  packageRoot: __dirname,
  // The backends are subpath exports (`@lenke/native/ffi`, `/wasm`) so a
  // browser bundle never pulls in the Bun-only `bun:ffi` import.
  // `src/arrow.ts` is a subpath export (`@lenke/native/arrow`) so the core never
  // pulls in the optional `apache-arrow` peer dependency unless you import it.
  additionalEntrypoints: [
    'src/backend-ffi.ts',
    'src/backend-wasm.ts',
    'src/backend-ffi-engine.ts',
    'src/backend-wasm-engine.ts',
    'src/arrow.ts',
  ],
  skipCjs: true,
  skipMin: true,
});
