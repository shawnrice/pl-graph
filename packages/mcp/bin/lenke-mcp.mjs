#!/usr/bin/env node
// Thin launcher: the server lives in src/ (built to dist/) and is unit-tested
// there; this wires stdio and turns a fatal error into a clean exit.
import { main } from '../dist/esm/index.mjs';

main().catch((err) => {
  process.stderr.write(`lenke-mcp: ${err?.message ?? err}\n`);
  process.exit(1);
});
