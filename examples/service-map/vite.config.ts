import { defineConfig } from 'vite';

export default defineConfig({
  // No react plugin: the built-in transformer handles automatic JSX, trading fast
  // refresh for zero extra dependencies in an example. (`oxc` is vite 8.2's
  // replacement for the deprecated `esbuild` option.)
  oxc: { jsx: { runtime: 'automatic' } },
  server: {
    fs: {
      // The wasm artifact is imported by URL from the crate's target dir,
      // outside this app's root.
      allow: ['../..'],
    },
  },
  worker: {
    format: 'es',
  },
});
