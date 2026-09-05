import { defineConfig } from 'vite';

export default defineConfig({
  // No react plugin: the built-in transformer handles automatic JSX, keeping the
  // example's deps minimal — trading fast-refresh for zero extra tooling. (`oxc`
  // is vite 8.2's replacement for the deprecated `esbuild` option.)
  oxc: { jsx: { runtime: 'automatic' } },
});
