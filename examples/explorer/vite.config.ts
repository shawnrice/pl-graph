import { defineConfig } from 'vite';

export default defineConfig({
  // No react plugin: esbuild handles automatic JSX, keeping the example's deps
  // minimal — trading fast-refresh for zero extra tooling.
  esbuild: { jsx: 'automatic' },
});
