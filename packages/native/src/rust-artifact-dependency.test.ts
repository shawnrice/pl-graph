import { describe, expect, test } from 'bun:test';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';

// Guard for the failure mode the `boundaries/no-cross-package-relative-import`
// lint rule CANNOT catch: a package that depends on the compiled Rust engine by
// loading its artifact at runtime — `createFfiEngineBackend('…/liblenke_engine.so')` or
// the wasm module by path. There is no import, so nx's project graph sees no edge
// to lenke-core. If such a package's nx `test` target does not key on `rustEngine`,
// a Rust-only change leaves its cache key unchanged and nx serves a STALE PASS
// (exactly the class of bug that let a broken native test cache as green).
//
// This test walks every workspace package, finds the ones whose source loads the
// artifact, and asserts each declares the dependency in its project.json. A new
// package that starts loading the cdylib/wasm and forgets to wire `rustEngine`
// fails here — the automatic backstop the import linter can't provide.

const ROOT = path.resolve(import.meta.dir, '../../..');
const PACKAGES_DIR = path.join(ROOT, 'packages');

// Runtime paths to the compiled cdylib / wasm — the couplings invisible to nx.
const ARTIFACT_SIGNATURE = /crates\/lenke-engine\/target\/|liblenke_engine|lenke_engine\.wasm/;

const tsFilesUnder = (dir: string): string[] => {
  const out: string[] = [];

  for (const entry of readdirSync(dir)) {
    if (entry === 'node_modules' || entry === 'dist') {
      continue;
    }

    const full = path.join(dir, entry);

    if (statSync(full).isDirectory()) {
      out.push(...tsFilesUnder(full));
    } else if (full.endsWith('.ts') || full.endsWith('.tsx')) {
      out.push(full);
    }
  }

  return out;
};

const filesLoadingArtifact = (pkgDir: string): string[] => {
  const src = path.join(pkgDir, 'src');

  if (!existsSync(src)) {
    return [];
  }

  return tsFilesUnder(src).filter((f) => ARTIFACT_SIGNATURE.test(readFileSync(f, 'utf8')));
};

const testTargetInputs = (pkgDir: string): string[] => {
  const projectFile = path.join(pkgDir, 'project.json');

  if (!existsSync(projectFile)) {
    return [];
  }

  const parsed = JSON.parse(readFileSync(projectFile, 'utf8')) as {
    targets?: { test?: { inputs?: string[] } };
  };

  return parsed.targets?.test?.inputs ?? [];
};

const packageDirs = readdirSync(PACKAGES_DIR).filter((d) =>
  statSync(path.join(PACKAGES_DIR, d)).isDirectory(),
);

const consumers = packageDirs.filter(
  (p) => filesLoadingArtifact(path.join(PACKAGES_DIR, p)).length,
);

describe('nx cache correctness: packages that load the Rust artifact declare it', () => {
  for (const pkg of consumers) {
    test(`@lenke/${pkg} keys its nx test target on rustEngine`, () => {
      const pkgDir = path.join(PACKAGES_DIR, pkg);
      const inputs = testTargetInputs(pkgDir);

      if (!inputs.includes('rustEngine')) {
        const hits = filesLoadingArtifact(pkgDir).map((f) => path.relative(ROOT, f));

        throw new Error(
          `packages/${pkg} loads the compiled Rust engine at runtime (${hits.join(', ')}) but its ` +
            `project.json test target does not list "rustEngine" in inputs. Without it a Rust-only ` +
            `change leaves this package's test cache key unchanged, so nx serves a STALE PASS. ` +
            `Add packages/${pkg}/project.json:\n` +
            `  { "name": "@lenke/${pkg}", "targets": { "test": {\n` +
            `      "inputs": ["default", "^production", "rustEngine"],\n` +
            `      "dependsOn": ["lenke-engine:build"] } } }`,
        );
      }

      expect(inputs).toContain('rustEngine');
    });
  }

  // Sanity: the scanner must actually find the known consumers, or a broken
  // signature/path would make every assertion above vacuously pass.
  test('the scanner sees the known consumers (native, sync)', () => {
    expect(consumers).toContain('native');
    expect(consumers).toContain('sync');
  });
});
