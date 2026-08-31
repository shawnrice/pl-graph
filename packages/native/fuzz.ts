/**
 * Unified runner for the byte-identity fuzzers.
 *
 * The fuzzers all share one convention already — `FUZZ_SEED=<n>` replays a run
 * exactly, and each prints its failing seed — so this script is the missing entry
 * point on top of that: run them ALL for a regression sweep, or ONE deterministically
 * on a chosen seed, without remembering which artifacts each needs rebuilt first.
 *
 * That last part is the point. `backend-parity` compares the wasm build against the
 * FFI build, and NEITHER `bun run build` nor `cargo build` rebuilds the wasm — a stale
 * `.wasm` turns it into a comparison of your change against an old copy of itself
 * (see CLAUDE.md). `gremlin` runs against the built `@lenke/gremlin` dist. This runner
 * rebuilds exactly the artifacts the selected fuzzers need, in order, before running.
 *
 * Usage (from packages/native, or `bun run fuzz …` from anywhere in the repo):
 *   bun run fuzz                      # all fuzzers, random seeds — the sweep
 *   bun run fuzz gremlin              # just gremlin, random seed
 *   bun run fuzz gremlin --seed 2977  # just gremlin, deterministic replay
 *   FUZZ_SEED=2977 bun run fuzz gremlin          # same, via the env convention
 *   bun run fuzz codec differential --seed 7     # a subset, one shared seed
 *   bun run fuzz parity --no-build    # skip the rebuild (you just built)
 *   bun run fuzz --list               # what's available
 *
 * A shared `--seed` makes every selected fuzzer replay that exact seed; with no seed
 * each fuzzer randomizes its own (a broad sweep, as CI runs them).
 */

type Artifact = 'rust' | 'wasm' | 'gremlin';

type Fuzzer = {
  file: string;
  needs: Artifact[];
  blurb: string;
};

// Order here is the run order; `needs` drives which artifacts get rebuilt.
const FUZZERS: Record<string, Fuzzer> = {
  codec: {
    file: 'src/codec-fuzz.test.ts',
    needs: ['rust'],
    blurb: 'snapshot/codec round-trip — TS vs the FFI build are byte-identical',
  },
  differential: {
    file: 'src/differential-fuzz.test.ts',
    needs: ['rust'],
    blurb: 'random GQL — the pure-TS engine vs the Rust engine',
  },
  write: {
    file: 'src/write-fuzz.test.ts',
    needs: ['rust'],
    blurb: 'random writes — TS vs the FFI build',
  },
  injection: {
    file: 'src/injection-fuzz.test.ts',
    needs: ['rust'],
    blurb: 'hostile params stay inert — TS vs the FFI build',
  },
  algo: {
    file: 'src/algo-fuzz.test.ts',
    needs: ['rust'],
    blurb: 'graph algorithms — TS vs the FFI build',
  },
  gremlin: {
    file: 'src/gremlin-fuzz.test.ts',
    needs: ['rust', 'gremlin'],
    blurb: 'random Gremlin — the built @lenke/gremlin dist vs the Rust engine',
  },
  parity: {
    file: 'src/backend-parity-fuzz.test.ts',
    needs: ['rust', 'wasm'],
    blurb: 'the wasm build vs the FFI build (needs a FRESH wasm)',
  },
};

// Build commands, run from packages/native. `rust`/`wasm` are the package scripts;
// the gremlin dist is an nx target resolved from the repo root.
const BUILD: Record<Artifact, { label: string; cmd: string[] }> = {
  rust: { label: 'build:rust (the .so)', cmd: ['bun', 'run', 'build:rust'] },
  wasm: { label: 'build:wasm (the .wasm)', cmd: ['bun', 'run', 'build:wasm'] },
  gremlin: {
    label: 'nx build @lenke/gremlin (dist)',
    cmd: ['bunx', 'nx', 'build', '@lenke/gremlin'],
  },
};

const ARTIFACT_ORDER: Artifact[] = ['rust', 'wasm', 'gremlin'];

const list = (): void => {
  console.log('Fuzzers (name — needs — what it checks):\n');

  for (const [name, f] of Object.entries(FUZZERS)) {
    console.log(`  ${name.padEnd(13)} [${f.needs.join(', ')}]  ${f.blurb}`);
  }

  console.log('\n  all            every fuzzer (the default)');
};

const help = (): void => {
  console.log(
    [
      'bun run fuzz [targets…] [--seed <n>] [--no-build] [--list]',
      '',
      '  targets     one or more fuzzer names (default: all). See --list.',
      '  --seed <n>  replay this exact seed on every selected fuzzer',
      '              (or set FUZZ_SEED=<n>). No seed → each randomizes (a sweep).',
      '  --no-build  skip rebuilding artifacts (use when you just built).',
      '  --list      list the fuzzers and what each needs.',
    ].join('\n'),
  );
};

const args = process.argv.slice(2);
let seed: string | undefined = process.env.FUZZ_SEED;
let noBuild = false;
const targets: string[] = [];

for (let i = 0; i < args.length; i++) {
  const a = args[i];

  if (a === '--seed' || a === '-s') {
    seed = args[++i];
  } else if (a.startsWith('--seed=')) {
    seed = a.slice('--seed='.length);
  } else if (a === '--no-build') {
    noBuild = true;
  } else if (a === '--list') {
    list();
    process.exit(0);
  } else if (a === '--help' || a === '-h') {
    help();
    process.exit(0);
  } else if (a === '--') {
    continue;
  } else if (a === 'all') {
    targets.push(...Object.keys(FUZZERS));
  } else if (a in FUZZERS) {
    targets.push(a);
  } else {
    console.error(`unknown fuzzer '${a}'. Run \`bun run fuzz --list\`.`);
    process.exit(2);
  }
}

// Default is the whole sweep. Dedup while preserving the registry's run order.
const wanted = new Set(targets.length > 0 ? targets : Object.keys(FUZZERS));
const selected = Object.keys(FUZZERS).filter((n) => wanted.has(n));

if (seed !== undefined && !/^\d+$/.test(seed)) {
  console.error(`--seed must be a non-negative integer, got '${seed}'`);
  process.exit(2);
}

// Rebuild exactly the artifacts the selected fuzzers need — the anti-stale guard.
if (!noBuild) {
  const need = new Set<Artifact>();

  for (const t of selected) {
    for (const a of FUZZERS[t].needs) {
      need.add(a);
    }
  }

  for (const a of ARTIFACT_ORDER) {
    if (!need.has(a)) {
      continue;
    }

    console.log(`\n▸ ${BUILD[a].label}`);
    const r = Bun.spawnSync({
      cmd: BUILD[a].cmd,
      stdout: 'inherit',
      stderr: 'inherit',
      stdin: 'inherit',
    });

    if (r.exitCode !== 0) {
      console.error(`\nbuild failed (${a}); aborting before the fuzzers run.`);
      process.exit(r.exitCode ?? 1);
    }
  }
}

const files = selected.map((t) => FUZZERS[t].file);
const seedNote = seed !== undefined ? `FUZZ_SEED=${seed} (deterministic)` : 'random seeds';
console.log(`\n▸ fuzzing: ${selected.join(', ')}  —  ${seedNote}\n`);

const env = { ...process.env };

if (seed !== undefined) {
  env.FUZZ_SEED = seed;
} else {
  delete env.FUZZ_SEED;
}

const run = Bun.spawnSync({
  cmd: ['bun', 'test', ...files],
  env,
  stdout: 'inherit',
  stderr: 'inherit',
  stdin: 'inherit',
});
process.exit(run.exitCode ?? 0);
