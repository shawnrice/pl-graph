# Dogfooding lenke — persona-driven DX validation

We repeatedly hand fresh "personas" (LLM agents that have **never used lenke**)
only the public docs + API and ask them to **build a real, runnable app** and
report every friction point. Then we fix what they hit, **rebuild** (re-run the
same personas cold), and compare — iterating until the pain points are gone.

This directory is the durable record so the loop survives context compaction.

## The loop

1. **Build** — spawn N personas on distinct app domains; each writes a runnable
   slice under `.dogfood/roundN/<persona>/` (gitignored scratch), from docs only.
2. **Report** — each returns: working slice? / ranked rough spots / what was
   smooth / doc-accuracy (did examples run as written?). Ambitious rounds add
   **capability gaps** + **scale/perf**.
3. **Verify** — check each claimed bug against the code before acting (personas
   are LLMs; some findings are wrong).
4. **Fix** — commit fixes locally (NEVER push — standing constraint).
5. **Rebuild** — re-run fresh personas on the same domains; measure the delta
   (which friction is gone, what's newly smooth, any regressions).

See [`findings/`](./findings/) for each round's raw persona reports. The `ROADMAP.md` pain-point tracker was retired once every item it tracked had shipped or been decided against.

## Confirmed use cases we want to support

The round-4 ambitious apps are use cases the project **intends to support**
(owner-confirmed). They define the bar:

| Domain                                                | Persona       | What it stresses                                                                                   |
| ----------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------- |
| **ReBAC authorization** (Zanzibar-style)              | Priyanka (r4) | traversal `check()` at 200k tuples, multi-anchor index seeking, sync tuple propagation             |
| **Graph analytics / feature engineering**             | Marcus (r4)   | PageRank/components/centrality at 100k–600k nodes, iterative expressiveness, Arrow columnar export |
| **Schema-validated data layer ("Prisma for graphs")** | Lena (r4)     | constraints, migrations, event-veto enforcement, transactions                                      |
| **Real-time multiplayer state server**                | Kenji (r4)    | 48 concurrent WS clients, per-viewport scoping, presence, reconnect, convergence                   |
| **Bitemporal knowledge graph**                        | Sofia (r4)    | valid-time + transaction-time, as-of queries, event-sourcing, snapshots-as-checkpoints             |

Earlier rounds (1–3) covered: browser knowledge-graph (wasm+React+GQL), Node
recommendation service (napi+bulk+prepared), ETL format migration
(serialization), Gremlin dependency analysis, offline-first sync, in-process
Graph + event system, CSV paired-file import, encrypted-snapshot WS sync,
tree/list + advanced Gremlin.

## Persona code index + how to re-run

Code lives under `.dogfood/roundN/<persona>/` (source is committed; generated
`*.ndjson` / data `*.json` / `out/` are gitignored — regenerate with each
persona's `gen*.ts`). Run each with `bun <entry>.ts` from its dir.

- **round1** (`.dogfood/{maya,raj,elena,priya,tomas}/`) — the code sits at the
  top level (no `round1/` prefix; it predates the convention).
- **round2** (`.dogfood/round2/{maya,raj,elena,priya,tomas}/`)
- **round3** (`.dogfood/round3/{diego,yuki,sven,tomas→nadia,amara}/`) — Arrow,
  Graph-hooks+events, CSV paired, WS+crypto, tree/list.
- **round4** (`.dogfood/round4/{priyanka,marcus,lena,kenji,sofia}/`) — the
  ambitious set above. Entry points: `authz.ts`, `analytics.ts`, `demo.ts`,
  `canvas.ts`, `demo.ts` respectively; each has a `gen.ts` for its data.

Environment the personas assume (repo plumbing, not part of the DX eval):
`@lenke/*` resolve from any dir under the repo root; FFI lib at
`crates/lenke-core/target/release/liblenke_core.so`; napi via
`@lenke/node/backend`; wasm at
`crates/lenke-core/target/wasm32-unknown-unknown/release/lenke_core.wasm`.

**Before a rebuild:** `bun run build` (dist must be current) — personas import
built `@lenke/*`.
