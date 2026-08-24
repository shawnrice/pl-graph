# @lenke/cli

> A graph shell and command-line tool for lenke: explore a graph interactively in **GQL** (or Gremlin), load bundled sample graphs, and convert between codecs.

For quick analysis and trying queries against a graph file without writing a
script. It runs the Rust engine as WebAssembly, so it works on plain Node or Bun
with no native addon.

## Install / run

In this monorepo it's the `lenke` bin:

```sh
bun run build                     # builds the JS packages
bun run build:wasm                # builds the wasm engine the CLI loads (separate step)
./packages/cli/bin/lenke.mjs      # or `lenke` once linked / installed
```

The CLI needs the wasm engine (`lenke_engine.wasm`). It looks at the `--wasm <path>`
flag, then `$LENKE_WASM`, then the workspace build output — so after
`bun run build:wasm` it just works in-repo.

## The shell

A `psql`-style shell. **GQL is the default** — type a query, press Enter, get a
table. Backslash meta-commands drive the session, and the prompt always shows the
mode, so there's no guessing what a line means. Runs on Node **and** Bun.

```sh
lenke                      # empty graph
lenke graph.ndjson         # load a file (codec inferred from the extension)
```

Start with a bundled sample — no data file needed:

```text
lenke=# \l
  modern   Apache TinkerPop "modern": 6 person/software vertices, weighted edges.
  dunder   Dunder Mifflin (The Office) — 24 employees over 9 seasons, bitemporal.
lenke=# \c dunder
loaded dunder — 26 vertices, 44 edges
lenke=# \clock 2007-06-01
clock: 2007-06-01 (fixed)
lenke=# -- who ran the Scranton branch then?
lenke=# MATCH (p:Person)-[m:MANAGES]->(:Company) WHERE m.vf <= current_date AND m.vt > current_date
lenke-#   RETURN p.name AS regional_manager
┌──────────────────┐
│ regional_manager │
├──────────────────┤
│ Michael Scott    │
└──────────────────┘
```

The manager's chair turns over almost every season, so re-run that with
`\clock 2013-04-01` and you get `Dwight Schrute`. The `dunder` graph is
**bitemporal** — Ryan Howard's VP tenure was recorded open-ended, then corrected
when he was fired, so his record reads differently depending on the system-time
you ask "as recorded on".

A query that isn't finished (an open bracket, or a trailing `\`) continues on the
next line — the prompt turns to `lenke-#`. No `;` needed.

### Modes

The language is an explicit mode, shown in the prompt — never sniffed:

| Command     | Prompt             | What you type                                 |
| ----------- | ------------------ | --------------------------------------------- |
| _(default)_ | `lenke=#`          | GQL: `MATCH (p:Person) RETURN p.name`         |
| `\gremlin`  | `lenke(gremlin)=#` | Gremlin: `g.V().hasLabel('Person').count()`   |
| `\js`       | `lenke(js)=#`      | JavaScript, where `_` is the last result rows |

`\js` is the escape hatch for keeping data in JavaScript:

```text
lenke=# MATCH (p:Person)-[e:WORKS_AT]->(:Company) WHERE e.vt > current_date
lenke-#   RETURN p.name AS person, e.role AS role, e.dept AS dept
… (a table of current employees) …
lenke=# \js
lenke(js)=# _.filter((r) => r.dept === 'Sales').map((r) => r.person)
┌────────────────┐
│ value          │
├────────────────┤
│ Jim Halpert    │
│ Phyllis Vance  │
…
```

### Meta-commands

| Command                       | Does                                                          |
| ----------------------------- | ------------------------------------------------------------- |
| `\l`                          | list the bundled sample graphs                                |
| `\c <name\|file> [fmt]`       | load a sample by name, or a file (codec from extension/`fmt`) |
| `\d`                          | describe the graph (labels + counts)                          |
| `\dv` / `\de`                 | list vertex / edge labels                                     |
| `\d <Label>`                  | property keys + element count for a label                     |
| `\clock [date\|now\|off]`     | as-of date for `current_date` (defaults to the system clock)  |
| `\format table\|json\|ndjson` | how results render (`json`/`ndjson` are pipe-friendly)        |
| `\timing on\|off`             | show query time                                               |
| `\i <file>`                   | run queries from a file (one per line)                        |
| `\o <file>\|off`              | also write query output to a file                             |
| `\save <file> [fmt]`          | serialize the graph to a file                                 |
| `\r`                          | reset the current input buffer                                |
| `\?` / `\q`                   | help / quit                                                   |

Tab completes meta-commands, GQL keywords / Gremlin steps, and the loaded graph's
labels. History persists in `~/.lenke_history`.

## One-shot & conversion

```sh
# run a single query and exit
lenke graph.csv -q "MATCH (p:Person) RETURN p.name, p.age"
lenke graph.ndjson -q "g.V().hasLabel('Person').count()"

# convert between codecs (load one, save another)
lenke graph.graphson -o graph.ndjson
```

Codecs (`--format` / `--out-format`, or inferred from the extension):
`ndjson` · `csv` · `graphson` · `pg-json` · `pg-text`. Run `lenke --help` for all
options.

## License

Apache-2.0
