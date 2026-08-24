# @lenke/cli

> A REPL and command-line tool for lenke: load a graph through any codec, query it in **GQL or Gremlin**, and serialize it back out.

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

## The REPL

The REPL **is Node's REPL** with the lenke helpers preloaded — so you get the
whole language (multiline, `await`, history, tab-completion) alongside the graph,
and query results auto-render as tables.

```sh
lenke                      # empty graph
lenke graph.ndjson         # load a file (codec inferred from the extension)
```

```text
lenke> g
Graph — 2 vertices, 1 edges (version 3)
Vertex labels
  Person  2
…
lenke> query('MATCH (p:Person) RETURN p.name, p.age')
┌────────┬───────┐
│ p.name │ p.age │
├────────┼───────┤
│ marko  │ 29    │
│ josh   │ 32    │
└────────┴───────┘
lenke> query('MATCH (p:Person) RETURN p.name, p.age').filter((r) => r['p.age'] > 30)
┌────────┬───────┐
│ p.name │ p.age │
├────────┼───────┤
│ josh   │ 32    │
└────────┴───────┘
lenke> gremlin("g.V().hasLabel('Person').count()")
┌───────┐
│ value │
├───────┤
│ 2     │
└───────┘
```

Because it's the real REPL, a query returns plain data you can keep working with
in JavaScript (`.filter`, `.map`, assign to a variable, …); it just renders as a
table when it's the value of the line.

Helpers available in the session:

| Helper                 | Does                                                |
| ---------------------- | --------------------------------------------------- |
| `g`                    | the current graph (type it for a summary)           |
| `query('…'[, params])` | run a GQL query → rows                              |
| `gremlin('…')`         | run a Gremlin traversal → results                   |
| `describe([graph])`    | the graph summary as data (counts, labels, indexes) |
| `table(rows)`          | render rows as a table string                       |
| `load('file'[, fmt])`  | load a graph from a file (replaces `g`)             |
| `save('file'[, fmt])`  | serialize the graph to a file                       |
| `formats`              | the list of codec names                             |

> The interactive REPL is **Node-only** — Bun's `node:repl` has no `start()`. The
> one-shot (`-q`) and conversion (`-o`) modes below work on both Node and Bun.

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
