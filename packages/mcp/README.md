# @lenke/mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server for [lenke](../../README.md). Point an MCP-capable assistant (Claude Code, Claude Desktop, Cursor, …) at it and it can run GQL and graph algorithms against a scratch graph, validate queries, and read how-to guides — so it helps you build with lenke against real behavior instead of guessing.

## What it exposes

**Tools** (run against the portable pure-TS engine — no native binary needed):

| Tool            | What it does                                                                                                                            |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `gql_run`       | Run an ISO-GQL query against a scratch graph (optionally seeded from NDJSON) and return the rows.                                       |
| `gql_check`     | Parse a query without running it; report validity, the parse error, and a syntax hint.                                                  |
| `algorithm_run` | Run an in-engine algorithm (pagerank, degree, components, centrality, shortest path, neighbor aggregation, …) and return per-node rows. |

**Resources** — how-to guides addressed `lenke://guide/<id>`: `overview`, `getting-started`, `gql`, `gremlin`, `algorithms`, `arrow`, `multiplayer-sync`, `workers`, `transactions`, `typed-nodes`, `serialization`.

## Configure it

The server speaks MCP over stdio. Add it to your client's MCP config, pointing at the `lenke-mcp` binary.

**Claude Code**

```sh
claude mcp add lenke -- npx -y @lenke/mcp
```

**Claude Desktop / Cursor** (`claude_desktop_config.json` / `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "lenke": {
      "command": "npx",
      "args": ["-y", "@lenke/mcp"]
    }
  }
}
```

Running from a checkout instead of a published package:

```json
{
  "mcpServers": {
    "lenke": { "command": "node", "args": ["/path/to/pl-graph/packages/mcp/bin/lenke-mcp.mjs"] }
  }
}
```

## Try it

Once connected, ask your assistant things like _"use lenke to find the shortest path between these accounts,"_ or _"help me write a GQL query for multi-hop transfers under a per-hop amount limit."_ It can validate the query with `gql_check`, run it on sample data with `gql_run`, and consult the guides.

## Notes

- Zero runtime dependencies beyond `@lenke/*`; the MCP framing is hand-written.
- The tools use the pure-TS engine, so a query runs anywhere Node runs. For native throughput, Arrow egress, Gremlin string execution, and the sync/worker setups, see the `arrow`, `gremlin`, `multiplayer-sync`, and `workers` guides.
