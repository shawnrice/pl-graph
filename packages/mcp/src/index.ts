// @lenke/mcp — a Model Context Protocol server for lenke. Point an MCP-capable
// assistant (e.g. Claude) at the `lenke-mcp` binary and it can run GQL and graph
// algorithms against a scratch graph, validate queries, and read how-to guides —
// so it can help you build with lenke instead of guessing.

import { RESOURCES } from './guides.js';
import { createServer, type Dispatch, runStdio } from './protocol.js';
import { TOOLS } from './tools.js';

export * from './protocol.js';
export { TOOLS } from './tools.js';
export { GUIDES, RESOURCES, type Guide } from './guides.js';

const VERSION = '0.1.0';

/** Build the lenke MCP dispatcher (tools + guide resources), for tests or a
 * custom transport. */
export const createLenkeServer = (): Dispatch =>
  createServer({ name: 'lenke', version: VERSION, tools: TOOLS, resources: RESOURCES });

/** Entry point: serve over stdio. Wired by `bin/lenke-mcp.mjs`. */
export const main = async (): Promise<void> => {
  await runStdio(createLenkeServer());
};
