// A small, dependency-free Model Context Protocol server over stdio. MCP is
// JSON-RPC 2.0 with newline-delimited messages on stdin/stdout; we implement the
// handful of methods a tools+resources server needs (initialize, tools/list,
// tools/call, resources/list, resources/read) by hand rather than pull in an SDK.

/** The protocol revision we advertise; we echo the client's if it sends one. */
const PROTOCOL_VERSION = '2024-11-05';

export type JsonRpcId = string | number | null;

export type JsonRpcMessage = {
  jsonrpc: '2.0';
  id?: JsonRpcId;
  method?: string;
  params?: unknown;
};

export type JsonRpcResponse = {
  jsonrpc: '2.0';
  id: JsonRpcId;
  result?: unknown;
  error?: { code: number; message: string };
};

/** A callable tool: a JSON-Schema for its arguments plus a handler that returns
 * text (or throws — a throw becomes an `isError` tool result, not a protocol
 * error, so the model sees the message and can retry). */
export type ToolDef = {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  handle: (args: Record<string, unknown>) => Promise<string> | string;
};

/** A readable resource (a how-to guide), addressed by `uri`. */
export type ResourceDef = {
  uri: string;
  name: string;
  description: string;
  mimeType: string;
  read: () => string;
};

export type ServerConfig = {
  name: string;
  version: string;
  tools: readonly ToolDef[];
  resources: readonly ResourceDef[];
};

/** A dispatcher: takes one JSON-RPC message and returns its response, or `null`
 * for a notification (no `id` ⇒ nothing to reply). Pure and directly testable. */
export type Dispatch = (msg: JsonRpcMessage) => Promise<JsonRpcResponse | null>;

export const createServer = (config: ServerConfig): Dispatch => {
  const toolByName = new Map(config.tools.map((t) => [t.name, t]));
  const resByUri = new Map(config.resources.map((r) => [r.uri, r]));

  return async (msg) => {
    const id = msg.id ?? null;
    const isNotification = msg.id === undefined;
    const ok = (result: unknown): JsonRpcResponse | null =>
      isNotification ? null : { jsonrpc: '2.0', id, result };
    const fail = (code: number, message: string): JsonRpcResponse | null =>
      isNotification ? null : { jsonrpc: '2.0', id, error: { code, message } };

    try {
      switch (msg.method) {
        case 'initialize': {
          const params = msg.params as { protocolVersion?: unknown } | undefined;
          const protocolVersion =
            typeof params?.protocolVersion === 'string' ? params.protocolVersion : PROTOCOL_VERSION;

          return ok({
            protocolVersion,
            capabilities: { tools: {}, resources: {} },
            serverInfo: { name: config.name, version: config.version },
          });
        }
        // Post-init and cancellation notifications need no reply.
        case 'notifications/initialized':
        case 'notifications/cancelled':
          return null;
        case 'ping':
          return ok({});
        case 'tools/list':
          return ok({
            tools: config.tools.map(({ name, description, inputSchema }) => ({
              name,
              description,
              inputSchema,
            })),
          });
        case 'tools/call': {
          const params = msg.params as { name?: string; arguments?: Record<string, unknown> };
          const tool = params?.name ? toolByName.get(params.name) : undefined;

          if (!tool) {
            return fail(-32602, `unknown tool: ${params?.name ?? '(none)'}`);
          }

          try {
            const text = await tool.handle(params.arguments ?? {});

            return ok({ content: [{ type: 'text', text }] });
          } catch (e) {
            const message = e instanceof Error ? e.message : String(e);

            return ok({ content: [{ type: 'text', text: `Error: ${message}` }], isError: true });
          }
        }
        case 'resources/list':
          return ok({
            resources: config.resources.map(({ uri, name, description, mimeType }) => ({
              uri,
              name,
              description,
              mimeType,
            })),
          });
        case 'resources/templates/list':
          return ok({ resourceTemplates: [] });
        case 'resources/read': {
          const params = msg.params as { uri?: string };
          const res = params?.uri ? resByUri.get(params.uri) : undefined;

          if (!res) {
            return fail(-32602, `unknown resource: ${params?.uri ?? '(none)'}`);
          }

          return ok({ contents: [{ uri: res.uri, mimeType: res.mimeType, text: res.read() }] });
        }
        case 'prompts/list':
          return ok({ prompts: [] });
        default:
          return fail(-32601, `method not found: ${msg.method ?? '(none)'}`);
      }
    } catch (e) {
      return fail(-32603, `internal error: ${e instanceof Error ? e.message : String(e)}`);
    }
  };
};

const writeResponse = (resp: JsonRpcResponse): void => {
  process.stdout.write(`${JSON.stringify(resp)}\n`);
};

/** Drive a {@link Dispatch} over stdin/stdout with newline-delimited JSON. */
export const runStdio = async (dispatch: Dispatch): Promise<void> => {
  const decoder = new TextDecoder();
  let buffer = '';

  for await (const chunk of process.stdin) {
    buffer += decoder.decode(chunk as Uint8Array, { stream: true });

    let nl = buffer.indexOf('\n');

    while (nl !== -1) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      nl = buffer.indexOf('\n');

      if (line === '') {
        continue;
      }

      let msg: JsonRpcMessage;

      try {
        msg = JSON.parse(line) as JsonRpcMessage;
      } catch {
        continue; // ignore a malformed line rather than crash the server
      }

      const resp = await dispatch(msg);

      if (resp) {
        writeResponse(resp);
      }
    }
  }
};
