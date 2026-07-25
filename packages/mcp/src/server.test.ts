import { describe, expect, test } from 'bun:test';

import { createLenkeServer } from './index.js';
import type { JsonRpcMessage } from './protocol.js';

const server = createLenkeServer();

let nextId = 1;
const call = async (method: string, params?: unknown): Promise<Record<string, unknown>> => {
  const msg: JsonRpcMessage = { jsonrpc: '2.0', id: nextId++, method, params };
  const resp = await server(msg);

  if (!resp) {
    throw new Error(`no response for ${method}`);
  }

  if (resp.error) {
    throw new Error(`${method} errored: ${resp.error.message}`);
  }

  return resp.result as Record<string, unknown>;
};

const callTool = async (name: string, args: Record<string, unknown>): Promise<string> => {
  const result = await call('tools/call', { name, arguments: args });
  const content = result.content as { type: string; text: string }[];

  if (result.isError) {
    throw new Error(content[0]?.text);
  }

  return content[0]?.text ?? '';
};

const NDJSON = [
  '{"type":"node","id":"a","labels":["Person"],"properties":{"name":"ada","age":36}}',
  '{"type":"node","id":"b","labels":["Person"],"properties":{"name":"lin","age":29}}',
  '{"type":"edge","from":"a","to":"b","labels":["KNOWS"],"properties":{}}',
].join('\n');

describe('lenke MCP server', () => {
  test('initialize advertises tools + resources', async () => {
    const result = await call('initialize', { protocolVersion: '2024-11-05' });

    expect(result.protocolVersion).toBe('2024-11-05');
    expect((result.serverInfo as { name: string }).name).toBe('lenke');
    expect(result.capabilities).toEqual({ tools: {}, resources: {} });
  });

  test('a notification (no id) gets no response', async () => {
    expect(await server({ jsonrpc: '2.0', method: 'notifications/initialized' })).toBeNull();
  });

  test('tools/list exposes the three tools', async () => {
    const { tools } = (await call('tools/list')) as { tools: { name: string }[] };

    expect(tools.map((t) => t.name).sort()).toEqual(['algorithm_run', 'gql_check', 'gql_run']);
    // Each tool carries a JSON-Schema for its arguments.
    expect(
      tools.every(
        (t) => (t as unknown as { inputSchema: { type: string } }).inputSchema.type === 'object',
      ),
    ).toBe(true);
  });

  test('gql_run executes a query against seeded data', async () => {
    const text = await callTool('gql_run', {
      query: 'MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS src, b.name AS dst',
      data: NDJSON,
    });

    expect(text).toContain('1 row');
    expect(text).toContain('"src": "ada"');
    expect(text).toContain('"dst": "lin"');
  });

  test('gql_run binds params', async () => {
    const text = await callTool('gql_run', {
      query: 'MATCH (a:Person) WHERE a.age > $min RETURN a.name AS name ORDER BY name',
      data: NDJSON,
      params: { min: 30 },
    });

    expect(text).toContain('"name": "ada"');
    expect(text).not.toContain('lin');
  });

  test('gql_check accepts valid GQL and explains invalid GQL', async () => {
    expect(await callTool('gql_check', { query: 'MATCH (a:Person) RETURN a.name' })).toContain(
      'Valid GQL',
    );

    // A Cypher-style variable-length path — invalid here, with a pointed note.
    const bad = await callTool('gql_check', { query: 'MATCH (a)-[:R*1..5]->(b) RETURN b' });
    expect(bad).toContain('Invalid GQL');
    expect(bad).toContain('{1,5}');
  });

  test('algorithm_run computes degree over seeded data', async () => {
    const text = await callTool('algorithm_run', {
      name: 'degree',
      config: { direction: 'both' },
      data: NDJSON,
    });

    expect(text).toContain('2 rows');
    expect(text).toContain('"degree": 1'); // a↔b, each degree 1
  });

  test('a tool error is reported as an isError result, not a protocol error', async () => {
    const resp = await server({
      jsonrpc: '2.0',
      id: 99,
      method: 'tools/call',
      params: { name: 'gql_run', arguments: { query: 'THIS IS NOT GQL' } },
    });

    expect(resp?.error).toBeUndefined();
    const result = resp?.result as { isError?: boolean; content: { text: string }[] };
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain('Error:');
  });

  test('resources/list + resources/read serve the guides', async () => {
    const { resources } = (await call('resources/list')) as {
      resources: { uri: string; mimeType: string }[];
    };

    expect(resources.length).toBeGreaterThanOrEqual(10);
    expect(resources.every((r) => r.uri.startsWith('lenke://guide/'))).toBe(true);

    const gql = resources.find((r) => r.uri === 'lenke://guide/gql');
    expect(gql).toBeDefined();

    const read = (await call('resources/read', { uri: gql!.uri })) as {
      contents: { text: string; mimeType: string }[];
    };
    expect(read.contents[0].mimeType).toBe('text/markdown');
    expect(read.contents[0].text).toContain('Variable-length paths');
  });

  test('unknown method / tool / resource error cleanly', async () => {
    expect((await server({ jsonrpc: '2.0', id: 1, method: 'nope' }))?.error?.code).toBe(-32601);
    expect(
      (await server({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'ghost' } }))
        ?.error?.code,
    ).toBe(-32602);
    expect(
      (await server({ jsonrpc: '2.0', id: 3, method: 'resources/read', params: { uri: 'x' } }))
        ?.error?.code,
    ).toBe(-32602);
  });
});
