import { ErrorCode, LenkeError } from '@lenke/errors';

import type { Edge, Graph, Vertex } from './core/index.js';

/**
 * The [Standard Schema](https://standardschema.dev) v1 interface, inlined
 * (types-only, zero-dependency by design — the spec is meant to be copied). Any
 * validation library that implements it — Zod ≥3.24, Valibot, ArkType, … —
 * exposes a `~standard` property and is accepted here verbatim. lenke owns no
 * schema DSL of its own: you bring your validator, we consume the spec.
 *
 * We deliberately flatten the spec's `namespace` into standalone types (the lint
 * gate disallows TS namespaces) — the shape is otherwise identical.
 */
export type StandardSchemaV1<Input = unknown, Output = Input> = {
  readonly '~standard': StandardSchemaProps<Input, Output>;
};

type StandardSchemaProps<Input, Output> = {
  readonly version: 1;
  readonly vendor: string;
  readonly validate: (value: unknown) => StandardResult<Output> | Promise<StandardResult<Output>>;
  readonly types?: { readonly input: Input; readonly output: Output } | undefined;
};

type StandardResult<Output> =
  | { readonly value: Output; readonly issues?: undefined }
  | { readonly issues: ReadonlyArray<StandardIssue> };

type StandardIssue = {
  readonly message: string;
  readonly path?: ReadonlyArray<PropertyKey | { readonly key: PropertyKey }> | undefined;
};

/** The input type a Standard Schema accepts (what you pass to `create`). */
export type InferInput<S extends StandardSchemaV1> = NonNullable<S['~standard']['types']>['input'];

/** The output type a Standard Schema produces (what gets stored). */
export type InferOutput<S extends StandardSchemaV1> = NonNullable<
  S['~standard']['types']
>['output'];

/** Render a `key.0.name`-style path for an issue, for a legible error message. */
const formatPath = (issue: StandardIssue): string =>
  (issue.path ?? [])
    .map((seg) => (typeof seg === 'object' ? String(seg.key) : String(seg)))
    .join('.');

/**
 * Run a Standard Schema over `input`, throwing a `ConstraintViolation` that
 * lists every issue on failure. Awaits the validator — a Standard Schema may
 * validate asynchronously (regex/refinements/network), so this is always async
 * even for a synchronous schema.
 */
const validateWith = async <S extends StandardSchemaV1>(
  label: string,
  schema: S,
  input: InferInput<S>,
): Promise<InferOutput<S>> => {
  const result = await schema['~standard'].validate(input);

  if (result.issues) {
    // Normalize each Standard-Schema issue to `{ message, path }` (path as the
    // `key.0.name` dotted string used in the message). Attached as
    // `details.issues` so a caller can handle failures field-by-field without
    // parsing the joined message string.
    const issues = result.issues.map((i) => ({ message: i.message, path: formatPath(i) }));
    const detail = issues.map((i) => (i.path ? `${i.path}: ${i.message}` : i.message)).join('; ');

    throw new LenkeError(`lenke: ${label} failed schema validation — ${detail}`, {
      code: ErrorCode.ConstraintViolation,
      details: { issues },
    });
  }

  return result.value;
};

/**
 * A label bound to a Standard Schema: `create` validates an input host-side and,
 * on success, writes it as a `label`-tagged vertex. `parse` validates without
 * writing. Reusable across graphs (the schema holds no graph).
 */
export type NodeSchema<S extends StandardSchemaV1> = {
  readonly label: string;
  readonly schema: S;
  /** Validate `input` (throwing `ConstraintViolation` on failure) without writing. */
  parse: (input: InferInput<S>) => Promise<InferOutput<S>>;
  /**
   * Validate `input`, then add it as a `label`-tagged vertex to `graph`. The
   * validated (parsed/coerced) OUTPUT is what gets stored, so a schema that
   * trims/defaults/coerces persists its normalized value.
   */
  create: (graph: Graph, input: InferInput<S>) => Promise<Vertex>;
};

/**
 * Bind a node `label` to any [Standard Schema](https://standardschema.dev) —
 * a Zod/Valibot/ArkType object schema, or anything exposing `~standard`. You
 * get the schema's own type inference for free (`create`'s input is typed) plus
 * validate-before-write, with no schema DSL to learn and no dependency added.
 *
 * This validates HOST-side, before the write — it guards `create` calls, not a
 * raw GQL `INSERT` or `mergeNdjson` (those write through the engine, which can't
 * run a JS schema). For in-engine, both-engine enforcement that also guards raw
 * writes, use the constraint surface (`createTypeConstraint` /
 * `createRequiredConstraint` / `createUniqueConstraint` / `createValidator`);
 * the two compose — a schema at the app boundary, constraints at the engine.
 *
 * **Type inference needs `~standard.types`.** `create`'s input/output typing comes
 * from `InferInput`/`InferOutput`, which read the schema's `~standard.types`
 * carrier. Zod/Valibot/ArkType expose it, so inference is automatic. A HAND-ROLLED
 * validator that only implements `~standard.validate` (no `~standard.types`) still
 * validates at runtime, but `create`/`parse` silently infer `never` for the input
 * — so add a `~standard.types` (typically a phantom `{} as { input: T; output: T }`)
 * if you want typed calls.
 *
 * @example
 * const User = defineNode('User', z.object({ name: z.string(), age: z.number().optional() }))
 * const v = await User.create(graph, { name: 'ada' }) // typed + validated, then stored
 */
export const defineNode = <S extends StandardSchemaV1>(
  label: string,
  schema: S,
): NodeSchema<S> => ({
  label,
  schema,
  parse: (input) => validateWith(label, schema, input),
  create: async (graph, input) => {
    const value = await validateWith(label, schema, input);

    return graph.addVertex({ labels: [label], properties: value as Record<string, unknown> });
  },
});

/**
 * An endpoint accepted by `EdgeSchema.create`: a live `Vertex` OR a bare vertex
 * `id` (string). Passing ids avoids the ergonomic tax of holding onto `Vertex`
 * objects — the edge only needs an endpoint the graph can resolve.
 *
 * ⚠️ A string `VertexRef` is resolved as the vertex **id** (the graph's UUID /
 * `Vertex.id`, via `getVertexById`), NOT as a natural/business key property. A
 * common footgun: passing `'alice'` because a vertex has `{ name: 'alice' }` will
 * NOT match — it looks up the vertex whose *id* is `'alice'` and throws
 * `MissingVertex` if none exists. To key off a business property, look the vertex
 * up yourself first (e.g. a GQL/Gremlin match on that property) and pass the
 * resulting `Vertex` (or its `.id`).
 */
export type VertexRef = Vertex | string;

/** Resolve a `Vertex`-or-id endpoint to the live vertex, or throw `MissingVertex`. */
const resolveEndpoint = (graph: Graph, ref: VertexRef, role: 'from' | 'to'): Vertex => {
  if (typeof ref !== 'string') {
    return ref;
  }

  const vertex = graph.getVertexById(ref);

  if (!vertex) {
    throw new LenkeError(`lenke: edge '${role}' endpoint vertex '${ref}' is not in this graph`, {
      code: ErrorCode.MissingVertex,
    });
  }

  return vertex;
};

/**
 * An edge type bound to a Standard Schema: `create` validates typed edge
 * properties host-side and, on success, writes an `edgeType`-labeled edge
 * between two endpoints. `parse` validates without writing. Reusable across
 * graphs (the schema holds no graph). The `edgeType` mirror of `defineNode`.
 */
export type EdgeSchema<S extends StandardSchemaV1> = {
  readonly edgeType: string;
  readonly schema: S;
  /** Validate `input` (throwing `ConstraintViolation` on failure) without writing. */
  parse: (input: InferInput<S>) => Promise<InferOutput<S>>;
  /**
   * Validate `input`, then add an `edgeType`-labeled edge from `from` to `to`.
   * Endpoints may be `Vertex` objects OR bare vertex ids (strings) — a string is
   * the vertex **id** (UUID), NOT a business-key property (see {@link VertexRef}),
   * resolved against `graph` and throwing `MissingVertex` if absent. The validated
   * (parsed/coerced) OUTPUT is what gets stored.
   */
  create: (graph: Graph, from: VertexRef, to: VertexRef, input: InferInput<S>) => Promise<Edge>;
};

/**
 * Bind an `edgeType` to any [Standard Schema](https://standardschema.dev) — the
 * edge-property mirror of `defineNode`. You get the schema's own type inference
 * for free (`create`'s input is typed) plus validate-before-write, with no
 * schema DSL to learn and no dependency added.
 *
 * `create` takes the two endpoints (`Vertex` objects OR bare vertex ids —
 * Marcus's ergonomic tax: you rarely still hold the `Vertex`) plus the typed
 * props. Like `defineNode`, this validates HOST-side, before the write — it
 * guards `create` calls, not a raw GQL `INSERT` or `mergeNdjson`. For
 * both-engine enforcement that also guards raw writes, compose with the engine
 * edge constraints (`createEdgeRequiredConstraint` / `createEdgeTypeConstraint`
 * / `createEdgeUniqueConstraint`): a schema at the app boundary, constraints at
 * the engine.
 *
 * **Type inference needs `~standard.types`** (as with `defineNode`): a hand-rolled
 * validator that omits the `~standard.types` carrier still validates at runtime,
 * but `create` infers `never` for the props — add `~standard.types` for typed calls.
 *
 * @example
 * const Follows = defineEdge('FOLLOWS', z.object({ since: z.number() }))
 * const e = await Follows.create(graph, ada.id, lin.id, { since: 2020 }) // typed + validated
 */
export const defineEdge = <S extends StandardSchemaV1>(
  edgeType: string,
  schema: S,
): EdgeSchema<S> => ({
  edgeType,
  schema,
  parse: (input) => validateWith(edgeType, schema, input),
  create: async (graph, from, to, input) => {
    const value = await validateWith(edgeType, schema, input);

    return graph.addEdge({
      from: resolveEndpoint(graph, from, 'from'),
      to: resolveEndpoint(graph, to, 'to'),
      labels: [edgeType],
      properties: value as Record<string, unknown>,
    });
  },
});
