# @lenke/errors

> Canonical, stable error codes and a shared error type for the lenke packages.

The toolkit's parsers, evaluators, and FFI layers all fail in similar ways (syntax errors, invalid input, resource limits, FFI failures), and message text is not a reliable thing to branch on. This package defines a fixed set of opaque `E_*` codes plus a `LenkeError` carrying one, so consumers can match on `error.code` rather than parsing message strings. Reach for it when you need to detect or classify a failure programmatically across packages.

## Install

```bash
bun add @lenke/errors
```

## Usage

```ts
import { ErrorCode, LenkeError, hasErrorCode, isLenkeError } from '@lenke/errors';

function parse(text: string) {
  throw new LenkeError('unexpected token at line 3', {
    code: ErrorCode.Syntax,
    details: { line: 3, column: 12 },
  });
}

try {
  parse('MATCH (n');
} catch (error) {
  // Branch on the stable code, not the message text.
  if (hasErrorCode(error, ErrorCode.Syntax)) {
    console.error('parse failed:', (error as LenkeError).details);
  } else if (isLenkeError(error)) {
    console.error(error.code, error.message);
  } else {
    throw error;
  }
}
```

## API

`ErrorCode` is a const object (and a type of the same name) of `E_*` string values. The code is the contract: it is stable across releases, while a message may be reworded freely.

```ts
class LenkeError extends Error {
  readonly code: ErrorCode;
  readonly details?: Readonly<Record<string, unknown>>;
  constructor(message: string, options: {
    code: ErrorCode;
    cause?: unknown;       // attached as the standard Error.cause
    details?: Readonly<Record<string, unknown>>;
  });
}

isLenkeError(error: unknown): error is LenkeError;
hasErrorCode(error: unknown, code: ErrorCode): boolean;
```

`hasErrorCode` matches any object that adopts the `code` convention, not only `LenkeError` instances or subclasses, so it works across package and FFI boundaries.

## Error codes

| Code                   | Meaning                                                                                                                                                                                                                                                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `E_SYNTAX`             | A query/text parse or lex failure (GQL / Gremlin / `.pg`).                                                                                                                                                                                                                                                                    |
| `E_INVALID_JSON`       | Input wasn't valid JSON.                                                                                                                                                                                                                                                                                                      |
| `E_INVALID_SHAPE`      | Input parsed but didn't match the expected document shape.                                                                                                                                                                                                                                                                    |
| `E_UNKNOWN_FORMAT`     | An unknown serialization format name.                                                                                                                                                                                                                                                                                         |
| `E_INVALID_VALUE`      | A value outside the LPG property-value model.                                                                                                                                                                                                                                                                                 |
| `E_DATA_EXCEPTION`     | An ISO data exception at evaluation time: division by zero, type mismatch, out-of-range numeric, **temporal arithmetic overflow** (`date`/`datetime`/`duration` out of representable range — never a silent null), or an **unsupported temporal aggregate** (`avg` over a temporal, or `sum` over a non-`DURATION` temporal). |
| `E_MISSING_VERTEX`     | An edge or operation referenced a vertex id that doesn't exist.                                                                                                                                                                                                                                                               |
| `E_INVALID_GRAPH_OP`   | An invalid graph mutation (e.g. a cycle, a self-reference).                                                                                                                                                                                                                                                                   |
| `E_INVALID_TREE`       | An invalid tree structure or operation.                                                                                                                                                                                                                                                                                       |
| `E_NOT_IMPLEMENTED`    | A recognized-but-not-yet-implemented feature.                                                                                                                                                                                                                                                                                 |
| `E_UNSUPPORTED`        | A feature/clause/predicate that isn't supported.                                                                                                                                                                                                                                                                              |
| `E_RESOURCE_EXHAUSTED` | Evaluation hit a resource limit (e.g. a path enumeration exceeded its budget).                                                                                                                                                                                                                                                |
| `E_UNKNOWN_FUNCTION`   | An unknown function/step/symbol referenced in a query.                                                                                                                                                                                                                                                                        |
| `E_FFI`                | A failure crossing the native/wasm FFI boundary.                                                                                                                                                                                                                                                                              |

The `ErrorCode` values are the source of truth for a generated Rust mirror in the `lenke-engine-core` crate, so both languages share the same wire strings.

## License

Apache-2.0
