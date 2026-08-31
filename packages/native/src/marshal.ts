import { ErrorCode, LenkeError } from '@lenke/errors';

/**
 * Shared marshalling guards for the two backends. The FFI boundary is *our* seam
 * — both sides are lenke code — so these are not defenses against a hostile
 * peer; they are contract checks at a border crossing. A length that can't be
 * trusted or an error report that doesn't match the agreed `{code, message}`
 * shape means our own ABI drifted, and we'd rather surface that as a coded
 * {@link ErrorCode.Ffi} fault than let it silently truncate a buffer or hand back
 * a `LenkeError` whose `code` is `undefined`.
 */

/** The shape Rust writes into the last-error slot (mirrors `LenkeError`). */
export type ErrorReport = {
  readonly code: ErrorCode;
  readonly message: string;
  readonly details?: Readonly<Record<string, unknown>> | null;
};

/**
 * Coerce a length the crate wrote into `out_len` to a JS array length, checking
 * it's a value we can actually use. `out_len` is a native `usize`; over bun:ffi
 * it arrives as a `bigint` that can exceed `Number.MAX_SAFE_INTEGER` (where
 * `Number()` starts dropping bits). A non-finite, negative, or unsafe length is
 * a broken result contract — throw rather than read a corrupt range.
 */
export const asByteLength = (raw: number | bigint, op: string): number => {
  const len = typeof raw === 'bigint' ? Number(raw) : raw;

  if (!Number.isSafeInteger(len) || len < 0) {
    throw new LenkeError(
      `lenke: ${op}: native returned an implausible buffer length (${String(raw)})`,
      { code: ErrorCode.Ffi, details: { len: String(raw) } },
    );
  }

  return len;
};

/**
 * Parse and shape-check the JSON last-error report the crate hands back. A report
 * that isn't valid JSON, or that lacks the agreed `{code, message}` strings, is
 * itself an FFI fault: return null so the caller falls back to a generic code
 * instead of surfacing a `LenkeError` whose `code` field is `undefined`.
 */
export const parseErrorReport = (json: string): ErrorReport | null => {
  let parsed: unknown;

  try {
    parsed = JSON.parse(json);
  } catch {
    return null;
  }

  if (
    typeof parsed !== 'object' ||
    parsed === null ||
    typeof (parsed as { code?: unknown }).code !== 'string' ||
    typeof (parsed as { message?: unknown }).message !== 'string'
  ) {
    return null;
  }

  return parsed as ErrorReport;
};

/**
 * Rebuild a {@link LenkeError} from the message an N-API exception carries. The
 * napi addon (`@lenke/node`) throws `lenke: <op>: <message> [E_CODE]`, with the
 * stable wire code in a trailing `[…]`; its adapter runs the message through here
 * to recover the SAME coded error the bun:ffi and wasm backends surface via the
 * last-error channel. A message with no code tail (an adapter-level fault, e.g. a
 * bad handle) passes through as a generic {@link ErrorCode.Ffi} error.
 */
export const errorFromNapi = (message: string | undefined): LenkeError => {
  if (message === undefined) {
    return new LenkeError('lenke: native call failed', { code: ErrorCode.Ffi });
  }

  // The tail is always `[E_UPPER_SNAKE]` from ErrorCode::as_str(); the greedy
  // `.*` (dotall for multi-line messages) binds it to the LAST such tail.
  const tagged = /^(.*) \[(E_[A-Z_]+)\]$/s.exec(message);

  return tagged
    ? new LenkeError(tagged[1], { code: tagged[2] as ErrorCode })
    : new LenkeError(message, { code: ErrorCode.Ffi });
};
