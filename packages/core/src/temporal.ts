/**
 * ISO/IEC 39075 temporal values — `DATE`, `LOCAL TIME`, `LOCAL DATETIME`,
 * `ZONED TIME`, `ZONED DATETIME`, `DURATION`.
 *
 * A byte-for-byte port of the now-removed lenke-core's `temporal` module: the calendar math
 * is Howard Hinnant's civil-from-days algorithm and the ISO-8601 parse/format is
 * hand-rolled, so both engines produce the identical wire string. **Byte-identity
 * is defined by the ISO-8601 string (`format`) and the comparison order**, not by
 * the field layout. JS has no native calendar-date / calendar-duration, so each
 * type is a small branded class (discriminated by `instanceof` + a `kind` tag),
 * with a `toJSON()` that emits the tagged form the JSON codecs round-trip.
 *
 * The ZONED variants carry a numeric UTC offset (`±HH:MM`/`Z`), never a named
 * zone, so they stay dependency-free; the offset is preserved for round-trip.
 */

import { ErrorCode, LenkeError } from '@lenke/errors';

const SECS_PER_DAY = 86_400;

/** Rust integer division truncates toward zero — JS `/` does not. */
const tdiv = (a: number, b: number): number => Math.trunc(a / b);

/** A calendar date, no time/zone: days since 1970-01-01. */
export class LocalDate {
  readonly kind = 'date' as const;
  constructor(readonly days: number) {
    // The argument is an epoch-day count, not calendar fields. `new
    // LocalDate(2026, 1, 15)` is a common mistake (TS flags the excess args, but
    // a JS/`bun`-stripped call runs) that would silently store day 2026 (=
    // 1975-07-20) and drop the month/day. Reject it so it fails loudly; use
    // `LocalDate.of(y, m, d)` or `parseDate('YYYY-MM-DD')` for calendar fields.
    if (arguments.length !== 1) {
      throw new LenkeError(
        `LocalDate(days) takes a single epoch-day count, not calendar fields — ` +
          `use LocalDate.of(year, month, day) or parseDate('YYYY-MM-DD').`,
        { code: ErrorCode.InvalidValue },
      );
    }
  }
  /** Construct from calendar fields, e.g. `LocalDate.of(2026, 1, 15)`. */
  static of(year: number, month: number, day: number): LocalDate {
    return new LocalDate(daysFromCivil(year, month, day));
  }
  /** The ISO-8601 string, e.g. `2020-01-01` — the interop lingua franca. */
  toString(): string {
    return formatDate(this);
  }
  toISOString(): string {
    return formatDate(this);
  }
  toJSON(): { '@date': string } {
    return { '@date': formatDate(this) };
  }
  /** A TC39 `Temporal.PlainDate` (throws if the runtime lacks `Temporal`). */
  toTemporal(): unknown {
    return toTemporalGlobal('PlainDate', formatDate(this));
  }
  /** Parse an ISO date string. */
  static parse(s: string): LocalDate {
    return parseDate(s);
  }
  /** Take the calendar date of a native `Date`, in an explicit zone (default UTC). */
  static fromJSDate(d: Date, opts?: { zone?: 'utc' | 'local' }): LocalDate {
    return new LocalDate(Math.floor(wallMs(d, opts?.zone ?? 'utc') / 86_400_000));
  }
}

/** A zone-less time of day (ISO "LOCAL TIME"): secs since midnight + nanos. */
export class LocalTime {
  readonly kind = 'localtime' as const;
  constructor(
    readonly secs: number, // 0..86_400
    readonly nanos: number,
  ) {}
  toString(): string {
    return formatTime(this);
  }
  toISOString(): string {
    return formatTime(this);
  }
  toJSON(): { '@localtime': string } {
    return { '@localtime': formatTime(this) };
  }
  /** A TC39 `Temporal.PlainTime` (throws if the runtime lacks `Temporal`). */
  toTemporal(): unknown {
    return toTemporalGlobal('PlainTime', formatTime(this));
  }
  static parse(s: string): LocalTime {
    return parseLocalTime(s);
  }
}

/** A zone-less datetime (ISO "LOCAL DATETIME"): secs since the epoch + nanos. */
export class LocalDateTime {
  readonly kind = 'datetime' as const;
  constructor(
    readonly secs: number,
    readonly nanos: number,
  ) {}
  toString(): string {
    return formatDateTime(this);
  }
  toISOString(): string {
    return formatDateTime(this);
  }
  toJSON(): { '@datetime': string } {
    return { '@datetime': formatDateTime(this) };
  }
  /** A TC39 `Temporal.PlainDateTime` (throws if the runtime lacks `Temporal`). */
  toTemporal(): unknown {
    return toTemporalGlobal('PlainDateTime', formatDateTime(this));
  }
  static parse(s: string): LocalDateTime {
    return parseDateTime(s);
  }
  /**
   * Take the wall-clock datetime of a native `Date`, in an explicit zone
   * (default UTC). Explicit because a `Date` is a zoned instant and our types
   * are zone-less — silently guessing a zone would corrupt the stored value.
   */
  static fromJSDate(d: Date, opts?: { zone?: 'utc' | 'local' }): LocalDateTime {
    const ms = wallMs(d, opts?.zone ?? 'utc');
    const secs = Math.floor(ms / 1000);

    return new LocalDateTime(secs, (ms - secs * 1000) * 1_000_000);
  }
}

/**
 * A datetime with a UTC offset (ISO "ZONED DATETIME" / "TIMESTAMP WITH TIME
 * ZONE"): the UTC instant + the offset it was written in (whole minutes; `Z`=0).
 * The offset is preserved for round-trip and participates in identity/ordering
 * (instant first, offset second). ISO carries a numeric offset, never a named zone.
 */
export class ZonedDateTime {
  readonly kind = 'zoned_datetime' as const;
  constructor(
    readonly secs: number, // UTC instant seconds since the epoch
    readonly nanos: number,
    readonly offset: number, // whole minutes
  ) {}
  toString(): string {
    return formatZonedDateTime(this);
  }
  toISOString(): string {
    return formatZonedDateTime(this);
  }
  toJSON(): { '@zoned_datetime': string } {
    return { '@zoned_datetime': formatZonedDateTime(this) };
  }
  static parse(s: string): ZonedDateTime {
    return parseZonedDateTime(s);
  }
}

/** A time of day with a UTC offset (ISO "ZONED TIME" / "TIME WITH TIME ZONE"). */
export class ZonedTime {
  readonly kind = 'zoned_time' as const;
  constructor(
    readonly secs: number, // UTC seconds-of-day (0..86_400)
    readonly nanos: number,
    readonly offset: number, // whole minutes
  ) {}
  toString(): string {
    return formatZonedTime(this);
  }
  toISOString(): string {
    return formatZonedTime(this);
  }
  toJSON(): { '@zoned_time': string } {
    return { '@zoned_time': formatZonedTime(this) };
  }
  static parse(s: string): ZonedTime {
    return parseZonedTime(s);
  }
}

/** An ISO-8601 calendar duration (months/days kept separate from seconds). */
export class Duration {
  readonly kind = 'duration' as const;
  constructor(
    readonly months: number,
    readonly days: number,
    readonly secs: number,
    readonly nanos: number,
  ) {}
  /** The ISO-8601 duration string, e.g. `P14M3DT4H5M6S`. Round-trips with
   * `Temporal.Duration` / Luxon `Duration.fromISO` — the interop path for
   * durations (JS has no native duration type). */
  toString(): string {
    return formatDuration(this);
  }
  toISOString(): string {
    return formatDuration(this);
  }
  toJSON(): { '@duration': string } {
    return { '@duration': formatDuration(this) };
  }
  /** A TC39 `Temporal.Duration` (throws if the runtime lacks `Temporal`). */
  toTemporal(): unknown {
    return toTemporalGlobal('Duration', formatDuration(this));
  }
  static parse(s: string): Duration {
    return parseDuration(s);
  }
}

export type Temporal = LocalDate | LocalTime | LocalDateTime | ZonedTime | ZonedDateTime | Duration;

/**
 * A host clock: a nullary function the host wires into a query runner to supply
 * the reserved `$__now` param that the ISO now-functions (`current_date`,
 * `current_timestamp`, `local_timestamp`) desugar to. Return a `LocalDateTime`
 * for a timestamp (or a `LocalDate` for a date-only clock).
 *
 * The engine NEVER calls a clock itself — it stays a pure function of (graph,
 * params). The clock runs in the HOST, once per query, and its result is bound
 * as a value, so the impurity is explicit and opt-in: wiring a clock is how you
 * declare "this runner reads wall time." A query that passes an explicit
 * `$__now` overrides the clock (deterministic, for tests/repro), and with no
 * clock and no `$__now` the now-functions read as null (the honest default).
 */
export type Clock = () => LocalDate | LocalDateTime;

export const isTemporal = (v: unknown): v is Temporal =>
  v instanceof LocalDate ||
  v instanceof LocalTime ||
  v instanceof LocalDateTime ||
  v instanceof ZonedTime ||
  v instanceof ZonedDateTime ||
  v instanceof Duration;

// --- civil calendar (Hinnant) ------------------------------------------------

/** Days since 1970-01-01 for a proleptic-Gregorian (y, m, d). `m` in 1..=12. */
export function daysFromCivil(y: number, m: number, d: number): number {
  const yy = m <= 2 ? y - 1 : y;
  const era = tdiv(yy >= 0 ? yy : yy - 399, 400);
  const yoe = yy - era * 400;
  const doy = tdiv(153 * (m > 2 ? m - 3 : m + 9) + 2, 5) + d - 1;
  const doe = yoe * 365 + tdiv(yoe, 4) - tdiv(yoe, 100) + doy;

  return era * 146_097 + doe - 719_468;
}

/** Proleptic-Gregorian [y, m, d] for days since 1970-01-01. */
export function civilFromDays(z: number): [number, number, number] {
  const zz = z + 719_468;
  const era = tdiv(zz >= 0 ? zz : zz - 146_096, 146_097);
  const doe = zz - era * 146_097;
  const yoe = tdiv(doe - tdiv(doe, 1460) + tdiv(doe, 36_524) - tdiv(doe, 146_096), 365);
  const y = yoe + era * 400;
  const doy = doe - (365 * yoe + tdiv(yoe, 4) - tdiv(yoe, 100));
  const mp = tdiv(5 * doy + 2, 153);
  const d = doy - tdiv(153 * mp + 2, 5) + 1;
  const m = mp < 10 ? mp + 3 : mp - 9;

  return [m <= 2 ? y + 1 : y, m, d];
}

// --- small helpers -----------------------------------------------------------

const pad = (n: number, width: number): string => String(n).padStart(width, '0');

// Zero-pad a year to 4 digits AFTER the sign — matching Rust `format!("{y:04}")`.
// A negative (BCE) year (reachable only via temporal arithmetic; both engines
// reject a leading `-` on input) must render as `-0009`, not `pad`'s `00-9` (which
// pads across the minus). Only the year is ever negative in a formatted temporal.
const padYear = (y: number): string =>
  y < 0 ? `-${String(-y).padStart(4, '0')}` : String(y).padStart(4, '0');

const parseIntStrict = (s: string): number => {
  if (!/^-?\d+$/.test(s)) {
    throw new LenkeError(`invalid integer '${s}'`, { code: ErrorCode.InvalidValue });
  }

  return Number(s);
};

/** A fractional-second string (up to 9 digits) → nanoseconds. */
const parseFrac = (frac: string | undefined): number => {
  if (frac === undefined) {
    return 0;
  }

  if (frac.length === 0 || frac.length > 9 || !/^\d+$/.test(frac)) {
    throw new LenkeError(`bad fractional seconds '.${frac}'`, { code: ErrorCode.InvalidValue });
  }

  return Number(frac.padEnd(9, '0'));
};

/** Render `nanos` as `.fraction` (trailing zeros trimmed), or '' when zero. */
const fmtFrac = (nanos: number): string => {
  if (nanos === 0) {
    return '';
  }

  return `.${pad(nanos, 9).replace(/0+$/, '')}`;
};

/** Render a UTC offset (whole minutes) as `Z` (=0) or `±HH:MM`. */
const fmtOffset = (offset: number): string => {
  if (offset === 0) {
    return 'Z';
  }

  const sign = offset < 0 ? '-' : '+';
  const a = Math.abs(offset);

  return `${sign}${pad(Math.trunc(a / 60), 2)}:${pad(a % 60, 2)}`;
};

/**
 * Split a trailing UTC offset (`Z` / `±HH:MM` / `±HHMM`) off `s`, returning
 * `[part-before, offset-minutes]`; throws if none (a ZONED value requires one).
 * Only the tail is inspected, so a date's `-` separators are never mistaken for
 * the offset sign.
 */
const splitOffset = (s: string): [string, number] => {
  if (s.endsWith('Z')) {
    return [s.slice(0, -1), 0];
  }

  const n = s.length;

  for (const [width, colon] of [
    [6, true],
    [5, false],
  ] as const) {
    if (n < width) {
      continue;
    }

    const start = n - width;
    const sign = s[start];

    if (sign !== '+' && sign !== '-') {
      continue;
    }

    const hh = s.slice(start + 1, start + 3);
    let mm: string | undefined;

    if (colon) {
      mm = s[start + 3] === ':' ? s.slice(start + 4, start + 6) : undefined;
    } else {
      mm = s.slice(start + 3, start + 5);
    }

    if (mm !== undefined && /^\d\d$/.test(hh) && /^\d\d$/.test(mm)) {
      const h = Number(hh);
      const m = Number(mm);

      if (h <= 23 && m < 60) {
        const mag = h * 60 + m;

        return [s.slice(0, start), sign === '-' ? -mag : mag];
      }
    }
  }

  throw new LenkeError(`missing/invalid time-zone offset in '${s}'`, {
    code: ErrorCode.InvalidValue,
  });
};

/** Parse `HH:MM:SS[.fraction]` into [seconds-of-day, nanos]. */
const parseTime = (s: string): [number, number] => {
  const dot = s.indexOf('.');
  const hms = dot === -1 ? s : s.slice(0, dot);
  const frac = dot === -1 ? undefined : s.slice(dot + 1);
  const parts = hms.split(':');

  if (parts.length !== 3) {
    throw new LenkeError(`bad time '${s}'`, { code: ErrorCode.InvalidValue });
  }

  const [h, m, sec] = parts.map(parseIntStrict);

  if (h < 0 || h >= 24 || m < 0 || m >= 60 || sec < 0 || sec >= 60) {
    throw new LenkeError(`time out of range '${s}'`, { code: ErrorCode.InvalidValue });
  }

  return [h * 3600 + m * 60 + sec, parseFrac(frac)];
};

// --- Date --------------------------------------------------------------------

export function parseDate(s: string): LocalDate {
  const parts = s.split('-');

  if (parts.length !== 3) {
    throw new LenkeError(`bad date '${s}'`, { code: ErrorCode.InvalidValue });
  }

  const [y, m, d] = parts.map(parseIntStrict);

  if (m < 1 || m > 12 || d < 1 || d > 31) {
    throw new LenkeError(`date out of range '${s}'`, { code: ErrorCode.InvalidValue });
  }

  return new LocalDate(daysFromCivil(y, m, d));
}

export function formatDate(dt: LocalDate): string {
  const [y, m, d] = civilFromDays(dt.days);

  return `${padYear(y)}-${pad(m, 2)}-${pad(d, 2)}`;
}

// --- Time --------------------------------------------------------------------

export function parseLocalTime(s: string): LocalTime {
  const [tod, nanos] = parseTime(s);

  return new LocalTime(tod, nanos);
}

export function formatTime(t: LocalTime): string {
  const h = tdiv(t.secs, 3600);
  const m = tdiv(t.secs % 3600, 60);
  const s = t.secs % 60;

  return `${pad(h, 2)}:${pad(m, 2)}:${pad(s, 2)}${fmtFrac(t.nanos)}`;
}

// --- Zoned -------------------------------------------------------------------

export function parseZonedDateTime(s: string): ZonedDateTime {
  const [dtStr, offset] = splitOffset(s);
  const local = parseDateTime(dtStr);

  return new ZonedDateTime(local.secs - offset * 60, local.nanos, offset);
}

export function formatZonedDateTime(z: ZonedDateTime): string {
  const local = new LocalDateTime(z.secs + z.offset * 60, z.nanos);

  return `${formatDateTime(local)}${fmtOffset(z.offset)}`;
}

export function parseZonedTime(s: string): ZonedTime {
  const [tStr, offset] = splitOffset(s);
  const [tod, nanos] = parseTime(tStr);
  const utc = (((tod - offset * 60) % SECS_PER_DAY) + SECS_PER_DAY) % SECS_PER_DAY;

  return new ZonedTime(utc, nanos, offset);
}

export function formatZonedTime(z: ZonedTime): string {
  const local = (((z.secs + z.offset * 60) % SECS_PER_DAY) + SECS_PER_DAY) % SECS_PER_DAY;
  const h = tdiv(local, 3600);
  const m = tdiv(local % 3600, 60);
  const s = local % 60;

  return `${pad(h, 2)}:${pad(m, 2)}:${pad(s, 2)}${fmtFrac(z.nanos)}${fmtOffset(z.offset)}`;
}

// --- DateTime ----------------------------------------------------------------

export function parseDateTime(s: string): LocalDateTime {
  const sep = s.search(/[T ]/);

  if (sep === -1) {
    throw new LenkeError(`datetime missing time part '${s}'`, { code: ErrorCode.InvalidValue });
  }

  const date = parseDate(s.slice(0, sep));
  const [tod, nanos] = parseTime(s.slice(sep + 1));

  return new LocalDateTime(date.days * SECS_PER_DAY + tod, nanos);
}

export function formatDateTime(dt: LocalDateTime): string {
  // Floor-divide so a pre-epoch time-of-day stays in [0, 86400).
  const days = Math.floor(dt.secs / SECS_PER_DAY);
  const tod = ((dt.secs % SECS_PER_DAY) + SECS_PER_DAY) % SECS_PER_DAY;
  const [y, mo, d] = civilFromDays(days);
  const h = tdiv(tod, 3600);
  const m = tdiv(tod % 3600, 60);
  const s = tod % 60;

  return `${padYear(y)}-${pad(mo, 2)}-${pad(d, 2)}T${pad(h, 2)}:${pad(m, 2)}:${pad(s, 2)}${fmtFrac(dt.nanos)}`;
}

// --- Duration ----------------------------------------------------------------

export function parseDuration(s: string): Duration {
  if (!s.startsWith('P')) {
    throw new LenkeError(`duration must start with 'P': '${s}'`, { code: ErrorCode.InvalidValue });
  }

  const rest = s.slice(1);
  const tIdx = rest.indexOf('T');
  const datePart = tIdx === -1 ? rest : rest.slice(0, tIdx);
  const timePart = tIdx === -1 ? undefined : rest.slice(tIdx + 1);
  let months = 0;
  let days = 0;
  let num = '';
  const takeNum = (designator: string): number => {
    if (num === '') {
      throw new LenkeError(`missing number before '${designator}' in '${s}'`, {
        code: ErrorCode.InvalidValue,
      });
    }

    const v = parseIntStrict(num);
    num = '';

    return v;
  };

  for (const c of datePart) {
    if (/[0-9-]/.test(c)) {
      num += c;
    } else if (c === 'Y') {
      months += takeNum('Y') * 12;
    } else if (c === 'M') {
      months += takeNum('M');
    } else if (c === 'W') {
      days += takeNum('W') * 7;
    } else if (c === 'D') {
      days += takeNum('D');
    } else {
      throw new LenkeError(`bad duration date field '${c}' in '${s}'`, {
        code: ErrorCode.InvalidValue,
      });
    }
  }

  if (num !== '') {
    throw new LenkeError(`dangling number in duration '${s}'`, { code: ErrorCode.InvalidValue });
  }

  let secs = 0;
  let nanos = 0;

  if (timePart !== undefined) {
    for (const c of timePart) {
      if (/[0-9.-]/.test(c)) {
        num += c;
      } else if (c === 'H') {
        secs += takeNum('H') * 3600;
      } else if (c === 'M') {
        secs += takeNum('M') * 60;
      } else if (c === 'S') {
        const dot = num.indexOf('.');
        const whole = dot === -1 ? num : num.slice(0, dot);
        const frac = dot === -1 ? undefined : num.slice(dot + 1);

        if (whole === '') {
          throw new LenkeError(`missing number before 'S' in '${s}'`, {
            code: ErrorCode.InvalidValue,
          });
        }

        secs += parseIntStrict(whole);
        nanos = parseFrac(frac);
        num = '';
      } else {
        throw new LenkeError(`bad duration time field '${c}' in '${s}'`, {
          code: ErrorCode.InvalidValue,
        });
      }
    }

    if (num !== '') {
      throw new LenkeError(`dangling number in duration '${s}'`, { code: ErrorCode.InvalidValue });
    }
  }

  const d = new Duration(months, days, secs, nanos);

  // A component at/beyond 2^53 isn't a JS-safe integer — f64 can't hold it exactly,
  // and native rejects it too (Duration::representable) — so reject rather than
  // silently round, keeping the engines byte-identical.
  if (!durationRepresentable(d)) {
    throw new LenkeError(`duration component is not representable as float64: '${s}'`, {
      code: ErrorCode.InvalidValue,
    });
  }

  return d;
}

export function formatDuration(d: Duration): string {
  let out = 'P';

  if (d.months !== 0) {
    out += `${d.months}M`;
  }

  if (d.days !== 0) {
    out += `${d.days}D`;
  }

  if (d.secs !== 0 || d.nanos !== 0) {
    out += `T${d.secs}${fmtFrac(d.nanos)}S`;
  }

  return out === 'P' ? 'PT0S' : out;
}

// --- Temporal (kind-agnostic) helpers ----------------------------------------

type TemporalTag = 'date' | 'localtime' | 'datetime' | 'zoned_time' | 'zoned_datetime' | 'duration';

export const temporalTag = (t: Temporal): TemporalTag => t.kind;

/**
 * The single source of truth for each temporal kind: its wire tag, its GraphSON
 * `@type`, and its format/parse pair. Every temporal type is one row here — the
 * dispatchers below derive from it, so adding a type is one entry rather than
 * four parallel `instanceof`/tag ladders that must be kept in lockstep.
 */
type TemporalSpec = {
  readonly tag: TemporalTag;
  readonly graphson: string;
  readonly format: (t: Temporal) => string;
  readonly parse: (s: string) => Temporal;
};

// Bind one row: the generic ties `format`/`parse` to a single concrete class, so
// a mismatched pair (e.g. `formatDate` with `parseLocalTime`) is a type error.
// The widening casts are sound because the dispatchers only ever look a row up by
// that same class's `kind` discriminant.
const specRow = <T extends Temporal>(
  tag: TemporalTag,
  graphson: string,
  format: (t: T) => string,
  parse: (s: string) => T,
): TemporalSpec => ({ tag, graphson, format: format as (t: Temporal) => string, parse });

const TEMPORAL_SPEC: Record<TemporalTag, TemporalSpec> = {
  date: specRow('date', 'gx:LocalDate', formatDate, parseDate),
  localtime: specRow('localtime', 'gx:LocalTime', formatTime, parseLocalTime),
  datetime: specRow('datetime', 'gx:LocalDateTime', formatDateTime, parseDateTime),
  zoned_time: specRow('zoned_time', 'gx:OffsetTime', formatZonedTime, parseZonedTime),
  zoned_datetime: specRow(
    'zoned_datetime',
    'gx:OffsetDateTime',
    formatZonedDateTime,
    parseZonedDateTime,
  ),
  duration: specRow('duration', 'gx:Duration', formatDuration, parseDuration),
};

// Reverse index for `graphsonTag`. A `Map` (not a plain object) so an unknown
// `@type` like `'toString'` misses cleanly instead of hitting a prototype method.
const GRAPHSON_TO_TAG = new Map<string, TemporalTag>(
  Object.values(TEMPORAL_SPEC).map((s) => [s.graphson, s.tag]),
);

/**
 * The valid tagged-temporal JSON keys — the `@`-prefixed wire form every temporal
 * `toJSON()` emits and `fromTaggedJson` accepts (`@date`, `@localtime`, `@datetime`,
 * `@zoned_time`, `@zoned_datetime`, `@duration`). Derived from TEMPORAL_SPEC so it
 * can never drift from the kinds themselves — the single authority for "is this
 * object a tagged temporal?", shared by the host-side param gate in `@lenke/native`.
 */
export const TEMPORAL_TAG_KEYS: ReadonlySet<string> = new Set(
  Object.values(TEMPORAL_SPEC).map((s) => `@${s.tag}`),
);

export function temporalFormat(t: Temporal): string {
  return TEMPORAL_SPEC[t.kind].format(t);
}

/** Build from a kind tag + ISO string (the codec decode path); throws on error. */
export function temporalParse(tag: string, s: string): Temporal {
  if (!Object.hasOwn(TEMPORAL_SPEC, tag)) {
    throw new LenkeError(`unknown temporal kind '${tag}'`, { code: ErrorCode.InvalidValue });
  }

  return TEMPORAL_SPEC[tag as TemporalTag].parse(s);
}

/** GraphSON v3 `@type` name (TinkerPop extended types). */
export function graphsonType(t: Temporal): string {
  return TEMPORAL_SPEC[t.kind].graphson;
}

/** GraphSON `@type` → kind tag, or `undefined` if not a temporal type. */
export function graphsonTag(ty: string): TemporalTag | undefined {
  return GRAPHSON_TO_TAG.get(ty);
}

/** Recognize a single-key tagged temporal object `{"@date":"…"}`, else null. */
export function fromTaggedJson(v: unknown): Temporal | null {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) {
    return null;
  }

  const keys = Object.keys(v);

  if (keys.length !== 1) {
    return null;
  }

  const [key] = keys;

  if (!TEMPORAL_TAG_KEYS.has(key)) {
    return null;
  }

  const s = (v as Record<string, unknown>)[key];

  if (typeof s !== 'string') {
    return null;
  }

  return temporalParse(key.slice(1), s);
}

// --- interop (TC39 Temporal / native Date bridge) ----------------------------

/** Wall-clock ms for a native `Date` in an explicit zone. */
const wallMs = (d: Date, zone: 'utc' | 'local'): number =>
  zone === 'local' ? d.getTime() - d.getTimezoneOffset() * 60_000 : d.getTime();

/** Build a TC39 `Temporal.<kind>` from an ISO string, or throw if unavailable. */
const toTemporalGlobal = (
  kind: 'PlainDate' | 'PlainTime' | 'PlainDateTime' | 'Duration',
  iso: string,
): unknown => {
  const T = (globalThis as unknown as { Temporal?: Record<string, { from(s: string): unknown }> })
    .Temporal;

  if (T === undefined) {
    throw new LenkeError(
      'TC39 Temporal is not available in this runtime — use .toISOString() with your date library',
      { code: ErrorCode.Unsupported },
    );
  }

  return T[kind].from(iso);
};

/** `[object Temporal.PlainDate]` → `Temporal.PlainDate`, else `''`. */
const brandOf = (v: unknown): string => {
  const tag = Object.prototype.toString.call(v);

  return tag.startsWith('[object Temporal.') ? tag.slice(8, -1) : '';
};

/** Strip a Temporal calendar annotation (`[u-ca=iso8601]`) from an ISO string. */
const stripAnnotation = (s: string): string => s.replace(/\[[^\]]*\]/g, '');

/**
 * Coerce a foreign temporal value into lenke's model at the value boundary: a
 * lenke instance passes through; a TC39 `Temporal.PlainDate`/`PlainDateTime`/
 * `Duration` converts via its ISO string. Returns `null` for anything else (the
 * caller decides whether that's an error). A native `Date` deliberately does NOT
 * coerce — it's a zoned instant; use `LocalDateTime.fromJSDate(d, { zone })`.
 */
export function coerceTemporal(v: unknown): Temporal | null {
  if (isTemporal(v)) {
    return v;
  }

  switch (brandOf(v)) {
    case 'Temporal.PlainDate':
      return parseDate(stripAnnotation(String(v)));
    case 'Temporal.PlainTime':
      return parseLocalTime(stripAnnotation(String(v)));
    case 'Temporal.PlainDateTime':
      return parseDateTime(stripAnnotation(String(v)));
    case 'Temporal.Duration':
      return parseDuration(stripAnnotation(String(v)));
    default:
      return null;
  }
}

const kindRank = (t: Temporal): number => {
  if (t instanceof LocalDate) {
    return 0;
  }

  if (t instanceof LocalTime) {
    return 1;
  }

  if (t instanceof LocalDateTime) {
    return 2;
  }

  if (t instanceof ZonedTime) {
    return 3;
  }

  if (t instanceof ZonedDateTime) {
    return 4;
  }

  return 5;
};

const cmpNum = (a: number, b: number): number => {
  if (a < b) {
    return -1;
  }

  if (a > b) {
    return 1;
  }

  return 0;
};

const cmpTuple = (a: readonly number[], b: readonly number[]): number => {
  for (let i = 0; i < a.length; i++) {
    const c = cmpNum(a[i], b[i]);

    if (c !== 0) {
      return c;
    }
  }

  return 0;
};

/**
 * Deterministic TOTAL order over all temporals (for `ORDER BY`/min/max): by kind,
 * then chronologically within date/datetime, lexicographically within duration.
 */
export function temporalCmpTotal(a: Temporal, b: Temporal): number {
  if (a instanceof LocalDate && b instanceof LocalDate) {
    return cmpNum(a.days, b.days);
  }

  if (a instanceof LocalTime && b instanceof LocalTime) {
    return cmpTuple([a.secs, a.nanos], [b.secs, b.nanos]);
  }

  if (a instanceof LocalDateTime && b instanceof LocalDateTime) {
    return cmpTuple([a.secs, a.nanos], [b.secs, b.nanos]);
  }

  if (a instanceof ZonedTime && b instanceof ZonedTime) {
    return cmpTuple([a.secs, a.nanos, a.offset], [b.secs, b.nanos, b.offset]);
  }

  if (a instanceof ZonedDateTime && b instanceof ZonedDateTime) {
    return cmpTuple([a.secs, a.nanos, a.offset], [b.secs, b.nanos, b.offset]);
  }

  if (a instanceof Duration && b instanceof Duration) {
    return cmpTuple([a.months, a.days, a.secs, a.nanos], [b.months, b.days, b.secs, b.nanos]);
  }

  return cmpNum(kindRank(a), kindRank(b));
}

/**
 * Relational order for `< > <= >=`: date/datetime (same kind) are instants;
 * durations and cross-kind pairs are UNKNOWN (`null`).
 */
export function temporalRelCmp(a: Temporal, b: Temporal): number | null {
  if (a instanceof LocalDate && b instanceof LocalDate) {
    return cmpNum(a.days, b.days);
  }

  if (a instanceof LocalTime && b instanceof LocalTime) {
    return cmpTuple([a.secs, a.nanos], [b.secs, b.nanos]);
  }

  if (a instanceof LocalDateTime && b instanceof LocalDateTime) {
    return cmpTuple([a.secs, a.nanos], [b.secs, b.nanos]);
  }

  if (a instanceof ZonedTime && b instanceof ZonedTime) {
    return cmpTuple([a.secs, a.nanos, a.offset], [b.secs, b.nanos, b.offset]);
  }

  if (a instanceof ZonedDateTime && b instanceof ZonedDateTime) {
    return cmpTuple([a.secs, a.nanos, a.offset], [b.secs, b.nanos, b.offset]);
  }

  // Two DURATIONS are only PARTIALLY ordered (W3C XML Schema) — incomparable → null.
  if (a instanceof Duration && b instanceof Duration) {
    return durationPartialCmp(a, b);
  }

  return null;
}

// --- calendar arithmetic -----------------------------------------------------

const isLeap = (y: number): boolean => (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0;

/** Days in month `m` (1..=12) of proleptic-Gregorian year `y`. */
const daysInMonth = (y: number, m: number): number => {
  if (m === 2) {
    return isLeap(y) ? 29 : 28;
  }

  return m === 4 || m === 6 || m === 9 || m === 11 ? 30 : 31;
};

// A LocalDate is `i32` days from 1970 in the native engine (≈±5.88M years). Date
// arithmetic that lands outside this range yields null (non-representable → null,
// matching the numeric-overflow policy) rather than a value the engines disagree
// on — byte-identical with native's `Date::add_calendar`.
const DATE_DAYS_MIN = -2_147_483_648;
const DATE_DAYS_MAX = 2_147_483_647;
const inDateRange = (days: number): boolean => days >= DATE_DAYS_MIN && days <= DATE_DAYS_MAX;

/**
 * Add `months` (calendar), CLAMPING the day to the new month's length
 * (`Jan 31 + 1 month → Feb 28/29`), then `extraDays` as plain days. Returns null
 * when the result falls outside the representable date range.
 */
const addCalendar = (date: LocalDate, months: number, extraDays: number): LocalDate | null => {
  const [y, m, d] = civilFromDays(date.days);
  const total = y * 12 + (m - 1) + months;
  const ny = Math.floor(total / 12);
  const nm = (((total % 12) + 12) % 12) + 1;
  const nd = Math.min(d, daysInMonth(ny, nm));
  const days = daysFromCivil(ny, nm, nd) + extraDays;

  return inDateRange(days) ? new LocalDate(days) : null;
};

/** Negate the whole span, keeping `nanos` in `[0, 1e9)`. */
const negateDuration = (d: Duration): Duration => {
  const secs = d.nanos === 0 ? -d.secs : -d.secs - 1;
  const nanos = d.nanos === 0 ? 0 : 1_000_000_000 - d.nanos;

  return new Duration(-d.months, -d.days, secs, nanos);
};

/**
 * A duration is representable only when every component is a JS **safe integer**
 * (|x| ≤ 2^53−1); beyond that f64 rounds and native (i64) would keep an exact value
 * or wrap, so the two engines would diverge. Such a duration is non-representable →
 * `null` (arithmetic) / rejected (parse), matching the native `Duration::representable`
 * contract and the numeric-overflow → null policy.
 */
const durationRepresentable = (d: Duration): boolean =>
  Number.isSafeInteger(d.months) && Number.isSafeInteger(d.days) && Number.isSafeInteger(d.secs);

/** A duration whose sum/scale overflows the safe-integer range is a value error (a
 *  real duration we can't represent), not a silent null — fail loud, byte-identical to
 *  the native engine (like division by zero). The engine collapses every runtime fault
 *  to `E_INVALID_VALUE` at the FFI boundary, so this matches that code, not a distinct
 *  `E_DATA_EXCEPTION`. */
const durationOverflow = (): never => {
  throw new LenkeError(
    'duration overflow: a component exceeds the representable (float64-safe-integer) range',
    { code: ErrorCode.InvalidValue },
  );
};

/** Instant ± duration whose result leaves the representable date range (a date is
 *  `i32` days, ≈±5.88M years) is a value error — the target date is real but
 *  unstorable — not a silent null. Fail loud with `E_INVALID_VALUE`, byte-identical to
 *  native's `FAULT_DATE_OVERFLOW` (which the FFI surfaces as `E_INVALID_VALUE`;
 *  supersedes the old D4 → null). */
const dateOverflow = (): never => {
  throw new LenkeError('date overflow: arithmetic result is outside the representable date range', {
    code: ErrorCode.InvalidValue,
  });
};

/** Anchor a duration to an instant (`addDurationTo`), raising `dateOverflow` when
 *  the result is out of the representable date range (`addDurationTo` → null). A
 *  bare time never overflows (it wraps), so this only throws for date-carrying
 *  instants. */
const anchorOrThrow = (t: Temporal, d: Duration): Temporal => addDurationTo(t, d) ?? dateOverflow();

/** Component-wise sum of two nominal durations (nanos carry into secs). Throws
 *  `DataException` when a component leaves the safe-integer range. */
const addDurations = (a: Duration, b: Duration): Duration => {
  let secs = a.secs + b.secs;
  let nanos = a.nanos + b.nanos;

  if (nanos >= 1_000_000_000) {
    nanos -= 1_000_000_000;
    secs += 1;
  }

  const sum = new Duration(a.months + b.months, a.days + b.days, secs, nanos);

  return durationRepresentable(sum) ? sum : durationOverflow();
};

/** Scale every component by an integer factor (nanos carry into secs). Throws
 *  `DataException` when a component leaves the safe-integer range. */
const scaleDuration = (d: Duration, n: number): Duration => {
  const totalNanos = d.nanos * n;
  const carry = Math.floor(totalNanos / 1_000_000_000);
  const nanos = ((totalNanos % 1_000_000_000) + 1_000_000_000) % 1_000_000_000;

  const scaled = new Duration(d.months * n, d.days * n, d.secs * n + carry, nanos);

  return durationRepresentable(scaled) ? scaled : durationOverflow();
};

/** `t + duration` for a date/datetime (months clamped, then days, then time). */
const addDurationTo = (t: Temporal, d: Duration): Temporal | null => {
  if (t instanceof LocalDate) {
    return addCalendar(t, d.months, d.days);
  }

  // A bare time has no calendar part; the duration's time component wraps within
  // the 24h day (months/days are ignored).
  if (t instanceof LocalTime) {
    const carryNanos = t.nanos + d.nanos;
    const secs = t.secs + d.secs + Math.floor(carryNanos / 1_000_000_000);
    const wrapped = ((secs % 86_400) + 86_400) % 86_400;
    const nanos = ((carryNanos % 1_000_000_000) + 1_000_000_000) % 1_000_000_000;

    return new LocalTime(wrapped, nanos);
  }

  if (t instanceof LocalDateTime) {
    const days0 = Math.floor(t.secs / 86_400);
    const tod = ((t.secs % 86_400) + 86_400) % 86_400;

    if (!inDateRange(days0)) {
      return null;
    }

    const date = addCalendar(new LocalDate(days0), d.months, d.days);

    if (!date) {
      return null;
    }

    let secs = date.days * 86_400 + tod + d.secs;
    let nanos = t.nanos + d.nanos;

    if (nanos >= 1_000_000_000) {
      nanos -= 1_000_000_000;
      secs += 1;
    }

    return new LocalDateTime(secs, nanos);
  }

  // Zoned ± duration: apply to the LOCAL wall clock (calendar-correct in its own
  // zone), re-anchor to the UTC instant, keep the offset.
  if (t instanceof ZonedDateTime) {
    const shifted = addDurationTo(new LocalDateTime(t.secs + t.offset * 60, t.nanos), d);

    return shifted instanceof LocalDateTime
      ? new ZonedDateTime(shifted.secs - t.offset * 60, shifted.nanos, t.offset)
      : null;
  }

  if (t instanceof ZonedTime) {
    const localSecs = (((t.secs + t.offset * 60) % 86_400) + 86_400) % 86_400;
    const shifted = addDurationTo(new LocalTime(localSecs, t.nanos), d);

    if (shifted instanceof LocalTime) {
      const utc = (((shifted.secs - t.offset * 60) % 86_400) + 86_400) % 86_400;

      return new ZonedTime(utc, shifted.nanos, t.offset);
    }

    return null;
  }

  return null;
};

// The four reference dateTimes W3C XML Schema Part 2 §3.2.6.2 fixes for the duration
// order — a non-leap and a leap February, and months of 31 and 30 days, enough to expose
// any month-length ambiguity between two durations.
const DURATION_REFS: LocalDateTime[] = (
  [
    [1696, 9, 1],
    [1697, 2, 1],
    [1903, 3, 1],
    [1903, 7, 1],
  ] as const
).map(([y, m, d]) => new LocalDateTime(daysFromCivil(y, m, d) * SECS_PER_DAY, 0));

/**
 * The 3-valued PREDICATE order on two durations, per W3C XML Schema §3.2.6.2: `a < b` iff
 * `s + a < s + b` for EACH of the four reference dateTimes; if the four do not all agree,
 * the pair is INDETERMINATE (`null`) — durations are only partially ordered because a
 * month is 28-31 days (so `P1M` vs `P30D` is null, while `P1D` vs `P2D` orders). `null`
 * also on a date overflow. The TOTAL order for `ORDER BY` / min / max is separate.
 */
const durationPartialCmp = (a: Duration, b: Duration): number | null => {
  let acc: number | null = null;

  for (const s of DURATION_REFS) {
    const sa = addDurationTo(s, a);
    const sb = addDurationTo(s, b);

    if (sa === null || sb === null) {
      return null; // out of representable range → cannot determine
    }

    const ord = temporalRelCmp(sa, sb);

    if (acc === null) {
      acc = ord;
    } else if (acc !== ord) {
      return null; // references disagree → indeterminate
    }
  }

  return acc;
};

/**
 * `duration_between(a, b)` = the exact span b − a (a measurement, so fixed units
 * only): whole days for two dates, secs+nanos for two datetimes; else null.
 */
export function durationBetween(a: Temporal, b: Temporal): Temporal | null {
  if (a instanceof LocalDate && b instanceof LocalDate) {
    return new Duration(0, b.days - a.days, 0, 0);
  }

  if (a instanceof LocalDateTime && b instanceof LocalDateTime) {
    let secs = b.secs - a.secs;
    let nanos = b.nanos - a.nanos;

    if (nanos < 0) {
      nanos += 1_000_000_000;
      secs -= 1;
    }

    return new Duration(0, 0, secs, nanos);
  }

  return null;
}

/**
 * Temporal `+`/`-`/`*` when either operand is temporal (mirrors the Rust
 * `temporal_arith`): instant ± a nominal duration anchors it (calendar months
 * clamped, then days, then time); instant − instant is the exact span; duration
 * ± duration is component-wise; duration × integer scales. Else null.
 */
export function temporalArith(op: string, l: unknown, r: unknown): unknown {
  if (l === null || l === undefined || r === null || r === undefined) {
    return null;
  }

  if (op === '+') {
    if (l instanceof Duration && r instanceof Duration) {
      return addDurations(l, r);
    }

    if (isTemporal(l) && r instanceof Duration) {
      return anchorOrThrow(l, r);
    }

    if (l instanceof Duration && isTemporal(r)) {
      return anchorOrThrow(r, l);
    }

    return null;
  }

  if (op === '-') {
    if (l instanceof Duration && r instanceof Duration) {
      return addDurations(l, negateDuration(r));
    }

    if (isTemporal(l) && r instanceof Duration) {
      return anchorOrThrow(l, negateDuration(r));
    }

    if (isTemporal(l) && isTemporal(r)) {
      return durationBetween(r, l); // l − r = span from r to l
    }

    return null;
  }

  if (op === '*') {
    // Only an INTEGER factor scales a duration — a calendar duration (with a
    // `months` component) has no meaningful fractional multiple, so a
    // non-integer factor is invalid → null (never a silently-truncated result).
    if (l instanceof Duration && typeof r === 'number') {
      return Number.isInteger(r) ? scaleDuration(l, r) : null;
    }

    if (typeof l === 'number' && r instanceof Duration) {
      return Number.isInteger(l) ? scaleDuration(r, l) : null;
    }
  }

  return null;
}

/** The textual kind name and ISO payload of a temporal, derived from its tagged
 *  JSON form. `{"@date": "2020-01-01"}` -> `{ kind: 'date', iso: '2020-01-01' }`.
 *
 *  The kind names are the ones both textual dialects use as literal constructors
 *  (`date('…')`, `zoned_datetime('…')`), so a caller can format a literal without
 *  restating the tag table. Note `@localtime` maps to `time`, which is why a
 *  plain `tag.slice(1)` will not do.
 */
// Also derived from TEMPORAL_SPEC (keys can't drift from the tag set). The only
// value that isn't its own tag is `@localtime`, whose literal-constructor keyword
// is `time` (see the doc above) — the single documented exception.
const TEMPORAL_TAG_KIND: Readonly<Record<string, string>> = Object.fromEntries(
  Object.values(TEMPORAL_SPEC).map((s) => [`@${s.tag}`, s.tag === 'localtime' ? 'time' : s.tag]),
);

export const temporalLiteralParts = (v: Temporal): { kind: string; iso: string } | null => {
  const tagged = v.toJSON() as Record<string, string>;
  const [tag] = Object.keys(tagged);
  const kind = TEMPORAL_TAG_KIND[tag];

  return kind === undefined ? null : { kind, iso: tagged[tag] };
};
