import { describe, expect, test } from 'bun:test';

import {
  civilFromDays,
  daysFromCivil,
  coerceTemporal,
  Duration,
  formatDuration,
  fromTaggedJson,
  LocalDate,
  LocalDateTime,
  parseDate,
  parseDateTime,
  parseDuration,
  TEMPORAL_TAG_KEYS,
  temporalArith,
  temporalCmpTotal,
  temporalParse,
  temporalRelCmp,
} from './temporal.js';

// These mirror the Rust `temporal::tests` one-for-one — identical inputs and
// expected outputs pin the two engines' calendar math + ISO parse/format to the
// same byte string (the real cross-engine differential rides on top of this).

describe('temporal: canonical tagged-temporal keys', () => {
  // TEMPORAL_TAG_KEYS is the single source of accepted `@`-keys, derived from
  // TEMPORAL_SPEC. It must exactly equal what fromTaggedJson revives — and must
  // NOT drift into bogus tags (a hand-maintained copy once had '@time' and
  // dropped '@zoned_time', silently rejecting valid ZONED TIME params).
  const CANONICAL = [
    '@date',
    '@localtime',
    '@datetime',
    '@zoned_time',
    '@zoned_datetime',
    '@duration',
  ];

  test('the set is exactly the six canonical keys', () => {
    expect([...TEMPORAL_TAG_KEYS].sort()).toEqual([...CANONICAL].sort());
    expect(TEMPORAL_TAG_KEYS.has('@time')).toBe(false);
    expect(TEMPORAL_TAG_KEYS.has('@zoned_time')).toBe(true);
  });

  test('every key in the set is revived by fromTaggedJson (and only those)', () => {
    const sample: Record<string, string> = {
      '@date': '2020-01-01',
      '@localtime': '08:30:00',
      '@datetime': '2020-01-01T08:30:00',
      '@zoned_time': '12:00:00Z',
      '@zoned_datetime': '2020-01-01T08:30:00Z',
      '@duration': 'P1Y2M3DT4H',
    };

    for (const key of TEMPORAL_TAG_KEYS) {
      expect(fromTaggedJson({ [key]: sample[key] })).not.toBeNull();
    }

    expect(fromTaggedJson({ '@time': '12:00:00' })).toBeNull();
  });
});

describe('temporal: negative (BCE) year formatting', () => {
  // A negative year (reachable only via arithmetic — both engines reject a leading
  // `-` on input) must pad AFTER the sign, matching Rust `format!("{y:04}")`:
  // `-0009`, not the old `pad`'s `00-9` (which padded across the minus).
  test('year is padded after the sign for a pre-year-1 date/datetime', () => {
    const d = temporalArith('-', parseDate('0001-01-01'), parseDuration('P10Y'));
    expect(String(d)).toBe('-0009-01-01');
    const dt = temporalArith('-', parseDateTime('0001-01-01T12:00:00'), parseDuration('P10Y'));
    expect(String(dt)).toBe('-0009-01-01T12:00:00');
  });
});

describe('temporal: civil calendar', () => {
  test('round-trips known dates', () => {
    for (const [y, m, d] of [
      [1970, 1, 1],
      [2000, 1, 1],
      [2020, 2, 29],
      [1969, 12, 31],
      [1600, 12, 31],
      [2262, 4, 11],
    ] as const) {
      expect(civilFromDays(daysFromCivil(y, m, d))).toEqual([y, m, d]);
    }

    expect(daysFromCivil(1970, 1, 1)).toBe(0);
    expect(daysFromCivil(1970, 1, 2)).toBe(1);
    expect(daysFromCivil(1969, 12, 31)).toBe(-1);
  });
});

describe('temporal: parse/format round-trips', () => {
  test('date', () => {
    for (const s of ['1970-01-01', '2020-02-29', '1999-12-31', '2026-07-11']) {
      expect(parseDate(s).toJSON()['@date']).toBe(s);
    }

    expect(() => parseDate('2020-13-01')).toThrow();
    expect(() => parseDate('not-a-date')).toThrow();
  });

  test('datetime (incl. fraction + space separator + pre-epoch)', () => {
    for (const s of [
      '2020-01-01T00:00:00',
      '2026-07-11T13:45:06',
      '2020-01-01T10:15:30.5',
      '1969-12-31T23:59:59',
    ]) {
      expect(parseDateTime(s).toJSON()['@datetime']).toBe(s);
    }

    expect(parseDateTime('2020-01-01 10:15:30').toJSON()['@datetime']).toBe('2020-01-01T10:15:30');
  });

  test('duration normalizes years->months, weeks->days', () => {
    expect(formatDuration(parseDuration('P1Y2M3W4DT5H6M7S'))).toBe('P14M25DT18367S');
    expect(formatDuration(parseDuration('P1Y'))).toBe('P12M');
    expect(formatDuration(parseDuration('PT0S'))).toBe('PT0S');
    expect(formatDuration(parseDuration('P0D'))).toBe('PT0S');
    expect(formatDuration(parseDuration('PT1.5S'))).toBe('PT1.5S');
    // canonical output re-parses to itself
    const canon = parseDuration('P14M25DT18367S');
    expect(formatDuration(parseDuration(formatDuration(canon)))).toBe(formatDuration(canon));
    expect(() => parseDuration('1Y')).toThrow();
  });
});

describe('temporal: ordering', () => {
  test('is deterministic and matches the Rust policy', () => {
    const d1 = parseDate('2020-01-01');
    const d2 = parseDate('2020-06-01');
    expect(temporalRelCmp(d1, d2)).toBe(-1);
    expect(temporalCmpTotal(d1, d2)).toBe(-1);

    const t1 = parseDateTime('2020-01-01T00:00:00');
    expect(temporalRelCmp(d1, t1)).toBeNull(); // cross-kind: UNKNOWN
    expect(temporalCmpTotal(d1, t1)).toBe(-1); // date kind-rank < datetime

    const du = parseDuration('P1M');
    expect(temporalRelCmp(du, du)).toBe(0); // a duration equals itself (determinate)
    expect(temporalCmpTotal(du, du)).toBe(0);
    expect(temporalCmpTotal(t1, du)).toBe(-1); // datetime kind-rank < duration
  });

  // W3C XML Schema Part 2: Datatypes §3.2.6.2 "order relation on duration": comparable
  // only when the four reference dateTimes agree; a month vs a spanning day-count is null.
  test('durations follow the W3C XML Schema partial order', () => {
    const cmp = (a: string, b: string): number | null =>
      temporalRelCmp(parseDuration(a), parseDuration(b));

    // Determinate — day/time only, or ranges that cannot overlap.
    expect(cmp('P1D', 'P2D')).toBe(-1);
    expect(cmp('P1D', 'PT25H')).toBe(-1); // 24h < 25h
    expect(cmp('PT25H', 'P1D')).toBe(1);
    expect(cmp('P1M', 'P27D')).toBe(1); // a month is >= 28 days > 27
    expect(cmp('P1M', 'P32D')).toBe(-1); // a month is <= 31 days < 32
    expect(cmp('P1Y', 'P360D')).toBe(1); // a year is >= 365 days
    expect(cmp('P1Y', 'P400D')).toBe(-1);

    // Indeterminate — the spec's own examples (a month is 28-31 days; a year 365-366).
    for (const days of ['P28D', 'P29D', 'P30D', 'P31D']) {
      expect(cmp('P1M', days)).toBeNull();
    }

    expect(cmp('P1Y', 'P365D')).toBeNull();
    expect(cmp('P1Y', 'P366D')).toBeNull();
    // The total order stays defined for every pair (ORDER BY), even indeterminate ones.
    expect(temporalCmpTotal(parseDuration('P1M'), parseDuration('P30D'))).not.toBe(0);
  });

  test('instances expose Rust-identical fields', () => {
    expect(parseDate('1970-01-02')).toEqual(new LocalDate(1));
    expect(parseDateTime('1970-01-01T00:00:01').secs).toBe(1);
    expect(parseDuration('P2M3DT4S')).toEqual(new Duration(2, 3, 4, 0));
  });
});

describe('temporal: JS interop', () => {
  test('toString / toISOString give the ISO string (the bridge to any library)', () => {
    expect(String(parseDate('2020-02-29'))).toBe('2020-02-29');
    expect(parseDateTime('2021-06-15T08:30:00.5').toISOString()).toBe('2021-06-15T08:30:00.5');
    // Durations: ISO-8601 string round-trips with Temporal.Duration / Luxon.
    expect(parseDuration('P1Y2M').toString()).toBe('P14M');
  });

  test('fromJSDate takes the wall clock in an explicit zone', () => {
    // 2020-01-02T03:04:05.678Z as UTC wall clock.
    const d = new Date('2020-01-02T03:04:05.678Z');
    const dt = LocalDateTime.fromJSDate(d, { zone: 'utc' });
    expect(dt.toISOString()).toBe('2020-01-02T03:04:05.678');
    expect(LocalDate.fromJSDate(d, { zone: 'utc' }).toISOString()).toBe('2020-01-02');
  });

  test('coerceTemporal accepts our instances and TC39 Temporal.Plain* (duck-typed)', () => {
    const mine = parseDate('2020-01-01');
    expect(coerceTemporal(mine)).toBe(mine);

    // Simulate a TC39 Temporal.PlainDate via its @@toStringTag brand + ISO toString.
    const fakePlainDate = {
      [Symbol.toStringTag]: 'Temporal.PlainDate',
      toString: () => '2022-03-04[u-ca=iso8601]', // with a calendar annotation
    };
    expect(coerceTemporal(fakePlainDate)?.toString()).toBe('2022-03-04');

    // A native Date is NOT coerced (zoned instant) — returns null here.
    expect(coerceTemporal(new Date())).toBeNull();
    expect(coerceTemporal('2020-01-01')).toBeNull(); // a bare string stays a string
    expect(coerceTemporal(42)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Every temporal parse failure carries an error CODE.
//
// These parsers sit behind `fromTaggedJson`, which the graph mutation boundary,
// the GQL param binder, the Gremlin predicates and every codec decoder all call
// on caller-supplied data. They used to throw bare `Error`s, so a hostile
// `{"@date": "…"}` produced an UNCODED failure on each of those surfaces while
// the Rust engine coded the same input — a boundary that maps coded errors to a
// 4xx and everything else to a 5xx would answer attacker-controlled input with a
// 500. Nothing executed and nothing was injectable; the discipline was the gap.
// Found by extending the injection fuzz to structured (non-string) values.
// ---------------------------------------------------------------------------

describe('temporal parse failures are coded', () => {
  const code = (fn: () => unknown): string | undefined => {
    try {
      fn();
    } catch (e) {
      return (e as { code?: string }).code;
    }

    return undefined;
  };

  test.each([
    ['date', 'nope'],
    ['date', "'); DROP //"],
    ['date', '2020-13-01'],
    ['date', ''],
    ['datetime', 'nope'],
    ['datetime', '2020-01-01'],
    ['localtime', 'nope'],
    ['localtime', '99:99:99'],
    ['duration', 'nope'],
    ['duration', 'P1X'],
    ['duration', 'PT1'],
    ['zoned_datetime', '2020-01-01T00:00:00'],
    ['zoned_time', 'nope'],
  ])('{"@%s": %j} throws E_INVALID_VALUE', (tag, bad) => {
    expect(code(() => fromTaggedJson({ [`@${tag}`]: bad }))).toBe('E_INVALID_VALUE');
  });

  test('an unknown temporal kind is coded too', () => {
    expect(code(() => temporalParse('bogus', '2020-01-01'))).toBe('E_INVALID_VALUE');
  });

  test('the direct parsers are coded as well', () => {
    expect(code(() => parseDate('nope'))).toBe('E_INVALID_VALUE');
    expect(code(() => parseDateTime('nope'))).toBe('E_INVALID_VALUE');
    expect(code(() => parseDuration('nope'))).toBe('E_INVALID_VALUE');
  });

  test('a well-formed tagged temporal still parses, and a non-temporal is still null', () => {
    expect(fromTaggedJson({ '@date': '2020-01-01' })).toBeInstanceOf(LocalDate);
    expect(fromTaggedJson({ notATag: 'x' })).toBeNull();
    expect(fromTaggedJson({ '@date': 5 })).toBeNull();
    expect(fromTaggedJson(null)).toBeNull();
  });
});
