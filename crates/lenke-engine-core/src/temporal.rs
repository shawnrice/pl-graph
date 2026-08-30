//! ISO/IEC 39075 temporal values — all six kinds: `DATE`, `LOCAL TIME`,
//! `LOCAL DATETIME`, `ZONED TIME`, `ZONED DATETIME`, `DURATION`. Ported faithfully
//! from the now-removed `lenke-core`'s `temporal.rs` so the two engines AGREE (same ISO-8601 wire
//! form, same order, same calendar arithmetic).
//!
//! Dependency-free: the calendar math is Howard Hinnant's civil-from-days
//! algorithm and the ISO-8601 parse/format is hand-rolled, so the wire form is a
//! pure function. Byte-identity is defined by the ISO-8601 string ([`format`],
//! [`Temporal::format`]) and the comparison order, not by the field layout.

use std::cmp::Ordering;

/// A calendar date: days since 1970-01-01 (proleptic Gregorian), ordered
/// chronologically. `days` is an `i32`, matching the TS engine (parse overflows the
/// `i32` range with an error).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Date {
    pub days: i32,
}

/// A zone-less time of day: seconds since midnight plus a sub-second nanosecond
/// part. Ordered chronologically within the day.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Time {
    /// Seconds since midnight, 0..86_400.
    pub secs: u32,
    /// 0..1_000_000_000
    pub nanos: u32,
}

/// A zone-less datetime: seconds since 1970-01-01T00:00:00 plus a sub-second
/// nanosecond part. Ordered chronologically (`secs` then `nanos`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DateTime {
    pub secs: i64,
    /// 0..1_000_000_000
    pub nanos: u32,
}

/// An ISO-8601 calendar duration. Months and days are kept SEPARATE from seconds
/// (a month is not a fixed number of seconds), matching the Cypher/GQL model.
/// PARTIALLY ordered for relational comparison (`partial_cmp_spec`, per W3C XML
/// Schema — a month vs a spanning day-count is incomparable → UNKNOWN), but given a
/// deterministic TOTAL order for `ORDER BY` (`total_key`, lexicographic over components).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Duration {
    pub months: i64,
    pub days: i64,
    pub secs: i64,
    /// 0..1_000_000_000
    pub nanos: u32,
}

/// A datetime with a UTC offset. Stored as the UTC instant plus the offset it was
/// written in — the offset is PRESERVED for round-trip and participates in
/// identity/ordering (instant first, offset second). Fields ordered
/// `secs, nanos, offset` so the derived `Ord` is instant-primary.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ZonedDateTime {
    /// UTC instant: seconds since 1970-01-01T00:00:00Z.
    pub secs: i64,
    /// 0..1_000_000_000
    pub nanos: u32,
    /// Offset from UTC in whole minutes (`Z` = 0), for round-trip rendering.
    pub offset: i16,
}

/// A time of day with a UTC offset. Stored as the UTC seconds-of-day + the
/// offset; ordered by UTC time-of-day then offset. No date component.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ZonedTime {
    /// UTC seconds-of-day, 0..86_400 (the wall clock minus the offset, wrapped).
    pub secs: u32,
    /// 0..1_000_000_000
    pub nanos: u32,
    /// Offset from UTC in whole minutes, for round-trip rendering.
    pub offset: i16,
}

const SECS_PER_DAY: i64 = 86_400;
const NANOS_PER_SEC: i64 = 1_000_000_000;

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Days in month `m` (1..=12) of proleptic-Gregorian year `y`.
fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(y) => 29,
        _ => 28,
    }
}

// --- civil calendar (Hinnant) ------------------------------------------------

/// Days since 1970-01-01 for a proleptic-Gregorian (y, m, d). `m` in 1..=12.
#[must_use]
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Proleptic-Gregorian (y, m, d) for days since 1970-01-01.
#[must_use]
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (
        if m <= 2 { y + 1 } else { y },
        u32::try_from(m).expect("month in 1..=12"),
        u32::try_from(d).expect("day in 1..=31"),
    )
}

// --- small parse/format helpers ----------------------------------------------

fn parse_int(s: &str) -> Result<i64, String> {
    s.parse::<i64>()
        .map_err(|_| format!("invalid integer '{s}'"))
}

/// Parse `HH:MM:SS[.fraction]` into (seconds-of-day, nanos).
fn parse_time(s: &str) -> Result<(i64, u32), String> {
    let (hms, frac) = match s.split_once('.') {
        Some((a, b)) => (a, Some(b)),
        None => (s, None),
    };
    let mut it = hms.split(':');
    let h = parse_int(it.next().unwrap_or(""))?;
    let m = parse_int(it.next().ok_or("missing minutes")?)?;
    let sec = parse_int(it.next().ok_or("missing seconds")?)?;
    if it.next().is_some() {
        return Err(format!("bad time '{s}'"));
    }
    if !(0..24).contains(&h) || !(0..60).contains(&m) || !(0..60).contains(&sec) {
        return Err(format!("time out of range '{s}'"));
    }
    let nanos = parse_frac(frac)?;
    Ok((h * 3600 + m * 60 + sec, nanos))
}

/// A fractional-second string (up to 9 digits) → nanoseconds.
fn parse_frac(frac: Option<&str>) -> Result<u32, String> {
    let Some(f) = frac else { return Ok(0) };
    if f.is_empty() || f.len() > 9 || !f.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("bad fractional seconds '.{f}'"));
    }
    let mut padded = f.to_string();
    while padded.len() < 9 {
        padded.push('0');
    }
    parse_int(&padded).map(|n| u32::try_from(n).expect("9-digit fraction fits u32"))
}

/// Render `nanos` as `.fraction` (trailing zeros trimmed), or empty if zero.
fn fmt_frac(nanos: u32) -> String {
    if nanos == 0 {
        return String::new();
    }
    let s = format!("{nanos:09}");
    format!(".{}", s.trim_end_matches('0'))
}

/// Zero-pad a year to at least 4 digits (a negative year renders as `-0009`).
fn fmt_year(y: i64) -> String {
    if y < 0 {
        format!("-{:04}", -y)
    } else {
        format!("{y:04}")
    }
}

/// Render a UTC offset (whole minutes) as `Z` (=0) or `±HH:MM`.
fn fmt_offset(offset: i16) -> String {
    if offset == 0 {
        return "Z".to_string();
    }
    let sign = if offset < 0 { '-' } else { '+' };
    let a = offset.unsigned_abs();
    format!("{sign}{:02}:{:02}", a / 60, a % 60)
}

/// Split a trailing UTC offset (`Z` / `±HH:MM` / `±HHMM`) off `s`, returning the
/// part before it and the offset in whole minutes. Errors if none is present (a
/// ZONED value requires one). Only the tail is inspected, so a date's `-`
/// separators are never mistaken for the offset sign.
fn split_offset(s: &str) -> Result<(&str, i16), String> {
    if let Some(rest) = s.strip_suffix('Z') {
        return Ok((rest, 0));
    }
    let b = s.as_bytes();
    let n = b.len();
    for (width, colon) in [(6usize, true), (5usize, false)] {
        if n < width {
            continue;
        }
        let start = n - width;
        let sign = b[start];
        if sign != b'+' && sign != b'-' {
            continue;
        }
        let Some(hh) = s.get(start + 1..start + 3) else {
            continue;
        };
        let mm = if colon {
            if b[start + 3] != b':' {
                continue;
            }
            s.get(start + 4..start + 6)
        } else {
            s.get(start + 3..start + 5)
        };
        let Some(mm) = mm else {
            continue;
        };
        if let (Ok(h), Ok(m)) = (hh.parse::<i16>(), mm.parse::<i16>()) {
            if (0..=23).contains(&h) && (0..60).contains(&m) {
                let mag = h * 60 + m;
                return Ok((&s[..start], if sign == b'-' { -mag } else { mag }));
            }
        }
    }
    Err(format!("missing/invalid time-zone offset in '{s}'"))
}

/// Consume the pending numeric buffer as an integer for duration designator `d`.
fn take_num(num: &mut String, d: char, whole: &str) -> Result<i64, String> {
    if num.is_empty() {
        return Err(format!("missing number before '{d}' in '{whole}'"));
    }
    let v = parse_int(num)?;
    num.clear();
    Ok(v)
}

/// Consume the pending buffer as `seconds[.fraction]` → (whole secs, nanos).
fn take_secs(num: &mut String, whole: &str) -> Result<(i64, u32), String> {
    if num.is_empty() {
        return Err(format!("missing number before 'S' in '{whole}'"));
    }
    let (w, frac) = match num.split_once('.') {
        Some((a, b)) => (a, Some(b.to_string())),
        None => (num.as_str(), None),
    };
    let secs = parse_int(w)?;
    let nanos = parse_frac(frac.as_deref())?;
    num.clear();
    Ok((secs, nanos))
}

// --- Date --------------------------------------------------------------------

impl Date {
    /// Parse `YYYY-MM-DD`.
    ///
    /// # Errors
    /// A malformed or out-of-range date (including one outside the `i32`-day range).
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut it = s.splitn(3, '-');
        let y = parse_int(it.next().ok_or("empty date")?)?;
        let m = parse_int(it.next().ok_or_else(|| format!("bad date '{s}'"))?)?;
        let d = parse_int(it.next().ok_or_else(|| format!("bad date '{s}'"))?)?;
        if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
            return Err(format!("date out of range '{s}'"));
        }
        let days = days_from_civil(y, m, d);
        i32::try_from(days)
            .map(|days| Self { days })
            .map_err(|_| format!("date out of range '{s}'"))
    }

    #[must_use]
    pub fn format(&self) -> String {
        let (y, m, d) = civil_from_days(i64::from(self.days));
        format!("{}-{m:02}-{d:02}", fmt_year(y))
    }

    /// Add `months` (calendar), CLAMPING the day to the new month's length
    /// (`Jan 31 + 1 month → Feb 28/29`), then `extra_days` as plain days. `None`
    /// when the result leaves the representable (`i32` day) range — the arithmetic
    /// layer turns that into a thrown fault, not a silent wrap.
    fn add_calendar(&self, months: i64, extra_days: i64) -> Option<Self> {
        let (y, m, d) = civil_from_days(i64::from(self.days));
        let total = y * 12 + (i64::from(m) - 1) + months;
        let ny = total.div_euclid(12);
        let nm = u32::try_from(total.rem_euclid(12) + 1).expect("1..=12");
        let nd = d.min(days_in_month(ny, nm));
        let base = days_from_civil(ny, i64::from(nm), i64::from(nd));
        i32::try_from(base + extra_days)
            .ok()
            .map(|days| Self { days })
    }
}

// --- DateTime ----------------------------------------------------------------

impl DateTime {
    /// Parse `YYYY-MM-DDTHH:MM:SS[.fraction]` (also accepts a space separator).
    ///
    /// # Errors
    /// A missing time part or any malformed/out-of-range component.
    pub fn parse(s: &str) -> Result<Self, String> {
        let sep = s
            .find(['T', ' '])
            .ok_or_else(|| format!("datetime missing time part '{s}'"))?;
        let date = Date::parse(&s[..sep])?;
        let (tod, nanos) = parse_time(&s[sep + 1..])?;
        Ok(Self {
            secs: i64::from(date.days) * SECS_PER_DAY + tod,
            nanos,
        })
    }

    #[must_use]
    pub fn format(&self) -> String {
        // Floor-divide so a pre-epoch time-of-day stays in [0, 86400).
        let days = self.secs.div_euclid(SECS_PER_DAY);
        let tod = self.secs.rem_euclid(SECS_PER_DAY);
        let (y, mo, d) = civil_from_days(days);
        let (h, m, s) = (tod / 3600, (tod % 3600) / 60, tod % 60);
        format!(
            "{}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}{}",
            fmt_year(y),
            fmt_frac(self.nanos)
        )
    }

    fn key(&self) -> (i64, u32) {
        (self.secs, self.nanos)
    }
}

impl Ord for DateTime {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key().cmp(&other.key())
    }
}
impl PartialOrd for DateTime {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// --- Time --------------------------------------------------------------------

impl Time {
    /// Parse `HH:MM:SS[.fraction]`.
    ///
    /// # Errors
    /// A malformed or out-of-range time.
    pub fn parse(s: &str) -> Result<Self, String> {
        let (tod, nanos) = parse_time(s)?;
        Ok(Self {
            secs: u32::try_from(tod).expect("time-of-day in 0..86_400"),
            nanos,
        })
    }

    #[must_use]
    pub fn format(&self) -> String {
        let (h, m, s) = (self.secs / 3600, (self.secs % 3600) / 60, self.secs % 60);
        format!("{h:02}:{m:02}:{s:02}{}", fmt_frac(self.nanos))
    }
}

// --- ZonedDateTime -----------------------------------------------------------

impl ZonedDateTime {
    /// Parse `YYYY-MM-DDTHH:MM:SS[.frac](Z|±HH:MM)`. The wall clock is the
    /// pre-offset datetime; the stored instant is that minus the offset.
    ///
    /// # Errors
    /// A missing/invalid offset, or a malformed datetime part.
    pub fn parse(s: &str) -> Result<Self, String> {
        let (dt_str, offset) = split_offset(s)?;
        let local = DateTime::parse(dt_str)?;
        Ok(Self {
            secs: local.secs - i64::from(offset) * 60,
            nanos: local.nanos,
            offset,
        })
    }

    #[must_use]
    pub fn format(&self) -> String {
        let local = DateTime {
            secs: self.secs + i64::from(self.offset) * 60,
            nanos: self.nanos,
        };
        format!("{}{}", local.format(), fmt_offset(self.offset))
    }
}

// --- ZonedTime ---------------------------------------------------------------

impl ZonedTime {
    /// Parse `HH:MM:SS[.frac](Z|±HH:MM)`. The wall clock is the pre-offset time;
    /// the stored UTC seconds-of-day is that minus the offset, wrapped into a day.
    ///
    /// # Errors
    /// A missing/invalid offset, or a malformed time part.
    pub fn parse(s: &str) -> Result<Self, String> {
        let (t_str, offset) = split_offset(s)?;
        let (tod, nanos) = parse_time(t_str)?;
        let utc = (tod - i64::from(offset) * 60).rem_euclid(SECS_PER_DAY);
        Ok(Self {
            secs: u32::try_from(utc).expect("rem_euclid keeps it in 0..86_400"),
            nanos,
            offset,
        })
    }

    #[must_use]
    pub fn format(&self) -> String {
        let local = (i64::from(self.secs) + i64::from(self.offset) * 60).rem_euclid(SECS_PER_DAY);
        let (h, m, s) = (local / 3600, (local % 3600) / 60, local % 60);
        format!(
            "{h:02}:{m:02}:{s:02}{}{}",
            fmt_frac(self.nanos),
            fmt_offset(self.offset)
        )
    }
}

// --- Duration ----------------------------------------------------------------

impl Duration {
    /// The float64-safe integer bound each component must stay under.
    const MAX_SAFE: u64 = 1 << 53;

    /// Parse ISO-8601 `PnYnMnWnDTnHnMnS` (years→months, weeks→days). Fractional
    /// seconds allowed on the seconds field.
    ///
    /// # Errors
    /// A malformed field, a dangling number, or a component outside the
    /// float64-safe integer range.
    pub fn parse(s: &str) -> Result<Self, String> {
        let rest = s
            .strip_prefix('P')
            .ok_or_else(|| format!("duration must start with 'P': '{s}'"))?;
        let (date_part, time_part) = match rest.split_once('T') {
            Some((d, t)) => (d, Some(t)),
            None => (rest, None),
        };
        let mut months = 0i64;
        let mut days = 0i64;
        let mut num = String::new();
        for c in date_part.chars() {
            match c {
                '0'..='9' | '-' => num.push(c),
                'Y' => months += take_num(&mut num, 'Y', s)? * 12,
                'M' => months += take_num(&mut num, 'M', s)?,
                'W' => days += take_num(&mut num, 'W', s)? * 7,
                'D' => days += take_num(&mut num, 'D', s)?,
                _ => return Err(format!("bad duration date field '{c}' in '{s}'")),
            }
        }
        if !num.is_empty() {
            return Err(format!("dangling number in duration '{s}'"));
        }
        let mut secs = 0i64;
        let mut nanos = 0u32;
        if let Some(tp) = time_part {
            for c in tp.chars() {
                match c {
                    '0'..='9' | '-' | '.' => num.push(c),
                    'H' => secs += take_num(&mut num, 'H', s)? * 3600,
                    'M' => secs += take_num(&mut num, 'M', s)? * 60,
                    'S' => {
                        let (whole, frac) = take_secs(&mut num, s)?;
                        secs += whole;
                        nanos = frac;
                    }
                    _ => return Err(format!("bad duration time field '{c}' in '{s}'")),
                }
            }
            if !num.is_empty() {
                return Err(format!("dangling number in duration '{s}'"));
            }
        }
        Self {
            months,
            days,
            secs,
            nanos,
        }
        .representable()
        .ok_or_else(|| format!("duration component is not representable as float64: '{s}'"))
    }

    /// Canonical ISO-8601: `P<months>M<days>DT<secs>S`, each component omitted
    /// when zero; all-zero renders `PT0S`. Total months / total days (no Y/W
    /// split) so the form is deterministic and round-trips to itself.
    #[must_use]
    pub fn format(&self) -> String {
        let mut out = String::from("P");
        if self.months != 0 {
            out.push_str(&format!("{}M", self.months));
        }
        if self.days != 0 {
            out.push_str(&format!("{}D", self.days));
        }
        if self.secs != 0 || self.nanos != 0 {
            out.push_str(&format!("T{}{}S", self.secs, fmt_frac(self.nanos)));
        }
        if out == "P" {
            out.push_str("T0S");
        }
        out
    }

    /// Deterministic total order (NOT chronological — a month has no fixed
    /// length): lexicographic over (months, days, secs, nanos). For `ORDER BY`.
    fn total_key(&self) -> (i64, i64, i64, u32) {
        (self.months, self.days, self.secs, self.nanos)
    }

    /// The 3-valued PREDICATE order (`< > <= >=`) on two durations, per W3C XML Schema
    /// Part 2: Datatypes §3.2.6.2 ("order relation on duration"): `self < other` iff
    /// `s + self < s + other` for EACH of the four reference dateTimes the spec fixes. If
    /// the four do not all agree, the pair is INDETERMINATE (`None`) — durations are only
    /// PARTIALLY ordered because a month is 28-31 days (so `P1M` vs `P30D` is indeterminate,
    /// while `P1D` vs `P2D`, and `P1M` vs `P27D`, are determinate). `None` also on a date
    /// overflow. The TOTAL order for `ORDER BY` / min / max is separate (`total_key`).
    #[must_use]
    pub fn partial_cmp_spec(&self, other: &Self) -> Option<Ordering> {
        // The spec's four reference instants: a non-leap and a leap February, and months
        // of 31 and 30 days — enough to expose any month-length ambiguity between the two.
        const REFS: [(i64, i64, i64); 4] = [(1696, 9, 1), (1697, 2, 1), (1903, 3, 1), (1903, 7, 1)];
        let mut acc: Option<Ordering> = None;
        for (y, m, d) in REFS {
            let base = Temporal::DateTime(DateTime {
                secs: days_from_civil(y, m, d) * SECS_PER_DAY,
                nanos: 0,
            });
            let ord = base
                .add_duration(self)?
                .cmp_total(&base.add_duration(other)?);
            match acc {
                None => acc = Some(ord),
                Some(prev) if prev != ord => return None, // references disagree → indeterminate
                Some(_) => {}
            }
        }
        acc
    }

    /// Negate the whole span, keeping `nanos` in `[0, 1e9)`.
    #[must_use]
    pub fn negate(&self) -> Self {
        let (secs, nanos) = if self.nanos == 0 {
            (-self.secs, 0)
        } else {
            (-self.secs - 1, 1_000_000_000 - self.nanos)
        };
        Self {
            months: -self.months,
            days: -self.days,
            secs,
            nanos,
        }
    }

    /// Component-wise sum of two (nominal) durations, nanos carrying into secs.
    /// `None` on i64 overflow or leaving the f64-safe range.
    #[must_use]
    pub fn add(&self, o: &Self) -> Option<Self> {
        let mut secs = self.secs.checked_add(o.secs)?;
        let mut nanos = i64::from(self.nanos) + i64::from(o.nanos);
        if nanos >= NANOS_PER_SEC {
            nanos -= NANOS_PER_SEC;
            secs = secs.checked_add(1)?;
        }
        Self {
            months: self.months.checked_add(o.months)?,
            days: self.days.checked_add(o.days)?,
            secs,
            nanos: u32::try_from(nanos).expect("0..1e9 after carry"),
        }
        .representable()
    }

    /// Scale every component by an integer factor (nanos carry into secs). `None`
    /// on i64 overflow or leaving the f64-safe range.
    #[must_use]
    pub fn scale(&self, n: i64) -> Option<Self> {
        let total_nanos = i64::from(self.nanos).checked_mul(n)?;
        Some(Self {
            months: self.months.checked_mul(n)?,
            days: self.days.checked_mul(n)?,
            secs: self
                .secs
                .checked_mul(n)?
                .checked_add(total_nanos.div_euclid(NANOS_PER_SEC))?,
            nanos: u32::try_from(total_nanos.rem_euclid(NANOS_PER_SEC)).expect("0..1e9"),
        })
        .and_then(Self::representable)
    }

    fn representable(self) -> Option<Self> {
        if self.months.unsigned_abs() >= Self::MAX_SAFE
            || self.days.unsigned_abs() >= Self::MAX_SAFE
            || self.secs.unsigned_abs() >= Self::MAX_SAFE
        {
            None
        } else {
            Some(self)
        }
    }
}

// --- the value family --------------------------------------------------------

/// The temporal value family, carried as one `Value::Temporal` variant so each
/// exhaustive match gains a single arm.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Temporal {
    Date(Date),
    Time(Time),
    DateTime(DateTime),
    ZonedTime(ZonedTime),
    ZonedDateTime(ZonedDateTime),
    Duration(Duration),
}

/// The discriminant of a [`Temporal`] (no payload) — used to type a homogeneous
/// packed temporal storage column (a column holds exactly one kind).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TemporalKind {
    Date,
    Time,
    DateTime,
    ZonedTime,
    ZonedDateTime,
    Duration,
}

impl TemporalKind {
    /// A canonical zero value of this kind — the placeholder an ABSENT slot in a
    /// typed temporal column holds (never read; the presence bitmap gates it).
    #[must_use]
    pub fn zero(self) -> Temporal {
        match self {
            Self::Date => Temporal::Date(Date { days: 0 }),
            Self::Time => Temporal::Time(Time { secs: 0, nanos: 0 }),
            Self::DateTime => Temporal::DateTime(DateTime { secs: 0, nanos: 0 }),
            Self::ZonedTime => Temporal::ZonedTime(ZonedTime {
                secs: 0,
                nanos: 0,
                offset: 0,
            }),
            Self::ZonedDateTime => Temporal::ZonedDateTime(ZonedDateTime {
                secs: 0,
                nanos: 0,
                offset: 0,
            }),
            Self::Duration => Temporal::Duration(Duration {
                months: 0,
                days: 0,
                secs: 0,
                nanos: 0,
            }),
        }
    }
}

impl Temporal {
    /// The discriminant (no payload), for typing a packed temporal column.
    #[must_use]
    pub fn kind(&self) -> TemporalKind {
        match self {
            Self::Date(_) => TemporalKind::Date,
            Self::Time(_) => TemporalKind::Time,
            Self::DateTime(_) => TemporalKind::DateTime,
            Self::ZonedTime(_) => TemporalKind::ZonedTime,
            Self::ZonedDateTime(_) => TemporalKind::ZonedDateTime,
            Self::Duration(_) => TemporalKind::Duration,
        }
    }

    /// The kind tag used by codecs and the value key.
    #[must_use]
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Date(_) => "date",
            Self::Time(_) => "localtime",
            Self::DateTime(_) => "datetime",
            Self::ZonedTime(_) => "zoned_time",
            Self::ZonedDateTime(_) => "zoned_datetime",
            Self::Duration(_) => "duration",
        }
    }

    /// The ISO-8601 string form (the byte-identity wire value).
    #[must_use]
    pub fn format(&self) -> String {
        match self {
            Self::Date(d) => d.format(),
            Self::Time(t) => t.format(),
            Self::DateTime(dt) => dt.format(),
            Self::ZonedTime(t) => t.format(),
            Self::ZonedDateTime(dt) => dt.format(),
            Self::Duration(du) => du.format(),
        }
    }

    /// Build from a kind tag + ISO string.
    ///
    /// # Errors
    /// An unknown tag, or a string that does not parse for that kind.
    pub fn parse(tag: &str, s: &str) -> Result<Self, String> {
        match tag {
            "date" => Date::parse(s).map(Temporal::Date),
            "localtime" => Time::parse(s).map(Temporal::Time),
            "datetime" => DateTime::parse(s).map(Temporal::DateTime),
            "zoned_time" => ZonedTime::parse(s).map(Temporal::ZonedTime),
            "zoned_datetime" => ZonedDateTime::parse(s).map(Temporal::ZonedDateTime),
            "duration" => Duration::parse(s).map(Temporal::Duration),
            _ => Err(format!("unknown temporal kind '{tag}'")),
        }
    }

    /// Kind rank for the cross-kind total order (date < localtime < datetime <
    /// zoned_time < zoned_datetime < duration).
    fn kind_rank(&self) -> u8 {
        match self {
            Self::Date(_) => 0,
            Self::Time(_) => 1,
            Self::DateTime(_) => 2,
            Self::ZonedTime(_) => 3,
            Self::ZonedDateTime(_) => 4,
            Self::Duration(_) => 5,
        }
    }

    /// Deterministic TOTAL order over all temporals (for `ORDER BY`/min/max): by
    /// kind, then chronologically within date/time/datetime and zoned kinds, and
    /// lexicographically within duration.
    #[must_use]
    pub fn cmp_total(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Date(a), Self::Date(b)) => a.cmp(b),
            (Self::Time(a), Self::Time(b)) => a.cmp(b),
            (Self::DateTime(a), Self::DateTime(b)) => a.cmp(b),
            (Self::ZonedTime(a), Self::ZonedTime(b)) => a.cmp(b),
            (Self::ZonedDateTime(a), Self::ZonedDateTime(b)) => a.cmp(b),
            (Self::Duration(a), Self::Duration(b)) => a.total_key().cmp(&b.total_key()),
            _ => self.kind_rank().cmp(&other.kind_rank()),
        }
    }

    /// The 3-valued PREDICATE order (`< > <= >=` in an expression / WHERE) for two
    /// temporals of the SAME kind (the caller checks `kind()`). Date/time/datetime kinds
    /// are totally ordered chronologically; two DURATIONS follow the W3C partial order
    /// (`Duration::partial_cmp_spec`), so an incomparable pair (a month vs a spanning
    /// day-count) is `None` → UNKNOWN. This is DISTINCT from `cmp_total`, which forces a
    /// total order for `ORDER BY` / min / max determinism.
    #[must_use]
    pub fn partial_cmp_pred(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Duration(a), Self::Duration(b)) => a.partial_cmp_spec(b),
            _ => Some(self.cmp_total(other)),
        }
    }

    /// `self + duration` for a date/time/datetime (and their zoned forms): apply
    /// calendar months (clamped), then days, then the sub-day part. A bare `Time`
    /// wraps within the day (months/days ignored); a zoned value applies it to the
    /// LOCAL wall clock, then re-anchors to UTC keeping the offset. `None` when the
    /// result leaves the representable date range, or when `self` is a duration.
    #[must_use]
    pub fn add_duration(&self, d: &Duration) -> Option<Self> {
        match self {
            Self::Date(date) => date.add_calendar(d.months, d.days).map(Self::Date),
            Self::Time(t) => {
                let carry_nanos = i64::from(t.nanos) + i64::from(d.nanos);
                let secs = i64::from(t.secs) + d.secs + carry_nanos.div_euclid(NANOS_PER_SEC);
                Some(Self::Time(Time {
                    secs: u32::try_from(secs.rem_euclid(SECS_PER_DAY)).expect("0..86_400"),
                    nanos: u32::try_from(carry_nanos.rem_euclid(NANOS_PER_SEC)).expect("0..1e9"),
                }))
            }
            Self::DateTime(dt) => {
                let days0 = dt.secs.div_euclid(SECS_PER_DAY);
                let tod = dt.secs.rem_euclid(SECS_PER_DAY);
                let date = Date {
                    days: i32::try_from(days0).ok()?,
                }
                .add_calendar(d.months, d.days)?;
                let mut secs = i64::from(date.days) * SECS_PER_DAY + tod + d.secs;
                let mut nanos = i64::from(dt.nanos) + i64::from(d.nanos);
                if nanos >= NANOS_PER_SEC {
                    nanos -= NANOS_PER_SEC;
                    secs += 1;
                }
                Some(Self::DateTime(DateTime {
                    secs,
                    nanos: u32::try_from(nanos).expect("0..1e9 after carry"),
                }))
            }
            Self::ZonedDateTime(zdt) => {
                let local = DateTime {
                    secs: zdt.secs + i64::from(zdt.offset) * 60,
                    nanos: zdt.nanos,
                };
                let Self::DateTime(nl) = Self::DateTime(local).add_duration(d)? else {
                    return None;
                };
                Some(Self::ZonedDateTime(ZonedDateTime {
                    secs: nl.secs - i64::from(zdt.offset) * 60,
                    nanos: nl.nanos,
                    offset: zdt.offset,
                }))
            }
            Self::ZonedTime(zt) => {
                let local_secs =
                    (i64::from(zt.secs) + i64::from(zt.offset) * 60).rem_euclid(SECS_PER_DAY);
                let local = Time {
                    secs: u32::try_from(local_secs).expect("0..86_400"),
                    nanos: zt.nanos,
                };
                let Self::Time(nt) = Self::Time(local).add_duration(d)? else {
                    return None;
                };
                let utc = (i64::from(nt.secs) - i64::from(zt.offset) * 60).rem_euclid(SECS_PER_DAY);
                Some(Self::ZonedTime(ZonedTime {
                    secs: u32::try_from(utc).expect("0..86_400"),
                    nanos: nt.nanos,
                    offset: zt.offset,
                }))
            }
            Self::Duration(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// W3C XML Schema Part 2: Datatypes §3.2.6.2 "order relation on duration". A duration
    /// pair is comparable only when the four reference dateTimes all agree; a month vs a
    /// day-count that spans a month's 28-31-day range is INDETERMINATE.
    #[test]
    fn duration_partial_order_follows_w3c_xml_schema() {
        use Ordering::{Equal, Greater, Less};
        let d = |s: &str| Duration::parse(s).unwrap();
        let cmp = |a: &str, b: &str| d(a).partial_cmp_spec(&d(b));

        // Determinate — day/time only, or ranges that cannot overlap.
        assert_eq!(cmp("P1D", "P2D"), Some(Less));
        assert_eq!(cmp("P2D", "P1D"), Some(Greater));
        assert_eq!(cmp("P1D", "P1D"), Some(Equal));
        assert_eq!(cmp("P1D", "PT25H"), Some(Less)); // 24h < 25h (the old compare got this backwards)
        assert_eq!(cmp("PT25H", "P1D"), Some(Greater));
        assert_eq!(cmp("P1M", "P27D"), Some(Greater)); // a month is >= 28 days > 27
        assert_eq!(cmp("P1M", "P32D"), Some(Less)); // a month is <= 31 days < 32
        assert_eq!(cmp("P1Y", "P360D"), Some(Greater)); // a year is >= 365 days
        assert_eq!(cmp("P1Y", "P400D"), Some(Less)); // a year is <= 366 days

        // Indeterminate — the spec's own examples (a month is 28-31 days; a year 365-366).
        for days in ["P28D", "P29D", "P30D", "P31D"] {
            assert_eq!(
                cmp("P1M", days),
                None,
                "P1M vs {days} must be indeterminate"
            );
        }
        assert_eq!(cmp("P1Y", "P365D"), None);
        assert_eq!(cmp("P1Y", "P366D"), None);
        // The total order (ORDER BY) stays defined for every pair, even indeterminate ones.
        assert_eq!(
            Temporal::Duration(d("P1M")).cmp_total(&Temporal::Duration(d("P30D"))),
            Greater
        );
    }

    #[test]
    fn civil_round_trips_the_epoch_and_neighbours() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1970, 1, 2), 1);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        for &(y, m, d) in &[(1970, 1, 1), (2000, 2, 29), (1, 1, 1), (2024, 12, 31)] {
            let days = days_from_civil(y, m, d);
            assert_eq!(civil_from_days(days), (y, m as u32, d as u32));
        }
    }

    #[test]
    fn date_parse_format_round_trip() {
        let d = Date::parse("2024-01-15").unwrap();
        assert_eq!(d.format(), "2024-01-15");
        // 2024-01-15 is 19737 days after the epoch (hand: 54 years incl. 13 leap
        // days is far off; trust the round trip + a known anchor).
        assert_eq!(Date::parse("1970-01-01").unwrap().days, 0);
        assert_eq!(Date::parse("1970-01-02").unwrap().days, 1);
    }

    #[test]
    fn date_out_of_range_and_malformed_error() {
        assert!(Date::parse("2024-13-01").is_err()); // month
        assert!(Date::parse("2024-01-32").is_err()); // day
        assert!(Date::parse("2024-01").is_err()); // missing day
    }

    #[test]
    fn time_and_datetime_round_trip() {
        assert_eq!(Time::parse("13:45:06").unwrap().format(), "13:45:06");
        assert_eq!(Time::parse("00:00:00.5").unwrap().format(), "00:00:00.5");
        assert_eq!(
            DateTime::parse("2024-01-15T13:45:06").unwrap().format(),
            "2024-01-15T13:45:06"
        );
        // A space separator parses; it formats back with `T`.
        assert_eq!(
            DateTime::parse("2024-01-15 00:00:00").unwrap().format(),
            "2024-01-15T00:00:00"
        );
    }

    #[test]
    fn duration_parse_and_canonical_format() {
        // Years->months, weeks->days; the canonical form is total M / total D.
        assert_eq!(
            Duration::parse("P1Y2M3DT4H5M6S").unwrap().format(),
            "P14M3DT14706S"
        );
        assert_eq!(Duration::parse("P1W").unwrap().format(), "P7D");
        assert_eq!(Duration::parse("PT0S").unwrap().format(), "PT0S");
        assert_eq!(Duration::parse("P0D").unwrap().format(), "PT0S"); // all-zero
        assert_eq!(Duration::parse("PT1.5S").unwrap().format(), "PT1.5S");
        assert!(Duration::parse("1Y").is_err()); // missing leading P
        assert!(Duration::parse("P1X").is_err()); // bad field
    }

    #[test]
    fn zoned_datetime_round_trip_and_offset_order() {
        assert_eq!(
            ZonedDateTime::parse("2024-01-15T12:00:00+01:00")
                .unwrap()
                .format(),
            "2024-01-15T12:00:00+01:00"
        );
        assert_eq!(
            ZonedDateTime::parse("2024-01-15T12:00:00Z")
                .unwrap()
                .format(),
            "2024-01-15T12:00:00Z"
        );
        // Same UTC instant, different offset: ordered by instant then offset,
        // so Z (0) sorts before +01:00 (+60).
        let a = Temporal::ZonedDateTime(ZonedDateTime::parse("2024-01-15T12:00:00Z").unwrap());
        let b = Temporal::ZonedDateTime(ZonedDateTime::parse("2024-01-15T13:00:00+01:00").unwrap());
        assert_eq!(a.cmp_total(&b), Ordering::Less);
        // A missing offset is an error (a ZONED value requires one).
        assert!(ZonedDateTime::parse("2024-01-15T12:00:00").is_err());
    }

    #[test]
    fn zoned_time_round_trip() {
        assert_eq!(
            ZonedTime::parse("13:45:00+02:00").unwrap().format(),
            "13:45:00+02:00"
        );
        assert_eq!(ZonedTime::parse("13:45:00Z").unwrap().format(), "13:45:00Z");
    }

    #[test]
    fn cross_kind_rank_spans_all_six() {
        let d = Temporal::Date(Date::parse("2024-01-01").unwrap());
        let du = Temporal::Duration(Duration::parse("P1D").unwrap());
        // date (rank 0) < duration (rank 5).
        assert_eq!(d.cmp_total(&du), Ordering::Less);
        // Two durations order lexicographically by component.
        let d1 = Temporal::Duration(Duration::parse("P1D").unwrap());
        let d2 = Temporal::Duration(Duration::parse("P2D").unwrap());
        assert_eq!(d1.cmp_total(&d2), Ordering::Less);
    }

    #[test]
    fn chronological_order_and_cross_kind_rank() {
        let d1 = Temporal::Date(Date::parse("2024-01-01").unwrap());
        let d2 = Temporal::Date(Date::parse("2024-06-01").unwrap());
        assert_eq!(d1.cmp_total(&d2), Ordering::Less);
        // Cross-kind falls back to the kind rank: date < datetime.
        let dt = Temporal::DateTime(DateTime::parse("1900-01-01T00:00:00").unwrap());
        assert_eq!(d1.cmp_total(&dt), Ordering::Less);
    }
}
