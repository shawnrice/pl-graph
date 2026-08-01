//! ISO/IEC 39075 temporal values — `DATE`, `LOCAL TIME`, `LOCAL DATETIME`,
//! `ZONED TIME`, `ZONED DATETIME`, `DURATION`.
//!
//! Dependency-free (no `chrono`/`time`): the calendar math is Howard Hinnant's
//! civil-from-days algorithm and the ISO-8601 parse/format is hand-rolled, so
//! the wire form is a pure function we can reproduce byte-for-byte in the TS
//! engine. The internal field layout is private to each engine; **byte-identity
//! is defined by the ISO-8601 string** (`format`) and the comparison order, not
//! by the representation.
//!
//! The ZONED variants carry a numeric UTC offset (`±HH:MM`/`Z`), never a named
//! zone, so they stay dependency-free; the offset is preserved for round-trip and
//! participates in identity/ordering (instant first, offset second).

use std::cmp::Ordering;

/// A calendar date with no time or zone: days since 1970-01-01 (proleptic
/// Gregorian). Ordered chronologically.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Date {
    /// Days since the Unix epoch (1970-01-01).
    pub days: i32,
}

/// A zone-less time of day (ISO "LOCAL TIME"): seconds since midnight plus a
/// sub-second nanosecond part. Ordered chronologically within the day.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Time {
    /// Seconds since midnight, 0..86_400.
    pub secs: u32,
    /// 0..1_000_000_000
    pub nanos: u32,
}

/// A zone-less datetime (ISO "LOCAL DATETIME"): seconds since 1970-01-01T00:00:00
/// plus a sub-second nanosecond part. Ordered chronologically.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DateTime {
    pub secs: i64,
    /// 0..1_000_000_000
    pub nanos: u32,
}

/// An ISO-8601 calendar duration. Months and days are kept SEPARATE from seconds
/// (a month is not a fixed number of seconds), matching the Cypher/GQL duration
/// model. Relationally unordered (like SQL intervals), but given a deterministic
/// TOTAL order for `ORDER BY` (lexicographic over the normalized components).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Duration {
    pub months: i64,
    pub days: i64,
    pub secs: i64,
    /// 0..1_000_000_000
    pub nanos: u32,
}

/// A datetime with a UTC offset (ISO "ZONED DATETIME" / "TIMESTAMP WITH TIME
/// ZONE"). Stored as the UTC instant plus the offset it was written in — the
/// offset is PRESERVED for round-trip rendering, and it participates in identity
/// and ordering (so `12:00Z` and `13:00+01:00` — same instant, different offset
/// — are ordered by instant first, offset second; a deliberate simplification of
/// ISO's inferred `=`-by-instant rule, kept for a consistent total/relational
/// model). Fields ordered `secs, nanos, offset` so the derived `Ord` is
/// instant-primary. ISO stores a numeric offset, never a named zone.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ZonedDateTime {
    /// UTC instant: seconds since 1970-01-01T00:00:00Z.
    pub secs: i64,
    /// 0..1_000_000_000
    pub nanos: u32,
    /// Offset from UTC in whole minutes (`Z` = 0), for round-trip rendering.
    pub offset: i16,
}

/// A time of day with a UTC offset (ISO "ZONED TIME" / "TIME WITH TIME ZONE").
/// Stored as the UTC seconds-of-day + the offset; ordered by UTC time-of-day then
/// offset. No date component. ISO uses a numeric offset, never a named zone.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ZonedTime {
    /// UTC seconds-of-day, 0..86_400 (the wall clock minus the offset, wrapped).
    pub secs: u32,
    /// 0..1_000_000_000
    pub nanos: u32,
    /// Offset from UTC in whole minutes, for round-trip rendering.
    pub offset: i16,
}

/// The temporal value family, carried as one `Value`/`Val`/`GVal` variant so each
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

/// Which temporal variant a value is — the discriminant only, no payload. Used by
/// the columnar store to key a homogeneous packed temporal column (a column holds
/// exactly one kind; a key that mixes kinds falls back to `Mixed`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TemporalKind {
    Date,
    Time,
    DateTime,
    ZonedTime,
    ZonedDateTime,
    Duration,
}

impl Temporal {
    /// The discriminant (no payload), for typing a packed temporal column.
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

    /// Total-ordered secondary-index key as `(kind_rank, i128)`. The `i128` mirrors
    /// [`TemporalCol::monotonic_key`](crate::graph::TemporalCol::monotonic_key) **bit
    /// for bit** (guarded by `temporal_index_key_matches_column`), so a value built
    /// from a stored column and one built from a query literal/param compare equal.
    /// The leading `kind_rank` keeps distinct temporal kinds in disjoint key ranges
    /// (the `i128` is only comparable *within* a kind — Date days vs DateTime
    /// secs<<32 don't share a scale), so a range seek never bleeds Date into DateTime.
    /// `None` for `Duration` — same exclusion as the column (no monotonic key).
    pub(crate) fn index_key(&self) -> Option<(u8, i128)> {
        const OFF_BIAS: i128 = 32_768; // offset (whole minutes, i16) → non-negative 0..=65535
        Some(match self {
            Self::Date(d) => (0, d.days as i128),
            Self::Time(t) => (1, ((t.secs as i128) << 32) | (t.nanos as i128)),
            Self::DateTime(dt) => (2, ((dt.secs as i128) << 32) | (dt.nanos as i128)),
            Self::ZonedTime(z) => (
                3,
                ((z.secs as i128) << 48)
                    | ((z.nanos as i128) << 16)
                    | ((z.offset as i128) + OFF_BIAS),
            ),
            Self::ZonedDateTime(z) => (
                4,
                ((z.secs as i128) << 48)
                    | ((z.nanos as i128) << 16)
                    | ((z.offset as i128) + OFF_BIAS),
            ),
            Self::Duration(_) => return None,
        })
    }
}

const SECS_PER_DAY: i64 = 86_400;

// --- civil calendar (Hinnant) ------------------------------------------------

/// Days since 1970-01-01 for a proleptic-Gregorian (y, m, d). `m` in 1..=12.
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Proleptic-Gregorian (y, m, d) for days since 1970-01-01.
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
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

// --- small parse helpers -----------------------------------------------------

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
    parse_int(&padded).map(|n| n as u32)
}

/// Render `nanos` as `.fraction` (trailing zeros trimmed), or empty if zero.
fn fmt_frac(nanos: u32) -> String {
    if nanos == 0 {
        return String::new();
    }
    let s = format!("{nanos:09}");
    format!(".{}", s.trim_end_matches('0'))
}

/// Zero-pad a year to at least 4 digits, matching the TS `padYear`. A NEGATIVE year
/// must render as `-0009` (sign, then 4 digits), NOT Rust's `{:04}` which counts the
/// sign inside the width and yields `-009`. Years ≥ 10000 render with their natural
/// width (no leading `+`), same as TS.
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
/// part before it and the offset in whole minutes. Errors if no offset is present
/// (a ZONED value requires one). Only the tail is inspected, so a date's `-`
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
        // `.get` (not `&s[..]`) so an offset window containing a non-ASCII byte —
        // e.g. `…+0é` — yields None and falls to the Err path below, rather than
        // panicking on a non-char-boundary slice (which under `panic = "abort"`
        // would abort the host; the TS engine returns an error here).
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

// --- Date --------------------------------------------------------------------

impl Date {
    /// Parse `YYYY-MM-DD`.
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut it = s.splitn(3, '-');
        // A leading '-' (negative year) is unusual; require the common form.
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

    pub fn format(&self) -> String {
        let (y, m, d) = civil_from_days(self.days as i64);
        format!("{}-{m:02}-{d:02}", fmt_year(y))
    }
}

// --- DateTime ----------------------------------------------------------------

impl DateTime {
    /// Parse `YYYY-MM-DDTHH:MM:SS[.fraction]` (also accepts a space separator).
    pub fn parse(s: &str) -> Result<Self, String> {
        let sep = s
            .find(['T', ' '])
            .ok_or_else(|| format!("datetime missing time part '{s}'"))?;
        let date = Date::parse(&s[..sep])?;
        let (tod, nanos) = parse_time(&s[sep + 1..])?;
        Ok(Self {
            secs: date.days as i64 * SECS_PER_DAY + tod,
            nanos,
        })
    }

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
    pub fn parse(s: &str) -> Result<Self, String> {
        let (tod, nanos) = parse_time(s)?;
        Ok(Self {
            secs: tod as u32,
            nanos,
        })
    }

    pub fn format(&self) -> String {
        let (h, m, s) = (self.secs / 3600, (self.secs % 3600) / 60, self.secs % 60);
        format!("{h:02}:{m:02}:{s:02}{}", fmt_frac(self.nanos))
    }
}

// --- ZonedDateTime -----------------------------------------------------------

impl ZonedDateTime {
    /// Parse `YYYY-MM-DDTHH:MM:SS[.frac](Z|±HH:MM)`. The wall clock is the
    /// pre-offset datetime; the stored instant is that minus the offset.
    pub fn parse(s: &str) -> Result<Self, String> {
        let (dt_str, offset) = split_offset(s)?;
        let local = DateTime::parse(dt_str)?;
        Ok(Self {
            secs: local.secs - i64::from(offset) * 60,
            nanos: local.nanos,
            offset,
        })
    }

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
    pub fn parse(s: &str) -> Result<Self, String> {
        let (t_str, offset) = split_offset(s)?;
        let (tod, nanos) = parse_time(t_str)?;
        let utc = (tod - i64::from(offset) * 60).rem_euclid(SECS_PER_DAY);
        Ok(Self {
            secs: utc as u32,
            nanos,
            offset,
        })
    }

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
    /// Parse ISO-8601 `PnYnMnWnDTnHnMnS` (years→months, weeks→days). Fractional
    /// seconds allowed on the seconds field.
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
                'Y' => {
                    months += take_num(&mut num, 'Y', s)? * 12;
                }
                'M' => {
                    months += take_num(&mut num, 'M', s)?;
                }
                'W' => {
                    days += take_num(&mut num, 'W', s)? * 7;
                }
                'D' => {
                    days += take_num(&mut num, 'D', s)?;
                }
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
}

/// Consume the pending numeric buffer as an integer for designator `d`.
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

// --- Temporal (the value-model variant) --------------------------------------

impl Temporal {
    /// The kind tag used by codecs and the value key: `date`/`datetime`/`duration`.
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

    /// The GraphSON v3 `@type` name (TinkerPop extended types).
    pub fn graphson_type(&self) -> &'static str {
        match self {
            Self::Date(_) => "gx:LocalDate",
            Self::Time(_) => "gx:LocalTime",
            Self::DateTime(_) => "gx:LocalDateTime",
            Self::ZonedTime(_) => "gx:OffsetTime",
            Self::ZonedDateTime(_) => "gx:OffsetDateTime",
            Self::Duration(_) => "gx:Duration",
        }
    }

    /// The GraphSON `@type` → kind tag, for decode. `None` if not temporal.
    pub fn graphson_tag(ty: &str) -> Option<&'static str> {
        match ty {
            "gx:LocalDate" => Some("date"),
            "gx:LocalTime" => Some("localtime"),
            "gx:LocalDateTime" => Some("datetime"),
            "gx:OffsetTime" => Some("zoned_time"),
            "gx:OffsetDateTime" => Some("zoned_datetime"),
            "gx:Duration" => Some("duration"),
            _ => None,
        }
    }

    /// The ISO-8601 string form (the byte-identity wire value).
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

    /// Compact tagged JSON, e.g. `{"@date":"2020-01-01"}` — the representation
    /// for our JSON-family codecs (ndjson/pg-json/query results). The ISO form
    /// never contains JSON-special characters, so no escaping is needed. (Its
    /// decode twin is [`Temporal::from_json_tag`].)
    pub fn json_tagged(&self) -> String {
        format!("{{\"@{}\":\"{}\"}}", self.tag(), self.format())
    }

    /// Decode from a `@date`/`@datetime`/`@duration` JSON object key + its string
    /// value (the single-key tagged form). `None` if the key isn't a temporal tag.
    /// Is `key` one of the tagged-temporal keys, independent of its value? Lets a
    /// caller decide an object's SHAPE before it commits to reading the value.
    pub fn is_json_tag(key: &str) -> bool {
        key.strip_prefix('@').is_some_and(|tag| {
            matches!(
                tag,
                "date" | "localtime" | "datetime" | "zoned_time" | "zoned_datetime" | "duration"
            )
        })
    }

    pub fn from_json_tag(key: &str, s: &str) -> Option<Result<Self, String>> {
        let tag = key.strip_prefix('@')?;
        matches!(
            tag,
            "date" | "localtime" | "datetime" | "zoned_time" | "zoned_datetime" | "duration"
        )
        .then(|| Self::parse(tag, s))
    }

    /// Build from a kind tag + ISO string (the codec decode path).
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

    /// Deterministic TOTAL order over all temporals (for `ORDER BY`/min/max):
    /// by kind, then chronologically within date/datetime, lexicographically
    /// within duration.
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

    /// Relational order for `< > <= >=`: date/datetime are instants (chronological);
    /// durations and cross-kind pairs are UNKNOWN (`None`) — SQL-interval-like,
    /// and consistent with the engine's partial relational / total sort split.
    pub fn rel_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Date(a), Self::Date(b)) => Some(a.cmp(b)),
            (Self::Time(a), Self::Time(b)) => Some(a.cmp(b)),
            (Self::DateTime(a), Self::DateTime(b)) => Some(a.cmp(b)),
            // Zoned instants compare within-kind (by UTC instant, then offset);
            // zoned-vs-local is a type mismatch → UNKNOWN, like the other pairs.
            (Self::ZonedTime(a), Self::ZonedTime(b)) => Some(a.cmp(b)),
            (Self::ZonedDateTime(a), Self::ZonedDateTime(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }
}

// --- calendar arithmetic -----------------------------------------------------

const NANOS_PER_SEC: i64 = 1_000_000_000;

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Days in month `m` (1..=12) of proleptic-Gregorian year `y`.
fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

impl Date {
    /// Add `months` (calendar), **clamping the day to the new month's length**
    /// (`Jan 31 + 1 month → Feb 28/29`), then `extra_days` as plain days. Returns
    /// `None` when the result falls outside the representable date range (a `Date`
    /// is `i32` days from 1970, ≈±5.88M years) instead of silently wrapping to a
    /// bogus year. This stays a pure range check; the arithmetic layer
    /// (`temporal_arith`) turns a `None` into a loud `E_DATA_EXCEPTION` (byte-
    /// identical to the TS engine), like duration overflow and division by zero.
    fn add_calendar(&self, months: i64, extra_days: i64) -> Option<Self> {
        let (y, m, d) = civil_from_days(self.days as i64);
        let total = y * 12 + (m as i64 - 1) + months;
        let ny = total.div_euclid(12);
        let nm = (total.rem_euclid(12) + 1) as u32;
        let nd = d.min(days_in_month(ny, nm));
        let base = days_from_civil(ny, i64::from(nm), i64::from(nd));
        i32::try_from(base + extra_days)
            .ok()
            .map(|days| Self { days })
    }
}

impl Duration {
    /// Negate the whole span, keeping `nanos` in `[0, 1e9)`.
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

    /// Component-wise sum of two durations (nanos carry into secs). Both are
    /// nominal — no anchoring — so months/days/secs simply add.
    /// A duration component at or beyond 2^53 isn't a JS *safe* integer, so the TS
    /// engine (f64) can't represent it exactly — it rounds — while native (i64)
    /// would keep an unrepresentable exact value or wrap on overflow. Both engines
    /// treat such a duration as **non-representable** (this returns `None`); the
    /// arithmetic layer turns that into a loud `E_DATA_EXCEPTION` (like out-of-range
    /// date arithmetic and division by zero), so they stay byte-identical instead of
    /// diverging (round vs wrap). The `>=` bound matches JS `Number.isSafeInteger`
    /// (|x| ≤ 2^53−1) exactly, so a value TS could round is rejected on both sides
    /// regardless of which engine parses/computes it.
    const MAX_SAFE: u64 = 1 << 53;

    /// `Some(self)` when every component is a JS-safe integer, else `None`
    /// (→ `E_DATA_EXCEPTION` at the arithmetic layer).
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

    /// Nominal add (no anchoring). `None` if the result overflows i64 or leaves the
    /// f64-exact range (→ null; see [`Self::representable`]).
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
            nanos: nanos as u32,
        }
        .representable()
    }

    /// Scale every component by an integer factor (nanos carry into secs). `None` on
    /// i64 overflow or leaving the f64-exact range (→ null).
    pub fn scale(&self, n: i64) -> Option<Self> {
        let total_nanos = i64::from(self.nanos).checked_mul(n)?;
        Some(Self {
            months: self.months.checked_mul(n)?,
            days: self.days.checked_mul(n)?,
            secs: self
                .secs
                .checked_mul(n)?
                .checked_add(total_nanos.div_euclid(NANOS_PER_SEC))?,
            nanos: total_nanos.rem_euclid(NANOS_PER_SEC) as u32,
        })
        .and_then(Self::representable)
    }
}

impl Temporal {
    /// `self + duration` for a date/datetime: apply calendar months (clamped),
    /// then days, then the time part (a `DATE` ignores the sub-day part — it has
    /// no time). `None` if `self` is a duration.
    pub fn add_duration(&self, d: &Duration) -> Option<Self> {
        match self {
            Self::Date(date) => date.add_calendar(d.months, d.days).map(Self::Date),
            // A bare time has no calendar part; the duration's time component wraps
            // within the 24h day (months/days are ignored).
            Self::Time(t) => {
                let carry_nanos = i64::from(t.nanos) + i64::from(d.nanos);
                let secs = i64::from(t.secs) + d.secs + carry_nanos.div_euclid(NANOS_PER_SEC);
                Some(Self::Time(Time {
                    secs: secs.rem_euclid(SECS_PER_DAY) as u32,
                    nanos: carry_nanos.rem_euclid(NANOS_PER_SEC) as u32,
                }))
            }
            Self::DateTime(dt) => {
                let days0 = dt.secs.div_euclid(SECS_PER_DAY);
                let tod = dt.secs.rem_euclid(SECS_PER_DAY);
                let date = Date {
                    days: i32::try_from(days0).ok()?,
                }
                .add_calendar(d.months, d.days)?;
                let mut secs = date.days as i64 * SECS_PER_DAY + tod + d.secs;
                let mut nanos = i64::from(dt.nanos) + i64::from(d.nanos);
                if nanos >= NANOS_PER_SEC {
                    nanos -= NANOS_PER_SEC;
                    secs += 1;
                }
                Some(Self::DateTime(DateTime {
                    secs,
                    nanos: nanos as u32,
                }))
            }
            // Zoned ± duration: apply it to the LOCAL wall clock (so calendar
            // months/days are correct in the value's own zone), then re-anchor to
            // the UTC instant, keeping the offset.
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
                    secs: local_secs as u32,
                    nanos: zt.nanos,
                };
                let Self::Time(nt) = Self::Time(local).add_duration(d)? else {
                    return None;
                };
                let utc = (i64::from(nt.secs) - i64::from(zt.offset) * 60).rem_euclid(SECS_PER_DAY);
                Some(Self::ZonedTime(ZonedTime {
                    secs: utc as u32,
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

    #[test]
    fn civil_round_trips_known_dates() {
        for &(y, m, d) in &[
            (1970, 1, 1),
            (2000, 1, 1),
            (2020, 2, 29), // leap day
            (1969, 12, 31),
            (1600, 12, 31),
            (2262, 4, 11),
        ] {
            let days = days_from_civil(y, m, d);
            assert_eq!(
                civil_from_days(days),
                (y, m as u32, d as u32),
                "{y}-{m}-{d}"
            );
        }
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1970, 1, 2), 1);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
    }

    #[test]
    fn negative_year_pads_to_four_digits_like_ts() {
        // A negative year renders as sign + 4 digits (`-0001`), matching the TS
        // `padYear` — NOT Rust's `{:04}`, which counts the sign inside the width and
        // yields `-001`. Found by the differential fuzzer via a date arithmetic that
        // crosses into a negative year (`date('0001-01-01') - duration('P1Y2M')`).
        let d = Date {
            days: i32::try_from(days_from_civil(-1, 11, 1)).unwrap(),
        };
        assert_eq!(d.format(), "-0001-11-01");
        let dt = DateTime {
            secs: days_from_civil(-1, 11, 1) * SECS_PER_DAY,
            nanos: 0,
        };
        assert_eq!(dt.format(), "-0001-11-01T00:00:00");
        // A single-digit negative year still gets 4 magnitude digits.
        let d9 = Date {
            days: i32::try_from(days_from_civil(-9, 6, 15)).unwrap(),
        };
        assert_eq!(d9.format(), "-0009-06-15");
        // Positive years are unchanged: 4-pad small, natural width for large.
        let d_small = Date {
            days: i32::try_from(days_from_civil(9, 1, 1)).unwrap(),
        };
        assert_eq!(d_small.format(), "0009-01-01");
    }

    #[test]
    fn date_arithmetic_out_of_range_is_none_not_wrapped() {
        // `DATE + huge DURATION` must not silently wrap the i32 day count to a
        // negative year — the pure calendar op yields None, which the arithmetic
        // layer (`temporal_arith`) turns into a loud `E_DATA_EXCEPTION` (see the
        // gql-level `date_arithmetic_overflow_raises_data_exception`). An in-range
        // shift still succeeds.
        let base = Temporal::Date(Date::parse("2020-01-01").unwrap());
        let ten_million_years = Duration {
            months: 10_000_000 * 12,
            days: 0,
            secs: 0,
            nanos: 0,
        };
        assert_eq!(base.add_duration(&ten_million_years), None);
        assert_eq!(base.add_duration(&ten_million_years.negate()), None);
        // ~5M years stays within i32 days (≈±5.88M years) and succeeds.
        let five_million_years = Duration {
            months: 5_000_000 * 12,
            days: 0,
            secs: 0,
            nanos: 0,
        };
        assert!(base.add_duration(&five_million_years).is_some());
        // DateTime + a huge duration is None too.
        let dt = Temporal::DateTime(DateTime::parse("2020-01-01T00:00:00").unwrap());
        assert_eq!(dt.add_duration(&ten_million_years), None);
    }

    #[test]
    fn date_parse_format_round_trip() {
        for s in ["1970-01-01", "2020-02-29", "1999-12-31", "2026-07-11"] {
            assert_eq!(Date::parse(s).unwrap().format(), s);
        }
        assert!(Date::parse("2020-13-01").is_err());
        assert!(Date::parse("not-a-date").is_err());
    }

    #[test]
    fn datetime_parse_format_round_trip() {
        for s in [
            "2020-01-01T00:00:00",
            "2026-07-11T13:45:06",
            "2020-01-01T10:15:30.5",
            "1969-12-31T23:59:59", // pre-epoch time-of-day stays in range
        ] {
            assert_eq!(DateTime::parse(s).unwrap().format(), s);
        }
        // space separator accepted on input, normalized to 'T'.
        assert_eq!(
            DateTime::parse("2020-01-01 10:15:30").unwrap().format(),
            "2020-01-01T10:15:30"
        );
    }

    #[test]
    fn zoned_parse_format_round_trip_and_instant() {
        // Offset and `Z` both round-trip byte-for-byte.
        for s in [
            "2020-01-01T12:00:00+05:00",
            "2020-01-01T12:00:00Z",
            "2020-06-15T08:30:00.25-08:00",
            "2020-01-01T02:00:00+05:30",
        ] {
            assert_eq!(ZonedDateTime::parse(s).unwrap().format(), s);
        }
        for s in ["12:20:02+08:00", "03:02:11.7-06:00", "00:00:00Z"] {
            assert_eq!(ZonedTime::parse(s).unwrap().format(), s);
        }
        // Same instant, different offset → equal UTC secs (compare-by-instant core),
        // but distinct values (offset is a tiebreaker in the total order).
        let a = ZonedDateTime::parse("2020-01-01T12:00:00Z").unwrap();
        let b = ZonedDateTime::parse("2020-01-01T13:00:00+01:00").unwrap();
        assert_eq!(a.secs, b.secs, "same UTC instant");
        assert_ne!(a, b, "different offset → distinct value");
        assert!(a < b, "instant-equal, ordered by offset (+00 < +01)");
        // A later instant sorts after, regardless of wall-clock/offset.
        let later = ZonedDateTime::parse("2020-01-01T12:00:01Z").unwrap();
        assert!(a < later);
    }

    #[test]
    fn duration_parse_normalizes_and_round_trips() {
        // years→months, weeks→days on input; canonical uses total M/D and T…S.
        assert_eq!(
            Duration::parse("P1Y2M3W4DT5H6M7S").unwrap().format(),
            "P14M25DT18367S"
        );
        assert_eq!(Duration::parse("P1Y").unwrap().format(), "P12M");
        assert_eq!(Duration::parse("PT0S").unwrap().format(), "PT0S");
        assert_eq!(Duration::parse("P0D").unwrap().format(), "PT0S");
        assert_eq!(Duration::parse("PT1.5S").unwrap().format(), "PT1.5S");
        // canonical output re-parses to itself.
        let canon = Duration::parse("P14M25DT18367S").unwrap();
        assert_eq!(Duration::parse(&canon.format()).unwrap(), canon);
        assert!(Duration::parse("1Y").is_err());
    }

    #[test]
    fn ordering_is_deterministic() {
        let d1 = Temporal::Date(Date::parse("2020-01-01").unwrap());
        let d2 = Temporal::Date(Date::parse("2020-06-01").unwrap());
        assert_eq!(d1.rel_cmp(&d2), Some(Ordering::Less));
        assert_eq!(d1.cmp_total(&d2), Ordering::Less);

        let t1 = Temporal::DateTime(DateTime::parse("2020-01-01T00:00:00").unwrap());
        // cross-kind: relationally UNKNOWN, but a deterministic total order.
        assert_eq!(d1.rel_cmp(&t1), None);
        assert_eq!(d1.cmp_total(&t1), Ordering::Less); // date kind-rank < datetime

        let du = Temporal::Duration(Duration::parse("P1M").unwrap());
        assert_eq!(du.rel_cmp(&du), None); // durations not relationally ordered
        assert_eq!(du.cmp_total(&du), Ordering::Equal);
        assert_eq!(t1.cmp_total(&du), Ordering::Less); // datetime kind-rank < duration
    }

    #[test]
    fn zoned_offset_with_non_ascii_byte_errors_not_panics() {
        // A non-ASCII byte in the trailing offset window must surface an error, not
        // panic on a non-char-boundary slice (which under panic=abort aborts the
        // host). `é` is two bytes, so the offset window `+0éxy` crosses a char.
        assert!(ZonedDateTime::parse("2020-01-01T00:00:00+0éxy").is_err());
        assert!(ZonedTime::parse("00:00:00+0é:0").is_err());
        // A well-formed offset still parses.
        assert!(ZonedDateTime::parse("2020-01-01T00:00:00+05:30").is_ok());
    }
}
