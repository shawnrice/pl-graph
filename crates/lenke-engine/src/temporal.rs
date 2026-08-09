//! ISO/IEC 39075 temporal values — this engine's zone-less subset: `DATE`,
//! `LOCAL TIME`, `LOCAL DATETIME`. Ported faithfully from `lenke-core`'s
//! `temporal.rs` so the two engines AGREE (same ISO-8601 wire form, same order);
//! the ZONED variants and `DURATION` join in a later slice.
//!
//! Dependency-free: the calendar math is Howard Hinnant's civil-from-days
//! algorithm and the ISO-8601 parse/format is hand-rolled, so the wire form is a
//! pure function. Byte-identity is defined by the ISO-8601 string ([`format`],
//! [`Temporal::format`]) and the comparison order, not by the field layout.

use std::cmp::Ordering;

/// A calendar date: days since 1970-01-01 (proleptic Gregorian), ordered
/// chronologically. `days` is an `i32`, matching lenke-core (parse overflows the
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

const SECS_PER_DAY: i64 = 86_400;

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

// --- the value family --------------------------------------------------------

/// The temporal value family, carried as one `Value::Temporal` variant so each
/// exhaustive match gains a single arm. Zone-less subset for now.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Temporal {
    Date(Date),
    Time(Time),
    DateTime(DateTime),
}

impl Temporal {
    /// The kind tag used by codecs and the value key.
    #[must_use]
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Date(_) => "date",
            Self::Time(_) => "localtime",
            Self::DateTime(_) => "datetime",
        }
    }

    /// The ISO-8601 string form (the byte-identity wire value).
    #[must_use]
    pub fn format(&self) -> String {
        match self {
            Self::Date(d) => d.format(),
            Self::Time(t) => t.format(),
            Self::DateTime(dt) => dt.format(),
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
            _ => Err(format!("unknown temporal kind '{tag}'")),
        }
    }

    /// Kind rank for the cross-kind total order (date < localtime < datetime).
    fn kind_rank(&self) -> u8 {
        match self {
            Self::Date(_) => 0,
            Self::Time(_) => 1,
            Self::DateTime(_) => 2,
        }
    }

    /// Deterministic TOTAL order over all temporals (for `ORDER BY`/min/max): by
    /// kind, then chronologically within a kind.
    #[must_use]
    pub fn cmp_total(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Date(a), Self::Date(b)) => a.cmp(b),
            (Self::Time(a), Self::Time(b)) => a.cmp(b),
            (Self::DateTime(a), Self::DateTime(b)) => a.cmp(b),
            _ => self.kind_rank().cmp(&other.kind_rank()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn chronological_order_and_cross_kind_rank() {
        let d1 = Temporal::Date(Date::parse("2024-01-01").unwrap());
        let d2 = Temporal::Date(Date::parse("2024-06-01").unwrap());
        assert_eq!(d1.cmp_total(&d2), Ordering::Less);
        // Cross-kind falls back to the kind rank: date < datetime.
        let dt = Temporal::DateTime(DateTime::parse("1900-01-01T00:00:00").unwrap());
        assert_eq!(d1.cmp_total(&dt), Ordering::Less);
    }
}
