//! Robustness fuzzing: feed seeded-random and mutated strings to every parser and
//! codec-decoder, asserting they NEVER panic (only Ok/Err). The release cdylib is
//! built `panic = "abort"` with no `catch_unwind` on the FFI boundary, so a panic on
//! malformed input aborts the host process — exactly the R1/R2/R3 char-boundary bugs
//! this round found. Under the test profile a panic unwinds into a test failure, so
//! this pins "no input can crash a parser".
//!
//! Seed: random each run so every run explores fresh inputs (fuzzing is discovery —
//! the specific panics found are pinned as their own deterministic unit tests). The
//! seed is printed at the start and cargo shows captured output on failure, so a
//! crash is reproducible: re-run with `FUZZ_SEED=<n>` (decimal or 0x-hex). This is
//! the property-based-testing convention: random by default, seed on failure.

/// The base seed for a run: `FUZZ_SEED` if set (decimal or `0x…` hex), else derived
/// from the wall clock. Never 0 — xorshift needs nonzero state. (`SystemTime` is
/// fine in a normal `cargo test`; only Workflow scripts forbid the wall clock.)
pub(crate) fn fuzz_seed() -> u64 {
    if let Ok(s) = std::env::var("FUZZ_SEED") {
        let s = s.trim();
        let parsed = s.strip_prefix("0x").map_or_else(
            || s.parse::<u64>().ok(),
            |h| u64::from_str_radix(h, 16).ok(),
        );
        if let Some(v) = parsed {
            return v | 1;
        }
    }
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0x1234_5678_9abc_def1, |d| d.as_nanos() as u64);
    nanos | 1
}

// A tiny deterministic PRNG (xorshift64*), so a run replays exactly from its seed.
pub(crate) struct Rng(pub u64);
impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

// An alphabet mixing structural parser chars, temporal/query keywords, escape
// triggers, control chars, and multi-byte UTF-8 (the char-boundary trap).
const ALPHABET: &[&str] = &[
    "a", "Z", "0", "9", " ", "\"", "'", "\\", "/", "[", "]", "{", "}", "(", ")", ":", ";", ",",
    ".", "+", "-", "*", "%", "|", "&", "<", ">", "=", "~", "@", "$", "!", "`", "\n", "\t", "\u{0}",
    "u", "U", "x", "e", "E", "T", "Z", "P", "Y", "M", "D", "H", "S", "W", "g", "V", "é", "中",
    "😀", "\u{e000}", "ñ", "🎉", "\u{feff}",
];

fn random_string(rng: &mut Rng, max_len: usize) -> String {
    let len = rng.below(max_len + 1);
    (0..len)
        .map(|_| ALPHABET[rng.below(ALPHABET.len())])
        .collect()
}

// Run one string through EVERY parser/decoder. A panic in any of these unwinds
// into a test failure (release: it would abort the host).
fn exercise(s: &str) {
    let _ = crate::gql::parse(s);
    let _ = crate::gql::lexer::tokenize(s);
    let _ = crate::gremlin::parse(s);
    let _ = crate::temporal::Date::parse(s);
    let _ = crate::temporal::Time::parse(s);
    let _ = crate::temporal::DateTime::parse(s);
    let _ = crate::temporal::ZonedTime::parse(s);
    let _ = crate::temporal::ZonedDateTime::parse(s);
    let _ = crate::temporal::Duration::parse(s);
    let _ = crate::ndjson::decode(s);
    let _ = crate::codec::csv::decode(s);
}

#[test]
fn fuzz_random_strings_never_panic() {
    let seed = fuzz_seed();
    eprintln!("fuzz_random_strings_never_panic: FUZZ_SEED={seed} (0x{seed:x}) to reproduce");
    let mut rng = Rng(seed);
    for _ in 0..40_000 {
        let s = random_string(&mut rng, 48);
        exercise(&s);
    }
}

// Mutate mostly-valid templates: insert a random char (often multi-byte) at a
// random CHAR position, and/or splice a byte-truncation-adjacent form. This is
// what catches char-boundary slicing — a valid prefix with a multi-byte char
// landing where the parser computed an offset assuming ASCII (R1 temporal
// offset, R3 lexer \u escape).
#[test]
fn fuzz_template_mutations_never_panic() {
    const TEMPLATES: &[&str] = &[
            "2020-01-01T00:00:00+05:30",
            "2020-01-01",
            "12:00:00Z",
            "2020-01-01T00:00:00",
            "P1Y2M3DT4H5M6S",
            "PT-2.5S",
            "RETURN \"\\u0041\"",
            "RETURN 1 + 2",
            "MATCH (n:T) RETURN n.x",
            "g.V('1').out('E').values('n')",
            "{\"type\":\"node\",\"id\":\"1\",\"labels\":[\"T\"],\"properties\":{\"d\":{\"@date\":\"2020-01-01\"}}}",
        ];
    // Print the BASE seed (what you pass to FUZZ_SEED); the RNG xors a constant so
    // this test explores a different stream than the random-string test under the
    // same base seed.
    let base = fuzz_seed();
    eprintln!("fuzz_template_mutations_never_panic: FUZZ_SEED={base} (0x{base:x}) to reproduce");
    let mut rng = Rng(base ^ 0x5555_5555_5555_5555);
    for _ in 0..40_000 {
        let template = TEMPLATES[rng.below(TEMPLATES.len())];
        let mut chars: Vec<char> = template.chars().collect();
        // 1-3 random single-char insertions at random char positions (keeps the
        // string valid UTF-8 but shifts byte offsets past ASCII assumptions).
        for _ in 0..=rng.below(3) {
            let ins = ALPHABET[rng.below(ALPHABET.len())]
                .chars()
                .next()
                .unwrap_or('x');
            let pos = rng.below(chars.len() + 1);
            chars.insert(pos, ins);
        }
        // Sometimes truncate to a random char length (a cut-short valid prefix).
        if rng.below(2) == 0 && !chars.is_empty() {
            chars.truncate(rng.below(chars.len()));
        }
        let mutated: String = chars.into_iter().collect();
        exercise(&mutated);
    }
}
