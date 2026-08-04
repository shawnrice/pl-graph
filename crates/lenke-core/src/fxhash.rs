//! One fast hash for both engines.
//!
//! FxHash — the hasher rustc uses — for the internal grouping and dedup maps,
//! where the default SipHash dominates: a short key hashed per row, and
//! SipHash's ~40 ns there is the wall. It processes 8-byte words with a
//! multiply-rotate-xor, ~3-4x faster on these keys, and needs no dependency.
//! These maps are internal — never keyed by untrusted external data in a way a
//! hash-flood would matter — so the DoS resistance SipHash buys is not needed.
//!
//! This lived inside `gql::eval`, private, while the Gremlin side used
//! `std::collections`' SipHash for the same job: deduplicating a column and
//! tallying a `groupCount`. That was 3x and 8x on the identical graph question,
//! and neither was a difference in what the two engines DO.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// A fast, non-cryptographic hasher (FxHash — the one rustc uses) for the internal
/// grouping / dedup maps, where the default SipHash dominates: `GROUP BY <node>`
/// over a big result hashes a short key per row, and SipHash's ~40 ns there is the
/// wall. FxHash processes 8-byte words with a multiply-rotate-xor, ~3–4× faster on
/// these keys, and needs no dependency. These maps are internal (never keyed by
/// untrusted external data in a way that a hash-flood would matter), so the DoS
/// resistance SipHash buys is not needed here.
#[derive(Default)]
pub(crate) struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        // FxHash's raw accumulator has weak high-bit avalanche, so structured keys
        // (`@v0`, `@v1`, … — a common prefix + a small varying suffix) cluster and
        // the map probes more. A splitmix64 finalize (3 mul-xor-shift, once per
        // hash) fully mixes it — restoring good distribution while keeping the fast
        // per-word write. Without this, FxHash was *slower* than SipHash here.
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
        let mut h = self.0;
        while bytes.len() >= 8 {
            let w = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            h = (h.rotate_left(5) ^ w).wrapping_mul(SEED);
            bytes = &bytes[8..];
        }
        if !bytes.is_empty() {
            let mut w = 0u64;
            for (i, &b) in bytes.iter().enumerate() {
                w |= (b as u64) << (i * 8);
            }
            h = (h.rotate_left(5) ^ w).wrapping_mul(SEED);
        }
        self.0 = h;
    }
    #[inline]
    fn write_u64(&mut self, w: u64) {
        const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(SEED);
    }
}

pub(crate) type FxBuild = BuildHasherDefault<FxHasher>;
pub(crate) type FxHashMap<K, V> = HashMap<K, V, FxBuild>;
pub(crate) type FxHashSet<K> = HashSet<K, FxBuild>;
