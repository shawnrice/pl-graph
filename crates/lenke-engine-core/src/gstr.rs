//! `GStr` — a compact owned string ("German string" / Umbra layout), the storage- and
//! value-layer replacement for `Arc<str>`.
//!
//! 16 bytes on every target: a `u32` length, a 4-byte prefix, and an 8-byte tail. A
//! string ≤ 12 bytes lives ENTIRELY INLINE (prefix + tail hold the bytes) — no heap
//! allocation at all, which is the common case for graph property values (labels,
//! codes, statuses, short names). A longer string keeps its first 4 bytes in the prefix
//! (for a fast comparison reject) and an owning, atomically-refcounted pointer in the
//! tail; it is freed when the last `GStr` drops, so it is safe under arbitrary
//! edit/delete workloads (unlike an append-only arena).
//!
//! All `unsafe` is contained here; the public API is a safe drop-in for `Arc<str>`
//! (`Deref<Target = str>`, `Borrow<str>`, `Hash`/`Eq`/`Ord` byte-identical to `str`,
//! `Clone`, `From<&str>`). Ordering and hashing delegate to the string bytes, so a
//! `GStr` sorts and hashes exactly like the `&str` it holds — the cross-engine
//! byte-identity contract is unchanged.
//!
//! The long-string pointer is stored via exposed provenance (`expose_provenance` /
//! `with_exposed_provenance`) because the tail bytes double as inline data or a pointer
//! — the union trick every small-string type relies on. The module is verified UB-free
//! under Miri's default (exposed-provenance) model; `-Zmiri-strict-provenance` cannot
//! apply to any int-as-pointer layout and is not a goal here.

use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::atomic::{fence, AtomicUsize, Ordering as AtomicOrdering};

/// Longest string stored inline (the 4-byte prefix + 8-byte tail).
const INLINE_CAP: usize = 12;
/// Size of the refcount header prepended to a heap-allocated long string.
const HDR: usize = std::mem::size_of::<AtomicUsize>();

#[repr(C, align(8))]
pub struct GStr {
    len: u32,
    /// Long: the string's first 4 bytes. Inline: bytes `[0..4]` (zero-padded).
    prefix: [u8; 4],
    /// Long: the block pointer (as an integer, for a fixed 16-byte size on 32-bit
    /// targets too). Inline: bytes `[4..12]` (zero-padded).
    tail: u64,
}

impl GStr {
    #[must_use]
    pub fn new(s: &str) -> Self {
        let b = s.as_bytes();
        let len = b.len();
        if len <= INLINE_CAP {
            let mut prefix = [0u8; 4];
            let mut tail = [0u8; 8];
            let head = len.min(4);
            prefix[..head].copy_from_slice(&b[..head]);
            if len > 4 {
                tail[..len - 4].copy_from_slice(&b[4..len]);
            }
            Self {
                len: len as u32,
                prefix,
                tail: u64::from_ne_bytes(tail),
            }
        } else {
            let layout = Self::heap_layout(len);
            // SAFETY: `layout` reserves the refcount header plus exactly `len` bytes. We
            // initialise the refcount to 1 and copy `len` valid bytes into the data
            // region; nothing else reads the block until then.
            unsafe {
                let block = alloc(layout);
                if block.is_null() {
                    handle_alloc_error(layout);
                }
                (block as *mut AtomicUsize).write(AtomicUsize::new(1));
                std::ptr::copy_nonoverlapping(b.as_ptr(), block.add(HDR), len);
                let mut prefix = [0u8; 4];
                prefix.copy_from_slice(&b[..4]);
                // Store the address with its provenance exposed, so the read side can
                // reconstruct a valid pointer (strict-provenance clean).
                Self {
                    len: len as u32,
                    prefix,
                    tail: block.expose_provenance() as u64,
                }
            }
        }
    }

    #[inline]
    fn heap_layout(len: usize) -> Layout {
        Layout::from_size_align(HDR + len, std::mem::align_of::<AtomicUsize>())
            .expect("string length fits the address space")
    }

    #[inline]
    fn is_inline(&self) -> bool {
        self.len as usize <= INLINE_CAP
    }

    /// The heap block pointer for a long string, reconstructed from the exposed
    /// address. Only valid when `!is_inline()`.
    #[inline]
    fn heap_ptr(&self) -> *mut u8 {
        std::ptr::with_exposed_provenance_mut(self.tail as usize)
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        let len = self.len as usize;
        if len <= INLINE_CAP {
            // The 12 inline bytes are contiguous at struct offset 4 (after `len`), by
            // `repr(C)`. SAFETY: reading this struct's own initialised inline storage,
            // bounded by `len ≤ 12`.
            unsafe { std::slice::from_raw_parts((self as *const Self as *const u8).add(4), len) }
        } else {
            // SAFETY: `tail` is a live block from `new` that we hold a refcount on; the
            // string data begins just past the header and is `len` bytes long.
            unsafe { std::slice::from_raw_parts(self.heap_ptr().add(HDR), len) }
        }
    }

    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        // SAFETY: a `GStr` is only ever built from a valid `&str`, and its bytes are
        // never mutated in place, so they remain valid UTF-8.
        unsafe { std::str::from_utf8_unchecked(self.as_bytes()) }
    }
}

impl Clone for GStr {
    fn clone(&self) -> Self {
        if !self.is_inline() {
            // SAFETY: `tail` points at a live refcount we currently hold a share of.
            unsafe {
                (*(self.heap_ptr() as *const AtomicUsize)).fetch_add(1, AtomicOrdering::Relaxed)
            };
        }
        Self {
            len: self.len,
            prefix: self.prefix,
            tail: self.tail,
        }
    }
}

impl Drop for GStr {
    fn drop(&mut self) {
        if !self.is_inline() {
            // SAFETY: standard atomic refcount release; the last owner frees the block it
            // allocated with the matching layout (Arc's exact pattern).
            unsafe {
                let rc = &*(self.heap_ptr() as *const AtomicUsize);
                if rc.fetch_sub(1, AtomicOrdering::Release) == 1 {
                    fence(AtomicOrdering::Acquire);
                    dealloc(self.heap_ptr(), Self::heap_layout(self.len as usize));
                }
            }
        }
    }
}

// The bytes are immutable and the refcount is atomic, so sharing across threads is safe.
unsafe impl Send for GStr {}
unsafe impl Sync for GStr {}

impl Deref for GStr {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}
impl AsRef<str> for GStr {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}
impl Borrow<str> for GStr {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq for GStr {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Reject on length and the 4-byte prefix (a cheap early-out, and for long
        // strings it avoids a deref), then confirm on the full bytes — the prefix alone
        // is NOT sufficient for an inline string, whose bytes run past it.
        self.len == other.len && self.prefix == other.prefix && self.as_bytes() == other.as_bytes()
    }
}
impl Eq for GStr {}
impl PartialEq<str> for GStr {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

// Ordering and hashing MUST match `str` exactly (byte-lexicographic), because they are
// part of the cross-engine result/order contract. Delegate to the bytes — no `u32`
// prefix shortcut, which would order by native-endian integer, not lexicographically.
impl Ord for GStr {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}
impl PartialOrd for GStr {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Hash for GStr {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Delegate to `str`'s Hash so `Borrow<str>` lookups (probe a `GStr` map by
        // `&str`) find the key — identical hashing is the Borrow contract.
        self.as_str().hash(state);
    }
}

impl From<&str> for GStr {
    #[inline]
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}
impl From<String> for GStr {
    #[inline]
    fn from(s: String) -> Self {
        Self::new(&s)
    }
}
impl From<&String> for GStr {
    #[inline]
    fn from(s: &String) -> Self {
        Self::new(s)
    }
}
impl From<std::sync::Arc<str>> for GStr {
    #[inline]
    fn from(s: std::sync::Arc<str>) -> Self {
        Self::new(&s)
    }
}
impl std::fmt::Debug for GStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.as_str(), f)
    }
}
impl std::fmt::Display for GStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn h(x: impl Hash) -> u64 {
        let mut s = DefaultHasher::new();
        x.hash(&mut s);
        s.finish()
    }

    #[test]
    fn layout_is_16_bytes() {
        assert_eq!(std::mem::size_of::<GStr>(), 16);
        assert_eq!(std::mem::align_of::<GStr>(), 8);
    }

    #[test]
    fn roundtrips_every_length_class() {
        for s in [
            "",
            "a",
            "abcd",          // == 4 (prefix boundary)
            "abcdefghijkl",  // == 12 (inline boundary)
            "abcdefghijklm", // == 13 (first heap)
            "category-value-0001234",
            "π≈3.14159 — with unicode 🎯 and more length here",
            &"x".repeat(5000),
        ] {
            let g = GStr::new(s);
            assert_eq!(g.as_str(), s);
            assert_eq!(g.len(), s.len());
            assert_eq!(&*g, s); // Deref
        }
    }

    #[test]
    fn clone_and_drop_keep_the_value_alive() {
        let g = GStr::new("a reasonably long heap-backed string value");
        let c1 = g.clone();
        let c2 = c1.clone();
        drop(c1);
        assert_eq!(c2.as_str(), "a reasonably long heap-backed string value");
        assert_eq!(g.as_str(), "a reasonably long heap-backed string value");
        drop(g);
        assert_eq!(c2.as_str(), "a reasonably long heap-backed string value");
    }

    #[test]
    fn eq_and_ord_match_str_bytewise() {
        let samples = [
            "",
            "a",
            "ab",
            "abc",
            "abcd",
            "abce",
            "abcdefghijkl",
            "abcdefghijklm",
            "abcdefghijklZ",
            "zzz",
            "z",
            // Same length AND same 4-byte prefix, differing only PAST the prefix — the
            // inline case a prefix-only compare would wrongly call equal.
            "key0005",
            "key0006",
            "key0999",
        ];
        for &a in &samples {
            for &b in &samples {
                assert_eq!(GStr::new(a) == GStr::new(b), a == b, "eq {a:?} {b:?}");
                assert_eq!(GStr::new(a).cmp(&GStr::new(b)), a.cmp(b), "ord {a:?} {b:?}");
            }
        }
    }

    #[test]
    fn hash_matches_str_so_borrow_lookup_works() {
        use std::collections::HashMap;
        for s in [
            "",
            "short",
            "abcdefghijkl",
            "a much longer heap-backed key value",
        ] {
            assert_eq!(h(GStr::new(s)), h(s), "hash {s:?}");
        }
        let mut m: HashMap<GStr, u32> = HashMap::new();
        m.insert(GStr::new("status"), 1);
        m.insert(
            GStr::new("a long heap-backed key that is definitely not inline"),
            2,
        );
        // probe by &str via Borrow<str>
        assert_eq!(m.get("status"), Some(&1));
        assert_eq!(
            m.get("a long heap-backed key that is definitely not inline"),
            Some(&2)
        );
        assert_eq!(m.get("missing"), None);
    }
}
