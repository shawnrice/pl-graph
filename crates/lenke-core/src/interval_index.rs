//! Relational Interval Tree (RI-tree, Kriegel/Pötke/Seidl) — interval **stabbing**
//! (which stored `[lo, hi]` intervals contain a query point `q`) answered on top of
//! ordinary sorted maps, so it inherits incremental B-tree maintenance and never needs
//! a rebuild. Used as an optional interval index over a temporal `[vf, vt)` pair to make
//! an as-of predicate (`vf <= v AND vt > v`) an `O(log² + k)` seek instead of a scan or a
//! materialize-and-intersect.
//!
//! A *virtual* complete binary tree spans the whole (biased) `u128` key domain — it is
//! never stored, only walked arithmetically. Each interval registers at its **fork
//! node**: the highest virtual node whose value lies inside `[lo, hi]`. Two sorted maps
//! hold the registrations — `lower` keyed by `(fork, lo)` and `upper` by `(fork, hi)`.
//! A stab at `q` walks the root→`q` path (~128 nodes); at each path node `v` it emits, in
//! one range scan, the registrations that must contain `q` (`hi >= q` when `q >= v`,
//! else `lo <= q`).

use std::collections::BTreeMap;

/// Order-preserving `i128` → `u128` (flip the sign bit) so the RI-tree's positive-domain
/// arithmetic works on the signed monotonic keys, negative dates included.
#[inline]
fn bias(k: i128) -> u128 {
    (k as u128) ^ (1u128 << 127)
}

/// The virtual tree's root and the initial half-step (one below the root).
const ROOT: u128 = 1u128 << 127;
const ROOT_STEP: u128 = 1u128 << 126;

/// The fork node of `[lo, hi]`: the highest virtual node whose value is inside the
/// interval. Descend from the root, stepping toward the interval, until a node lands
/// inside it (a non-empty `[lo, hi]` always contains a node — at worst a leaf).
fn fork(lo: u128, hi: u128) -> u128 {
    let mut v = ROOT;
    let mut step = ROOT_STEP;
    loop {
        if hi < v {
            v -= step; // whole interval is left of v
        } else if lo > v {
            v += step; // whole interval is right of v
        } else {
            return v; // v ∈ [lo, hi]
        }
        if step == 0 {
            return v; // leaf reached
        }
        step >>= 1;
    }
}

#[derive(Default, Clone)]
pub(crate) struct RiTree {
    lower: BTreeMap<(u128, u128), Vec<u32>>, // (fork, lo) -> ids
    upper: BTreeMap<(u128, u128), Vec<u32>>, // (fork, hi) -> ids
    len: usize,
}

impl RiTree {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Register interval `[lo, hi]` for element `id`. `lo <= hi` assumed.
    pub(crate) fn insert(&mut self, lo: i128, hi: i128, id: u32) {
        let (lo, hi) = (bias(lo), bias(hi));
        let f = fork(lo, hi);
        self.lower.entry((f, lo)).or_default().push(id);
        self.upper.entry((f, hi)).or_default().push(id);
        self.len += 1;
    }

    /// Deregister a previously inserted `[lo, hi] / id` (one occurrence).
    pub(crate) fn remove(&mut self, lo: i128, hi: i128, id: u32) {
        let (lo, hi) = (bias(lo), bias(hi));
        let f = fork(lo, hi);
        for (map, key) in [(&mut self.lower, (f, lo)), (&mut self.upper, (f, hi))] {
            if let Some(v) = map.get_mut(&key) {
                if let Some(pos) = v.iter().position(|&x| x == id) {
                    v.swap_remove(pos);
                }
                if v.is_empty() {
                    map.remove(&key);
                }
            }
        }
        self.len -= 1;
    }

    /// Walk the stab of point `q`, calling `visit(ids)` for each matching bucket.
    fn stab_visit(&self, q: i128, mut visit: impl FnMut(&[u32])) {
        use std::ops::Bound::Included;
        let q = bias(q);
        let mut v = ROOT;
        let mut step = ROOT_STEP;
        loop {
            if q >= v {
                // lo <= v <= q for anything registered here, so it contains q iff hi >= q.
                for (_, ids) in self
                    .upper
                    .range((Included((v, q)), Included((v, u128::MAX))))
                {
                    visit(ids);
                }
                if step == 0 {
                    break;
                }
                v += step;
            } else {
                // hi >= v > q for anything registered here, so it contains q iff lo <= q.
                for (_, ids) in self.lower.range((Included((v, 0)), Included((v, q)))) {
                    visit(ids);
                }
                if step == 0 {
                    break;
                }
                v -= step;
            }
            step >>= 1;
        }
    }

    /// All ids whose interval contains the point `q` (`lo <= q <= hi`). For a half-open
    /// as-of `[vf, vt)` the caller passes `hi = vt - 1` so the seek stays a superset that
    /// the final `WHERE vt > v` verifies (and `vf <= v` too).
    pub(crate) fn stab(&self, q: i128) -> Vec<u32> {
        let mut out = Vec::new();
        self.stab_visit(q, |ids| out.extend_from_slice(ids));
        out
    }

    /// The stab result SIZE, without materializing the ids — sums bucket lengths in one
    /// walk (O(buckets on the path), not O(result)). Lets a multi-index seek compare
    /// axes' selectivity CHEAPLY and seed from the smallest, never materializing a
    /// non-selective stab (e.g. "everything believed now").
    pub(crate) fn stab_len(&self, q: i128) -> usize {
        let mut n = 0;
        self.stab_visit(q, |ids| n += ids.len());
        n
    }

    /// All ids whose interval `[lo, hi]` overlaps the closed query range `[d1, d2]`
    /// (`lo <= d2 AND hi >= d1`). A superset for a half-open `lo < d2 AND hi > d1`
    /// predicate, which the caller's `WHERE` then verifies. Three disjoint fork
    /// classes (an interval registers at exactly one fork, so no double-count):
    ///   • fork ∈ [d1, d2]  — the fork itself is in range ⇒ the interval overlaps.
    ///   • fork < d1        — on the root→d1 path (it must contain d1); overlaps iff hi >= d1.
    ///   • fork > d2        — on the root→d2 path (it must contain d2); overlaps iff lo <= d2.
    fn overlap_visit(&self, d1: i128, d2: i128, mut visit: impl FnMut(&[u32])) {
        use std::ops::Bound::Included;
        let (a, b) = (bias(d1), bias(d2));
        if a > b {
            return;
        }
        // Inner: every interval whose fork ∈ [a, b] (one `lower` entry each).
        for (_, ids) in self
            .lower
            .range((Included((a, 0)), Included((b, u128::MAX))))
        {
            visit(ids);
        }
        // Left boundary: nodes v < a on the root→a path — overlaps iff hi >= a.
        let (mut v, mut step) = (ROOT, ROOT_STEP);
        loop {
            if v < a {
                for (_, ids) in self
                    .upper
                    .range((Included((v, a)), Included((v, u128::MAX))))
                {
                    visit(ids);
                }
            }
            if step == 0 {
                break;
            }
            v = if a < v { v - step } else { v + step };
            step >>= 1;
        }
        // Right boundary: nodes v > b on the root→b path — overlaps iff lo <= b.
        let (mut v, mut step) = (ROOT, ROOT_STEP);
        loop {
            if v > b {
                for (_, ids) in self.lower.range((Included((v, 0)), Included((v, b)))) {
                    visit(ids);
                }
            }
            if step == 0 {
                break;
            }
            v = if b < v { v - step } else { v + step };
            step >>= 1;
        }
    }

    /// All ids whose interval `[lo, hi]` overlaps `[d1, d2]` (superset the WHERE verifies).
    pub(crate) fn overlap(&self, d1: i128, d2: i128) -> Vec<u32> {
        let mut out = Vec::new();
        self.overlap_visit(d1, d2, |ids| out.extend_from_slice(ids));
        out
    }

    /// The overlap result SIZE without materializing — the analogue of [`Self::stab_len`].
    pub(crate) fn overlap_len(&self, d1: i128, d2: i128) -> usize {
        let mut n = 0;
        self.overlap_visit(d1, d2, |ids| n += ids.len());
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A tiny deterministic LCG so the test needs no rng crate and no Math.random ban.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 16
        }
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % ((hi - lo) as u64)) as i64
        }
    }

    #[test]
    fn stab_matches_brute_force() {
        let mut rng = Lcg(0x1234_5678);
        // Mix of tiny and large magnitudes, negatives, and equal endpoints.
        let intervals: Vec<(i128, i128)> = (0..800)
            .map(|_| {
                let a = rng.range(-1_000_000, 1_000_000) as i128;
                let w = rng.range(0, 500) as i128;
                (a, a + w)
            })
            .collect();
        let mut t = RiTree::new();
        for (id, &(lo, hi)) in intervals.iter().enumerate() {
            t.insert(lo, hi, id as u32);
        }
        assert_eq!(t.len(), intervals.len());
        for _ in 0..2000 {
            let q = rng.range(-1_100_000, 1_100_000) as i128;
            let mut got = t.stab(q);
            got.sort_unstable();
            let mut want: Vec<u32> = intervals
                .iter()
                .enumerate()
                .filter(|(_, &(lo, hi))| lo <= q && q <= hi)
                .map(|(id, _)| id as u32)
                .collect();
            want.sort_unstable();
            assert_eq!(got, want, "stab({q}) mismatch");
        }
    }

    #[test]
    fn overlap_matches_brute_force() {
        let mut rng = Lcg(0xabcd_ef01);
        let intervals: Vec<(i128, i128)> = (0..800)
            .map(|_| {
                let a = rng.range(-1_000_000, 1_000_000) as i128;
                (a, a + rng.range(0, 500) as i128)
            })
            .collect();
        let mut t = RiTree::new();
        for (id, &(lo, hi)) in intervals.iter().enumerate() {
            t.insert(lo, hi, id as u32);
        }
        for _ in 0..2000 {
            let d1 = rng.range(-1_100_000, 1_100_000) as i128;
            let d2 = d1 + rng.range(0, 3000) as i128; // windows from empty to wide
            let mut got = t.overlap(d1, d2);
            got.sort_unstable();
            let mut want: Vec<u32> = intervals
                .iter()
                .enumerate()
                .filter(|(_, &(lo, hi))| lo <= d2 && hi >= d1)
                .map(|(id, _)| id as u32)
                .collect();
            want.sort_unstable();
            assert_eq!(got, want, "overlap([{d1},{d2}]) mismatch");
        }
    }

    #[test]
    fn insert_remove_keeps_stab_correct() {
        let mut rng = Lcg(0xdead_beef);
        let mut t = RiTree::new();
        let mut live: Vec<(i128, i128, u32)> = Vec::new();
        for id in 0..400u32 {
            let a = rng.range(-10_000, 10_000) as i128;
            let hi = a + rng.range(0, 200) as i128;
            t.insert(a, hi, id);
            live.push((a, hi, id));
        }
        // Remove half (every other), then stab must match the survivors.
        for i in (0..live.len()).step_by(2).rev() {
            let (lo, hi, id) = live.remove(i);
            t.remove(lo, hi, id);
        }
        for _ in 0..1000 {
            let q = rng.range(-11_000, 11_000) as i128;
            let mut got = t.stab(q);
            got.sort_unstable();
            let mut want: Vec<u32> = live
                .iter()
                .filter(|&&(lo, hi, _)| lo <= q && q <= hi)
                .map(|&(_, _, id)| id)
                .collect();
            want.sort_unstable();
            assert_eq!(got, want, "post-remove stab({q}) mismatch");
        }
    }

    #[test]
    fn boundary_keys() {
        // Extremes of the signed domain must bias + fork without overflow.
        let mut t = RiTree::new();
        t.insert(i128::MIN, i128::MIN + 10, 1);
        t.insert(-5, 5, 2);
        t.insert(i128::MAX - 10, i128::MAX, 3);
        assert_eq!(t.stab(i128::MIN + 3), vec![1]);
        assert_eq!(t.stab(0), vec![2]);
        assert_eq!(t.stab(i128::MAX - 3), vec![3]);
        assert!(t.stab(1_000_000).is_empty());
    }
}
