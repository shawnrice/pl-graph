//! A generational handle table for prepared statements, kept OFF the `Store` so a
//! statement's lifecycle never looks like a graph mutation (freeing one bumps no
//! `version`/`epoch`, emits no CDC, invalidates no read snapshot).
//!
//! The point is safety for a JavaScript caller. JS has `using` /
//! `FinalizationRegistry` for the *lifetime* side (a forgotten handle is freed on
//! GC), but nothing guards against **use-after-free** or **double-free** — and to
//! a JS developer those are the foreign, dangerous failures. A raw pointer handle
//! turns them into undefined behavior; this table turns them into a clean error:
//! every handle carries the generation of its slot, and a stale handle (freed, or
//! its slot reused) fails validation instead of dereferencing freed memory.
//!
//! Single-threaded by design (bun:ffi / wasm drive the ABI on one thread), so the
//! table is a `thread_local`.

use crate::ir::Plan;
use std::cell::RefCell;

struct Slot {
    /// Bumped on every free, so a handle minted for a prior occupant no longer
    /// validates once the slot is reused.
    generation: u32,
    plan: Option<Plan>,
}

/// A slab of prepared plans addressed by generational handles.
#[derive(Default)]
pub struct PreparedSlab {
    slots: Vec<Slot>,
    /// Indices of freed slots, reused before growing.
    free: Vec<usize>,
}

/// Pack `(index, generation)` into an opaque handle. The host treats it as an
/// opaque token (carried as a decimal string, since it can exceed a JS `f64`).
fn pack(index: usize, generation: u32) -> u64 {
    ((index as u64) << 32) | u64::from(generation)
}
fn unpack(handle: u64) -> (usize, u32) {
    ((handle >> 32) as usize, (handle & 0xFFFF_FFFF) as u32)
}

impl PreparedSlab {
    /// Store `plan` and return its handle (reusing a freed slot when available).
    fn insert(&mut self, plan: Plan) -> u64 {
        if let Some(i) = self.free.pop() {
            self.slots[i].plan = Some(plan);
            pack(i, self.slots[i].generation)
        } else {
            self.slots.push(Slot {
                generation: 0,
                plan: Some(plan),
            });
            pack(self.slots.len() - 1, 0)
        }
    }

    /// Clone the plan for `handle`, or `None` if the handle is stale/freed/unknown
    /// (a use-after-free surfaces here as `None`, never a bad dereference).
    fn get_clone(&self, handle: u64) -> Option<Plan> {
        let (i, generation) = unpack(handle);
        let slot = self.slots.get(i)?;
        if slot.generation == generation {
            slot.plan.clone()
        } else {
            None
        }
    }

    /// Free `handle`. Returns `false` if it was already freed / stale / unknown (a
    /// double-free surfaces here as `false`, never a double `Box::from_raw`).
    fn remove(&mut self, handle: u64) -> bool {
        let (i, generation) = unpack(handle);
        let Some(slot) = self.slots.get_mut(i) else {
            return false;
        };
        if slot.generation != generation || slot.plan.is_none() {
            return false;
        }
        slot.plan = None;
        // Invalidate every outstanding handle to this slot before it is reused.
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(i);
        true
    }
}

thread_local! {
    static SLAB: RefCell<PreparedSlab> = RefCell::new(PreparedSlab::default());
}

/// Store a prepared plan, returning its handle.
pub fn insert(plan: Plan) -> u64 {
    SLAB.with(|s| s.borrow_mut().insert(plan))
}

/// Clone the plan for `handle` (`None` if stale / freed / unknown).
pub fn get_clone(handle: u64) -> Option<Plan> {
    SLAB.with(|s| s.borrow().get_clone(handle))
}

/// Free `handle` (`false` if it was already freed / stale / unknown).
pub fn free(handle: u64) -> bool {
    SLAB.with(|s| s.borrow_mut().remove(handle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_and_use_after_free_is_clean() {
        let mut slab = PreparedSlab::default();
        let h = slab.insert(Plan::Row);
        assert!(slab.get_clone(h).is_some(), "live handle resolves");
        assert!(slab.remove(h), "first free succeeds");
        // Use-after-free: resolving a freed handle is None, not a bad deref.
        assert!(
            slab.get_clone(h).is_none(),
            "freed handle no longer resolves"
        );
        // Double-free: the second free is a clean false, not a double drop.
        assert!(!slab.remove(h), "double free is rejected");
    }

    #[test]
    fn reused_slot_invalidates_the_old_handle() {
        let mut slab = PreparedSlab::default();
        let h1 = slab.insert(Plan::Row);
        assert!(slab.remove(h1));
        let h2 = slab.insert(Plan::Row); // reuses slot 0 at a new generation
        assert_ne!(h1, h2, "the reused slot mints a distinct handle");
        assert!(slab.get_clone(h2).is_some(), "the new handle resolves");
        assert!(
            slab.get_clone(h1).is_none(),
            "the stale handle to the reused slot does not resolve"
        );
    }

    #[test]
    fn unknown_handle_is_rejected() {
        let mut slab = PreparedSlab::default();
        assert!(slab.get_clone(pack(999, 0)).is_none());
        assert!(!slab.remove(pack(999, 0)));
    }
}
