//! Bounded, dedicated worker pool for the graph algorithms (feature `parallel`,
//! native-only — wasm32 has no threads and never enables this module).
//!
//! The whole point is to NOT touch rayon's global pool: an embedder runs the engine
//! inside a Node/Bun process whose event loop and libuv threadpool must keep running,
//! so a global rayon pool sized to every core would starve the host. Instead each
//! parallel algorithm run builds its OWN pool of exactly the configured thread count
//! and executes inside `pool.install(...)`, so every `par_iter` it spawns stays on
//! these workers and nothing else on the machine is affected.
//!
//! Byte-identity across thread counts is the callers' responsibility, not this
//! module's: the algorithms only ever parallelize work whose float reduction is
//! either per-cell (each unit writes its own output) or folded back in a fixed
//! canonical order (ascending source/node id). This helper just supplies the pool.

use rayon::prelude::*;

/// Format the index range `0..n` in parallel and concatenate the pieces IN ORDER,
/// returning one `String` byte-identical to a serial left-to-right build. The range is
/// cut into contiguous ascending chunks (≈4 per worker), each chunk rendered by
/// `render(lo, hi)` into its own buffer on a pool thread, and the buffers joined in
/// chunk order. Because chunks are contiguous and joined in order, the output bytes are
/// exactly the serial order — the byte-identity rule for encoders (each element's text
/// is independent; only the concatenation order matters, and it is preserved).
pub(crate) fn concat_ranges(
    threads: u32,
    n: u32,
    render: impl Fn(u32, u32) -> String + Sync,
) -> String {
    let total = n as usize;
    let workers = threads.max(1) as usize;
    let chunk = (total / (workers * 4)).max(1);
    let ranges: Vec<(u32, u32)> = (0..total)
        .step_by(chunk)
        .map(|lo| (lo as u32, (lo + chunk).min(total) as u32))
        .collect();
    let parts: Vec<String> = with_pool(threads, || {
        ranges.par_iter().map(|&(lo, hi)| render(lo, hi)).collect()
    });
    let mut out = String::with_capacity(parts.iter().map(String::len).sum());
    for p in &parts {
        out.push_str(p);
    }
    out
}

/// Run `f` on a dedicated pool of `threads` workers and return its result. `threads
/// <= 1` runs `f` inline on the caller (no pool built). A pool-build failure also
/// falls back to inline execution rather than faulting — parallelism is an
/// optimization, never a correctness requirement.
pub(crate) fn with_pool<R: Send>(threads: u32, f: impl FnOnce() -> R + Send) -> R {
    let n = threads.max(1) as usize;
    if n <= 1 {
        return f();
    }
    match rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("lenke-algo-{i}"))
        .build()
    {
        Ok(pool) => pool.install(f),
        Err(_) => f(),
    }
}
