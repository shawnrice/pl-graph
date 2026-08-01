//! Storage-layout probe: should we change adjacency storage, and can we do it
//! without penalizing writes? Measures, at two average degrees:
//!   (1) READ CEILING — a 1-hop and 2-hop neighbor walk over the live
//!       `Vec<Vec<Adj>>` vs a transient CSR snapshot (same logical work; the
//!       delta is *pure memory locality*, the most any layout change could buy).
//!   (2) WRITE BASELINE — current `add_edge` throughput (O(1) push).
//!   (3) WRITE PENALTY — the same inserts done as a sorted-by-etype insert
//!       (O(degree) shift), the cost a sorted-neighbor layout would add.
//! Run: cargo run --release --example storage_probe

use std::time::Instant;

use lenke_core::graph::{Adj, Builder, EdgeRec, Graph, NodeRec};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// `nv` vertices, each with `deg` out-edges of one of 3 edge types.
fn build_graph(nv: usize, deg: usize) -> Graph {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut b = Builder::default();
    for i in 0..nv {
        b.nodes.push(NodeRec::owned(
            format!("v{i}"),
            vec!["N".to_string()],
            vec![],
        ));
    }
    let types = ["A", "B", "C"];
    for i in 0..nv {
        for k in 0..deg {
            b.edges.push(EdgeRec::owned(
                format!("v{i}"),
                format!("v{}", rng.below(nv)),
                types[k % 3].to_string(),
                vec![],
                None,
            ));
        }
    }
    b.finalize()
}

/// Build a transient CSR snapshot (offsets + flat neighbor array) from the live
/// edge arrays. Read-only — models a freeze/delta design's bulk store.
fn build_csr(g: &Graph) -> (Vec<u32>, Vec<u32>) {
    let n = g.n;
    let mut off = vec![0u32; n + 1];
    for &s in &g.e_src {
        off[s as usize + 1] += 1;
    }
    for i in 0..n {
        off[i + 1] += off[i];
    }
    let mut nbr = vec![0u32; g.e_src.len()];
    let mut cur = off.clone();
    for e in 0..g.e_src.len() {
        let s = g.e_src[e] as usize;
        nbr[cur[s] as usize] = g.e_dst[e];
        cur[s] += 1;
    }
    (off, nbr)
}

fn ms(t: Instant, iters: u32) -> f64 {
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn main() {
    for (nv, deg) in [(50_000usize, 8usize), (8_000usize, 64usize)] {
        let g = build_graph(nv, deg);
        let n = g.n;
        let (off, nbr) = build_csr(&g);
        let iters = 50u32;
        println!(
            "\n=== {nv} vertices, avg degree {deg} ({} edges) ===",
            g.edge_count()
        );

        // --- (1) READ CEILING: 1-hop neighbor sum ---
        let t = Instant::now();
        let mut acc = 0u64;
        for _ in 0..iters {
            for v in 0..n as u32 {
                for a in g.out_adj(v) {
                    acc = acc.wrapping_add(a.nbr as u64);
                }
            }
        }
        let veclist_1 = ms(t, iters);

        let t = Instant::now();
        let mut acc2 = 0u64;
        for _ in 0..iters {
            for v in 0..n {
                for &x in &nbr[off[v] as usize..off[v + 1] as usize] {
                    acc2 = acc2.wrapping_add(x as u64);
                }
            }
        }
        let csr_1 = ms(t, iters);
        assert_eq!(acc, acc2);

        // --- 2-hop neighbor sum (amplifies locality) ---
        let t = Instant::now();
        let mut acc = 0u64;
        for _ in 0..iters {
            for v in 0..n as u32 {
                for a in g.out_adj(v) {
                    for b in g.out_adj(a.nbr) {
                        acc = acc.wrapping_add(b.nbr as u64);
                    }
                }
            }
        }
        let veclist_2 = ms(t, iters);

        let t = Instant::now();
        let mut acc2 = 0u64;
        for _ in 0..iters {
            for v in 0..n {
                for &m in &nbr[off[v] as usize..off[v + 1] as usize] {
                    let m = m as usize;
                    for &x in &nbr[off[m] as usize..off[m + 1] as usize] {
                        acc2 = acc2.wrapping_add(x as u64);
                    }
                }
            }
        }
        let csr_2 = ms(t, iters);
        assert_eq!(acc, acc2);

        println!(
            "  1-hop walk : Vec<Vec> {veclist_1:6.2} ms   CSR {csr_1:6.2} ms   ({:.2}x)",
            veclist_1 / csr_1
        );
        println!(
            "  2-hop walk : Vec<Vec> {veclist_2:6.2} ms   CSR {csr_2:6.2} ms   ({:.2}x)",
            veclist_2 / csr_2
        );

        // --- (2) WRITE BASELINE vs (3) SORTED-INSERT PENALTY ---
        // Replicate add_edge's adjacency maintenance in isolation: plain push
        // (current) vs sorted-by-etype insert (the sorted-neighbor layout).
        let medges = nv * deg;
        let mut rng = Rng(0xDEAD_BEEF);
        let edges: Vec<(usize, u32, u32)> = (0..medges)
            .map(|_| (rng.below(nv), rng.below(nv) as u32, (rng.next() % 3) as u32))
            .collect();

        let mut plain: Vec<Vec<Adj>> = vec![Vec::new(); nv];
        let t = Instant::now();
        for (i, &(s, d, ty)) in edges.iter().enumerate() {
            plain[s].push(Adj {
                eidx: i as u32,
                nbr: d,
                etype: ty,
            });
        }
        let push_ms = t.elapsed().as_secs_f64() * 1e3;

        let mut sorted: Vec<Vec<Adj>> = vec![Vec::new(); nv];
        let t = Instant::now();
        for (i, &(s, d, ty)) in edges.iter().enumerate() {
            let lst = &mut sorted[s];
            let pos = lst.partition_point(|a| a.etype < ty);
            lst.insert(
                pos,
                Adj {
                    eidx: i as u32,
                    nbr: d,
                    etype: ty,
                },
            );
        }
        let sorted_ms = t.elapsed().as_secs_f64() * 1e3;

        println!(
            "  {medges} inserts: push {push_ms:6.2} ms   sorted-insert {sorted_ms:6.2} ms   ({:.2}x slower writes)",
            sorted_ms / push_ms
        );
    }
}
