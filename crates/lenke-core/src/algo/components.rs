//! Weakly-connected components via union-find. Edges are treated as undirected
//! (each out-edge unions its two endpoints), and union is **by smaller vertex
//! index** so every component's root is its first-inserted (lowest dense-id)
//! vertex — the component id is then that root's external id string. Root choice
//! is independent of edge-processing order, so the TS mirror (which unions by
//! insertion index) is byte-identical. O(V + E·α).

use super::AlgoConfig;
use crate::graph::{Graph, Value};

/// Find with full path compression.
fn find(parent: &mut [u32], x: u32) -> u32 {
    let mut root = x;
    while parent[root as usize] != root {
        root = parent[root as usize];
    }
    // Compress: point every node on the path straight at the root.
    let mut cur = x;
    while parent[cur as usize] != root {
        let next = parent[cur as usize];
        parent[cur as usize] = root;
        cur = next;
    }
    root
}

/// Union `a` and `b`, keeping the smaller-indexed root (deterministic).
fn union(parent: &mut [u32], a: u32, b: u32) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra == rb {
        return;
    }
    let (keep, drop) = if ra < rb { (ra, rb) } else { (rb, ra) };
    parent[drop as usize] = keep;
}

pub fn connected_components(graph: &Graph, cfg: &AlgoConfig) -> Vec<(u32, Value)> {
    // A named-but-unknown edge type → no edges → every vertex is its own component.
    let etype = cfg.etype(graph);

    // parent indexed by dense id over the whole slot space (0..n); tombstoned slots
    // are self-parented and simply never visited via `vertex_indices()`.
    let mut parent: Vec<u32> = (0..graph.n as u32).collect();

    if let Some(etype) = etype {
        // One flat sweep over the edge columns (contiguous) unions each edge's
        // endpoints — union-by-min is order-independent, so the forest is identical.
        for ei in 0..graph.edge_slots() {
            if graph.is_edge_live(ei as u32) && super::edge_type_ok(graph, etype, ei as u32) {
                union(&mut parent, graph.e_src[ei], graph.e_dst[ei]);
            }
        }
    }

    graph
        .vertex_indices()
        .map(|v| {
            let root = find(&mut parent, v);
            (v, Value::Str(graph.vid.arc(root)))
        })
        .collect()
}
