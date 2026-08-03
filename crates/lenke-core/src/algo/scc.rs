//! Strongly-connected components via **iterative** Tarjan (an explicit work stack,
//! never native recursion — a deep chain must not blow the thread stack). Two
//! vertices share a component iff each is reachable from the other along directed
//! edges. The SCC partition is unique, so — exactly like weakly-connected components
//! — each component's id is its **first-inserted (lowest dense-id) member's external
//! id**, chosen independently of DFS order. That makes the TS mirror byte-identical:
//! both engines compute the same partition and pick the same min-index representative,
//! regardless of the order they happen to walk neighbours. O(V + E).

use super::AlgoConfig;
use crate::graph::{Graph, Value};

/// The SCC partition as `comp[slot] = the component's min-dense-index member` — the
/// shared byte-identical Tarjan pass behind both `strongly_connected_components` and
/// `on_cycle`. Dead/unvisited slots stay `u32::MAX`.
fn scc_reps(graph: &Graph, cfg: &AlgoConfig) -> Vec<u32> {
    let slots = graph.n;

    // Some(None) = every type; Some(Some(t)) = one type; None = a named-but-unknown
    // type → no edges → every vertex is its own singleton component.
    let etype = cfg.etype(graph);
    // ANY of the edge's labels — see `algo::edge_type_ok`.
    let type_ok = |ei: usize| match etype {
        Some(Some(t)) => graph.edge_has_label(ei as u32, t),
        Some(None) => true,
        None => false,
    };

    // Forward adjacency as a flat CSR (`adj_off[v]..adj_off[v+1]` indexes `adj_tgt`),
    // built by two edge sweeps in insertion order — the same shape PageRank uses.
    let mut adj_off = vec![0usize; slots + 1];
    for ei in 0..graph.edge_slots() {
        if graph.is_edge_live(ei as u32) && type_ok(ei) {
            adj_off[graph.e_src[ei] as usize + 1] += 1;
        }
    }
    for v in 0..slots {
        adj_off[v + 1] += adj_off[v];
    }
    let mut adj_tgt = vec![0u32; adj_off[slots]];
    let mut cursor = adj_off[..slots].to_vec();
    for ei in 0..graph.edge_slots() {
        if graph.is_edge_live(ei as u32) && type_ok(ei) {
            let src = graph.e_src[ei] as usize;
            adj_tgt[cursor[src]] = graph.e_dst[ei];
            cursor[src] += 1;
        }
    }

    const UNVISITED: u32 = u32::MAX;
    let mut order = vec![UNVISITED; slots]; // DFS discovery index (Tarjan's `index`)
    let mut low = vec![0u32; slots]; // lowlink
    let mut on_stack = vec![false; slots];
    let mut comp = vec![UNVISITED; slots]; // resolved component representative
    let mut tstack: Vec<u32> = Vec::new(); // Tarjan's component stack
    let mut counter: u32 = 0;

    // Each DFS frame is `(vertex, next-neighbour cursor into adj_tgt)`.
    let mut frames: Vec<(u32, usize)> = Vec::new();

    for s in graph.vertex_indices() {
        if order[s as usize] != UNVISITED {
            continue;
        }
        order[s as usize] = counter;
        low[s as usize] = counter;
        counter += 1;
        on_stack[s as usize] = true;
        tstack.push(s);
        frames.push((s, adj_off[s as usize]));

        while let Some(&(v, ci)) = frames.last() {
            let vu = v as usize;
            if ci < adj_off[vu + 1] {
                // Advance this frame's cursor, then process neighbour `w`.
                frames.last_mut().unwrap().1 = ci + 1;
                let w = adj_tgt[ci];
                let wu = w as usize;
                if order[wu] == UNVISITED {
                    // Tree edge: descend into `w`.
                    order[wu] = counter;
                    low[wu] = counter;
                    counter += 1;
                    on_stack[wu] = true;
                    tstack.push(w);
                    frames.push((w, adj_off[wu]));
                } else if on_stack[wu] {
                    // Back/cross edge to a vertex still on the stack.
                    low[vu] = low[vu].min(order[wu]);
                }
            } else {
                // `v` is fully explored. If it's an SCC root, pop its component and
                // stamp every member with the component's min-dense-index member.
                if low[vu] == order[vu] {
                    let mut members: Vec<u32> = Vec::new();
                    loop {
                        let m = tstack.pop().unwrap();
                        on_stack[m as usize] = false;
                        members.push(m);
                        if m == v {
                            break;
                        }
                    }
                    let rep = *members.iter().min().unwrap();
                    for m in members {
                        comp[m as usize] = rep;
                    }
                }
                frames.pop();
                // Propagate the finished child's lowlink up to its parent.
                if let Some(&(p, _)) = frames.last() {
                    low[p as usize] = low[p as usize].min(low[vu]);
                }
            }
        }
    }

    comp
}

pub fn strongly_connected_components(graph: &Graph, cfg: &AlgoConfig) -> Vec<(u32, Value)> {
    let comp = scc_reps(graph, cfg);
    graph
        .vertex_indices()
        .map(|v| (v, Value::Str(graph.vid.arc(comp[v as usize]))))
        .collect()
}

/// Per-vertex cycle membership: `true` iff the vertex lies on a directed cycle —
/// i.e. its SCC has more than one member OR it has a self-loop (a 1-cycle). Derived
/// from the same byte-identical SCC partition + a deterministic self-loop scan, so
/// the TS mirror matches. A named-but-unknown edge type → no edges → every vertex
/// is `false`.
pub fn on_cycle(graph: &Graph, cfg: &AlgoConfig) -> Vec<(u32, Value)> {
    let slots = graph.n;
    let comp = scc_reps(graph, cfg);

    // Component sizes: a component with >1 member is a cycle (every member is on it).
    let mut size = vec![0u32; slots];
    for v in graph.vertex_indices() {
        let rep = comp[v as usize];
        if (rep as usize) < slots {
            size[rep as usize] += 1;
        }
    }

    // Self-loops (v→v of the selected type) put a singleton on a 1-cycle too.
    let etype = cfg.etype(graph);
    // ANY of the edge's labels — see `algo::edge_type_ok`.
    let type_ok = |ei: usize| match etype {
        Some(Some(t)) => graph.edge_has_label(ei as u32, t),
        Some(None) => true,
        None => false,
    };
    let mut self_loop = vec![false; slots];
    for ei in 0..graph.edge_slots() {
        if graph.is_edge_live(ei as u32) && type_ok(ei) && graph.e_src[ei] == graph.e_dst[ei] {
            self_loop[graph.e_src[ei] as usize] = true;
        }
    }

    graph
        .vertex_indices()
        .map(|v| {
            let vu = v as usize;
            let cyclic = size[comp[vu] as usize] > 1 || self_loop[vu];
            (v, Value::Bool(cyclic))
        })
        .collect()
}
