//! Shared GQL "hard shape" generator: quantified var-length hops, subpath GROUPS
//! with group variables, NESTED groups (`( (…){a,b} ){c,d}`), per-rep WHERE, and
//! shortest paths — the constructs the flat differential/perf generators never
//! emit. It produces a full `MATCH … RETURN` query whose RETURN reduces every path
//! / group binding to SCALARS (`size`, `gv[i].prop`, `path_length`, the endpoint
//! id), so two engines' results compare as a plain MULTISET of Num/Str/Null cells —
//! no element-identity, list-rendering, or enumeration-order dependence.
//!
//! Reused by both `tests/differential_fuzz.rs` (correctness: run both engines,
//! compare) and `examples/perf_fuzz.rs` (perf: time both) via `#[path]`. The graph
//! schema (labels, edge type, prop names) is passed in as a `Schema`, so each
//! harness drives its own fixture. `Caps` gates which families are emitted, so a
//! not-yet-implemented family (e.g. `nested`) can be turned on to DRIVE its
//! implementation and off to keep a suite green.
#![allow(dead_code)] // each includer uses a subset

// ── deterministic RNG (xorshift64*) ──────────────────────────────────────────
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    pub fn chance(&mut self, num: u32, den: u32) -> bool {
        (self.next() % u64::from(den)) < u64::from(num)
    }
    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        let i = self.below(xs.len());
        &xs[i]
    }
}

/// The fixture's GQL vocabulary the generator writes against.
pub struct Schema {
    /// A node label every node carries (the pattern anchor's label).
    pub label: &'static str,
    /// The edge type to traverse.
    pub etype: &'static str,
    /// A numeric node property (may be absent/null on some nodes).
    pub num: &'static str,
    /// A unique numeric node property — the anchor selector `{id: k}`.
    pub id: &'static str,
    /// A numeric EDGE property (for per-rep / per-hop edge predicates).
    pub ew: &'static str,
}

/// Which hard families to emit. Turn a family OFF to keep a suite green while its
/// engine support is missing; ON to drive/verify it.
#[derive(Clone, Copy)]
pub struct Caps {
    pub varlen: bool,   // `-[:R]->{lo,hi}(t)`
    pub group: bool,    // `((x)-[e:R]->(y)){lo,hi} (t)` with group vars
    pub per_rep: bool,  // a `WHERE` inside the group / a per-hop edge WHERE
    pub shortest: bool, // `ANY SHORTEST (s)-[:R]->*(t)`
    pub nested: bool,   // `( (…){a,b} ){c,d} (t)` and group-over-var-length
}
impl Caps {
    pub fn all() -> Self {
        Self {
            varlen: true,
            group: true,
            per_rep: true,
            shortest: true,
            nested: true,
        }
    }
    /// Everything the engine is known to support (nested excluded).
    pub fn supported() -> Self {
        Self {
            nested: false,
            ..Self::all()
        }
    }
    fn any(&self) -> bool {
        self.varlen || self.group || self.per_rep || self.shortest || self.nested
    }
}

/// A generated query and how to compare it.
pub struct Hard {
    pub text: String,
    pub tags: Vec<&'static str>,
    /// Group/var-length/shortest results have unspecified row order — compare as a
    /// multiset. (Always true here; kept explicit for the caller.)
    pub multiset: bool,
}

/// A small quantifier with `hi <= 3` — used by SINGLE-level families where a few
/// hops on a small graph enumerate cheaply.
fn quant(rng: &mut Rng) -> (String, u32, u32) {
    match rng.below(6) {
        0 => ("+".into(), 1, 3),
        1 => ("*".into(), 0, 3),
        2 => {
            let n = 1 + rng.below(2) as u32;
            (format!("{{{n}}}"), n, n)
        }
        _ => {
            let lo = rng.below(3) as u32;
            let hi = lo + 1 + rng.below(2) as u32;
            (format!("{{{lo},{hi}}}"), lo, hi)
        }
    }
}

/// A BOUNDED, progress-guaranteed quantifier for NESTED families: `1 <= lo <= hi <=
/// 2` (or an exact `{1}`/`{2}`), never `*`/`+`. An unbounded outer over an inner that
/// can match zero hops (`( … {0,n} )*`) is a degenerate no-progress loop that
/// explodes BOTH engines — the corpus's nested cases are all small and bounded, so
/// mirror that.
fn quant_bounded(rng: &mut Rng) -> (String, u32, u32) {
    match rng.below(3) {
        0 => ("{1}".into(), 1, 1),
        1 => ("{2}".into(), 2, 2),
        _ => ("{1,2}".into(), 1, 2),
    }
}

/// A scalar reducer over a NODE group variable `g` (list of nodes; or list-of-lists
/// when `nested`), yielding a Num/Str/Null. `depth` is the list nesting.
fn node_reducer(rng: &mut Rng, s: &Schema, g: &str, depth: u8) -> String {
    match depth {
        1 => match rng.below(3) {
            0 => format!("size({g}) AS r"),
            1 => format!("{g}[{}].{} AS r", rng.below(3), s.id),
            _ => format!("{g}[{}].{} AS r", rng.below(3), s.num),
        },
        _ => match rng.below(4) {
            0 => format!("size({g}) AS r"),
            1 => format!("size({g}[{}]) AS r", rng.below(2)),
            2 => format!("{g}[{}][{}].{} AS r", rng.below(2), rng.below(2), s.id),
            _ => format!("{g}[{}][{}].{} AS r", rng.below(2), rng.below(2), s.num),
        },
    }
}

/// A scalar reducer over an EDGE group variable `g`.
fn edge_reducer(rng: &mut Rng, s: &Schema, g: &str, depth: u8) -> String {
    if depth >= 2 {
        return format!("size({g}[{}]) AS r", rng.below(2));
    }
    match rng.below(2) {
        0 => format!("size({g}) AS r"),
        _ => format!("{g}[{}].{} AS r", rng.below(3), s.ew),
    }
}

/// Generate one hard-shape query. `src_id` selects the anchor node `(s {id:src_id})`
/// to bound the traversal. Returns `None` if no enabled family fit (caller retries).
pub fn gen_hard(rng: &mut Rng, s: &Schema, caps: &Caps, src_id: usize) -> Option<Hard> {
    if !caps.any() {
        return None;
    }
    // Pick an ENABLED family.
    let mut families: Vec<u8> = Vec::new();
    if caps.varlen {
        families.push(0);
    }
    if caps.group {
        families.push(1);
    }
    if caps.shortest {
        families.push(2);
    }
    if caps.nested {
        families.push(3);
        families.push(4);
    }
    let fam = *rng.pick(&families);
    let src = format!("(src:{} {{{}: {src_id}}})", s.label, s.id);
    let mut tags: Vec<&'static str> = Vec::new();

    let text = match fam {
        // ── var-length hop → endpoint ────────────────────────────────────────
        0 => {
            tags.push("varlen");
            let (q, _, _) = quant(rng);
            let where_ = if caps.per_rep && rng.chance(1, 3) {
                tags.push("per-hop-edge");
                format!(
                    "MATCH {src}-[e:{ety} WHERE e.{ew} {op} {k}]->{q}(t) ",
                    ety = s.etype,
                    ew = s.ew,
                    op = rng.pick(&["<", ">=", "<>"]),
                    k = rng.below(1000),
                )
            } else {
                format!("MATCH {src}-[:{}]->{q}(t) ", s.etype)
            };
            format!("{where_}RETURN t.{} AS r", s.id)
        }
        // ── subpath group with group vars → endpoint + reducer ────────────────
        1 => {
            tags.push("group");
            let (q, _, _) = quant(rng);
            // Bind a node group var (x/y) and/or an edge group var (e).
            let bind_edge = rng.chance(1, 2);
            let ev = if bind_edge { "e" } else { "" };
            let inner_where = if caps.per_rep && rng.chance(1, 3) {
                tags.push("per-rep-where");
                // `size(e) = k` needs the edge var; else a node predicate.
                if bind_edge {
                    format!(
                        " WHERE size(e) {} {}",
                        rng.pick(&["=", ">=", "<"]),
                        1 + rng.below(2)
                    )
                } else {
                    format!(
                        " WHERE x.{} {} {}",
                        s.num,
                        rng.pick(&[">=", "<", "<>"]),
                        rng.below(100)
                    )
                }
            } else {
                String::new()
            };
            let group = format!(
                "((x)-[{ev}:{ety}]->(y){iw}){q}",
                ety = s.etype,
                iw = inner_where,
            );
            // A reducer over one of the bound group vars.
            let reducer = if bind_edge && rng.chance(1, 2) {
                edge_reducer(rng, s, "e", 1)
            } else {
                let g = if rng.chance(1, 2) { "x" } else { "y" };
                node_reducer(rng, s, g, 1)
            };
            format!(
                "MATCH {src} {group} (t) RETURN t.{} AS tid, {reducer}",
                s.id
            )
        }
        // ── shortest path → endpoint + length ────────────────────────────────
        2 => {
            tags.push("shortest");
            let star = *rng.pick(&["*", "+"]);
            format!(
                "MATCH p = ANY SHORTEST {src}-[:{ety}]->{star}(t) RETURN t.{id} AS tid, path_length(p) AS len",
                ety = s.etype,
                id = s.id,
            )
        }
        // ── NESTED group: `( ((x)-[e]->(y)){a,b} ){c,d} (t)` ──────────────────
        3 => {
            tags.push("nested");
            tags.push("nested-lol");
            let (inner, _, _) = quant_bounded(rng);
            let (outer, _, _) = quant_bounded(rng);
            let bind_edge = rng.chance(1, 2);
            let ev = if bind_edge { "e" } else { "" };
            let inner_where = if caps.per_rep && rng.chance(1, 4) {
                tags.push("nested-per-rep-where");
                if bind_edge {
                    format!(
                        " WHERE size(e) {} {}",
                        rng.pick(&["=", ">="]),
                        1 + rng.below(2)
                    )
                } else {
                    String::new()
                }
            } else {
                String::new()
            };
            let group = format!(
                "( ((x)-[{ev}:{ety}]->(y){iw}){inner} ){outer}",
                ety = s.etype,
                iw = inner_where,
            );
            // Depth-2 reducer over x (or e).
            let reducer = if bind_edge && rng.chance(1, 2) {
                edge_reducer(rng, s, "e", 2)
            } else {
                node_reducer(rng, s, "x", 2)
            };
            format!(
                "MATCH {src} {group} (t) RETURN t.{} AS tid, {reducer}",
                s.id
            )
        }
        // ── group over a VAR-LENGTH inner: `( (x)-[e]->{lo,hi}(y) ){c,d} (t)` ─
        _ => {
            tags.push("nested");
            tags.push("nested-varlen-inner");
            let (inner, _, _) = quant(rng);
            let (outer, _, _) = quant_bounded(rng);
            let bind_edge = rng.chance(1, 2);
            let ev = if bind_edge { "e" } else { "" };
            let group = format!("( (x)-[{ev}:{ety}]->{inner}(y) ){outer}", ety = s.etype,);
            // With a var-length inner, x/y bind once per OUTER rep (depth 1); e is a
            // list-of-lists (depth 2) only when the inner is var-length AND bound.
            let reducer = if bind_edge && rng.chance(1, 2) {
                edge_reducer(rng, s, "e", 2)
            } else {
                let g = if rng.chance(1, 2) { "x" } else { "y" };
                node_reducer(rng, s, g, 1)
            };
            format!(
                "MATCH {src} {group} (t) RETURN t.{} AS tid, {reducer}",
                s.id
            )
        }
    };

    Some(Hard {
        text,
        tags,
        multiset: true,
    })
}
