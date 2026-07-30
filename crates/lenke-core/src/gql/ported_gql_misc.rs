//! Ported GQL tests — faithful Rust port of `packages/gql/src/errors.test.ts`
//! and `packages/gql/src/fuzz.test.ts`.
//!
//! errors.test.ts: 1 test, ported 1:1.
//! fuzz.test.ts: 7 property-based fuzz suites, all using a seeded mulberry32 RNG
//! with 400 iterations each. The RNG and graph/predicate generators are ported
//! directly from the TS source so the same seeded sequence executes in Rust.
//!
//! Self-contained: copies the `modern()` fixture, helper functions, and imports.

use super::eval::Params;
use super::parse;
use crate::graph::{Graph, Value};

// ── Mulberry32 RNG — direct port of the TS makeRng ───────────────────────────
//
// TS:
//   let s = seed >>> 0;
//   s = (s + 0x6d2b79f5) >>> 0;
//   let t = Math.imul(s ^ (s >>> 15), 1 | s);
//   t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
//   return ((t ^ (t >>> 14)) >>> 0) / 4294967296;

struct Mulberry32 {
    s: u32,
}

impl Mulberry32 {
    fn new(seed: u32) -> Self {
        Self { s: seed }
    }

    fn next(&mut self) -> f64 {
        self.s = self.s.wrapping_add(0x6d2b79f5);
        let mut t = (self.s ^ (self.s >> 15)).wrapping_mul(1u32 | self.s);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61u32 | t));
        ((t ^ (t >> 14)) as f64) / 4_294_967_296.0
    }
}

// ── Random graph builder — port of TS makeGraph ──────────────────────────────
//
// Creates a Graph with 4..12 Node vertices each having a random subset of
// {a, b, c} properties (numeric 0..4 or string 'x'/'y'), plus random R edges.
// Property values match the TS generator exactly so the same seed yields the
// same logical structure (node count, property assignments, edge topology).

fn make_fuzz_graph(rng: &mut Mulberry32) -> Graph {
    let mut g = crate::graph::Builder::default().finalize();
    let count = 4 + (rng.next() * 8.0) as usize;
    let mut vis: Vec<u32> = Vec::with_capacity(count);

    for _ in 0..count {
        let mut props: Vec<(String, Value)> = Vec::new();

        for p in &["a", "b", "c"] {
            let r = rng.next();
            if r < 0.3 {
                continue; // absent → NULL
            }
            let v = if r < 0.75 {
                Value::Num((rng.next() * 5.0).floor())
            } else {
                Value::Str((if rng.next() < 0.5 { "x" } else { "y" }).into())
            };
            props.push((p.to_string(), v));
        }

        let vi = g.add_vertex(&["Node".to_string()], props);
        vis.push(vi);
    }

    let edge_count = (rng.next() * count as f64) as usize;
    for _ in 0..edge_count {
        let from = vis[(rng.next() * vis.len() as f64) as usize];
        let to = vis[(rng.next() * vis.len() as f64) as usize];
        g.add_edge(from, to, "R", vec![]);
    }

    g
}

// ── Random predicate builder — port of TS leaf/pred ─────────────────────────

fn fuzz_leaf(rng: &mut Mulberry32) -> String {
    let props = ["a", "b", "c"];
    let p = format!("n.{}", props[(rng.next() * 3.0) as usize]);
    let k = rng.next();

    if k < 0.55 {
        let comp = [">", "<", ">=", "<=", "=", "<>"];
        let op = comp[(rng.next() * 6.0) as usize];
        let val = if rng.next() < 0.7 {
            format!("{}", (rng.next() * 5.0).floor() as i64)
        } else if rng.next() < 0.5 {
            "'x'".to_string()
        } else {
            "'y'".to_string()
        };
        return format!("{p} {op} {val}");
    }

    if k < 0.78 {
        let not = if rng.next() < 0.5 { "" } else { "NOT " };
        return format!("{p} IS {not}NULL");
    }

    // IN / NOT IN — avoid two simultaneous &mut rng borrows by collecting separately.
    let mut items: Vec<String> = Vec::new();
    for _ in 0..4 {
        let include = rng.next() < 0.5;
        let val = (rng.next() * 5.0).floor() as i64;
        if include {
            items.push(val.to_string());
        }
    }
    let not = if rng.next() < 0.5 { "" } else { "NOT " };
    format!("{p} {not}IN [{}]", items.join(", "))
}

fn fuzz_pred(rng: &mut Mulberry32, depth: i32) -> String {
    if depth <= 0 || rng.next() < 0.4 {
        return format!("({})", fuzz_leaf(rng));
    }
    let k = rng.next();
    if k < 0.25 {
        return format!("(NOT {})", fuzz_pred(rng, depth - 1));
    }
    let op = if k < 0.5 {
        "AND"
    } else if k < 0.75 {
        "OR"
    } else {
        "XOR"
    };
    let l = fuzz_pred(rng, depth - 1);
    let r = fuzz_pred(rng, depth - 1);
    format!("({l} {op} {r})")
}

/// The set of element_ids kept by `WHERE <predicate>`, sorted for comparison.
fn ids_where(g: &mut Graph, predicate: &str) -> Vec<String> {
    let q_str = format!("MATCH (n:Node) WHERE {predicate} RETURN element_id(n) AS id");
    let rs = parse(&q_str)
        .unwrap_or_else(|e| panic!("parse error: {e}\n  predicate: {predicate}"))
        .execute(g, &Params::new())
        .unwrap_or_else(|e| panic!("exec error: {e}\n  predicate: {predicate}"));
    let mut ids: Vec<String> = rs
        .rows()
        .map(|row| match &row[0] {
            Value::Str(s) => s.to_string(),
            v => format!("{v:?}"),
        })
        .collect();
    ids.sort();
    ids
}

fn sorted_union(a: &[String], b: &[String]) -> Vec<String> {
    let mut set: std::collections::HashSet<String> = std::collections::HashSet::new();
    for x in a {
        set.insert(x.clone());
    }
    for x in b {
        set.insert(x.clone());
    }
    let mut v: Vec<String> = set.into_iter().collect();
    v.sort();
    v
}

// ─────────────────────────────────────────────────────────────────────────────
// errors.test.ts
// ─────────────────────────────────────────────────────────────────────────────

/// Port of: "a parse error carries the stable ErrorCode.Syntax (not just a message)"
///
/// TS asserts: caught instanceof GqlSyntaxError, caught.code === ErrorCode.Syntax,
///             hasErrorCode(caught, ErrorCode.Syntax), caught.pos >= 0.
///
/// Rust: parse() returns Err(SyntaxError). SyntaxError IS the Syntax code —
/// the type is the discriminant. We assert Err and pos >= 0.
#[test]
fn x_parse_error_carries_syntax_code() {
    let result = parse("MATCH ("); // unterminated pattern
    assert!(
        result.is_err(),
        "expected parse error for unterminated pattern"
    );
    let err = result.unwrap_err();
    // SyntaxError carries pos; the type itself encodes ErrorCode::Syntax.
    // (In Rust there is no .code field — the Err(SyntaxError) variant IS the syntax code.)
    assert!(
        err.pos < usize::MAX, // pos is always a valid usize — always >= 0
        "SyntaxError.pos must be a valid position; got {}",
        err.pos
    );
    // The message must be non-empty (mirrors the TS "not just a message" intent).
    assert!(
        !err.message.is_empty(),
        "SyntaxError.message must not be empty"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// fuzz.test.ts  (400 iterations each, seeded mulberry32)
// ─────────────────────────────────────────────────────────────────────────────

const ITERATIONS: usize = 400;

/// Port of: GQL fuzz: Kleene boolean-algebra laws
/// "OR is the union of trues; De Morgan; AND commutes; NOT involutes"
///
/// For each seed: generates a random graph and two random predicates P, Q, then
/// asserts:
///   1. ids_where(P OR Q) == sorted_union(ids_where(P), ids_where(Q))
///   2. ids_where(NOT (P AND Q)) == ids_where((NOT P) OR (NOT Q))  [De Morgan]
///   3. ids_where(P AND Q) == ids_where(Q AND P)                   [AND commutes]
///   4. ids_where(NOT (NOT P)) == ids_where(P)                     [double negation]
#[test]
fn x_fuzz_kleene_boolean_algebra_laws() {
    for i in 0..ITERATIONS {
        let seed: u32 = 0x5eed_0000u32.wrapping_add(i as u32);
        let mut rng = Mulberry32::new(seed);
        let mut g = make_fuzz_graph(&mut rng);
        let p = fuzz_pred(&mut rng, 3);
        let q = fuzz_pred(&mut rng, 3);

        // 1. P OR Q is the union of their true-sets.
        let or_ids = ids_where(&mut g, &format!("{p} OR {q}"));
        let union_ids = sorted_union(&ids_where(&mut g, &p), &ids_where(&mut g, &q));
        assert_eq!(or_ids, union_ids, "OR≠union  seed={seed}\nP={p}\nQ={q}");

        // 2. De Morgan: NOT (P AND Q) == (NOT P) OR (NOT Q)
        let de_m_lhs = ids_where(&mut g, &format!("NOT ({p} AND {q})"));
        let de_m_rhs = ids_where(&mut g, &format!("(NOT {p}) OR (NOT {q})"));
        assert_eq!(de_m_lhs, de_m_rhs, "De Morgan  seed={seed}\nP={p}\nQ={q}");

        // 3. AND commutes.
        let and_pq = ids_where(&mut g, &format!("{p} AND {q}"));
        let and_qp = ids_where(&mut g, &format!("{q} AND {p}"));
        assert_eq!(and_pq, and_qp, "AND commute  seed={seed}\nP={p}\nQ={q}");

        // 4. Double negation involutes.
        let not_not_p = ids_where(&mut g, &format!("NOT (NOT {p})"));
        let just_p = ids_where(&mut g, &p);
        assert_eq!(not_not_p, just_p, "NOT involute  seed={seed}\nP={p}");
    }
}

/// Port of: GQL fuzz: feature cross-consistency
/// "WHERE P ⟺ WHERE (P) IS TRUE ⟺ WHERE CASE over P"
///
/// For each seed: generates a random graph and predicate P, then asserts:
///   1. ids_where("(P) IS TRUE") == ids_where(P)
///   2. ids_where("CASE WHEN P THEN true ELSE false END") == ids_where(P)
///   3. ids_where("(P) IS NOT TRUE").len() == all.len() - base.len()   [complement]
#[test]
fn x_fuzz_feature_cross_consistency() {
    for i in 0..ITERATIONS {
        let seed: u32 = 0xca5e_0000u32.wrapping_add(i as u32);
        let mut rng = Mulberry32::new(seed);
        let mut g = make_fuzz_graph(&mut rng);
        let p = fuzz_pred(&mut rng, 3);

        let base = ids_where(&mut g, &p);

        // P IS TRUE keeps exactly the rows where P is TRUE.
        let is_true = ids_where(&mut g, &format!("({p}) IS TRUE"));
        assert_eq!(is_true, base, "IS TRUE  seed={seed}\nP={p}");

        // CASE WHEN P THEN true ELSE false END
        let case_ids = ids_where(&mut g, &format!("CASE WHEN {p} THEN true ELSE false END"));
        assert_eq!(case_ids, base, "CASE  seed={seed}\nP={p}");

        // P IS NOT TRUE is the exact complement.
        let all_len = ids_where(&mut g, "n.a = n.a OR true").len();
        let is_not_true_len = ids_where(&mut g, &format!("({p}) IS NOT TRUE")).len();
        assert_eq!(
            is_not_true_len,
            all_len - base.len(),
            "complement  seed={seed}\nP={p}"
        );
    }
}

/// Port of: GQL fuzz: EXISTS ⟺ COUNT{} > 0
/// "the existential and the counted subquery agree"
///
/// For each seed: generates a random graph and asserts:
///   1. EXISTS { (n)-[:R]->() } == COUNT { (n)-[:R]->() } > 0
///   2. NOT EXISTS { (n)-[:R]->() } == COUNT { (n)-[:R]->() } = 0
#[test]
fn x_fuzz_exists_equals_count_gt_zero() {
    for i in 0..ITERATIONS {
        let seed: u32 = 0x00c0_0000u32.wrapping_add(i as u32);
        let mut rng = Mulberry32::new(seed);
        let mut g = make_fuzz_graph(&mut rng);

        let exists_ids = ids_where(&mut g, "EXISTS { (n)-[:R]->() }");
        let count_gt_0 = ids_where(&mut g, "COUNT { (n)-[:R]->() } > 0");
        assert_eq!(exists_ids, count_gt_0, "EXISTS=COUNT>0  seed={seed}");

        let not_exists = ids_where(&mut g, "NOT EXISTS { (n)-[:R]->() }");
        let count_eq_0 = ids_where(&mut g, "COUNT { (n)-[:R]->() } = 0");
        assert_eq!(not_exists, count_eq_0, "NOT EXISTS=COUNT=0  seed={seed}");
    }
}

/// Port of: GQL fuzz: compiled plan agrees with the one-shot path
/// "compile(parse(q)) reused matches query(); reuse is stable"
///
/// For each seed: generates a random graph and predicate P, then asserts:
///   1. prepare(q).execute(g) == parse(q).execute(g)   [plan == one-shot]
///   2. running the plan again gives the same result     [pure / stable]
#[test]
fn x_fuzz_compiled_plan_agrees_with_oneshot() {
    for i in 0..ITERATIONS {
        let seed: u32 = 0xc0de_0000u32.wrapping_add(i as u32);
        let mut rng = Mulberry32::new(seed);
        let mut g = make_fuzz_graph(&mut rng);
        let p = fuzz_pred(&mut rng, 3);
        let q_str = format!("MATCH (n:Node) WHERE {p} RETURN element_id(n) AS id ORDER BY id");

        // one-shot
        let one_shot: Vec<Vec<Value>> = parse(&q_str)
            .unwrap_or_else(|e| panic!("parse(one-shot): {e}\nseed={seed}\nq={q_str}"))
            .execute(&mut g, &Params::new())
            .unwrap_or_else(|e| panic!("exec(one-shot): {e}\nseed={seed}"))
            .rows()
            .map(|r| r.to_vec())
            .collect();

        // prepared plan
        let plan = super::prepare(&q_str)
            .unwrap_or_else(|e| panic!("prepare: {e}\nseed={seed}\nq={q_str}"));

        let plan_run1: Vec<Vec<Value>> = plan
            .execute(&mut g, &Params::new())
            .unwrap_or_else(|e| panic!("plan.execute (run 1): {e}\nseed={seed}"))
            .rows()
            .map(|r| r.to_vec())
            .collect();
        assert_eq!(plan_run1, one_shot, "plan≠query  seed={seed}\nq={q_str}");

        // second run of the same plan — must be identical (purity / stability)
        let plan_run2: Vec<Vec<Value>> = plan
            .execute(&mut g, &Params::new())
            .unwrap_or_else(|e| panic!("plan.execute (run 2): {e}\nseed={seed}"))
            .rows()
            .map(|r| r.to_vec())
            .collect();
        assert_eq!(
            plan_run2, one_shot,
            "plan not stable  seed={seed}\nq={q_str}"
        );
    }
}

/// Port of: GQL fuzz: aggregate identities
/// "sum(1) = count(*); count(DISTINCT x) ≤ count(x) ≤ count(*)"
///
/// For each seed: generates a random graph, picks a random property p, runs
///   MATCH (n:Node)
///   RETURN sum(1) AS s, count(*) AS star, count(n.p) AS nonNull, count(DISTINCT n.p) AS dis
/// and asserts:
///   1. s == star
///   2. dis <= nonNull && nonNull <= star
#[test]
fn x_fuzz_aggregate_identities() {
    for i in 0..ITERATIONS {
        let seed: u32 = 0x0a66_0000u32.wrapping_add(i as u32);
        let mut rng = Mulberry32::new(seed);
        let mut g = make_fuzz_graph(&mut rng);
        let props = ["a", "b", "c"];
        let p = props[(rng.next() * 3.0) as usize];

        let q_str = format!(
            "MATCH (n:Node) RETURN sum(1) AS s, count(*) AS star, count(n.{p}) AS nonNull, count(DISTINCT n.{p}) AS dis"
        );
        let rs = parse(&q_str)
            .unwrap_or_else(|e| panic!("parse: {e}\nseed={seed}"))
            .execute(&mut g, &Params::new())
            .unwrap_or_else(|e| panic!("exec: {e}\nseed={seed}"));
        let row: Vec<Value> = rs.rows().next().unwrap().to_vec();

        // sum(1) must equal count(*)
        assert_eq!(row[0], row[1], "sum(1)=count(*)  seed={seed} prop={p}");

        // count ordering: dis <= nonNull <= star
        let star = match row[1] {
            Value::Num(x) => x,
            _ => panic!("star not a number"),
        };
        let non_null = match row[2] {
            Value::Num(x) => x,
            _ => panic!("nonNull not a number"),
        };
        let dis = match row[3] {
            Value::Num(x) => x,
            _ => panic!("dis not a number"),
        };
        assert!(
            dis <= non_null && non_null <= star,
            "count ordering  seed={seed} prop={p}  dis={dis} nonNull={non_null} star={star}"
        );
    }
}

/// Port of: GQL fuzz: ORDER BY + SKIP/LIMIT slicing
/// "SKIP s LIMIT l equals slicing the fully ordered result"
///
/// For each seed: generates a random graph, picks a random property p and
/// direction, builds the full ORDER BY result, picks random s and l, then
/// asserts the paginated query equals full[s..s+l].
#[test]
fn x_fuzz_order_by_skip_limit_slicing() {
    for i in 0..ITERATIONS {
        let seed: u32 = 0x005d_0000u32.wrapping_add(i as u32);
        let mut rng = Mulberry32::new(seed);
        let mut g = make_fuzz_graph(&mut rng);
        let props = ["a", "b", "c"];
        let p = props[(rng.next() * 3.0) as usize];
        let dir = if rng.next() < 0.5 { "ASC" } else { "DESC" };

        // element_id is a total-order tiebreak so the full order is deterministic.
        let order = format!("ORDER BY n.{p} {dir}, element_id(n)");
        let full_q = format!("MATCH (n:Node) RETURN element_id(n) AS id {order}");

        let full_ids: Vec<String> = parse(&full_q)
            .unwrap_or_else(|e| panic!("parse full: {e}\nseed={seed}"))
            .execute(&mut g, &Params::new())
            .unwrap_or_else(|e| panic!("exec full: {e}\nseed={seed}"))
            .rows()
            .map(|row| match &row[0] {
                Value::Str(s) => s.to_string(),
                v => format!("{v:?}"),
            })
            .collect();

        let full_len = full_ids.len();
        let s = (rng.next() * (full_len as f64 + 2.0)) as usize;
        let l = (rng.next() * (full_len as f64 + 2.0)) as usize;

        let paged_q =
            format!("MATCH (n:Node) RETURN element_id(n) AS id {order} SKIP {s} LIMIT {l}");
        let paged_ids: Vec<String> = parse(&paged_q)
            .unwrap_or_else(|e| panic!("parse paged: {e}\nseed={seed}"))
            .execute(&mut g, &Params::new())
            .unwrap_or_else(|e| panic!("exec paged: {e}\nseed={seed}"))
            .rows()
            .map(|row| match &row[0] {
                Value::Str(sv) => sv.to_string(),
                v => format!("{v:?}"),
            })
            .collect();

        let end = (s + l).min(full_len);
        let expected: Vec<String> = if s >= full_len {
            vec![]
        } else {
            full_ids[s..end].to_vec()
        };

        assert_eq!(
            paged_ids, expected,
            "slice  seed={seed}\n{order} SKIP {s} LIMIT {l}"
        );
    }
}

/// Port of: GQL fuzz: set-operation laws
/// "UNION ALL is additive; EXCEPT self is empty; INTERSECT self is DISTINCT self"
///
/// For each seed: generates a random graph and two predicates A, B, then asserts:
///   1. (A UNION ALL B).len() == A.len() + B.len()
///   2. (A EXCEPT A) == []
///   3. sort(A INTERSECT A) == sort(DISTINCT ids of A)
#[test]
fn x_fuzz_set_operation_laws() {
    for i in 0..ITERATIONS {
        let seed: u32 = 0x05e7_0000u32.wrapping_add(i as u32);
        let mut rng = Mulberry32::new(seed);
        let mut g = make_fuzz_graph(&mut rng);
        let pred_a = fuzz_pred(&mut rng, 2);
        let pred_b = fuzz_pred(&mut rng, 2);

        let a_q = format!("MATCH (n:Node) WHERE {pred_a} RETURN element_id(n) AS id");
        let b_q = format!("MATCH (n:Node) WHERE {pred_b} RETURN element_id(n) AS id");

        let run = |g: &mut Graph, qs: &str| -> Vec<Vec<Value>> {
            parse(qs)
                .unwrap_or_else(|e| panic!("parse: {e}\nseed={seed}\n{qs}"))
                .execute(g, &Params::new())
                .unwrap_or_else(|e| panic!("exec: {e}\nseed={seed}"))
                .rows()
                .map(|r| r.to_vec())
                .collect()
        };

        let a_rows = run(&mut g, &a_q);
        let b_rows = run(&mut g, &b_q);

        // 1. UNION ALL is additive.
        let union_all = run(&mut g, &format!("{a_q} UNION ALL {b_q}"));
        assert_eq!(
            union_all.len(),
            a_rows.len() + b_rows.len(),
            "UNION ALL additive  seed={seed}"
        );

        // 2. EXCEPT self is empty.
        let except_self = run(&mut g, &format!("{a_q} EXCEPT {a_q}"));
        assert!(
            except_self.is_empty(),
            "EXCEPT self empty  seed={seed}\ngot {except_self:?}"
        );

        // 3. INTERSECT self equals DISTINCT self.
        let intersect_self = run(&mut g, &format!("{a_q} INTERSECT {a_q}"));
        let mut inter_ids: Vec<String> = intersect_self
            .iter()
            .map(|r| match &r[0] {
                Value::Str(s) => s.to_string(),
                v => format!("{v:?}"),
            })
            .collect();
        inter_ids.sort();

        let mut distinct_a: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            a_rows
                .iter()
                .filter_map(|r| match &r[0] {
                    Value::Str(s) => {
                        let sv = s.to_string();
                        if seen.insert(sv.clone()) {
                            Some(sv)
                        } else {
                            None
                        }
                    }
                    v => {
                        let sv = format!("{v:?}");
                        if seen.insert(sv.clone()) {
                            Some(sv)
                        } else {
                            None
                        }
                    }
                })
                .collect()
        };
        distinct_a.sort();

        assert_eq!(inter_ids, distinct_a, "INTERSECT self  seed={seed}");
    }
}

/// Port of: GQL fuzz: arithmetic precedence matches a reference evaluator
/// "random +,-,* expressions evaluate to the precedence-correct value"
///
/// Generates flat expressions `a*b + c*d - e*f` (no parens) where * binds
/// tighter than +/-. The reference value is computed with that precedence so a
/// parser/evaluator bug would diverge. Uses an empty graph (RETURN only).
#[test]
fn x_fuzz_arithmetic_precedence() {
    let mut g = crate::graph::Builder::default().finalize();

    for i in 0..ITERATIONS {
        let seed: u32 = 0xa817_0000u32.wrapping_add(i as u32);
        let mut rng = Mulberry32::new(seed);

        let (expr_str, expected_val) = gen_flat_arith(&mut rng);

        let q_str = format!("RETURN {expr_str} AS r");
        let rs = parse(&q_str)
            .unwrap_or_else(|e| panic!("parse: {e}\nseed={seed}\n{q_str}"))
            .execute(&mut g, &Params::new())
            .unwrap_or_else(|e| panic!("exec: {e}\nseed={seed}"));

        let got = rs.rows().next().unwrap()[0].clone();
        assert_eq!(
            got,
            Value::Num(expected_val as f64),
            "arith  seed={seed}  {expr_str}"
        );
    }
}

/// Port of TS `genFlat`: a flat expression of terms joined by +/-, each term a
/// product of 1..3 integers from 1..5. Returns (expression string, expected value).
fn gen_flat_arith(rng: &mut Mulberry32) -> (String, i64) {
    let n_terms = 1 + (rng.next() * 3.0) as usize;
    let mut terms: Vec<String> = Vec::with_capacity(n_terms);
    let mut vals: Vec<i64> = Vec::with_capacity(n_terms);

    for _ in 0..n_terms {
        let n_factors = 1 + (rng.next() * 3.0) as usize;
        let mut factors: Vec<String> = Vec::with_capacity(n_factors);
        let mut v: i64 = 1;
        for _ in 0..n_factors {
            let x = 1 + (rng.next() * 5.0) as i64;
            factors.push(x.to_string());
            v *= x;
        }
        terms.push(factors.join(" * "));
        vals.push(v);
    }

    let mut expr = terms[0].clone();
    let mut val = vals[0];
    for t in 1..n_terms {
        let plus = rng.next() < 0.5;
        if plus {
            expr.push_str(" + ");
            val += vals[t];
        } else {
            expr.push_str(" - ");
            val -= vals[t];
        }
        expr.push_str(&terms[t]);
    }

    (expr, val)
}
