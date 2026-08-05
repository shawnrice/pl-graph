//! The lowered intermediate representation (IR) and the `lower` pass.
//!
//! A parsed [`Query`](super::ast::Query) is *lowered* once into a `CQuery` that
//! bakes every graph- and param-independent decision: `$param` → a positional
//! slot, **variable → a binding slot**, function name → an enum, aggregate
//! detection, projection column names and group keys. This is the artifact a
//! [`prepared`](super::prepare) statement holds, paid once and reused.
//!
//! Variable slots: a `Scope` maps each in-scope variable name to an index into a
//! `Vec<Option<Val>>` binding, so the per-row hot path indexes an array instead
//! of scanning a name list. `WITH` starts a fresh scope (its output columns);
//! correlated sub-queries extend the scope with their own pattern variables.
//!
//! Graph-dependent resolution (property key → id) stays at execute time — the
//! graph is mutable and key ids are graph-specific.

use std::sync::Arc;

use super::ast::*;

/// A variable reference that resolves to no in-scope slot reads as NULL.
pub const UNBOUND: usize = usize::MAX;

/// Scalar (non-aggregate) functions, resolved from a name once. `Unknown` keeps
/// the engine total (an unknown function evaluates to NULL, as before).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarFn {
    Abs,
    Ceil,
    Floor,
    Sqrt,
    Exp,
    Ln,
    Log10,
    Sin,
    Cos,
    Tan,
    Cot,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Degrees,
    Radians,
    Round,
    Sign,
    Pi,
    E,
    Upper,
    Lower,
    Trim,
    Ltrim,
    Rtrim,
    CharLength,
    Power,
    Mod,
    Log,
    Atan2,
    Size,
    Left,
    Right,
    Coalesce,
    Nullif,
    ElementId,
    // Graph functions.
    Labels,
    Type,
    Keys,
    // Path functions (ISO GQL): the vertices, the edges, and the interleaved
    // vertex/edge sequence of a bound path. Hop count reuses `Size`.
    PathNodes,
    PathEdges,
    PathElements,
    // Conversion.
    ToString,
    ToInteger,
    ToFloat,
    ToBoolean,
    ToList,
    // String predicates / measurement.
    Contains,
    StartsWith,
    EndsWith,
    ByteLength,
    // String / list.
    Substring,
    Split,
    Replace,
    Head,
    Last,
    Reverse,
    Tail,
    Append,
    Range,
    ListUnion,
    Intersection,
    Difference,
    ListContains,
    ListSort,
    // Temporal constructors: parse a string (or convert a temporal) into a
    // `DATE` / `LOCAL DATETIME` / `DURATION`.
    DateOf,
    LocalTimeOf,
    DateTimeOf,
    ZonedTimeOf,
    ZonedDateTimeOf,
    DurationOf,
    /// `duration_between(a, b)` — the EXACT elapsed span (a measurement between
    /// two pinned points), never calendar months: whole days for two dates,
    /// seconds+nanos for two datetimes.
    DurationBetween,
    /// Temporal component extraction (`_year(x)`/`_month`/`_day`/`_hour`/
    /// `_minute`/`_second`). **A lenke extension** — date-part extraction is NOT
    /// in the ISO GQL function catalogue (not mandatory, not a catalogued optional
    /// feature; verified against the 39075 Feature-ID taxonomy), so it wears the
    /// leading-underscore sigil per `docs/design/gql-extensions.md`. Argument must
    /// be a temporal value carrying the requested component (a string is NOT
    /// coerced — it faults); zoned values are read in their own offset (local wall
    /// clock). NOT SQL `EXTRACT`, NOT Cypher's `.year` accessor.
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggFn {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    CollectList,
    /// ISO ordered-set aggregates: `percentile_cont(value, fraction)` (linear
    /// interpolation between ranks) and `percentile_disc(value, fraction)` (the
    /// value at the smallest rank whose cumulative fraction ≥ `fraction`). The
    /// fraction is carried on [`CAgg::frac`] / [`CExpr::Aggregate`].
    PercentileCont,
    PercentileDisc,
    /// ISO population / sample standard deviation over a group's numeric values.
    /// Both are one-pass (running count, sum, sum-of-squares). `stddev_pop` divides
    /// the summed squared deviations by `n` (null over 0 rows); `stddev_samp` by
    /// `n-1` (null over fewer than 2 rows).
    StddevPop,
    StddevSamp,
}

fn agg_fn(name: &str) -> Option<AggFn> {
    Some(match name {
        "count" => AggFn::Count,
        "sum" => AggFn::Sum,
        "avg" => AggFn::Avg,
        "min" => AggFn::Min,
        "max" => AggFn::Max,
        "collect_list" => AggFn::CollectList,
        "percentile_cont" => AggFn::PercentileCont,
        "percentile_disc" => AggFn::PercentileDisc,
        "stddev_pop" => AggFn::StddevPop,
        "stddev_samp" => AggFn::StddevSamp,
        _ => return None,
    })
}

fn scalar_fn(name: &str) -> ScalarFn {
    match name {
        "abs" => ScalarFn::Abs,
        "ceil" | "ceiling" => ScalarFn::Ceil,
        "floor" => ScalarFn::Floor,
        "sqrt" => ScalarFn::Sqrt,
        "exp" => ScalarFn::Exp,
        "ln" => ScalarFn::Ln,
        "log10" => ScalarFn::Log10,
        "sin" => ScalarFn::Sin,
        "cos" => ScalarFn::Cos,
        "tan" => ScalarFn::Tan,
        "cot" => ScalarFn::Cot,
        "asin" => ScalarFn::Asin,
        "acos" => ScalarFn::Acos,
        "atan" => ScalarFn::Atan,
        "sinh" => ScalarFn::Sinh,
        "cosh" => ScalarFn::Cosh,
        "tanh" => ScalarFn::Tanh,
        "degrees" => ScalarFn::Degrees,
        "radians" => ScalarFn::Radians,
        "round" => ScalarFn::Round,
        "sign" => ScalarFn::Sign,
        "pi" => ScalarFn::Pi,
        "e" => ScalarFn::E,
        "upper" => ScalarFn::Upper,
        "lower" => ScalarFn::Lower,
        "trim" | "btrim" => ScalarFn::Trim,
        "ltrim" => ScalarFn::Ltrim,
        "rtrim" => ScalarFn::Rtrim,
        "char_length" | "character_length" => ScalarFn::CharLength,
        "power" => ScalarFn::Power,
        "mod" => ScalarFn::Mod,
        "log" => ScalarFn::Log,
        "atan2" => ScalarFn::Atan2,
        // `cardinality` is the ISO GQL / SQL name for a collection's element count;
        // `size` is the openCypher spelling (`length`/`path_length` alias paths).
        "size" | "cardinality" | "length" | "path_length" => ScalarFn::Size,
        "left" => ScalarFn::Left,
        "right" => ScalarFn::Right,
        "coalesce" => ScalarFn::Coalesce,
        "nullif" => ScalarFn::Nullif,
        "element_id" => ScalarFn::ElementId,
        "labels" => ScalarFn::Labels,
        "type" => ScalarFn::Type,
        // `property_names` is the ISO GQL name for an element's property-name list;
        // `keys` is the openCypher spelling of the same thing.
        "keys" | "property_names" => ScalarFn::Keys,
        "nodes" => ScalarFn::PathNodes,
        // `edges` is the ISO GQL name for a path's edge list. Cypher's
        // `relationships` is deliberately NOT accepted — GQL's element
        // vocabulary is node/edge (Spanner + Fabric GQL both use `edges`).
        "edges" => ScalarFn::PathEdges,
        "elements" => ScalarFn::PathElements,
        "tostring" | "to_string" => ScalarFn::ToString,
        "tointeger" | "to_integer" => ScalarFn::ToInteger,
        "tofloat" | "to_float" => ScalarFn::ToFloat,
        "toboolean" | "to_boolean" => ScalarFn::ToBoolean,
        "tolist" | "to_list" => ScalarFn::ToList,
        "contains" => ScalarFn::Contains,
        "starts_with" => ScalarFn::StartsWith,
        "ends_with" => ScalarFn::EndsWith,
        "byte_length" | "octet_length" => ScalarFn::ByteLength,
        "substring" => ScalarFn::Substring,
        "split" => ScalarFn::Split,
        "replace" => ScalarFn::Replace,
        "head" => ScalarFn::Head,
        "last" => ScalarFn::Last,
        "reverse" => ScalarFn::Reverse,
        "tail" => ScalarFn::Tail,
        "append" => ScalarFn::Append,
        "range" => ScalarFn::Range,
        "list_union" => ScalarFn::ListUnion,
        "intersection" => ScalarFn::Intersection,
        "difference" => ScalarFn::Difference,
        "list_contains" => ScalarFn::ListContains,
        "list_sort" => ScalarFn::ListSort,
        "date" => ScalarFn::DateOf,
        "local_time" => ScalarFn::LocalTimeOf,
        "zoned_time" => ScalarFn::ZonedTimeOf,
        "zoned_datetime" => ScalarFn::ZonedDateTimeOf,
        "local_datetime" | "datetime" => ScalarFn::DateTimeOf,
        "duration" => ScalarFn::DurationOf,
        "duration_between" => ScalarFn::DurationBetween,
        // Temporal component extraction — a lenke EXTENSION (not in the ISO GQL
        // function catalogue), so sigil-prefixed. The bare `year`/… names stay
        // unknown, signalling non-portability at the call site.
        "_year" => ScalarFn::Year,
        "_month" => ScalarFn::Month,
        "_day" => ScalarFn::Day,
        "_hour" => ScalarFn::Hour,
        "_minute" => ScalarFn::Minute,
        "_second" => ScalarFn::Second,
        _ => ScalarFn::Unknown,
    }
}

/// A lowered label expression: each label name is a `ref` index resolved once
/// per execution to a (vertex-label id, edge-type id) pair (a name can be both).
#[derive(Debug, Clone)]
pub enum CLabelExpr {
    Label(usize),
    Wildcard,
    Not(Box<Self>),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
}

/// Lowered expression. Variables and properties carry a binding slot; `$param` a
/// positional slot; property keys and label names a ref resolved per execution;
/// functions a resolved enum tag.
#[derive(Debug, Clone)]
pub enum CExpr {
    Var(usize),
    Param(usize),
    Prop {
        var_slot: usize,
        key_ref: usize,
    },
    /// A field access on a stored record: `n.meta.city` (or `n.meta['city']`).
    /// The `Prop`-rooted `Field`/`Index` chain is collapsed here at plan time so
    /// eval navigates the stored `Value` in place and materializes ONLY the leaf,
    /// instead of cloning the whole `meta` map into a `Val::Record` first. `descent`
    /// is the field-name path after the root property.
    PropField {
        var_slot: usize,
        root_key_ref: usize,
        descent: Vec<Arc<str>>,
    },
    /// `PROPERTY_EXISTS(n, key)` — a presence test; resolves the element + key
    /// exactly like `Prop`, but yields a `Bool` (or `Null` on a non-element).
    PropertyExists {
        var_slot: usize,
        key_ref: usize,
    },
    Lit(Lit),
    List(Vec<Self>),
    /// ISO `<record constructor>` — `{ field: expr, … }`. Field name → value.
    Record(Vec<(Arc<str>, Self)>),
    /// ISO GQL list element access `base[index]` — 0-based; out of range → null.
    Index {
        base: Box<Self>,
        index: Box<Self>,
    },
    /// Property access on an arbitrary expression (`base.key`) — the postfix
    /// chain form (`edges(p)[0].amount`). `key_ref` resolves like `Prop`.
    Field {
        base: Box<Self>,
        key_ref: usize,
        /// The field name as a string — used when the base is a record/map (the
        /// `key_ref` only resolves against a graph element's property columns).
        name: Arc<str>,
    },
    Compare {
        op: CompareOp,
        left: Box<Self>,
        right: Box<Self>,
    },
    /// n-ary left-associative arithmetic run (mirrors `ast::Expr::Arith`): a flat
    /// `Vec`, not a chain-deep tree, so eval/drop never overflow the stack (C1).
    Arith {
        head: Box<Self>,
        tail: Vec<(ArithOp, Self)>,
    },
    /// n-ary left-associative string-concat run.
    Concat(Vec<Self>),
    Neg(Box<Self>),
    /// n-ary boolean runs (three-valued folds; same-operator run flattened).
    And(Vec<Self>),
    Or(Vec<Self>),
    Xor(Vec<Self>),
    Not(Box<Self>),
    IsNull {
        expr: Box<Self>,
        negated: bool,
    },
    IsTruth {
        expr: Box<Self>,
        truth: Option<bool>,
        negated: bool,
    },
    IsLabeled {
        expr: Box<Self>,
        label: CLabelExpr,
        negated: bool,
    },
    /// `x IS [NOT] TYPED <value type> [NOT NULL]` — the ISO value-type predicate.
    IsTyped {
        expr: Box<Self>,
        ty: TypeTest,
        not_null: bool,
        negated: bool,
    },
    /// A graph-element predicate (`IS DIRECTED` / `IS SOURCE|DESTINATION OF` /
    /// `ALL_DIFFERENT` / `SAME`). Resolves like a Vec-args scalar; tree-walked.
    GraphPred {
        kind: super::ast::GraphPredKind,
        args: Vec<Self>,
        negated: bool,
    },
    In {
        expr: Box<Self>,
        list: Box<Self>,
        negated: bool,
    },
    /// Correlated sub-pattern existence; `sub_len` is the sub-scope slot count.
    Exists {
        patterns: Vec<CPath>,
        where_: Option<Box<Self>>,
        sub_len: usize,
    },
    CountSubquery {
        patterns: Vec<CPath>,
        where_: Option<Box<Self>>,
        sub_len: usize,
    },
    ValueSubquery {
        patterns: Vec<CPath>,
        where_: Option<Box<Self>>,
        ret: Box<Self>,
        /// RETURN is (or contains) an aggregate → fold over the group rather than
        /// enforcing the one-row cardinality rule.
        is_agg: bool,
        sub_len: usize,
    },
    /// `LET (slot = expr)+ IN body END` — each binding is written into a per-eval
    /// clone of the row binding (left-to-right, later sees earlier), then `body`
    /// is evaluated against it.
    LetIn {
        bindings: Vec<(usize, Box<Self>)>,
        body: Box<Self>,
    },
    Case {
        subject: Option<Box<Self>>,
        whens: Vec<(Self, Self)>,
        else_: Option<Box<Self>>,
    },
    Scalar {
        func: ScalarFn,
        args: Vec<Self>,
    },
    Aggregate {
        func: AggFn,
        arg: Option<Box<Self>>,
        distinct: bool,
        star: bool,
        /// The literal fraction for `percentile_cont` / `percentile_disc` (already
        /// clamped to `[0, 1]`); `None` for every other aggregate.
        frac: Option<f64>,
    },
    /// Reference to a projection's `i`th extracted aggregate (its folded value).
    /// Projection/ORDER BY expressions have their aggregates lifted out into
    /// `CProjection::aggs` and replaced by these, so a group folds incrementally.
    AggRef(usize),
}

/// A bytecode instruction for the expression VM (a stack machine). Compiled from
/// `CExpr` once at lower time; executed by a flat loop over a `Vec<Op>` against a
/// small operand stack — contiguous instructions instead of a pointer-chased
/// boxed tree. `Tree` is the escape hatch: control-flow / subquery / aggregate
/// nodes fall back to the tree-walking interpreter for that subexpression.
#[derive(Debug, Clone)]
pub enum Op {
    Const(Lit),
    Var(usize),
    Param(usize),
    Prop {
        var_slot: usize,
        key_ref: usize,
    },
    MakeList(usize),
    Arith(ArithOp),
    Compare(CompareOp),
    Concat,
    Neg,
    Not,
    And,
    Or,
    Xor,
    IsNull(bool),
    IsTruth(Option<bool>, bool),
    IsLabeled(CLabelExpr, bool),
    In(bool),
    Scalar(ScalarFn, usize),
    AggRef(usize),
    /// Fall back to the tree-walk for this subexpression (CASE / EXISTS / COUNT{}
    /// / aggregate) and push its value.
    Tree(CExpr),
}

/// A compiled expression: a flat instruction stream for the VM.
#[derive(Debug, Clone)]
pub struct Program(pub Vec<Op>);

impl Program {
    /// Push every input slot this program READS into `out`.
    ///
    /// Returns `false` if the program contains an opaque [`Op::Tree`] — a
    /// subquery or aggregate the flattener kept as an expression, whose slots
    /// this cannot see. A caller using the result to SKIP work must then assume
    /// every slot is read: under-reporting here would silently drop a column,
    /// and a dropped group-variable column reads back as `null` rather than
    /// failing.
    #[must_use]
    pub fn read_slots(&self, out: &mut Vec<usize>) -> bool {
        for op in &self.0 {
            match op {
                Op::Var(s) => out.push(*s),
                Op::Prop { var_slot, .. } => out.push(*var_slot),
                Op::Tree(_) => return false,
                _ => {}
            }
        }

        true
    }
}

fn emit(e: &CExpr, out: &mut Vec<Op>) {
    match e {
        CExpr::Lit(l) => out.push(Op::Const(l.clone())),
        CExpr::Var(s) => out.push(Op::Var(*s)),
        CExpr::Param(s) => out.push(Op::Param(*s)),
        CExpr::Prop { var_slot, key_ref } => out.push(Op::Prop {
            var_slot: *var_slot,
            key_ref: *key_ref,
        }),
        CExpr::List(items) => {
            for it in items {
                emit(it, out);
            }
            out.push(Op::MakeList(items.len()));
        }
        // n-ary nodes emit a left-associative sequence of binary stack ops:
        // `head e1 <op1> e2 <op2> …`, computing the same fold as the evaluator.
        CExpr::Arith { head, tail } => {
            emit(head, out);
            for (op, e) in tail {
                emit(e, out);
                out.push(Op::Arith(*op));
            }
        }
        CExpr::Compare { op, left, right } => {
            emit(left, out);
            emit(right, out);
            out.push(Op::Compare(*op));
        }
        CExpr::Concat(items) => {
            emit(&items[0], out);
            for e in &items[1..] {
                emit(e, out);
                out.push(Op::Concat);
            }
        }
        CExpr::Neg(x) => {
            emit(x, out);
            out.push(Op::Neg);
        }
        CExpr::And(items) => {
            emit(&items[0], out);
            for e in &items[1..] {
                emit(e, out);
                out.push(Op::And);
            }
        }
        CExpr::Or(items) => {
            emit(&items[0], out);
            for e in &items[1..] {
                emit(e, out);
                out.push(Op::Or);
            }
        }
        CExpr::Xor(items) => {
            emit(&items[0], out);
            for e in &items[1..] {
                emit(e, out);
                out.push(Op::Xor);
            }
        }
        CExpr::Not(x) => {
            emit(x, out);
            out.push(Op::Not);
        }
        CExpr::IsNull { expr, negated } => {
            emit(expr, out);
            out.push(Op::IsNull(*negated));
        }
        CExpr::IsTruth {
            expr,
            truth,
            negated,
        } => {
            emit(expr, out);
            out.push(Op::IsTruth(*truth, *negated));
        }
        CExpr::IsLabeled {
            expr,
            label,
            negated,
        } => {
            emit(expr, out);
            out.push(Op::IsLabeled(label.clone(), *negated));
        }
        CExpr::In {
            expr,
            list,
            negated,
        } => {
            emit(expr, out);
            emit(list, out);
            out.push(Op::In(*negated));
        }
        CExpr::Scalar { func, args } => {
            for a in args {
                emit(a, out);
            }
            out.push(Op::Scalar(*func, args.len()));
        }
        CExpr::AggRef(i) => out.push(Op::AggRef(*i)),
        // Control flow / subquery / aggregate / list subscript: tree-walk it.
        CExpr::Case { .. }
        | CExpr::Exists { .. }
        | CExpr::CountSubquery { .. }
        | CExpr::ValueSubquery { .. }
        | CExpr::LetIn { .. }
        | CExpr::Record(_)
        | CExpr::Index { .. }
        | CExpr::Field { .. }
        | CExpr::PropField { .. }
        | CExpr::PropertyExists { .. }
        | CExpr::IsTyped { .. }
        | CExpr::GraphPred { .. }
        | CExpr::Aggregate { .. } => out.push(Op::Tree(e.clone())),
    }
}

/// Compile a lowered expression to a VM `Program`.
pub fn compile_program(e: &CExpr) -> Program {
    let mut out = Vec::new();
    emit(e, &mut out);
    Program(out)
}

/// An aggregate lifted out of a projection expression (folded once per group).
#[derive(Debug, Clone)]
pub struct CAgg {
    pub func: AggFn,
    pub arg: Option<CExpr>,
    pub distinct: bool,
    pub star: bool,
    /// Percentile fraction (clamped to `[0, 1]`); `None` for other aggregates.
    pub frac: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct CPropConstraint {
    /// Key name (for INSERT, which creates the property) …
    pub key: String,
    /// … and its resolved ref (for MATCH, which reads the property).
    pub key_ref: usize,
    pub value: CExpr,
}

#[derive(Debug, Clone)]
pub struct CNode {
    /// Binding slot this node's variable occupies (`None` if anonymous).
    pub var_slot: Option<usize>,
    pub label: Option<CLabelExpr>,
    pub props: Vec<CPropConstraint>,
    pub where_: Option<CExpr>,
}

#[derive(Debug, Clone)]
pub struct CRel {
    pub var_slot: Option<usize>,
    pub label: Option<CLabelExpr>,
    pub direction: Direction,
    pub props: Vec<CPropConstraint>,
    pub where_: Option<CExpr>,
    pub quantifier: Option<Quantifier>,
}

/// The repetition UNIT of an ISO quantified parenthesized subpath
/// `((x)-[e1]->(m)-[e2]->(y) [WHERE]){n,m}`: a fixed linear sub-path repeated k ≥ 1
/// hops. Every inner variable is a GROUP variable (exposed to the outer query as
/// the list of each repetition's value). A single-edge subpath is a **one-hop
/// unit** — the SAME matcher, just `k = 1` — so the single and multi cases can't
/// drift.
#[derive(Debug, Clone)]
pub struct CUnit {
    /// The unit's linear element sequence — each a single hop or a nested quantified
    /// sub-unit. A plain linear subpath is all `Hop`s.
    pub elems: Vec<CElem>,
    /// The unit SOURCE node `(x)`'s group-variable slot (`None` if anonymous).
    pub start_slot: Option<usize>,
    /// The per-unit predicate (the subpath `WHERE`), checked once the whole unit is
    /// matched (every inner variable bound), so it can reference any of them.
    pub where_: Option<CExpr>,
}

impl CUnit {
    /// Whether this unit binds any GROUP variable (source, an edge, a hop target, or
    /// anything a nested sub-unit binds). When it binds nothing, the walk's
    /// `verts`/`edges` need not be reconstructed — the matcher can skip the per-end
    /// path rebuild (the abbreviated form's speed).
    pub fn exposes(&self) -> bool {
        self.start_slot.is_some()
            || self.elems.iter().any(|e| match e {
                CElem::Hop(h) => h.target_slot.is_some() || h.rel.var_slot.is_some(),
                // A `Sub` exposes its LANDING (`y` in `-[]->{a,b}(y)`, a flat list at this
                // unit's depth) and — for a nested parenthesized subpath — its inner unit's
                // own variables (one list level deeper). An anonymous-endpoint `Sub` with a
                // bare per-hop edge predicate exposes nothing.
                CElem::Sub(s) => s.target_slot.is_some() || s.unit.exposes(),
            })
    }

    /// Every slot this unit binds as a group variable, in a stable order.
    ///
    /// Exactly the slots [`CUnit::exposes`] answers `true` for, enumerated — the
    /// columnar scan needs to know WHICH columns a repetition produces, not just
    /// that it produces some.
    pub fn group_slots(&self, out: &mut Vec<usize>) {
        out.extend(self.start_slot);
        for e in &self.elems {
            match e {
                CElem::Hop(h) => {
                    out.extend(h.rel.var_slot);
                    out.extend(h.target_slot);
                }
                CElem::Sub(s) => {
                    out.extend(s.target_slot);
                    s.unit.group_slots(out);
                }
            }
        }
    }

    /// Whether every element is a plain `Hop` (no nested `Sub`). A flat unit has a fixed
    /// `k` hops per rep, so its group variables can be bound by the cheap `k`-stride over
    /// the flat walk — the hot path — instead of the general structured binder.
    pub fn is_flat(&self) -> bool {
        self.elems.iter().all(|e| matches!(e, CElem::Hop(_)))
    }

    /// The element at `i` as a hop (a flat unit's elements are all hops).
    pub fn hop(&self, i: usize) -> &CHop {
        match &self.elems[i] {
            CElem::Hop(h) => h,
            CElem::Sub(_) => unreachable!("a nested sub-unit is not bound by the k-stride path"),
        }
    }
}

/// One element of a unit's linear sequence: a single edge, or a nested quantified
/// sub-unit (repeated `[min, max]` times — the general `( … ){n,m}` nesting).
#[derive(Debug, Clone)]
pub enum CElem {
    Hop(CHop),
    Sub(CSub),
}

#[derive(Debug, Clone)]
pub struct CHop {
    /// The relationship (for expansion + its own inline label/property filter).
    pub rel: CRel,
    /// The hop's target node's group-variable slot (`None` if anonymous).
    pub target_slot: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct CSub {
    pub unit: Box<CUnit>,
    pub min: u32,
    pub max: Option<u32>,
    /// The subpath's LANDING node variable (`y` in `( (x)-[]->{1,3}(y) ){2}`) — the
    /// vertex after the whole sub-unit completes, exposed as a group variable at the
    /// enclosing unit's depth. `None` until nested exposure is compiled in.
    pub target_slot: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct CSegment {
    pub rel: CRel,
    pub node: CNode,
    /// For a quantified parenthesized subpath: the repetition unit (`rel.quantifier`
    /// carries the bounds; `node` is the outer landing endpoint). `None` for a plain
    /// hop or the abbreviated `-[e]->{n,m}` form.
    pub unit: Option<CUnit>,
}

#[derive(Debug, Clone)]
pub struct CPath {
    pub start: CNode,
    pub segments: Vec<CSegment>,
    /// Slot the whole path binds to (`p = …`), or `None` if the path is unnamed.
    pub path_var_slot: Option<usize>,
    /// Which matching paths to keep (`Walk` = all; `AnyShortest` = one per pair).
    pub selector: PathSelector,
    /// The repeated-element restrictor on a var-length walk (`Trail` default =
    /// no repeated edges; `Simple`/`Acyclic` = no repeated nodes; `Walk` = none).
    pub mode: PathMode,
}

#[derive(Debug, Clone)]
pub struct CReturnItem {
    pub expr: CExpr,
    /// Compiled form of `expr` for the stack-machine VM (hot per-row site).
    pub prog: Program,
    pub name: String,
    pub is_agg: bool,
}

#[derive(Debug, Clone)]
pub struct CSortItem {
    pub expr: CExpr,
    pub descending: bool,
    pub nulls_first: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct CProjection {
    pub star: bool,
    pub distinct: bool,
    pub items: Vec<CReturnItem>,
    /// True when any item aggregates OR `GROUP BY` is present → grouping.
    pub aggregating: bool,
    /// Explicit `GROUP BY` keys (compiled). When non-empty they DRIVE grouping;
    /// empty → implicit grouping by the non-aggregate items. See
    /// [`CProjection::group_keys`].
    pub group_by: Vec<CReturnItem>,
    /// Aggregates lifted out of the item/ORDER BY expressions, folded per group;
    /// the expressions reference them via [`CExpr::AggRef`].
    pub aggs: Vec<CAgg>,
    /// Output slot count (= column count). Output slot `i` holds column `i`.
    pub out_len: usize,
    /// Output column names, indexed by output slot.
    pub out_names: Vec<String>,
    /// For `*`: the input slots to carry across (aligned with `out_names`).
    pub star_cols: Vec<usize>,
    pub order_by: Vec<CSortItem>,
    /// Input slots appended after the output slots to form the ORDER BY scope —
    /// lets a sort key reference an input variable not in the output.
    pub order_overlay: Vec<usize>,
    /// True if any ORDER BY key references an output column (slot < `out_len`).
    /// When false, sort keys come from the input alone, so `ORDER BY … LIMIT n`
    /// can keep only the top-k *input* bindings and project just those.
    pub order_needs_output: bool,
    /// ISO `HAVING` — a post-aggregation predicate on each group, lowered against
    /// the input scope (group keys) with its aggregates lifted into `aggs`. A
    /// group survives only when this is TRUE. `None` on `RETURN`/`WITH`.
    pub having: Option<CExpr>,
    pub skip: Option<CCount>,
    pub limit: Option<CCount>,
}

impl CProjection {
    /// The grouping keys: explicit `GROUP BY` items if present, else the
    /// non-aggregate output items (implicit grouping).
    ///
    /// OPEN — `GROUP BY` does not resolve an OUTPUT ALIAS, and silently groups
    /// everything into one. `ORDER BY` does resolve one (5f028be), so the two
    /// clauses disagree about what a bare name in them means:
    ///
    /// ```text
    ///   RETURN n.t AS t, count(*) AS c GROUP BY n.t   →  [x, 3], [y, 2]   correct
    ///   RETURN n.t AS t, count(*) AS c GROUP BY t     →  [x, 5]           one group
    ///   RETURN n.t AS u, count(*) AS c GROUP BY u     →  [x, 5]           same
    /// ```
    ///
    /// The surviving row takes the FIRST row's value, which is what grouping by
    /// a constant looks like. Not a cross-engine divergence — the TS engine
    /// returns the same three answers, byte for byte — so the fuzzers cannot see
    /// it, and it is not new.
    ///
    /// Deciding it needs the ISO grammar, not a patch: if `GROUP BY <alias>` is
    /// not legal GQL then the fix is to REJECT it at plan time (a wrong answer
    /// becoming an error), and if it is legal the fix is to resolve it the way
    /// `ORDER BY` already does. Either way it lands in both engines together.
    /// Found while checking NaN ordering (7855bfb).
    pub fn group_keys(&self) -> Vec<&CReturnItem> {
        if self.group_by.is_empty() {
            self.items.iter().filter(|i| !i.is_agg).collect()
        } else {
            self.group_by.iter().collect()
        }
    }
}

/// A lowered `LIMIT` / `OFFSET` bound: an integer literal, or a `$param` slot
/// resolved — and validated to be a non-negative integer — at execute time.
#[derive(Debug, Clone)]
pub enum CCount {
    Lit(usize),
    Param(usize),
}

#[derive(Debug, Clone)]
pub enum CSetItem {
    Prop {
        var_slot: usize,
        key: String,
        value: CExpr,
    },
    Label {
        var_slot: usize,
        label: String,
    },
}

#[derive(Debug, Clone)]
pub enum CRemoveItem {
    Prop { var_slot: usize, key: String },
    Label { var_slot: usize, label: String },
}

#[derive(Debug, Clone)]
pub enum CClause {
    /// `scope_len` is the binding slot count after this match (incl. its vars).
    Match {
        optional: bool,
        patterns: Vec<CPath>,
        where_: Option<CExpr>,
        /// Compiled form of `where_` for the VM (hot per-row site).
        where_prog: Option<Program>,
        scope_len: usize,
    },
    With {
        projection: CProjection,
        where_: Option<CExpr>,
        where_prog: Option<Program>,
    },
    Return(CProjection),
    /// `FILTER [WHERE] <cond>` — drop rows where `pred` is not TRUE.
    Filter {
        pred: CExpr,
        prog: Program,
    },
    /// `ORDER BY … [OFFSET n] [LIMIT n]` as a STATEMENT — sort and/or slice the
    /// working binding table in place. Distinct from a projection's trailing
    /// paging: there is no projection here, so the sort keys read the binding
    /// scope directly and a later RETURN only ever projects the surviving rows.
    Page {
        order_by: Vec<CSortItem>,
        skip: Option<CCount>,
        limit: Option<CCount>,
    },
    /// `LET x = e, …` — bind new value variables (additive). Each `(slot, expr,
    /// prog)` is evaluated and stored, left-to-right (a later item sees earlier).
    Let(Vec<(usize, CExpr, Program)>),
    /// `FOR alias IN list [WITH ORDINALITY|OFFSET]` — unwind `list` into one row
    /// per element. `ord` is `(is_ordinality, slot)`: a counter bound alongside
    /// each element, 1-based when `is_ordinality`, else 0-based. `scope_len` is
    /// the binding width after the alias (+ counter) are bound.
    For {
        list: CExpr,
        alias_slot: usize,
        ord: Option<(bool, usize)>,
        scope_len: usize,
    },
    Insert(Vec<CPath>),
    Merge(CMerge),
    Set(Vec<CSetItem>),
    Remove(Vec<CRemoveItem>),
    Delete {
        detach: bool,
        targets: Vec<CExpr>,
    },
    Finish,
    /// `[OPTIONAL] CALL name(config) [YIELD …]` — a named procedure call. `algo`
    /// is the resolved algorithm dispatch name (`None` = unknown procedure, faults
    /// at run time). `config` = each config field's compiled value expr. `binds`
    /// maps a procedure output column to the binding slot it yields into.
    CallNamed {
        optional: bool,
        proc_name: String,
        algo: Option<&'static str>,
        config: Vec<(String, CExpr)>,
        binds: Vec<CallBind>,
        scope_len: usize,
    },
    /// `[OPTIONAL] CALL (scope) { … }`. `imports` maps each scoped variable's
    /// outer slot to the nested slot the subquery reads it from; `out_binds` maps
    /// each nested `RETURN` output column (in order) to the outer slot it merges
    /// into. The nested query shares the plan's key/label/param tables.
    CallInline {
        optional: bool,
        imports: Vec<(usize, usize)>,
        body: CLinear,
        /// Additional set-op parts (`… UNION … EXCEPT …`) after the first. Empty
        /// for a plain single-part body. Each part shares the same imported scope
        /// and produces the same output columns; results are folded with `combine`.
        body_more: Vec<(SetOp, CLinear)>,
        out_binds: Vec<usize>,
        /// True when the body's RETURN is `*` — its output columns are the scope
        /// vars (imports included). On an OPTIONAL empty run we then keep the outer
        /// row untouched (leaving fresh vars unbound) rather than null-filling
        /// `out_binds`, which would clobber an imported var. (A named RETURN's
        /// columns are genuinely produced by the subquery, so those DO null-fill.)
        body_star: bool,
        /// True if the nested body only reads — then every correlated run reuses
        /// the caller's resolved Ctx (no per-outer-row resolve). A writing body
        /// resolves per row (a mutation may invalidate the shared tables).
        body_read_only: bool,
    },
}

/// One `YIELD` binding: procedure output column → binding slot.
#[derive(Debug, Clone)]
pub struct CallBind {
    pub column: String,
    pub slot: usize,
}

/// The built-in procedure catalog: procedure name → (algorithm dispatch name,
/// its non-`node` result column). Output columns are always `[node, <result>]`.
pub fn procedure_spec(name: &str) -> Option<(&'static str, &'static str)> {
    Some(match name {
        "pagerank" => ("pagerank", "score"),
        "personalized_pagerank" => ("personalizedPagerank", "score"),
        "connected_components" => ("connectedComponents", "componentId"),
        "strongly_connected_components" => ("stronglyConnectedComponents", "componentId"),
        "on_cycle" => ("onCycle", "onCycle"),
        "label_propagation" => ("labelPropagation", "label"),
        "peer_pressure" => ("peerPressure", "cluster"),
        "degree" => ("degree", "degree"),
        "betweenness" => ("betweenness", "centrality"),
        "closeness" => ("closeness", "centrality"),
        "shortest_path" => ("shortestPath", "distance"),
        "neighbor_aggregate" => ("neighborAggregate", "vector"),
        _ => return None,
    })
}

/// The canonical (snake_case) names of the built-in procedures — the surface a
/// `CALL` accepts. Kept in sync with [`procedure_spec`].
pub const PROCEDURE_NAMES: &[&str] = &[
    "pagerank",
    "personalized_pagerank",
    "connected_components",
    "strongly_connected_components",
    "on_cycle",
    "label_propagation",
    "peer_pressure",
    "degree",
    "betweenness",
    "closeness",
    "shortest_path",
    "neighbor_aggregate",
];

/// For an unknown procedure name, the canonical snake_case name it most likely
/// meant — matched by ignoring case and `_` separators, so a camelCase spelling
/// (`pageRank`, `connectedComponents`) resolves to its surface name. `None` when
/// nothing plausibly matches. Drives the `CALL` "did you mean" hint; the TS
/// engine mirrors this so both faults read identically.
pub fn suggest_procedure(name: &str) -> Option<&'static str> {
    let norm = |s: &str| -> String {
        s.chars()
            .filter(|c| *c != '_')
            .flat_map(char::to_lowercase)
            .collect()
    };
    let target = norm(name);
    PROCEDURE_NAMES.iter().copied().find(|n| norm(n) == target)
}

/// Compiled `_MERGE` (see [`crate::gql::ast::MergeClause`]).
#[derive(Debug, Clone)]
pub struct CMerge {
    pub pattern: CPath,
    pub on_create: Option<Vec<CSetItem>>,
    pub on_update: Option<CMergeUpdate>,
}

#[derive(Debug, Clone)]
pub enum CMergeUpdate {
    Set {
        items: Vec<CSetItem>,
        where_: Option<CExpr>,
    },
    Nothing,
}

#[derive(Debug, Clone)]
pub struct CLinear {
    pub clauses: Vec<CClause>,
}

#[derive(Debug, Clone)]
pub struct CQuery {
    pub parts: Vec<CLinear>,
    pub ops: Vec<SetOp>,
    /// Property-key names, indexed by `key_ref`; resolved to ids per execution.
    pub key_names: Vec<String>,
    /// Label/edge-type names, indexed by label ref; resolved to ids per execution.
    pub label_names: Vec<String>,
    /// Names of unknown/unimplemented functions the query references — surfaced
    /// in the `UnknownFunction` error when one faults.
    pub unknown_fns: Vec<String>,
}

/// Collapse a field access on a stored-property base into a single `PropField`
/// (which reads only the leaf). Returns `Ok(PropField)` when `base_c` is a
/// `Prop`/`PropField` chain rooted at a stored property; otherwise `Err(base_c)`
/// hands the lowered base back so the caller keeps the general `Field`/`Index`.
fn prop_field(base_c: CExpr, name: Arc<str>) -> Result<CExpr, CExpr> {
    match base_c {
        CExpr::Prop { var_slot, key_ref } => Ok(CExpr::PropField {
            var_slot,
            root_key_ref: key_ref,
            descent: vec![name],
        }),
        CExpr::PropField {
            var_slot,
            root_key_ref,
            mut descent,
        } => {
            descent.push(name);
            Ok(CExpr::PropField {
                var_slot,
                root_key_ref,
                descent,
            })
        }
        other => Err(other),
    }
}

/// Does a lowered expression contain an aggregate anywhere?
fn has_aggregate(expr: &CExpr) -> bool {
    match expr {
        CExpr::Aggregate { .. } => true,
        CExpr::Scalar { args, .. } => args.iter().any(has_aggregate),
        CExpr::GraphPred { args, .. } => args.iter().any(has_aggregate),
        CExpr::Neg(e) | CExpr::Not(e) => has_aggregate(e),
        CExpr::IsNull { expr, .. }
        | CExpr::IsTruth { expr, .. }
        | CExpr::IsLabeled { expr, .. }
        | CExpr::IsTyped { expr, .. } => has_aggregate(expr),
        CExpr::Arith { head, tail } => {
            has_aggregate(head) || tail.iter().any(|(_, e)| has_aggregate(e))
        }
        CExpr::Concat(items) | CExpr::And(items) | CExpr::Or(items) | CExpr::Xor(items) => {
            items.iter().any(has_aggregate)
        }
        CExpr::Compare { left, right, .. } => has_aggregate(left) || has_aggregate(right),
        CExpr::LetIn { bindings, body } => {
            bindings.iter().any(|(_, e)| has_aggregate(e)) || has_aggregate(body)
        }
        CExpr::In { expr, list, .. } => has_aggregate(expr) || has_aggregate(list),
        CExpr::List(items) => items.iter().any(has_aggregate),
        CExpr::Record(fields) => fields.iter().any(|(_, e)| has_aggregate(e)),
        CExpr::Index { base, index } => has_aggregate(base) || has_aggregate(index),
        CExpr::Field { base, .. } => has_aggregate(base),
        CExpr::Case {
            subject,
            whens,
            else_,
        } => {
            subject.as_deref().is_some_and(has_aggregate)
                || whens
                    .iter()
                    .any(|(w, t)| has_aggregate(w) || has_aggregate(t))
                || else_.as_deref().is_some_and(has_aggregate)
        }
        _ => false,
    }
}

/// True if any aggregate in the plan has an argument that itself contains an
/// aggregate. ISO forbids nested aggregates (`sum(avg(x))`); lowering leaves an
/// aggregate's argument intact (it never recurses into one), so an inner
/// aggregate survives in `CAgg::arg` and we can reject it before execution.
pub fn has_nested_aggregate(plan: &CQuery) -> bool {
    plan.parts
        .iter()
        .flat_map(|part| &part.clauses)
        .filter_map(|clause| match clause {
            CClause::With { projection, .. } => Some(projection),
            CClause::Return(projection) => Some(projection),
            _ => None,
        })
        .flat_map(|projection| &projection.aggs)
        .any(|agg| agg.arg.as_ref().is_some_and(has_aggregate))
}

/// True if any aggregate is argless and is not `count(*)`. Only `count(*)` is a
/// valid argless aggregate; `sum()`, `avg()`, `count()` with no argument, etc.
/// are meaningless and must be rejected (ISO; matches the TS engine).
pub fn has_argless_aggregate(plan: &CQuery) -> bool {
    plan.parts
        .iter()
        .flat_map(|part| &part.clauses)
        .filter_map(|clause| match clause {
            CClause::With { projection, .. } => Some(projection),
            CClause::Return(projection) => Some(projection),
            _ => None,
        })
        .flat_map(|projection| &projection.aggs)
        .any(|agg| agg.arg.is_none() && !agg.star)
}

/// Lift aggregate sub-expressions out of `expr` into `aggs`, replacing each with
/// an [`CExpr::AggRef`]. An aggregate's own argument is left intact (a nested
/// aggregate is invalid), so this never recurses into an `Aggregate`.
fn extract_aggs(expr: CExpr, aggs: &mut Vec<CAgg>) -> CExpr {
    let b = |e: Box<CExpr>, aggs: &mut Vec<CAgg>| Box::new(extract_aggs(*e, aggs));
    match expr {
        CExpr::Aggregate {
            func,
            arg,
            distinct,
            star,
            frac,
        } => {
            let idx = aggs.len();
            aggs.push(CAgg {
                func,
                arg: arg.map(|a| *a),
                distinct,
                star,
                frac,
            });
            CExpr::AggRef(idx)
        }
        CExpr::List(items) => {
            CExpr::List(items.into_iter().map(|e| extract_aggs(e, aggs)).collect())
        }
        CExpr::Record(fields) => CExpr::Record(
            fields
                .into_iter()
                .map(|(k, e)| (k, extract_aggs(e, aggs)))
                .collect(),
        ),
        CExpr::Index { base, index } => CExpr::Index {
            base: b(base, aggs),
            index: b(index, aggs),
        },
        CExpr::Field {
            base,
            key_ref,
            name,
        } => CExpr::Field {
            base: b(base, aggs),
            key_ref,
            name,
        },
        CExpr::Compare { op, left, right } => CExpr::Compare {
            op,
            left: b(left, aggs),
            right: b(right, aggs),
        },
        CExpr::Arith { head, tail } => CExpr::Arith {
            head: b(head, aggs),
            tail: tail
                .into_iter()
                .map(|(op, e)| (op, extract_aggs(e, aggs)))
                .collect(),
        },
        CExpr::Concat(items) => {
            CExpr::Concat(items.into_iter().map(|e| extract_aggs(e, aggs)).collect())
        }
        CExpr::Neg(e) => CExpr::Neg(b(e, aggs)),
        CExpr::And(items) => CExpr::And(items.into_iter().map(|e| extract_aggs(e, aggs)).collect()),
        CExpr::Or(items) => CExpr::Or(items.into_iter().map(|e| extract_aggs(e, aggs)).collect()),
        CExpr::Xor(items) => CExpr::Xor(items.into_iter().map(|e| extract_aggs(e, aggs)).collect()),
        CExpr::Not(e) => CExpr::Not(b(e, aggs)),
        CExpr::IsNull { expr, negated } => CExpr::IsNull {
            expr: b(expr, aggs),
            negated,
        },
        CExpr::IsTruth {
            expr,
            truth,
            negated,
        } => CExpr::IsTruth {
            expr: b(expr, aggs),
            truth,
            negated,
        },
        CExpr::IsLabeled {
            expr,
            label,
            negated,
        } => CExpr::IsLabeled {
            expr: b(expr, aggs),
            label,
            negated,
        },
        CExpr::IsTyped {
            expr,
            ty,
            not_null,
            negated,
        } => CExpr::IsTyped {
            expr: b(expr, aggs),
            ty,
            not_null,
            negated,
        },
        CExpr::In {
            expr,
            list,
            negated,
        } => CExpr::In {
            expr: b(expr, aggs),
            list: b(list, aggs),
            negated,
        },
        CExpr::Case {
            subject,
            whens,
            else_,
        } => CExpr::Case {
            subject: subject.map(|s| b(s, aggs)),
            whens: whens
                .into_iter()
                .map(|(w, t)| (extract_aggs(w, aggs), extract_aggs(t, aggs)))
                .collect(),
            else_: else_.map(|e| b(e, aggs)),
        },
        CExpr::Scalar { func, args } => CExpr::Scalar {
            func,
            args: args.into_iter().map(|e| extract_aggs(e, aggs)).collect(),
        },
        // leaves and the (correlated) sub-queries carry no grouping aggregate
        other => other,
    }
}

/// Does a lowered expression reference any variable/property slot matching `f`?
///
/// The one traversal over `CExpr`'s slot references. Two questions are asked of
/// it: "does an ORDER BY key read an output column" (slot < out_len) and "does a
/// WHERE say anything about a slot other than the one being seeded" — the second
/// because a fast path that seeds and filters ONE element must decline a
/// predicate about any other, or it silently drops the predicate.
pub(crate) fn refs_slot(expr: &CExpr, f: &dyn Fn(usize) -> bool) -> bool {
    match expr {
        CExpr::Var(s) => f(*s),
        CExpr::Prop { var_slot, .. } | CExpr::PropField { var_slot, .. } => f(*var_slot),
        CExpr::List(items) => items.iter().any(|e| refs_slot(e, f)),
        CExpr::Record(fields) => fields.iter().any(|(_, e)| refs_slot(e, f)),
        CExpr::Index { base, index } => refs_slot(base, f) || refs_slot(index, f),
        CExpr::Field { base, .. } => refs_slot(base, f),
        CExpr::Neg(e) | CExpr::Not(e) => refs_slot(e, f),
        CExpr::IsNull { expr, .. }
        | CExpr::IsTruth { expr, .. }
        | CExpr::IsLabeled { expr, .. }
        | CExpr::IsTyped { expr, .. } => refs_slot(expr, f),
        CExpr::Arith { head, tail } => {
            refs_slot(head, f) || tail.iter().any(|(_, e)| refs_slot(e, f))
        }
        CExpr::Concat(items) | CExpr::And(items) | CExpr::Or(items) | CExpr::Xor(items) => {
            items.iter().any(|e| refs_slot(e, f))
        }
        CExpr::Compare { left, right, .. } => refs_slot(left, f) || refs_slot(right, f),
        CExpr::In { expr, list, .. } => refs_slot(expr, f) || refs_slot(list, f),
        CExpr::Case {
            subject,
            whens,
            else_,
        } => {
            subject.as_deref().is_some_and(|e| refs_slot(e, f))
                || whens
                    .iter()
                    .any(|(w, t)| refs_slot(w, f) || refs_slot(t, f))
                || else_.as_deref().is_some_and(|e| refs_slot(e, f))
        }
        CExpr::Scalar { args, .. } => args.iter().any(|e| refs_slot(e, f)),
        CExpr::GraphPred { args, .. } => args.iter().any(|e| refs_slot(e, f)),
        CExpr::Aggregate { arg, .. } => arg.as_deref().is_some_and(|e| refs_slot(e, f)),
        // `PROPERTY_EXISTS(v, k)` resolves the element in slot `var_slot`, exactly
        // like `Prop` — so an ORDER BY over `PROPERTY_EXISTS(<output col>, …)` reads
        // an output slot and must NOT take the input-scope vectorized fast path.
        CExpr::PropertyExists { var_slot, .. } => f(*var_slot),
        // `LET x = e IN body`: the binding exprs read the enclosing scope (which may
        // include an output column < n); the body may too. The LET-bound slots are
        // fresh (≥ n), so a body ref to one is harmless — and a false positive only
        // makes the fast path more conservative, never wrong.
        CExpr::LetIn { bindings, body } => {
            bindings.iter().any(|(_, e)| refs_slot(e, f)) || refs_slot(body, f)
        }
        // A correlated subquery can read an output column through its patterns /
        // WHERE / RETURN correlation, which `refs_slot_below` can't see (they live in
        // `CPath`, not `CExpr`). Conservatively force the safe non-vectorized path
        // whenever ORDER BY contains one — correctness over a rarely-hit fast path.
        CExpr::Exists { .. } | CExpr::CountSubquery { .. } | CExpr::ValueSubquery { .. } => true,
        // Lits / params / aggref reference no input slot.
        _ => false,
    }
}

/// Does `expr` reference any variable/property slot below `n`?
fn refs_slot_below(expr: &CExpr, n: usize) -> bool {
    refs_slot(expr, &|s| s < n)
}

/// The default output column name (from the source AST, which still has names).
fn column_name(expr: &Expr) -> String {
    match expr {
        Expr::Var(name) => name.clone(),
        Expr::Prop { variable, key } => format!("{variable}.{key}"),
        _ => "expr".to_string(),
    }
}

/// Lowers a `Query`, allocating `$param` slots and per-scope variable slots.
struct Lowerer {
    /// param slot -> name (the order positional args are bound in at execute).
    params: Vec<String>,
    /// current scope: variable slot -> name.
    scope: Vec<String>,
    /// property-key ref -> name (resolved to ids per execution).
    keys: Vec<String>,
    /// label/edge-type ref -> name (resolved to ids per execution).
    labels: Vec<String>,
    /// Names of any unknown/unimplemented functions lowered — for the error
    /// message when one of them faults at execute time (they eval to a fault,
    /// not a value).
    unknown_fns: Vec<String>,
}

/// Intern `name` into `table`, returning its ref index.
fn intern_ref(table: &mut Vec<String>, name: &str) -> usize {
    if let Some(i) = table.iter().position(|n| n == name) {
        i
    } else {
        table.push(name.to_string());
        table.len() - 1
    }
}

/// A subpath hop element: a plain `Hop`, or a nested single-edge `Sub` when the hop
/// carries its own quantifier (`-[e]->{a,b}` — a nested repetition). The matcher steps
/// one edge, so a `Hop`'s `CRel` is cleared of the hop quantifier; the `Sub` carries the
/// bounds, and the hop's target becomes the `Sub`'s landing group variable.
fn hop_or_sub(mut rel: CRel, target_slot: Option<usize>, q: Option<Quantifier>) -> CElem {
    rel.quantifier = None;
    match q {
        None => CElem::Hop(CHop { rel, target_slot }),
        // A nested `-[]->{a,b}(y)` hop: the inner hop's own target is an ANONYMOUS
        // intermediate; the landing `y` is the WHOLE sub-unit's target, bound once per
        // enclosing rep as a group variable (`CSub.target_slot`).
        Some(quant) => CElem::Sub(CSub {
            unit: Box::new(CUnit {
                elems: vec![CElem::Hop(CHop {
                    rel,
                    target_slot: None,
                })],
                start_slot: None,
                where_: None,
            }),
            min: quant.min,
            max: quant.max,
            target_slot,
        }),
    }
}

impl Lowerer {
    fn param_slot(&mut self, name: &str) -> usize {
        if let Some(i) = self.params.iter().position(|n| n == name) {
            i
        } else {
            self.params.push(name.to_string());
            self.params.len() - 1
        }
    }

    /// Lower a label expression, assigning a ref to each label name.
    fn label_expr(&mut self, e: &LabelExpr) -> CLabelExpr {
        match e {
            LabelExpr::Label(name) => CLabelExpr::Label(intern_ref(&mut self.labels, name)),
            LabelExpr::Wildcard => CLabelExpr::Wildcard,
            LabelExpr::Not(b) => CLabelExpr::Not(Box::new(self.label_expr(b))),
            LabelExpr::And(l, r) => {
                CLabelExpr::And(Box::new(self.label_expr(l)), Box::new(self.label_expr(r)))
            }
            LabelExpr::Or(l, r) => {
                CLabelExpr::Or(Box::new(self.label_expr(l)), Box::new(self.label_expr(r)))
            }
        }
    }

    /// Slot of an in-scope variable, or `UNBOUND` (reads as NULL).
    fn slot_of(&self, name: &str) -> usize {
        self.scope.iter().position(|n| n == name).unwrap_or(UNBOUND)
    }

    /// Add a variable to the current scope (reusing an existing slot if present).
    fn add_var(&mut self, name: &str) -> usize {
        if let Some(i) = self.scope.iter().position(|n| n == name) {
            i
        } else {
            self.scope.push(name.to_string());
            self.scope.len() - 1
        }
    }

    /// Bring every variable a set of patterns introduces into scope (in order),
    /// before lowering the patterns' predicates (which may reference any of them).
    fn add_pattern_vars(&mut self, patterns: &[PathPattern]) {
        for p in patterns {
            if let Some(v) = &p.path_var {
                self.add_var(v);
            }
            if let Some(v) = &p.start.variable {
                self.add_var(v);
            }
            for seg in &p.segments {
                self.add_segment_vars(seg);
            }
        }
    }

    /// Register a segment's variables. Recurses into a nested parenthesized subpath so
    /// its inner variables get slots too (else `slot_of` returns the `UNBOUND` sentinel
    /// and the binder writes to a bogus slot).
    fn add_segment_vars(&mut self, seg: &Segment) {
        // The inner source `(x)` and target `(y)` group variables of a quantified
        // parenthesized subpath, plus any intermediate nodes / edges of a MULTI-element
        // repetition unit (`(x)-[e1]->(m)-[e2]->(y)`).
        for inner in [&seg.hop_from, &seg.hop_to].into_iter().flatten() {
            if let Some(v) = &inner.variable {
                self.add_var(v);
            }
        }
        for extra in &seg.unit_rest {
            if let Some(v) = &extra.rel.variable {
                self.add_var(v);
            }
            if let Some(v) = &extra.node.variable {
                self.add_var(v);
            }
        }
        if let Some(inner) = &seg.nested {
            self.add_segment_vars(inner);
        }
        if let Some(v) = &seg.rel.variable {
            self.add_var(v);
        }
        if let Some(v) = &seg.node.variable {
            self.add_var(v);
        }
    }

    fn expr(&mut self, e: &Expr) -> CExpr {
        match e {
            Expr::Var(n) => CExpr::Var(self.slot_of(n)),
            Expr::Param(n) => CExpr::Param(self.param_slot(n)),
            Expr::Prop { variable, key } => CExpr::Prop {
                var_slot: self.slot_of(variable),
                key_ref: intern_ref(&mut self.keys, key),
            },
            Expr::PropertyExists { variable, key } => CExpr::PropertyExists {
                var_slot: self.slot_of(variable),
                key_ref: intern_ref(&mut self.keys, key),
            },
            Expr::Lit(l) => CExpr::Lit(l.clone()),
            Expr::List(items) => CExpr::List(items.iter().map(|x| self.expr(x)).collect()),
            Expr::Record(fields) => CExpr::Record(
                fields
                    .iter()
                    .map(|(k, e)| (Arc::from(k.as_str()), self.expr(e)))
                    .collect(),
            ),
            Expr::Index { base, index } => {
                // `base['field']` on a stored-record base is field access — collapse
                // it into a `PropField` (reads only the leaf), same as `.field`.
                if let Expr::Lit(Lit::Str(s)) = index.as_ref() {
                    match prop_field(self.expr(base), Arc::from(s.as_str())) {
                        Ok(pf) => return pf,
                        Err(other) => {
                            return CExpr::Index {
                                base: Box::new(other),
                                index: self.boxed(index),
                            }
                        }
                    }
                }
                CExpr::Index {
                    base: self.boxed(base),
                    index: self.boxed(index),
                }
            }
            Expr::Field { base, key } => {
                let name = Arc::from(key.as_str());
                match prop_field(self.expr(base), Arc::clone(&name)) {
                    Ok(pf) => pf,
                    // A computed (non-stored-prop) base keeps the general Field.
                    Err(other) => CExpr::Field {
                        base: Box::new(other),
                        key_ref: intern_ref(&mut self.keys, key),
                        name,
                    },
                }
            }
            Expr::Compare { op, left, right } => CExpr::Compare {
                op: *op,
                left: self.boxed(left),
                right: self.boxed(right),
            },
            Expr::Arith { head, tail } => CExpr::Arith {
                head: self.boxed(head),
                tail: tail.iter().map(|(op, e)| (*op, self.expr(e))).collect(),
            },
            Expr::Concat(items) => CExpr::Concat(items.iter().map(|x| self.expr(x)).collect()),
            Expr::Neg(x) => CExpr::Neg(self.boxed(x)),
            Expr::And(items) => CExpr::And(items.iter().map(|x| self.expr(x)).collect()),
            Expr::Or(items) => CExpr::Or(items.iter().map(|x| self.expr(x)).collect()),
            Expr::Xor(items) => CExpr::Xor(items.iter().map(|x| self.expr(x)).collect()),
            Expr::Not(x) => CExpr::Not(self.boxed(x)),
            Expr::IsNull { expr, negated } => CExpr::IsNull {
                expr: self.boxed(expr),
                negated: *negated,
            },
            Expr::IsTruth {
                expr,
                truth,
                negated,
            } => CExpr::IsTruth {
                expr: self.boxed(expr),
                truth: *truth,
                negated: *negated,
            },
            Expr::IsLabeled {
                expr,
                label,
                negated,
            } => CExpr::IsLabeled {
                expr: self.boxed(expr),
                label: self.label_expr(label),
                negated: *negated,
            },
            Expr::IsTyped {
                expr,
                ty,
                not_null,
                negated,
            } => CExpr::IsTyped {
                expr: self.boxed(expr),
                ty: ty.clone(),
                not_null: *not_null,
                negated: *negated,
            },
            Expr::GraphPred {
                kind,
                args,
                negated,
            } => CExpr::GraphPred {
                kind: *kind,
                args: args.iter().map(|a| self.expr(a)).collect(),
                negated: *negated,
            },
            Expr::In {
                expr,
                list,
                negated,
            } => CExpr::In {
                expr: self.boxed(expr),
                list: self.boxed(list),
                negated: *negated,
            },
            Expr::Exists { patterns, where_ } => {
                let (patterns, where_, sub_len) = self.sub_patterns(patterns, where_.as_deref());
                CExpr::Exists {
                    patterns,
                    where_,
                    sub_len,
                }
            }
            Expr::CountSubquery { patterns, where_ } => {
                let (patterns, where_, sub_len) = self.sub_patterns(patterns, where_.as_deref());
                CExpr::CountSubquery {
                    patterns,
                    where_,
                    sub_len,
                }
            }
            Expr::ValueSubquery {
                patterns,
                where_,
                ret,
            } => {
                // The RETURN expression reads the subquery's own bindings, so lower
                // it inside the sub-scope alongside the patterns/WHERE, then restore.
                let parent_len = self.scope.len();
                self.add_pattern_vars(patterns);
                let cpatterns = patterns.iter().map(|p| self.path(p)).collect();
                let cwhere = where_.as_deref().map(|w| self.boxed(w));
                let cret = self.boxed(ret);
                let sub_len = self.scope.len();
                self.scope.truncate(parent_len);
                let is_agg = has_aggregate(&cret);
                CExpr::ValueSubquery {
                    patterns: cpatterns,
                    where_: cwhere,
                    ret: cret,
                    is_agg,
                    sub_len,
                }
            }
            Expr::LetIn { bindings, body } => {
                // Each binding compiles against the scope so far (prior LET vars
                // included), THEN its variable is added so later bindings and the
                // body can see it — mirroring the LET *statement*. The locals are
                // scoped to this expression, so restore the outer scope after.
                let parent_len = self.scope.len();
                let cbindings = bindings
                    .iter()
                    .map(|it| {
                        let cexpr = self.boxed(&it.expr);
                        let slot = self.add_var(&it.var);
                        (slot, cexpr)
                    })
                    .collect();
                let cbody = self.boxed(body);
                self.scope.truncate(parent_len);
                CExpr::LetIn {
                    bindings: cbindings,
                    body: cbody,
                }
            }
            Expr::Case {
                subject,
                whens,
                else_,
            } => CExpr::Case {
                subject: subject.as_ref().map(|s| self.boxed(s)),
                whens: whens
                    .iter()
                    .map(|(w, t)| (self.expr(w), self.expr(t)))
                    .collect(),
                else_: else_.as_ref().map(|e| self.boxed(e)),
            },
            Expr::Func {
                name,
                args,
                distinct,
                star,
            } => {
                let cargs: Vec<CExpr> = args.iter().map(|a| self.expr(a)).collect();
                if let Some(func) = agg_fn(name) {
                    // Percentile aggregates are `(value, literal fraction)`. A
                    // malformed call (wrong arity / non-literal fraction) falls
                    // through to an unknown-function error rather than silently
                    // mis-evaluating.
                    let is_pct = matches!(func, AggFn::PercentileCont | AggFn::PercentileDisc);
                    let frac = if is_pct {
                        match (cargs.len(), cargs.get(1)) {
                            (2, Some(CExpr::Lit(Lit::Num(f)))) => Some(f.clamp(0.0, 1.0)),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    if is_pct && frac.is_none() {
                        if !self.unknown_fns.iter().any(|n| n == name) {
                            self.unknown_fns.push(name.to_string());
                        }
                        CExpr::Scalar {
                            func: ScalarFn::Unknown,
                            args: cargs,
                        }
                    } else {
                        CExpr::Aggregate {
                            func,
                            arg: cargs.into_iter().next().map(Box::new),
                            distinct: *distinct,
                            star: *star,
                            frac,
                        }
                    }
                } else {
                    let func = scalar_fn(name);

                    if matches!(func, ScalarFn::Unknown)
                        && !self.unknown_fns.iter().any(|n| n == name)
                    {
                        self.unknown_fns.push(name.to_string());
                    }

                    CExpr::Scalar { func, args: cargs }
                }
            }
        }
    }

    fn boxed(&mut self, e: &Expr) -> Box<CExpr> {
        Box::new(self.expr(e))
    }

    /// Lower a correlated sub-query's patterns: extend the scope with the sub's
    /// new variables (outer vars keep their slots — that's the correlation),
    /// lower, then truncate the scope back so the sub's vars don't leak out.
    fn sub_patterns(
        &mut self,
        patterns: &[PathPattern],
        where_: Option<&Expr>,
    ) -> (Vec<CPath>, Option<Box<CExpr>>, usize) {
        let parent_len = self.scope.len();
        self.add_pattern_vars(patterns);
        let cpatterns = patterns.iter().map(|p| self.path(p)).collect();
        let cwhere = where_.map(|w| self.boxed(w));
        let sub_len = self.scope.len();
        self.scope.truncate(parent_len);
        (cpatterns, cwhere, sub_len)
    }

    fn prop(&mut self, p: &PropertyConstraint) -> CPropConstraint {
        CPropConstraint {
            key: p.key.clone(),
            key_ref: intern_ref(&mut self.keys, &p.key),
            value: self.expr(&p.value),
        }
    }

    fn node(&mut self, n: &NodePattern) -> CNode {
        CNode {
            var_slot: n.variable.as_ref().map(|v| self.slot_of(v)),
            label: n.label.as_ref().map(|l| self.label_expr(l)),
            props: n.props.iter().map(|p| self.prop(p)).collect(),
            where_: n.where_.as_ref().map(|w| self.expr(w)),
        }
    }

    fn rel(&mut self, r: &RelPattern) -> CRel {
        CRel {
            var_slot: r.variable.as_ref().map(|v| self.slot_of(v)),
            label: r.label.as_ref().map(|l| self.label_expr(l)),
            direction: r.direction,
            props: r.props.iter().map(|p| self.prop(p)).collect(),
            where_: r.where_.as_ref().map(|w| self.expr(w)),
            quantifier: r.quantifier,
        }
    }

    /// A node's binding slot (`None` if anonymous).
    fn node_slot(&mut self, n: &NodePattern) -> Option<usize> {
        n.variable.as_ref().map(|v| self.slot_of(v))
    }

    fn segment(&mut self, s: &Segment) -> CSegment {
        let node = self.node(&s.node);
        let rel = self.rel(&s.rel);
        // A quantified parenthesized subpath compiles to a repetition UNIT (its inner hops
        // with group-variable slots + the per-unit `WHERE`); `node` is the separate outer
        // endpoint, and `rel.quantifier` stays the OUTER subpath quantifier for the matcher.
        // A plain hop has no unit.
        let unit = if s.hop_from.is_some() || s.nested.is_some() {
            Some(self.subpath_unit(s))
        } else {
            None
        };
        CSegment { rel, node, unit }
    }

    /// Build one repetition unit from a quantified-subpath segment. A NESTED parenthesized
    /// subpath (`( ((x)-[e]->(y)){a,b} ){n,m}`) recurses: the outer unit's sole element is
    /// a `Sub` wrapping the inner subpath's unit, so its variables nest one list level
    /// deeper. Otherwise it's the inner hop chain (`(x)-[e1]->(m)-[e2]->(y)`).
    fn subpath_unit(&mut self, s: &Segment) -> CUnit {
        if let Some(inner) = &s.nested {
            let inner_unit = self.subpath_unit(inner);
            let q = inner
                .rel
                .quantifier
                .expect("a nested subpath is quantified");
            // The outer subpath-level `WHERE` (per OUTER rep, inner vars bound as lists).
            let where_ = s.subpath_where.as_ref().map(|w| self.expr(w));
            return CUnit {
                elems: vec![CElem::Sub(CSub {
                    unit: Box::new(inner_unit),
                    min: q.min,
                    max: q.max,
                    // The nested subpath's LANDING is the outer segment's endpoint node,
                    // matched separately — not a group variable of this unit.
                    target_slot: None,
                })],
                start_slot: None,
                where_,
            };
        }
        let from = s.hop_from.as_ref().expect("a subpath has a source");
        let mut rel = self.rel(&s.rel);
        // The subpath-level `WHERE` is the per-repetition predicate. A PLAIN first hop's
        // inline edge `WHERE` is also lifted to the unit level and AND-ed (checked once the
        // rep is bound); a NESTED hop's inline `WHERE` (`-[e WHERE …]->{a,b}`) is a per-inner-
        // edge predicate that stays on the `Sub`'s inner hop, NOT lifted.
        let mut where_ = s.subpath_where.as_ref().map(|w| self.expr(w));
        if s.inner_q.is_none() {
            if let Some(w) = rel.where_.take() {
                where_ = Some(match where_.take() {
                    Some(prev) => CExpr::And(vec![prev, w]),
                    None => w,
                });
            }
        }
        let first_target = s.hop_to.as_ref().and_then(|to| self.node_slot(to));
        let mut elems = vec![hop_or_sub(rel.clone(), first_target, s.inner_q)];
        for extra in &s.unit_rest {
            let mut extra_rel = self.rel(&extra.rel);
            let q = extra.rel.quantifier;
            if q.is_none() {
                if let Some(w) = extra_rel.where_.take() {
                    where_ = Some(match where_.take() {
                        Some(prev) => CExpr::And(vec![prev, w]),
                        None => w,
                    });
                }
            }
            let target = self.node_slot(&extra.node);
            elems.push(hop_or_sub(extra_rel, target, q));
        }
        CUnit {
            elems,
            start_slot: self.node_slot(from),
            where_,
        }
    }

    fn path(&mut self, p: &PathPattern) -> CPath {
        CPath {
            start: self.node(&p.start),
            segments: p.segments.iter().map(|s| self.segment(s)).collect(),
            path_var_slot: p.path_var.as_ref().map(|v| self.slot_of(v)),
            selector: p.selector,
            mode: p.mode,
        }
    }

    /// Lower a projection body. Sets the scope for what follows: a non-terminal
    /// (`WITH`) projection's output columns become the next scope; a terminal
    /// (`RETURN`) leaves the scope as-is (nothing follows).
    fn projection(&mut self, p: &Projection, terminal: bool) -> CProjection {
        let input_scope = self.scope.clone();
        let mut aggs: Vec<CAgg> = Vec::new();
        let items: Vec<CReturnItem> = p
            .items
            .iter()
            .map(|it| {
                let expr = self.expr(&it.expr);
                let is_agg = has_aggregate(&expr);
                let name = it.alias.clone().unwrap_or_else(|| column_name(&it.expr));
                // Lift aggregates out of aggregating items so groups fold incrementally.
                let expr = extract_aggs(expr, &mut aggs);
                let prog = compile_program(&expr);
                CReturnItem {
                    expr,
                    prog,
                    name,
                    is_agg,
                }
            })
            .collect();

        let (out_names, star_cols): (Vec<String>, Vec<usize>) = if p.star {
            (input_scope.clone(), (0..input_scope.len()).collect())
        } else {
            (items.iter().map(|i| i.name.clone()).collect(), Vec::new())
        };
        let out_len = out_names.len();
        // `GROUP BY` keys are compiled like items (input-scope expressions used
        // purely for keying, never output). They force grouping on.
        let group_by: Vec<CReturnItem> = p
            .group_by
            .iter()
            .map(|e| {
                let expr = self.expr(e);
                let prog = compile_program(&expr);
                CReturnItem {
                    expr,
                    prog,
                    name: String::new(),
                    is_agg: false,
                }
            })
            .collect();
        // HAVING is lowered here, while the scope is still the input scope, so it
        // reads group keys (input vars); its aggregates lift into `aggs` alongside
        // the items' so they fold per group. Evaluated post-aggregation in
        // `Aggregator::finish`.
        let having = p
            .having
            .as_ref()
            .map(|h| extract_aggs(self.expr(h), &mut aggs));
        let aggregating =
            !p.star && (items.iter().any(|i| i.is_agg) || !group_by.is_empty() || having.is_some());

        // ORDER BY scope = output columns, then input vars not shadowed by one.
        let mut sort_scope = out_names.clone();
        let mut order_overlay = Vec::new();
        for (i, name) in input_scope.iter().enumerate() {
            if !out_names.contains(name) {
                sort_scope.push(name.clone());
                order_overlay.push(i);
            }
        }
        self.scope = sort_scope;
        let order_by: Vec<CSortItem> = p
            .order_by
            .iter()
            .map(|s| CSortItem {
                expr: extract_aggs(self.expr(&s.expr), &mut aggs),
                descending: s.descending,
                nulls_first: s.nulls_first,
            })
            .collect();

        self.scope = if terminal {
            input_scope
        } else {
            out_names.clone()
        };
        let order_needs_output = order_by.iter().any(|s| refs_slot_below(&s.expr, out_len));
        CProjection {
            star: p.star,
            distinct: p.distinct,
            items,
            aggregating,
            group_by,
            aggs,
            out_len,
            out_names,
            star_cols,
            order_by,
            order_overlay,
            order_needs_output,
            having,
            skip: self.count_bound(&p.skip),
            limit: self.count_bound(&p.limit),
        }
    }

    /// Lower a `LIMIT` / `OFFSET` bound, allocating a positional slot for a
    /// `$param` bound (so `positional` bound-checks it like any other param).
    fn count_bound(&mut self, b: &Option<CountBound>) -> Option<CCount> {
        b.as_ref().map(|c| match c {
            CountBound::Lit(n) => CCount::Lit(*n),
            CountBound::Param(name) => CCount::Param(self.param_slot(name)),
        })
    }

    fn compile_set_items(&mut self, items: &[SetItem]) -> Vec<CSetItem> {
        items
            .iter()
            .map(|i| match i {
                SetItem::Prop {
                    variable,
                    key,
                    value,
                } => CSetItem::Prop {
                    var_slot: self.slot_of(variable),
                    key: key.clone(),
                    value: self.expr(value),
                },
                SetItem::Label { variable, label } => CSetItem::Label {
                    var_slot: self.slot_of(variable),
                    label: label.clone(),
                },
            })
            .collect()
    }

    fn clause(&mut self, c: &Clause) -> CClause {
        match c {
            Clause::Match(m) => {
                self.add_pattern_vars(&m.patterns);
                let patterns = m.patterns.iter().map(|p| self.path(p)).collect();
                let where_ = m.where_.as_ref().map(|w| self.expr(w));
                let where_prog = where_.as_ref().map(compile_program);
                CClause::Match {
                    optional: m.optional,
                    patterns,
                    where_,
                    where_prog,
                    scope_len: self.scope.len(),
                }
            }
            Clause::With(w) => {
                let projection = self.projection(&w.projection, false);
                // WITH's WHERE filters the projected output columns (new scope).
                let where_ = w.where_.as_ref().map(|e| self.expr(e));
                let where_prog = where_.as_ref().map(compile_program);
                CClause::With {
                    projection,
                    where_,
                    where_prog,
                }
            }
            Clause::Filter(cond) => {
                // FILTER's condition is a predicate over the CURRENT scope (the
                // working table) — no projection, no new vars.
                let pred = self.expr(cond);
                let prog = compile_program(&pred);
                CClause::Filter { pred, prog }
            }
            Clause::Page(p) => {
                // Sort keys are ordinary expressions over the CURRENT scope — the
                // working table — so they lower exactly like a FILTER predicate.
                // No projection, no new vars, no aggregate lifting (an aggregate
                // here has no group to fold and is rejected by the same rule that
                // rejects one in a FILTER).
                let order_by = p
                    .order_by
                    .iter()
                    .map(|it| CSortItem {
                        expr: self.expr(&it.expr),
                        descending: it.descending,
                        nulls_first: it.nulls_first,
                    })
                    .collect();
                CClause::Page {
                    order_by,
                    skip: self.count_bound(&p.skip),
                    limit: self.count_bound(&p.limit),
                }
            }
            Clause::Let(items) => {
                // Each binding is compiled against the scope so far (prior LET vars
                // included), THEN its variable is added so later items can see it.
                let compiled = items
                    .iter()
                    .map(|it| {
                        let expr = self.expr(&it.expr);
                        let prog = compile_program(&expr);
                        let slot = self.add_var(&it.var);
                        (slot, expr, prog)
                    })
                    .collect();
                CClause::Let(compiled)
            }
            Clause::For(f) => {
                // Lower the list in the pre-FOR scope (it cannot reference the
                // alias), THEN bind the alias (+ any ordinality/offset var) so
                // downstream clauses resolve them.
                let list = self.expr(&f.list);
                let alias_slot = self.add_var(&f.alias);
                let ord = f
                    .ordinal
                    .as_ref()
                    .map(|o| (matches!(o.kind, OrdKind::Ordinality), self.add_var(&o.var)));
                CClause::For {
                    list,
                    alias_slot,
                    ord,
                    scope_len: self.scope.len(),
                }
            }
            Clause::CallNamed(c) => {
                let spec = procedure_spec(&c.name);
                // Config exprs compile in the pre-CALL scope (constants — no vars).
                let config = c
                    .config
                    .iter()
                    .map(|p| (p.key.clone(), self.expr(&p.value)))
                    .collect();
                // Output columns of a known procedure: [node, <result>].
                let columns: Vec<String> = spec
                    .map(|(_, col)| vec!["node".to_string(), col.to_string()])
                    .unwrap_or_default();
                let binds = match &c.yields {
                    // Explicit YIELD: bind each named column to alias|name.
                    Some(items) => items
                        .iter()
                        .map(|it| CallBind {
                            column: it.name.clone(),
                            slot: self.add_var(it.alias.as_ref().unwrap_or(&it.name)),
                        })
                        .collect(),
                    // No YIELD: bind every output column under its own name.
                    None => columns
                        .iter()
                        .map(|col| CallBind {
                            column: col.clone(),
                            slot: self.add_var(col),
                        })
                        .collect(),
                };
                CClause::CallNamed {
                    optional: c.optional,
                    proc_name: c.name.clone(),
                    algo: spec.map(|(dispatch, _)| dispatch),
                    config,
                    binds,
                    scope_len: self.scope.len(),
                }
            }
            Clause::CallInline(c) => {
                // Compile the nested body in its OWN scope (the imported variables
                // occupy nested slots 0..k), sharing the key/label/param tables so
                // a single Ctx resolves both queries.
                let saved = std::mem::replace(&mut self.scope, c.scope.clone());
                let body = CLinear {
                    clauses: c.body.parts[0]
                        .clauses
                        .iter()
                        .map(|cl| self.clause(cl))
                        .collect(),
                };
                // Additional set-op parts (`… UNION …`). Each imports the SAME base
                // scope, so reset to it before compiling — parts don't share bindings.
                let mut body_more = Vec::with_capacity(c.body.ops.len());
                for (i, op) in c.body.ops.iter().enumerate() {
                    self.scope = c.scope.clone();
                    let part = CLinear {
                        clauses: c.body.parts[i + 1]
                            .clauses
                            .iter()
                            .map(|cl| self.clause(cl))
                            .collect(),
                    };
                    body_more.push((*op, part));
                }
                // The nested RETURN's output columns, in order. All set-op parts share
                // the same output columns, so the first part is authoritative.
                let ret_proj = body.clauses.iter().rev().find_map(|cl| match cl {
                    CClause::Return(proj) => Some(proj),
                    _ => None,
                });
                let out_cols = ret_proj.map(|p| p.out_names.clone()).unwrap_or_default();
                let body_star = ret_proj.is_some_and(|p| p.star);
                // Restore the outer scope, then resolve imports + merge slots.
                self.scope = saved;
                let imports = c
                    .scope
                    .iter()
                    .enumerate()
                    .map(|(nested_slot, name)| (self.slot_of(name), nested_slot))
                    .collect();
                let out_binds = out_cols.iter().map(|n| self.add_var(n)).collect();
                let is_read_only = |lin: &CLinear| {
                    lin.clauses.iter().all(|cl| {
                        !matches!(
                            cl,
                            CClause::Insert(_)
                                | CClause::Merge(_)
                                | CClause::Set(_)
                                | CClause::Remove(_)
                                | CClause::Delete { .. }
                        )
                    })
                };
                let body_read_only =
                    is_read_only(&body) && body_more.iter().all(|(_, p)| is_read_only(p));
                CClause::CallInline {
                    optional: c.optional,
                    imports,
                    body,
                    body_more,
                    out_binds,
                    body_star,
                    body_read_only,
                }
            }
            Clause::Return(p) => CClause::Return(self.projection(p, true)),
            Clause::Insert(ps) => {
                self.add_pattern_vars(ps); // INSERT introduces new bindable vars
                CClause::Insert(ps.iter().map(|p| self.path(p)).collect())
            }
            Clause::Merge(m) => {
                // Register the pattern's vars (like INSERT) so _ON_CREATE/_ON_UPDATE
                // SET items resolve to the pattern node's slot, then compile.
                self.add_pattern_vars(std::slice::from_ref(&m.pattern));
                let pattern = self.path(&m.pattern);
                let on_create = m
                    .on_create
                    .as_ref()
                    .map(|items| self.compile_set_items(items));
                let on_update = m.on_update.as_ref().map(|u| match u {
                    MergeUpdate::Nothing => CMergeUpdate::Nothing,
                    MergeUpdate::Set { items, where_ } => CMergeUpdate::Set {
                        items: self.compile_set_items(items),
                        where_: where_.as_ref().map(|e| self.expr(e)),
                    },
                });
                CClause::Merge(CMerge {
                    pattern,
                    on_create,
                    on_update,
                })
            }
            Clause::Set(items) => CClause::Set(self.compile_set_items(items)),
            Clause::Remove(items) => CClause::Remove(
                items
                    .iter()
                    .map(|i| match i {
                        RemoveItem::Prop { variable, key } => CRemoveItem::Prop {
                            var_slot: self.slot_of(variable),
                            key: key.clone(),
                        },
                        RemoveItem::Label { variable, label } => CRemoveItem::Label {
                            var_slot: self.slot_of(variable),
                            label: label.clone(),
                        },
                    })
                    .collect(),
            ),
            Clause::Delete { detach, targets } => CClause::Delete {
                detach: *detach,
                targets: targets.iter().map(|t| self.expr(t)).collect(),
            },
            Clause::Finish => CClause::Finish,
        }
    }

    fn linear(&mut self, l: &LinearQuery) -> CLinear {
        self.scope.clear(); // each linear query starts with a fresh scope
                            // Decorrelate safe correlated inline subqueries into flat MATCH+WITH first.
        let clauses = decorrelate_clauses(&l.clauses);
        CLinear {
            clauses: fold_filters_into_matches(clauses.iter().map(|c| self.clause(c)).collect()),
        }
    }
}

/// Merge adjacent plain `MATCH` clauses, and fold a `FILTER` into the `MATCH`
/// before it.
///
/// `MATCH p1 MATCH p2 WHERE w` and `MATCH p1, p2 WHERE w` are the same query, and
/// they were 35x apart (3.478 ms vs 0.096 ms on 20k vertices, degree 3) purely
/// because the comma spelling arrived as ONE clause with two patterns — which
/// `fuse_chain` splices into a single path — while the two-clause spelling did
/// not, and every multi-clause shape refuses the vectorized frame.
///
/// Sound because it is where the work already happened: the executor DEFERS
/// consecutive `MATCH`es and materializes them together at the next barrier, so
/// merging them changes nothing about which rows are produced or in what order.
/// A clause `WHERE` is checked once every pattern in its clause has bound, which
/// is also where a following `FILTER` ran.
///
/// `OPTIONAL MATCH` is left alone on both counts: it is a left join, so merging
/// it into a plain match would drop the null-extended rows, and a `WHERE` folded
/// onto it would remove them.
fn fold_filters_into_matches(clauses: Vec<CClause>) -> Vec<CClause> {
    let mut out: Vec<CClause> = Vec::with_capacity(clauses.len());

    for c in clauses {
        // Adjacent plain MATCHes become one clause with both pattern lists.
        if let CClause::Match {
            optional: false,
            patterns,
            where_,
            scope_len,
            ..
        } = &c
        {
            if let Some(CClause::Match {
                optional: false,
                patterns: prev_pats,
                where_: prev_where,
                where_prog: prev_prog,
                scope_len: prev_scope,
            }) = out.last_mut()
            {
                prev_pats.extend(patterns.iter().cloned());
                *prev_scope = (*prev_scope).max(*scope_len);
                *prev_where = match (prev_where.take(), where_.clone()) {
                    (Some(a), Some(b)) => Some(CExpr::And(vec![a, b])),
                    (a, b) => a.or(b),
                };
                *prev_prog = prev_where.as_ref().map(compile_program);
                continue;
            }
        }

        let CClause::Filter { pred, prog } = c else {
            out.push(c);
            continue;
        };

        match out.last_mut() {
            Some(CClause::Match {
                optional: false,
                where_,
                where_prog,
                ..
            }) => {
                *where_ = Some(match where_.take() {
                    Some(prev) => CExpr::And(vec![prev, pred]),
                    None => pred,
                });
                *where_prog = where_.as_ref().map(compile_program);
            }
            _ => out.push(CClause::Filter { pred, prog }),
        }
    }

    out
}

/// A compiled VALIDATOR predicate: the lowered boolean expression plus the
/// key/label name tables it references (resolved to graph ids per evaluation).
/// The validated element binds to slot 0 — the sole in-scope variable — so a
/// reference to any *other* name lowers to `UNBOUND` and reads as NULL, exactly
/// like a `WHERE` over a lone pattern variable. Evaluated by
/// [`crate::gql::eval::eval_predicate`].
#[derive(Debug, Clone)]
pub struct CPredicate {
    pub expr: CExpr,
    pub key_names: Vec<String>,
    pub label_names: Vec<String>,
    pub unknown_fns: Vec<String>,
}

/// Lower a bare predicate `Expr` with a single in-scope variable `var` (slot 0)
/// into a [`CPredicate`] — the compiled form a validator evaluates against one
/// element. Mirrors the TS `compileValidator`, which compiles the same `Expr`
/// against a binding `{ [var]: element }`.
pub fn lower_predicate(var: &str, e: &Expr) -> CPredicate {
    let mut l = Lowerer {
        params: Vec::new(),
        scope: vec![var.to_string()],
        keys: Vec::new(),
        labels: Vec::new(),
        unknown_fns: Vec::new(),
    };
    let expr = l.expr(e);
    CPredicate {
        expr,
        key_names: l.keys,
        label_names: l.labels,
        unknown_fns: l.unknown_fns,
    }
}

/// Add every variable a sub-pattern introduces (start node, each hop's rel +
/// node) to `bound`.
fn pattern_bound_vars(p: &PathPattern, bound: &mut Vec<String>) {
    let add = |v: &Option<String>, bound: &mut Vec<String>| {
        if let Some(name) = v {
            if !bound.iter().any(|n| n == name) {
                bound.push(name.clone());
            }
        }
    };
    add(&p.path_var, bound);
    add(&p.start.variable, bound);
    for seg in &p.segments {
        add(&seg.rel.variable, bound);
        add(&seg.node.variable, bound);
    }
}

/// Collect every FREE variable a predicate references — a `Var`/`Prop` name NOT
/// bound by an enclosing `EXISTS`/`COUNT` sub-pattern. A VALIDATOR predicate has
/// exactly one legitimate free variable, the declared `var` (the element under
/// test); a reference to any *other* free name (a typo like `x.age` when the
/// binding is `u`, or a bare `age`) is unbound, so the predicate silently reads
/// UNKNOWN and the SQL-`CHECK` never fires. [`Graph::create_validator`] rejects
/// such a predicate at declare time. Sub-query pattern variables are bound
/// *within* the sub-query, so they are correctly NOT free. Mirrors the TS
/// `freePredicateVars`.
pub fn free_predicate_vars(e: &Expr) -> Vec<String> {
    let mut free = Vec::new();
    collect_free_vars(e, &[], &mut free);
    free
}

fn note_free(name: &str, bound: &[String], free: &mut Vec<String>) {
    if !bound.iter().any(|n| n == name) && !free.iter().any(|n| n == name) {
        free.push(name.to_string());
    }
}

fn collect_free_vars(e: &Expr, bound: &[String], free: &mut Vec<String>) {
    match e {
        Expr::Var(n) => note_free(n, bound, free),
        Expr::Prop { variable, .. } => note_free(variable, bound, free),
        Expr::PropertyExists { variable, .. } => note_free(variable, bound, free),
        Expr::Lit(_) | Expr::Param(_) => {}
        Expr::List(items) => {
            for it in items {
                collect_free_vars(it, bound, free);
            }
        }
        Expr::Record(fields) => {
            for (_, e) in fields {
                collect_free_vars(e, bound, free);
            }
        }
        Expr::Neg(x) | Expr::Not(x) => collect_free_vars(x, bound, free),
        Expr::Index { base, index } => {
            collect_free_vars(base, bound, free);
            collect_free_vars(index, bound, free);
        }
        Expr::Field { base, .. } => collect_free_vars(base, bound, free),
        Expr::IsNull { expr, .. }
        | Expr::IsTruth { expr, .. }
        | Expr::IsLabeled { expr, .. }
        | Expr::IsTyped { expr, .. } => collect_free_vars(expr, bound, free),
        Expr::Compare { left, right, .. } => {
            collect_free_vars(left, bound, free);
            collect_free_vars(right, bound, free);
        }
        Expr::Arith { head, tail } => {
            collect_free_vars(head, bound, free);
            for (_, e) in tail {
                collect_free_vars(e, bound, free);
            }
        }
        Expr::Concat(items) | Expr::And(items) | Expr::Or(items) | Expr::Xor(items) => {
            for e in items {
                collect_free_vars(e, bound, free);
            }
        }
        Expr::In { expr, list, .. } => {
            collect_free_vars(expr, bound, free);
            collect_free_vars(list, bound, free);
        }
        Expr::Case {
            subject,
            whens,
            else_,
        } => {
            if let Some(s) = subject {
                collect_free_vars(s, bound, free);
            }
            for (w, t) in whens {
                collect_free_vars(w, bound, free);
                collect_free_vars(t, bound, free);
            }
            if let Some(el) = else_ {
                collect_free_vars(el, bound, free);
            }
        }
        Expr::Func { args, .. } => {
            for a in args {
                collect_free_vars(a, bound, free);
            }
        }
        Expr::GraphPred { args, .. } => {
            for a in args {
                collect_free_vars(a, bound, free);
            }
        }
        Expr::Exists { patterns, where_ } | Expr::CountSubquery { patterns, where_ } => {
            // The sub-pattern binds its own variables; extend the bound set before
            // descending into its inline predicates and WHERE so those bindings
            // are not mistaken for free references. Outer names still read free.
            let mut inner = bound.to_vec();
            for p in patterns {
                pattern_bound_vars(p, &mut inner);
            }
            for p in patterns {
                collect_pattern_free_vars(p, &inner, free);
            }
            if let Some(w) = where_ {
                collect_free_vars(w, &inner, free);
            }
        }
        Expr::ValueSubquery {
            patterns,
            where_,
            ret,
        } => {
            // Same as EXISTS/COUNT, plus the RETURN expression, which also reads the
            // subquery's own bindings.
            let mut inner = bound.to_vec();
            for p in patterns {
                pattern_bound_vars(p, &mut inner);
            }
            for p in patterns {
                collect_pattern_free_vars(p, &inner, free);
            }
            if let Some(w) = where_ {
                collect_free_vars(w, &inner, free);
            }
            collect_free_vars(ret, &inner, free);
        }
        Expr::LetIn { bindings, body } => {
            // Each binding may reference outer vars and prior LET locals; the LET
            // variable it introduces then shadows for later bindings and the body.
            let mut inner = bound.to_vec();
            for it in bindings {
                collect_free_vars(&it.expr, &inner, free);
                if !inner.iter().any(|n| n == &it.var) {
                    inner.push(it.var.clone());
                }
            }
            collect_free_vars(body, &inner, free);
        }
    }
}

fn collect_pattern_free_vars(p: &PathPattern, bound: &[String], free: &mut Vec<String>) {
    let node = |n: &NodePattern, free: &mut Vec<String>| {
        for c in &n.props {
            collect_free_vars(&c.value, bound, free);
        }
        if let Some(w) = &n.where_ {
            collect_free_vars(w, bound, free);
        }
    };
    node(&p.start, free);
    for seg in &p.segments {
        for c in &seg.rel.props {
            collect_free_vars(&c.value, bound, free);
        }
        if let Some(w) = &seg.rel.where_ {
            collect_free_vars(w, bound, free);
        }
        node(&seg.node, free);
    }
}

/// Lower a parsed query into the IR plus the parameter slot order (slot → name).
/// Decorrelate a linear query's clauses: rewrite a correlated **non-aggregating**
/// inline `CALL (scope) { MATCH … RETURN <items> }` into flat `[OPTIONAL] MATCH …
/// WITH <outer vars>, <items>`. This produces byte-IDENTICAL output (same rows,
/// same order) — the flat MATCH nests exactly as the per-outer-row subquery did —
/// but runs in a single pass through the optimized MATCH path instead of
/// re-executing the subquery per outer row. Only fires when provably safe (see
/// `try_decorrelate`); anything else stays correlated. Aggregating bodies are left
/// alone (a grouping rewrite would reorder rows, breaking the identical-output
/// guarantee this relies on).
fn decorrelate_clauses(clauses: &[Clause]) -> Vec<Clause> {
    let mut out = Vec::with_capacity(clauses.len());
    let mut bound: Vec<String> = Vec::new();
    // The rewrite is only sound while the prefix is plain MATCH clauses (so
    // `bound` == the exact in-scope variable set). A WITH/FOR/write/CALL resets or
    // complicates the scope, after which we stop decorrelating.
    let mut simple_prefix = true;

    for clause in clauses {
        if simple_prefix {
            if let Clause::CallInline(c) = clause {
                if let Some((mc, wc)) = try_decorrelate(c, &bound) {
                    for p in &mc.patterns {
                        pattern_bound_vars(p, &mut bound);
                    }
                    out.push(Clause::Match(mc));
                    out.push(Clause::With(wc));
                    simple_prefix = false; // the injected WITH ends the plain prefix
                    continue;
                }
            }
        }

        match clause {
            Clause::Match(m) => {
                for p in &m.patterns {
                    pattern_bound_vars(p, &mut bound);
                }
            }
            _ => simple_prefix = false,
        }
        out.push(clause.clone());
    }

    out
}

/// Try to flatten one non-aggregating correlated inline CALL into `MATCH` + `WITH`.
/// Returns `None` (→ stay correlated) unless every safety guard holds.
fn try_decorrelate(c: &CallInline, outer: &[String]) -> Option<(MatchClause, WithClause)> {
    // Shape: body is a single part (no set-ops) that is exactly
    // `MATCH <non-optional> RETURN <plain projection>`.
    if !c.body.ops.is_empty() || c.body.parts.len() != 1 {
        return None;
    }
    let part = &c.body.parts[0];
    if part.clauses.len() != 2 {
        return None;
    }
    let Clause::Match(m) = &part.clauses[0] else {
        return None;
    };
    let Clause::Return(proj) = &part.clauses[1] else {
        return None;
    };
    if m.optional
        || proj.star
        || proj.distinct
        || !proj.order_by.is_empty()
        || proj.skip.is_some()
        || proj.limit.is_some()
    {
        return None;
    }

    // Vars the nested pattern introduces.
    let mut nested = Vec::new();
    for p in &m.patterns {
        pattern_bound_vars(p, &mut nested);
    }
    // Collision guard: a nested-only var (not an imported scope var) must not
    // clash with an outer var — decorrelating would wrongly join on it.
    for v in &nested {
        if !c.scope.contains(v) && outer.contains(v) {
            return None;
        }
    }
    // Isolation guard: the body must reference only scope vars + its own pattern
    // vars. If it references an *unscoped* outer var, the correlated form reads
    // NULL (isolated) but the flat form would read the value — a divergence.
    let mut free = Vec::new();
    for p in &m.patterns {
        collect_pattern_free_vars(p, &nested, &mut free);
    }
    if let Some(w) = &m.where_ {
        collect_free_vars(w, &nested, &mut free);
    }
    for it in &proj.items {
        collect_free_vars(&it.expr, &nested, &mut free);
    }
    for v in &free {
        if outer.contains(v) && !c.scope.contains(v) {
            return None;
        }
    }

    // Compile the projection with the same aggregate detection the real lowering
    // uses (no divergence risk), to get `aggregating` + the output column names.
    let mut probe = Lowerer {
        params: Vec::new(),
        scope: [outer, nested.as_slice()].concat(),
        keys: Vec::new(),
        labels: Vec::new(),
        unknown_fns: Vec::new(),
    };
    let cproj = probe.projection(proj, true);
    // A GLOBAL-aggregate body (all items aggregate, no inner grouping key) yields
    // exactly ONE row per outer row — the zero-aggregate over an empty match (e.g.
    // `count` = 0) — so the outer row is always kept. Decorrelating to `OPTIONAL
    // MATCH … WITH <outer>, <aggs>` reproduces that: the null-fill row groups under
    // the outer key with the aggregate over null (count = 0, sum = null). Now that
    // grouped + OPTIONAL aggregation is columnar (fast), this is a real win, not
    // just correct. A body that GROUPS (has a non-agg item beyond the outer vars)
    // yields ZERO rows for an empty match — dropping the outer row — which the
    // null-fill would NOT reproduce, so leave those correlated.
    if cproj.aggregating && !cproj.items.iter().all(|i| i.is_agg) {
        return None;
    }
    let aggregating = cproj.aggregating;
    // Column-collision guard: a projected output name must not shadow an outer var.
    for name in &cproj.out_names {
        if outer.iter().any(|v| v == name) {
            return None;
        }
    }

    // Flatten: `[OPTIONAL] MATCH <patterns> [WHERE …]` then a WITH that carries the
    // outer vars unchanged and adds the subquery's projected columns (dropping the
    // nested-only pattern vars — exactly the merge semantics of the CALL).
    let mut items: Vec<ReturnItem> = outer
        .iter()
        .map(|v| ReturnItem {
            expr: Expr::Var(v.clone()),
            alias: None,
        })
        .collect();
    items.extend(proj.items.iter().cloned());

    let mat = MatchClause {
        // A global-aggregate body must keep every outer row (the aggregate is
        // defined over the empty match), so its flat form is always OPTIONAL —
        // regardless of whether the CALL itself was optional.
        optional: c.optional || aggregating,
        patterns: m.patterns.clone(),
        where_: m.where_.clone(),
    };
    let with = WithClause {
        projection: Projection {
            star: false,
            items,
            distinct: false,
            group_by: Vec::new(),
            having: None,
            order_by: Vec::new(),
            skip: None,
            limit: None,
        },
        where_: None,
    };
    Some((mat, with))
}

pub fn lower(query: &Query) -> (CQuery, Vec<String>) {
    let mut l = Lowerer {
        params: Vec::new(),
        scope: Vec::new(),
        keys: Vec::new(),
        labels: Vec::new(),
        unknown_fns: Vec::new(),
    };
    let parts = query.parts.iter().map(|p| l.linear(p)).collect();
    let cquery = CQuery {
        parts,
        ops: query.ops.clone(),
        key_names: l.keys,
        label_names: l.labels,
        unknown_fns: l.unknown_fns,
    };
    (cquery, l.params)
}
