//! A dual Gremlin builder: every method has the SAME signature as core's
//! `lenke_core::gremlin::Traversal`, but each call drives BOTH core's builder AND an
//! equivalent engine query string. `to_core()` hands back the core `Traversal`;
//! `query()` the engine string. A ported test runs the two, compares them, and
//! asserts against the same expected value — so core's test bodies run VERBATIM
//! against both engines. `P`, `Order`, `Column`, `Pop`, `Token`, `Scope`, `SackOp`
//! are re-used from core directly; a shim `P` carries the engine fragment alongside.
//!
//! A method whose engine spelling is not yet handled still builds the string; the
//! engine's parser then errors on it — surfacing the real gap when the test runs.

#![allow(
    dead_code,
    non_snake_case,
    clippy::wrong_self_convention,
    clippy::should_implement_trait
)]

use lenke_core::gremlin::{self as cg};
pub use lenke_core::gremlin::{Column, GVal, Order, Pop, SackOp, Scope, Token};

fn dir_word(d: Order) -> &'static str {
    match d {
        Order::Desc => "desc",
        _ => "asc",
    }
}

/// Render a `GVal` as an engine literal fragment.
pub fn val(v: &GVal) -> String {
    match v {
        GVal::Str(s) => format!("'{s}'"),
        GVal::Num(n) if n.fract() == 0.0 && n.is_finite() => format!("{}", *n as i64),
        GVal::Num(n) => format!("{n}"),
        GVal::Bool(b) => format!("{b}"),
        GVal::Null => "null".into(),
        // A list renders as a `[…]` literal (each element via `val`), so
        // inject([1,2,3]) drives a real list-literal query, not Debug output.
        GVal::List(items) => {
            let inner = items.iter().map(val).collect::<Vec<_>>().join(",");
            format!("[{inner}]")
        }
        other => format!("{other:?}"),
    }
}

fn labels(ls: &[&str]) -> String {
    ls.iter()
        .map(|l| format!("'{l}'"))
        .collect::<Vec<_>>()
        .join(",")
}

// ── predicate shim ───────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct P {
    core: cg::P,
    eng: String,
}

impl P {
    pub fn eq(v: impl Into<GVal>) -> Self {
        let g = v.into();
        Self {
            eng: format!("eq({})", val(&g)),
            core: cg::P::eq(g),
        }
    }
    pub fn neq(v: impl Into<GVal>) -> Self {
        let g = v.into();
        Self {
            eng: format!("neq({})", val(&g)),
            core: cg::P::neq(g),
        }
    }
    pub fn gt(v: impl Into<GVal>) -> Self {
        let g = v.into();
        Self {
            eng: format!("gt({})", val(&g)),
            core: cg::P::gt(g),
        }
    }
    pub fn gte(v: impl Into<GVal>) -> Self {
        let g = v.into();
        Self {
            eng: format!("gte({})", val(&g)),
            core: cg::P::gte(g),
        }
    }
    pub fn lt(v: impl Into<GVal>) -> Self {
        let g = v.into();
        Self {
            eng: format!("lt({})", val(&g)),
            core: cg::P::lt(g),
        }
    }
    pub fn lte(v: impl Into<GVal>) -> Self {
        let g = v.into();
        Self {
            eng: format!("lte({})", val(&g)),
            core: cg::P::lte(g),
        }
    }
    pub fn between(a: impl Into<GVal>, b: impl Into<GVal>) -> Self {
        let (ga, gb) = (a.into(), b.into());
        Self {
            eng: format!("between({},{})", val(&ga), val(&gb)),
            core: cg::P::between(ga, gb),
        }
    }
    pub fn inside(a: impl Into<GVal>, b: impl Into<GVal>) -> Self {
        let (ga, gb) = (a.into(), b.into());
        Self {
            eng: format!("inside({},{})", val(&ga), val(&gb)),
            core: cg::P::inside(ga, gb),
        }
    }
    pub fn outside(a: impl Into<GVal>, b: impl Into<GVal>) -> Self {
        let (ga, gb) = (a.into(), b.into());
        Self {
            eng: format!("outside({},{})", val(&ga), val(&gb)),
            core: cg::P::outside(ga, gb),
        }
    }
    pub fn within<V: Into<GVal>>(vs: impl IntoIterator<Item = V>) -> Self {
        let gs: Vec<GVal> = vs.into_iter().map(Into::into).collect();
        let frag = gs.iter().map(val).collect::<Vec<_>>().join(",");
        Self {
            eng: format!("within({frag})"),
            core: cg::P::within(gs),
        }
    }
    pub fn without<V: Into<GVal>>(vs: impl IntoIterator<Item = V>) -> Self {
        let gs: Vec<GVal> = vs.into_iter().map(Into::into).collect();
        let frag = gs.iter().map(val).collect::<Vec<_>>().join(",");
        Self {
            eng: format!("without({frag})"),
            core: cg::P::without(gs),
        }
    }
    pub fn starts_with(s: &str) -> Self {
        Self {
            eng: format!("startingWith('{s}')"),
            core: cg::P::starts_with(s),
        }
    }
    pub fn containing(s: &str) -> Self {
        Self {
            eng: format!("containing('{s}')"),
            core: cg::P::containing(s),
        }
    }
    pub fn regex(s: &str) -> Self {
        Self {
            eng: format!("regex('{s}')"),
            core: cg::P::regex(s),
        }
    }
    pub fn not(p: Self) -> Self {
        Self {
            eng: format!("not({})", p.eng),
            core: cg::P::not(p.core),
        }
    }
}

// ── traversal shim ───────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Traversal {
    core: cg::Traversal,
    eng: String,
}

pub fn g() -> Traversal {
    Traversal {
        core: cg::g(),
        eng: "g".into(),
    }
}

pub fn __() -> Traversal {
    Traversal {
        core: cg::__(),
        eng: "__".into(),
    }
}

impl Traversal {
    pub fn to_core(self) -> cg::Traversal {
        self.core
    }
    pub fn core_ref(&self) -> &cg::Traversal {
        &self.core
    }
    pub fn query(&self) -> String {
        self.eng.clone()
    }

    fn step(mut self, core: cg::Traversal, frag: &str) -> Self {
        self.core = core;
        self.eng.push('.');
        self.eng.push_str(frag);
        self
    }

    // sources
    pub fn V(self) -> Self {
        let c = self.core.clone().V();
        self.step(c, "V()")
    }
    pub fn E(self) -> Self {
        let c = self.core.clone().E();
        self.step(c, "E()")
    }
    pub fn v_ids(self, ids: &[&str]) -> Self {
        let c = self.core.clone().v_ids(ids);
        self.step(c, &format!("V({})", labels(ids)))
    }
    pub fn e_ids(self, ids: &[&str]) -> Self {
        let c = self.core.clone().e_ids(ids);
        self.step(c, &format!("E({})", labels(ids)))
    }

    // hops
    pub fn out(self, l: &[&str]) -> Self {
        let c = self.core.clone().out(l);
        self.step(c, &format!("out({})", labels(l)))
    }
    pub fn in_(self, l: &[&str]) -> Self {
        let c = self.core.clone().in_(l);
        self.step(c, &format!("in({})", labels(l)))
    }
    pub fn both(self, l: &[&str]) -> Self {
        let c = self.core.clone().both(l);
        self.step(c, &format!("both({})", labels(l)))
    }
    pub fn out_e(self, l: &[&str]) -> Self {
        let c = self.core.clone().out_e(l);
        self.step(c, &format!("outE({})", labels(l)))
    }
    pub fn in_e(self, l: &[&str]) -> Self {
        let c = self.core.clone().in_e(l);
        self.step(c, &format!("inE({})", labels(l)))
    }
    pub fn both_e(self, l: &[&str]) -> Self {
        let c = self.core.clone().both_e(l);
        self.step(c, &format!("bothE({})", labels(l)))
    }
    pub fn out_v(self) -> Self {
        let c = self.core.clone().out_v();
        self.step(c, "outV()")
    }
    pub fn in_v(self) -> Self {
        let c = self.core.clone().in_v();
        self.step(c, "inV()")
    }
    pub fn other_v(self) -> Self {
        let c = self.core.clone().other_v();
        self.step(c, "otherV()")
    }
    pub fn both_v(self) -> Self {
        let c = self.core.clone().both_v();
        self.step(c, "bothV()")
    }

    // filters
    pub fn has(self, key: &str, pred: P) -> Self {
        let c = self.core.clone().has(key, pred.core);
        self.step(c, &format!("has('{key}',{})", pred.eng))
    }
    pub fn has_val(self, key: &str, v: impl Into<GVal>) -> Self {
        let g = v.into();
        let c = self.core.clone().has_val(key, g.clone());
        self.step(c, &format!("has('{key}',{})", val(&g)))
    }
    pub fn has_label_key(self, label: &str, key: &str, pred: P) -> Self {
        let c = self.core.clone().has_label_key(label, key, pred.core);
        self.step(c, &format!("has('{label}','{key}',{})", pred.eng))
    }
    pub fn has_label(self, l: &[&str]) -> Self {
        let c = self.core.clone().has_label(l);
        self.step(c, &format!("hasLabel({})", labels(l)))
    }
    pub fn has_id(self, ids: &[&str]) -> Self {
        let c = self.core.clone().has_id(ids);
        self.step(c, &format!("hasId({})", labels(ids)))
    }
    pub fn has_key(self, keys: &[&str]) -> Self {
        let c = self.core.clone().has_key(keys);
        self.step(c, &format!("hasKey({})", labels(keys)))
    }
    pub fn has_not(self, keys: &[&str]) -> Self {
        let c = self.core.clone().has_not(keys);
        self.step(c, &format!("hasNot({})", labels(keys)))
    }
    pub fn has_value<V: Into<GVal>>(self, vs: impl IntoIterator<Item = V>) -> Self {
        let gs: Vec<GVal> = vs.into_iter().map(Into::into).collect();
        let frag = gs.iter().map(val).collect::<Vec<_>>().join(",");
        let c = self.core.clone().has_value(gs);
        self.step(c, &format!("hasValue({frag})"))
    }
    pub fn is(self, pred: P) -> Self {
        let c = self.core.clone().is(pred.core);
        self.step(c, &format!("is({})", pred.eng))
    }
    pub fn where_(self, sub: Self) -> Self {
        let c = self.core.clone().where_(sub.core);
        self.step(c, &format!("where({})", sub.eng))
    }
    pub fn where_key(self, start: &str, pred: P) -> Self {
        let c = self.core.clone().where_key(start, pred.core);
        self.step(c, &format!("where('{start}',{})", pred.eng))
    }
    pub fn where_pred(self, pred: P) -> Self {
        let c = self.core.clone().where_pred(pred.core);
        self.step(c, &format!("where({})", pred.eng))
    }
    pub fn and(self, plans: Vec<Self>) -> Self {
        let frag = plans
            .iter()
            .map(|p| p.eng.clone())
            .collect::<Vec<_>>()
            .join(",");
        let c = self
            .core
            .clone()
            .and(plans.into_iter().map(|p| p.core).collect());
        self.step(c, &format!("and({frag})"))
    }
    pub fn or(self, plans: Vec<Self>) -> Self {
        let frag = plans
            .iter()
            .map(|p| p.eng.clone())
            .collect::<Vec<_>>()
            .join(",");
        let c = self
            .core
            .clone()
            .or(plans.into_iter().map(|p| p.core).collect());
        self.step(c, &format!("or({frag})"))
    }
    pub fn not(self, sub: Self) -> Self {
        let c = self.core.clone().not(sub.core);
        self.step(c, &format!("not({})", sub.eng))
    }
    pub fn dedup(self) -> Self {
        let c = self.core.clone().dedup();
        self.step(c, "dedup()")
    }
    pub fn dedup_labels(self, ls: Vec<String>) -> Self {
        let frag = ls
            .iter()
            .map(|l| format!("'{l}'"))
            .collect::<Vec<_>>()
            .join(",");
        let c = self.core.clone().dedup_labels(ls);
        self.step(c, &format!("dedup({frag})"))
    }
    pub fn simple_path(self) -> Self {
        let c = self.core.clone().simple_path();
        self.step(c, "simplePath()")
    }
    pub fn cyclic_path(self) -> Self {
        let c = self.core.clone().cyclic_path();
        self.step(c, "cyclicPath()")
    }

    // projections
    pub fn values(self, keys: &[&str]) -> Self {
        let c = self.core.clone().values(keys);
        self.step(c, &format!("values({})", labels(keys)))
    }
    pub fn value_map(self, keys: &[&str]) -> Self {
        let c = self.core.clone().value_map(keys);
        self.step(c, &format!("valueMap({})", labels(keys)))
    }
    pub fn property_map(self, keys: &[&str]) -> Self {
        let c = self.core.clone().property_map(keys);
        self.step(c, &format!("propertyMap({})", labels(keys)))
    }
    pub fn element_map(self, keys: &[&str]) -> Self {
        let c = self.core.clone().element_map(keys);
        self.step(c, &format!("elementMap({})", labels(keys)))
    }
    pub fn properties(self, keys: &[&str]) -> Self {
        let c = self.core.clone().properties(keys);
        self.step(c, &format!("properties({})", labels(keys)))
    }
    pub fn value(self) -> Self {
        let c = self.core.clone().value();
        self.step(c, "value()")
    }
    pub fn id(self) -> Self {
        let c = self.core.clone().id();
        self.step(c, "id()")
    }
    pub fn label(self) -> Self {
        let c = self.core.clone().label();
        self.step(c, "label()")
    }
    pub fn path(self) -> Self {
        let c = self.core.clone().path();
        self.step(c, "path()")
    }
    pub fn project(self, keys: &[&str]) -> Self {
        let c = self.core.clone().project(keys);
        self.step(c, &format!("project({})", labels(keys)))
    }
    pub fn tree(self) -> Self {
        let c = self.core.clone().tree();
        self.step(c, "tree()")
    }

    // modulators
    pub fn by(self, key: &str) -> Self {
        let c = self.core.clone().by(key);
        self.step(c, &format!("by('{key}')"))
    }
    pub fn by_identity(self) -> Self {
        let c = self.core.clone().by_identity();
        self.step(c, "by()")
    }
    pub fn by_dir(self, key: &str, dir: Order) -> Self {
        let c = self.core.clone().by_dir(key, dir);
        self.step(c, &format!("by('{key}',{})", dir_word(dir)))
    }
    pub fn by_identity_dir(self, dir: Order) -> Self {
        let c = self.core.clone().by_identity_dir(dir);
        self.step(c, &format!("by({})", dir_word(dir)))
    }
    pub fn by_t(self, t: Self) -> Self {
        let c = self.core.clone().by_t(t.core);
        self.step(c, &format!("by({})", t.eng))
    }
    pub fn by_t_dir(self, t: Self, dir: Order) -> Self {
        let c = self.core.clone().by_t_dir(t.core, dir);
        self.step(c, &format!("by({},{})", t.eng, dir_word(dir)))
    }

    // paging / bounds
    pub fn limit(self, n: usize) -> Self {
        let c = self.core.clone().limit(n);
        self.step(c, &format!("limit({n})"))
    }
    pub fn limit_local(self, n: usize) -> Self {
        let c = self.core.clone().limit_local(n);
        self.step(c, &format!("limit(local,{n})"))
    }
    pub fn skip(self, n: usize) -> Self {
        let c = self.core.clone().skip(n);
        self.step(c, &format!("skip({n})"))
    }
    pub fn range(self, a: usize, b: usize) -> Self {
        let c = self.core.clone().range(a, b);
        self.step(c, &format!("range({a},{b})"))
    }
    pub fn range_local(self, a: usize, b: usize) -> Self {
        let c = self.core.clone().range_local(a, b);
        self.step(c, &format!("range(local,{a},{b})"))
    }
    pub fn tail(self, n: usize) -> Self {
        let c = self.core.clone().tail(n);
        self.step(c, &format!("tail({n})"))
    }
    pub fn tail_local(self, n: usize) -> Self {
        let c = self.core.clone().tail_local(n);
        self.step(c, &format!("tail(local,{n})"))
    }
    pub fn sample(self, n: usize) -> Self {
        let c = self.core.clone().sample(n);
        self.step(c, &format!("sample({n})"))
    }

    // reducers
    pub fn count(self) -> Self {
        let c = self.core.clone().count();
        self.step(c, "count()")
    }
    pub fn count_local(self) -> Self {
        let c = self.core.clone().count_local();
        self.step(c, "count(local)")
    }
    pub fn fold(self) -> Self {
        let c = self.core.clone().fold();
        self.step(c, "fold()")
    }
    pub fn sum(self) -> Self {
        let c = self.core.clone().sum();
        self.step(c, "sum()")
    }
    pub fn sum_local(self) -> Self {
        let c = self.core.clone().sum_local();
        self.step(c, "sum(local)")
    }
    pub fn min(self) -> Self {
        let c = self.core.clone().min();
        self.step(c, "min()")
    }
    pub fn min_local(self) -> Self {
        let c = self.core.clone().min_local();
        self.step(c, "min(local)")
    }
    pub fn max(self) -> Self {
        let c = self.core.clone().max();
        self.step(c, "max()")
    }
    pub fn max_local(self) -> Self {
        let c = self.core.clone().max_local();
        self.step(c, "max(local)")
    }
    pub fn mean(self) -> Self {
        let c = self.core.clone().mean();
        self.step(c, "mean()")
    }
    pub fn mean_local(self) -> Self {
        let c = self.core.clone().mean_local();
        self.step(c, "mean(local)")
    }
    pub fn group(self) -> Self {
        let c = self.core.clone().group();
        self.step(c, "group()")
    }
    pub fn group_count(self) -> Self {
        let c = self.core.clone().group_count();
        self.step(c, "groupCount()")
    }

    // order
    pub fn order(self) -> Self {
        let c = self.core.clone().order();
        self.step(c, "order()")
    }
    pub fn order_local(self) -> Self {
        let c = self.core.clone().order_local();
        self.step(c, "order(local)")
    }
    pub fn order_dir(self, dir: Order, scope: Scope) -> Self {
        let c = self.core.clone().order_dir(dir, scope);
        let sc = if matches!(scope, Scope::Local) {
            "local"
        } else {
            "global"
        };
        let _ = sc;
        self.step(c, &format!("order({}).by({})", "", dir_word(dir)))
    }
    pub fn order_by(self, key: &str, dir: Order) -> Self {
        let c = self.core.clone().order_by(key, dir);
        self.step(c, &format!("order().by('{key}',{})", dir_word(dir)))
    }

    // branch / control
    pub fn union(self, plans: Vec<Self>) -> Self {
        let frag = plans
            .iter()
            .map(|p| p.eng.clone())
            .collect::<Vec<_>>()
            .join(",");
        let c = self
            .core
            .clone()
            .union(plans.into_iter().map(|p| p.core).collect());
        self.step(c, &format!("union({frag})"))
    }
    pub fn coalesce(self, plans: Vec<Self>) -> Self {
        let frag = plans
            .iter()
            .map(|p| p.eng.clone())
            .collect::<Vec<_>>()
            .join(",");
        let c = self
            .core
            .clone()
            .coalesce(plans.into_iter().map(|p| p.core).collect());
        self.step(c, &format!("coalesce({frag})"))
    }
    pub fn optional(self, sub: Self) -> Self {
        let c = self.core.clone().optional(sub.core);
        self.step(c, &format!("optional({})", sub.eng))
    }
    pub fn local(self, sub: Self) -> Self {
        let c = self.core.clone().local(sub.core);
        self.step(c, &format!("local({})", sub.eng))
    }
    pub fn choose(self, test: Self, then_: Self) -> Self {
        let c = self.core.clone().choose(test.core, then_.core);
        self.step(c, &format!("choose({},{})", test.eng, then_.eng))
    }
    pub fn choose_else(self, test: Self, then_: Self, else_: Self) -> Self {
        let c = self
            .core
            .clone()
            .choose_else(test.core, then_.core, else_.core);
        self.step(
            c,
            &format!("choose({},{},{})", test.eng, then_.eng, else_.eng),
        )
    }
    pub fn branch(self, test: Self) -> Self {
        let c = self.core.clone().branch(test.core);
        self.step(c, &format!("branch({})", test.eng))
    }
    pub fn option(self, m: impl Into<GVal>, plan: Self) -> Self {
        let g = m.into();
        let c = self.core.clone().option(g.clone(), plan.core);
        self.step(c, &format!("option({},{})", val(&g), plan.eng))
    }
    pub fn option_none(self, plan: Self) -> Self {
        let c = self.core.clone().option_none(plan.core);
        self.step(c, &format!("option(none,{})", plan.eng))
    }
    pub fn map(self, sub: Self) -> Self {
        let c = self.core.clone().map(sub.core);
        self.step(c, &format!("map({})", sub.eng))
    }
    pub fn flat_map(self, sub: Self) -> Self {
        let c = self.core.clone().flat_map(sub.core);
        self.step(c, &format!("flatMap({})", sub.eng))
    }
    pub fn filter(self, sub: Self) -> Self {
        let c = self.core.clone().filter(sub.core);
        self.step(c, &format!("filter({})", sub.eng))
    }
    pub fn side_effect(self, sub: Self) -> Self {
        let c = self.core.clone().side_effect(sub.core);
        self.step(c, &format!("sideEffect({})", sub.eng))
    }
    pub fn match_(self, patterns: Vec<Self>) -> Self {
        let frag = patterns
            .iter()
            .map(|p| p.eng.clone())
            .collect::<Vec<_>>()
            .join(",");
        let c = self
            .core
            .clone()
            .match_(patterns.into_iter().map(|p| p.core).collect());
        self.step(c, &format!("match({frag})"))
    }

    // side effects / bags
    pub fn aggregate(self, key: &str) -> Self {
        let c = self.core.clone().aggregate(key);
        self.step(c, &format!("aggregate('{key}')"))
    }
    pub fn store(self, key: &str) -> Self {
        let c = self.core.clone().store(key);
        self.step(c, &format!("store('{key}')"))
    }
    pub fn cap(self, key: &str) -> Self {
        let c = self.core.clone().cap(key);
        self.step(c, &format!("cap('{key}')"))
    }
    pub fn subgraph(self, key: &str) -> Self {
        let c = self.core.clone().subgraph(key);
        self.step(c, &format!("subgraph('{key}')"))
    }
    pub fn barrier(self) -> Self {
        let c = self.core.clone().barrier();
        self.step(c, "barrier()")
    }

    // OLAP
    pub fn shortest_path(self) -> Self {
        let c = self.core.clone().shortest_path();
        self.step(c, "shortestPath()")
    }
    pub fn shortest_path_to(self, target: Self) -> Self {
        let c = self.core.clone().shortest_path_to(target.core);
        // Engine has no target-filtered form; render the target selector so the engine
        // parser surfaces it as a gap rather than silently computing all paths.
        self.step(c, &format!("shortestPath().with(target,{})", target.eng))
    }
    pub fn from_tag(self, label: &str) -> Self {
        let c = self.core.clone().from_tag(label);
        self.step(c, &format!("from('{label}')"))
    }
    pub fn page_rank(self, alpha: Option<f64>) -> Self {
        let c = self.core.clone().page_rank(alpha);
        match alpha {
            Some(a) => self.step(c, &format!("pageRank({a})")),
            None => self.step(c, "pageRank()"),
        }
    }
    pub fn connected_component(self) -> Self {
        let c = self.core.clone().connected_component();
        self.step(c, "connectedComponent()")
    }
    pub fn peer_pressure(self) -> Self {
        let c = self.core.clone().peer_pressure();
        self.step(c, "peerPressure()")
    }

    // repeat
    pub fn repeat(self, body: Self) -> Self {
        let c = self.core.clone().repeat(body.core);
        self.step(c, &format!("repeat({})", body.eng))
    }
    pub fn times(self, n: usize) -> Self {
        let c = self.core.clone().times(n);
        self.step(c, &format!("times({n})"))
    }
    pub fn until(self, cond: Self) -> Self {
        let c = self.core.clone().until(cond.core);
        self.step(c, &format!("until({})", cond.eng))
    }
    pub fn emit(self, cond: Self) -> Self {
        let c = self.core.clone().emit(cond.core);
        self.step(c, &format!("emit({})", cond.eng))
    }
    pub fn emit_all(self) -> Self {
        let c = self.core.clone().emit_all();
        self.step(c, "emit()")
    }

    // tags / select
    pub fn as_(self, label: &str) -> Self {
        let c = self.core.clone().as_(label);
        self.step(c, &format!("as('{label}')"))
    }
    pub fn select(self, labels_: &[&str]) -> Self {
        let c = self.core.clone().select(labels_);
        self.step(c, &format!("select({})", labels(labels_)))
    }
    pub fn select_pop(self, pop: Pop, labels_: &[&str]) -> Self {
        let c = self.core.clone().select_pop(pop, labels_);
        let pw = format!("{pop:?}");
        self.step(c, &format!("select({},{})", pw, labels(labels_)))
    }
    pub fn select_column(self, col: Column) -> Self {
        let c = self.core.clone().select_column(col);
        let cw = if matches!(col, Column::Keys) {
            "keys"
        } else {
            "values"
        };
        self.step(c, &format!("select({cw})"))
    }

    // misc
    pub fn unfold(self) -> Self {
        let c = self.core.clone().unfold();
        self.step(c, "unfold()")
    }
    pub fn constant(self, v: impl Into<GVal>) -> Self {
        let g = v.into();
        let c = self.core.clone().constant(g.clone());
        self.step(c, &format!("constant({})", val(&g)))
    }
    pub fn math(self, expr: &str) -> Self {
        let c = self.core.clone().math(expr);
        self.step(c, &format!("math('{expr}')"))
    }
    pub fn identity(self) -> Self {
        let c = self.core.clone().identity();
        self.step(c, "identity()")
    }
    pub fn inject<V: Into<GVal>>(self, vs: impl IntoIterator<Item = V>) -> Self {
        let gs: Vec<GVal> = vs.into_iter().map(Into::into).collect();
        let frag = gs.iter().map(val).collect::<Vec<_>>().join(",");
        let c = self.core.clone().inject(gs);
        self.step(c, &format!("inject({frag})"))
    }
    pub fn none(self) -> Self {
        let c = self.core.clone().none();
        self.step(c, "none()")
    }
    pub fn none_pred(self, pred: P) -> Self {
        let c = self.core.clone().none_pred(pred.core);
        self.step(c, &format!("none({})", pred.eng))
    }
    pub fn fail(self, msg: &str) -> Self {
        let c = self.core.clone().fail(msg);
        self.step(c, &format!("fail('{msg}')"))
    }
    pub fn index(self) -> Self {
        let c = self.core.clone().index();
        self.step(c, "index()")
    }
    pub fn loops(self) -> Self {
        let c = self.core.clone().loops();
        self.step(c, "loops()")
    }

    // sack
    pub fn with_sack(self, init: impl Into<GVal>) -> Self {
        let g = init.into();
        let c = self.core.clone().with_sack(g.clone());
        // withSack is a source prefix — it must precede the head. Model as prefix.
        let mut t = self;
        t.core = c;
        t.eng = t.eng.replacen("g", &format!("g.withSack({})", val(&g)), 1);
        t
    }
    pub fn sack(self) -> Self {
        let c = self.core.clone().sack();
        self.step(c, "sack()")
    }
    pub fn sack_op(self, op: SackOp) -> Self {
        let c = self.core.clone().sack_op(op);
        let ow = match op {
            SackOp::Sum => "sum",
            SackOp::Mult => "mult",
            SackOp::Min => "min",
            SackOp::Max => "max",
            SackOp::Assign => "assign",
        };
        self.step(c, &format!("sack({ow})"))
    }

    pub fn by_label(self) -> Self {
        let c = self.core.clone().by_label();
        self.step(c, "by(label)")
    }
    pub fn by_id(self) -> Self {
        let c = self.core.clone().by_id();
        self.step(c, "by(id)")
    }
    pub fn by_token(self, tok: Token) -> Self {
        let c = self.core.clone().by_token(tok);
        let w = match tok {
            Token::Id => "id",
            Token::Label => "label",
            Token::Key => "key",
            Token::Value => "value",
        };
        self.step(c, &format!("by({w})"))
    }

    // writes
    pub fn add_v(self, label: Option<&str>) -> Self {
        let c = self.core.clone().add_v(label);
        match label {
            Some(l) => self.step(c, &format!("addV('{l}')")),
            None => self.step(c, "addV()"),
        }
    }
    pub fn add_e(self, label: &str) -> Self {
        let c = self.core.clone().add_e(label);
        self.step(c, &format!("addE('{label}')"))
    }
    pub fn property(self, key: &str, v: impl Into<GVal>) -> Self {
        let g = v.into();
        let c = self.core.clone().property(key, g.clone());
        self.step(c, &format!("property('{key}',{})", val(&g)))
    }
    pub fn drop(self) -> Self {
        let c = self.core.clone().drop();
        self.step(c, "drop()")
    }

    // OLAP config modulators (attach to the preceding algo step)
    pub fn with_algo_property(self, name: String) -> Self {
        let c = self.core.clone().with_algo_property(name.clone());
        self.step(c, &format!("with(propertyName,'{name}')"))
    }
    pub fn with_algo_times(self, n: u32) -> Self {
        let c = self.core.clone().with_algo_times(n);
        self.step(c, &format!("with(times,{n})"))
    }

    /// Run on core (returns core's result). The dual comparison lives in the ported
    /// test's `q`, which also parses+runs `query()` on the engine.
    pub fn run(self, graph: &mut lenke_core::graph::Graph) -> Vec<GVal> {
        self.core.run(graph)
    }
}
