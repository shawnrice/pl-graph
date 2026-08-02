//! The runtime value both query engines carry.
//!
//! GQL's `Val` and Gremlin's `GVal` were two enums that agreed on eight variants
//! (`Null`, `Bool`, `Num`, `Str`, `Temporal`, the two element handles, `List`)
//! and differed on three. This is their union, and both are now aliases of it.
//!
//! # Why a union rather than a common subset
//!
//! The three differing variants are not accidents — each belongs to one
//! language's data model and has no meaning in the other:
//!
//! - [`Value::Record`] is the ISO `<record>`: string field names, **keys kept
//!   sorted** so equality is a slice compare, `Arc`-boxed so a per-row binding
//!   clone is a refcount bump.
//! - [`Value::Map`] is TinkerPop's map: **any** value as a key, **insertion
//!   ordered**, because `valueMap`/`group`/`project` preserve the order the
//!   traversal produced.
//! - [`Value::Path`] is a GQL walked path; [`Value::Property`] is a Gremlin
//!   property element.
//!
//! Collapsing `Record` and `Map` into one representation would mean giving one
//! language the other's ordering and key rules, which is a semantic change, not
//! a refactor. So each language uses the subset it has always used, and the
//! compiler stops caring which file the variant is spelled in.
//!
//! # Size
//!
//! `Path` and `Property` are boxed, which is what makes this free. Measured:
//!
//! ```text
//!   Val (before)        48 bytes
//!   GVal (before)       40
//!   union, naive        48     ← GVal would have grown 20%
//!   union, boxed        40     ← Gremlin unchanged, GQL 17% smaller
//! ```
//!
//! Both boxed variants are rare (a path is bound only by a quantified pattern; a
//! property element only by `.properties(k)`), while `Temporal` — 40 bytes, the
//! variant that actually sets the floor — is common.
//!
//! # Equality
//!
//! [`PartialEq`] is **TinkerPop's**, and it exists because Gremlin steps compare
//! values directly. GQL must not use it: ISO equality is three-valued (null
//! propagates, cross-type is UNKNOWN rather than false) and lives in
//! `gql::eval::compare_vals`. `Val` never had a `PartialEq` before this merge
//! and GQL code should keep behaving as though it still doesn't.

use std::sync::Arc;

use crate::temporal::Temporal;

/// A GQL walked path: interleaved vertices and edges, `vertices.len() ==
/// edges.len() + 1`. Boxed inside [`Value`] — see the module docs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathVal {
    pub vertices: Vec<u32>,
    pub edges: Vec<u32>,
}

/// A Gremlin property element from `.properties(k)`.
///
/// The owner is carried EXPLICITLY rather than recovered from the traverser
/// path, so `.drop()` deletes exactly this property and can never mistake a
/// `project('key')` map for a property element.
#[derive(Clone, Debug)]
pub struct PropertyVal {
    pub owner: Value,
    pub key: Arc<str>,
    pub value: Value,
}

/// The runtime value of both engines. See the module docs for which variants
/// belong to which language.
#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    /// Interned: cloning is a refcount bump, not an allocation.
    Str(Arc<str>),
    /// An ISO temporal scalar (`DATE` / `LOCAL DATETIME` / `DURATION`).
    Temporal(Temporal),
    /// A vertex handle by dense index. Gremlin spelled this `Vertex`.
    Node(u32),
    Edge(u32),
    List(Vec<Self>),
    /// ISO record — string keys, kept SORTED, `Arc`-boxed. GQL only.
    Record(Arc<[(Arc<str>, Self)]>),
    /// TinkerPop map — any-value keys, INSERTION ordered. Gremlin only.
    Map(Vec<(Self, Self)>),
    /// GQL only.
    Path(Box<PathVal>),
    /// Gremlin only.
    Property(Box<PropertyVal>),
}

impl Value {
    /// This value as a property-index key, or `None` if no index can hold it.
    ///
    /// One function for both engines, which is the first thing the merged type
    /// bought. The two copies had already drifted: Gremlin's had no `Temporal`
    /// arm, so `has('when', DATE '…')` could not seek a temporal index while the
    /// same predicate in GQL could. GQL's rebuilt the `Arc<str>` (`as_ref().into()`)
    /// where a clone is a refcount bump.
    #[must_use]
    pub fn index_key(&self) -> Option<crate::graph::IdxKey> {
        use crate::graph::IdxKey;

        match self {
            Self::Str(s) => Some(IdxKey::Str(s.clone())),
            Self::Num(n) => Some(IdxKey::Num(*n)),
            Self::Bool(b) => Some(IdxKey::Bool(*b)),
            Self::Temporal(t) => t.index_key().map(|(k, key)| IdxKey::Temporal(k, key)),
            // Null, lists, records, maps, element handles, paths and property
            // elements are not indexable.
            _ => None,
        }
    }

    /// A stored [`crate::graph::Value`] as a runtime value.
    ///
    /// Every arm but one is the same for both engines, and the one that differs
    /// is the reason `Record` and `Map` are separate variants: GQL reads a
    /// stored map as an ISO record (keys already canonical/sorted in the store),
    /// Gremlin as a TinkerPop map so it flows through `valueMap`/`select`/
    /// `order(local)` like any other map. `as_record` picks.
    ///
    /// Shared because the two copies of this had already drifted once — see
    /// [`Value::index_key`] — and six identical arms is exactly the surface that
    /// drift hides in.
    #[must_use]
    pub fn from_stored(v: &crate::graph::Value, as_record: bool) -> Self {
        use crate::graph::Value as Stored;

        match v {
            Stored::Null => Self::Null,
            Stored::Bool(b) => Self::Bool(*b),
            Stored::Num(n) => Self::Num(*n),
            // Shared `Arc` — a refcount bump, not an allocation.
            Stored::Str(s) => Self::Str(s.clone()),
            Stored::Temporal(t) => Self::Temporal(*t),
            Stored::List(items) => Self::List(
                items
                    .iter()
                    .map(|x| Self::from_stored(x, as_record))
                    .collect(),
            ),
            Stored::Map(pairs) if as_record => Self::Record(
                pairs
                    .iter()
                    .map(|(k, v)| (k.clone(), Self::from_stored(v, as_record)))
                    .collect(),
            ),
            Stored::Map(pairs) => Self::Map(
                pairs
                    .iter()
                    .map(|(k, v)| (Self::Str(k.clone()), Self::from_stored(v, as_record)))
                    .collect(),
            ),
        }
    }

    /// A path value, boxing the payload.
    #[must_use]
    pub fn path(vertices: Vec<u32>, edges: Vec<u32>) -> Self {
        Self::Path(Box::new(PathVal { vertices, edges }))
    }

    /// A property element, boxing the payload.
    #[must_use]
    pub fn property(owner: Self, key: Arc<str>, value: Self) -> Self {
        Self::Property(Box::new(PropertyVal { owner, key, value }))
    }
}

/// TinkerPop equality — see the module docs. NOT ISO equality.
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Num(a), Self::Num(b)) => a == b, // f64: NaN != NaN, as derive would
            (Self::Str(a), Self::Str(b)) => a == b,
            (Self::Temporal(a), Self::Temporal(b)) => a == b,
            (Self::Node(a), Self::Node(b)) => a == b,
            (Self::Edge(a), Self::Edge(b)) => a == b,
            (Self::List(a), Self::List(b)) => a == b,
            (Self::Map(a), Self::Map(b)) => a == b,
            (Self::Record(a), Self::Record(b)) => a == b,
            (Self::Path(a), Self::Path(b)) => a == b,
            // Owner ignored: a property element's observable identity is its
            // key + value (the owner is internal drop-routing metadata).
            (Self::Property(a), Self::Property(b)) => a.key == b.key && a.value == b.value,
            _ => false,
        }
    }
}

impl From<f64> for Value {
    fn from(n: f64) -> Self {
        Self::Num(n)
    }
}
impl From<i32> for Value {
    fn from(n: i32) -> Self {
        Self::Num(f64::from(n))
    }
}
impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}
impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Self::Str(Arc::from(s))
    }
}
impl From<String> for Value {
    fn from(s: String) -> Self {
        Self::Str(Arc::from(s.as_str()))
    }
}
