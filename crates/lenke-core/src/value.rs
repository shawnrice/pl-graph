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

/// A TinkerPop map: insertion-ordered, any value permitted as a key.
///
/// Keys are held SEPARATELY from values, and `Arc`-shared, because the maps that
/// dominate are projections — every element of a `valueMap()` / `elementMap()` /
/// `project()` stream has the SAME keys. Storing pairs meant cloning each key per
/// element: 600k atomic refcount bumps on twelve hot cache lines to project 50k
/// vertices, where one bump per element will do.
///
/// `keys.len() == vals.len()` is an invariant; the constructors are the only way
/// to build one.
#[derive(Clone, Debug)]
pub struct MapVal {
    /// `Arc<Vec<_>>` rather than `Arc<[_]>`: a few maps are BUILT incrementally
    /// (`tree()` grows one as it walks paths), and a shared slice cannot grow.
    /// `Arc::make_mut` copies only when the keys are actually shared.
    keys: Arc<Vec<Value>>,
    vals: Vec<Value>,
}

impl MapVal {
    /// Independent keys — the general case (`group()`, where every map differs).
    #[must_use]
    pub fn from_pairs(pairs: Vec<(Value, Value)>) -> Self {
        let mut keys = Vec::with_capacity(pairs.len());
        let mut vals = Vec::with_capacity(pairs.len());

        for (k, v) in pairs {
            keys.push(k);
            vals.push(v);
        }

        Self {
            keys: Arc::new(keys),
            vals,
        }
    }

    /// Keys shared with other maps — a projection over a stream.
    ///
    /// # Panics
    /// If `keys` and `vals` differ in length.
    #[must_use]
    pub fn with_keys(keys: Arc<Vec<Value>>, vals: Vec<Value>) -> Self {
        assert_eq!(
            keys.len(),
            vals.len(),
            "a map's keys and values must pair up"
        );

        Self { keys, vals }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.vals.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vals.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Value, &Value)> {
        self.keys.iter().zip(&self.vals)
    }

    /// The value for `key`, by TinkerPop equality.
    #[must_use]
    pub fn get(&self, key: &Value) -> Option<&Value> {
        self.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    #[must_use]
    pub fn keys(&self) -> &[Value] {
        &self.keys
    }

    /// Append an entry. Clones the key vector only if it is shared.
    pub fn push(&mut self, key: Value, val: Value) {
        Arc::make_mut(&mut self.keys).push(key);
        self.vals.push(val);
    }

    /// The value for `key`, mutably — for the few maps built incrementally.
    pub fn get_mut(&mut self, key: &Value) -> Option<&mut Value> {
        let i = self.keys.iter().position(|k| k == key)?;

        self.vals.get_mut(i)
    }

    #[must_use]
    pub fn values(&self) -> &[Value] {
        &self.vals
    }

    /// Owned pairs, for the few callers that rebuild a map.
    #[must_use]
    pub fn into_pairs(self) -> Vec<(Value, Value)> {
        self.keys.iter().cloned().zip(self.vals).collect()
    }
}

impl PartialEq for MapVal {
    fn eq(&self, other: &Self) -> bool {
        self.keys.len() == other.keys.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
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
    /// `Arc`-ed, like [`Value::Record`] and for the same reason: a list is built
    /// once and read many times, and the per-row `Binding` clone (GQL) and the
    /// per-step `Trav::tags` clone (Gremlin) both used to DEEP-COPY it. Nothing
    /// mutates a list through a pattern match, so there is no owner to lose.
    List(Arc<[Self]>),
    /// ISO record — string keys, kept SORTED, `Arc`-boxed. GQL only.
    Record(Arc<[(Arc<str>, Self)]>),
    /// TinkerPop map — any-value keys, INSERTION ordered. Gremlin only.
    Map(MapVal),
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
            Stored::Map(pairs) => Self::map(
                pairs
                    .iter()
                    .map(|(k, v)| (Self::Str(k.clone()), Self::from_stored(v, as_record)))
                    .collect(),
            ),
        }
    }

    /// A map from independent key/value pairs. See [`MapVal`] for when the keys
    /// can be shared instead.
    #[must_use]
    pub fn map(pairs: Vec<(Self, Self)>) -> Self {
        Self::Map(MapVal::from_pairs(pairs))
    }

    /// A list value. Takes the `Vec` callers naturally build and shares it.
    #[must_use]
    pub fn list(items: Vec<Self>) -> Self {
        Self::List(items.into())
    }

    /// Element `idx`'s value for the already-resolved column `kid`, read straight
    /// off the typed column.
    ///
    /// Both engines wanted this and only one had it. GQL's `prop_of` read the
    /// column directly — a string is a refcount bump, numbers and bools are
    /// copied, nothing is boxed. Gremlin's `prop` went through
    /// `Properties::value_id`, which builds a `graph::Value` first and then
    /// converts, so every property read allocated an intermediate. The only thing
    /// that ever differed between them is `as_record`, the same flag
    /// [`Value::from_stored`] already takes: a stored map is an ISO record to GQL
    /// and a TinkerPop map to Gremlin.
    #[must_use]
    pub fn from_column(
        store: &crate::graph::Properties,
        kid: u32,
        idx: usize,
        strs: &crate::graph::Dict,
        as_record: bool,
    ) -> Self {
        use crate::graph::Column;

        match store.cols.get(kid as usize) {
            Some(Column::Num { data, present }) if present.get(idx) => Self::Num(data[idx]),
            Some(Column::Bool { data, present }) if present.get(idx) => Self::Bool(data[idx]),
            Some(Column::Str { data, present }) if present.get(idx) => {
                Self::Str(strs.arc(data[idx]))
            }
            Some(Column::Temporal { data, present }) if present.get(idx) => {
                Self::Temporal(data.get(idx))
            }
            // A typed vector column reconstructs the same list the boxed form
            // would yield, via the zero-copy slice accessor.
            Some(Column::Vec { .. }) => store
                .vector_id(idx, kid)
                .map(|v| Self::list(v.iter().map(|x| Self::Num(*x)).collect()))
                .unwrap_or(Self::Null),
            Some(Column::Mixed { data }) => data[idx]
                .as_ref()
                .map(|v| Self::from_stored(v, as_record))
                .unwrap_or(Self::Null),
            // A de-boxed record synthesizes its map (or reads an escapee).
            Some(Column::Record { .. }) => {
                Self::from_stored(&store.value_id(idx, kid, strs), as_record)
            }
            _ => Self::Null,
        }
    }

    /// An element's EXTERNAL id, the string a user sees — or `Null` for anything
    /// that is not an element.
    ///
    /// Both engines needed this and both wrote it out. That duplication has
    /// already cost one wrong answer: Gremlin's copy used to pass a non-element
    /// value THROUGH instead of nulling it, so `path().id()` handed back the paths
    /// untouched and a following `sum()` faulted where the TS engine summed nulls.
    /// The Gremlin differential fuzzer found it; GQL's copy was correct all along.
    ///
    /// An edge's assigned id shadows the canonical `e{index}` — a distinction GQL's
    /// copy also had to be corrected for, separately.
    #[must_use]
    pub fn element_id(graph: &crate::graph::Graph, v: &Self) -> Self {
        match v {
            Self::Node(i) => Self::Str(graph.vid.arc(*i)),
            Self::Edge(e) => Self::Str(graph.edge_id_arc(*e)),
            _ => Self::Null,
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

/// The extreme of `values` under `cmp` — `min`/`max` for both languages.
///
/// Nulls are skipped: neither language's `min`/`max` considers one a candidate
/// (GQL filters them upstream, TinkerPop ignores them), so the rule is the same
/// and lives here once.
///
/// `cmp` is NOT. Ordering is a per-language contract — GQL's `cmp_total` puts
/// nulls last and NaN greatest, Gremlin's raises a type fault on a cross-type
/// pair before falling back to `gcmp_total` — so the comparator is the argument
/// and the FOLD is shared. That split is the whole point: it was three copies of
/// this loop, and the third had drifted.
///
/// The drift: `min(local)` used `cmp_or_fault(..) == Some(want)` with no total
/// fallback, so a NaN answered `None`, never matched `want`, and whichever NaN
/// arrived first held `best` forever. `math('sqrt _').min()` gave 2.0 and
/// `math('sqrt _').fold().min(local)` gave NaN — the same question, two scopes,
/// two answers.
pub fn fold_extreme(
    values: impl IntoIterator<Item = Value>,
    want: std::cmp::Ordering,
    mut cmp: impl FnMut(&Value, &Value) -> std::cmp::Ordering,
) -> Value {
    let mut best: Option<Value> = None;

    for v in values {
        if matches!(v, Value::Null) {
            continue;
        }

        best = Some(match best {
            None => v,
            Some(b) => {
                if cmp(&v, &b) == want {
                    v
                } else {
                    b
                }
            }
        });
    }

    best.unwrap_or(Value::Null)
}

/// A COLUMN of runtime values — the unit a lowered query carries from one step
/// to the next, in either language.
///
/// Both engines arrived at this independently and kept their own: GQL's `VVec`
/// was `Num`/`Bool`/`Gen`, Gremlin's `Col` was `Elems`/`Nums`/`Vals`. They are
/// the same idea — a column, unboxed when its type allows — and now that `Val`
/// and `GVal` are one type there is nothing left keeping them apart. The union
/// carries every variant either had: an element frontier, unboxed numbers,
/// unboxed booleans, and anything else boxed.
///
/// # What lives here and what does not
///
/// The STRUCTURE is shared: how long a column is, how it is sliced, how it
/// materializes. Anything carrying a language's SEMANTICS is not — grouping
/// identity, ordering, and three-valued truth differ between GQL and Gremlin by
/// contract, so those stay with their engine and take this as an argument. It is
/// the same split [`fold_extreme`] and [`keep_smallest`] already make: the
/// traversal is shared, the comparator is a parameter.
///
/// # Validity
///
/// `valid: None` means every row is valid, which is not the same as an all-true
/// mask — it is the absence of one. Gremlin's `values(k)` DROPS a row whose key
/// is missing, so its columns never need a mask; GQL's projection KEEPS the row
/// with a null in it, so its columns usually do. Making the mask optional is what
/// lets one type serve both without Gremlin allocating a mask per column it will
/// never read.
#[derive(Clone, Debug)]
pub enum Col<'a> {
    /// Graph elements by dense index — a frontier. Borrowed where it can be, so
    /// a paging step over one does not copy it.
    Elems {
        ids: std::borrow::Cow<'a, [u32]>,
        is_edge: bool,
    },
    Num {
        d: Vec<f64>,
        valid: Option<Vec<bool>>,
    },
    Bool {
        t: Vec<bool>,
        valid: Option<Vec<bool>>,
    },
    /// Anything the typed variants cannot hold, boxed.
    Gen(Vec<Value>),
}

impl<'a> Col<'a> {
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Elems { ids, .. } => ids.len(),
            Self::Num { d, .. } => d.len(),
            Self::Bool { t, .. } => t.len(),
            Self::Gen(v) => v.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Is row `i` a value rather than a null? An element is always one, and an
    /// absent mask means every row is.
    #[must_use]
    pub fn valid_at(&self, i: usize) -> bool {
        match self {
            Self::Elems { .. } => true,
            Self::Num { valid, .. } | Self::Bool { valid, .. } => {
                valid.as_ref().is_none_or(|v| v[i])
            }
            Self::Gen(v) => !matches!(v[i], Value::Null),
        }
    }

    /// `self[lo..hi]`, clamped to the column. Borrowed ids stay borrowed.
    #[must_use]
    pub fn page(self, lo: usize, hi: usize) -> Self {
        let n = self.len();
        let lo = lo.min(n);
        let hi = hi.clamp(lo, n);
        // Owned data is MOVED into the window, not copied out of it. A borrowed id
        // column reslices for free; everything else truncates and splits, which is
        // what the code this replaced did in place. Copying instead cost 1.7x on a
        // projection of one string column — the window is on the hot path of every
        // paged query, and a column is the whole result.
        fn cut<T>(mut v: Vec<T>, lo: usize, hi: usize) -> Vec<T> {
            v.truncate(hi);
            v.split_off(lo)
        }

        let mask = |v: Option<Vec<bool>>| v.map(|v| cut(v, lo, hi));

        match self {
            Self::Elems { ids, is_edge } => Self::Elems {
                ids: match ids {
                    std::borrow::Cow::Borrowed(s) => std::borrow::Cow::Borrowed(&s[lo..hi]),
                    std::borrow::Cow::Owned(v) => std::borrow::Cow::Owned(cut(v, lo, hi)),
                },
                is_edge,
            },
            Self::Num { d, valid } => Self::Num {
                d: cut(d, lo, hi),
                valid: mask(valid),
            },
            Self::Bool { t, valid } => Self::Bool {
                t: cut(t, lo, hi),
                valid: mask(valid),
            },
            Self::Gen(v) => Self::Gen(cut(v, lo, hi)),
        }
    }

    /// Append one row.
    ///
    /// An element column takes the id out of a `Node`/`Edge`; a typed column
    /// takes the value and marks the row invalid when it is a `Null`, allocating
    /// the mask at that point rather than up front — a column of a thousand
    /// numbers and one null pays for the mask once, and one with no nulls never
    /// pays at all.
    pub fn push_val(&mut self, v: &Value) {
        match self {
            Self::Elems { ids, .. } => match v {
                Value::Node(i) | Value::Edge(i) => ids.to_mut().push(*i),
                // Not an element: an element column cannot hold it, and silently
                // dropping the row would misalign every other column.
                _ => panic!("pushed a non-element into an element column"),
            },
            Self::Num { d, valid } => {
                let n = d.len();

                match v {
                    Value::Num(x) => {
                        d.push(*x);

                        if let Some(m) = valid {
                            m.push(true);
                        }
                    }
                    _ => {
                        d.push(f64::NAN);
                        valid.get_or_insert_with(|| vec![true; n]).push(false);
                    }
                }
            }
            Self::Bool { t, valid } => {
                let n = t.len();

                match v {
                    Value::Bool(b) => {
                        t.push(*b);

                        if let Some(m) = valid {
                            m.push(true);
                        }
                    }
                    _ => {
                        t.push(false);
                        valid.get_or_insert_with(|| vec![true; n]).push(false);
                    }
                }
            }
            Self::Gen(vs) => vs.push(v.clone()),
        }
    }

    /// Each row held for `k` consecutive output rows — the LEFT side of a cross
    /// product.
    #[must_use]
    pub fn repeat_each(self, k: usize) -> Self {
        self.rebuild(&|n| (0..n).flat_map(|i| std::iter::repeat_n(i, k)).collect())
    }

    /// The whole column laid down `k` times — the RIGHT side of a cross product.
    #[must_use]
    pub fn tile(self, k: usize) -> Self {
        self.rebuild(&|n| (0..k).flat_map(|_| 0..n).collect())
    }

    /// Rebuild the column by taking rows in the order `order(len)` gives.
    ///
    /// One gather for every reshaping — cross products, sorts, group
    /// representatives — so a new one costs an index list rather than a match arm
    /// per representation.
    #[must_use]
    fn rebuild(self, order: &dyn Fn(usize) -> Vec<usize>) -> Self {
        let idx = order(self.len());
        let pick =
            |valid: Option<Vec<bool>>| valid.map(|v| idx.iter().map(|&i| v[i]).collect::<Vec<_>>());

        match self {
            Self::Elems { ids, is_edge } => Self::Elems {
                ids: std::borrow::Cow::Owned(idx.iter().map(|&i| ids[i]).collect()),
                is_edge,
            },
            Self::Num { d, valid } => Self::Num {
                d: idx.iter().map(|&i| d[i]).collect(),
                valid: pick(valid),
            },
            Self::Bool { t, valid } => Self::Bool {
                t: idx.iter().map(|&i| t[i]).collect(),
                valid: pick(valid),
            },
            Self::Gen(v) => Self::Gen(idx.iter().map(|&i| v[i].clone()).collect()),
        }
    }

    /// Read a stored property column at `ids` into a runtime column.
    ///
    /// One gather, dispatching on what the STORAGE holds — which is the question
    /// every caller was asking separately. GQL had three (`gather_num`,
    /// `gather_str`, `gather_temporal`) chained with `or_else` so that a column
    /// type nobody had written a gather for fell through to a per-row rebuild;
    /// Gremlin had a fourth that boxed everything.
    ///
    /// Absent is NULL here, one row per id — the alignment a keyed modulator
    /// needs (`order().by(k)`, `GROUP BY k`). A caller that wants absent rows
    /// DROPPED instead — which is what `values(k)` means — filters afterwards.
    ///
    /// `None` for a column shape with no unboxed form — `Mixed`, `Vec`, `Record`
    /// — because reading one is a per-LANGUAGE question: GQL takes a stored map
    /// as an ISO record and Gremlin as a map. The caller falls back to its own
    /// per-row read, which is what it did before this existed.
    #[must_use]
    pub fn from_property(
        col: Option<&crate::graph::Column>,
        ids: &[u32],
        strs: &crate::graph::Dict,
    ) -> Option<Self> {
        use crate::graph::Column;

        let masked = |present: &crate::graph::BitSet| {
            let mut valid = Vec::with_capacity(ids.len());
            let mut any_absent = false;

            for &vi in ids {
                let ok = present.get(vi as usize);

                any_absent |= !ok;
                valid.push(ok);
            }

            any_absent.then_some(valid)
        };

        Some(match col {
            Some(Column::Num { data, present }) => Self::Num {
                d: ids.iter().map(|&vi| data[vi as usize]).collect(),
                valid: masked(present),
            },
            Some(Column::Bool { data, present }) => Self::Bool {
                t: ids.iter().map(|&vi| data[vi as usize]).collect(),
                valid: masked(present),
            },
            Some(Column::Str { data, present }) => Self::Gen(
                ids.iter()
                    .map(|&vi| {
                        let i = vi as usize;

                        if present.get(i) {
                            Value::Str(strs.arc(data[i]))
                        } else {
                            Value::Null
                        }
                    })
                    .collect(),
            ),
            Some(Column::Temporal { data, present }) => Self::Gen(
                ids.iter()
                    .map(|&vi| {
                        let i = vi as usize;

                        if present.get(i) {
                            Value::Temporal(data.get(i))
                        } else {
                            Value::Null
                        }
                    })
                    .collect(),
            ),
            // The key has no column at all: every row is a null, which no
            // per-row read would improve on.
            None => Self::Gen(vec![Value::Null; ids.len()]),
            // `Mixed` / `Vec` / `Record`: the caller's own read.
            Some(_) => return None,
        })
    }

    /// Row `i` as a value. Cheap for the typed variants; a `Str` is an `Arc`
    /// bump.
    #[must_use]
    pub fn val_at(&self, i: usize) -> Value {
        if !self.valid_at(i) {
            return Value::Null;
        }

        match self {
            Self::Elems { ids, is_edge } => {
                if *is_edge {
                    Value::Edge(ids[i])
                } else {
                    Value::Node(ids[i])
                }
            }
            Self::Num { d, .. } => Value::Num(d[i]),
            Self::Bool { t, .. } => Value::Bool(t[i]),
            Self::Gen(v) => v[i].clone(),
        }
    }

    /// Row `i`, handed to `f` BY REFERENCE.
    ///
    /// A boxed column lends its value; a typed one has to build a temporary, and
    /// `f` borrows that. The distinction is worth an accessor because the callers
    /// that read a whole column cell by cell — building a row, keying a DISTINCT —
    /// are exactly the ones for which a clone per cell shows up: routing them
    /// through `val_at` instead cost 2.1x on a projection of one string column.
    pub fn with_val_at<R>(&self, i: usize, f: impl FnOnce(&Value) -> R) -> R {
        match self {
            Self::Gen(v) if self.valid_at(i) => f(&v[i]),
            _ => f(&self.val_at(i)),
        }
    }

    /// Append `other`'s rows.
    ///
    /// Two columns of the same representation stay in it; anything else boxes,
    /// because a column has ONE representation and the alternative is a variant
    /// that is secretly two. Chunked work — a projection split across threads —
    /// is the caller that needs this, and its chunks agree by construction.
    pub fn append(&mut self, other: Self) {
        match (&mut *self, other) {
            (
                Self::Num { d, valid },
                Self::Num {
                    d: od,
                    valid: ovalid,
                },
            ) => {
                let (n, on) = (d.len(), od.len());

                d.extend(od);

                match (valid.as_mut(), ovalid) {
                    (None, None) => {}
                    (Some(m), None) => m.extend(std::iter::repeat_n(true, on)),
                    (None, Some(om)) => {
                        let mut m = vec![true; n];

                        m.extend(om);
                        *valid = Some(m);
                    }
                    (Some(m), Some(om)) => m.extend(om),
                }
            }
            (a, b) => {
                let mut vals = std::mem::replace(a, Self::Gen(Vec::new())).into_vals();

                vals.extend(b.into_vals());
                *a = Self::Gen(vals);
            }
        }
    }

    /// Drop the rows where `keep[i]` is false, in place.
    ///
    /// The validity mask is compacted with the data, which is the kind of thing
    /// that goes wrong when a column is two parallel vectors and only one of them
    /// is a column's business.
    pub fn retain_rows(&mut self, keep: &[bool]) {
        fn mask(valid: &mut Option<Vec<bool>>, keep: &[bool]) {
            if let Some(v) = valid {
                let mut i = 0;

                v.retain(|_| {
                    let k = keep[i];

                    i += 1;
                    k
                });
            }
        }

        match self {
            Self::Elems { ids, .. } => {
                let mut i = 0;

                ids.to_mut().retain(|_| {
                    let k = keep[i];

                    i += 1;
                    k
                });
            }
            Self::Num { d, valid } => {
                let mut i = 0;

                d.retain(|_| {
                    let k = keep[i];

                    i += 1;
                    k
                });
                mask(valid, keep);
            }
            Self::Bool { t, valid } => {
                let mut i = 0;

                t.retain(|_| {
                    let k = keep[i];

                    i += 1;
                    k
                });
                mask(valid, keep);
            }
            Self::Gen(v) => {
                let mut i = 0;

                v.retain(|_| {
                    let k = keep[i];

                    i += 1;
                    k
                });
            }
        }
    }

    /// The column as boxed values, one per row. An invalid row is a `Null`.
    #[must_use]
    pub fn into_vals(self) -> Vec<Value> {
        let boxed = |n: usize, valid: Option<Vec<bool>>, f: &dyn Fn(usize) -> Value| {
            (0..n)
                .map(|i| {
                    if valid.as_ref().is_none_or(|v| v[i]) {
                        f(i)
                    } else {
                        Value::Null
                    }
                })
                .collect()
        };

        match self {
            Self::Elems { ids, is_edge } => ids
                .iter()
                .map(|&id| {
                    if is_edge {
                        Value::Edge(id)
                    } else {
                        Value::Node(id)
                    }
                })
                .collect(),
            Self::Num { d, valid } => boxed(d.len(), valid, &|i| Value::Num(d[i])),
            Self::Bool { t, valid } => boxed(t.len(), valid, &|i| Value::Bool(t[i])),
            Self::Gen(v) => v,
        }
    }
}

/// First-seen grouping: one entry per distinct key, in the order the keys first
/// appear, with a caller-chosen accumulator per group.
///
/// Returns `(representative row, accumulator)` — the representative being the
/// FIRST row of the group, which is what makes `DISTINCT` and `dedup()` well
/// defined about which duplicate survives.
///
/// # The key is the parameter, as always
///
/// Grouping identity is a language contract and the two engines disagree: GQL
/// keys a composite row on `val_key`, Gremlin on `dedup_key`, and a number column
/// keys on raw bits with signed zeros collapsed. The BUCKETING is the same in
/// every case, and was written four times.
///
/// `key(i)` returning `None` means row `i` has NO key: it is equal to nothing,
/// including another keyless row, so each gets a group of its own.
///
/// That is GREMLIN's NaN rule — a NaN is never a duplicate — and NOT GQL's, which
/// keys a NaN like any other value and puts them in one group (`RETURN DISTINCT
/// sqrt(-1) + 0` is one row). Both are reachable only through a COMPUTED column:
/// a NaN cannot be stored, since every write normalizes a non-finite number to
/// null. So a mutation that merges keyless rows survives the suite today, and
/// this is the note that says why rather than a test that cannot fail.
///
/// `key` is called exactly ONCE per row, for rows `0..n` in ascending order, up
/// to wherever `cap` stops it. That is part of the contract, not an accident of
/// the loop: a caller with keys it cannot clone hands them over by move.
///
/// `cap` stops once that many groups exist. `DISTINCT … LIMIT 5` over a large
/// frame does not need the sixth group, and pushing the bound in here is the
/// difference between an early exit and a full pass.
pub fn group_first_seen<K: Eq + std::hash::Hash, A>(
    n: usize,
    mut key: impl FnMut(usize) -> Option<K>,
    init: impl Fn() -> A,
    mut add: impl FnMut(&mut A, usize),
    cap: Option<usize>,
) -> Vec<(usize, A)> {
    let mut groups: Vec<(usize, A)> = Vec::new();
    let mut index: crate::fxhash::FxHashMap<K, usize> = crate::fxhash::FxHashMap::default();

    for i in 0..n {
        match key(i) {
            Some(k) => match index.get(&k) {
                Some(&g) => add(&mut groups[g].1, i),
                None => {
                    if cap.is_some_and(|c| groups.len() >= c) {
                        break;
                    }

                    index.insert(k, groups.len());

                    let mut a = init();

                    add(&mut a, i);
                    groups.push((i, a));
                }
            },
            None => {
                if cap.is_some_and(|c| groups.len() >= c) {
                    break;
                }

                let mut a = init();

                add(&mut a, i);
                groups.push((i, a));
            }
        }
    }

    groups
}

/// Keep the `cap` smallest by `cmp`, in order — an ORDER BY with a LIMIT.
///
/// Quickselect partitions at `cap` in O(n) and only the kept prefix is sorted, so
/// the cost is O(n + k log k) rather than O(n log n) for a full sort whose tail
/// is then discarded. A `cap` of `None`, or one that is not smaller than the
/// input, sorts everything.
///
/// Same split as [`fold_extreme`]: the ALGORITHM is shared and the comparator is
/// the argument, because ordering is a per-language contract. GQL's projection
/// accumulator had this and Gremlin's `order()` did not — 20k vertices took
/// 1.015ms there against 0.109ms here for the same top-10 question.
///
/// `select_nth_unstable_by` is unstable and so is the sort, which is why this
/// takes a comparator that is expected to be total. Both callers pass one that
/// breaks ties (GQL's `cmp_keyed` walks every sort key; Gremlin's arm declines
/// any key set that is not uniformly comparable), so an unstable partition
/// cannot make the answer depend on the input order.
pub fn keep_smallest<T>(
    items: &mut Vec<T>,
    cap: Option<usize>,
    mut cmp: impl FnMut(&T, &T) -> std::cmp::Ordering,
) {
    if let Some(cap) = cap {
        retain_smallest(items, cap, &mut cmp);
    }

    items.sort_by(cmp);
}

/// The `cap` smallest by `cmp`, in NO particular order — the partition half of
/// [`keep_smallest`], for a caller that will sort later or not at all.
///
/// A streaming top-k wants exactly this: it trims its buffer whenever it grows
/// past a threshold, and sorting on every trim would be the cost the trimming
/// exists to avoid.
pub fn retain_smallest<T>(
    items: &mut Vec<T>,
    cap: usize,
    cmp: impl FnMut(&T, &T) -> std::cmp::Ordering,
) {
    if cap == 0 {
        items.clear();
        return;
    }

    if cap < items.len() {
        items.select_nth_unstable_by(cap - 1, cmp);
        items.truncate(cap);
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
